"""Matched cloud-free CPU/Metal simulation and retrieval benchmark.

This uses ROBERT's distributed R=100 opacity tables, an isothermal free-
chemistry atmosphere, exact conservative RORR, linear-source thermal emission,
a blackbody star, and the same fixed instrument projection on both backends.
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from robert_exoplanets import bundled_k_table_directory
from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, SpectralGrid
from robert_exoplanets.metal import (
    molecular_weights_for_species,
    prepare_cloud_free_emission_from_provider,
    require_metal_runtime,
)
from robert_exoplanets.opacity import CorrelatedKOpacityProvider
from robert_exoplanets.retrieval import (
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)
from robert_exoplanets.rt.emission import _planck_radiance_wavelength
from robert_exoplanets.rt.optical_depth import assemble_gas_optical_depth
from robert_exoplanets.rt.thermal_integration import (
    integrate_thermal_emission_spectrum,
)


def _time_cpu(function, repeats: int):
    value = function()
    started = perf_counter()
    for _ in range(repeats):
        value = function()
    return (perf_counter() - started) / repeats, value


def _time_device(function, repeats: int):
    started = perf_counter()
    value = function()
    value.block_until_ready()
    compile_seconds = perf_counter() - started
    started = perf_counter()
    for _ in range(repeats):
        value = function()
        value.block_until_ready()
    return compile_seconds, (perf_counter() - started) / repeats, value


def main() -> None:
    runtime = require_metal_runtime()
    species = ("H2O", "CO2", "CH4")
    provider = CorrelatedKOpacityProvider.from_exomol_kta_directory(
        bundled_k_table_directory(),
        species=species,
        resolution="R100",
        interpolation="log_pressure_temperature_log_k",
        nonfinite_policy="floor",
        nonfinite_fill_value=np.finfo(np.float32).tiny,
    )
    reference = provider.tables[species[0]]
    wavelength_mask = (reference.wavelength_micron >= 1.0) & (
        reference.wavelength_micron <= 12.0
    )
    selected = np.flatnonzero(wavelength_mask)
    wavelength = reference.wavelength_micron[selected]
    wavenumber = reference.wavenumber_cm_inverse[selected]
    pressure = PressureGrid.logspace(1.0e-5, 10.0, 40)
    spectral = SpectralGrid.from_array(wavenumber, unit="cm^-1", role="opacity")
    prepared_cpu = provider.prepare(spectral, pressure, species)
    molecular_weights = molecular_weights_for_species(species).astype(float)
    g_weights = reference.g_weights
    radius_ratio_squared = (75781520.0 / 565568100.0) ** 2

    def cpu_native(parameters):
        temperature = float(parameters[0])
        vmr = 10.0 ** np.asarray(parameters[1:], dtype=float)
        mean_molecular_weight = (1.0 - np.sum(vmr)) * 2.3 + np.sum(
            vmr * molecular_weights
        )
        atmosphere = AtmosphereState(
            pressure_grid=pressure,
            temperature=np.full(pressure.n_layers, temperature),
            temperature_edges=np.full(pressure.n_layers + 1, temperature),
            composition={
                name: np.full(pressure.n_layers, value)
                for name, value in zip(species, vmr, strict=True)
            },
            mean_molecular_weight=np.full(pressure.n_layers, mean_molecular_weight),
        )
        opacity = provider.evaluate(atmosphere, prepared_cpu)
        gas = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=6.0,
            gas_combination="random_overlap",
            retain_species_tau=False,
        )
        source = np.array(
            [
                _planck_radiance_wavelength(wavelength, temperature)
                for _ in range(pressure.n_layers + 1)
            ]
        )
        radiance = integrate_thermal_emission_spectrum(
            gas.total_tau,
            source[:-1],
            g_weights,
            np.ones((1, pressure.n_layers)),
            [1.0],
            level_source_ordered=source,
            bottom_source=source[-1],
            bottom_visible=[True],
            backend="numba",
        ).radiance
        star = _planck_radiance_wavelength(wavelength, 4750.0)
        return radius_ratio_squared * radiance / star

    truth = np.array([1100.0, -3.0, -4.0, -4.5])
    truth_native = cpu_native(truth)
    projection = np.zeros((24, selected.size))
    for row, indices in enumerate(np.array_split(np.arange(selected.size), 24)):
        projection[row, indices] = 1.0 / indices.size
    observation = projection @ truth_native
    uncertainty = np.maximum(np.abs(observation) * 0.01, 1.0e-9)
    metal_model = prepare_cloud_free_emission_from_provider(
        provider=provider,
        species=species,
        spectral_indices=selected,
        pressure_center_bar=pressure.centers,
        pressure_edge_bar=pressure.edges,
        projection_matrix=projection,
        observation=observation,
        uncertainty=uncertainty,
        runtime=runtime,
        gravity_m_s2=6.0,
        planet_radius_m=75781520.0,
        star_radius_m=565568100.0,
        star_temperature_k=4750.0,
    )
    parameter_set = RetrievalParameterSet(
        (
            RetrievalParameter("temperature_k", UniformPrior(700.0, 1600.0)),
            RetrievalParameter("log_H2O", UniformPrior(-8.0, -1.0)),
            RetrievalParameter("log_CO2", UniformPrior(-10.0, -1.0)),
            RetrievalParameter("log_CH4", UniformPrior(-10.0, -1.0)),
        )
    )
    problem = metal_model.retrieval_problem(
        name="bundled-r100-cloud-free-metal", parameters=parameter_set
    )
    compiled_native = runtime.jax.jit(metal_model.native_eclipse_depth)
    truth_device = runtime.put(truth)
    compile_seconds, metal_seconds, metal_native = _time_device(
        lambda: compiled_native(truth_device), 5
    )
    cpu_seconds, _ = _time_cpu(lambda: cpu_native(truth), 10)
    metal_array = np.asarray(metal_native)
    difference = metal_array - truth_native
    relative = difference / np.maximum(np.abs(truth_native), np.finfo(float).tiny)

    offsets = np.linspace(-2.0, 2.0, 21)
    cpu_loglike = []
    metal_loglike = []
    for offset in offsets:
        vector = truth.copy()
        vector[0] += offset
        model = projection @ cpu_native(vector)
        cpu_loglike.append(-0.5 * np.sum(((observation - model) / uncertainty) ** 2))
        metal_loglike.append(problem.log_likelihood_from_vector(vector))
    cpu_loglike = np.asarray(cpu_loglike)
    metal_loglike = np.asarray(metal_loglike)

    def posterior(loglike):
        weight = np.exp(loglike - np.max(loglike))
        weight /= weight.sum()
        mean = np.sum(weight * offsets)
        width = np.sqrt(np.sum(weight * (offsets - mean) ** 2))
        evidence = np.max(loglike) + np.log(np.mean(np.exp(loglike - np.max(loglike))))
        return {"mean_k": float(mean), "width_k": float(width), "logz": float(evidence)}

    loglike_difference = np.abs(metal_loglike - cpu_loglike)
    result = {
        "runtime": runtime.metadata,
        "physics": {
            "species": list(species),
            "layers": pressure.n_layers,
            "wavelengths": selected.size,
            "g_ordinates": g_weights.size,
            "opacity": "bundled R100 KTA",
            "clouds": "none",
        },
        "speed": {
            "metal_compile_seconds": compile_seconds,
            "metal_warm_seconds": metal_seconds,
            "cpu_warm_seconds": cpu_seconds,
            "speedup_cpu_over_metal": cpu_seconds / metal_seconds,
        },
        "spectrum_accuracy": {
            "max_absolute_eclipse_depth": float(np.max(np.abs(difference))),
            "rms_relative": float(np.sqrt(np.mean(relative**2))),
            "max_relative": float(np.max(np.abs(relative))),
        },
        "likelihood_accuracy": {
            "median_absolute_delta_loglike": float(np.median(loglike_difference)),
            "p99_absolute_delta_loglike": float(np.quantile(loglike_difference, 0.99)),
            "max_absolute_delta_loglike": float(np.max(loglike_difference)),
        },
        "posterior_grid": {
            "cpu": posterior(cpu_loglike),
            "metal": posterior(metal_loglike),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
