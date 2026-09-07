"""Benchmark ROBERT's isolated JAX Metal kernels against CPU references.

Run only on Apple Silicon after installing ``robert-exoplanets[metal]``::

    ENABLE_PJRT_COMPATIBILITY=1 conda run -n robert-exoplanets \
      python scripts/benchmark_metal_backend.py

Compilation and warmed execution are reported separately.  Every timed Metal
result is synchronized with ``block_until_ready``.
"""

from __future__ import annotations

import json
import platform
from time import perf_counter

import numpy as np

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, RobertValidationError, SpectralGrid
from robert_exoplanets.metal import (
    MetalLinearGaussianLikelihood,
    MetalLinearProjection,
    MetalResourcePolicy,
    PreparedMetalCorrelatedK,
    assemble_gas_optical_depth_device,
    compile_metal_batched_likelihood,
    compile_metal_likelihood,
    integrate_clear_thermal_device,
    metal_random_overlap_species_tau,
    metal_solve_thermal_sh4_spectrum,
    mie_efficiencies_and_moments_device,
    planck_radiance_wavelength_device,
    require_metal_runtime,
)
from robert_exoplanets.opacity import CorrelatedKOpacityProvider, CorrelatedKTable
from robert_exoplanets.rt.emission import _planck_radiance_wavelength
from robert_exoplanets.rt.mie import mie_efficiencies, mie_phase_function_moments
from robert_exoplanets.rt.optical_depth import assemble_gas_optical_depth
from robert_exoplanets.rt.random_overlap import random_overlap_species_tau
from robert_exoplanets.rt.sh4 import solve_thermal_sh4_spectrum
from robert_exoplanets.rt.thermal_integration import (
    integrate_thermal_emission_spectrum,
)


def _time_cpu(function, repeats: int) -> tuple[float, object]:
    value = function()
    started = perf_counter()
    for _ in range(repeats):
        value = function()
    return (perf_counter() - started) / repeats, value


def _time_metal(function, repeats: int) -> tuple[float, float, object]:
    started = perf_counter()
    value = function()
    value.block_until_ready()
    compile_seconds = perf_counter() - started
    started = perf_counter()
    for _ in range(repeats):
        value = function()
        value.block_until_ready()
    return compile_seconds, (perf_counter() - started) / repeats, value


def _error(actual, expected) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    difference = actual_array - expected_array
    denominator = np.maximum(np.abs(expected_array), np.finfo(float).tiny)
    return {
        "max_absolute": float(np.max(np.abs(difference))),
        "rms_relative": float(np.sqrt(np.mean((difference / denominator) ** 2))),
        "max_relative": float(np.max(np.abs(difference) / denominator)),
    }


def benchmark_rorr(runtime) -> dict[str, object]:
    rng = np.random.default_rng(101)
    tau = np.exp(rng.uniform(-9.0, 1.0, size=(3, 40, 64, 8)))
    weights = np.arange(1.0, 9.0)
    weights /= weights.sum()
    policy = MetalResourcePolicy.laptop_safe()
    policy.require_safe(
        "RORR benchmark",
        policy.estimate_rorr_bytes(
            species=tau.shape[0],
            layers=tau.shape[1],
            spectral=tau.shape[2],
            g_ordinates=tau.shape[3],
        ),
    )
    cpu_seconds, cpu = _time_cpu(
        lambda: random_overlap_species_tau(tau, weights, backend="numba"), 30
    )

    def function():
        return metal_random_overlap_species_tau(tau, weights, runtime=runtime)

    compile_seconds, metal_seconds, metal = _time_metal(function, 30)
    return {
        "shape": list(tau.shape),
        "compile_seconds": compile_seconds,
        "metal_warm_seconds": metal_seconds,
        "cpu_numba_seconds": cpu_seconds,
        "speedup_cpu_over_metal": cpu_seconds / metal_seconds,
        "accuracy": _error(np.asarray(metal), cpu),
    }


def benchmark_mie(runtime) -> dict[str, object]:
    x = np.geomspace(0.1, 50.0, 128)
    real = np.linspace(1.3, 2.0, x.size)
    imaginary = np.geomspace(1.0e-5, 0.3, x.size)
    mu, weight = np.polynomial.legendre.leggauss(87)

    def cpu_function():
        efficiencies = np.array(
            [
                mie_efficiencies(value, complex(n, k))
                for value, n, k in zip(x, real, imaginary, strict=True)
            ]
        )
        moments = np.array(
            [
                mie_phase_function_moments(value, complex(n, k))
                for value, n, k in zip(x, real, imaginary, strict=True)
            ]
        )
        return efficiencies, moments

    cpu_seconds, cpu = _time_cpu(cpu_function, 3)

    def metal_function():
        result = mie_efficiencies_and_moments_device(
            runtime.put(x),
            runtime.put(real),
            runtime.put(imaginary),
            runtime.put(mu),
            runtime.put(weight),
            maximum_order=80,
            maximum_downward_order=116,
        )
        return result[0]

    compile_seconds, metal_seconds, _ = _time_metal(metal_function, 10)
    result = mie_efficiencies_and_moments_device(
        runtime.put(x),
        runtime.put(real),
        runtime.put(imaginary),
        runtime.put(mu),
        runtime.put(weight),
        maximum_order=80,
        maximum_downward_order=116,
    )
    for item in result:
        item.block_until_ready()
    metal_efficiencies = np.column_stack(
        (np.asarray(result[0]), np.asarray(result[1]), np.asarray(result[2]))
    )
    return {
        "particles": x.size,
        "compile_seconds": compile_seconds,
        "metal_warm_seconds": metal_seconds,
        "cpu_reference_seconds": cpu_seconds,
        "speedup_cpu_over_metal": cpu_seconds / metal_seconds,
        "efficiency_accuracy": _error(metal_efficiencies, cpu[0]),
        "moment_accuracy": _error(np.asarray(result[3]), cpu[1]),
    }


def benchmark_sh4(runtime) -> dict[str, object]:
    rng = np.random.default_rng(902)
    # Keep the standalone solver benchmark bounded.  The report separately
    # records that an 80 x 32 x 8 compile exceeded the operational time limit.
    layers, spectral, g_count = 80, 4, 4
    tau = np.exp(rng.uniform(-5.0, -0.2, size=(layers, spectral, g_count)))
    omega = rng.uniform(0.1, 0.95, size=tau.shape)
    asymmetry = rng.uniform(-0.1, 0.8, size=tau.shape)
    planck = (
        np.linspace(1.0, 5.0, layers + 1)[:, None]
        * np.linspace(0.8, 1.2, spectral)[None, :]
    )
    mu = np.array([0.2113248654, 0.7886751346])
    angle_weight = np.array([0.5, 0.5])
    g_weight = np.full(g_count, 1.0 / g_count)
    bottom = planck[-1] * 1.05
    arguments = (tau, omega, asymmetry, planck, mu, angle_weight, g_weight)
    policy = MetalResourcePolicy.laptop_safe()
    policy.require_safe(
        "SH4 benchmark",
        policy.estimate_sh4_bytes(
            layers=layers,
            spectral=spectral,
            g_ordinates=g_count,
        ),
    )
    cpu_seconds, cpu = _time_cpu(
        lambda: (
            solve_thermal_sh4_spectrum(
                *arguments,
                bottom_planck_radiance=bottom,
                backend="numba",
            ).radiance
        ),
        3,
    )
    device_arguments = tuple(runtime.put(value) for value in arguments)

    def function():
        return metal_solve_thermal_sh4_spectrum(
            *device_arguments,
            bottom_planck_radiance=runtime.put(bottom),
        )

    compile_seconds, metal_seconds, metal = _time_metal(function, 5)
    return {
        "shape": [layers, spectral, g_count],
        "compile_seconds": compile_seconds,
        "metal_warm_seconds": metal_seconds,
        "cpu_numba_seconds": cpu_seconds,
        "speedup_cpu_over_metal": cpu_seconds / metal_seconds,
        "accuracy": _error(np.asarray(metal), cpu),
    }


def benchmark_complete_clear_graph(runtime) -> dict[str, object]:
    rng = np.random.default_rng(314)
    species_names = ("H2O", "CO2", "CH4")
    layers, spectral, g_count = 80, 128, 8
    pressure_grid = PressureGrid.logspace(1.0e-6, 100.0, layers)
    pressure_table = np.logspace(-7.0, 3.0, 12)
    temperature_table = np.linspace(400.0, 2400.0, 12)
    wavelength = np.linspace(1.0, 12.0, spectral)
    wavenumber = 10000.0 / wavelength
    g_weight = np.arange(1.0, g_count + 1)
    g_weight /= g_weight.sum()
    base = rng.uniform(
        -55.0,
        -45.0,
        size=(
            len(species_names),
            pressure_table.size,
            temperature_table.size,
            spectral,
            g_count,
        ),
    )
    kcoeff = np.exp(base)
    tables = {
        name: CorrelatedKTable(
            species=name,
            pressure_bar=pressure_table,
            temperature_K=temperature_table,
            wavenumber_cm_inverse=wavenumber,
            wavelength_micron=wavelength,
            g_samples=np.linspace(0.05, 0.95, g_count),
            g_weights=g_weight,
            kcoeff=kcoeff[index],
        )
        for index, name in enumerate(species_names)
    }
    provider = CorrelatedKOpacityProvider(
        tables=tables, interpolation="log_pressure_temperature_log_k"
    )
    spectral_grid = SpectralGrid.from_array(wavenumber, unit="cm^-1", role="opacity")
    prepared_cpu = provider.prepare(spectral_grid, pressure_grid, species_names)
    composition = {
        name: np.full(layers, value)
        for name, value in zip(species_names, (1.0e-3, 2.0e-4, 8.0e-5), strict=True)
    }
    base_temperature = np.linspace(700.0, 1700.0, layers)
    base_level_temperature = np.linspace(680.0, 1720.0, layers + 1)
    paths = np.ones((1, layers))
    point_weights = np.ones(1)
    projection = np.zeros((32, spectral))
    for row, indices in enumerate(np.array_split(np.arange(spectral), 32)):
        projection[row, indices] = 1.0 / indices.size

    def cpu_native(parameter: float):
        temperature = base_temperature + parameter
        level_temperature = base_level_temperature + parameter
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=temperature,
            temperature_edges=level_temperature,
            composition=composition,
            mean_molecular_weight=np.full(layers, 2.3),
        )
        opacity = provider.evaluate(atmosphere, prepared_cpu)
        gas = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=6.0,
            gas_combination="random_overlap",
            retain_species_tau=False,
        )
        level_source = np.array(
            [
                _planck_radiance_wavelength(wavelength, value)
                for value in level_temperature
            ]
        )
        radiance = integrate_thermal_emission_spectrum(
            gas.total_tau,
            level_source[:-1],
            g_weight,
            paths,
            point_weights,
            level_source_ordered=level_source,
            bottom_source=level_source[-1],
            bottom_visible=np.ones(1, dtype=bool),
            backend="numba",
        ).radiance
        return projection @ radiance

    reference_observation = cpu_native(0.0)
    uncertainty = np.maximum(np.abs(reference_observation) * 0.01, 1.0)

    def cpu_loglike_at(shift: float):
        residual = reference_observation - cpu_native(shift)
        return -0.5 * np.sum((residual / uncertainty) ** 2)

    cpu_seconds, cpu_value = _time_cpu(lambda: cpu_loglike_at(10.0), 10)
    indices = np.broadcast_to(np.arange(spectral), (len(species_names), spectral))
    prepared_metal = PreparedMetalCorrelatedK.from_arrays(
        kcoeff,
        pressure_table,
        temperature_table,
        indices,
        g_weight,
        runtime,
    )
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        reference_observation,
        uncertainty,
        MetalLinearProjection.from_matrix(projection, runtime),
        runtime,
    )
    pressure = runtime.put(pressure_grid.centers)
    pressure_edges = runtime.put(pressure_grid.edges)
    wavelength_device = runtime.put(wavelength)
    vmr = runtime.put(np.stack(tuple(composition.values())))
    mmw = runtime.put(np.full(layers, 2.3))
    paths_device = runtime.put(paths)
    point_weights_device = runtime.put(point_weights)

    def graph(parameters):
        shift = parameters[0]
        temperature = runtime.put(base_temperature) + shift
        levels = runtime.put(base_level_temperature) + shift
        interpolated = prepared_metal.interpolate(pressure, temperature)
        tau = assemble_gas_optical_depth_device(
            interpolated,
            vmr,
            pressure_edges,
            mmw,
            runtime.jnp.float32(6.0),
            prepared_metal.g_weights,
            opacity_unit_scale_m2=prepared_metal.opacity_unit_scale_m2,
        )
        source = planck_radiance_wavelength_device(wavelength_device, levels)
        radiance = integrate_clear_thermal_device(
            tau,
            source,
            prepared_metal.g_weights,
            paths_device,
            point_weights_device,
            source[-1],
            runtime.jnp.array([True]),
        )
        return likelihood.loglike(radiance)

    compiled = compile_metal_likelihood(graph, runtime)

    def function():
        return compiled(runtime.put([10.0]))

    compile_seconds, metal_seconds, metal_value = _time_metal(function, 20)
    # Resolve the synthetic posterior (about 0.5 K wide) rather than sampling
    # only its far tails.
    ensemble_shifts = np.linspace(-2.0, 2.0, 21)
    ensemble_cpu = np.array([cpu_loglike_at(value) for value in ensemble_shifts])
    ensemble_metal_values = []
    for value in ensemble_shifts:
        device_value = compiled(runtime.put([value]))
        device_value.block_until_ready()
        ensemble_metal_values.append(float(device_value))
    ensemble_metal = np.asarray(ensemble_metal_values)

    def posterior_summary(loglike_values):
        shifted = loglike_values - np.max(loglike_values)
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        mean = np.sum(weights * ensemble_shifts)
        standard_deviation = np.sqrt(np.sum(weights * (ensemble_shifts - mean) ** 2))
        log_evidence = np.max(loglike_values) + np.log(
            np.mean(np.exp(loglike_values - np.max(loglike_values)))
        )
        return {
            "mean_temperature_shift_k": float(mean),
            "standard_deviation_k": float(standard_deviation),
            "log_evidence": float(log_evidence),
        }

    ensemble_difference = np.abs(ensemble_metal - ensemble_cpu)
    batch_results = {}
    policy = MetalResourcePolicy.laptop_safe()
    for batch_size in (8, 32):
        try:
            policy.require_safe_batch("likelihood benchmark", batch_size)
        except RobertValidationError as exc:
            batch_results[str(batch_size)] = {
                "status": "refused_by_memory_policy",
                "reason": str(exc),
            }
            continue
        estimated = policy.estimate_rorr_bytes(
            species=len(species_names),
            layers=layers,
            spectral=spectral,
            g_ordinates=g_count,
            batch=batch_size,
        )
        if estimated > policy.working_set_limit_bytes:
            batch_results[str(batch_size)] = {
                "status": "refused_by_memory_policy",
                "estimated_peak_gib": estimated / 1024**3,
                "limit_gib": policy.working_set_limit_bytes / 1024**3,
            }
            continue
        shifts = np.linspace(-20.0, 20.0, batch_size)
        parameter_batch = runtime.put(shifts[:, None])
        batched_graph = compile_metal_batched_likelihood(
            graph,
            runtime,
            batch_size=batch_size,
            resource_policy=policy,
        )

        def metal_batch():
            return batched_graph(parameter_batch)

        batch_compile, batch_metal_seconds, batch_metal_value = _time_metal(
            metal_batch, 3
        )
        batch_cpu_seconds, batch_cpu_value = _time_cpu(
            lambda: np.array([cpu_loglike_at(value) for value in shifts]), 3
        )
        batch_results[str(batch_size)] = {
            "compile_seconds": batch_compile,
            "metal_batch_seconds": batch_metal_seconds,
            "cpu_sequential_batch_seconds": batch_cpu_seconds,
            "metal_likelihoods_per_second": batch_size / batch_metal_seconds,
            "cpu_likelihoods_per_second": batch_size / batch_cpu_seconds,
            "speedup_cpu_over_metal": batch_cpu_seconds / batch_metal_seconds,
            "max_absolute_loglike_difference": float(
                np.max(
                    np.abs(np.asarray(batch_metal_value, dtype=float) - batch_cpu_value)
                )
            ),
        }
    return {
        "shape": [len(species_names), layers, spectral, g_count],
        "compile_seconds": compile_seconds,
        "metal_warm_seconds": metal_seconds,
        "cpu_production_seconds": cpu_seconds,
        "speedup_cpu_over_metal": cpu_seconds / metal_seconds,
        "loglike_cpu": float(cpu_value),
        "loglike_metal": float(metal_value),
        "absolute_loglike_difference": abs(float(metal_value) - float(cpu_value)),
        "sequential_21_point_reproducibility": {
            "median_absolute_loglike_difference": float(np.median(ensemble_difference)),
            "p99_absolute_loglike_difference": float(
                np.quantile(ensemble_difference, 0.99)
            ),
            "maximum_absolute_loglike_difference": float(np.max(ensemble_difference)),
            "cpu_posterior": posterior_summary(ensemble_cpu),
            "metal_posterior": posterior_summary(ensemble_metal),
        },
        "batches": batch_results,
    }


def main() -> None:
    runtime = require_metal_runtime()
    result = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "runtime": runtime.metadata,
        "rorr": benchmark_rorr(runtime),
        "mie": benchmark_mie(runtime),
        "sh4": benchmark_sh4(runtime),
        "complete_clear_likelihood": benchmark_complete_clear_graph(runtime),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
