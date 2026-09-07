"""Matched bundled-opacity cloud-free CPU/JAX scientific validation."""

from __future__ import annotations

import os

import numpy as np
import pytest

from robert_exoplanets import bundled_k_table_directory
from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.atmosphere.temperature import _parmentier_guillot_eta
from robert_exoplanets.core import PressureGrid, SpectralGrid
from robert_exoplanets.metal import (
    MetalLinearGaussianLikelihood,
    MetalLinearProjection,
    MetalRuntime,
    PreparedMetalCloudFreeEmission,
    PreparedMetalCorrelatedK,
    molecular_weights_for_species,
)
from robert_exoplanets.opacity import CorrelatedKOpacityProvider
from robert_exoplanets.rt.emission import _planck_radiance_wavelength
from robert_exoplanets.rt.geometry import gauss_legendre_disk_geometry
from robert_exoplanets.rt.optical_depth import assemble_gas_optical_depth
from robert_exoplanets.rt.thermal_integration import (
    integrate_thermal_emission_spectrum,
)


def _runtime() -> MetalRuntime:
    if os.environ.get("ROBERT_RUN_JAX_CPU_TESTS") != "1":
        pytest.skip("set ROBERT_RUN_JAX_CPU_TESTS=1 for cloud-free JAX parity")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    return MetalRuntime(jax=jax, jnp=jnp, device=jax.devices("cpu")[0])


def _pg14_temperature(pressure_bar: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    kappa_ir, gamma1, gamma2, irradiation, alpha = parameters[:5]
    tau = kappa_ir * pressure_bar * 1.0e5 / 6.0
    fourth = 0.75 * 100.0**4 * (2.0 / 3.0 + tau)
    fourth += (
        0.75
        * irradiation**4
        * (
            (1.0 - alpha) * _parmentier_guillot_eta(gamma1, tau)
            + alpha * _parmentier_guillot_eta(gamma2, tau)
        )
    )
    return fourth**0.25


def test_bundled_cloud_free_spectrum_and_likelihood_match_cpu() -> None:
    runtime = _runtime()
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
    selected = np.arange(10, reference.wavenumber_cm_inverse.size - 10, 30)
    wavenumber = reference.wavenumber_cm_inverse[selected]
    wavelength = reference.wavelength_micron[selected]
    spectral_grid = SpectralGrid.from_array(wavenumber, unit="cm^-1", role="opacity")
    pressure_grid = PressureGrid.logspace(1.0e-5, 10.0, 8)
    prepared_cpu = provider.prepare(spectral_grid, pressure_grid, species)
    parameters = np.array([0.00316, 0.12, 1.7, 1510.0, 0.38, -3.0, -4.0, -4.5])
    molecular_weights = molecular_weights_for_species(species).astype(float)
    geometry = gauss_legendre_disk_geometry(4)
    paths = np.broadcast_to(
        1.0 / geometry.emission_angle_cosines[:, None],
        (geometry.n_points, pressure_grid.n_layers),
    )

    def cpu_spectrum(vector: np.ndarray) -> np.ndarray:
        abundance = 10.0 ** vector[5:]
        mmw = (1.0 - np.sum(abundance)) * 2.3 + np.sum(abundance * molecular_weights)
        temperature = _pg14_temperature(pressure_grid.centers, vector)
        edge_temperature = _pg14_temperature(pressure_grid.edges, vector)
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=temperature,
            temperature_edges=edge_temperature,
            composition={
                name: np.full(pressure_grid.n_layers, value)
                for name, value in zip(species, abundance, strict=True)
            },
            mean_molecular_weight=np.full(pressure_grid.n_layers, mmw),
        )
        evaluated = provider.evaluate(atmosphere, prepared_cpu)
        gas = assemble_gas_optical_depth(
            atmosphere,
            evaluated,
            gravity_m_s2=6.0,
            gas_combination="random_overlap",
            retain_species_tau=False,
        )
        level_source = np.array(
            [
                _planck_radiance_wavelength(wavelength, temperature_value)
                for temperature_value in edge_temperature
            ]
        )
        planet = integrate_thermal_emission_spectrum(
            gas.total_tau,
            level_source[:-1],
            reference.g_weights,
            paths,
            geometry.emission_angle_weights,
            level_source_ordered=level_source,
            bottom_source=level_source[-1],
            bottom_visible=np.ones(geometry.n_points, dtype=bool),
            backend="numpy",
        ).radiance
        stellar = _planck_radiance_wavelength(wavelength, 4750.0)
        return (75781520.0 / 565568100.0) ** 2 * planet / stellar

    expected = cpu_spectrum(parameters)

    kcoeff = np.stack([provider.tables[name].kcoeff for name in species])
    prepared_metal = PreparedMetalCorrelatedK.from_arrays(
        kcoeff,
        reference.pressure_bar,
        reference.temperature_K,
        np.broadcast_to(selected, (len(species), selected.size)),
        reference.g_weights,
        runtime,
    )
    projection = MetalLinearProjection.from_matrix(np.eye(selected.size), runtime)
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        expected,
        np.maximum(np.abs(expected) * 0.01, 1.0e-9),
        projection,
        runtime,
    )
    model = PreparedMetalCloudFreeEmission(
        runtime=runtime,
        opacity=prepared_metal,
        species=species,
        molecular_weights_amu=runtime.put(molecular_weights),
        pressure_center_bar=runtime.put(pressure_grid.centers),
        pressure_edge_bar=runtime.put(pressure_grid.edges),
        wavelength_micron=runtime.put(wavelength),
        gravity_m_s2=6.0,
        planet_radius_m=75781520.0,
        star_radius_m=565568100.0,
        star_temperature_k=4750.0,
        emission_path_factors=runtime.put(paths),
        emission_point_weights=runtime.put(geometry.emission_angle_weights),
        likelihood=likelihood,
    )
    device_parameters = runtime.put(parameters)
    actual = model.native_eclipse_depth(device_parameters)
    actual.block_until_ready()
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2.0e-4, atol=2.0e-9)
    assert abs(float(model.loglike(device_parameters))) < 1.0e-3
    assert np.ptp(_pg14_temperature(pressure_grid.centers, parameters)) > 1.0

    for species_index in range(len(species)):
        perturbed = parameters.copy()
        perturbed[5 + species_index] += 0.5
        expected_perturbed = cpu_spectrum(perturbed)
        actual_perturbed = model.native_eclipse_depth(runtime.put(perturbed))
        actual_perturbed.block_until_ready()
        assert np.max(np.abs(expected_perturbed - expected)) > 1.0e-10
        np.testing.assert_allclose(
            np.asarray(actual_perturbed),
            expected_perturbed,
            rtol=2.0e-4,
            atol=2.0e-9,
        )
