"""Opt-in tests that must execute on a real Apple Metal device."""

from __future__ import annotations

import os

import numpy as np
import pytest

from robert_exoplanets import bundled_k_table_directory
from robert_exoplanets.core import PressureGrid
from robert_exoplanets.metal import (
    MetalBinnedProjection,
    MetalLinearGaussianLikelihood,
    MetalLinearProjection,
    PreparedMetalCorrelatedK,
    PreparedMetalOpacitySampling,
    PreparedMetalOpacitySamplingCloudFreeEmission,
    assemble_gas_optical_depth_device,
    cloud_extinction_tau_device,
    compile_metal_likelihood,
    integrate_clear_thermal_device,
    lognormal_mie_optics_device,
    metal_random_overlap_species_tau,
    metal_solve_thermal_sh4_spectrum,
    mie_efficiencies_and_moments_device,
    planck_radiance_wavelength_device,
    prepare_cloud_free_emission_from_provider,
    require_metal_runtime,
)
from robert_exoplanets.opacity import CorrelatedKOpacityProvider
from robert_exoplanets.rt.geometry import gauss_legendre_disk_geometry

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_METAL_TESTS") != "1",
    reason="set ROBERT_RUN_METAL_TESTS=1 on Apple Silicon with jax-metal",
)


def _assert_metal(array) -> None:
    platforms = {
        str(getattr(device, "platform", "")).lower() for device in array.devices()
    }
    assert platforms == {"metal"}


def test_real_metal_kernels_are_resident_finite_and_deterministic() -> None:
    runtime = require_metal_runtime()
    rng = np.random.default_rng(72)
    tau = np.exp(rng.uniform(-8.0, 0.0, size=(3, 4, 6, 8)))
    weights = np.arange(1.0, 9.0)
    mixed = metal_random_overlap_species_tau(tau, weights, runtime=runtime)
    repeated = metal_random_overlap_species_tau(tau, weights, runtime=runtime)
    mixed.block_until_ready()
    repeated.block_until_ready()
    _assert_metal(mixed)
    assert np.all(np.isfinite(np.asarray(mixed)))
    np.testing.assert_array_equal(np.asarray(mixed), np.asarray(repeated))

    angular_mu, angular_weight = np.polynomial.legendre.leggauss(47)
    mie = mie_efficiencies_and_moments_device(
        runtime.put([0.1, 1.0, 5.0]),
        runtime.put([1.4, 1.6, 2.0]),
        runtime.put([0.0, 0.02, 0.2]),
        runtime.put(angular_mu),
        runtime.put(angular_weight),
        maximum_order=32,
        maximum_downward_order=47,
    )
    for value in mie:
        value.block_until_ready()
        _assert_metal(value)
        assert np.all(np.isfinite(np.asarray(value)))

    layer_tau = runtime.put(np.full((2, 2, 1), 0.4))
    omega = runtime.put(np.full((2, 2, 1), 0.6))
    asymmetry = runtime.put(np.full((2, 2, 1), 0.3))
    sh4 = metal_solve_thermal_sh4_spectrum(
        layer_tau,
        omega,
        asymmetry,
        runtime.put([[1.0, 1.2], [2.0, 2.4], [3.5, 4.0]]),
        runtime.put([0.3, 0.8]),
        runtime.put([0.4, 0.6]),
        runtime.put([1.0]),
        bottom_planck_radiance=runtime.put([4.0, 4.5]),
        delta_m=True,
    )
    sh4.block_until_ready()
    _assert_metal(sh4)
    assert np.all(np.isfinite(np.asarray(sh4)))


def test_real_metal_complete_parameter_to_likelihood_graph() -> None:
    runtime = require_metal_runtime()
    prepared = PreparedMetalCorrelatedK.from_arrays(
        np.full((1, 3, 3, 2, 4), 2.0e-24),
        [1.0e-5, 1.0e-2, 1.0],
        [500.0, 1000.0, 1800.0],
        [[0, 1]],
        [0.1, 0.2, 0.3, 0.4],
        runtime,
    )
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        [1.0e8, 8.0e7],
        [1.0e7, 1.0e7],
        MetalLinearProjection.from_matrix(np.eye(2), runtime),
        runtime,
    )
    pressure = runtime.put([1.0e-3, 0.1])
    edges = runtime.put([1.0e-5, 1.0e-2, 1.0])
    wavelengths = runtime.put([3.0, 5.0])

    def graph(parameters):
        temperature = parameters[:2]
        k_values = prepared.interpolate(pressure, temperature)
        tau = assemble_gas_optical_depth_device(
            k_values,
            runtime.jnp.full((1, 2), runtime.jnp.float32(1.0e-3)),
            edges,
            runtime.jnp.full((2,), runtime.jnp.float32(2.3)),
            runtime.jnp.float32(6.0),
            prepared.g_weights,
            opacity_unit_scale_m2=prepared.opacity_unit_scale_m2,
        )
        level_temperature = runtime.jnp.array(
            [temperature[0], 0.5 * (temperature[0] + temperature[1]), temperature[1]]
        )
        source = planck_radiance_wavelength_device(wavelengths, level_temperature)
        radiance = integrate_clear_thermal_device(
            tau,
            source,
            prepared.g_weights,
            runtime.put([[1.0, 1.0]]),
            runtime.put([1.0]),
            source[-1],
            runtime.jnp.array([True]),
        )
        return likelihood.loglike(radiance)

    compiled = compile_metal_likelihood(graph, runtime)
    first = compiled(runtime.put([900.0, 1300.0]))
    first.block_until_ready()
    second = compiled(runtime.put([900.0, 1300.0]))
    second.block_until_ready()
    _assert_metal(first)
    assert np.isfinite(float(first))
    assert float(first) == float(second)


def test_real_metal_complete_mie_cloud_sh4_likelihood_graph() -> None:
    """The retrieved cloud path remains on Metal through multiple scattering."""
    runtime = require_metal_runtime()
    wavelength = runtime.put([1.5, 3.0])
    angular_mu, angular_weight = np.polynomial.legendre.leggauss(47)
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        [2.0, 2.5],
        [0.2, 0.2],
        MetalLinearProjection.from_matrix(np.eye(2), runtime),
        runtime,
    )

    def graph(parameters):
        optics = lognormal_mie_optics_device(
            wavelength,
            runtime.jnp.array([1.6, 1.6]),
            runtime.jnp.array([0.02, 0.02]),
            parameters[0],
            runtime.jnp.float32(1.0),
            runtime.jnp.float32(3200.0),
            runtime.jnp.array([0.0]),
            runtime.jnp.array([1.0]),
            runtime.put(angular_mu),
            runtime.put(angular_weight),
            maximum_order=32,
            maximum_downward_order=47,
        )
        mass_fraction = runtime.jnp.full((2,), parameters[1])
        delta_pressure = runtime.jnp.array([100.0, 300.0])
        cloud_extinction = cloud_extinction_tau_device(
            delta_pressure,
            runtime.jnp.float32(6.0),
            mass_fraction,
            optics["mass_extinction_m2_kg"],
        )
        cloud_scattering = cloud_extinction_tau_device(
            delta_pressure,
            runtime.jnp.float32(6.0),
            mass_fraction,
            optics["mass_scattering_m2_kg"],
        )
        extinction = runtime.jnp.float32(0.1) + cloud_extinction[:, :, None]
        scattering = cloud_scattering[:, :, None]
        omega = runtime.jnp.clip(scattering / extinction, 0.0, 1.0)
        asymmetry = runtime.jnp.broadcast_to(
            optics["asymmetry_factor"][None, :, None], extinction.shape
        )
        moments = runtime.jnp.broadcast_to(
            optics["phase_function_moments"][:4, None, :, None],
            (4,) + extinction.shape,
        )
        forward = runtime.jnp.broadcast_to(
            optics["phase_function_moments"][4, None, :, None]
            / runtime.jnp.float32(9.0),
            extinction.shape,
        )
        spectrum = metal_solve_thermal_sh4_spectrum(
            extinction,
            omega,
            asymmetry,
            runtime.jnp.array([[1.0, 1.2], [2.0, 2.4], [3.5, 4.0]]),
            runtime.jnp.array([0.3, 0.8]),
            runtime.jnp.array([0.4, 0.6]),
            runtime.jnp.array([1.0]),
            bottom_planck_radiance=runtime.jnp.array([4.0, 4.5]),
            phase_function_moments=moments,
            delta_m_forward_fraction=forward,
            delta_m=True,
        )
        return likelihood.loglike(spectrum)

    compiled = compile_metal_likelihood(graph, runtime)
    parameters = runtime.put([0.3, 1.0e-4])
    first = compiled(parameters)
    first.block_until_ready()
    second = compiled(parameters)
    second.block_until_ready()
    _assert_metal(first)
    assert np.isfinite(float(first))
    assert float(first) == float(second)


def test_bundled_r100_cloud_free_model_runs_on_real_metal() -> None:
    """Exercise the phase-one user-facing model with distributed opacity."""
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
    selected = np.arange(10, reference.wavenumber_cm_inverse.size - 10, 30)
    pressure = PressureGrid.logspace(1.0e-5, 10.0, 8)
    model = prepare_cloud_free_emission_from_provider(
        provider=provider,
        species=species,
        spectral_indices=selected,
        pressure_center_bar=pressure.centers,
        pressure_edge_bar=pressure.edges,
        projection_matrix=np.eye(selected.size),
        observation=np.zeros(selected.size),
        uncertainty=np.ones(selected.size),
        runtime=runtime,
        gravity_m_s2=6.0,
        planet_radius_m=75781520.0,
        star_radius_m=565568100.0,
        star_temperature_k=4750.0,
    )
    compiled = runtime.jax.jit(model.native_eclipse_depth)
    spectrum = compiled(
        runtime.put([0.00316, 0.12, 1.7, 1510.0, 0.38, -3.0, -4.0, -4.5])
    )
    spectrum.block_until_ready()
    _assert_metal(spectrum)
    assert spectrum.shape == (selected.size,)
    assert np.all(np.isfinite(np.asarray(spectrum)))
    assert np.all(np.asarray(spectrum) > 0.0)


def test_real_metal_opacity_sampling_cloud_free_likelihood() -> None:
    """Run the direct sampled-opacity PG14 graph on the Metal device."""

    runtime = require_metal_runtime()
    species = ("H2O", "CO2")
    pressure_table = np.array([1.0e-6, 1.0e-3, 1.0, 100.0])
    temperature_table = np.array([500.0, 1200.0, 2200.0])
    wavelength = np.linspace(1.5, 5.0, 8)
    cross_sections = np.empty((2, 4, 3, 8))
    for species_index in range(2):
        cross_sections[species_index] = (
            (species_index + 1.0)
            * pressure_table[:, None, None] ** 0.1
            * np.exp(temperature_table[None, :, None] / 5000.0)
            * (1.0 + wavelength[None, None, :])
            * 1.0e-25
        )
    opacity = PreparedMetalOpacitySampling.from_arrays(
        cross_sections,
        pressure_table,
        temperature_table,
        runtime,
    )
    pressure = PressureGrid.logspace(1.0e-5, 10.0, 4)
    geometry = gauss_legendre_disk_geometry(4)
    path_factors = np.broadcast_to(
        1.0 / geometry.emission_angle_cosines[:, None],
        (geometry.n_points, pressure.n_layers),
    )
    likelihood = MetalLinearGaussianLikelihood.from_arrays(
        np.zeros(2),
        np.ones(2),
        MetalBinnedProjection.from_edges([0, 4, 8], wavelength.size, runtime),
        runtime,
    )
    model = PreparedMetalOpacitySamplingCloudFreeEmission(
        runtime=runtime,
        opacity=opacity,
        species=species,
        molecular_weights_amu=runtime.put([18.01528, 44.0095]),
        pressure_center_bar=runtime.put(pressure.centers),
        pressure_edge_bar=runtime.put(pressure.edges),
        wavelength_micron=runtime.put(wavelength),
        gravity_m_s2=10.0,
        planet_radius_m=7.1492e7,
        star_radius_m=6.957e8,
        star_temperature_k=5000.0,
        emission_path_factors=runtime.put(path_factors),
        emission_point_weights=runtime.put(geometry.emission_angle_weights),
        likelihood=likelihood,
    )
    compiled = runtime.jax.jit(
        lambda values: (model.native_eclipse_depth(values), model.loglike(values))
    )
    spectrum, loglike = compiled(
        runtime.put([0.00316, 0.12, 1.7, 1510.0, 0.38, -3.0, -4.0])
    )
    spectrum.block_until_ready()
    loglike.block_until_ready()
    _assert_metal(spectrum)
    _assert_metal(loglike)
    assert np.all(np.isfinite(np.asarray(spectrum)))
    assert np.isfinite(float(loglike))
