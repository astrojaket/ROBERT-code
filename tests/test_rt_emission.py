"""Tests for clear-sky thermal-emission reference solver."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    GasOpticalDepth,
    LayerOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    disk_average_quadrature,
    gauss_legendre_disk_geometry,
    geometry_from_emission_angles,
    planck_radiance_wavelength,
    solve_clear_sky_emission,
)
from robert_exoplanets.core import RobertValidationError


def test_isothermal_clear_sky_with_blackbody_bottom_recovers_planck_radiance() -> None:
    temperature = 1400.0
    gas_tau = _gas_tau(
        temperature=[temperature, temperature],
        kcoeff=np.array([[[[5.0e-23, 8.0e-23]], [[2.0e-22, 4.0e-22]]]]),
    )
    mu, weights = disk_average_quadrature(4)

    result = solve_clear_sky_emission(
        gas_tau,
        emission_angle_cosines=mu,
        emission_angle_weights=weights,
        bottom_boundary="blackbody",
    )

    expected = planck_radiance_wavelength(result.radiance.spectral_grid.values, temperature)
    np.testing.assert_allclose(result.radiance.values, expected, rtol=1.0e-12)
    np.testing.assert_allclose(
        np.sum(result.layer_contribution_radiance, axis=0) + result.bottom_contribution_radiance,
        result.radiance.values,
    )
    assert result.metadata["scattering_treatment"] == "none"


def test_single_layer_without_bottom_matches_absorbing_slab_solution() -> None:
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3]),
        centers=np.array([1.0e-4]),
        unit="bar",
    )
    temperature = 1000.0
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([temperature]),
        composition={"H2O": np.array([1.0e-3])},
        mean_molecular_weight=2.3,
    )
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        np.array([[[[1.0e-28]]]]),
        unit="m^2/molecule",
    )
    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)

    result = solve_clear_sky_emission(gas_tau, bottom_boundary="none")

    source = planck_radiance_wavelength([2.0], temperature)
    expected = source * (-np.expm1(-gas_tau.total_tau[0, :, 0]))
    np.testing.assert_allclose(result.radiance.values, expected)
    np.testing.assert_allclose(result.bottom_contribution_radiance, 0.0)


def test_clear_sky_emission_accepts_additional_layer_extinction() -> None:
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3]),
        centers=np.array([1.0e-4]),
        unit="bar",
    )
    temperature = 1000.0
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([temperature]),
        composition={"H2O": np.array([1.0e-3])},
        mean_molecular_weight=2.3,
    )
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        np.array([[[[1.0e-28]]]]),
        unit="m^2/molecule",
    )
    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)
    continuum = LayerOpticalDepth(
        name="test continuum",
        tau=np.array([[0.3]]),
        spectral_grid=gas_tau.spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
    )

    result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[continuum],
    )

    source = planck_radiance_wavelength([2.0], temperature)
    expected_tau = gas_tau.total_tau[0, :, 0] + 0.3
    expected = source * (-np.expm1(-expected_tau))
    np.testing.assert_allclose(result.radiance.values, expected)
    np.testing.assert_allclose(result.total_optical_depth[0, :, 0], expected_tau)
    assert "test continuum" in result.metadata["opacity_sources"]


def test_clear_sky_emission_returns_blackbody_eclipse_depth_when_star_is_blackbody() -> None:
    planet_temperature = 1250.0
    star_temperature = 6000.0
    planet_radius = 1.4e8
    star_radius = 9.2e8
    gas_tau = _gas_tau(
        temperature=[planet_temperature, planet_temperature],
        kcoeff=np.array([[[[1.0e-22, 2.0e-22]], [[3.0e-22, 4.0e-22]]]]),
    )

    result = solve_clear_sky_emission(
        gas_tau,
        planet_radius_m=planet_radius,
        star_radius_m=star_radius,
        star_temperature_k=star_temperature,
    )

    assert result.eclipse_depth is not None
    wavelength = result.radiance.spectral_grid.values
    expected = (
        planck_radiance_wavelength(wavelength, planet_temperature)
        / planck_radiance_wavelength(wavelength, star_temperature)
        * (planet_radius / star_radius) ** 2
    )
    np.testing.assert_allclose(result.eclipse_depth.values, expected, rtol=1.0e-12)


def test_normalized_layer_contribution_sums_to_one_for_nonzero_contribution() -> None:
    gas_tau = _gas_tau(
        temperature=[900.0, 1200.0],
        kcoeff=np.array([[[[1.0e-23, 2.0e-23]], [[3.0e-23, 4.0e-23]]]]),
    )

    result = solve_clear_sky_emission(gas_tau, bottom_boundary="none")

    normalized = result.normalized_layer_contribution()
    np.testing.assert_allclose(np.sum(normalized, axis=0), np.ones(result.radiance.spectral_grid.size))


def test_disk_average_quadrature_weights_sum_to_one() -> None:
    mu, weights = disk_average_quadrature(5)

    assert mu.shape == weights.shape
    assert np.all(mu > 0.0)
    assert np.all(mu <= 1.0)
    np.testing.assert_allclose(np.sum(weights), 1.0)


def test_clear_sky_emission_geometry_matches_legacy_quadrature_inputs() -> None:
    gas_tau = _gas_tau(
        temperature=[900.0, 1200.0],
        kcoeff=np.array([[[[1.0e-23, 2.0e-23]], [[3.0e-23, 4.0e-23]]]]),
    )
    mu, weights = disk_average_quadrature(3)
    geometry = geometry_from_emission_angles(mu, weights, name="test_disc")

    legacy = solve_clear_sky_emission(
        gas_tau,
        emission_angle_cosines=mu,
        emission_angle_weights=weights,
        bottom_boundary="none",
    )
    with_geometry = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="none",
    )

    np.testing.assert_allclose(with_geometry.radiance.values, legacy.radiance.values)
    np.testing.assert_allclose(
        with_geometry.layer_contribution_radiance,
        legacy.layer_contribution_radiance,
    )
    assert with_geometry.geometry is not None
    assert with_geometry.geometry.name == "test_disc"
    assert with_geometry.point_radiance is not None
    np.testing.assert_allclose(
        np.tensordot(geometry.emission_angle_weights, with_geometry.point_radiance, axes=(0, 0)),
        with_geometry.radiance.values,
    )
    assert with_geometry.metadata["geometry"] == "test_disc"


def test_clear_sky_emission_accepts_disk_geometry_object() -> None:
    temperature = 1400.0
    gas_tau = _gas_tau(
        temperature=[temperature, temperature],
        kcoeff=np.array([[[[5.0e-23, 8.0e-23]], [[2.0e-22, 4.0e-22]]]]),
    )
    geometry = gauss_legendre_disk_geometry(4)

    result = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="blackbody",
    )

    expected = planck_radiance_wavelength(result.radiance.spectral_grid.values, temperature)
    np.testing.assert_allclose(result.radiance.values, expected, rtol=1.0e-12)
    assert result.point_layer_contribution_radiance is not None
    assert result.point_layer_contribution_radiance.shape[0] == geometry.n_points
    assert result.metadata["geometry_quadrature"] == "gauss_legendre_mu"


def test_clear_sky_emission_rejects_geometry_combined_with_legacy_angles() -> None:
    gas_tau = _gas_tau(
        temperature=[1000.0, 1000.0],
        kcoeff=np.array([[[[1.0e-23, 2.0e-23]], [[3.0e-23, 4.0e-23]]]]),
    )
    geometry = gauss_legendre_disk_geometry(2)

    with pytest.raises(RobertValidationError, match="geometry cannot be combined"):
        solve_clear_sky_emission(
            gas_tau,
            geometry=geometry,
            emission_angle_cosines=[1.0],
            emission_angle_weights=[1.0],
        )


def test_clear_sky_emission_rejects_partial_eclipse_depth_inputs() -> None:
    gas_tau = _gas_tau(
        temperature=[1000.0, 1000.0],
        kcoeff=np.array([[[[1.0e-23, 2.0e-23]], [[3.0e-23, 4.0e-23]]]]),
    )

    with pytest.raises(RobertValidationError, match="all required for eclipse depth"):
        solve_clear_sky_emission(gas_tau, planet_radius_m=1.0)


def _gas_tau(
    *,
    temperature: list[float],
    kcoeff: np.ndarray,
) -> GasOpticalDepth:
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 1.0e-1]),
        centers=np.array([1.0e-4, 1.0e-2]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.asarray(temperature, dtype=float),
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-3)},
        mean_molecular_weight=2.3,
    )
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    opacity = _evaluated_opacity(pressure_grid, spectral_grid, kcoeff)
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


def _evaluated_opacity(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    kcoeff: np.ndarray,
    *,
    unit: str = "cm^2/molecule",
) -> EvaluatedCorrelatedKOpacity:
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test-correlated-k",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.25, 0.75]) if kcoeff.shape[-1] == 2 else np.array([0.5]),
        g_weights=np.array([0.4, 0.6]) if kcoeff.shape[-1] == 2 else np.array([1.0]),
        cache_key="test-clear-sky-cache-key",
    )
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit=unit,
    )
