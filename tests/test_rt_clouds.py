"""Tests for cloud/aerosol optical properties and scattering RT hooks."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    CloudOpticalProperties,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    grey_cloud_deck,
    planck_radiance_wavelength,
    power_law_haze,
    solve_clear_sky_emission,
    two_stream_scattering_diagnostics,
)
from robert_exoplanets.core import RobertValidationError


def test_cloud_optical_properties_split_absorption_and_scattering_components() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1.0, 2.0], unit="micron", role="opacity")
    cloud = CloudOpticalProperties(
        name="test cloud",
        extinction_tau=np.array([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2]]),
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=0.25,
        asymmetry_factor=np.array([0.0, 0.5]),
    )

    np.testing.assert_allclose(cloud.absorption_tau, cloud.extinction_tau * 0.75)
    np.testing.assert_allclose(cloud.scattering_tau, cloud.extinction_tau * 0.25)
    np.testing.assert_allclose(
        cloud.transport_scattering_tau,
        cloud.scattering_tau * np.array([[1.0, 0.5]]),
    )

    absorption, scattering = cloud.as_layer_optical_depths()
    assert absorption.kind == "cloud_absorption"
    assert scattering.kind == "cloud_scattering_extinction"
    assert scattering.metadata["cloud_name"] == "test cloud"


def test_cloud_optical_properties_validate_bounds() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1.0], unit="micron", role="opacity")

    with pytest.raises(RobertValidationError, match="single_scattering_albedo"):
        CloudOpticalProperties(
            name="bad cloud",
            extinction_tau=np.ones((pressure_grid.n_layers, spectral_grid.size)),
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            single_scattering_albedo=1.2,
        )

    with pytest.raises(RobertValidationError, match="asymmetry_factor"):
        CloudOpticalProperties(
            name="bad cloud",
            extinction_tau=np.ones((pressure_grid.n_layers, spectral_grid.size)),
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            asymmetry_factor=-1.2,
        )


def test_grey_cloud_deck_distributes_tau_below_cloud_top() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1.0, 2.0], unit="micron", role="opacity")

    cloud = grey_cloud_deck(
        pressure_grid,
        spectral_grid,
        cloud_top_pressure=1.0e-3,
        cloud_top_pressure_unit="bar",
        optical_depth=0.6,
    )

    expected = np.array([[0.0, 0.0], [0.3, 0.3], [0.3, 0.3]])
    np.testing.assert_allclose(cloud.extinction_tau, expected)
    assert cloud.metadata["vertical_model"] == "grey_cloud_deck_uniform_tau_below_top"


def test_power_law_haze_scales_with_wavelength() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1.0, 2.0], unit="micron", role="opacity")

    haze = power_law_haze(
        pressure_grid,
        spectral_grid,
        optical_depth_at_reference=0.9,
        reference_wavelength_micron=1.0,
        slope=-4.0,
    )

    np.testing.assert_allclose(np.sum(haze.extinction_tau, axis=0), [0.9, 0.9 / 16.0])
    np.testing.assert_allclose(haze.single_scattering_albedo, 1.0)
    assert haze.metadata["spectral_model"] == "power_law"


def test_emission_solver_accepts_cloud_optical_properties() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)
    cloud = CloudOpticalProperties(
        name="test cloudy slab",
        extinction_tau=np.array([[0.5]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.2,
    )

    result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[cloud],
    )

    expected = planck_radiance_wavelength([2.0], 1000.0) * (-np.expm1(-0.5))
    np.testing.assert_allclose(result.radiance.values, expected)
    np.testing.assert_allclose(result.extinction_optical_depth[0, :, 0], [0.5])
    np.testing.assert_allclose(result.total_optical_depth[0, :, 0], [0.5])
    assert result.metadata["scattering_treatment"] == "extinction_only_no_scattering_source"
    assert result.metadata["total_optical_depth_role"] == "extinction"


def test_cloud_object_and_split_layer_optical_depths_are_equivalent() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)
    cloud = CloudOpticalProperties(
        name="test cloudy slab",
        extinction_tau=np.array([[0.5]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.2,
    )

    direct = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[cloud],
    )
    split = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=list(cloud.as_layer_optical_depths()),
    )

    np.testing.assert_allclose(split.radiance.values, direct.radiance.values)
    np.testing.assert_allclose(split.total_optical_depth, direct.total_optical_depth)


def test_two_stream_backend_increases_effective_tau_for_mixed_absorption_scattering() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)
    cloud = CloudOpticalProperties(
        name="test cloudy slab",
        extinction_tau=np.array([[0.5]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.5,
        asymmetry_factor=0.0,
    )

    result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="two_stream",
    )

    expected_tau = 0.5 + np.sqrt(3.0 * 0.25 * 0.25)
    expected = planck_radiance_wavelength([2.0], 1000.0) * (-np.expm1(-expected_tau))
    np.testing.assert_allclose(result.total_optical_depth[0, :, 0], [expected_tau])
    np.testing.assert_allclose(result.extinction_optical_depth[0, :, 0], [0.5])
    np.testing.assert_allclose(result.radiance.values, expected)
    assert result.metadata["scattering_treatment"] == "two_stream_multiple_scattering_reference"
    assert result.metadata["multiple_scattering_applied"] == "true"
    assert result.metadata["total_optical_depth_role"] == "two_stream_effective_extinction"


def test_two_stream_diagnostics_preserve_no_scattering_limit() -> None:
    total = np.array([[[0.4], [0.8]]])
    diagnostics = two_stream_scattering_diagnostics(total, np.zeros_like(total))

    np.testing.assert_allclose(diagnostics.effective_tau, total)
    np.testing.assert_allclose(diagnostics.single_scattering_albedo, 0.0)
    assert diagnostics.metadata["closure"] == "two_stream_effective_extinction_reference"


def test_emission_solver_rejects_unknown_multiple_scattering_backend() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)

    with pytest.raises(RobertValidationError, match="multiple_scattering_backend"):
        solve_clear_sky_emission(
            gas_tau,
            bottom_boundary="none",
            multiple_scattering_backend="fancy",
        )


def _pressure_grid() -> PressureGrid:
    return PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 1.0e-1, 1.0]),
        centers=np.array([1.0e-4, 1.0e-2, 5.0e-1]),
        unit="bar",
    )


def _zero_gas_tau(*, spectral_grid: SpectralGrid, temperature: float):
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3]),
        centers=np.array([1.0e-4]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([temperature]),
        composition={"H2O": np.array([1.0e-3])},
        mean_molecular_weight=2.3,
    )
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        np.zeros((1, pressure_grid.n_layers, spectral_grid.size, 1)),
    )
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


def _evaluated_opacity(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    kcoeff: np.ndarray,
) -> EvaluatedCorrelatedKOpacity:
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test-correlated-k",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="test-cloud-cache-key",
    )
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit="m^2/molecule",
    )
