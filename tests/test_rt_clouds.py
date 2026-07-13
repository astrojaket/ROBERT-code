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
    grey_cloud_from_mass_extinction,
    planck_radiance_wavelength,
    power_law_haze,
    solve_clear_sky_emission,
    solve_clear_sky_emission_spectrum,
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


def test_grey_mass_extinction_uses_hydrostatic_bulk_mass_column() -> None:
    spectral_grid = SpectralGrid.from_array([2.0, 5.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)

    cloud = grey_cloud_from_mass_extinction(
        gas_tau,
        mass_extinction_cm2_g=2.0,
        single_scattering_albedo=1.0,
        asymmetry_factor=0.0,
    )

    expected = 2.0 * 0.1 * gas_tau.layer_pressure_thickness_pa / gas_tau.gravity_m_s2
    np.testing.assert_allclose(
        cloud.extinction_tau,
        np.repeat(expected[:, None], spectral_grid.size, axis=1),
    )
    np.testing.assert_allclose(cloud.single_scattering_albedo, 1.0)
    assert cloud.metadata["hydrostatic_conversion"] == (
        "tau=kappa*delta_pressure/gravity"
    )


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
    assert (
        result.metadata["scattering_treatment"]
        == "extinction_only_no_scattering_source"
    )
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


def test_two_stream_backend_solves_coupled_mixed_absorption_scattering() -> None:
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

    extinction_only = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[cloud],
    )

    np.testing.assert_allclose(result.total_optical_depth[0, :, 0], [0.5])
    np.testing.assert_allclose(result.extinction_optical_depth[0, :, 0], [0.5])
    assert np.all(result.radiance.values > 0.0)
    assert np.all(result.radiance.values < extinction_only.radiance.values)
    assert (
        result.metadata["scattering_treatment"]
        == "toon_hemispheric_mean_thermal_two_stream"
    )
    assert result.metadata["multiple_scattering_applied"] == "true"
    assert result.metadata["total_optical_depth_role"] == "extinction"
    assert (
        result.metadata["thermal_integration_backend"] == "numpy_toon_hemispheric_mean"
    )


def test_two_stream_one_layer_preserves_physical_tau_and_contributions() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=900.0)
    cloud = CloudOpticalProperties(
        name="analytic one-layer cloud",
        extinction_tau=np.array([[0.8]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.25,
        asymmetry_factor=0.4,
    )

    result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="two_stream",
    )

    np.testing.assert_allclose(result.total_optical_depth[0, 0, 0], 0.8)
    np.testing.assert_allclose(
        result.radiance.values,
        np.sum(result.layer_contribution_radiance, axis=0)
        + result.bottom_contribution_radiance,
    )


def test_sh4_backend_uses_higher_order_multiple_scattering() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)
    cloud = CloudOpticalProperties(
        name="forward-scattering cloud",
        extinction_tau=np.array([[0.8]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.9,
        asymmetry_factor=0.7,
    )

    result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="sh4",
    )

    assert np.all(result.radiance.values > 0.0)
    assert (
        result.metadata["multiple_scattering_backend"]
        == "sh4_henyey_greenstein_delta_m"
    )
    assert result.metadata["scattering_treatment"] == (
        "rooney_p3_sh4_mixed_phase_moments_delta_m"
    )
    assert result.metadata["thermal_integration_backend"] == (
        "scipy_banded_sh4_phase_moments_delta_m"
    )
    np.testing.assert_allclose(
        result.radiance.values,
        np.sum(result.layer_contribution_radiance, axis=0)
        + result.bottom_contribution_radiance,
    )


def test_sh4_spectrum_only_backend_matches_diagnostic_solver() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)
    cloud = CloudOpticalProperties(
        name="spectrum-only cloud",
        extinction_tau=np.array([[0.8]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.9,
        asymmetry_factor=0.7,
    )
    reference = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="blackbody",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="sh4",
    )
    spectrum = solve_clear_sky_emission_spectrum(
        gas_tau,
        bottom_boundary="blackbody",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="sh4",
        thermal_integration_backend="numpy",
    )

    np.testing.assert_allclose(spectrum.values, reference.radiance.values, rtol=2.0e-13)
    assert spectrum.metadata["rt_solver"] == "sh4_spectrum_only"
    assert spectrum.metadata["diagnostics"] == "disabled"
    assert spectrum.metadata["multiple_scattering_applied"] == "true"


def test_sh4_uses_supplied_non_hg_phase_moments() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)
    exact_moments = np.array([1.0, 1.5, 1.0, 0.5, 0.2])[:, None]
    exact = CloudOpticalProperties(
        name="explicit moment cloud",
        extinction_tau=np.array([[0.8]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.85,
        asymmetry_factor=0.5,
        phase_function_moments=exact_moments,
    )
    hg = CloudOpticalProperties(
        name="HG cloud",
        extinction_tau=np.array([[0.8]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.85,
        asymmetry_factor=0.5,
    )

    exact_result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="blackbody",
        additional_optical_depths=[exact],
        multiple_scattering_backend="sh4",
    )
    hg_result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="blackbody",
        additional_optical_depths=[hg],
        multiple_scattering_backend="sh4",
    )

    assert not np.allclose(
        exact_result.radiance.values, hg_result.radiance.values, rtol=1.0e-5
    )


def test_cloud_split_preserves_phase_moments_for_sh4() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    gas_tau = _zero_gas_tau(spectral_grid=spectral_grid, temperature=1000.0)
    moments = np.array([1.0, 1.2, 0.8, 0.3, 0.1])[:, None]
    cloud = CloudOpticalProperties(
        name="moment cloud",
        extinction_tau=np.array([[0.5]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        single_scattering_albedo=0.7,
        asymmetry_factor=0.4,
        phase_function_moments=moments,
    )

    direct = solve_clear_sky_emission(
        gas_tau,
        additional_optical_depths=[cloud],
        multiple_scattering_backend="sh4",
    )
    split = solve_clear_sky_emission(
        gas_tau,
        additional_optical_depths=list(cloud.as_layer_optical_depths()),
        multiple_scattering_backend="sh4",
    )

    np.testing.assert_allclose(
        split.radiance.values, direct.radiance.values, rtol=2.0e-13
    )


def test_two_stream_two_layer_uses_explicit_level_temperatures() -> None:
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 1.0e-1]),
        centers=np.array([1.0e-4, 1.0e-2]),
        unit="bar",
    )
    gas_tau = _zero_gas_tau_for_profile(
        pressure_grid=pressure_grid,
        spectral_grid=spectral_grid,
        temperature=np.array([700.0, 1100.0]),
    )
    extinction = np.array([[0.2], [0.7]])
    cloud = CloudOpticalProperties(
        name="analytic two-layer cloud",
        extinction_tau=extinction,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=np.array([0.5, 0.2]),
        asymmetry_factor=np.array([0.0, 0.5]),
    )

    result = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="two_stream",
    )

    np.testing.assert_allclose(result.total_optical_depth[:, 0, 0], extinction[:, 0])
    assert gas_tau.atmosphere.temperature_edges is not None
    assert np.all(result.layer_contribution_radiance >= 0.0)
    np.testing.assert_allclose(
        result.radiance.values,
        np.sum(result.layer_contribution_radiance, axis=0)
        + result.bottom_contribution_radiance,
    )


def test_two_stream_diagnostics_preserve_no_scattering_limit() -> None:
    total = np.array([[[0.4], [0.8]]])
    diagnostics = two_stream_scattering_diagnostics(total, np.zeros_like(total))

    np.testing.assert_allclose(diagnostics.effective_tau, total)
    np.testing.assert_allclose(diagnostics.single_scattering_albedo, 0.0)
    assert (
        diagnostics.metadata["closure"] == "two_stream_effective_extinction_reference"
    )


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
    return _zero_gas_tau_for_profile(
        pressure_grid=pressure_grid,
        spectral_grid=spectral_grid,
        temperature=np.array([temperature]),
    )


def _zero_gas_tau_for_profile(
    *,
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    temperature: np.ndarray,
):
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=np.interp(
            np.log(pressure_grid.edges),
            np.log(pressure_grid.centers),
            temperature,
        ),
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-3)},
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
