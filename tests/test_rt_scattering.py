"""Tests for single-scattering source-function helpers."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    DirectStellarBeam,
    DiscGeometry,
    DiscPoint,
    EvaluatedCorrelatedKOpacity,
    LayerOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SingleScatteringSource,
    SpectralGrid,
    assemble_gas_optical_depth,
    rayleigh_phase_function,
    solve_clear_sky_emission,
)
from robert_exoplanets.core import RobertValidationError


def test_rayleigh_phase_function_is_back_and_forward_enhanced() -> None:
    phase = rayleigh_phase_function([0.0, 90.0, 180.0])

    np.testing.assert_allclose(phase, [1.5, 0.75, 1.5])


def test_direct_stellar_beam_blackbody_uses_finite_solid_angle() -> None:
    spectral_grid = SpectralGrid.from_array([1.0, 2.0], unit="micron", role="opacity")

    beam = DirectStellarBeam.blackbody(
        spectral_grid,
        star_temperature_k=5000.0,
        star_radius_m=7.0e8,
        semi_major_axis_m=1.0e11,
    )

    assert beam.values.shape == (2,)
    assert np.all(beam.values > 0.0)
    assert float(beam.metadata["stellar_solid_angle_sr"]) > 0.0


def test_direct_stellar_beam_accepts_covered_descending_targets() -> None:
    source = SpectralGrid.from_array([1.0, 2.0, 3.0], unit="micron")
    target = SpectralGrid.from_array([2.5, 1.5], unit="micron")
    beam = DirectStellarBeam(source, [10.0, 20.0, 30.0])

    np.testing.assert_allclose(beam.values_on(target), [25.0, 15.0])


def test_single_scattering_source_matches_one_layer_reference_solution() -> None:
    spectral_grid = SpectralGrid.from_array([1.0], unit="micron", role="opacity")
    gas_tau = _gas_tau(spectral_grid)
    scattering_tau = 0.2
    scattering = LayerOpticalDepth(
        name="test Rayleigh",
        tau=np.array([[scattering_tau]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        kind="scattering_extinction",
    )
    geometry = DiscGeometry(
        points=(
            DiscPoint(
                emission_mu=1.0,
                weight=1.0,
                stellar_mu=1.0,
                stellar_azimuth_deg=180.0,
            ),
        ),
        name="dayside_point",
    )
    beam_value = 10.0
    source = SingleScatteringSource(
        stellar_beam=DirectStellarBeam(spectral_grid=spectral_grid, values=np.array([beam_value])),
        phase_function="rayleigh",
    )

    result = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="none",
        additional_optical_depths=[scattering],
        scattering_source=source,
    )

    phase = 1.5
    expected_source = beam_value * phase / (4.0 * np.pi) * np.exp(-0.5 * scattering_tau)
    expected_contribution = expected_source * (-np.expm1(-scattering_tau))
    np.testing.assert_allclose(result.radiance.values, [expected_contribution])
    np.testing.assert_allclose(result.scattering_layer_contribution_radiance, [[expected_contribution]])
    np.testing.assert_allclose(result.point_scattering_source_function, [[[expected_source]]])
    assert result.metadata["scattering_treatment"] == "single_scattering_direct_beam"
    assert result.metadata["scattering_phase_function"] == "rayleigh"


def test_single_scattering_source_is_zero_on_nightside() -> None:
    spectral_grid = SpectralGrid.from_array([1.0], unit="micron", role="opacity")
    gas_tau = _gas_tau(spectral_grid)
    scattering = LayerOpticalDepth(
        name="test Rayleigh",
        tau=np.array([[0.2]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        kind="scattering_extinction",
    )
    geometry = DiscGeometry(
        points=(DiscPoint(emission_mu=1.0, weight=1.0, stellar_mu=-1.0, stellar_azimuth_deg=0.0),),
        name="nightside_point",
    )
    source = SingleScatteringSource(
        stellar_beam=DirectStellarBeam(spectral_grid=spectral_grid, values=np.array([10.0])),
    )

    result = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="none",
        additional_optical_depths=[scattering],
        scattering_source=source,
    )

    np.testing.assert_allclose(result.radiance.values, [0.0])
    np.testing.assert_allclose(result.scattering_layer_contribution_radiance, [[0.0]])


def test_single_scattering_source_requires_scattering_optical_depth() -> None:
    spectral_grid = SpectralGrid.from_array([1.0], unit="micron", role="opacity")
    gas_tau = _gas_tau(spectral_grid)
    geometry = DiscGeometry(
        points=(DiscPoint(emission_mu=1.0, weight=1.0, stellar_mu=1.0, stellar_azimuth_deg=180.0),),
    )
    source = SingleScatteringSource(
        stellar_beam=DirectStellarBeam(spectral_grid=spectral_grid, values=np.array([10.0])),
    )

    with pytest.raises(RobertValidationError, match="requires at least one positive scattering"):
        solve_clear_sky_emission(
            gas_tau,
            geometry=geometry,
            bottom_boundary="none",
            scattering_source=source,
        )


def test_single_scattering_source_requires_stellar_geometry() -> None:
    spectral_grid = SpectralGrid.from_array([1.0], unit="micron", role="opacity")
    gas_tau = _gas_tau(spectral_grid)
    scattering = LayerOpticalDepth(
        name="test Rayleigh",
        tau=np.array([[0.2]]),
        spectral_grid=spectral_grid,
        pressure_grid=gas_tau.pressure_grid,
        kind="scattering_extinction",
    )
    geometry = DiscGeometry(points=(DiscPoint(emission_mu=1.0, weight=1.0),))
    source = SingleScatteringSource(
        stellar_beam=DirectStellarBeam(spectral_grid=spectral_grid, values=np.array([10.0])),
    )

    with pytest.raises(RobertValidationError, match="finite geometry stellar_mu"):
        solve_clear_sky_emission(
            gas_tau,
            geometry=geometry,
            bottom_boundary="none",
            additional_optical_depths=[scattering],
            scattering_source=source,
        )


def _gas_tau(spectral_grid: SpectralGrid):
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3]),
        centers=np.array([1.0e-4]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([10.0]),
        composition={
            "H2O": np.full(pressure_grid.n_layers, 1.0e-3),
            "H2": np.full(pressure_grid.n_layers, 0.84),
            "He": np.full(pressure_grid.n_layers, 0.159),
        },
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
        cache_key="test-scattering-cache-key",
    )
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit="m^2/molecule",
    )
