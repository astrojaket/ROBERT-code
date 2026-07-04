"""Tests for gas optical-depth assembly and tau diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
)
from robert_exoplanets.core import RobertValidationError

AMU_KG = 1.66053906660e-27


def test_assemble_gas_optical_depth_uses_hydrostatic_column_and_cm2_conversion() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1000.0, 2000.0], unit="cm^-1", role="opacity")
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={"H2O": np.array([1.0e-3, 2.0e-3])},
        mean_molecular_weight=np.array([2.3, 2.4]),
    )
    kcoeff = np.array([[[[2.0e-24, 3.0e-24], [4.0e-24, 5.0e-24]], [[6.0e-24, 7.0e-24], [8.0e-24, 9.0e-24]]]])
    opacity = _evaluated_opacity(pressure_grid, spectral_grid, ("H2O",), kcoeff)

    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)

    delta_pressure_pa = np.abs(np.diff(pressure_grid.edges)) * 1.0e5
    total_column = delta_pressure_pa / (atmosphere.mean_molecular_weight * AMU_KG * 10.0)
    species_column = atmosphere.composition["H2O"] * total_column
    expected_tau = kcoeff[0] * 1.0e-4 * species_column[:, None, None]
    np.testing.assert_allclose(gas_tau.layer_pressure_thickness_pa, delta_pressure_pa)
    np.testing.assert_allclose(gas_tau.layer_column_density_molecules_m2, total_column)
    np.testing.assert_allclose(gas_tau.species_column_density_molecules_m2[0], species_column)
    np.testing.assert_allclose(gas_tau.species_tau[0], expected_tau)
    np.testing.assert_allclose(gas_tau.total_tau, expected_tau)
    assert gas_tau.metadata["column_model"] == "hydrostatic_plane_parallel"


def test_assemble_gas_optical_depth_sums_multiple_species() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1000.0], unit="cm^-1", role="opacity")
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={
            "H2O": np.array([1.0e-3, 1.0e-3]),
            "CO": np.array([2.0e-4, 3.0e-4]),
        },
        mean_molecular_weight=2.3,
    )
    kcoeff = np.array(
        [
            [[[1.0e-28, 2.0e-28]], [[3.0e-28, 4.0e-28]]],
            [[[5.0e-28, 6.0e-28]], [[7.0e-28, 8.0e-28]]],
        ]
    )
    opacity = _evaluated_opacity(pressure_grid, spectral_grid, ("H2O", "CO"), kcoeff, unit="m^2/molecule")

    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=np.array([8.0, 12.0]),
    )

    np.testing.assert_allclose(gas_tau.total_tau, np.sum(gas_tau.species_tau, axis=0))
    np.testing.assert_allclose(gas_tau.gravity_m_s2, [8.0, 12.0])
    assert gas_tau.species == ("H2O", "CO")


def test_gas_optical_depth_rejects_missing_composition_species() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1000.0], unit="cm^-1", role="opacity")
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={"H2O": np.array([1.0e-3, 1.0e-3])},
        mean_molecular_weight=2.3,
    )
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        ("H2O", "CO"),
        np.ones((2, 2, 1, 2)) * 1.0e-24,
    )

    with pytest.raises(RobertValidationError, match="missing opacity species: CO"):
        assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


def test_gas_optical_depth_rejects_pressure_grid_mismatch() -> None:
    pressure_grid = _pressure_grid()
    opacity_pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 2.0e-1]),
        centers=np.array([1.0e-4, 2.0e-2]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([1000.0], unit="cm^-1", role="opacity")
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={"H2O": np.array([1.0e-3, 1.0e-3])},
        mean_molecular_weight=2.3,
    )
    opacity = _evaluated_opacity(
        opacity_pressure_grid,
        spectral_grid,
        ("H2O",),
        np.ones((1, 2, 1, 2)) * 1.0e-24,
    )

    with pytest.raises(RobertValidationError, match="pressure grids must match"):
        assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


def test_gas_optical_depth_rejects_unsupported_units_and_conventions() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1000.0], unit="cm^-1", role="opacity")
    mass_fraction_atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={"H2O": np.array([1.0e-3, 1.0e-3])},
        mean_molecular_weight=2.3,
        composition_convention="mass_fraction",
    )
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        ("H2O",),
        np.ones((1, 2, 1, 2)) * 1.0e-24,
    )

    with pytest.raises(RobertValidationError, match="requires VMR composition"):
        assemble_gas_optical_depth(mass_fraction_atmosphere, opacity, gravity_m_s2=10.0)

    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={"H2O": np.array([1.0e-3, 1.0e-3])},
        mean_molecular_weight=2.3,
    )
    bad_unit_opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        ("H2O",),
        np.ones((1, 2, 1, 2)) * 1.0e-24,
        unit="barn/molecule",
    )

    with pytest.raises(RobertValidationError, match="unsupported opacity unit"):
        assemble_gas_optical_depth(atmosphere, bad_unit_opacity, gravity_m_s2=10.0)


def test_tau_diagnostics_respect_top_of_atmosphere_for_decreasing_pressure_grid() -> None:
    pressure_grid = PressureGrid(
        edges=np.array([1.0, 1.0e-2, 1.0e-4]),
        centers=np.array([1.0e-1, 1.0e-3]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([1000.0], unit="cm^-1", role="opacity")
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([1200.0, 900.0]),
        composition={"H2O": np.array([1.0e-3, 1.0e-3])},
        mean_molecular_weight=2.3,
    )
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        ("H2O",),
        np.array([[[[4.0e-24, 8.0e-24]], [[1.0e-24, 2.0e-24]]]]),
    )
    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)

    cumulative = gas_tau.cumulative_tau_from_top()
    above = gas_tau.tau_above_layer()

    np.testing.assert_allclose(cumulative[1], gas_tau.total_tau[1])
    np.testing.assert_allclose(cumulative[0], gas_tau.total_tau[1] + gas_tau.total_tau[0])
    np.testing.assert_allclose(above[1], 0.0)
    np.testing.assert_allclose(above[0], gas_tau.total_tau[1])


def test_layer_transmission_weighting_can_integrate_and_normalize_g_ordinates() -> None:
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([1000.0], unit="cm^-1", role="opacity")
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={"H2O": np.array([1.0e-3, 1.0e-3])},
        mean_molecular_weight=2.3,
    )
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        ("H2O",),
        np.array([[[[1.0e-23, 2.0e-23]], [[3.0e-23, 4.0e-23]]]]),
    )
    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)

    raw = gas_tau.layer_transmission_weighting(integrate_g=False)
    integrated = gas_tau.layer_transmission_weighting()
    normalized = gas_tau.layer_transmission_weighting(normalize=True)

    expected_raw = np.exp(-gas_tau.tau_above_layer()) * (-np.expm1(-gas_tau.total_tau))
    expected_integrated = np.sum(expected_raw * np.array([0.4, 0.6])[None, None, :], axis=-1)
    np.testing.assert_allclose(raw, expected_raw)
    np.testing.assert_allclose(integrated, expected_integrated)
    np.testing.assert_allclose(np.sum(normalized, axis=0), np.ones(spectral_grid.size))


def _pressure_grid() -> PressureGrid:
    return PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 1.0e-1]),
        centers=np.array([1.0e-4, 1.0e-2]),
        unit="bar",
    )


def _evaluated_opacity(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    species: tuple[str, ...],
    kcoeff: np.ndarray,
    *,
    unit: str = "cm^2/molecule",
) -> EvaluatedCorrelatedKOpacity:
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test-correlated-k",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=species,
        g_samples=np.array([0.25, 0.75]),
        g_weights=np.array([0.4, 0.6]),
        cache_key="test-correlated-k-cache-key",
    )
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit=unit,
    )
