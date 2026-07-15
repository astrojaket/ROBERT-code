"""Tests for hydrostatic radius and spherical path geometry."""

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
    hydrostatic_path_geometry,
    inverse_square_hydrostatic_path_geometry,
    normal_emission_geometry,
    solve_emission,
)
from robert_exoplanets.core import RobertValidationError

BOLTZMANN_CONSTANT_J_K = 1.380649e-23
AMU_KG = 1.66053906660e-27


def test_hydrostatic_path_geometry_anchors_reference_radius_and_pressure() -> None:
    atmosphere = _isothermal_atmosphere()
    reference_radius = 1.0e8
    reference_pressure_bar = 1.0e-2

    path = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=10.0,
        reference_radius_m=reference_radius,
        reference_pressure=reference_pressure_bar,
        reference_pressure_unit="bar",
    )

    scale_height = BOLTZMANN_CONSTANT_J_K * 1000.0 / (2.3 * AMU_KG * 10.0)
    expected_edge_radius = reference_radius - scale_height * np.log(
        atmosphere.pressure_grid.edges / reference_pressure_bar
    )
    expected_center_radius = reference_radius - scale_height * np.log(
        atmosphere.pressure_grid.centers / reference_pressure_bar
    )
    np.testing.assert_allclose(path.scale_height_m, np.full(atmosphere.n_layers, scale_height))
    np.testing.assert_allclose(path.edge_radius_m, expected_edge_radius)
    np.testing.assert_allclose(path.center_radius_m, expected_center_radius)
    np.testing.assert_allclose(path.radius_at_pressure(reference_pressure_bar * 1.0e5), reference_radius)
    assert path.top_radius_m > reference_radius
    assert path.bottom_radius_m < reference_radius


def test_hydrostatic_path_geometry_returns_spherical_shell_path_factors() -> None:
    atmosphere = _isothermal_atmosphere()
    path = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=10.0,
        reference_radius_m=1.0e8,
        reference_pressure=1.0e-2,
    )

    factors = path.emission_path_factors([1.0, 0.5])

    np.testing.assert_allclose(factors[0], np.ones(atmosphere.n_layers), rtol=1.0e-12)
    top_radius = path.top_radius_m
    impact_parameter = top_radius * np.sqrt(1.0 - 0.5**2)
    shell_inner = np.minimum(path.edge_radius_m[:-1], path.edge_radius_m[1:])
    shell_outer = np.maximum(path.edge_radius_m[:-1], path.edge_radius_m[1:])
    expected_path = np.sqrt(np.maximum(shell_outer**2 - impact_parameter**2, 0.0)) - np.sqrt(
        np.maximum(shell_inner**2 - impact_parameter**2, 0.0)
    )
    expected_path = np.where(impact_parameter < shell_outer, expected_path, 0.0)
    expected = expected_path / (shell_outer - shell_inner)
    np.testing.assert_allclose(factors[1], expected)


def test_inverse_square_geometry_decreases_gravity_with_altitude() -> None:
    atmosphere = _isothermal_atmosphere()
    reference_radius = 1.0e8
    reference_gravity = 10.0

    path = inverse_square_hydrostatic_path_geometry(
        atmosphere,
        reference_radius_m=reference_radius,
        reference_pressure=1.0,
        reference_gravity_m_s2=reference_gravity,
    )

    expected_gravity = reference_gravity * (
        reference_radius / path.center_radius_m
    ) ** 2
    np.testing.assert_allclose(
        path.gravity_m_s2,
        expected_gravity,
        rtol=2.0e-12,
    )
    assert path.gravity_m_s2[0] < path.gravity_m_s2[-1]
    assert path.metadata["gravity_model"] == (
        "inverse_square_layer_center_fixed_point"
    )
    np.testing.assert_allclose(
        path.radius_at_pressure(1.0e5),
        reference_radius,
    )


def test_inverse_square_geometry_converges_to_isothermal_analytic_radius() -> None:
    pressure_grid = PressureGrid.from_log_centers(
        10.0,
        1.0e-5,
        n_layers=80,
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.full(pressure_grid.n_layers, 1000.0),
        composition={"H2": np.ones(pressure_grid.n_layers)},
        mean_molecular_weight=2.3,
    )
    reference_radius = 1.0e8
    reference_gravity = 10.0

    path = inverse_square_hydrostatic_path_geometry(
        atmosphere,
        reference_radius_m=reference_radius,
        reference_pressure=10.0,
        reference_gravity_m_s2=reference_gravity,
    )

    gravitational_parameter = reference_gravity * reference_radius**2
    coefficient = (
        BOLTZMANN_CONSTANT_J_K
        * 1000.0
        / (gravitational_parameter * 2.3 * AMU_KG)
    )
    top_pressure = float(np.min(pressure_grid.edges))
    analytic_top_radius = 1.0 / (
        1.0 / reference_radius + coefficient * np.log(top_pressure / 10.0)
    )
    assert path.top_radius_m == pytest.approx(analytic_top_radius, abs=15.0)


def test_hydrostatic_geometry_rejects_reference_pressure_outside_grid() -> None:
    with pytest.raises(RobertValidationError, match="reference pressure"):
        hydrostatic_path_geometry(
            _isothermal_atmosphere(),
            gravity_m_s2=10.0,
            reference_radius_m=1.0e8,
            reference_pressure=10.0,
        )


def test_emission_with_normal_spherical_path_matches_plane_parallel() -> None:
    atmosphere = _isothermal_atmosphere()
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    opacity = _evaluated_opacity(
        atmosphere.pressure_grid,
        spectral_grid,
        np.array([[[[1.0e-23]], [[2.0e-23]]]]),
    )
    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)
    path = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=10.0,
        reference_radius_m=1.0e8,
        reference_pressure=1.0e-2,
    )

    plane = solve_emission(
        gas_tau,
        geometry=normal_emission_geometry(),
        bottom_boundary="blackbody",
    )
    spherical = solve_emission(
        gas_tau,
        geometry=normal_emission_geometry(),
        bottom_boundary="blackbody",
        path_geometry=path,
    )

    np.testing.assert_allclose(spherical.radiance.values, plane.radiance.values)
    np.testing.assert_allclose(spherical.layer_contribution_radiance, plane.layer_contribution_radiance)
    assert spherical.metadata["path_geometry"] == "hydrostatic_spherical_shell"


def _isothermal_atmosphere() -> AtmosphereState:
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-4, 1.0e-2, 1.0]),
        centers=np.array([1.0e-3, 1.0e-1]),
        unit="bar",
    )
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.full(pressure_grid.n_layers, 1000.0),
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-3)},
        mean_molecular_weight=2.3,
    )


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
        cache_key="test-hydrostatic-path-cache-key",
    )
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit="cm^2/molecule",
    )
