"""Physics tests for spherical absorption transmission."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    LayerOpticalDepth,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    hydrostatic_path_geometry,
    solve_absorption_transmission,
)
from robert_exoplanets.core import RobertValidationError


def test_transparent_atmosphere_returns_opaque_base_radius() -> None:
    gas_tau, path = _one_layer_case(np.array([0.0]))
    star_radius = 7.0e8

    result = solve_absorption_transmission(
        gas_tau,
        path,
        star_radius_m=star_radius,
    )

    expected = (path.bottom_radius_m / star_radius) ** 2
    np.testing.assert_allclose(result.transit_depth.values, expected, rtol=2.0e-14)
    np.testing.assert_allclose(result.annulus_area_contribution_m2, 0.0)


def test_opaque_atmosphere_returns_top_radius() -> None:
    gas_tau, path = _one_layer_case(np.array([1.0]))
    star_radius = 7.0e8

    result = solve_absorption_transmission(
        gas_tau,
        path,
        star_radius_m=star_radius,
        impact_quadrature_order=12,
    )

    expected = (path.top_radius_m / star_radius) ** 2
    np.testing.assert_allclose(result.transit_depth.values, expected, rtol=2.0e-12)


def test_correlated_k_transmission_is_integrated_before_annulus_area() -> None:
    gas_tau, path = _one_layer_case(
        np.array([0.0, 1.0]),
        g_weights=np.array([0.25, 0.75]),
    )
    star_radius = 7.0e8

    result = solve_absorption_transmission(
        gas_tau,
        path,
        star_radius_m=star_radius,
        impact_quadrature_order=12,
    )

    expected_radius_squared = path.bottom_radius_m**2 + 0.75 * (
        path.top_radius_m**2 - path.bottom_radius_m**2
    )
    np.testing.assert_allclose(
        result.transit_depth.values,
        expected_radius_squared / star_radius**2,
        rtol=2.0e-12,
    )


def test_additional_extinction_increases_transit_depth() -> None:
    gas_tau, path = _one_layer_case(np.array([0.0]))
    clear = solve_absorption_transmission(gas_tau, path, star_radius_m=7.0e8)
    extinct = solve_absorption_transmission(
        gas_tau,
        path,
        star_radius_m=7.0e8,
        additional_optical_depths=[np.array([[1.0]])],
    )

    assert np.all(extinct.transit_depth.values > clear.transit_depth.values)


def test_transmission_rejects_mismatched_optical_depth_grid() -> None:
    gas_tau, path = _one_layer_case(np.array([0.0]))
    mismatched = LayerOpticalDepth(
        name="mismatched haze",
        tau=np.ones((1, 1)),
        pressure_grid=gas_tau.pressure_grid,
        spectral_grid=SpectralGrid.from_array(
            [3.0],
            unit="micron",
            role="opacity",
        ),
    )

    with pytest.raises(RobertValidationError, match="spectral grid"):
        solve_absorption_transmission(
            gas_tau,
            path,
            star_radius_m=7.0e8,
            additional_optical_depths=[mismatched],
        )


def test_transmission_requires_integer_impact_quadrature_order() -> None:
    gas_tau, path = _one_layer_case(np.array([0.0]))

    with pytest.raises(RobertValidationError, match="integer"):
        solve_absorption_transmission(
            gas_tau,
            path,
            star_radius_m=7.0e8,
            impact_quadrature_order=2.5,
        )


def _one_layer_case(
    kcoeff_g: np.ndarray,
    *,
    g_weights: np.ndarray | None = None,
):
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-3, 1.0]),
        centers=np.array([np.sqrt(1.0e-3)]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([1000.0]),
        temperature_edges=np.array([1000.0, 1000.0]),
        composition={"H2O": np.array([1.0])},
        mean_molecular_weight=2.3,
    )
    spectral_grid = SpectralGrid.from_array([2.0], unit="micron", role="opacity")
    weights = (
        np.full(kcoeff_g.size, 1.0 / kcoeff_g.size)
        if g_weights is None
        else np.asarray(g_weights, dtype=float)
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="transmission-test",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.linspace(0.25, 0.75, kcoeff_g.size),
        g_weights=weights,
        cache_key=f"transmission-test-{kcoeff_g.size}",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff_g.reshape(1, 1, 1, -1),
        unit="m^2/molecule",
    )
    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)
    path = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=10.0,
        reference_radius_m=1.0e8,
        reference_pressure=float(pressure_grid.centers[0]),
    )
    return gas_tau, path
