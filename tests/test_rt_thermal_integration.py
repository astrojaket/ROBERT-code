"""Tests for thermal source-function integration backends."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    GasOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    integrate_thermal_emission,
    solve_clear_sky_emission,
    thermal_integration_backend_name,
)
from robert_exoplanets.core import RobertValidationError


def test_thermal_integration_numpy_matches_manual_one_layer_solution() -> None:
    tau = np.array([[[0.2, 0.6]]])
    source = np.array([[4.0]])
    weights = np.array([0.25, 0.75])
    path_factors = np.array([[2.0]])
    bottom = np.array([10.0])

    result = integrate_thermal_emission(
        tau,
        source,
        weights,
        path_factors,
        bottom_source=bottom,
        backend="numpy",
    )

    slant_tau = tau[0, 0] * 2.0
    expected_layer = np.sum(source[0, 0] * (-np.expm1(-slant_tau)) * weights)
    expected_bottom = np.sum(np.exp(-slant_tau) * bottom[0] * weights)
    np.testing.assert_allclose(result.point_layer_contribution_radiance, [[[expected_layer]]])
    np.testing.assert_allclose(result.point_bottom_contribution_radiance, [[expected_bottom]])
    assert result.backend == "numpy"


def test_thermal_integration_auto_matches_numpy_reference() -> None:
    tau = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ]
    )
    source = np.array([[2.0, 3.0], [4.0, 5.0]])
    weights = np.array([0.4, 0.6])
    path_factors = np.array([[1.0, 1.0], [2.0, 1.5]])
    bottom = np.array([7.0, 8.0])

    reference = integrate_thermal_emission(
        tau,
        source,
        weights,
        path_factors,
        bottom_source=bottom,
        bottom_visible=np.array([True, False]),
        backend="numpy",
    )
    accelerated = integrate_thermal_emission(
        tau,
        source,
        weights,
        path_factors,
        bottom_source=bottom,
        bottom_visible=np.array([True, False]),
        backend="auto",
    )

    np.testing.assert_allclose(
        accelerated.point_layer_contribution_radiance,
        reference.point_layer_contribution_radiance,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        accelerated.point_bottom_contribution_radiance,
        reference.point_bottom_contribution_radiance,
        rtol=1.0e-12,
    )
    assert accelerated.backend in {"numpy", "numba"}


def test_linear_source_matches_exact_rutten_formal_integral() -> None:
    tau_value = 0.7
    mu = 0.43
    source_top = 2.0
    source_bottom = 5.0
    bottom_boundary = 7.0
    tau = np.array([[[tau_value]]])
    level_source = np.array([[source_top], [source_bottom]])

    reference = integrate_thermal_emission(
        tau,
        np.array([[(source_top + source_bottom) / 2.0]]),
        np.array([1.0]),
        np.array([[1.0 / mu]]),
        level_source_ordered=level_source,
        bottom_source=np.array([bottom_boundary]),
        backend="numpy",
    )
    accelerated = integrate_thermal_emission(
        tau,
        np.array([[999.0]]),
        np.array([1.0]),
        np.array([[1.0 / mu]]),
        level_source_ordered=level_source,
        bottom_source=np.array([bottom_boundary]),
        backend="auto",
    )

    slope = (source_bottom - source_top) / tau_value
    transmission = np.exp(-tau_value / mu)
    atmospheric = (
        source_top * (1.0 - transmission)
        + slope * (mu - (tau_value + mu) * transmission)
    )
    expected = atmospheric + bottom_boundary * transmission
    np.testing.assert_allclose(
        np.sum(reference.point_layer_contribution_radiance)
        + np.sum(reference.point_bottom_contribution_radiance),
        expected,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        accelerated.point_layer_contribution_radiance,
        reference.point_layer_contribution_radiance,
        rtol=2.0e-13,
    )


def test_emission_solver_numpy_and_auto_thermal_backends_match() -> None:
    gas_tau = _gas_tau(
        temperature=[900.0, 1200.0],
        kcoeff=np.array([[[[1.0e-23, 2.0e-23]], [[3.0e-23, 4.0e-23]]]]),
    )

    reference = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        thermal_integration_backend="numpy",
    )
    accelerated = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="none",
        thermal_integration_backend="auto",
    )

    np.testing.assert_allclose(accelerated.radiance.values, reference.radiance.values, rtol=1.0e-12)
    np.testing.assert_allclose(
        accelerated.layer_contribution_radiance,
        reference.layer_contribution_radiance,
        rtol=1.0e-12,
    )
    assert accelerated.metadata["thermal_integration_backend"] in {"numpy", "numba"}


def test_thermal_integration_rejects_unknown_backend() -> None:
    with pytest.raises(RobertValidationError, match="thermal_integration_backend"):
        thermal_integration_backend_name("fast-ish")


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
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test-correlated-k",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.25, 0.75]),
        g_weights=np.array([0.4, 0.6]),
        cache_key="test-thermal-integration-cache-key",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit="cm^2/molecule",
    )
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)
