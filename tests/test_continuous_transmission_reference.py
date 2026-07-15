"""Tests for the independent continuous transmission reference."""

from __future__ import annotations

import numpy as np
import pytest

from examples.benchmark_continuous_transmission_reference import (
    _continuous_transit_depth,
    _contract,
    _evaluate_robert,
    _pressure_at_radius,
    _radius_at_pressure,
)
from examples.benchmark_continuous_pt_cloud_reference import (
    _base_contract as _pt_cloud_contract,
    _continuous_transit_depth as _pt_cloud_continuous_depth,
    _pressure_edges as _pt_cloud_pressure_edges,
    _radius_from_pressure as _pt_cloud_radius_from_pressure,
    _state_at_radius as _pt_cloud_state_at_radius,
    _temperature_from_u,
)


def test_inverse_square_pressure_radius_mapping_round_trips() -> None:
    contract = _contract((32,))
    pressure_pa = np.geomspace(1.0e-4, 1.0e6, 41)

    radius_m = _radius_at_pressure(pressure_pa, contract)

    np.testing.assert_allclose(
        _pressure_at_radius(radius_m, contract),
        pressure_pa,
        rtol=5.0e-14,
    )


def test_continuous_reference_is_numerically_converged() -> None:
    contract = _contract((32,))

    order_128 = _continuous_transit_depth(contract, 128)
    order_256 = _continuous_transit_depth(contract, 256)

    np.testing.assert_allclose(order_128, order_256, atol=5.0e-13, rtol=0.0)


def test_robert_shell_discretization_converges_to_continuous_reference() -> None:
    contract = _contract((32, 64))
    reference = _continuous_transit_depth(contract, 256)
    error_32 = np.sqrt(np.mean((_evaluate_robert(contract, 32) - reference) ** 2))
    error_64 = np.sqrt(np.mean((_evaluate_robert(contract, 64) - reference) ** 2))

    assert error_64 < error_32 / 3.0


def test_nonisothermal_pressure_radius_mapping_and_temperature_endpoints() -> None:
    contract = _pt_cloud_contract((40,))
    pressure_pa = np.geomspace(1.0e-4, 1.0e6, 51)

    radius_m = _pt_cloud_radius_from_pressure(pressure_pa, contract)
    recovered_pressure, _ = _pt_cloud_state_at_radius(radius_m, contract)

    np.testing.assert_allclose(recovered_pressure, pressure_pa, rtol=8.0e-14)
    assert _temperature_from_u(0.0, contract) == 1600.0
    assert _temperature_from_u(1.0, contract) == pytest.approx(800.0)
    assert np.min(_temperature_from_u(np.linspace(0.0, 1.0, 101), contract)) > 0.0


def test_aligned_grid_contains_cloud_edge_and_uniform_grid_does_not() -> None:
    contract = _pt_cloud_contract((80,))
    cloud_pressure = float(contract["cloud_top_pressure_bar"])

    aligned = _pt_cloud_pressure_edges(contract, 80, "aligned")
    misaligned = _pt_cloud_pressure_edges(contract, 80, "misaligned")

    assert np.any(aligned == cloud_pressure)
    assert not np.any(np.isclose(misaligned, cloud_pressure, rtol=1.0e-12))


def test_sharp_cloud_continuous_reference_is_converged() -> None:
    contract = _pt_cloud_contract((40,))

    order_128 = _pt_cloud_continuous_depth(contract, 128, cloudy=True)
    order_256 = _pt_cloud_continuous_depth(contract, 256, cloudy=True)

    np.testing.assert_allclose(order_128, order_256, atol=2.0e-8, rtol=0.0)
