"""Tests for retrieval-facing temperature profile parameterizations."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    MadhusudhanSeager2009TemperatureProfile,
    ParmentierGuillot2014TemperatureProfile,
    PressureGrid,
    SplineTemperatureProfile,
)


def test_spline_temperature_profile_passes_through_fixed_knots() -> None:
    grid = PressureGrid(edges=[0.5, 2.0, 20.0, 200.0], centers=[1.0, 10.0, 100.0])
    profile = SplineTemperatureProfile(
        knot_pressure=np.array([1.0, 10.0, 100.0]),
        knot_temperature=np.array([1000.0, 1500.0, 2000.0]),
    )

    temperature = profile.evaluate({}, grid)

    np.testing.assert_allclose(temperature, np.array([1000.0, 1500.0, 2000.0]))
    assert temperature.flags.writeable is False
    assert profile.required_parameters() == ()


def test_spline_temperature_profile_orders_retrieval_parameters_with_pressure() -> None:
    grid = PressureGrid(edges=[0.5, 2.0, 20.0, 200.0], centers=[1.0, 10.0, 100.0])
    profile = SplineTemperatureProfile(
        knot_pressure=np.array([100.0, 1.0, 10.0]),
        parameter_names=("T_deep", "T_top", "T_mid"),
    )

    assert profile.required_parameters() == ("T_top", "T_mid", "T_deep")
    temperature = profile.evaluate(
        {"T_top": 1000.0, "T_mid": 1500.0, "T_deep": 2000.0},
        grid,
    )

    np.testing.assert_allclose(temperature, np.array([1000.0, 1500.0, 2000.0]))


def test_spline_temperature_profile_rejects_out_of_range_grid() -> None:
    grid = PressureGrid(edges=[0.25, 0.75, 2.0], centers=[0.5, 1.0])
    profile = SplineTemperatureProfile(
        knot_pressure=np.array([1.0, 10.0]),
        knot_temperature=np.array([1000.0, 2000.0]),
    )

    with pytest.raises(ValueError, match="outside spline"):
        profile.evaluate({}, grid)


def test_madhusudhan_seager_2009_profile_matches_piecewise_form() -> None:
    grid = PressureGrid(
        edges=[5.0e-7, 5.0e-5, 5.0e-1, 5.0, 20.0],
        centers=[1.0e-6, 1.0e-2, 1.0, 10.0],
    )
    profile = MadhusudhanSeager2009TemperatureProfile()
    parameters = {
        "P1": -2.0,
        "P2": -3.0,
        "P3": 0.3,
        "T0": 1200.0,
        "alpha1": 0.5,
        "alpha2": 0.25,
    }

    temperature = profile.evaluate(parameters, grid)

    p0 = 1.0e-6
    p1 = 10.0 ** parameters["P1"]
    p2 = 10.0 ** parameters["P2"]
    p3 = 10.0 ** parameters["P3"]
    t0 = parameters["T0"]
    alpha1 = parameters["alpha1"]
    alpha2 = parameters["alpha2"]
    t2 = ((1.0 / alpha1) * np.log10(p1 / p0)) ** 2
    t2 -= ((1.0 / alpha2) * np.log10(p1 / p2)) ** 2
    t2 += t0
    t3 = ((1.0 / alpha2) * np.log10(p3 / p2)) ** 2 + t2
    expected = np.array(
        [
            t0,
            ((1.0 / alpha2) * np.log10(1.0e-2 / p2)) ** 2 + t2,
            ((1.0 / alpha2) * np.log10(1.0 / p2)) ** 2 + t2,
            t3,
        ]
    )

    np.testing.assert_allclose(temperature, expected)
    assert temperature.flags.writeable is False
    assert profile.required_parameters() == ("P1", "P2", "P3", "T0", "alpha1", "alpha2")


def test_madhusudhan_seager_2009_profile_rejects_invalid_transition_order() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    profile = MadhusudhanSeager2009TemperatureProfile()

    with pytest.raises(ValueError, match="P3 must be deeper than P1"):
        profile.evaluate(
            {
                "P1": -1.0,
                "P2": -3.0,
                "P3": -2.0,
                "T0": 1200.0,
                "alpha1": 0.5,
                "alpha2": 0.25,
            },
            grid,
        )


def test_parmentier_guillot_2014_profile_returns_positive_temperatures() -> None:
    grid = PressureGrid.logspace(1.0e-6, 100.0, n_layers=8)
    profile = ParmentierGuillot2014TemperatureProfile(
        gravity=10.0,
        internal_temperature=200.0,
    )

    temperature = profile.evaluate(
        {
            "kappa_IR": 0.02,
            "gamma1": 0.5,
            "gamma2": 1.5,
            "T_irr": 1500.0,
            "alpha": 0.5,
        },
        grid,
    )

    assert temperature.shape == (8,)
    assert np.all(np.isfinite(temperature))
    assert np.all(temperature > 0.0)
    assert temperature.flags.writeable is False
    assert profile.required_parameters() == ("kappa_IR", "gamma1", "gamma2", "T_irr", "alpha")


def test_parmentier_guillot_2014_profile_alpha_is_irrelevant_for_equal_channels() -> None:
    grid = PressureGrid.logspace(1.0e-6, 1.0, n_layers=5)
    profile = ParmentierGuillot2014TemperatureProfile(gravity=12.0)
    base_parameters = {
        "kappa_IR": 0.01,
        "gamma1": 0.7,
        "gamma2": 0.7,
        "T_irr": 1400.0,
    }

    alpha_zero = profile.evaluate({**base_parameters, "alpha": 0.0}, grid)
    alpha_one = profile.evaluate({**base_parameters, "alpha": 1.0}, grid)

    np.testing.assert_allclose(alpha_zero, alpha_one)


def test_parmentier_guillot_2014_profile_can_retrieve_gravity_and_internal_temperature() -> None:
    profile = ParmentierGuillot2014TemperatureProfile(
        gravity=None,
        internal_temperature=None,
    )

    assert profile.required_parameters() == (
        "kappa_IR",
        "gamma1",
        "gamma2",
        "T_irr",
        "alpha",
        "gravity",
        "T_int",
    )
