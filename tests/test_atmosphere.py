"""Tests for v0.3 atmosphere construction."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereBuilder,
    AtmosphereState,
    ConstantChemistry,
    IsothermalTemperatureProfile,
    PressureGrid,
)


def test_isothermal_profile_returns_layer_temperatures() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    profile = IsothermalTemperatureProfile(temperature=1100.0)

    temperature = profile.evaluate({}, grid)

    np.testing.assert_allclose(temperature, np.full(3, 1100.0))
    assert temperature.flags.writeable is False
    assert profile.required_parameters() == ()


def test_isothermal_profile_can_read_parameter_value() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    profile = IsothermalTemperatureProfile()

    temperature = profile.evaluate({"temperature": 950.0}, grid)

    np.testing.assert_allclose(temperature, np.array([950.0, 950.0]))
    assert profile.required_parameters() == ("temperature",)


def test_constant_chemistry_repeats_mixing_ratios_by_layer() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    temperature = np.full(3, 1000.0)
    chemistry = ConstantChemistry({"H2O": 1.0e-3, "CO": 1.0e-4})

    composition = chemistry.evaluate({}, grid, temperature)

    assert chemistry.species == ("H2O", "CO")
    np.testing.assert_allclose(composition["H2O"], np.full(3, 1.0e-3))
    assert composition["H2O"].flags.writeable is False


def test_atmosphere_builder_returns_validated_state() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=4)
    builder = AtmosphereBuilder(
        pressure_grid=grid,
        temperature_profile=IsothermalTemperatureProfile(temperature=1200.0),
        chemistry_model=ConstantChemistry({"H2O": 1.0e-3}),
        mean_molecular_weight=2.3,
    )

    atmosphere = builder.build()

    assert atmosphere.n_layers == 4
    assert atmosphere.species == ("H2O",)
    np.testing.assert_allclose(atmosphere.mean_molecular_weight, np.full(4, 2.3))


def test_atmosphere_state_rejects_shape_mismatch() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)

    with pytest.raises(ValueError, match="temperature"):
        AtmosphereState(
            pressure_grid=grid,
            temperature=np.array([1000.0, 1000.0]),
            composition={"H2O": np.full(3, 1.0e-3)},
            mean_molecular_weight=2.3,
        )
