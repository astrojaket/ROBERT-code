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
    TabulatedTemperatureProfile,
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


def test_tabulated_temperature_profile_interpolates_in_log_pressure() -> None:
    grid = PressureGrid(edges=[0.5, 2.0, 20.0], centers=[1.0, 10.0])
    profile = TabulatedTemperatureProfile(
        pressure=np.array([1.0, 10.0]),
        temperature=np.array([1000.0, 2000.0]),
    )

    temperature = profile.evaluate({}, grid)

    np.testing.assert_allclose(temperature, np.array([1000.0, 2000.0]))
    assert temperature.flags.writeable is False
    assert profile.required_parameters() == ()


def test_tabulated_temperature_profile_sorts_pressure_values() -> None:
    grid = PressureGrid(edges=[0.5, 2.0, 20.0], centers=[1.0, 10.0])
    profile = TabulatedTemperatureProfile(
        pressure=np.array([10.0, 1.0]),
        temperature=np.array([2000.0, 1000.0]),
    )

    np.testing.assert_allclose(profile.pressure, np.array([1.0, 10.0]))
    np.testing.assert_allclose(profile.evaluate({}, grid), np.array([1000.0, 2000.0]))


def test_tabulated_temperature_profile_converts_pressure_units() -> None:
    grid = PressureGrid(edges=[0.5, 2.0, 20.0], centers=[1.0, 10.0], unit="bar")
    profile = TabulatedTemperatureProfile(
        pressure=np.array([1.0e5, 1.0e6]),
        temperature=np.array([1000.0, 2000.0]),
        pressure_unit="Pa",
    )

    np.testing.assert_allclose(profile.evaluate({}, grid), np.array([1000.0, 2000.0]))


def test_tabulated_temperature_profile_rejects_out_of_range_grid() -> None:
    grid = PressureGrid(edges=[0.25, 0.75, 2.0], centers=[0.5, 1.0])
    profile = TabulatedTemperatureProfile(
        pressure=np.array([1.0, 10.0]),
        temperature=np.array([1000.0, 2000.0]),
    )

    with pytest.raises(ValueError, match="outside tabulated"):
        profile.evaluate({}, grid)


def test_tabulated_temperature_profile_can_clip_out_of_range_grid() -> None:
    grid = PressureGrid(
        edges=[0.25, 0.75, 2.0, 20.0, 30.0],
        centers=[0.5, 1.0, 10.0, 20.0],
    )
    profile = TabulatedTemperatureProfile(
        pressure=np.array([1.0, 10.0]),
        temperature=np.array([1000.0, 2000.0]),
        extrapolation="clip",
    )

    np.testing.assert_allclose(
        profile.evaluate({}, grid),
        np.array([1000.0, 1000.0, 2000.0, 2000.0]),
    )


def test_tabulated_temperature_profile_loads_csv(tmp_path) -> None:
    profile_path = tmp_path / "pt.csv"
    profile_path.write_text(
        "level,pressure_bar,temperature_K\n"
        "1,1.0,1000.0\n"
        "2,10.0,2000.0\n",
        encoding="utf-8",
    )
    grid = PressureGrid(edges=[0.5, 2.0, 20.0], centers=[1.0, 10.0])

    profile = TabulatedTemperatureProfile.from_csv(profile_path)

    assert profile.source_path == profile_path
    assert profile.metadata["pressure_column"] == "pressure_bar"
    np.testing.assert_allclose(profile.evaluate({}, grid), np.array([1000.0, 2000.0]))


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
