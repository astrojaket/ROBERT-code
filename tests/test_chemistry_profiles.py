"""Tests for retrieval-facing chemistry parameterizations."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereBuilder,
    BackgroundGasMixture,
    FreeChemistry,
    IsothermalTemperatureProfile,
    PressureGrid,
)


def test_background_gas_mixture_normalizes_relative_fractions() -> None:
    mixture = BackgroundGasMixture({"H2": 3.0, "He": 1.0})

    assert mixture.species == ("H2", "He")
    assert mixture.fractions == {"H2": 0.75, "He": 0.25}


def test_background_gas_mixture_builds_standard_hydrogen_helium_split() -> None:
    mixture = BackgroundGasMixture.hydrogen_helium(h2_fraction=0.85)

    assert mixture.fractions == {"H2": 0.85, "He": pytest.approx(0.15)}


def test_free_chemistry_reads_linear_parameters_and_fills_background() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    temperature = np.full(3, 1200.0)
    chemistry = FreeChemistry(
        active_species=("H2O", "CO"),
        background=BackgroundGasMixture({"H2": 3.0, "He": 1.0}),
    )

    composition = chemistry.evaluate({"H2O": 1.0e-3, "CO": 2.0e-3}, grid, temperature)

    assert chemistry.species == ("H2O", "CO", "H2", "He")
    assert chemistry.required_parameters() == ("H2O", "CO")
    np.testing.assert_allclose(composition["H2O"], np.full(3, 1.0e-3))
    np.testing.assert_allclose(composition["CO"], np.full(3, 2.0e-3))
    np.testing.assert_allclose(composition["H2"], np.full(3, 0.997 * 0.75))
    np.testing.assert_allclose(composition["He"], np.full(3, 0.997 * 0.25))
    assert composition["H2O"].flags.writeable is False


def test_free_chemistry_can_use_log10_parameters() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    temperature = np.full(2, 1200.0)
    chemistry = FreeChemistry(
        active_species=("H2O",),
        parameter_mode="log10",
    )

    composition = chemistry.evaluate({"H2O": -3.0}, grid, temperature)

    np.testing.assert_allclose(composition["H2O"], np.full(2, 1.0e-3))


def test_free_chemistry_supports_fixed_species_and_custom_parameter_names() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    temperature = np.full(2, 1200.0)
    chemistry = FreeChemistry(
        active_species=("H2O", "CO2"),
        fixed_mixing_ratios={"CO2": 2.0e-4},
        parameter_names={"H2O": "log_H2O"},
        parameter_mode="log10",
        background=BackgroundGasMixture({"H2": 1.0}),
    )

    composition = chemistry.evaluate({"log_H2O": -3.0}, grid, temperature)

    assert chemistry.required_parameters() == ("log_H2O",)
    np.testing.assert_allclose(composition["H2O"], np.full(2, 1.0e-3))
    np.testing.assert_allclose(composition["CO2"], np.full(2, 2.0e-4))
    np.testing.assert_allclose(composition["H2"], np.full(2, 0.9988))


def test_free_chemistry_rejects_active_sum_above_one_by_default() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    temperature = np.full(2, 1200.0)
    chemistry = FreeChemistry(active_species=("H2O", "CO"))

    with pytest.raises(ValueError, match="sum to more than one"):
        chemistry.evaluate({"H2O": 0.8, "CO": 0.4}, grid, temperature)


def test_free_chemistry_can_normalize_active_sum_above_one() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    temperature = np.full(2, 1200.0)
    chemistry = FreeChemistry(
        active_species=("H2O", "CO"),
        excess_policy="normalize",
    )

    composition = chemistry.evaluate({"H2O": 0.8, "CO": 0.4}, grid, temperature)

    np.testing.assert_allclose(composition["H2O"], np.full(2, 2.0 / 3.0))
    np.testing.assert_allclose(composition["CO"], np.full(2, 1.0 / 3.0))
    np.testing.assert_allclose(composition["H2"], np.zeros(2))
    np.testing.assert_allclose(composition["He"], np.zeros(2))


def test_free_chemistry_rejects_active_background_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FreeChemistry(
            active_species=("H2O", "H2"),
            background=BackgroundGasMixture({"H2": 0.85, "He": 0.15}),
        )


def test_atmosphere_builder_accepts_free_chemistry_model() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    builder = AtmosphereBuilder(
        pressure_grid=grid,
        temperature_profile=IsothermalTemperatureProfile(temperature=1200.0),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            fixed_mixing_ratios={"H2O": 1.0e-3},
        ),
    )

    atmosphere = builder.build()

    assert atmosphere.species == ("H2O", "H2", "He")
    np.testing.assert_allclose(atmosphere.composition["H2O"], np.full(3, 1.0e-3))
