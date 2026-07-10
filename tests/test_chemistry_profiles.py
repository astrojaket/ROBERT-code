"""Tests for retrieval-facing chemistry parameterizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereBuilder,
    BackgroundGasMixture,
    CompositionMeanMolecularWeight,
    FastChemEquilibriumChemistry,
    FixedMeanMolecularWeight,
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


def test_fixed_mean_molecular_weight_returns_layer_profile() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    model = FixedMeanMolecularWeight(2.33)

    mean_molecular_weight = model.evaluate({}, grid)

    np.testing.assert_allclose(mean_molecular_weight, np.full(3, 2.33))
    assert mean_molecular_weight.flags.writeable is False


def test_composition_mean_molecular_weight_uses_vmr_weighted_masses() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    composition = {
        "H2": np.full(2, 0.8),
        "He": np.full(2, 0.19),
        "H2O": np.full(2, 0.01),
    }
    model = CompositionMeanMolecularWeight()

    mean_molecular_weight = model.evaluate(composition, grid)

    expected = 0.8 * 2.01588 + 0.19 * 4.002602 + 0.01 * 18.01528
    np.testing.assert_allclose(mean_molecular_weight, np.full(2, expected))
    assert mean_molecular_weight.flags.writeable is False


def test_composition_mean_molecular_weight_rejects_missing_species_mass() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    model = CompositionMeanMolecularWeight()

    with pytest.raises(ValueError, match="missing molecular mass"):
        model.evaluate({"MysteryGas": np.full(2, 1.0)}, grid)


def test_composition_mean_molecular_weight_requires_complete_vmr_budget() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    model = CompositionMeanMolecularWeight()

    with pytest.raises(ValueError, match="sum to one"):
        model.evaluate({"H2O": np.full(2, 1.0e-3)}, grid)


def test_composition_mean_molecular_weight_can_normalize_partial_budget() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    model = CompositionMeanMolecularWeight(normalization="normalize")

    mean_molecular_weight = model.evaluate(
        {
            "H2": np.full(2, 0.4),
            "He": np.full(2, 0.1),
        },
        grid,
    )

    expected = (0.4 * 2.01588 + 0.1 * 4.002602) / (0.4 + 0.1)
    np.testing.assert_allclose(mean_molecular_weight, np.full(2, expected))


def test_composition_mean_molecular_weight_can_preserve_raw_selected_species_sum() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    model = CompositionMeanMolecularWeight(normalization="raw_sum")

    mean_molecular_weight = model.evaluate(
        {
            "H2": np.full(2, 0.4),
            "He": np.full(2, 0.1),
        },
        grid,
    )

    expected = 0.4 * 2.01588 + 0.1 * 4.002602
    np.testing.assert_allclose(mean_molecular_weight, np.full(2, expected))


def test_fastchem_equilibrium_chemistry_smoke_test_when_available() -> None:
    pytest.importorskip("pyfastchem")
    fastchem_path = Path.home() / "Dropbox" / "fastchem"
    if not fastchem_path.exists():
        pytest.skip("local FastChem data files are not available")
    grid = PressureGrid.logspace(1.0e-4, 1.0e-2, n_layers=2, unit="bar")
    chemistry = FastChemEquilibriumChemistry(
        fastchem_path=fastchem_path,
        fastchem_species=("H2O1", "C1O1", "H2", "He"),
        labels=("H2O", "CO", "H2", "He"),
    )

    composition = chemistry.evaluate(
        {"metallicity": 0.0, "CtoO": 0.55},
        grid,
        np.full(grid.n_layers, 1500.0),
    )

    assert chemistry.required_parameters() == ("metallicity", "CtoO")
    assert set(composition) == {"H2O", "CO", "H2", "He"}
    assert np.all(composition["H2"] > 0.0)
    assert np.all(composition["He"] > 0.0)
    assert composition["H2"].flags.writeable is False
    enriched = chemistry.evaluate(
        {"metallicity": 1.0, "CtoO": 0.8},
        grid,
        np.full(grid.n_layers, 1500.0),
    )
    repeated = chemistry.evaluate(
        {"metallicity": 0.0, "CtoO": 0.55},
        grid,
        np.full(grid.n_layers, 1500.0),
    )
    assert not np.allclose(enriched["H2O"], composition["H2O"], rtol=1.0e-3)
    np.testing.assert_allclose(repeated["H2O"], composition["H2O"], rtol=0.0, atol=0.0)
    assert chemistry.metadata["fastchem_path"] == str(fastchem_path)


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


def test_atmosphere_builder_can_derive_mean_molecular_weight_from_composition() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    builder = AtmosphereBuilder(
        pressure_grid=grid,
        temperature_profile=IsothermalTemperatureProfile(temperature=1200.0),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            fixed_mixing_ratios={"H2O": 1.0e-3},
            background=BackgroundGasMixture({"H2": 0.75, "He": 0.25}),
        ),
        mean_molecular_weight_model=CompositionMeanMolecularWeight(),
    )

    atmosphere = builder.build()

    expected = 1.0e-3 * 18.01528 + 0.999 * 0.75 * 2.01588 + 0.999 * 0.25 * 4.002602
    np.testing.assert_allclose(atmosphere.mean_molecular_weight, np.full(3, expected))
