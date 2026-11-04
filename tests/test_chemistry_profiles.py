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


def test_composition_mean_molecular_weight_supports_fitted_phantom_gas_mass() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    model = CompositionMeanMolecularWeight(
        molecular_mass_parameters={"phantom": "phantom_mmw"}
    )
    composition = {
        "CO2": np.full(2, 0.2),
        "phantom": np.full(2, 0.8),
    }

    mean_molecular_weight = model.evaluate(
        composition,
        grid,
        parameters={"phantom_mmw": 60.0},
    )

    assert model.required_parameters() == ("phantom_mmw",)
    np.testing.assert_allclose(mean_molecular_weight, np.full(2, 0.2 * 44.0095 + 48.0))


def test_fitted_phantom_gas_mass_must_be_positive() -> None:
    grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    model = CompositionMeanMolecularWeight(
        molecular_mass_parameters={"phantom": "phantom_mmw"}
    )

    with pytest.raises(ValueError, match="must be positive"):
        model.evaluate(
            {"phantom": np.ones(2)},
            grid,
            parameters={"phantom_mmw": 0.0},
        )


def test_fastchem_equilibrium_chemistry_smoke_test_when_available() -> None:
    pytest.importorskip("pyfastchem")
    fastchem_path = Path(__file__).parents[1] / "data" / "chemistry" / "fastchem"
    if not fastchem_path.exists():
        pytest.skip("local FastChem data files are not available")
    grid = PressureGrid.logspace(1.0e-4, 1.0e-2, n_layers=2, unit="bar")
    chemistry = FastChemEquilibriumChemistry(
        fastchem_path=fastchem_path,
        fastchem_species=("H2O1", "C1O1", "O2S1", "H2", "He", "H"),
        labels=("H2O", "CO", "SO2", "H2", "He", "H"),
        constant_log10_vmr_parameters={"SO2": "log_SO2"},
    )

    composition = chemistry.evaluate(
        {"metallicity": 0.0, "CtoO": 0.55, "log_SO2": -7.0},
        grid,
        np.full(grid.n_layers, 1500.0),
    )

    assert chemistry.required_parameters() == ("metallicity", "CtoO", "log_SO2")
    assert set(composition) == {"H2O", "CO", "SO2", "H2", "He", "H"}
    np.testing.assert_allclose(composition["SO2"], np.full(2, 1.0e-7))
    assert np.all(composition["H2"] > 0.0)
    assert np.all(composition["H"] > 0.0)
    assert np.all(composition["He"] > 0.0)
    assert composition["H2"].flags.writeable is False
    mean_molecular_weight = CompositionMeanMolecularWeight(
        normalization="raw_sum"
    ).evaluate(composition, grid)
    expected = (
        composition["H2O"] * 18.01528
        + composition["CO"] * 28.0101
        + composition["SO2"] * 64.066
        + composition["H2"] * 2.01588
        + composition["He"] * 4.002602
        + composition["H"] * 1.00794
    )
    np.testing.assert_allclose(mean_molecular_weight, expected)
    enriched = chemistry.evaluate(
        {"metallicity": 1.0, "CtoO": 0.8, "log_SO2": -6.0},
        grid,
        np.full(grid.n_layers, 1500.0),
    )
    repeated = chemistry.evaluate(
        {"metallicity": 0.0, "CtoO": 0.55, "log_SO2": -7.0},
        grid,
        np.full(grid.n_layers, 1500.0),
    )
    assert not np.allclose(enriched["H2O"], composition["H2O"], rtol=1.0e-3)
    np.testing.assert_allclose(repeated["H2O"], composition["H2O"], rtol=0.0, atol=0.0)
    assert chemistry.metadata["fastchem_path"] == str(fastchem_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"H": "log_H_H"}, "cannot change H or He"),
        ({"Xx": "log_Xx_H"}, "unknown FastChem element"),
        ({"O": "same", "C": "same"}, "must be unique"),
        (
            {"O": "log_SO2"},
            "must be unique",
        ),
    ],
)
def test_fastchem_independent_abundance_configuration_is_validated(
    overrides: dict[str, str], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FastChemEquilibriumChemistry(
            fastchem_path="unused",
            metallicity_parameter_name=None,
            carbon_to_oxygen_parameter_name=None,
            element_log10_number_ratio_parameters=overrides,
            constant_log10_vmr_parameters={"SO2": "log_SO2"},
            labels=("SO2",),
            fastchem_species=("O2S1",),
        )


def test_fastchem_independent_elements_change_separately_and_reset() -> None:
    pytest.importorskip("pyfastchem")
    fastchem_path = Path(__file__).parents[1] / "data" / "chemistry" / "fastchem"
    if not fastchem_path.exists():
        pytest.skip("local FastChem data files are not available")
    species = ("H2O1", "C1O1", "O1Si1", "H2", "He", "H")
    labels = ("H2O", "CO", "SiO", "H2", "He", "H")
    independent = FastChemEquilibriumChemistry(
        fastchem_path=fastchem_path,
        fastchem_species=species,
        labels=labels,
        metallicity_parameter_name=None,
        carbon_to_oxygen_parameter_name=None,
        element_log10_number_ratio_parameters={
            "O": "log10_O_H",
            "C": "log10_C_H",
            "Si": "log10_Si_H",
        },
    )
    legacy = FastChemEquilibriumChemistry(
        fastchem_path=fastchem_path,
        fastchem_species=species,
        labels=labels,
    )
    runtime = independent._runtime
    hydrogen_index = runtime.fastchem.getElementIndex("H")
    solar = np.array(runtime.solar_abundances, copy=True)
    parameters = {
        f"log10_{element}_H": np.log10(
            solar[runtime.fastchem.getElementIndex(element)] / solar[hydrogen_index]
        )
        for element in ("O", "C", "Si")
    }
    grid = PressureGrid.logspace(1.0e-4, 1.0e-2, n_layers=2, unit="bar")
    temperature = np.full(grid.n_layers, 1500.0)

    solar_result = independent.evaluate(parameters, grid, temperature)
    # The selected equilibrium species must also form a usable bulk state.
    # This catches a missing SiO mass before a transmission calculation starts.
    mmw = CompositionMeanMolecularWeight(normalization="normalize").evaluate(
        solar_result, grid
    )
    assert np.all(np.isfinite(mmw))
    assert np.all((mmw > 2.0) & (mmw < 3.0))
    solar_legacy = legacy.evaluate(
        {
            "metallicity": 0.0,
            "CtoO": solar[runtime.fastchem.getElementIndex("C")]
            / solar[runtime.fastchem.getElementIndex("O")],
        },
        grid,
        temperature,
    )
    for label in labels:
        np.testing.assert_allclose(solar_result[label], solar_legacy[label], rtol=1.0e-12)

    oxygen_parameters = dict(parameters)
    oxygen_parameters["log10_O_H"] += 0.5
    oxygen_result = independent.evaluate(oxygen_parameters, grid, temperature)
    assert not np.allclose(oxygen_result["H2O"], solar_result["H2O"], rtol=1.0e-3)

    carbon_parameters = dict(parameters)
    carbon_parameters["log10_C_H"] += 0.5
    carbon_result = independent.evaluate(carbon_parameters, grid, temperature)
    assert not np.allclose(carbon_result["CO"], solar_result["CO"], rtol=1.0e-3)

    silicon_parameters = dict(parameters)
    silicon_parameters["log10_Si_H"] += 0.5
    silicon_result = independent.evaluate(silicon_parameters, grid, temperature)
    assert not np.allclose(silicon_result["SiO"], solar_result["SiO"], rtol=1.0e-3)

    repeated = independent.evaluate(parameters, grid, temperature)
    for label in labels:
        np.testing.assert_array_equal(repeated[label], solar_result[label])
    np.testing.assert_array_equal(independent._runtime.solar_abundances, solar)
    assert independent.required_parameters() == (
        "log10_O_H",
        "log10_C_H",
        "log10_Si_H",
    )
    assert independent.metadata["abundance_mode"] == "independent_elements"
    assert independent.metadata["element_log10_number_ratio_parameters"] == (
        "C:log10_C_H,O:log10_O_H,Si:log10_Si_H"
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
