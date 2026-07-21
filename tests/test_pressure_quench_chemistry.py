"""Tests for the generic pressure-quench chemistry decorator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
import pytest

from robert_exoplanets import (
    AtmosphereBuilder,
    CompositionMeanMolecularWeight,
    FastChemEquilibriumChemistry,
    IsothermalTemperatureProfile,
    PressureGrid,
    PressureQuenchChemistry,
    QuenchGroup,
)


@dataclass
class AnalyticChemistry:
    """Small profile-producing base chemistry used only by these unit tests."""

    profiles: Mapping[str, NDArray[np.float64]]
    parameters: tuple[str, ...] = ("base_parameter",)
    name: str = "analytic-equilibrium"
    convention: str = "volume_mixing_ratio"
    metadata: Mapping[str, str] = field(
        default_factory=lambda: {"base_source": "analytic-test"}
    )

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def required_parameters(self) -> tuple[str, ...]:
        return self.parameters

    def evaluate(self, parameters, pressure_grid, temperature):
        del parameters, temperature
        if any(np.asarray(value).shape != pressure_grid.centers.shape for value in self.profiles.values()):
            raise ValueError("test profile shape mismatch")
        return dict(self.profiles)


def _grid(*, decreasing: bool = False, unit: str = "bar") -> PressureGrid:
    centers_bar = np.array([0.01, 0.1, 1.0, 10.0])
    factors = {"bar": 1.0, "Pa": 1.0e5, "mbar": 1.0e3, "atm": 1.0 / 1.01325}
    centers = centers_bar * factors[unit]
    if decreasing:
        centers = centers[::-1]
    return PressureGrid.from_log_centers(
        float(centers[0]),
        float(centers[-1]),
        len(centers),
        unit=unit,
    )


def _base(*, decreasing: bool = False) -> AnalyticChemistry:
    profiles = {
        "CO2": np.array([1.0, 2.0, 4.0, 8.0]) * 1.0e-4,
        "H2O": np.array([8.0, 6.0, 4.0, 2.0]) * 1.0e-4,
        "H2": np.array([0.9991, 0.9992, 0.9992, 0.9990]),
    }
    if decreasing:
        profiles = {species: values[::-1].copy() for species, values in profiles.items()}
    return AnalyticChemistry(profiles)


def _evaluate(model, grid, **parameters):
    values = {"base_parameter": 1.0, **parameters}
    return model.evaluate(values, grid, np.full(grid.n_layers, 1200.0))


def test_exact_grid_quench_freezes_arbitrary_molecule_above_pressure() -> None:
    model = PressureQuenchChemistry.molecular(
        _base(),
        {"CO2": "log_Pq_CO2"},
    )

    composition = _evaluate(model, _grid(), log_Pq_CO2=-1.0)

    np.testing.assert_allclose(composition["CO2"], [2.0e-4, 2.0e-4, 4.0e-4, 8.0e-4])
    np.testing.assert_array_equal(composition["H2O"], _base().profiles["H2O"])
    np.testing.assert_array_equal(composition["H2"], _base().profiles["H2"])


def test_off_grid_quench_uses_linear_vmr_interpolation_in_log_pressure() -> None:
    model = PressureQuenchChemistry.molecular(
        _base(),
        {"CO2": "log_Pq_CO2"},
    )

    composition = _evaluate(model, _grid(), log_Pq_CO2=-0.5)

    np.testing.assert_allclose(composition["CO2"], [3.0e-4, 3.0e-4, 4.0e-4, 8.0e-4])


def test_multiple_single_and_multi_species_groups_use_independent_pressures() -> None:
    model = PressureQuenchChemistry(
        _base(),
        groups=(
            QuenchGroup("log_Pq_carbon", ("CO2", "H2O")),
            QuenchGroup.molecular("H2", "log_Pq_hydrogen"),
        ),
    )

    composition = _evaluate(
        model,
        _grid(),
        log_Pq_carbon=-1.0,
        log_Pq_hydrogen=0.0,
    )

    np.testing.assert_allclose(composition["CO2"], [2.0e-4, 2.0e-4, 4.0e-4, 8.0e-4])
    np.testing.assert_allclose(composition["H2O"], [6.0e-4, 6.0e-4, 4.0e-4, 2.0e-4])
    np.testing.assert_allclose(composition["H2"], [0.9992, 0.9992, 0.9992, 0.9990])


@pytest.mark.parametrize(
    ("log_quench_pressure", "expected"),
    [
        (-2.0, [1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4]),
        (1.0, [8.0e-4, 8.0e-4, 8.0e-4, 8.0e-4]),
    ],
)
def test_inclusive_layer_center_boundaries(log_quench_pressure, expected) -> None:
    model = PressureQuenchChemistry.molecular(_base(), {"CO2": "log_Pq_CO2"})

    composition = _evaluate(model, _grid(), log_Pq_CO2=log_quench_pressure)

    np.testing.assert_allclose(composition["CO2"], expected)


@pytest.mark.parametrize("log_quench_pressure", [-2.000001, 1.000001])
def test_out_of_domain_quench_pressure_is_rejected(log_quench_pressure) -> None:
    model = PressureQuenchChemistry.molecular(_base(), {"CO2": "log_Pq_CO2"})

    with pytest.raises(ValueError, match="outside the inclusive"):
        _evaluate(model, _grid(), log_Pq_CO2=log_quench_pressure)


def test_increasing_and_decreasing_grids_produce_same_physical_profile() -> None:
    increasing = PressureQuenchChemistry.molecular(
        _base(), {"CO2": "log_Pq_CO2"}
    )
    decreasing = PressureQuenchChemistry.molecular(
        _base(decreasing=True), {"CO2": "log_Pq_CO2"}
    )

    ascending = _evaluate(increasing, _grid(), log_Pq_CO2=-1.0)["CO2"]
    descending = _evaluate(
        decreasing, _grid(decreasing=True), log_Pq_CO2=-1.0
    )["CO2"]

    np.testing.assert_allclose(descending[::-1], ascending)


@pytest.mark.parametrize("unit", ["bar", "Pa", "mbar", "atm"])
def test_pressure_units_are_converted_explicitly(unit) -> None:
    model = PressureQuenchChemistry.molecular(_base(), {"CO2": "log_Pq_CO2"})

    composition = _evaluate(model, _grid(unit=unit), log_Pq_CO2=-1.0)

    np.testing.assert_allclose(composition["CO2"], [2.0e-4, 2.0e-4, 4.0e-4, 8.0e-4])


def test_group_and_parameter_validation_rejects_ambiguous_models() -> None:
    with pytest.raises(ValueError, match="at least one species"):
        QuenchGroup("log_Pq", ())
    with pytest.raises(ValueError, match="identifier"):
        QuenchGroup("log Pq", ("CO2",))
    with pytest.raises(ValueError, match="within a quench group"):
        QuenchGroup("log_Pq", ("CO2", "CO2"))
    with pytest.raises(ValueError, match="at least one quench group"):
        PressureQuenchChemistry(_base(), ())
    with pytest.raises(ValueError, match="only one quench group"):
        PressureQuenchChemistry(
            _base(),
            (
                QuenchGroup("log_Pq_1", ("CO2",)),
                QuenchGroup("log_Pq_2", ("CO2",)),
            ),
        )
    with pytest.raises(ValueError, match="missing from base chemistry"):
        PressureQuenchChemistry(
            _base(), (QuenchGroup("log_Pq_CH4", ("CH4",)),)
        )
    with pytest.raises(ValueError, match="collide with base chemistry"):
        PressureQuenchChemistry(
            _base(), (QuenchGroup("base_parameter", ("CO2",)),)
        )


def test_outputs_are_immutable_and_base_output_is_not_mutated() -> None:
    base = _base()
    original = {species: values.copy() for species, values in base.profiles.items()}
    model = PressureQuenchChemistry.molecular(base, {"CO2": "log_Pq_CO2"})

    composition = _evaluate(model, _grid(), log_Pq_CO2=-1.0)

    for species, values in base.profiles.items():
        np.testing.assert_array_equal(values, original[species])
        assert values.flags.writeable is True
    assert all(profile.flags.writeable is False for profile in composition.values())
    with pytest.raises(ValueError):
        composition["CO2"][0] = 0.0


def test_required_parameters_and_base_provenance_are_deterministic() -> None:
    model = PressureQuenchChemistry(
        _base(),
        (
            QuenchGroup("log_Pq_C", ("CO2", "H2O")),
            QuenchGroup("log_Pq_H", ("H2",)),
        ),
    )

    assert model.required_parameters() == (
        "base_parameter",
        "log_Pq_C",
        "log_Pq_H",
    )
    assert model.metadata["base_source"] == "analytic-test"
    assert model.metadata["quench_groups"] == "log_Pq_C:CO2|H2O;log_Pq_H:H2"
    assert model.metadata["quench_pressure_semantics"] == "log10(P_q/bar)"
    assert model.metadata["quench_closure_policy"] == "no_renormalization"
    assert "not_NemesisPy_parity" in model.metadata["quench_code_convention"]


def test_taylor_preset_has_paper_species_and_deliberately_omits_n2() -> None:
    profiles = {
        species: np.full(4, 1.0e-4)
        for species in ("H2O", "CO", "CO2", "CH4", "NH3", "N2")
    }

    model = PressureQuenchChemistry.taylor_2026_hot_jupiter_element_grouped(
        AnalyticChemistry(profiles)
    )

    assert model.groups == (
        QuenchGroup("log_Pq_C", ("H2O", "CO", "CO2", "CH4")),
        QuenchGroup("log_Pq_N", ("NH3",)),
    )
    assert all("N2" not in group.species for group in model.groups)


def test_no_renormalization_diagnostics_and_builder_mmw_recomputation() -> None:
    grid = _grid()
    base = AnalyticChemistry(
        {
            "CO2": np.array([0.01, 0.02, 0.04, 0.08]),
            "H2": np.array([0.99, 0.98, 0.96, 0.92]),
        },
        parameters=(),
    )
    model = PressureQuenchChemistry.molecular(base, {"CO2": "log_Pq_CO2"})
    composition, diagnostics = model.evaluate_with_diagnostics(
        {"log_Pq_CO2": -1.0},
        grid,
        np.full(grid.n_layers, 1200.0),
    )

    np.testing.assert_allclose(diagnostics.base_vmr_sum, np.ones(4))
    np.testing.assert_allclose(
        diagnostics.quenched_vmr_sum, [1.01, 1.0, 1.0, 1.0]
    )
    # A different H2 profile makes closure drift visible for an off-grid Pq.
    off_grid, off_grid_diagnostics = model.evaluate_with_diagnostics(
        {"log_Pq_CO2": -0.5},
        grid,
        np.full(grid.n_layers, 1200.0),
    )
    np.testing.assert_allclose(
        off_grid_diagnostics.quenched_vmr_sum,
        [1.02, 1.01, 1.0, 1.0],
    )
    np.testing.assert_allclose(off_grid_diagnostics.vmr_sum_drift, [0.02, 0.01, 0.0, 0.0])
    assert off_grid_diagnostics.elemental_budget_drift is not None
    np.testing.assert_allclose(
        off_grid_diagnostics.elemental_budget_drift["C"],
        [0.02, 0.01, 0.0, 0.0],
    )
    assert all(not values.flags.writeable for values in (
        diagnostics.base_vmr_sum,
        diagnostics.quenched_vmr_sum,
        diagnostics.vmr_sum_drift,
    ))

    builder = AtmosphereBuilder(
        pressure_grid=grid,
        temperature_profile=IsothermalTemperatureProfile(temperature=1200.0),
        chemistry_model=model,
        mean_molecular_weight_model=CompositionMeanMolecularWeight(
            normalization="normalize"
        ),
    )
    atmosphere = builder.build({"log_Pq_CO2": -0.5})
    expected = (
        off_grid["CO2"] * 44.0095 + off_grid["H2"] * 2.01588
    ) / (off_grid["CO2"] + off_grid["H2"])
    np.testing.assert_allclose(atmosphere.mean_molecular_weight, expected)


def test_fastchem_pressure_quench_smoke_when_optional_data_are_available() -> None:
    pytest.importorskip("pyfastchem")
    fastchem_path = Path(__file__).parents[1] / "data" / "chemistry" / "fastchem"
    if not fastchem_path.exists():
        pytest.skip("local FastChem data files are not available")
    grid = PressureGrid.from_log_centers(1.0e-4, 1.0e-1, 4, unit="bar")
    base = FastChemEquilibriumChemistry(
        fastchem_path=fastchem_path,
        fastchem_species=("H2O1", "C1O2", "H2", "He"),
        labels=("H2O", "CO2", "H2", "He"),
    )
    model = PressureQuenchChemistry.molecular(base, {"CO2": "log_Pq_CO2"})

    composition = model.evaluate(
        {"metallicity": 0.0, "CtoO": 0.55, "log_Pq_CO2": -2.0},
        grid,
        np.full(grid.n_layers, 1500.0),
    )

    assert model.required_parameters() == ("metallicity", "CtoO", "log_Pq_CO2")
    np.testing.assert_allclose(composition["CO2"][:3], composition["CO2"][2])
    assert composition["CO2"].flags.writeable is False
