"""Focused tests for the frozen Version-2 Stage-3 contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    load_version_2_common_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/emission_intercomparison/version_2"
MODULE_PATH = ROOT / "examples/benchmark_emission_intercomparison_v2_stage_3.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_emission_intercomparison_v2_stage_3", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
stage_3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_3)

RESOLUTIONS = (40, 80, 160)
PROFILES = ("isothermal", "pg14_non_inverted")
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")
FACTOR_CASES = (
    ("molecular_only", False, False),
    ("molecular_plus_h2_h2_cia", True, False),
    ("molecular_plus_h2_he_cia", False, True),
    ("molecular_plus_h2_h2_and_h2_he_cia", True, True),
)


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_contract_serializes_exact_cia_factorial(n_cells: int) -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_3.build_stage_3_contract(common, n_cells)
    expected_factors = [factor for _profile in PROFILES for factor in FACTOR_CASES]

    assert contract["case_id"].tolist() == [
        f"{profile}_{factor_name}_{n_cells}_cells"
        for profile in PROFILES
        for factor_name, _h2_h2, _h2_he in FACTOR_CASES
    ]
    assert contract["profile_name"].tolist() == [
        profile for profile in PROFILES for _factor in FACTOR_CASES
    ]
    assert contract["factor_name"].tolist() == [
        factor_name for factor_name, _h2_h2, _h2_he in expected_factors
    ]
    np.testing.assert_array_equal(
        contract["include_h2_h2_cia"],
        [h2_h2 for _factor_name, h2_h2, _h2_he in expected_factors],
    )
    np.testing.assert_array_equal(
        contract["include_h2_he_cia"],
        [h2_he for _factor_name, _h2_h2, h2_he in expected_factors],
    )


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_contract_preserves_frozen_composition_and_mmw(n_cells: int) -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_3.build_stage_3_contract(common, n_cells)
    gas_names = tuple(common.composition_vmr)
    authoritative_vmr = np.asarray(
        [common.composition_vmr[name] for name in gas_names]
    )

    assert contract["gas_name"].tolist() == list(gas_names)
    np.testing.assert_array_equal(
        contract["gas_vmr"],
        np.broadcast_to(authoritative_vmr, (len(PROFILES) * len(FACTOR_CASES), 6)),
    )
    declared_mmw = json.loads((DATA / "common_contract.json").read_text())[
        "composition"
    ]["mean_molecular_weight_u_declared"]
    np.testing.assert_array_equal(
        contract["mean_molecular_weight_u"],
        np.full(len(PROFILES) * len(FACTOR_CASES), declared_mmw),
    )
    assert contract["molecular_species_name"].tolist() == list(MOLECULAR_SPECIES)
    np.testing.assert_array_equal(
        contract["molecular_species_active"],
        np.ones((len(PROFILES) * len(FACTOR_CASES), len(MOLECULAR_SPECIES)), dtype=bool),
    )


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_contract_preserves_grids_and_evaluated_temperatures(
    n_cells: int,
) -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_3.build_stage_3_contract(common, n_cells)
    grid = next(grid for grid in common.pressure_grids if grid.n_cells == n_cells)

    assert contract["pressure_edges_bar"].shape == (n_cells + 1,)
    assert contract["pressure_centers_bar"].shape == (n_cells,)
    np.testing.assert_array_equal(contract["pressure_edges_bar"], grid.edges_bar)
    np.testing.assert_array_equal(contract["pressure_centers_bar"], grid.centers_bar)
    np.testing.assert_array_equal(
        contract["picaso_pressure_levels_bar"], grid.picaso_levels_bar
    )
    np.testing.assert_array_equal(
        contract["petitradtrans_pressure_nodes_bar"], grid.petitradtrans_nodes_bar
    )

    expected_temperature = np.asarray(
        [
            common.temperature_profiles_k[f"{profile}_{n_cells}_cells"]
            for profile in PROFILES
            for _factor in FACTOR_CASES
        ]
    )
    assert contract["temperature_cells_k"].shape == (8, n_cells)
    np.testing.assert_array_equal(
        contract["temperature_cells_k"], expected_temperature
    )
    for profile_index, profile in enumerate(PROFILES):
        start = profile_index * len(FACTOR_CASES)
        expected = common.temperature_profiles_k[f"{profile}_{n_cells}_cells"]
        np.testing.assert_array_equal(
            contract["temperature_cells_k"][start : start + len(FACTOR_CASES)],
            np.broadcast_to(expected, (len(FACTOR_CASES), n_cells)),
        )


def test_stage_3_acceptance_gates_are_frozen_before_matrix_execution() -> None:
    assert stage_3.STAGE_3_ACCEPTANCE_GATES == {
        "track_a_max_abs_symmetric_relative": 5.0e-4,
        "track_a_max_abs_eclipse_difference_ppm": 0.1,
        "track_a_80_to_160_max_abs_eclipse_difference_ppm": 0.1,
        "track_a_isothermal_max_abs_eclipse_difference_ppm": 0.1,
        "scattering_single_scattering_albedo_max_abs": 0.0,
        "pilot_projected_wall_time_max_s": 7200.0,
        "pilot_peak_rss_fraction_of_available_max": 0.60,
    }
