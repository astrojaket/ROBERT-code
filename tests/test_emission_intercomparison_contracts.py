"""Tests for the staged emission-intercomparison contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/emission_intercomparison_common.py"
SPEC = importlib.util.spec_from_file_location("emission_intercomparison_common", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def test_stage_1_contract_covers_grey_matrix_and_conserves_total_tau() -> None:
    contract = common.stage_1_contract(16)

    assert contract["case_id"].shape == (16,)
    assert contract["component_tau"].shape == (16, 1, 16, 320)
    assert np.isclose(np.sum(contract["disk_weights"]), 1.0)
    total_tau = np.sum(contract["component_tau"], axis=(1, 2))
    expected = np.tile(common.STAGE_1_TOTAL_TAU, 4)
    np.testing.assert_allclose(
        total_tau, np.broadcast_to(expected[:, None], total_tau.shape)
    )


def test_stage_2_contract_covers_species_temperature_and_abundance_matrix() -> None:
    contract = common.stage_2_contract(20)

    assert contract["case_id"].shape == (4 * 4 * 6,)
    assert contract["component_tau"].shape == (96, 4, 20, 320)
    np.testing.assert_allclose(np.sum(contract["gas_vmr"], axis=1), 1.0)
    active_components = np.any(contract["component_tau"] > 0.0, axis=(2, 3))
    np.testing.assert_array_equal(np.sum(active_components, axis=1), 1)
    anchors = np.tile(np.repeat(common.STAGE_1_TEMPERATURES_K, 6), 4)
    edge_offsets = contract["temperature_edges_k"] - anchors[:, None]
    np.testing.assert_allclose(edge_offsets[:, 0], -150.0)
    np.testing.assert_allclose(edge_offsets[:, -1], 150.0)


def test_stage_3_contract_fixes_composition_and_cia_switch() -> None:
    contract = common.stage_3_contract(80)

    assert contract["pressure_edges_bar"].shape == (81,)
    assert contract["temperature_edges_k"].shape == (1, 81)
    assert bool(contract["native_include_cia"])
    np.testing.assert_allclose(np.sum(contract["gas_vmr"], axis=1), 1.0)
    np.testing.assert_allclose(
        contract["gas_vmr"][0, 2:],
        [1.0e-3, 3.0e-4, 1.0e-4, 1.0e-5],
    )


def test_r100_grid_stays_inside_all_native_spectral_ranges() -> None:
    edges = common.r100_edges()

    assert edges[0] >= 0.5
    assert edges[-1] <= 12.0
    assert np.all(np.diff(edges) > 0.0)
