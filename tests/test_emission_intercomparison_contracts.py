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

WORKER_PATH = ROOT / "examples/run_emission_intercomparison_external.py"
WORKER_SPEC = importlib.util.spec_from_file_location(
    "run_emission_intercomparison_external", WORKER_PATH
)
assert WORKER_SPEC is not None and WORKER_SPEC.loader is not None
worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(worker)


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


def test_stage_4_contract_aligns_robert_cells_with_prt_nodes() -> None:
    contract = common.stage_4_contract(40)

    assert contract["case_id"].tolist() == [
        "isothermal_L40",
        "monotonic_L40",
        "inverted_L40",
        "retrieved_like_L40",
    ]
    assert contract["pressure_edges_bar"].shape == (41,)
    assert contract["pressure_centers_bar"].shape == (40,)
    assert contract["prt_pressure_bar"].shape == (40,)
    np.testing.assert_allclose(
        contract["prt_pressure_bar"],
        np.sqrt(
            contract["pressure_edges_bar"][:-1]
            * contract["pressure_edges_bar"][1:]
        ),
    )
    assert contract["temperature_edges_k"].shape == (4, 41)
    assert contract["temperature_cells_k"].shape == (4, 40)
    np.testing.assert_allclose(np.sum(contract["gas_vmr"], axis=1), 1.0)
    assert bool(contract["native_include_cia"])
    assert bool(contract["native_return_contribution"])


def test_stage_4_profiles_cover_requested_thermal_structures() -> None:
    contract = common.stage_4_contract(80)
    profiles = dict(
        zip(
            contract["profile_name"],
            contract["temperature_cells_k"],
            strict=True,
        )
    )

    assert np.ptp(profiles["isothermal"]) == 0.0
    assert np.all(np.diff(profiles["monotonic"]) > 0.0)
    assert np.any(np.diff(profiles["inverted"]) < 0.0)
    assert np.any(np.diff(profiles["inverted"]) > 0.0)
    retrieved_gradient = np.diff(profiles["retrieved_like"])
    assert np.all(retrieved_gradient >= 0.0)
    assert np.ptp(retrieved_gradient) > 10.0


def test_stage_5_contract_has_symmetric_localized_temperature_cases() -> None:
    contract = common.stage_5_contract(80)
    n_profiles = len(common.STAGE_4_PROFILE_NAMES)
    n_centers = len(common.STAGE_5_PERTURBATION_CENTERS_BAR)

    assert contract["case_id"].shape == (n_profiles * (1 + 2 * n_centers),)
    assert contract["pressure_edges_bar"].shape == (81,)
    assert contract["pressure_centers_bar"].shape == (80,)
    np.testing.assert_allclose(
        contract["prt_pressure_bar"], contract["pressure_centers_bar"]
    )
    np.testing.assert_allclose(
        contract["perturbation_centers_bar"],
        np.geomspace(1.0e-4, 10.0, 6),
    )
    assert np.count_nonzero(contract["native_contribution_case_mask"]) == 4
    for profile_index in range(n_profiles):
        baseline = contract["temperature_cells_k"][profile_index]
        for center_index in range(n_centers):
            selected = (
                (contract["profile_index"] == profile_index)
                & (contract["perturbation_center_index"] == center_index)
            )
            minus = np.flatnonzero(
                selected & (contract["perturbation_sign"] == -1)
            )
            plus = np.flatnonzero(
                selected & (contract["perturbation_sign"] == 1)
            )
            assert minus.size == plus.size == 1
            np.testing.assert_allclose(
                0.5
                * (
                    contract["temperature_cells_k"][minus[0]]
                    + contract["temperature_cells_k"][plus[0]]
                ),
                baseline,
            )


def test_temperature_localization_is_unit_peak_and_log_symmetric() -> None:
    center = 0.1
    pressure = np.array([center / 10.0, center, center * 10.0])
    localization = common.temperature_localization(
        pressure, center, sigma_dex=0.35
    )

    assert localization[1] == 1.0
    assert localization[0] == localization[2]
    assert np.all(localization > 0.0)


def test_temperature_jacobian_metrics_and_response_normalization() -> None:
    wavelength = np.geomspace(0.8, 8.0, 5)
    jacobian = np.arange(1.0, 16.0).reshape(3, 5)
    identical = common.temperature_jacobian_metrics(
        jacobian, jacobian.copy(), wavelength
    )
    response = common.normalize_temperature_response(jacobian)

    assert all(value == 0.0 for value in identical.values())
    np.testing.assert_allclose(np.sum(response, axis=0), 1.0)
    np.testing.assert_allclose(
        common.eclipse_jacobian_ppm_per_k(jacobian, wavelength),
        common.eclipse_depth(jacobian, wavelength) * 1.0e6,
    )


def test_shared_tau_contract_expands_profiles_to_cases() -> None:
    contract = common.stage_5_contract(40)
    shared = np.arange(4 * 40 * 3, dtype=float).reshape(4, 40, 3)
    contract["shared_total_tau"] = shared
    expanded = worker._shared_total_tau(contract)

    assert expanded.shape == (contract["case_id"].size, 40, 3)
    np.testing.assert_array_equal(expanded, shared[contract["profile_index"]])


def test_contribution_metrics_detect_identical_and_shifted_profiles() -> None:
    pressure = np.geomspace(1.0e-4, 10.0, 6)
    contribution = np.zeros((6, 3))
    contribution[2] = 1.0
    identical = common.contribution_metrics(
        contribution, contribution.copy(), pressure
    )

    assert all(value == 0.0 for value in identical.values())

    shifted = np.roll(contribution, 1, axis=0)
    different = common.contribution_metrics(contribution, shifted, pressure)
    assert different["centroid_pressure_rms_difference_dex"] == 1.0
    assert different["peak_pressure_rms_difference_dex"] == 1.0
    assert different["profile_total_variation_median"] == 1.0


def test_external_absorbing_contribution_is_normalized_by_wavelength() -> None:
    wavelength = np.geomspace(0.8, 8.0, 9)
    temperature_edges = np.array([900.0, 1100.0, 1300.0, 1500.0])
    layer_tau = np.full((3, wavelength.size, 2), 0.2)
    contribution = worker._absorbing_formal_contribution(
        wavelength,
        temperature_edges,
        layer_tau,
        np.array([0.4, 0.6]),
        np.array([0.25, 0.75]),
        np.array([0.2, 0.8]),
    )

    assert contribution.shape == (3, wavelength.size)
    assert np.all(contribution >= 0.0)
    np.testing.assert_allclose(np.sum(contribution, axis=0), 1.0)


def test_r100_grid_stays_inside_all_native_spectral_ranges() -> None:
    edges = common.r100_edges()

    assert edges[0] >= 0.5
    assert edges[-1] <= 12.0
    assert np.all(np.diff(edges) > 0.0)
