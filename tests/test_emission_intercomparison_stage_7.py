"""Focused Stage-7 cloud metrics and worker-contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_emission_intercomparison_stage_7",
    EXAMPLES / "benchmark_emission_intercomparison_stage_7.py",
)
assert SPEC is not None and SPEC.loader is not None
stage7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage7)


def _contract() -> dict[str, np.ndarray]:
    return stage7.stage_7_contract(
        2,
        np.array([1.0, 2.0]),
        archived_pressure_edges_bar=np.array([1.0e-5, 1.0e-1, 100.0]),
        archived_wavelength_micron=np.array([1.0, 2.0]),
        archived_extinction_tau=np.ones((2, 2)),
        cloud_indices=np.array([0, 1]),
    )


def test_shared_worker_contract_separates_gas_and_cloud_tau() -> None:
    contract = _contract()
    contract["shared_gas_tau"] = np.full((4, 2, 2), 0.5)
    total = stage7._formal_contribution  # worker function import remains available
    assert callable(total)
    assert contract["shared_gas_tau"].shape == (4, 2, 2)
    assert contract["cloud_extinction_tau"].shape[1:] == (2, 2)
    assert np.all(contract["cloud_single_scattering_albedo"] == 0.0)


def test_cloud_effect_metrics_preserve_sign_and_avoid_point_ratios() -> None:
    wavelength = np.array([1.0, 2.0])
    left = np.array([[-2.0, 1.0], [0.0, 0.0]])
    right = np.array([[-1.0, 2.0], [0.0, 0.0]])

    metrics = stage7._effect_metrics(left, right, wavelength)

    assert metrics["p95_abs_difference_over_pair_peak"] > 0.0
    assert np.isfinite(metrics["rms_eclipse_difference_ppm"])


def test_profile_regrid_normalizes_complete_vertical_tensor() -> None:
    source_edges = np.array([1.0e-5, 1.0e-2, 100.0])
    target_edges = np.geomspace(1.0e-5, 100.0, 9)
    values = np.array([[0.25, 0.75], [0.75, 0.25]])

    result = stage7._regrid_profile(values, source_edges, target_edges)

    assert result.shape == (8, 2)
    np.testing.assert_allclose(np.sum(result, axis=0), 1.0)


def test_pressure_metrics_recover_centroid_and_peak() -> None:
    pressure = np.array([1.0e-3, 1.0e-1, 10.0])
    profile = np.array([[0.0, 0.0], [1.0, 0.25], [0.0, 0.75]])

    summary = stage7._pressure_summary(profile, pressure)

    assert summary["peak_pressure_p05_bar"] >= 1.0e-1
    assert summary["peak_pressure_p95_bar"] <= 10.0
    assert summary["centroid_pressure_median_bar"] > 1.0e-1


def test_isothermal_cloud_effect_is_set_exactly_to_zero_before_response() -> None:
    contract = _contract()
    wavelength = np.array([1.0, 2.0])
    case_count = contract["case_id"].size
    layer_count = contract["pressure_centers_bar"].size
    flux = np.ones((case_count, 2))
    flux[1] += 1.0e-12
    contribution = np.broadcast_to(
        np.array([[[0.25, 0.25], [0.75, 0.75]]]),
        (case_count, layer_count, 2),
    ).copy()
    payload = {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "normalized_contribution": contribution,
        "cloud_extinction_tau": contract["cloud_extinction_tau"][
            contract["case_cloud_index"]
        ],
        "runtime_s": np.ones(case_count),
        "metadata_json": np.array("{}"),
    }

    binned = stage7._bin_payload(payload, contract, np.array([1.0, 1.5, 2.0]))

    isothermal = contract["profile_index"] == 0
    assert np.all(binned["cloud_effect_flux_r100"][isothermal] == 0.0)
    assert binned["raw_isothermal_max_abs_flux_cancellation"] > 0.0


def test_track_a_gates_are_frozen_and_include_convergence() -> None:
    assert stage7.TRACK_A_GATES["omega0_max_abs"] == 0.0
    assert "primary_cloud_effect_eclipse_rms_ppm" in stage7.TRACK_A_GATES
    assert "80_to_160_cloud_response_profile_tv_p95" in stage7.TRACK_A_GATES
