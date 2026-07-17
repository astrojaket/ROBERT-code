"""Integrity tests for committed Version-2 Stage-1 products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    Version2CommonContract,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/emission_intercomparison/version_2"
COMMON = DATA / "common_contract.json"
PROFILES = DATA / "version_2_common_profiles.npz"
REPORT = DATA / "stage_1_report.json"
ARRAYS = DATA / "stage_1_grey_isothermal_arrays.npz"
INTEGRITY = DATA / "stage_1_integrity.json"
LINEAGE = DATA / "version_lineage.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_version_2_common_contract_and_lineage_are_self_consistent() -> None:
    payload = json.loads(COMMON.read_text())
    contract = Version2CommonContract.from_dict(payload)
    lineage = json.loads(LINEAGE.read_text())

    assert contract.to_dict() == payload
    assert lineage["version_1"]["endpoint_commit"] == (
        "f00e0616c7aae7d37e0badda295c189ead17dde1"
    )
    assert lineage["version_1"]["tag_peels_to_endpoint"] is True
    assert lineage["separate_manuscript_repository"]["touched_by_stage_1"] is False
    assert lineage["large_raw_output_tree"]["recursively_hashed"] is False


def test_stage_1_checksums_and_manifest_match_committed_bytes() -> None:
    checksums = json.loads((DATA / "checksums.json").read_text())
    integrity = json.loads(INTEGRITY.read_text())
    expected = (COMMON, PROFILES, REPORT, ARRAYS, INTEGRITY, LINEAGE)

    for path in expected:
        assert checksums[path.name] == _sha256(path)
    for path in (COMMON, PROFILES, REPORT, ARRAYS, LINEAGE):
        assert integrity["artifacts"][path.name]["sha256"] == _sha256(path)


def test_stage_1_report_preserves_predeclared_failure_and_capability_limits() -> None:
    report = json.loads(REPORT.read_text())
    failed = [name for name, passed in report["gate_results"].items() if not passed]

    assert report["status"] == "fail"
    assert failed == ["analytic_max_abs_eclipse_difference_ppm"]
    assert report["observed_gate_values"][failed[0]] > report[
        "predeclared_acceptance_gates"
    ][failed[0]]
    assert report["pilot"]["authorized_full_matrix"] is True
    assert report["framework_scope"]["picaso"].startswith(
        "native exact-omega0=0"
    )
    probe = report["per_resolution"]["80"]["solver_metadata"][
        "picaso_absorbing_formal_reference"
    ]["native_exact_zero_probe"]
    assert probe == {
        "attempted": True,
        "finite": False,
        "result": "non_finite_at_exact_omega0_zero",
    }
    assert report["per_resolution"]["80"]["supported_case_count"][
        "petitradtrans"
    ] == 5


def test_stage_1_arrays_retain_native_r100_and_complete_vertical_fields() -> None:
    with np.load(ARRAYS, allow_pickle=False) as archive:
        assert archive["native_wavelength_micron"].size > archive[
            "r100_centers_micron"
        ].size
        for n_cells in (40, 80, 160):
            prefix = f"cells_{n_cells}"
            assert archive[f"{prefix}_shared_layer_tau"].shape == (
                10,
                n_cells,
                archive["native_wavelength_micron"].size,
            )
            for model in (
                "robert",
                "picaso_absorbing_formal_reference",
                "petitradtrans",
            ):
                assert archive[f"{prefix}_{model}_raw_flux_native_w_m2_m"].shape == (
                    10,
                    archive["native_wavelength_micron"].size,
                )
                assert archive[f"{prefix}_{model}_flux_r100_w_m2_m"].shape == (
                    10,
                    archive["r100_centers_micron"].size,
                )
                vertical = archive[
                    f"{prefix}_{model}_vertical_flux_contribution_native_w_m2_m"
                ]
                assert vertical.shape == (
                    10,
                    n_cells + 1,
                    archive["native_wavelength_micron"].size,
                )
                assert vertical.dtype == np.float32
        p_rt_supported = archive["cells_80_petitradtrans_supported_case_mask"]
        p_rt_flux = archive["cells_80_petitradtrans_raw_flux_native_w_m2_m"]
        assert np.all(np.isfinite(p_rt_flux[p_rt_supported]))
        assert np.all(np.isnan(p_rt_flux[~p_rt_supported]))


def test_common_profiles_cover_all_grids_and_pg14_families() -> None:
    with np.load(PROFILES, allow_pickle=False) as archive:
        for n_cells in (40, 80, 160):
            assert archive[f"pressure_edges_{n_cells}_bar"].shape == (n_cells + 1,)
            assert archive[f"pressure_centers_{n_cells}_bar"].shape == (n_cells,)
            assert archive[f"isothermal_{n_cells}_cells"].shape == (n_cells,)
            assert archive[f"pg14_non_inverted_{n_cells}_cells"].shape == (n_cells,)
            assert archive[f"pg14_inverted_{n_cells}_cells"].shape == (n_cells,)
