"""Focused tests for the report-only ROBERT H-minus evidence comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from examples import compare_wasp77ab_robert_hminus as comparator


def _source_hashes() -> dict[str, dict[str, Any]]:
    return {
        "cube14v3": {
            "sha256": "a" * 64,
            "size_bytes": 10,
            "verified": True,
        },
        "nirspec": {
            "sha256": "b" * 64,
            "size_bytes": 20,
            "verified": True,
        },
        "cia": {
            "sha256": "d" * 64,
            "verified": True,
        },
    }


def _source_manifest() -> dict[str, Any]:
    return {
        "status": "pass_primary_files_present",
        "files": [
            {
                "name": "cube14v3",
                "sha256_expected": "a" * 64,
                "size_bytes": 10,
            },
            {
                "name": "not_in_report",
                "sha256_expected": "c" * 64,
                "size_bytes": 30,
            },
        ]
    }


def _runs(log_evidence: float) -> dict[str, Any]:
    records = []
    for index, seed in enumerate(comparator.EXPECTED_SEEDS):
        records.append(
            {
                "seed": seed,
                "status": "pass",
                "converged": True,
                "resource_gate_passed": True,
                "peak_rss_bytes": 100_000_000,
                "log_evidence": log_evidence - index,
                "log_evidence_error": 0.2,
            }
        )
    return {
        "runs": records,
        "evidence_comparison": {
            "seed_pair": list(comparator.EXPECTED_SEEDS),
            "delta_log_evidence": 1.0,
            "gate_passed": True,
        },
        "gate_passed": True,
    }


def _report(*, enabled: bool, log_offset: float = 0.0) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "pass",
        "hminus_enabled": enabled,
        "hminus_policy": {
            "enabled": enabled,
            "same_parameterization_for_on_off_comparison": True,
        },
        "modes": ["hrs_only", "lrs_only", "joint"],
        "sampler_policy": {
            "sampler": "PyMultiNest",
            "backend": "MultiNest",
            "seeds": list(comparator.EXPECTED_SEEDS),
            "n_live_points": 64,
            "max_iter": 0,
            "sampling_efficiency": 0.8,
            "evidence_tolerance": 0.5,
            "mpi_processes": 1,
            "ultranest_allowed": False,
        },
        "composition_policy": {
            "robert_convention": "volume_mixing_ratio",
            "mass_fraction_parameters_in_robert": False,
            "mass_fraction_parameters": False,
        },
        "data": {
            "hrs_subset": "K band orders",
            "lrs_detectors": ["NRS1", "NRS2"],
            "hrs_order_indices": [1, 2],
            "hrs_point_count": 100,
            "lrs_point_count": 150,
            "source_hashes": _source_hashes(),
            "noise_identity": {"noise_seed": 20260831},
        },
        "stride_selection": {
            "hrs": 2,
            "lrs": 25,
            "release_claim": True,
        },
        "resources": {
            "thread_values": {name: 1 for name in comparator.THREAD_VARIABLES},
            "peak_rss_bytes": 100_000_000,
            "configured_process_rss_limit_bytes": int(1.9 * comparator.GIB),
            "gate_passed": True,
        },
        "runs": {
            mode: _runs(-100.0 + log_offset) for mode in ("hrs_only", "lrs_only", "joint")
        },
    }


def test_matched_reports_compute_per_mode_and_combined_differences() -> None:
    on = _report(enabled=True, log_offset=0.0)
    off = _report(enabled=False, log_offset=-2.0)
    result = comparator.compare_reports(on, off, source_manifest=_source_manifest())

    assert result["status"] == "pass"
    assert result["stride_selection"] == {"hrs": 2, "lrs": 25, "matched": True}
    assert result["source_manifest"] == {"checked": True, "overlap_count": 1}
    evidence = result["evidence_difference"]
    assert evidence["per_mode"]["joint"]["per_seed"][0][
        "delta_log_evidence_on_minus_off"
    ] == pytest.approx(2.0)
    assert evidence["per_mode"]["joint"]["per_seed"][0]["propagated_error"] == pytest.approx(
        0.2 * 2.0**0.5
    )
    assert evidence["combined"]["delta_log_evidence_on_minus_off"] == pytest.approx(12.0)
    assert result["hminus_identifiability"]["bound_free"]["k_band_status"] == "zero"
    assert "H, e-, and H- VMR" in result["hminus_identifiability"]["free_free"]["controls"]


def test_report_files_write_atomic_compact_json(tmp_path: Path) -> None:
    on_path = tmp_path / "on.json"
    off_path = tmp_path / "off.json"
    source_path = tmp_path / "sources.json"
    on_path.write_text(json.dumps(_report(enabled=True)), encoding="utf-8")
    off_path.write_text(json.dumps(_report(enabled=False, log_offset=-2.0)), encoding="utf-8")
    source_path.write_text(json.dumps(_source_manifest()), encoding="utf-8")
    output = tmp_path / "comparison.json"

    result = comparator.compare_report_files(
        on_path,
        off_path,
        output_path=output,
        source_manifest_path=source_path,
    )

    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["reports"]["on_sha256"] == comparator._sha256(on_path)
    assert result["reports"]["off_sha256"] == comparator._sha256(off_path)


def test_mismatched_identity_or_sampler_settings_are_rejected() -> None:
    on = _report(enabled=True)
    off = _report(enabled=False)
    off["data"]["noise_identity"] = {"noise_seed": 7}
    with pytest.raises(comparator.HMinusComparisonError, match="noise identities"):
        comparator.compare_reports(on, off)

    off = _report(enabled=False)
    off["sampler_policy"]["n_live_points"] = 32
    with pytest.raises(comparator.HMinusComparisonError, match="sampler controls"):
        comparator.compare_reports(on, off)


def test_missing_or_nonzero_mass_fraction_state_is_rejected() -> None:
    on = _report(enabled=True)
    off = _report(enabled=False)
    off["composition_policy"]["mass_fraction_parameters"] = ["H2O"]
    with pytest.raises(comparator.HMinusComparisonError, match="mass-fraction"):
        comparator.compare_reports(on, off)

    off = _report(enabled=False)
    off["sampler_policy"]["max_iter"] = 1
    with pytest.raises(comparator.HMinusComparisonError, match="max_iter"):
        comparator.compare_reports(on, off)


def test_real_robert_sampler_accepts_six_threads_and_rejects_seven() -> None:
    on = _report(enabled=True)
    off = _report(enabled=False, log_offset=-2.0)
    six_threads = {name: 6 for name in comparator.THREAD_VARIABLES}
    on["resources"]["thread_values"] = six_threads
    off["resources"]["thread_values"] = dict(six_threads)

    result = comparator.compare_reports(on, off)

    assert comparator.MAX_THREADS == 6
    assert result["sampler_contract"]["resource_policy"]["on"]["thread_limit"] == 6
    assert result["sampler_contract"]["resource_policy"]["on"]["threads_at_most_6"] is True
    assert "threads_at_most_3" not in result["sampler_contract"]["resource_policy"]["on"]

    on["resources"]["thread_values"][comparator.THREAD_VARIABLES[0]] = 7
    with pytest.raises(comparator.HMinusComparisonError, match="1..6"):
        comparator.compare_reports(on, off)
