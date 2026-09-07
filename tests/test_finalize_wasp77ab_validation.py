"""Small-fixture tests for the WASP-77Ab validation finalizer."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import pytest

from examples import finalize_wasp77ab_validation as finalizer


THREADS = {name: 1 for name in finalizer.THREAD_VARIABLES}


def _write(path: Path, text: str = "plot") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resources(*, peak: int = 100_000_000) -> dict[str, Any]:
    return {
        "thread_values": dict(THREADS),
        "peak_rss_bytes": peak,
        "process_rss_limit_bytes": int(1.9 * finalizer.GIB),
    }


def _source() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass_primary_files_present",
        "zenodo": {"doi": "10.5281/zenodo.10382053", "license": "CC-BY-4.0"},
        "acquisition": {"deserializes_archives": False},
        "resource_policy": {"max_threads": 3, "thread_variables": list(THREADS)},
        "files": [
            {
                "name": f"primary-{index}",
                "optional": False,
                "status": "present_verified",
                "sha256_expected": f"{index:064x}",
                "sha256": f"{index:064x}",
                "size_bytes": 10,
                "local_path": f"external_data/primary-{index}",
            }
            for index in range(7)
        ],
    }


def _lrs(plot: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "pass",
        "abundance_state": "ROBERT VMR only",
        "data": {
            "table_points": 150,
            "detectors": ["nirspec_g395h_nrs1", "nirspec_g395h_nrs2"],
        },
        "metrics": {"smith_consistency_pass": True},
        "plot": {"path": str(plot), "sha256": _sha256(plot)},
        "resources": _resources(),
    }


def _hrs(plot: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass",
        "method": {"abundance_convention": "ROBERT volume mixing ratios only"},
        "observation": {"n_orders": 44, "n_frames": 79},
        "response": {
            "stage_order": ["rotational-broadening", "gaussian-high-resolution"]
        },
        "acceptance": {"finite_diagnostics": True, "memory": True},
        "plot": {"path": str(plot), "sha256": _sha256(plot)},
        "resources": _resources(),
    }


def _stride() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass",
        "acceptance": {"all": True},
        "atmosphere": {
            "composition_convention": "volume_mixing_ratio",
            "mass_fraction_parameters": False,
            "hminus_continuum": {
                "config_species": ["H-", "H", "e-"],
                "included_in_all_stride_evaluations": True,
            },
        },
        "hrs": {
            "candidate_strides": [2, 4],
            "selected_finest_passing_stride": 2,
            "metrics": {"2": {"gate_pass": True}},
        },
        "lrs": {
            "candidate_strides": [25, 50],
            "selected_finest_passing_stride": 25,
            "metrics": {"25": {"gate_pass": True}},
        },
        "resources": {
            **_resources(),
            "opacity_estimate_limit_bytes": finalizer.OPACITY_MEMORY_LIMIT_BYTES,
        },
    }


def _sampler_policy() -> dict[str, Any]:
    return {
        "sampler": "PyMultiNest",
        "backend": "MultiNest",
        "seeds": list(finalizer.EXPECTED_SEEDS),
        "max_iter": 0,
        "mpi_processes": 1,
        "evidence_tolerance": 0.5,
    }


def _sampler_runs() -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "status": "pass",
            "converged": True,
            "resource_gate_passed": True,
            "peak_rss_bytes": 100_000_000,
        }
        for seed in finalizer.EXPECTED_SEEDS
    ]


def _sampler_report(*, notes: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "pass",
        "sampler_policy": _sampler_policy(),
        "runs": {
            "joint": {
                "runs": _sampler_runs(),
                "evidence_comparison": {
                    "delta_log_evidence": 0.1,
                    "gate_passed": True,
                },
            }
        },
        "resources": _resources(),
    }
    if notes:
        payload["hminus_policy"] = {
            "bound_free": "zero beyond 1.6421 micron",
            "free_free": "fixed H and retrieved electron VMR",
            "identifiability": "H-minus VMR prior-dominated in selected window",
        }
    return payload


def _injection() -> dict[str, Any]:
    scales = [float(order + 1) for order in range(19)]
    digest = hashlib.sha256(
        struct.pack("<19d", *(float(order + 1) for order in range(19)))
    ).hexdigest()
    branch = {
        "finite": True,
        "truth_loglike": 3.0,
        "null_loglike": 1.0,
        "off_velocity_loglike": 2.0,
        "truth_minus_null": 2.0,
        "truth_minus_off_velocity": 1.0,
        "off_velocity_delta_km_s": 13.0,
        "truth_preferred_to_null": True,
        "truth_preferred_to_off_velocity": True,
    }
    mode_results = {
        mode: {
            "hminus_on": dict(branch),
            "hminus_off": dict(branch),
            "hminus_on_off_finite": True,
            "gate_passed": True,
        }
        for mode in ("hrs_only", "lrs_only", "joint")
    }
    return {
        "schema_version": "1.0",
        "record_type": finalizer.EXPECTED_INJECTION_RECORD_TYPE,
        "status": "pass",
        "real_data_loaded": True,
        "sampler": {"used": False, "purpose": "deterministic only"},
        "acceptance": {
            "all_modes_finite": True,
            "truth_preferred_to_null": True,
            "truth_preferred_to_off_velocity": True,
            "hminus_on_off_evaluated": True,
            "resource_gate": True,
            "posterior_recovery_claim": False,
        },
        "checks": {"real_data_truth_recovery_claim": False},
        "composition_policy": {
            "robert_convention": "volume_mixing_ratio",
            "mass_fraction_parameters_in_robert": False,
            "shared_atmosphere": True,
            "hminus_species": ["H-", "H", "e-"],
        },
        "truth_parameters": {
            "temperature_K": 2700.0,
            "log10_H2O_VMR": -4.02,
            "log10_CO_VMR": -3.91,
            "log10_Hminus_VMR": -8.0,
            "log10_electron_VMR": -3.0,
            "Kp": 190.74,
            "dVsys": -5.27,
            "log10_a_hrs": 0.0,
            "lrs_scale": 1.0,
        },
        "generative_model": {
            "hrs_noise_calibration": {
                "n_orders": 19,
                "sigma_by_order": scales,
                "sigma_sha256": digest,
                "sigma_summary": {
                    "min": min(scales),
                    "max": max(scales),
                    "mean": math.fsum(scales) / len(scales),
                    "rms": math.sqrt(
                        math.fsum(value * value for value in scales) / len(scales)
                    ),
                },
                "seed": 20260832,
                "scale_definition": "per-order RMS",
                "distribution": "independent Gaussian",
                "chunking": "one order at a time",
            },
        },
        "source_hashes": {
            "synthetic_fixture_source": {
                "verified": True,
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
        },
        "mode_results": mode_results,
        "resource_policy": {
            "thread_variables": list(finalizer.THREAD_VARIABLES),
            "max_threads": 6,
            "process_memory_limit_bytes": finalizer.CONFIGURED_PROCESS_MEMORY_LIMIT_BYTES,
            "hard_process_memory_limit_bytes": finalizer.PROCESS_MEMORY_HARD_LIMIT_BYTES,
            "mpi_processes": 1,
        },
        "resources": {
            "thread_values": {
                name: 6 for name in finalizer.THREAD_VARIABLES
            },
            "thread_limit": 6,
            "peak_rss_bytes": 100_000_000,
            "configured_process_rss_limit_bytes": finalizer.CONFIGURED_PROCESS_MEMORY_LIMIT_BYTES,
            "process_rss_hard_limit_bytes": finalizer.PROCESS_MEMORY_HARD_LIMIT_BYTES,
            "mpi_processes_configured": 1,
            "mpi_world_size_observed": 1,
            "threads_pass": True,
            "mpi_pass": True,
            "process_rss_pass": True,
            "gate_passed": True,
        },
    }


def _write_fixture(
    tmp_path: Path,
    *,
    complete_samplers: bool = False,
    notes: bool = False,
) -> tuple[Path, Path]:
    report_dir = tmp_path / "reports"
    root = tmp_path
    plot_lrs = _write(root / "plots" / "lrs.png")
    plot_hrs = _write(root / "plots" / "hrs.png")
    payloads: dict[str, dict[str, Any]] = {
        "source": _source(),
        "lrs": _lrs(plot_lrs),
        "hrs": _hrs(plot_hrs),
        "stride": _stride(),
        "injection": _injection(),
    }
    if complete_samplers:
        payloads["fixed_template_sampler"] = _sampler_report(notes=notes)
        payloads["robert_joint_sampler"] = _sampler_report(notes=notes)
    else:
        payloads["fixed_template_sampler"] = {
            "status": "preflight_pass",
            "sampler_policy": _sampler_policy(),
            "runs": {},
            "resources": _resources(),
        }
    for role, payload in payloads.items():
        (report_dir / finalizer.REPORT_NAMES[role]).parent.mkdir(
            parents=True, exist_ok=True
        )
        (report_dir / finalizer.REPORT_NAMES[role]).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return report_dir, root


def test_nonrequired_mode_records_pending_and_hashes_small_plots(tmp_path: Path) -> None:
    report_dir, root = _write_fixture(tmp_path)
    output = tmp_path / "manifest.json"

    manifest = finalizer.build_validation_manifest(
        report_dir=report_dir,
        output_path=output,
        root=root,
    )

    assert manifest["status"] == "pending_samplers_or_notes"
    assert "fixed_template_sampler" in manifest["pending"]
    assert "robert_joint_sampler" in manifest["pending"]
    assert "hminus_notes" in manifest["pending"]
    assert "injection" not in manifest["pending"]
    assert manifest["reports"]["injection"]["status"] == "pass"
    assert manifest["validation"]["injection"]["sampler_used"] is False
    assert manifest["validation"]["injection"]["resources"][
        "threads_at_most_6"
    ] is True
    assert manifest["validation"]["injection"]["resources"][
        "strict_process_memory_under_configured_limit"
    ] is True
    assert manifest["validation"]["stride"]["selected_strides"] == {
        "hrs": 2,
        "lrs": 25,
    }
    assert all(
        artifact["status"] == "verified"
        for role in ("lrs", "hrs")
        for artifact in manifest["plots"][role]
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == manifest["status"]


def test_injection_report_requires_operator_contract(tmp_path: Path) -> None:
    report_dir, root = _write_fixture(tmp_path)
    path = report_dir / finalizer.REPORT_NAMES["injection"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["acceptance"]["all_modes_finite"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ValidationFinalizerError, match="acceptance gate"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )


def test_injection_report_is_required_even_when_samplers_are_deferred(
    tmp_path: Path,
) -> None:
    report_dir, root = _write_fixture(tmp_path)
    (report_dir / finalizer.REPORT_NAMES["injection"]).unlink()
    with pytest.raises(finalizer.ValidationFinalizerError, match="injection report"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )


def test_injection_report_requires_six_threads_mpi_one_and_memory_headroom(
    tmp_path: Path,
) -> None:
    report_dir, root = _write_fixture(tmp_path)
    path = report_dir / finalizer.REPORT_NAMES["injection"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["resources"]["peak_rss_bytes"] = finalizer.CONFIGURED_PROCESS_MEMORY_LIMIT_BYTES
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ValidationFinalizerError, match="configured 1.9 GiB"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )

    payload["resources"]["peak_rss_bytes"] = 100_000_000
    payload["resource_policy"]["mpi_processes"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ValidationFinalizerError, match="MPI size 1"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )


def test_injection_report_requires_truth_and_structured_noise_checksum(
    tmp_path: Path,
) -> None:
    report_dir, root = _write_fixture(tmp_path)
    path = report_dir / finalizer.REPORT_NAMES["injection"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["truth_parameters"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ValidationFinalizerError, match="truth_parameters"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )

    payload["truth_parameters"] = {"temperature_K": 2700.0}
    payload["generative_model"]["hrs_noise_calibration"]["sigma_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ValidationFinalizerError, match="checksum"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )


def test_require_samplers_fails_for_preflight_or_missing_reports(tmp_path: Path) -> None:
    report_dir, root = _write_fixture(tmp_path)
    with pytest.raises(finalizer.ValidationFinalizerError, match="pending"):
        finalizer.build_validation_manifest(
            report_dir=report_dir,
            output_path=tmp_path / "manifest.json",
            root=root,
            require_samplers=True,
        )


def test_complete_two_seed_sampler_contract_passes(tmp_path: Path) -> None:
    report_dir, root = _write_fixture(tmp_path, complete_samplers=True, notes=True)
    manifest = finalizer.build_validation_manifest(
        report_dir=report_dir,
        output_path=tmp_path / "manifest.json",
        root=root,
        require_samplers=True,
    )

    assert manifest["status"] == "pass"
    assert all(record["complete"] for record in manifest["samplers"].values())
    assert manifest["composition"]["hminus_notes_pending"] is False
    assert manifest["sampler_policy"]["max_iter"] == 0


def test_real_robert_sampler_allows_six_threads_but_reference_stays_at_three(
    tmp_path: Path,
) -> None:
    report_dir, root = _write_fixture(tmp_path, complete_samplers=True, notes=True)
    fixed_path = report_dir / finalizer.REPORT_NAMES["fixed_template_sampler"]
    fixed_payload = json.loads(fixed_path.read_text(encoding="utf-8"))
    fixed_payload["resources"]["thread_values"] = {
        name: 3 for name in finalizer.THREAD_VARIABLES
    }
    fixed_path.write_text(json.dumps(fixed_payload), encoding="utf-8")

    robert_path = report_dir / finalizer.REPORT_NAMES["robert_joint_sampler"]
    robert_payload = json.loads(robert_path.read_text(encoding="utf-8"))
    robert_payload["sampler_policy"]["max_threads"] = 6
    robert_payload["resources"]["thread_values"] = {
        name: 6 for name in finalizer.THREAD_VARIABLES
    }
    robert_path.write_text(json.dumps(robert_payload), encoding="utf-8")

    manifest = finalizer.build_validation_manifest(
        report_dir=report_dir,
        output_path=tmp_path / "manifest.json",
        root=root,
        require_samplers=True,
    )

    assert finalizer.MAX_THREADS == 6
    assert manifest["samplers"]["fixed_template_sampler"]["max_threads"] == 3
    assert manifest["samplers"]["robert_joint_sampler"]["max_threads"] == 6
    assert manifest["samplers"]["robert_joint_sampler"]["resources"][
        "threads_at_most_6"
    ] is True
    assert "threads_at_most_3" not in manifest["samplers"]["robert_joint_sampler"][
        "resources"
    ]


def test_reference_sampler_rejects_the_new_six_thread_limit(tmp_path: Path) -> None:
    report_dir, root = _write_fixture(tmp_path, complete_samplers=True, notes=True)
    fixed_path = report_dir / finalizer.REPORT_NAMES["fixed_template_sampler"]
    fixed_payload = json.loads(fixed_path.read_text(encoding="utf-8"))
    fixed_payload["resources"]["thread_values"] = {
        name: 4 for name in finalizer.THREAD_VARIABLES
    }
    fixed_path.write_text(json.dumps(fixed_payload), encoding="utf-8")

    with pytest.raises(finalizer.ValidationFinalizerError, match="1..3"):
        finalizer.build_validation_manifest(
            report_dir=report_dir,
            output_path=tmp_path / "manifest.json",
            root=root,
            require_samplers=True,
        )


def test_plot_hash_mismatch_and_mass_fraction_state_are_rejected(tmp_path: Path) -> None:
    report_dir, root = _write_fixture(tmp_path)
    path = report_dir / finalizer.REPORT_NAMES["lrs"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plot"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ValidationFinalizerError, match="SHA-256"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )

    payload["plot"]["sha256"] = _sha256(root / "plots" / "lrs.png")
    payload["mass_fraction_parameters"] = ["H2O"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ValidationFinalizerError, match="mass-fraction"):
        finalizer.build_validation_manifest(
            report_dir=report_dir, output_path=tmp_path / "manifest.json", root=root
        )
