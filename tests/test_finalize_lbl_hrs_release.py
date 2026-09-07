"""Focused tests for the compact LBL/HRS release finalizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples import finalize_lbl_hrs_release as finalizer


ROOT = Path(__file__).resolve().parents[1]
THREADS = {name: 1 for name in finalizer._THREAD_VARIABLES}
GIB = 1024**3


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table(path: Path) -> dict[str, Any]:
    _write(path, path.name.encode("utf-8"))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sampler(*, full: bool) -> dict[str, Any]:
    medians = {
        "temperature_K": 1499.0,
        "log10_h2o_vmr": -3.31,
        "log10_co_vmr": -3.02,
        "radial_velocity_km_s": 4.99,
    }
    runs = []
    for index, seed in enumerate((24680, 24681)):
        run: dict[str, Any] = {
            "seed": seed,
            "status": "completed",
            "converged": True,
            "likelihood_evaluations": 100 + index,
            "elapsed_seconds": 2.0 + index,
            "peak_rss_bytes": 120_000_000 + index,
        }
        if full:
            run.update(
                {
                    "posterior_medians": medians,
                    "log_evidence": -10.0 - index,
                    "log_evidence_error": 0.1,
                }
            )
        runs.append(run)
    return {
        "status": "pass",
        "scope": (
            "full four-parameter atmospheric posterior"
            if full
            else "one-dimensional H-minus VMR injection recovery"
        ),
        "settings": {
            "evidence_tolerance": 0.5,
            "max_iter": 0,
            "mpi_nprocs": 1,
            "n_live_points": 16,
            "sampling_efficiency": 0.8,
            "seed_pair": [24680, 24681],
        },
        "runs": runs,
        "gate": {"passed": True},
    }


def _write_reports(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    report_dir = tmp_path / "reports"
    h2o = _table(tmp_path / "opacity" / "h2o.h5")
    co = _table(tmp_path / "opacity" / "co.h5")
    tables = {"H2O": h2o, "CO": co}

    base_spectrum = _write(tmp_path / "plots" / "base_spectrum.npz", b"spectrum")
    base_plot = _write(tmp_path / "plots" / "kband.png", b"kband")
    science_output = _write(tmp_path / "oracle" / "science.npz", b"science")
    hminus_lrs = _write(tmp_path / "oracle" / "hminus_lrs.npz", b"lrs")
    hminus_hrs = _write(tmp_path / "oracle" / "hminus_hrs.npz", b"hrs")
    hminus_json = _write(tmp_path / "oracle" / "hminus.json", b"oracle")
    hminus_plot = _write(tmp_path / "plots" / "hminus.png", b"hminus")

    base = {
        "schema_version": 1,
        "status": "pass",
        "command": "python examples/benchmark_lbl_kband_emission.py",
        "acceptance": {"native": True, "gaussian": True},
        "resources": {
            "cpu_thread_limit": 1,
            "memory_limit_bytes": int(1.5 * GIB),
            "measured_peak_rss_bytes": 100_000_000,
        },
        "input_tables": tables,
        "artifacts": {
            "spectrum_npz": str(base_spectrum),
            "figure_png": str(base_plot),
        },
    }
    science = {
        "schema_version": 1,
        "status": "pass",
        "command": "python examples/benchmark_lbl_kband_science_grid.py",
        "acceptance": {"native_grid": True, "gaussian_response": True},
        "resources": {
            "cpu_thread_limit": 1,
            "memory_limit_bytes": int(1.5 * GIB),
            "measured_peak_rss_bytes": 110_000_000,
        },
        "table_inputs": tables,
        "oracle_inputs": {
            "baseline": {
                "path": str(science_output),
                "bytes": science_output.stat().st_size,
                "sha256": _sha256(science_output),
            }
        },
    }
    hminus = {
        "schema_version": 1,
        "status": "pass",
        "command": "python examples/benchmark_hminus_continuum.py",
        "resource_policy": {
            "thread_values": THREADS,
            "process_memory_ceiling_bytes": int(1.5 * GIB),
            "measured_peak_rss_bytes": 115_000_000,
            "processes": 1,
        },
        "source_table": {**co, "role": "zero-abundance carrier"},
        "lrs": {"status": "pass"},
        "hrs_native_and_gaussian_R100000": {
            "metrics": {
                "case": {
                    "native_gate_passed": True,
                    "gaussian_R100000_gate_passed": True,
                }
            }
        },
        "oracle": {
            "lrs_archive_path": str(hminus_lrs),
            "hrs_native_archive_path": str(hminus_hrs),
            "oracle_json_path": str(hminus_json),
        },
        "plot": str(hminus_plot),
        "pymultinest": _sampler(full=False),
    }
    full = {
        "schema_version": 1,
        "status": "pass_native_stride_1_stride_2_rejected",
        "command": "python examples/combined_resolution_full_pymultinest.py",
        "fit_passed": True,
        "inference_converged": True,
        "recovery_passed": True,
        "parameters_passed": True,
        "memory_guard_passed": True,
        "cpu_guard_passed": True,
        "truth": {
            "temperature_K": 1500.0,
            "log10_h2o_vmr": -3.3,
            "log10_co_vmr": -3.0,
            "radial_velocity_km_s": 5.0,
        },
        "resources": {
            "thread_values": THREADS,
            "max_memory_bytes": int(1.5 * GIB),
            "measured_peak_rss_bytes": 120_000_000,
        },
        "table_inputs": tables,
        "pymultinest": {
            **_sampler(full=True),
            "gate": {"passed": True},
        },
        "native_stride_comparison": {
            "status": "fail",
            "reference_stride": 1,
            "candidate_stride": 2,
            "nested_posterior_comparison": {
                "status": "fail",
                "evidence_tolerance": 0.5,
                "gate": {
                    "passed": False,
                    "evidence_agreement_with_reported_errors": False,
                    "paired_seed_runs": True,
                    "stride_1_natural_convergence": True,
                },
                "paired_runs": [
                    {
                        "seed": 24680,
                        "stride_1": {
                            "posterior_medians": {
                                "temperature_K": 1499.0,
                                "log10_h2o_vmr": -3.31,
                                "log10_co_vmr": -3.02,
                                "radial_velocity_km_s": 4.99,
                            },
                            "log_evidence": -10.0,
                        },
                        "stride_2": {
                            "posterior_medians": {
                                "temperature_K": 1501.0,
                                "log10_h2o_vmr": -3.30,
                                "log10_co_vmr": -3.01,
                                "radial_velocity_km_s": 4.98,
                            },
                            "log_evidence": -11.0,
                        },
                        "log_evidence_stride_2_minus_stride_1": -1.0,
                    },
                    {
                        "seed": 24681,
                        "stride_1": {
                            "posterior_medians": {
                                "temperature_K": 1498.0,
                                "log10_h2o_vmr": -3.32,
                                "log10_co_vmr": -3.03,
                                "radial_velocity_km_s": 5.01,
                            },
                            "log_evidence": -12.0,
                        },
                        "stride_2": {
                            "posterior_medians": {
                                "temperature_K": 1502.0,
                                "log10_h2o_vmr": -3.29,
                                "log10_co_vmr": -3.00,
                                "radial_velocity_km_s": 4.97,
                            },
                            "log_evidence": -14.0,
                        },
                        # Legacy spelling remains supported for old reports.
                        "evidence_difference_stride_2_minus_stride_1": -2.0,
                    }
                ],
            },
        },
    }
    paths = {
        "base": report_dir / finalizer._REQUIRED_REPORTS["base"],
        "science": report_dir / finalizer._REQUIRED_REPORTS["science"],
        "hminus": report_dir / finalizer._REQUIRED_REPORTS["hminus"],
        "full": report_dir / finalizer._REQUIRED_REPORTS["full"],
    }
    for role, payload in {
        "base": base,
        "science": science,
        "hminus": hminus,
        "full": full,
    }.items():
        paths[role].parent.mkdir(parents=True, exist_ok=True)
        paths[role].write_text(json.dumps(payload), encoding="utf-8")
    return report_dir, paths


def test_build_manifest_records_provenance_plots_and_rejected_stride(tmp_path: Path) -> None:
    report_dir, _ = _write_reports(tmp_path)
    output = tmp_path / "manifest.json"
    summary_plot = tmp_path / "plots" / "summary.png"

    manifest = finalizer.build_release_manifest(
        report_dir=report_dir,
        output_path=output,
        root=ROOT,
        summary_plot=summary_plot,
        command="python examples/finalize_lbl_hrs_release.py --summary-plot",
    )

    assert output.is_file()
    assert summary_plot.is_file()
    assert manifest["status"] == "complete_with_rejected_stride_2_candidate"
    assert set(manifest["report_hashes"]) == {"base", "science", "hminus", "full"}
    assert manifest["abundance_contract"]["mass_fractions_in_robert_state"] is False
    assert manifest["resources"]["thread_variables"] == list(finalizer._THREAD_VARIABLES)
    assert manifest["resources"]["mpi_processes"] == 1
    assert manifest["resources"]["strict_process_memory_under_2_gib"] is True
    assert manifest["stride_policy"]["stride_1_approval"]["status"] == (
        "approved_reference"
    )
    assert manifest["stride_policy"]["stride_1_approval"]["production_default"]
    assert manifest["stride_policy"]["stride_2_rejection"]["status"] == "rejected"
    assert manifest["stride_policy"]["stride_2_rejection"]["completed_negative_result"]
    assert manifest["stride_policy"][
        "evidence_differences_stride_2_minus_stride_1"
    ] == pytest.approx([-1.0, -2.0])
    assert len(manifest["opacity_inputs"]) == 2
    assert set(manifest["samplers"]) == {"hminus", "full"}
    assert manifest["plots"]["kband"]["command"].startswith("python examples")
    assert manifest["plots"]["hminus"]["input_data_sha256"]
    assert manifest["plots"]["summary"]["output_sha256"] == _sha256(summary_plot)
    assert all(
        sampler["backend"] == "PyMultiNest/MultiNest"
        for sampler in manifest["samplers"].values()
    )

    second = finalizer.build_release_manifest(
        report_dir=report_dir,
        output_path=output,
        root=ROOT,
        summary_plot=summary_plot,
        command="python examples/finalize_lbl_hrs_release.py --summary-plot",
    )
    assert second["plots"]["summary"]["output_sha256"] == manifest["plots"]["summary"][
        "output_sha256"
    ]


def test_manifest_rejects_pending_or_missing_artifact(tmp_path: Path) -> None:
    report_dir, paths = _write_reports(tmp_path)
    payload = json.loads(paths["base"].read_text(encoding="utf-8"))
    payload["unresolved"] = "PENDING_PARENT_MEASUREMENTS"
    paths["base"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ReleaseManifestError, match="unresolved"):
        finalizer.build_release_manifest(
            report_dir=report_dir,
            output_path=tmp_path / "pending.json",
            root=ROOT,
        )

    report_dir, paths = _write_reports(tmp_path / "missing")
    payload = json.loads(paths["hminus"].read_text(encoding="utf-8"))
    Path(payload["plot"]).unlink()
    with pytest.raises(finalizer.ReleaseManifestError, match="required artifact"):
        finalizer.build_release_manifest(
            report_dir=report_dir,
            output_path=tmp_path / "missing.json",
            root=ROOT,
        )


def test_manifest_rejects_failed_required_gate_and_unqualified_stride_failure(
    tmp_path: Path,
) -> None:
    report_dir, paths = _write_reports(tmp_path)
    payload = json.loads(paths["hminus"].read_text(encoding="utf-8"))
    payload["hrs_native_and_gaussian_R100000"]["metrics"]["case"][
        "native_gate_passed"
    ] = False
    paths["hminus"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ReleaseManifestError, match="hminus required gate"):
        finalizer.build_release_manifest(
            report_dir=report_dir,
            output_path=tmp_path / "hminus-fail.json",
            root=ROOT,
        )

    report_dir, paths = _write_reports(tmp_path / "stride")
    payload = json.loads(paths["full"].read_text(encoding="utf-8"))
    payload["native_stride_comparison"]["nested_posterior_comparison"]["gate"][
        "evidence_agreement_with_reported_errors"
    ] = True
    payload["native_stride_comparison"]["nested_posterior_comparison"]["gate"][
        "passed"
    ] = True
    paths["full"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ReleaseManifestError, match="explicit stride-2"):
        finalizer.build_release_manifest(
            report_dir=report_dir,
            output_path=tmp_path / "stride-fail.json",
            root=ROOT,
        )


def test_manifest_rejects_memory_at_two_gib(tmp_path: Path) -> None:
    report_dir, paths = _write_reports(tmp_path)
    payload = json.loads(paths["full"].read_text(encoding="utf-8"))
    payload["resources"]["max_memory_bytes"] = 2 * GIB
    paths["full"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(finalizer.ReleaseManifestError, match="strict process memory"):
        finalizer.build_release_manifest(
            report_dir=report_dir,
            output_path=tmp_path / "memory-fail.json",
            root=ROOT,
        )
