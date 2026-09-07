"""Build the compact LBL/HRS release manifest from measured reports.

This file reads compact JSON and small plot/output artifacts only. It does not
run a forward model, open an opacity table, or read sampler chain files.
External release work must finish before the default output can be written.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
from typing import Any


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
_MAX_THREADS = 3
_GIB = 1024**3
_PROCESS_MEMORY_LIMIT_BYTES = 2 * _GIB
_DEFAULT_REPORT_DIR_NAME = "docs/data"
_DEFAULT_OUTPUT_NAME = "docs/data/lbl_hrs_release_manifest_20260831.json"
_REQUIRED_REPORTS = {
    "base": "lbl_kband_validation_20260829.json",
    "science": "lbl_kband_science_grid_20260830.json",
    "hminus": "hminus_continuum_validation_20260831.json",
    "full": "combined_resolution_full_pymultinest_20260831.json",
}
_OPTIONAL_REPORTS = {
    "combined_proof": "combined_resolution_recovery_20260830.json",
}
_K_BAND_SOURCE = "examples/benchmark_lbl_kband_emission.py"
_HMINUS_SOURCE = "examples/benchmark_hminus_continuum.py"
_PYRT_SOURCE_URL = (
    "https://petitradtrans.readthedocs.io/en/latest/content/notebooks/"
    "high_resolution_spectra.html"
)
_NO_LICENSE = (
    "not recorded in the compact report; verify the upstream licence before "
    "archive publication"
)


# Keep the optional summary plot inside the laptop policy before matplotlib can
# load. The finalizer itself uses one thread by default.
for _name in _THREAD_VARIABLES:
    try:
        _value = int(os.environ.get(_name, "1"))
    except ValueError:
        _value = 1
    os.environ[_name] = str(min(_MAX_THREADS, max(1, _value)))


ROOT = Path(__file__).resolve().parents[1]


class ReleaseManifestError(RuntimeError):
    """Raised when measured release evidence is incomplete or inconsistent."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseManifestError(f"{label} must be a JSON object")
    return value


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseManifestError(f"{label} is missing")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, root: Path) -> Path:
    if isinstance(value, Path):
        raw = value.expanduser()
    else:
        raw = Path(_nonempty_text(value, "artifact path")).expanduser()
    if raw.is_absolute():
        return raw
    return (root / raw).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _scan_for_string(value: Any, target: str, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        if target in value:
            found.append(path)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_scan_for_string(child, target, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_scan_for_string(child, target, f"{path}[{index}]"))
    return found


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseManifestError(f"required report is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseManifestError(f"cannot read report {path}: {error}") from error
    return dict(_mapping(payload, f"report {path}"))


def _report_records(
    *,
    report_dir: Path,
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    report_specs = {**_REQUIRED_REPORTS, **_OPTIONAL_REPORTS}
    for role, filename in report_specs.items():
        path = (report_dir / filename).resolve()
        if not path.is_file() and role in _OPTIONAL_REPORTS:
            continue
        payload = _read_report(path)
        pending_paths = _scan_for_string(payload, "PENDING_PARENT_MEASUREMENTS")
        if pending_paths:
            raise ReleaseManifestError(
                f"{role} report contains unresolved "
                f"PENDING_PARENT_MEASUREMENTS at {pending_paths[0]}"
            )
        command = _nonempty_text(payload.get("command"), f"{role} report command")
        status = _nonempty_text(payload.get("status"), f"{role} report status")
        records[role] = {
            "path": _display_path(path, root),
            "sha256": _sha256(path),
            "status": status,
            "command": command,
            "schema_version": payload.get("schema_version"),
        }
        payloads[role] = payload
    return records, payloads


def _require_status(payload: Mapping[str, Any], role: str) -> None:
    status = payload.get("status")
    if status != "pass":
        raise ReleaseManifestError(f"{role} required gate has status {status!r}")


def _boolean_values(value: Any, key_suffix: str) -> list[tuple[str, bool]]:
    values: list[tuple[str, bool]] = []

    def visit(child: Any, path: str) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                nested_path = f"{path}.{key}"
                if key.endswith(key_suffix) and isinstance(nested, bool):
                    values.append((nested_path, nested))
                else:
                    visit(nested, nested_path)
        elif isinstance(child, list):
            for index, nested in enumerate(child):
                visit(nested, f"{path}[{index}]")

    visit(value, "$")
    return values


def _validate_native_report(role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_status(payload, role)
    acceptance = _mapping(payload.get("acceptance"), f"{role}.acceptance")
    boolean_gates = [value for value in acceptance.values() if isinstance(value, bool)]
    if not boolean_gates:
        raise ReleaseManifestError(f"{role} has no boolean native acceptance gates")
    failed = [key for key, value in acceptance.items() if value is False]
    if failed:
        raise ReleaseManifestError(f"{role} native gate failed: {failed[0]}")
    return {"status": "pass", "gates": dict(acceptance)}


def _validate_hminus(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_status(payload, "hminus")
    lrs = _mapping(payload.get("lrs"), "hminus.lrs")
    _require_status(lrs, "hminus LRS")
    hminus_gate_values = _boolean_values(payload, "_gate_passed")
    if not hminus_gate_values:
        raise ReleaseManifestError("hminus report has no component gate results")
    failed = [path for path, value in hminus_gate_values if not value]
    if failed:
        raise ReleaseManifestError(f"hminus required gate failed: {failed[0]}")
    sampler = payload.get("pymultinest")
    if isinstance(sampler, Mapping) and sampler.get("status") not in (None, "pass"):
        command = str(payload.get("command", ""))
        if "--run-pymultinest" in command:
            raise ReleaseManifestError("hminus PyMultiNest gate failed")
    return {
        "status": "pass",
        "component_gate_count": len(hminus_gate_values),
        "gate_paths": [path for path, _ in hminus_gate_values],
    }


def _validate_full(payload: Mapping[str, Any]) -> dict[str, Any]:
    pymultinest = _mapping(payload.get("pymultinest"), "full.pymultinest")
    if pymultinest.get("status") != "pass":
        raise ReleaseManifestError("full four-parameter PyMultiNest gate failed")
    scope = str(pymultinest.get("scope", ""))
    if "full four-parameter" not in scope:
        raise ReleaseManifestError(
            "full report does not cover the full four-parameter posterior"
        )
    gate = _mapping(pymultinest.get("gate"), "full.pymultinest.gate")
    if gate.get("passed") is not True:
        raise ReleaseManifestError("full PyMultiNest gate is not passed")
    runs = pymultinest.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ReleaseManifestError("full PyMultiNest report has fewer than two runs")
    for index, run in enumerate(runs):
        record = _mapping(run, f"full.pymultinest.runs[{index}]")
        if record.get("status") != "completed" or record.get("converged") is not True:
            raise ReleaseManifestError(f"full PyMultiNest run {index + 1} is incomplete")
    required_flags = (
        "fit_passed",
        "inference_converged",
        "recovery_passed",
        "parameters_passed",
        "memory_guard_passed",
        "cpu_guard_passed",
    )
    for key in required_flags:
        if payload.get(key) is not True:
            raise ReleaseManifestError(f"full required gate failed: {key}")
    return {
        "status": "pass",
        "scope": scope,
        "runs": len(runs),
        "gate": dict(gate),
    }


def _stride_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    comparison_value = payload.get("native_stride_comparison")
    if not isinstance(comparison_value, Mapping):
        return {"status": "not_recorded"}
    comparison = dict(comparison_value)
    reference_stride = comparison.get("reference_stride")
    candidate_stride = comparison.get("candidate_stride")
    nested = comparison.get("nested_posterior_comparison")
    nested_map = _mapping(nested, "full.native_stride_comparison.nested_posterior_comparison")
    nested_gate = _mapping(nested_map.get("gate"), "full stride nested gate")
    comparison_status = comparison.get("status")
    nested_status = nested_map.get("status")
    evidence_rejected = (
        nested_gate.get("evidence_agreement_with_reported_errors") is False
        or nested_gate.get("passed") is False
    )
    explicitly_rejected = (
        reference_stride == 1
        and candidate_stride == 2
        and comparison_status in {"fail", "rejected"}
        and nested_status in {"fail", "rejected"}
        and evidence_rejected
    )
    if comparison_status in {"fail", "rejected"} and not explicitly_rejected:
        raise ReleaseManifestError(
            "native stride report failed without an explicit stride-2 rejection"
        )
    if comparison_status not in {"pass", "fail", "rejected"}:
        raise ReleaseManifestError(
            f"native stride report has unknown status {comparison_status!r}"
        )
    paired_runs = nested_map.get("paired_runs")
    if not isinstance(paired_runs, list) or not paired_runs:
        raise ReleaseManifestError("native stride report has no paired runs")
    paired_records = [
        _mapping(run, f"full.native_stride_comparison.paired_runs[{index}]")
        for index, run in enumerate(paired_runs)
    ]
    evidence_differences: list[float] | None = None
    if explicitly_rejected:
        if len(paired_records) < 2:
            raise ReleaseManifestError(
                "explicit stride-2 rejection needs both paired seed runs"
            )
        if not isinstance(nested_map.get("evidence_tolerance"), (int, float)):
            raise ReleaseManifestError(
                "explicit stride-2 rejection has no evidence tolerance"
            )
        for index, paired in enumerate(paired_records):
            if paired.get("seed") is None:
                raise ReleaseManifestError(
                    f"explicit stride-2 rejection is missing seed {index + 1}"
                )
        evidence_differences = _summary_evidence(paired_records)
    stride_1_approved = (
        reference_stride == 1
        and nested_gate.get("paired_seed_runs") is True
        and nested_gate.get("stride_1_natural_convergence") is True
    )
    result: dict[str, Any] = {
        "reference_stride": reference_stride,
        "candidate_stride": candidate_stride,
        "stride_1_approval": {
            "status": "approved_reference" if stride_1_approved else "not_approved",
            "production_default": bool(stride_1_approved and explicitly_rejected),
        },
        "stride_2_rejection": {
            "status": "rejected" if explicitly_rejected else "not_rejected",
            "completed_negative_result": explicitly_rejected,
            "reason": (
                "reported evidence disagreement with the stride-1 reference"
                if explicitly_rejected
                else None
            ),
        },
        "comparison_status": comparison_status,
        "nested_status": nested_status,
        "evidence_tolerance": nested_map.get("evidence_tolerance"),
        "paired_run_count": len(paired_runs),
    }
    if evidence_differences is not None:
        result["evidence_differences_stride_2_minus_stride_1"] = evidence_differences
    return result


def _validate_full_outer_status(
    payload: Mapping[str, Any], stride: Mapping[str, Any]
) -> None:
    """Allow only a measured pass or a measured rejected-stride status."""

    status = payload.get("status")
    rejection_complete = (
        stride.get("stride_2_rejection", {}).get("completed_negative_result") is True
    )
    comparison_status = stride.get("comparison_status")
    if status == "pass":
        if comparison_status in {"fail", "rejected"}:
            raise ReleaseManifestError(
                "full report says pass but its stride-2 comparison failed"
            )
        return
    if status == "pass_native_stride_1_stride_2_rejected":
        if not rejection_complete:
            raise ReleaseManifestError(
                "pass_native_stride_1_stride_2_rejected needs complete stride-2 evidence"
            )
        return
    # Keep compatibility with the earlier compact report classification. It is
    # accepted only when the same explicit rejection evidence is present.
    if status == "fail" and rejection_complete:
        return
    raise ReleaseManifestError(
        f"full report has unsupported or unqualified outer status {status!r}"
    )


def _find_key(value: Any, names: set[str]) -> Any:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in names:
                return child
            found = _find_key(child, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, names)
            if found is not None:
                return found
    return None


def _thread_values(payload: Mapping[str, Any], role: str) -> dict[str, Any]:
    value = _find_key(payload, {"thread_values"})
    source = "reported.thread_values"
    if not isinstance(value, Mapping):
        limit = _find_key(payload, {"cpu_thread_limit", "cpu_threads"})
        if isinstance(limit, (int, float)) and not isinstance(limit, bool):
            value = {name: int(limit) for name in _THREAD_VARIABLES}
            source = "inferred_from_cpu_thread_limit"
    if not isinstance(value, Mapping):
        raise ReleaseManifestError(f"{role} has no numerical thread settings")
    values: dict[str, int] = {}
    for name in _THREAD_VARIABLES:
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ReleaseManifestError(f"{role} is missing thread variable {name}")
        integer = int(raw)
        if integer != raw or integer < 1 or integer > _MAX_THREADS:
            raise ReleaseManifestError(
                f"{role}.{name}={raw!r} is outside the allowed range 1..3"
            )
        values[name] = integer
    return {"values": values, "source": source}


def _mpi_processes(payload: Mapping[str, Any], role: str) -> dict[str, Any]:
    value = _find_key(payload, {"mpi_nprocs", "mpi_processes", "mpi_size"})
    source = "reported"
    if value is None:
        value = 1
        source = "default_for_non_sampler_report"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseManifestError(f"{role} has invalid MPI process count")
    integer = int(value)
    if integer != value or integer < 1 or integer > _MAX_THREADS:
        raise ReleaseManifestError(f"{role} has invalid MPI process count {value!r}")
    return {"count": integer, "source": source}


def _memory_values(
    value: Any, path: str = "$"
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    observed: list[tuple[str, int]] = []
    configured: list[tuple[str, int]] = []
    observed_names = {"measured_peak_rss_bytes", "peak_rss_bytes"}
    configured_names = {
        "memory_limit_bytes",
        "max_memory_bytes",
        "process_memory_ceiling_bytes",
        "max_process_bytes",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in observed_names and isinstance(child, (int, float)):
                observed.append((child_path, int(child)))
            elif key in configured_names and isinstance(child, (int, float)):
                configured.append((child_path, int(child)))
            else:
                child_observed, child_configured = _memory_values(child, child_path)
                observed.extend(child_observed)
                configured.extend(child_configured)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_observed, child_configured = _memory_values(child, f"{path}[{index}]")
            observed.extend(child_observed)
            configured.extend(child_configured)
    return observed, configured


def _resource_manifest(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    threads: dict[str, Any] = {}
    mpi: dict[str, Any] = {}
    observed: dict[str, int] = {}
    configured: dict[str, int] = {}
    for role, payload in payloads.items():
        threads[role] = _thread_values(payload, role)
        mpi[role] = _mpi_processes(payload, role)
        report_observed, report_configured = _memory_values(payload)
        observed.update({f"{role}:{path}": value for path, value in report_observed})
        configured.update({f"{role}:{path}": value for path, value in report_configured})
    all_memory = [*observed.values(), *configured.values()]
    if not all_memory:
        raise ReleaseManifestError("reports contain no process memory measurements")
    invalid_memory = [
        value
        for value in all_memory
        if value <= 0 or value >= _PROCESS_MEMORY_LIMIT_BYTES
    ]
    if invalid_memory:
        raise ReleaseManifestError("a report does not satisfy strict process memory <2 GiB")
    maximum_threads = {
        name: max(record["values"][name] for record in threads.values())
        for name in _THREAD_VARIABLES
    }
    maximum_mpi = max(record["count"] for record in mpi.values())
    return {
        "thread_variables": list(_THREAD_VARIABLES),
        "thread_values": maximum_threads,
        "thread_values_by_report": threads,
        "mpi_processes": maximum_mpi,
        "mpi_processes_by_report": mpi,
        "observed_peak_rss_bytes_by_report": observed,
        "configured_process_memory_bytes_by_report": configured,
        "strict_process_memory_limit_bytes": _PROCESS_MEMORY_LIMIT_BYTES,
        "strict_process_memory_under_2_gib": True,
        "threads_at_most_3": True,
        "mpi_processes_at_most_3": True,
    }


def _artifact(
    value: Any,
    *,
    label: str,
    root: Path,
    recorded_bytes: Any = None,
    recorded_sha256: Any = None,
) -> dict[str, Any]:
    path = _resolve_path(value, root)
    if not path.is_file():
        raise ReleaseManifestError(f"required artifact is missing: {path}")
    actual_bytes = path.stat().st_size
    if recorded_bytes is not None:
        try:
            expected_bytes = int(recorded_bytes)
        except (TypeError, ValueError) as error:
            raise ReleaseManifestError(f"{label} has invalid recorded size") from error
        if expected_bytes != actual_bytes:
            raise ReleaseManifestError(
                f"{label} size mismatch: report={expected_bytes}, file={actual_bytes}"
            )
    actual_sha256 = _sha256(path)
    if recorded_sha256 is not None and str(recorded_sha256).lower() != actual_sha256:
        raise ReleaseManifestError(f"{label} SHA-256 does not match its report")
    return {
        "path": _display_path(path, root),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def _table_records(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for role, payload in payloads.items():
        candidates: list[tuple[str, Mapping[str, Any]]] = []
        for key in ("input_tables", "table_inputs"):
            table_map = payload.get(key)
            if isinstance(table_map, Mapping):
                candidates.extend(
                    (
                        f"{role}.{key}.{species}",
                        _mapping(record, f"{role}.{key}.{species}"),
                    )
                    for species, record in table_map.items()
                )
        source_table = payload.get("source_table")
        if isinstance(source_table, Mapping) and source_table.get("path"):
            candidates.append((f"{role}.source_table", source_table))
        for label, record in candidates:
            path = _resolve_path(record.get("path"), root)
            if not path.is_file():
                raise ReleaseManifestError(f"required opacity table is missing: {path}")
            sha256_value = _nonempty_text(record.get("sha256"), f"{label}.sha256").lower()
            if len(sha256_value) != 64 or any(character not in "0123456789abcdef" for character in sha256_value):
                raise ReleaseManifestError(f"{label}.sha256 is not a SHA-256 value")
            recorded_bytes = record.get("file_bytes", record.get("bytes"))
            actual_bytes = path.stat().st_size
            if recorded_bytes is not None:
                try:
                    expected_bytes = int(recorded_bytes)
                except (TypeError, ValueError) as error:
                    raise ReleaseManifestError(
                        f"{label} has invalid recorded size"
                    ) from error
                if expected_bytes != actual_bytes:
                    raise ReleaseManifestError(
                        f"{label} file size does not match the report"
                    )
            key = (_display_path(path, root), sha256_value)
            if key in records:
                continue
            records[key] = {
                "path": key[0],
                "bytes": actual_bytes,
                "sha256": sha256_value,
                "sha256_source": "recorded compact report; not recomputed for large opacity tables",
                "source": record.get("source", "petitRADTRANS input-data release"),
                "source_url": record.get("source_url", _PYRT_SOURCE_URL),
                "licence": record.get("licence", record.get("license", _NO_LICENSE)),
                "licence_status": (
                    "recorded" if record.get("licence", record.get("license")) else "not_recorded"
                ),
            }
    if not records:
        raise ReleaseManifestError("no opacity table records were found")
    return sorted(records.values(), key=lambda record: record["path"])


def _plot_record(
    *,
    report_role: str,
    report_record: Mapping[str, Any],
    root: Path,
    source_script: str,
    output_value: Any,
    input_value: Any,
    input_label: str,
) -> dict[str, Any]:
    source_path = (root / source_script).resolve()
    if not source_path.is_file():
        raise ReleaseManifestError(f"plot source script is missing: {source_path}")
    input_artifact = _artifact(
        input_value,
        label=input_label,
        root=root,
    )
    output_artifact = _artifact(
        output_value,
        label=f"{report_role} plot",
        root=root,
    )
    return {
        "source_script": _display_path(source_path, root),
        "command": report_record["command"],
        "input_data_path": input_artifact["path"],
        "input_data_sha256": input_artifact["sha256"],
        "input_report_path": report_record["path"],
        "input_report_sha256": report_record["sha256"],
        "output_path": output_artifact["path"],
        "output_sha256": output_artifact["sha256"],
    }


def _artifact_records(
    payloads: Mapping[str, Mapping[str, Any]],
    report_records: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _mapping(payloads["base"].get("artifacts"), "base.artifacts")
    base_spectrum = _artifact(
        base.get("spectrum_npz"), label="base spectrum", root=root
    )
    base_plot = _plot_record(
        report_role="base",
        report_record=report_records["base"],
        root=root,
        source_script=_K_BAND_SOURCE,
        output_value=base.get("figure_png"),
        input_value=base.get("spectrum_npz"),
        input_label="base spectrum",
    )
    science_inputs = _mapping(
        payloads["science"].get("oracle_inputs"), "science.oracle_inputs"
    )
    science_artifacts = {
        name: _artifact(
            _mapping(record, f"science.oracle_inputs.{name}").get("path"),
            label=f"science oracle {name}",
            root=root,
            recorded_bytes=_mapping(record, f"science.oracle_inputs.{name}").get("bytes"),
            recorded_sha256=_mapping(record, f"science.oracle_inputs.{name}").get("sha256"),
        )
        for name, record in science_inputs.items()
    }
    hminus = payloads["hminus"]
    oracle = _mapping(hminus.get("oracle"), "hminus.oracle")
    hminus_artifacts = {
        name: _artifact(oracle.get(name), label=f"hminus {name}", root=root)
        for name in ("lrs_archive_path", "hrs_native_archive_path", "oracle_json_path")
    }
    hminus_plot = _plot_record(
        report_role="hminus",
        report_record=report_records["hminus"],
        root=root,
        source_script=_HMINUS_SOURCE,
        output_value=hminus.get("plot"),
        input_value=oracle.get("lrs_archive_path"),
        input_label="hminus LRS archive",
    )
    return (
        {
            "base_spectrum": base_spectrum,
            "science_oracle_outputs": science_artifacts,
            "hminus_oracle_outputs": hminus_artifacts,
        },
        {"kband": base_plot, "hminus": hminus_plot},
    )


def _sampler_record(payload: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    sampler = payload.get("pymultinest")
    if not isinstance(sampler, Mapping):
        return None
    settings = sampler.get("settings")
    runs = sampler.get("runs")
    if not isinstance(settings, Mapping) or not isinstance(runs, list):
        raise ReleaseManifestError(f"{role} sampler record is incomplete")
    report_peak = _find_key(payload, {"measured_peak_rss_bytes", "peak_rss_bytes"})
    sampler_peak = sampler.get("peak_rss_bytes")
    aggregate_peak = sampler_peak if sampler_peak is not None else report_peak
    seed_pair = settings.get("seed_pair", settings.get("sampler_seed_pair"))
    if not isinstance(seed_pair, list):
        seed_pair = []
    compact_runs = []
    for index, run in enumerate(runs):
        record = _mapping(run, f"{role}.pymultinest.runs[{index}]")
        seed = record.get("seed", record.get("sampler_seed"))
        if seed is None and index < len(seed_pair):
            seed = seed_pair[index]
        calls = record.get(
            "likelihood_evaluations", record.get("likelihood_callback_evaluations")
        )
        wall_time = record.get("elapsed_seconds")
        peak_rss = record.get("peak_rss_bytes", aggregate_peak)
        if seed is None or calls is None or wall_time is None or peak_rss is None:
            raise ReleaseManifestError(
                f"{role} sampler run {index + 1} lacks seed, calls, wall time, or RSS"
            )
        compact_runs.append(
            {
                "seed": seed,
                "likelihood_calls": calls,
                "wall_time_seconds": wall_time,
                "peak_rss_bytes": peak_rss,
                "peak_rss_source": (
                    "run"
                    if record.get("peak_rss_bytes") is not None
                    else "sampler_or_report_aggregate"
                ),
                "status": record.get("status"),
                "converged": record.get("converged"),
            }
        )
    return {
        "backend": "PyMultiNest/MultiNest",
        "status": sampler.get("status"),
        "scope": sampler.get("scope"),
        "settings": dict(settings),
        "elapsed_seconds": sampler.get("elapsed_seconds"),
        "peak_rss_bytes": aggregate_peak,
        "peak_rss_source": (
            "sampler" if sampler_peak is not None else "report_aggregate"
        ),
        "runs": compact_runs,
    }


def _environment_details() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for distribution in ("robert-exoplanets", "pymultinest", "mpi4py", "petitRADTRANS"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed in the ROBERT environment"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "robert_exoplanets_version": versions["robert-exoplanets"],
        "optional_package_versions": {
            key: value for key, value in versions.items() if key != "robert-exoplanets"
        },
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_available": os.cpu_count(),
    }


def _git_details(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseManifestError(f"cannot collect Git provenance: {error}") from error
    if not commit:
        raise ReleaseManifestError("Git commit is empty")
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_path_count": len(status),
        "dirty_paths": [line[3:] if len(line) >= 3 else line for line in status],
    }


def _signed_evidence_difference(paired: Mapping[str, Any], label: str) -> float:
    """Return signed stride-2 minus stride-1 evidence for one seed pair."""

    value = paired.get("log_evidence_stride_2_minus_stride_1")
    if value is None:
        # Keep compatibility with compact reports written before the field was
        # renamed. Both fields have the same signed convention.
        value = paired.get("evidence_difference_stride_2_minus_stride_1")
    if value is None:
        stride_1 = _mapping(paired.get("stride_1"), f"{label}.stride_1")
        stride_2 = _mapping(paired.get("stride_2"), f"{label}.stride_2")
        value = float(stride_2["log_evidence"]) - float(stride_1["log_evidence"])
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ReleaseManifestError(f"{label} has an invalid signed evidence difference") from error


def _summary_evidence(paired_runs: list[Mapping[str, Any]]) -> list[float]:
    """Return one signed evidence difference for every paired seed."""

    return [
        _signed_evidence_difference(paired, f"full paired stride run {index + 1}")
        for index, paired in enumerate(paired_runs)
    ]


def _summary_values(full: Mapping[str, Any]) -> tuple[dict[str, float], list[Mapping[str, Any]], float]:
    truth = _mapping(full.get("truth"), "full.truth")
    parameters = {
        name: float(truth[name])
        for name in (
            "temperature_K",
            "log10_h2o_vmr",
            "log10_co_vmr",
            "radial_velocity_km_s",
        )
        if name in truth
    }
    if len(parameters) != 4:
        raise ReleaseManifestError("full report does not contain four truth parameters")
    comparison = _mapping(full.get("native_stride_comparison"), "full stride comparison")
    nested = _mapping(comparison.get("nested_posterior_comparison"), "full nested stride comparison")
    paired_runs = nested.get("paired_runs")
    if not isinstance(paired_runs, list) or not paired_runs:
        raise ReleaseManifestError("full report has no paired stride runs for the summary plot")
    tolerance = nested.get("evidence_tolerance")
    if not isinstance(tolerance, (int, float)):
        tolerance = 0.0
    return parameters, [_mapping(run, "full paired stride run") for run in paired_runs], float(tolerance)


def _atomic_summary_plot(
    output: Path,
    *,
    full: Mapping[str, Any],
    hminus: Mapping[str, Any],
    science: Mapping[str, Any],
) -> None:
    parameters, paired_runs, tolerance = _summary_values(full)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), squeeze=False)
    axes_flat = axes.ravel()
    colors = {"stride_1": "mediumpurple", "stride_2": "rebeccapurple"}
    for axis, (parameter, truth) in zip(axes_flat, parameters.items()):
        axis.axhline(truth, color="black", linewidth=1.0, label="truth")
        for index, paired in enumerate(paired_runs):
            for stride in ("stride_1", "stride_2"):
                stride_record = _mapping(paired.get(stride), f"summary.{stride}")
                medians = _mapping(stride_record.get("posterior_medians"), "summary medians")
                if parameter not in medians:
                    raise ReleaseManifestError(
                        f"summary plot is missing {parameter} for {stride}"
                    )
                offset = -0.12 if stride == "stride_1" else 0.12
                axis.scatter(
                    index + offset,
                    float(medians[parameter]),
                    color=colors[stride],
                    s=28,
                    label=stride.replace("_", " ") if index == 0 else None,
                )
        axis.set_title(parameter)
        axis.set_xticks(range(len(paired_runs)))
        axis.set_xticklabels([str(run.get("seed", index)) for index, run in enumerate(paired_runs)])
        axis.set_xlabel("seed")
        axis.grid(alpha=0.2)
    evidence = _summary_evidence(paired_runs)
    status_text = (
        f"science={science.get('status', 'unknown')}  "
        f"H-={hminus.get('status', 'unknown')}  "
        f"full={full.get('status', 'unknown')}  "
        f"ΔlnZ stride2−stride1={', '.join(f'{value:.3g}' for value in evidence)}; "
        f"limit={tolerance:.3g}"
    )
    figure.suptitle("ROBERT LBL/HRS stride summary")
    figure.text(0.5, 0.01, status_text, ha="center", fontsize=8)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper right", fontsize=8)
    figure.tight_layout(rect=(0.0, 0.04, 0.96, 0.96))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".png", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, dpi=140, format="png")
        temporary.replace(output)
    finally:
        plt.close(figure)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _summary_plot_record(
    output_value: Any,
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    report_records: Mapping[str, Mapping[str, Any]],
    root: Path,
    command: str,
) -> dict[str, Any]:
    output = _resolve_path(output_value, root)
    _atomic_summary_plot(
        output,
        full=payloads["full"],
        hminus=payloads["hminus"],
        science=payloads["science"],
    )
    artifact = _artifact(output, label="summary plot", root=root)
    source = Path(__file__).resolve()
    return {
        "source_script": _display_path(source, root),
        "command": command,
        "input_report_sha256": {
            role: report_records[role]["sha256"]
            for role in ("full", "hminus", "science")
        },
        "output_path": artifact["path"],
        "output_sha256": artifact["sha256"],
    }


def _command_line() -> str:
    return " ".join(shlex.quote(argument) for argument in sys.argv)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_release_manifest(
    *,
    report_dir: Path = ROOT / _DEFAULT_REPORT_DIR_NAME,
    output_path: Path = ROOT / _DEFAULT_OUTPUT_NAME,
    root: Path = ROOT,
    summary_plot: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Validate compact reports and write one atomic release manifest."""

    report_dir = report_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    root = root.expanduser().resolve()
    report_records, payloads = _report_records(report_dir=report_dir, root=root)
    native_validation = {
        "base": _validate_native_report("base", payloads["base"]),
        "science": _validate_native_report("science", payloads["science"]),
    }
    hminus_validation = _validate_hminus(payloads["hminus"])
    full_validation = _validate_full(payloads["full"])
    stride = _stride_policy(payloads["full"])
    _validate_full_outer_status(payloads["full"], stride)
    resources = _resource_manifest(payloads)
    opacity = _table_records(payloads, root=root)
    artifacts, plots = _artifact_records(payloads, report_records, root=root)
    if summary_plot is not None:
        summary_command = command or _command_line()
        plots["summary"] = _summary_plot_record(
            summary_plot,
            payloads=payloads,
            report_records=report_records,
            root=root,
            command=summary_command,
        )
    samplers = {
        role: sampler
        for role, payload in payloads.items()
        if (sampler := _sampler_record(payload, role)) is not None
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_name": "lbl_hrs_release_manifest_20260831",
        "status": (
            "complete_with_rejected_stride_2_candidate"
            if stride.get("stride_2_rejection", {}).get("completed_negative_result")
            else "complete"
        ),
        "git": _git_details(root),
        "environment": _environment_details(),
        "reports": report_records,
        "report_hashes": {
            role: report_records[role]["sha256"]
            for role in ("hminus", "base", "science", "full")
        },
        "resources": resources,
        "abundance_contract": {
            "robert_state": "VMR only",
            "retrieval_fields": "VMR or log10(VMR)",
            "mass_fractions_in_robert_state": False,
            "pRT_boundary_conversion": (
                "private deterministic conversion from the same ROBERT VMR "
                "state for the external pRT oracle"
            ),
        },
        "opacity_inputs": opacity,
        "samplers": samplers,
        "artifacts": artifacts,
        "plots": plots,
        "validation": {
            "native": native_validation,
            "hminus": hminus_validation,
            "full_four_parameter_pymultinest": full_validation,
            "pending_token_absent": True,
            "strict_process_memory_under_2_gib": resources[
                "strict_process_memory_under_2_gib"
            ],
            "numerical_threads_at_most_3": resources["threads_at_most_3"],
            "mpi_processes_at_most_3": resources["mpi_processes_at_most_3"],
            "mpi_processes": resources["mpi_processes"],
        },
        "stride_policy": stride,
        "provenance_contract": {
            "sampler_backend": "PyMultiNest/MultiNest",
            "plot_inputs_are_compact_reports_or_small_artifacts": True,
            "sampler_chain_files_read": False,
            "external_tables_hashed_from_report": True,
        },
    }
    _atomic_write_json(output_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / _DEFAULT_REPORT_DIR_NAME)
    parser.add_argument("--output", type=Path, default=ROOT / _DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--summary-plot",
        type=Path,
        default=None,
        help="Optional bounded PNG summary of stride medians and evidence differences.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    try:
        manifest = build_release_manifest(
            report_dir=arguments.report_dir,
            output_path=arguments.output,
            root=ROOT,
            summary_plot=arguments.summary_plot,
        )
    except ReleaseManifestError as error:
        raise SystemExit(f"release manifest not written: {error}") from error
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
