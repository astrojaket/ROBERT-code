"""Compare matched ROBERT H-minus on/off PyMultiNest reports.

This is a report-only diagnostic.  It does not run a sampler, read sampler
chains, load an opacity table, or load a spectrum.  It checks that the two
reports describe the same WASP-77Ab observations and numerical experiment
before it subtracts their log evidences.

The reported difference is ``log Z(H-minus on) - log Z(H-minus off)``.  It is
evidence for the explicit ROBERT H-minus state only.  It is not a claim that
bound-free opacity is detected in the K band: the John bound-free term is zero
above 1.6421 micron, while the free-free term depends on the H, e-minus, and
H-minus VMR state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
# The physical ROBERT WASP-77Ab sampler uses the newer six-thread test
# allowance.  Older deterministic/reference reports are checked by the
# validation finalizer, not by this on/off comparison.
MAX_THREADS = 6
GIB = 1024**3
PROCESS_MEMORY_HARD_LIMIT_BYTES = 2 * GIB
EXPECTED_SEEDS = (24680, 24681)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ON_REPORT = ROOT / "docs/data/wasp77ab_robert_joint_hminus_on_20260831.json"
DEFAULT_OFF_REPORT = ROOT / "docs/data/wasp77ab_robert_joint_hminus_off_20260831.json"
DEFAULT_SOURCE_MANIFEST = ROOT / "docs/data/wasp77ab_smith2024_sources.json"
DEFAULT_OUTPUT = ROOT / "docs/data/wasp77ab_robert_hminus_evidence_20260831.json"


def _clamp_thread_environment() -> None:
    """Keep this report reader inside the six-thread ROBERT policy."""

    for name in THREAD_VARIABLES:
        raw = os.environ.get(name, "1")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 1
        os.environ[name] = str(min(MAX_THREADS, max(1, value)))


_clamp_thread_environment()


class HMinusComparisonError(RuntimeError):
    """Raised when the on/off reports are not a matched experiment."""


WASP77AbHMinusComparisonError = HMinusComparisonError


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HMinusComparisonError(f"{label} must be a JSON object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HMinusComparisonError(f"{label} is missing")
    return value.strip()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HMinusComparisonError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HMinusComparisonError(f"cannot read {label}: {error}") from error
    return dict(_mapping(payload, label))


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise HMinusComparisonError(f"identity value is not JSON serialisable: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HMinusComparisonError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HMinusComparisonError(f"{label} must be finite")
    return result


def _source_hashes(report: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    data = _mapping(report.get("data"), f"{label}.data")
    raw = data.get("source_hashes", report.get("source_hashes"))
    source = _mapping(raw, f"{label}.source_hashes")
    if not source:
        raise HMinusComparisonError(f"{label}.source_hashes is empty")
    result: dict[str, dict[str, Any]] = {}
    for name, raw_record in source.items():
        record = _mapping(raw_record, f"{label}.source_hashes.{name}")
        digest = record.get("sha256", record.get("expected_sha256"))
        digest_text = _text(digest, f"{label}.source_hashes.{name}.sha256").casefold()
        if len(digest_text) != 64 or any(character not in "0123456789abcdef" for character in digest_text):
            raise HMinusComparisonError(f"{label}.source_hashes.{name} has an invalid SHA-256")
        if "verified" in record and record["verified"] is not True:
            raise HMinusComparisonError(f"{label}.source_hashes.{name} is not verified")
        normalized: dict[str, Any] = {
            "name": str(record.get("name", name)),
            "sha256": digest_text,
        }
        size = record.get("size_bytes", record.get("expected_size_bytes"))
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise HMinusComparisonError(f"{label}.source_hashes.{name} has an invalid size")
            normalized["size_bytes"] = size
        md5 = record.get("md5", record.get("expected_md5"))
        if md5 is not None:
            md5_text = _text(md5, f"{label}.source_hashes.{name}.md5").casefold()
            if len(md5_text) != 32 or any(character not in "0123456789abcdef" for character in md5_text):
                raise HMinusComparisonError(f"{label}.source_hashes.{name} has an invalid MD5")
            normalized["md5"] = md5_text
        result[str(name)] = normalized
    return dict(sorted(result.items()))


def _active_ultranest_reference(value: Any, path: str = "$", key: str | None = None) -> str | None:
    """Return a path for an active UltraNest reference, if one exists."""

    if isinstance(value, str) and "ultranest" in value.casefold():
        if key != "ultranest_allowed":
            return path
        return None
    if isinstance(value, Mapping):
        for nested_key, nested in value.items():
            nested_key_text = str(nested_key)
            nested_path = f"{path}.{nested_key_text}"
            if nested_key_text.casefold() == "ultranest_allowed" and nested is False:
                continue
            if "ultranest" in nested_key_text.casefold() and nested is not False:
                return nested_path
            found = _active_ultranest_reference(nested, nested_path, nested_key_text)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _active_ultranest_reference(nested, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _check_source_manifest(
    report_sources: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Cross-check overlapping source hashes without reading external files."""

    if source_manifest is None:
        return {"checked": False, "overlap_count": 0}
    status = source_manifest.get("status")
    if not isinstance(status, str) or not status.startswith("pass"):
        raise HMinusComparisonError("source manifest does not have a passing status")
    files = source_manifest.get("files")
    if not isinstance(files, list):
        raise HMinusComparisonError("source manifest.files is missing")
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(files):
        record = _mapping(raw, f"source manifest.files[{index}]")
        name = record.get("name")
        if isinstance(name, str):
            by_name[name] = record
    overlap = 0
    for name, report_record in report_sources.items():
        source_name = str(report_record.get("name", name))
        source_record = by_name.get(source_name) or by_name.get(name)
        if source_record is None:
            continue
        expected = source_record.get("sha256_expected", source_record.get("sha256"))
        if expected is None or str(expected).casefold() != report_record["sha256"]:
            raise HMinusComparisonError(f"source manifest hash does not match {name}")
        overlap += 1
    return {"checked": True, "overlap_count": overlap}


def _sampler_policy(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    policy = _mapping(report.get("sampler_policy"), f"{label}.sampler_policy")
    sampler = str(policy.get("sampler", "")).casefold()
    backend = str(policy.get("backend", "")).casefold()
    if sampler != "pymultinest" or backend != "multinest":
        raise HMinusComparisonError(f"{label} is not a PyMultiNest/MultiNest report")
    if policy.get("max_iter") != 0:
        raise HMinusComparisonError(f"{label} must use max_iter=0")
    seeds = policy.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 2 or tuple(seeds) != EXPECTED_SEEDS:
        raise HMinusComparisonError(f"{label} must use seeds {EXPECTED_SEEDS}")
    n_live = policy.get("n_live_points")
    if isinstance(n_live, bool) or not isinstance(n_live, int) or n_live <= 0:
        raise HMinusComparisonError(f"{label}.n_live_points is invalid")
    efficiency = _finite_number(policy.get("sampling_efficiency"), f"{label}.sampling_efficiency")
    if not 0.0 < efficiency <= 1.0:
        raise HMinusComparisonError(f"{label}.sampling_efficiency is outside (0, 1]")
    tolerance = _finite_number(policy.get("evidence_tolerance"), f"{label}.evidence_tolerance")
    if tolerance < 0.0:
        raise HMinusComparisonError(f"{label}.evidence_tolerance must be non-negative")
    if policy.get("mpi_processes") != 1:
        raise HMinusComparisonError(f"{label} must use one MPI process")
    if policy.get("ultranest_allowed") is not False:
        raise HMinusComparisonError(f"{label} must explicitly disallow UltraNest")
    return {
        "sampler": "PyMultiNest",
        "backend": "MultiNest",
        "seeds": list(EXPECTED_SEEDS),
        "n_live_points": n_live,
        "max_iter": 0,
        "sampling_efficiency": efficiency,
        "evidence_tolerance": tolerance,
        "mpi_processes": 1,
    }


def _thread_values(report: Mapping[str, Any], label: str) -> dict[str, int]:
    resources = _mapping(report.get("resources"), f"{label}.resources")
    raw = resources.get("thread_values")
    if not isinstance(raw, Mapping):
        raise HMinusComparisonError(f"{label}.resources.thread_values is missing")
    result: dict[str, int] = {}
    for name in THREAD_VARIABLES:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_THREADS:
            raise HMinusComparisonError(f"{label} thread value {name} is outside 1..{MAX_THREADS}")
        result[name] = value
    return result


def _resource_contract(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    resources = _mapping(report.get("resources"), f"{label}.resources")
    if resources.get("gate_passed") is False:
        raise HMinusComparisonError(f"{label} resource gate failed")
    threads = _thread_values(report, label)
    observed: list[int] = []
    for key in ("peak_rss_bytes", "measured_peak_rss_bytes", "process_peak_rss_bytes"):
        value = resources.get(key)
        if value is not None:
            numeric = _finite_number(value, f"{label}.resources.{key}")
            if numeric <= 0 or numeric >= PROCESS_MEMORY_HARD_LIMIT_BYTES:
                raise HMinusComparisonError(f"{label} peak RSS is not below 2 GiB")
            observed.append(int(numeric))
    if not observed:
        raise HMinusComparisonError(f"{label} has no measured peak RSS")
    configured: list[int] = []
    for key in (
        "configured_process_rss_limit_bytes",
        "process_rss_limit_bytes",
        "process_memory_limit_bytes",
        "process_rss_hard_limit_bytes",
    ):
        value = resources.get(key)
        if value is not None:
            numeric = _finite_number(value, f"{label}.resources.{key}")
            if numeric <= 0 or numeric > PROCESS_MEMORY_HARD_LIMIT_BYTES:
                raise HMinusComparisonError(f"{label} has an invalid process memory limit")
            configured.append(int(numeric))
    return {
        "thread_values": threads,
        "thread_limit": MAX_THREADS,
        "observed_peak_rss_bytes": observed,
        "configured_process_memory_bytes": configured,
        "strict_process_memory_under_2_gib": True,
        "threads_at_most_6": True,
    }


def _identity(report: Mapping[str, Any], label: str, sources: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _mapping(report.get("data"), f"{label}.data")
    explicit_observation = report.get("observation_identity", data.get("observation_identity"))
    if explicit_observation is not None:
        observation = {"source": "reported", "value": explicit_observation}
    else:
        required = {
            "hrs_subset": data.get("hrs_subset"),
            "lrs_detectors": data.get("lrs_detectors"),
            "hrs_order_indices": data.get("hrs_order_indices"),
            "hrs_point_count": data.get("hrs_point_count"),
            "lrs_point_count": data.get("lrs_point_count"),
        }
        if any(value is None for value in required.values()):
            raise HMinusComparisonError(
                f"{label} has no complete observation identity; add observation_identity"
            )
        observation = {"source": "derived_from_report_data", "value": required}
    noise_value = report.get("noise_identity", data.get("noise_identity"))
    if noise_value is None:
        noise_seed = report.get("noise_seed", data.get("noise_seed"))
        if noise_seed is not None:
            noise_value = {"noise_seed": noise_seed}
            noise_source = "reported_noise_seed"
        else:
            # The real Smith/August comparison has no generated noise.  The
            # source hashes and observation identity are the noise identity.
            noise_value = {
                "kind": "real_observation_without_generated_noise",
                "source_hashes": sources,
            }
            noise_source = "derived_real_observation"
    else:
        noise_source = "reported"
    noise = {"source": noise_source, "value": noise_value}
    return observation, noise


def _hminus_enabled(report: Mapping[str, Any], label: str) -> bool:
    value = report.get("hminus_enabled")
    if not isinstance(value, bool):
        policy = _mapping(report.get("hminus_policy"), f"{label}.hminus_policy")
        value = policy.get("enabled")
    if not isinstance(value, bool):
        raise HMinusComparisonError(f"{label} has no hminus_enabled state")
    policy = report.get("hminus_policy")
    if isinstance(policy, Mapping) and policy.get("enabled") is not None and policy.get("enabled") != value:
        raise HMinusComparisonError(f"{label} has inconsistent H-minus state")
    return value


def _run_records(report: Mapping[str, Any], label: str, modes: Sequence[str], policy: Mapping[str, Any]) -> dict[str, list[dict[str, float | int]]]:
    raw_runs = _mapping(report.get("runs"), f"{label}.runs")
    output: dict[str, list[dict[str, float | int]]] = {}
    for mode in modes:
        group = _mapping(raw_runs.get(mode), f"{label}.runs.{mode}")
        if group.get("gate_passed") is not True:
            raise HMinusComparisonError(f"{label} mode {mode} sampler gate failed")
        comparison = _mapping(group.get("evidence_comparison"), f"{label}.runs.{mode}.evidence_comparison")
        if comparison.get("gate_passed") is not True:
            raise HMinusComparisonError(f"{label} mode {mode} evidence gate failed")
        group_seeds = comparison.get("seed_pair")
        if group_seeds is not None and group_seeds != list(policy["seeds"]):
            raise HMinusComparisonError(f"{label} mode {mode} has a mismatched seed pair")
        raw = group.get("runs")
        if not isinstance(raw, list) or len(raw) != 2:
            raise HMinusComparisonError(f"{label} mode {mode} does not have two runs")
        records: list[dict[str, float | int]] = []
        for index, item in enumerate(raw):
            run = _mapping(item, f"{label}.runs.{mode}[{index}]")
            if run.get("seed") != policy["seeds"][index]:
                raise HMinusComparisonError(f"{label} mode {mode} has the wrong seed")
            if run.get("status") not in {"pass", "completed"} or run.get("converged") is not True:
                raise HMinusComparisonError(f"{label} mode {mode} seed {run.get('seed')} did not converge")
            if run.get("resource_gate_passed") is not True:
                raise HMinusComparisonError(f"{label} mode {mode} seed {run.get('seed')} failed resource gate")
            evidence = _finite_number(run.get("log_evidence"), f"{label}.{mode}.log_evidence")
            error = _finite_number(run.get("log_evidence_error"), f"{label}.{mode}.log_evidence_error")
            if error < 0.0:
                raise HMinusComparisonError(f"{label}.{mode}.log_evidence_error is negative")
            peak = run.get("peak_rss_bytes")
            if peak is None:
                raise HMinusComparisonError(f"{label}.{mode} seed {run.get('seed')} has no peak RSS")
            peak_value = _finite_number(peak, f"{label}.{mode}.peak_rss_bytes")
            if peak_value <= 0.0 or peak_value >= PROCESS_MEMORY_HARD_LIMIT_BYTES:
                raise HMinusComparisonError(f"{label}.{mode} seed {run.get('seed')} exceeds 2 GiB RSS")
            records.append(
                {
                    "seed": int(run["seed"]),
                    "log_evidence": evidence,
                    "log_evidence_error": error,
                }
            )
        reported_delta = comparison.get("delta_log_evidence")
        if reported_delta is not None:
            reported = abs(_finite_number(reported_delta, f"{label}.{mode}.delta_log_evidence"))
            actual = abs(records[0]["log_evidence"] - records[1]["log_evidence"])
            if not math.isclose(reported, actual, rel_tol=1.0e-8, abs_tol=1.0e-8):
                raise HMinusComparisonError(f"{label} mode {mode} reported evidence delta is inconsistent")
        output[mode] = records
    return output


def _compare_controls(on: Mapping[str, Any], off: Mapping[str, Any]) -> dict[str, Any]:
    on_policy = _sampler_policy(on, "on report")
    off_policy = _sampler_policy(off, "off report")
    if _canonical(on_policy) != _canonical(off_policy):
        raise HMinusComparisonError("sampler controls are not matched")
    on_threads = _thread_values(on, "on report")
    off_threads = _thread_values(off, "off report")
    if on_threads != off_threads:
        raise HMinusComparisonError("thread settings are not matched")
    on_resources = _resource_contract(on, "on report")
    off_resources = _resource_contract(off, "off report")
    return {
        "sampler_policy": on_policy,
        "thread_values": on_threads,
        "resource_policy": {
            "on": on_resources,
            "off": off_resources,
        },
    }


def _evidence_difference(
    on_runs: Mapping[str, Sequence[Mapping[str, float | int]]],
    off_runs: Mapping[str, Sequence[Mapping[str, float | int]]],
    modes: Sequence[str],
) -> dict[str, Any]:
    per_mode: dict[str, Any] = {}
    combined_per_seed: list[dict[str, Any]] = []
    for mode in modes:
        entries: list[dict[str, Any]] = []
        for index, (on_run, off_run) in enumerate(zip(on_runs[mode], off_runs[mode])):
            on_evidence = float(on_run["log_evidence"])
            off_evidence = float(off_run["log_evidence"])
            on_error = float(on_run["log_evidence_error"])
            off_error = float(off_run["log_evidence_error"])
            delta = on_evidence - off_evidence
            propagated = math.hypot(on_error, off_error)
            entries.append(
                {
                    "seed": int(on_run["seed"]),
                    "log_evidence_on": on_evidence,
                    "log_evidence_off": off_evidence,
                    "delta_log_evidence_on_minus_off": delta,
                    "propagated_error": propagated,
                    "within_propagated_error": abs(delta) <= propagated,
                }
            )
        combined_delta = sum(float(entry["delta_log_evidence_on_minus_off"]) for entry in entries)
        combined_error = math.hypot(*(float(entry["propagated_error"]) for entry in entries))
        per_mode[mode] = {
            "per_seed": entries,
            "combined": {
                "delta_log_evidence_on_minus_off": combined_delta,
                "propagated_error": combined_error,
                "within_propagated_error": abs(combined_delta) <= combined_error,
            },
        }
        for index, entry in enumerate(entries):
            while len(combined_per_seed) <= index:
                combined_per_seed.append(
                    {
                        "seed": int(entry["seed"]),
                        "delta_log_evidence_on_minus_off": 0.0,
                        "propagated_error_terms": [],
                    }
                )
            combined_per_seed[index]["delta_log_evidence_on_minus_off"] += float(
                entry["delta_log_evidence_on_minus_off"]
            )
            combined_per_seed[index]["propagated_error_terms"].append(
                float(entry["propagated_error"])
            )
    for entry in combined_per_seed:
        terms = entry.pop("propagated_error_terms")
        entry["propagated_error"] = math.hypot(*terms)
        entry["within_propagated_error"] = abs(entry["delta_log_evidence_on_minus_off"]) <= entry[
            "propagated_error"
        ]
    total_delta = sum(float(entry["delta_log_evidence_on_minus_off"]) for entry in combined_per_seed)
    total_error = math.hypot(*(float(entry["propagated_error"]) for entry in combined_per_seed))
    return {
        "direction": "H-minus on minus H-minus off",
        "per_mode": per_mode,
        "combined": {
            "per_seed": combined_per_seed,
            "delta_log_evidence_on_minus_off": total_delta,
            "propagated_error": total_error,
            "within_propagated_error": abs(total_delta) <= total_error,
        },
    }


def compare_reports(
    on_report: Mapping[str, Any],
    off_report: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate two complete reports and return their compact comparison."""

    on = _mapping(on_report, "on report")
    off = _mapping(off_report, "off report")
    if on.get("status") != "pass" or off.get("status") != "pass":
        raise HMinusComparisonError("both reports must have status pass")
    for label, report in (("on report", on), ("off report", off)):
        active_reference = _active_ultranest_reference(report)
        if active_reference is not None:
            raise HMinusComparisonError(f"{label} contains an active UltraNest reference")
    on_enabled = _hminus_enabled(on, "on report")
    off_enabled = _hminus_enabled(off, "off report")
    if on_enabled is not True or off_enabled is not False:
        raise HMinusComparisonError("reports must be one H-minus-on and one H-minus-off run")
    on_modes = on.get("modes")
    off_modes = off.get("modes")
    if not isinstance(on_modes, list) or not on_modes or on_modes != off_modes:
        raise HMinusComparisonError("on/off mode lists are not matched")
    modes = tuple(str(mode) for mode in on_modes)
    if len(set(modes)) != len(modes):
        raise HMinusComparisonError("mode list contains duplicates")
    on_sources = _source_hashes(on, "on report")
    off_sources = _source_hashes(off, "off report")
    if on_sources != off_sources:
        raise HMinusComparisonError("source hashes are not matched")
    source_check = _check_source_manifest(on_sources, source_manifest)
    on_observation, on_noise = _identity(on, "on report", on_sources)
    off_observation, off_noise = _identity(off, "off report", off_sources)
    if _canonical(on_observation) != _canonical(off_observation):
        raise HMinusComparisonError("observation identities are not matched")
    if _canonical(on_noise) != _canonical(off_noise):
        raise HMinusComparisonError("noise identities are not matched")
    on_stride = _mapping(on.get("stride_selection"), "on report.stride_selection")
    off_stride = _mapping(off.get("stride_selection"), "off report.stride_selection")
    if _canonical(on_stride) != _canonical(off_stride):
        raise HMinusComparisonError("stride selections are not matched")
    if on_stride.get("hrs") != 2 or on_stride.get("lrs") != 25:
        raise HMinusComparisonError("reports must use measured HRS stride 2 and LRS stride 25")
    if on_stride.get("release_claim") is not True:
        raise HMinusComparisonError("stride selection is not a measured release selection")
    controls = _compare_controls(on, off)
    on_runs = _run_records(on, "on report", modes, controls["sampler_policy"])
    off_runs = _run_records(off, "off report", modes, controls["sampler_policy"])
    evidence = _evidence_difference(on_runs, off_runs, modes)
    composition = {
        "robert_state": "VMR only",
        "mass_fraction_parameters_in_robert": False,
        "same_parameterisation_required": True,
    }
    for label, report in (("on report", on), ("off report", off)):
        policy = _mapping(report.get("composition_policy"), f"{label}.composition_policy")
        if policy.get("mass_fraction_parameters_in_robert") is not False or policy.get("mass_fraction_parameters") is not False:
            raise HMinusComparisonError(f"{label} contains ROBERT mass-fraction parameters")
        if policy.get("robert_convention") != "volume_mixing_ratio":
            raise HMinusComparisonError(f"{label} is not VMR based")
    hminus_notes = {
        "evidence_scope": "evidence for the explicit ROBERT H-minus state only",
        "bound_free": {
            "k_band_status": "zero",
            "cutoff_micron": 1.6421,
            "reason": "the selected 1.90-2.45 micron K band is beyond the bound-free cutoff",
        },
        "free_free": {
            "controls": "depends on H, e-, and H- VMR profiles",
            "retrieval_limit": "the K-band comparison does not isolate the individual continuum VMRs",
        },
        "hminus_vmr_identifiability": "prior-dominated in the selected K-band window",
        "not_a_molecular_detection_claim": True,
    }
    return {
        "schema_version": 1,
        "record_type": "wasp77ab_robert_hminus_on_off_evidence_comparison",
        "status": "pass",
        "target": "WASP-77Ab",
        "reports": {"on": "in-memory", "off": "in-memory"},
        "modes": list(modes),
        "source_hashes": on_sources,
        "source_manifest": source_check,
        "observation_identity": on_observation,
        "noise_identity": on_noise,
        "stride_selection": {"hrs": 2, "lrs": 25, "matched": True},
        "sampler_contract": controls,
        "composition": composition,
        "hminus_identifiability": hminus_notes,
        "evidence_difference": evidence,
        "interpretation": "Evidence is for the explicit ROBERT H-minus state only; it is not a claim of K-band bound-free detection.",
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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


def compare_report_files(
    on_path: Path = DEFAULT_ON_REPORT,
    off_path: Path = DEFAULT_OFF_REPORT,
    *,
    output_path: Path | None = DEFAULT_OUTPUT,
    source_manifest_path: Path | None = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, Any]:
    """Read compact reports, compare them, and optionally write JSON."""

    on_file = Path(on_path).expanduser().resolve()
    off_file = Path(off_path).expanduser().resolve()
    source_manifest = None
    if source_manifest_path is not None:
        source_file = Path(source_manifest_path).expanduser().resolve()
        if source_file.is_file():
            source_manifest = _read_json(source_file, "source manifest")
        else:
            raise HMinusComparisonError(f"source manifest is missing: {source_file}")
    report = compare_reports(
        _read_json(on_file, "H-minus-on report"),
        _read_json(off_file, "H-minus-off report"),
        source_manifest=source_manifest,
    )
    report["reports"] = {
        "on": str(on_file),
        "off": str(off_file),
        "on_sha256": _sha256(on_file),
        "off_sha256": _sha256(off_file),
    }
    report["command"] = " ".join(json.dumps(argument) for argument in sys.argv)
    report["environment"] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "thread_values": {name: int(os.environ[name]) for name in THREAD_VARIABLES},
    }
    if output_path is not None:
        _atomic_write(Path(output_path).expanduser().resolve(), report)
    return report


# Descriptive aliases for callers that prefer a build-style name.
build_comparison_report = compare_report_files
compare_hminus_evidence = compare_reports


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-report", type=Path, default=DEFAULT_ON_REPORT)
    parser.add_argument("--off-report", type=Path, default=DEFAULT_OFF_REPORT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-source-manifest",
        action="store_true",
        help="skip the optional source-manifest cross-check",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = compare_report_files(
            arguments.on_report,
            arguments.off_report,
            output_path=arguments.output,
            source_manifest_path=(None if arguments.no_source_manifest else arguments.source_manifest),
        )
    except HMinusComparisonError as error:
        raise SystemExit(f"H-minus evidence comparison not written: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
