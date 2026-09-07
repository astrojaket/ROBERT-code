"""Check the compact WASP-77Ab validation evidence.

The finalizer reads JSON reports and small plot files.  It does not load an
IGRINS cube, an opacity table, a sampler chain, or a spectrum.  It writes one
small manifest only when :func:`build_validation_manifest` or the command line
entry point is called.

The five required reports are the source manifest, deterministic LRS and HRS
checks, the measured LBL stride report, and the real-data ROBERT operator
injection check.  The fixed-template PyMultiNest report and the shared-VMR
ROBERT sampler report are optional while the workflow is being developed.
Missing or preflight-only sampler evidence is reported as ``pending``.
``--require-samplers`` turns that state into a release error.  The
deterministic/reference reports retain their historical 1--3 thread contract.
The real shared-VMR ROBERT reports use the newer 1--6 thread contract.
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
import shlex
import struct
import subprocess
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
LEGACY_MAX_THREADS = 3
ROBERT_MAX_THREADS = 6
# Keep the public name aligned with the active real ROBERT sampler contract.
MAX_THREADS = ROBERT_MAX_THREADS
GIB = 1024**3
PROCESS_MEMORY_HARD_LIMIT_BYTES = 2 * GIB
OPACITY_MEMORY_LIMIT_BYTES = 1023 * 1024**2
MAX_LIGHTWEIGHT_ARTIFACT_BYTES = 64 * 1024**2
CONFIGURED_PROCESS_MEMORY_LIMIT_BYTES = int(1.9 * GIB)
EXPECTED_INJECTION_RECORD_TYPE = "wasp77ab_robert_operator_injection_validation"
EXPECTED_HRS_ORDER_COUNT = 19

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "docs" / "data"
DEFAULT_OUTPUT = DEFAULT_REPORT_DIR / "wasp77ab_validation_manifest_20260831.json"

REPORT_NAMES = {
    "source": "wasp77ab_smith2024_sources.json",
    "lrs": "wasp77ab_smith2024_lrs_validation_20260831.json",
    "hrs": "wasp77ab_smith2024_hrs_validation_20260831.json",
    "stride": "wasp77ab_lbl_sampling_20260831.json",
    "injection": "wasp77ab_robert_injection_20260831.json",
    "fixed_template_sampler": "wasp77ab_smith2024_reference_pymultinest.json",
    "robert_joint_sampler": "wasp77ab_robert_joint_20260831.json",
}
REQUIRED_REPORT_ROLES = ("source", "lrs", "hrs", "stride", "injection")
OPTIONAL_SAMPLER_ROLES = ("fixed_template_sampler", "robert_joint_sampler")
EXPECTED_SEEDS = (24680, 24681)


def _clamp_thread_environment() -> None:
    """Keep this small reader inside the repository CPU policy."""

    for name in THREAD_VARIABLES:
        raw = os.environ.get(name, "1")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 1
        os.environ[name] = str(min(MAX_THREADS, max(1, value)))


_clamp_thread_environment()


class ValidationFinalizerError(RuntimeError):
    """Raised when compact validation evidence breaks its contract."""


# A short compatibility alias makes the exception easy to discover in example
# code without introducing a second implementation.
WASP77AbValidationError = ValidationFinalizerError


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationFinalizerError(f"{label} must be a JSON object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationFinalizerError(f"{label} is missing")
    return value.strip()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationFinalizerError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationFinalizerError(f"cannot read {label}: {error}") from error
    return dict(_mapping(payload, label))


def _sha256(path: Path) -> str:
    """Hash a small report or plot with a bounded read buffer."""

    size = path.stat().st_size
    if size > MAX_LIGHTWEIGHT_ARTIFACT_BYTES:
        raise ValidationFinalizerError(
            f"refusing to read non-lightweight artifact {path} ({size} bytes)"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, root: Path, label: str) -> Path:
    if isinstance(value, Path):
        path = value.expanduser()
    else:
        path = Path(_text(value, label)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _find_values(value: Any, names: set[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []

    def visit(child: Any, path: str) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                key_text = str(key)
                nested_path = f"{path}.{key_text}"
                if key_text in names:
                    found.append((nested_path, nested))
                visit(nested, nested_path)
        elif isinstance(child, list):
            for index, nested in enumerate(child):
                visit(nested, f"{path}[{index}]")

    visit(value, "$")
    return found


def _first_value(value: Any, names: set[str]) -> Any:
    matches = _find_values(value, names)
    return matches[0][1] if matches else None


def _scan_forbidden_sampler_text(value: Any) -> list[str]:
    """Find active UltraNest mentions while allowing an explicit false flag."""

    found: list[str] = []

    def visit(child: Any, path: str, key: str | None = None) -> None:
        if isinstance(child, str) and "ultranest" in child.casefold():
            if not (key == "ultranest_allowed" and child.casefold() == "false"):
                found.append(path)
        elif isinstance(child, Mapping):
            for nested_key, nested in child.items():
                visit(nested, f"{path}.{nested_key}", str(nested_key))
        elif isinstance(child, list):
            for index, nested in enumerate(child):
                visit(nested, f"{path}[{index}]")

    visit(value, "$")
    return found


def _status(payload: Mapping[str, Any], label: str, accepted: set[str]) -> str:
    value = _text(payload.get("status"), f"{label}.status")
    if value not in accepted:
        raise ValidationFinalizerError(
            f"{label}.status={value!r}; expected one of {sorted(accepted)}"
        )
    return value


def _all_boolean_gates(payload: Mapping[str, Any], label: str) -> dict[str, bool]:
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValidationFinalizerError(f"{label}.acceptance is missing")
    values = {str(key): value for key, value in acceptance.items() if isinstance(value, bool)}
    if not values:
        raise ValidationFinalizerError(f"{label}.acceptance has no boolean gates")
    failed = [key for key, value in values.items() if not value]
    if failed:
        raise ValidationFinalizerError(f"{label} acceptance gate failed: {failed[0]}")
    return values


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    """Return a finite JSON number, with an optional strict positivity check."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationFinalizerError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValidationFinalizerError(f"{label} must be {qualifier}")
    return result


def _required_boolean(
    payload: Mapping[str, Any], label: str, key: str, expected: bool = True
) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool) or value is not expected:
        expectation = "true" if expected else "false"
        raise ValidationFinalizerError(f"{label}.{key} must be {expectation}")
    return value


def _thread_values(
    payload: Mapping[str, Any],
    label: str,
    *,
    max_threads: int = LEGACY_MAX_THREADS,
) -> dict[str, int]:
    """Read thread values using the report's explicit CPU contract."""

    if not 1 <= max_threads <= ROBERT_MAX_THREADS:
        raise ValueError("max_threads must be between 1 and the ROBERT limit")
    value = _first_value(payload, {"thread_values"})
    if not isinstance(value, Mapping):
        scalar = _first_value(payload, {"thread_limit", "max_threads", "cpu_thread_limit"})
        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
            value = {name: scalar for name in THREAD_VARIABLES}
        else:
            raise ValidationFinalizerError(f"{label} has no thread values")
    result: dict[str, int] = {}
    for name in THREAD_VARIABLES:
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValidationFinalizerError(f"{label} is missing thread variable {name}")
        integer = int(raw)
        if integer != raw or not 1 <= integer <= max_threads:
            raise ValidationFinalizerError(
                f"{label}.{name}={raw!r} is outside the allowed range 1..{max_threads}"
            )
        result[name] = integer
    return result


def _mpi_values(
    payload: Mapping[str, Any],
    label: str,
    *,
    sampler: bool,
    max_threads: int = LEGACY_MAX_THREADS,
) -> dict[str, int]:
    names = {"mpi_processes", "mpi_nprocs", "mpi_world_size_observed", "mpi_size"}
    values = _find_values(payload, names)
    result: dict[str, int] = {}
    for path, raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValidationFinalizerError(f"{label} has invalid MPI value at {path}")
        integer = int(raw)
        if integer != raw or not 1 <= integer <= max_threads:
            raise ValidationFinalizerError(f"{label} has invalid MPI value {raw!r}")
        result[path] = integer
    if sampler and not values:
        raise ValidationFinalizerError(f"{label} sampler has no MPI setting")
    if sampler and any(value != 1 for value in result.values()):
        raise ValidationFinalizerError(f"{label} sampler must use MPI size 1")
    if not result:
        result["default"] = 1
    return result


def _memory_values(payload: Mapping[str, Any]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    observed_names = {
        "peak_rss_bytes",
        "measured_peak_rss_bytes",
        "peak_memory_bytes",
        "process_peak_rss_bytes",
    }
    configured_names = {
        "configured_process_rss_limit_bytes",
        "process_rss_limit_bytes",
        "process_memory_limit_bytes",
        "max_memory_bytes",
        "memory_limit_bytes",
        "process_rss_hard_limit_bytes",
    }
    observed: list[tuple[str, int]] = []
    configured: list[tuple[str, int]] = []

    def visit(child: Any, path: str) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                nested_path = f"{path}.{key}"
                if key in observed_names and isinstance(nested, (int, float)):
                    observed.append((nested_path, int(nested)))
                elif key in configured_names and isinstance(nested, (int, float)):
                    configured.append((nested_path, int(nested)))
                else:
                    visit(nested, nested_path)
        elif isinstance(child, list):
            for index, nested in enumerate(child):
                visit(nested, f"{path}[{index}]")

    visit(payload, "$")
    return observed, configured


def _resource_record(
    payload: Mapping[str, Any],
    label: str,
    *,
    sampler: bool = False,
    require_observed: bool = False,
    max_threads: int = LEGACY_MAX_THREADS,
) -> dict[str, Any]:
    threads = _thread_values(payload, label, max_threads=max_threads)
    mpi = _mpi_values(payload, label, sampler=sampler, max_threads=max_threads)
    observed, configured = _memory_values(payload)
    if require_observed and not observed:
        raise ValidationFinalizerError(f"{label} has no measured peak RSS")
    bad_observed = [value for _, value in observed if value <= 0 or value >= PROCESS_MEMORY_HARD_LIMIT_BYTES]
    if bad_observed:
        raise ValidationFinalizerError(f"{label} violates strict process memory <2 GiB")
    for path, value in configured:
        if value <= 0 or value > PROCESS_MEMORY_HARD_LIMIT_BYTES:
            raise ValidationFinalizerError(f"{label} has invalid memory limit at {path}")
    return {
        "thread_values": threads,
        "thread_limit": max_threads,
        "mpi_values": mpi,
        "observed_peak_rss_bytes": {path: value for path, value in observed},
        "configured_memory_bytes": {path: value for path, value in configured},
        "strict_process_memory_under_2_gib": all(
            value < PROCESS_MEMORY_HARD_LIMIT_BYTES for _, value in observed
        ),
        f"threads_at_most_{max_threads}": True,
        "mpi_processes_are_valid": all(value <= max_threads for value in mpi.values()),
    }


def _report_record(path: Path, payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "status": payload.get("status"),
        "schema_version": payload.get("schema_version"),
        "command": payload.get("command"),
    }


def _validate_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = _status(payload, "source manifest", {"pass_primary_files_present", "pass"})
    if _scan_forbidden_sampler_text(payload):
        raise ValidationFinalizerError("source manifest contains an active UltraNest reference")
    zenodo = _mapping(payload.get("zenodo"), "source manifest.zenodo")
    _text(zenodo.get("doi"), "source manifest.zenodo.doi")
    _text(zenodo.get("license"), "source manifest.zenodo.license")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationFinalizerError("source manifest.files is empty")
    primary = []
    for index, item in enumerate(files):
        record = _mapping(item, f"source manifest.files[{index}]")
        if record.get("optional") is True:
            continue
        primary.append(record)
        if record.get("status") != "present_verified":
            raise ValidationFinalizerError(
                f"source manifest primary file {record.get('name')!r} is not verified"
            )
        expected = _text(record.get("sha256_expected"), "source expected SHA-256").lower()
        recorded = _text(record.get("sha256"), "source recorded SHA-256").lower()
        if expected != recorded or len(expected) != 64:
            raise ValidationFinalizerError(
                f"source manifest SHA-256 mismatch for {record.get('name')!r}"
            )
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValidationFinalizerError(f"source manifest has invalid size for {record.get('name')!r}")
        _text(record.get("local_path"), "source local path")
    if len(primary) != 7:
        raise ValidationFinalizerError(f"source manifest has {len(primary)} primary files; expected 7")
    acquisition = _mapping(payload.get("acquisition"), "source manifest.acquisition")
    if acquisition.get("deserializes_archives") is not False:
        raise ValidationFinalizerError("source acquisition must not deserialize archives")
    resources = _resource_record(
        payload,
        "source manifest",
        require_observed=False,
        max_threads=LEGACY_MAX_THREADS,
    )
    return {
        "status": status,
        "primary_file_count": len(primary),
        "primary_files_verified_from_manifest": True,
        "external_file_bytes_not_rehashed": True,
        "resources": resources,
    }


def _validate_lrs(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = _status(payload, "LRS report", {"pass"})
    metrics = _mapping(payload.get("metrics"), "LRS report.metrics")
    if metrics.get("smith_consistency_pass") is not True:
        raise ValidationFinalizerError("LRS Smith consistency gate did not pass")
    acceptance = payload.get("acceptance")
    gates = (
        _all_boolean_gates(payload, "LRS report")
        if isinstance(acceptance, Mapping)
        else {"smith_consistency_pass": True}
    )
    data = _mapping(payload.get("data"), "LRS report.data")
    if data.get("table_points") != 150:
        raise ValidationFinalizerError("LRS report does not contain the 150 published bins")
    detectors = data.get("detectors")
    if detectors != ["nirspec_g395h_nrs1", "nirspec_g395h_nrs2"]:
        raise ValidationFinalizerError("LRS report does not preserve the NRS1/NRS2 split")
    resources = _resource_record(
        payload,
        "LRS report",
        require_observed=True,
        max_threads=LEGACY_MAX_THREADS,
    )
    if "vmr" not in json.dumps(payload, sort_keys=True).casefold():
        raise ValidationFinalizerError("LRS report has no VMR composition statement")
    return {"status": status, "gates": gates, "resources": resources}


def _stage_order(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).casefold() for item in value)
    if isinstance(value, str):
        text = value.casefold()
        stages: list[str] = []
        if "rot" in text or "gray" in text:
            stages.append("rotational-broadening")
        if "gaussian" in text or "lsf" in text:
            stages.append("gaussian-high-resolution")
        if "pixel" in text:
            stages.append("pixel-bin-integration")
        return tuple(stages)
    return ()


def _validate_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    response = _mapping(payload.get("response"), "HRS report.response")
    order = _stage_order(response.get("stage_order"))
    if len(order) < 2 or "rot" not in order[0] and "gray" not in order[0]:
        raise ValidationFinalizerError("HRS response must start with Gray rotation")
    if "gaussian" not in order[1] and "lsf" not in order[1]:
        raise ValidationFinalizerError("HRS response must apply Gaussian LSF after rotation")
    if any("pixel" in stage for stage in order[:2]):
        raise ValidationFinalizerError("HRS pixel mapping cannot precede the Gaussian LSF")
    return {
        "stage_order": list(order),
        "rotation_then_gaussian": True,
        "pixel_mapping_after_lsf": True,
    }


def _validate_hrs(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = _status(payload, "HRS report", {"pass"})
    gates = _all_boolean_gates(payload, "HRS report")
    response = _validate_response(payload)
    method = _mapping(payload.get("method"), "HRS report.method")
    if method.get("abundance_convention") != "ROBERT volume mixing ratios only":
        raise ValidationFinalizerError("HRS report does not declare ROBERT VMR-only input")
    resources = _resource_record(
        payload,
        "HRS report",
        require_observed=True,
        max_threads=LEGACY_MAX_THREADS,
    )
    observation = _mapping(payload.get("observation"), "HRS report.observation")
    if observation.get("n_orders") != 44 or observation.get("n_frames") != 79:
        raise ValidationFinalizerError("HRS report does not describe the public 44 x 79 cube")
    return {
        "status": status,
        "gates": gates,
        "response": response,
        "resources": resources,
    }


def _validate_hminus_and_composition(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    combined = json.dumps(payloads, sort_keys=True).casefold()
    if "volume_mixing_ratio" not in combined and "vmr" not in combined:
        raise ValidationFinalizerError("reports do not declare volume mixing ratios")
    violations: list[str] = []

    def visit(value: Any, path: str, external: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key).casefold()
                context_external = external or "prt" in key_text or "boundary" in key_text
                if "mass_fraction" in key_text or "mass_fractions" in key_text:
                    allowed = (
                        nested is False
                        or nested is None
                        or nested == "none"
                        or nested == []
                        or context_external
                    )
                    if not allowed:
                        violations.append(f"{path}.{key}")
                visit(nested, f"{path}.{key}", context_external)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]", external)

    visit(payloads, "$")
    if violations:
        raise ValidationFinalizerError(f"ROBERT mass-fraction state found at {violations[0]}")

    # The benchmark report records the complete BF/FF note only after the
    # physical ROBERT report is written.  Keep this as an explicit pending
    # state during the pre-sampler phase.
    hminus_text = " ".join(
        json.dumps(payload, sort_keys=True).casefold() for payload in payloads.values()
    )
    has_bf = "bound-free" in hminus_text or "bound_free" in hminus_text or " bf " in f" {hminus_text} "
    has_ff = "free-free" in hminus_text or "free_free" in hminus_text or " ff " in f" {hminus_text} "
    has_identifiability = "identif" in hminus_text or "prior-dominated" in hminus_text or "prior_dominated" in hminus_text
    return {
        "robert_vmr_only": True,
        "mass_fraction_parameters_in_robert": False,
        "hminus_bound_free_free_free_identifiability_notes": bool(
            has_bf and has_ff and has_identifiability
        ),
        "hminus_notes_pending": not (has_bf and has_ff and has_identifiability),
        "hminus_notes": {
            "bound_free_present": has_bf,
            "free_free_present": has_ff,
            "identifiability_present": has_identifiability,
        },
    }


def _validate_source_hashes(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the lightweight source identities carried by the injection."""

    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValidationFinalizerError("injection report.source_hashes is missing")
    verified = 0
    for name, item in source_hashes.items():
        record = _mapping(item, f"injection report.source_hashes.{name}")
        if record.get("verified") is not True:
            raise ValidationFinalizerError(
                f"injection source hash {name!r} is not verified"
            )
        digest = _text(
            record.get("sha256"), f"injection source hash {name!r}.sha256"
        ).casefold()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValidationFinalizerError(
                f"injection source hash {name!r} is not a SHA-256 digest"
            )
        size = record.get("size_bytes")
        if size is not None and (
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
        ):
            raise ValidationFinalizerError(
                f"injection source hash {name!r} has an invalid size"
            )
        verified += 1
    return {"verified_records": verified}


def _validate_injection_noise_scales(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate finite, structured per-order Smith HRS noise scales.

    The report stores the values in a nested calibration record and hashes
    their contiguous little-endian float64 representation.  This keeps the
    report compact while making the order count and calibration identity
    auditable without reading the HRS cube.
    """

    model = _mapping(payload.get("generative_model"), "injection report.generative_model")
    calibration = model.get("hrs_noise_calibration")
    if not isinstance(calibration, Mapping):
        raise ValidationFinalizerError(
            "injection report.generative_model.hrs_noise_calibration is missing"
        )
    declared_orders = calibration.get("n_orders")
    if declared_orders != EXPECTED_HRS_ORDER_COUNT:
        raise ValidationFinalizerError(
            "injection HRS noise calibration must declare 19 orders"
        )
    raw = calibration.get("sigma_by_order")
    checksum = calibration.get("sigma_sha256")
    if not isinstance(raw, list) or len(raw) != EXPECTED_HRS_ORDER_COUNT:
        raise ValidationFinalizerError(
            "injection report must contain 19 HRS noise calibration scales"
        )

    scales: list[float] = []
    for index, item in enumerate(raw):
        sigma = _finite_number(
            item,
            f"injection HRS noise scale[{index}].sigma",
            positive=True,
        )
        scales.append(sigma)
    digest = _text(checksum, "injection HRS noise scale SHA-256").casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValidationFinalizerError(
            "injection HRS noise scale checksum is not a SHA-256 digest"
        )
    calculated = hashlib.sha256(struct.pack(f"<{len(scales)}d", *scales)).hexdigest()
    if digest != calculated:
        raise ValidationFinalizerError(
            "injection HRS noise scale checksum does not match the structured values"
        )
    seed = calibration.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValidationFinalizerError("injection HRS noise calibration seed is invalid")
    for key in ("scale_definition", "distribution", "chunking"):
        _text(calibration.get(key), f"injection HRS noise calibration.{key}")
    summary = _mapping(
        calibration.get("sigma_summary"),
        "injection HRS noise calibration.sigma_summary",
    )
    summary_values = {
        key: _finite_number(
            summary.get(key),
            f"injection HRS noise calibration.sigma_summary.{key}",
            positive=True,
        )
        for key in ("min", "max", "mean", "rms")
    }
    expected_summary = {
        "min": min(scales),
        "max": max(scales),
        "mean": math.fsum(scales) / len(scales),
        "rms": math.sqrt(math.fsum(value * value for value in scales) / len(scales)),
    }
    if any(
        not math.isclose(summary_values[key], expected_summary[key], rel_tol=1e-12, abs_tol=0.0)
        for key in expected_summary
    ):
        raise ValidationFinalizerError(
            "injection HRS noise calibration summary does not match sigma_by_order"
        )
    return {
        "order_count": len(scales),
        "sha256": digest,
        "all_positive_finite": True,
        "seed": seed,
        "summary_verified": True,
    }


def _validate_injection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the required real-data deterministic ROBERT injection report."""

    status = _status(payload, "ROBERT injection report", {"pass"})
    if payload.get("record_type") != EXPECTED_INJECTION_RECORD_TYPE:
        raise ValidationFinalizerError(
            "ROBERT injection report has the wrong record_type"
        )
    _required_boolean(payload, "ROBERT injection report", "real_data_loaded")

    sampler = _mapping(payload.get("sampler"), "ROBERT injection report.sampler")
    _required_boolean(sampler, "ROBERT injection report.sampler", "used", expected=False)
    acceptance = _mapping(
        payload.get("acceptance"), "ROBERT injection report.acceptance"
    )
    _required_boolean(
        acceptance, "ROBERT injection report.acceptance", "posterior_recovery_claim", expected=False
    )
    for key, value in acceptance.items():
        if key == "posterior_recovery_claim":
            continue
        if not isinstance(value, bool) or value is not True:
            raise ValidationFinalizerError(
                f"ROBERT injection report acceptance gate {key!r} is not true"
            )
    required_acceptance = (
        "all_modes_finite",
        "truth_preferred_to_null",
        "truth_preferred_to_off_velocity",
        "hminus_on_off_evaluated",
        "resource_gate",
    )
    for key in required_acceptance:
        _required_boolean(acceptance, "ROBERT injection report.acceptance", key)

    checks = _mapping(payload.get("checks"), "ROBERT injection report.checks")
    _required_boolean(checks, "ROBERT injection report.checks", "real_data_truth_recovery_claim", expected=False)
    mode_results = _mapping(
        payload.get("mode_results"), "ROBERT injection report.mode_results"
    )
    expected_modes = {"hrs_only", "lrs_only", "joint"}
    if set(mode_results) != expected_modes:
        raise ValidationFinalizerError(
            "ROBERT injection report must contain HRS-only, LRS-only, and joint modes"
        )
    for mode_name in ("hrs_only", "lrs_only", "joint"):
        mode = _mapping(mode_results[mode_name], f"injection mode {mode_name}")
        _required_boolean(mode, f"injection mode {mode_name}", "gate_passed")
        _required_boolean(mode, f"injection mode {mode_name}", "hminus_on_off_finite")
        for branch_name in ("hminus_on", "hminus_off"):
            branch = _mapping(
                mode.get(branch_name), f"injection mode {mode_name}.{branch_name}"
            )
            branch_label = f"injection mode {mode_name}.{branch_name}"
            _required_boolean(branch, branch_label, "finite")
            _required_boolean(branch, branch_label, "truth_preferred_to_null")
            _required_boolean(branch, branch_label, "truth_preferred_to_off_velocity")
            for key in (
                "truth_loglike",
                "null_loglike",
                "off_velocity_loglike",
                "truth_minus_null",
                "truth_minus_off_velocity",
                "off_velocity_delta_km_s",
            ):
                _finite_number(branch.get(key), f"{branch_label}.{key}")

    composition = _mapping(
        payload.get("composition_policy"), "ROBERT injection report.composition_policy"
    )
    if composition.get("robert_convention") != "volume_mixing_ratio":
        raise ValidationFinalizerError("ROBERT injection report is not VMR based")
    if composition.get("mass_fraction_parameters_in_robert") is not False:
        raise ValidationFinalizerError(
            "ROBERT injection report declares mass-fraction parameters"
        )
    if tuple(composition.get("hminus_species", ())) != ("H-", "H", "e-"):
        raise ValidationFinalizerError(
            "ROBERT injection report has an incomplete H-minus species set"
        )

    truth = _mapping(
        payload.get("truth_parameters"), "ROBERT injection report.truth_parameters"
    )
    if not truth:
        raise ValidationFinalizerError("ROBERT injection report.truth_parameters is empty")
    for name, value in truth.items():
        _finite_number(value, f"injection truth_parameters.{name}")
    noise = _validate_injection_noise_scales(payload)
    source_hashes = _validate_source_hashes(payload)

    resource_policy = _mapping(
        payload.get("resource_policy"), "ROBERT injection report.resource_policy"
    )
    if resource_policy.get("max_threads") != ROBERT_MAX_THREADS:
        raise ValidationFinalizerError(
            "ROBERT injection report must declare the exact six-thread cap"
        )
    if resource_policy.get("mpi_processes") != 1:
        raise ValidationFinalizerError("ROBERT injection report must declare MPI size 1")
    if resource_policy.get("thread_variables") != list(THREAD_VARIABLES):
        raise ValidationFinalizerError(
            "ROBERT injection report has the wrong thread-variable contract"
        )
    if resource_policy.get("process_memory_limit_bytes") != CONFIGURED_PROCESS_MEMORY_LIMIT_BYTES:
        raise ValidationFinalizerError(
            "ROBERT injection report must use the configured 1.9 GiB process limit"
        )
    if resource_policy.get("hard_process_memory_limit_bytes") != PROCESS_MEMORY_HARD_LIMIT_BYTES:
        raise ValidationFinalizerError(
            "ROBERT injection report must use the 2 GiB hard process limit"
        )

    resources_payload = _mapping(
        payload.get("resources"), "ROBERT injection report.resources"
    )
    for key in ("gate_passed", "mpi_pass", "process_rss_pass", "threads_pass"):
        _required_boolean(resources_payload, "ROBERT injection report.resources", key)
    if resources_payload.get("thread_limit") != ROBERT_MAX_THREADS:
        raise ValidationFinalizerError(
            "ROBERT injection report resources do not declare the six-thread cap"
        )
    thread_values = resources_payload.get("thread_values")
    if not isinstance(thread_values, Mapping) or set(thread_values) != set(THREAD_VARIABLES):
        raise ValidationFinalizerError(
            "ROBERT injection report resources have incomplete thread values"
        )
    for name in THREAD_VARIABLES:
        raw = thread_values[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= ROBERT_MAX_THREADS:
            raise ValidationFinalizerError(
                f"ROBERT injection report resources.{name} is outside 1..6"
            )
    if resources_payload.get("mpi_processes_configured") != 1 or resources_payload.get("mpi_world_size_observed") != 1:
        raise ValidationFinalizerError("ROBERT injection report resources must use MPI size 1")
    peak = resources_payload.get("peak_rss_bytes")
    configured = resources_payload.get("configured_process_rss_limit_bytes")
    hard = resources_payload.get("process_rss_hard_limit_bytes")
    peak_value = _finite_number(peak, "ROBERT injection report.resources.peak_rss_bytes", positive=True)
    configured_value = _finite_number(
        configured,
        "ROBERT injection report.resources.configured_process_rss_limit_bytes",
        positive=True,
    )
    hard_value = _finite_number(
        hard,
        "ROBERT injection report.resources.process_rss_hard_limit_bytes",
        positive=True,
    )
    if int(configured_value) != CONFIGURED_PROCESS_MEMORY_LIMIT_BYTES:
        raise ValidationFinalizerError(
            "ROBERT injection report resources do not record the configured 1.9 GiB limit"
        )
    if int(hard_value) != PROCESS_MEMORY_HARD_LIMIT_BYTES:
        raise ValidationFinalizerError(
            "ROBERT injection report resources do not record the 2 GiB hard limit"
        )
    if not peak_value < configured_value:
        raise ValidationFinalizerError(
            "ROBERT injection report peak RSS is not below the configured 1.9 GiB limit"
        )
    if not peak_value < hard_value:
        raise ValidationFinalizerError(
            "ROBERT injection report peak RSS is not below the 2 GiB hard limit"
        )
    resources = _resource_record(
        payload,
        "ROBERT injection report",
        require_observed=True,
        max_threads=ROBERT_MAX_THREADS,
    )
    resources["strict_process_memory_under_configured_limit"] = True
    resources["configured_process_memory_limit_bytes"] = CONFIGURED_PROCESS_MEMORY_LIMIT_BYTES
    return {
        "status": status,
        "record_type": EXPECTED_INJECTION_RECORD_TYPE,
        "real_data_loaded": True,
        "sampler_used": False,
        "posterior_recovery_claim": False,
        "modes": {
            "hrs_only": "finite truth/null/off-velocity and H-minus on/off",
            "lrs_only": "finite truth/null and H-minus on/off",
            "joint": "finite truth/null/off-velocity and H-minus on/off",
        },
        "truth_parameters": {
            "count": len(truth),
            "all_finite": True,
        },
        "hrs_noise_scales": noise,
        "source_hashes": source_hashes,
        "resources": resources,
    }


def _validate_stride(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = _status(payload, "LBL stride report", {"pass"})
    gates = _all_boolean_gates(payload, "LBL stride report")
    hrs = _mapping(payload.get("hrs"), "LBL stride report.hrs")
    lrs = _mapping(payload.get("lrs"), "LBL stride report.lrs")
    if hrs.get("selected_finest_passing_stride") != 2:
        raise ValidationFinalizerError("LBL stride report did not select HRS stride 2")
    if lrs.get("selected_finest_passing_stride") != 25:
        raise ValidationFinalizerError("LBL stride report did not select LRS stride 25")
    try:
        hrs_candidates = [int(value) for value in hrs.get("candidate_strides", [])]
        lrs_candidates = [int(value) for value in lrs.get("candidate_strides", [])]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationFinalizerError("LBL stride candidates are invalid") from error
    if 2 not in hrs_candidates:
        raise ValidationFinalizerError("LBL stride report has no HRS stride-2 candidate")
    if 25 not in lrs_candidates:
        raise ValidationFinalizerError("LBL stride report has no LRS stride-25 candidate")
    for branch, stride in ((hrs, 2), (lrs, 25)):
        metric = _mapping(branch.get("metrics"), "LBL stride metric")
        selected = _mapping(metric.get(str(stride)), "LBL selected stride metric")
        if selected.get("gate_pass") is not True:
            raise ValidationFinalizerError("selected LBL stride metric did not pass")
    atmosphere = _mapping(payload.get("atmosphere"), "LBL stride report.atmosphere")
    continuum = _mapping(atmosphere.get("hminus_continuum"), "LBL stride H-minus continuum")
    if tuple(continuum.get("config_species", ())) != ("H-", "H", "e-"):
        raise ValidationFinalizerError("LBL stride report has an incomplete H-minus species set")
    if continuum.get("included_in_all_stride_evaluations") is not True:
        raise ValidationFinalizerError("H-minus was not included in every LBL stride")
    if atmosphere.get("composition_convention") != "volume_mixing_ratio":
        raise ValidationFinalizerError("LBL stride report is not VMR based")
    if atmosphere.get("mass_fraction_parameters") is not False:
        raise ValidationFinalizerError("LBL stride report has ROBERT mass-fraction parameters")
    resources = _resource_record(
        payload,
        "LBL stride report",
        require_observed=True,
        max_threads=LEGACY_MAX_THREADS,
    )
    opacity_limits = _find_values(payload, {"opacity_estimate_limit_bytes"})
    if any(
        isinstance(value, (int, float)) and value >= 1024**3
        for _, value in opacity_limits
    ):
        raise ValidationFinalizerError("LBL opacity estimate limit is not below 1023 MiB")
    return {
        "status": status,
        "gates": gates,
        "selected_strides": {"hrs": 2, "lrs": 25},
        "resources": resources,
        "opacity_limit_below_1_gib": True,
    }


def _plot_records(
    role: str,
    payload: Mapping[str, Any],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    """Hash only paths explicitly located below plot/figure keys."""

    records: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add_artifact(value: Any, expected: Any = None) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        artifact = _resolve_path(value, root, f"{role} plot path")
        if artifact in seen:
            return
        seen.add(artifact)
        if artifact.is_file():
            actual = _sha256(artifact)
            if expected is not None and str(expected).casefold() != actual:
                raise ValidationFinalizerError(
                    f"{role} plot SHA-256 does not match: {artifact}"
                )
            records.append(
                {
                    "path": _display_path(artifact, root),
                    "bytes": artifact.stat().st_size,
                    "sha256": actual,
                    "status": "verified",
                }
            )
        else:
            records.append(
                {
                    "path": _display_path(artifact, root),
                    "sha256": expected,
                    "status": "missing_optional_artifact",
                }
            )

    def visit(value: Any, path: str, plot_context: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key).casefold()
                nested_context = plot_context or key_text in {"plot", "plots", "figure", "figures"}
                if isinstance(nested, str) and key_text in {"plot", "figure"}:
                    # Some compact reports use ``"plot": "/path/to/file"``.
                    add_artifact(nested)
                elif nested_context and key_text in {"path", "output", "output_path", "file"}:
                    expected = value.get("sha256")
                    add_artifact(nested, expected)
                # Do not interpret a plot's SHA string, label, or other text
                # as another filesystem path.
                if not (nested_context and isinstance(nested, str)):
                    visit(nested, f"{path}.{key_text}", nested_context)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]", plot_context)

    visit(payload, "$")
    return records


def _sampler_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("sampler_policy", "pymultinest", "sampler"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _run_groups(value: Any) -> list[tuple[str, list[Mapping[str, Any]], Mapping[str, Any] | None]]:
    """Flatten the fixed and ROBERT report run layouts."""

    groups: list[tuple[str, list[Mapping[str, Any]], Mapping[str, Any] | None]] = []
    if isinstance(value, list):
        records = [_mapping(item, "sampler run") for item in value]
        return [("default", records, None)]
    if not isinstance(value, Mapping):
        return groups
    direct_runs = value.get("runs")
    if isinstance(direct_runs, list):
        records = [_mapping(item, "sampler run") for item in direct_runs]
        groups.append(("default", records, value.get("evidence_comparison") if isinstance(value.get("evidence_comparison"), Mapping) else None))
        return groups
    for name, child in value.items():
        if isinstance(child, Mapping) and isinstance(child.get("runs"), list):
            records = [_mapping(item, f"sampler {name} run") for item in child["runs"]]
            comparison = child.get("evidence_comparison")
            groups.append((str(name), records, comparison if isinstance(comparison, Mapping) else None))
        elif isinstance(child, list):
            records = [_mapping(item, f"sampler {name} run") for item in child]
            groups.append((str(name), records, None))
    return groups


def _validate_sampler(
    payload: Mapping[str, Any],
    role: str,
    *,
    require_complete: bool,
    max_threads: int = LEGACY_MAX_THREADS,
) -> dict[str, Any]:
    active_text = _scan_forbidden_sampler_text(payload)
    # An explicit false policy flag is allowed.  Any other UltraNest mention
    # would make the sampler provenance ambiguous.
    if active_text:
        raise ValidationFinalizerError(f"{role} contains an active UltraNest reference")
    sampler = _sampler_mapping(payload)
    if sampler is None:
        if require_complete:
            raise ValidationFinalizerError(f"{role} has no PyMultiNest policy")
        return {"status": "pending_missing_sampler_policy", "complete": False}
    sampler_name = str(sampler.get("sampler", sampler.get("name", ""))).casefold()
    backend = str(sampler.get("backend", "")).casefold()
    if "pymultinest" not in sampler_name or "multinest" not in backend:
        raise ValidationFinalizerError(f"{role} is not a PyMultiNest/MultiNest report")
    declared_max_threads = sampler.get("max_threads")
    if declared_max_threads is not None:
        if (
            isinstance(declared_max_threads, bool)
            or not isinstance(declared_max_threads, (int, float))
            or int(declared_max_threads) != declared_max_threads
            or int(declared_max_threads) != max_threads
        ):
            raise ValidationFinalizerError(
                f"{role} declares max_threads={declared_max_threads!r}; "
                f"expected {max_threads}"
            )
    max_iter = sampler.get("max_iter")
    if max_iter != 0:
        raise ValidationFinalizerError(f"{role} must use max_iter=0")
    seeds = sampler.get("seeds", sampler.get("seed_pair"))
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValidationFinalizerError(f"{role} must declare two distinct sampler seeds")
    if tuple(seeds[:2]) != EXPECTED_SEEDS:
        raise ValidationFinalizerError(f"{role} must use the declared seeds {EXPECTED_SEEDS}")
    mpi = _mpi_values(payload, role, sampler=True)
    threads = _thread_values(payload, role, max_threads=max_threads)
    resources = _resource_record(
        payload,
        role,
        sampler=True,
        require_observed=False,
        max_threads=max_threads,
    )
    groups = _run_groups(payload.get("runs"))
    complete = bool(groups)
    compact_groups: list[dict[str, Any]] = []
    tolerance = sampler.get("evidence_tolerance")
    if not isinstance(tolerance, (int, float)):
        tolerance = 0.5
    for name, runs, comparison in groups:
        if len(runs) != 2:
            raise ValidationFinalizerError(f"{role} group {name} does not contain two runs")
        run_seeds = [run.get("seed", run.get("sampler_seed")) for run in runs]
        if run_seeds != list(EXPECTED_SEEDS):
            raise ValidationFinalizerError(f"{role} group {name} has the wrong seed pair")
        for index, run in enumerate(runs):
            state = run.get("status")
            if state not in {"pass", "completed"} or run.get("converged") is not True:
                raise ValidationFinalizerError(f"{role} group {name} run {index + 1} did not converge")
            if run.get("resource_gate_passed") is False:
                raise ValidationFinalizerError(f"{role} group {name} run {index + 1} failed resource gate")
            peak = run.get("peak_rss_bytes")
            if isinstance(peak, (int, float)) and peak >= PROCESS_MEMORY_HARD_LIMIT_BYTES:
                raise ValidationFinalizerError(f"{role} group {name} exceeds 2 GiB RSS")
        if comparison is not None:
            if comparison.get("gate_passed") is not True:
                raise ValidationFinalizerError(f"{role} group {name} evidence repeatability failed")
            delta = comparison.get("delta_log_evidence")
            if isinstance(delta, (int, float)) and abs(float(delta)) > float(tolerance):
                raise ValidationFinalizerError(f"{role} group {name} evidence difference exceeds tolerance")
        compact_groups.append({"name": name, "run_count": 2, "evidence_comparison": comparison})
    if not groups:
        complete = False
    if require_complete and not complete:
        raise ValidationFinalizerError(f"{role} sampler evidence is pending")
    report_status = str(payload.get("status", ""))
    if report_status == "fail":
        raise ValidationFinalizerError(f"{role} report has status fail")
    return {
        "status": "pass" if complete else "pending_sampler_runs",
        "complete": complete,
        "backend": "PyMultiNest/MultiNest",
        "seeds": list(EXPECTED_SEEDS),
        "max_iter": 0,
        "max_threads": max_threads,
        "mpi_processes": mpi,
        "thread_values": threads,
        "resources": resources,
        "groups": compact_groups,
    }


def _git_record(root: Path) -> dict[str, Any]:
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
        # Tiny test fixtures and copied release bundles may not have a Git
        # directory.  Keep the scientific checks useful and mark provenance
        # as unavailable instead of claiming a commit.
        return {"available": False, "error": str(error)}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "dirty_path_count": len(status),
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


def build_validation_manifest(
    *,
    report_dir: Path = DEFAULT_REPORT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    root: Path = ROOT,
    require_samplers: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    """Validate reports and atomically write a compact manifest."""

    report_dir = report_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    root = root.expanduser().resolve()
    paths = {role: (report_dir / name).resolve() for role, name in REPORT_NAMES.items()}
    payloads: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        if role in OPTIONAL_SAMPLER_ROLES and not path.is_file():
            records[role] = {"path": _display_path(path, root), "status": "pending_missing_report"}
            continue
        payload = _read_json(path, f"{role} report")
        payloads[role] = payload
        if _scan_forbidden_sampler_text(payload):
            raise ValidationFinalizerError(f"{role} report contains an active UltraNest reference")
        records[role] = _report_record(path, payload, root)

    validation = {
        "source": _validate_source(payloads["source"]),
        "lrs": _validate_lrs(payloads["lrs"]),
        "hrs": _validate_hrs(payloads["hrs"]),
        "stride": _validate_stride(payloads["stride"]),
        "injection": _validate_injection(payloads["injection"]),
    }
    composition = _validate_hminus_and_composition(payloads)
    plots = {
        role: _plot_records(role, payload, root=root)
        for role, payload in payloads.items()
    }
    samplers: dict[str, Any] = {}
    for role in OPTIONAL_SAMPLER_ROLES:
        if role not in payloads:
            samplers[role] = {"status": "pending_missing_report", "complete": False}
            continue
        sampler_max_threads = (
            ROBERT_MAX_THREADS
            if role == "robert_joint_sampler"
            else LEGACY_MAX_THREADS
        )
        samplers[role] = _validate_sampler(
            payloads[role],
            role,
            require_complete=require_samplers,
            max_threads=sampler_max_threads,
        )
    all_samplers_complete = all(
        bool(samplers[role].get("complete")) for role in OPTIONAL_SAMPLER_ROLES
    )
    if require_samplers and not composition["hminus_bound_free_free_free_identifiability_notes"]:
        raise ValidationFinalizerError("H-minus BF/FF identifiability notes are not complete")
    pending = [
        role for role, record in samplers.items() if record.get("status", "").startswith("pending")
    ]
    if composition["hminus_notes_pending"]:
        pending.append("hminus_notes")
    status = "pass" if not pending else "pending_samplers_or_notes"
    if require_samplers and not all_samplers_complete:
        raise ValidationFinalizerError("required PyMultiNest evidence is pending")
    resources = {
        "thread_variables": list(THREAD_VARIABLES),
        "max_threads": ROBERT_MAX_THREADS,
        "legacy_report_max_threads": LEGACY_MAX_THREADS,
        "mpi_processes_required_for_samplers": 1,
        "strict_process_memory_limit_bytes": PROCESS_MEMORY_HARD_LIMIT_BYTES,
        "opacity_estimate_limit_bytes": OPACITY_MEMORY_LIMIT_BYTES,
        "reports": {
            role: validation[role]["resources"]
            for role in ("lrs", "hrs", "stride", "injection")
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_name": "wasp77ab_validation_manifest_20260831",
        "status": status,
        "target": "WASP-77Ab",
        "comparison": "Smith et al. 2024 IGRINS HRS + August et al. 2023 NIRSpec LRS",
        "reports": records,
        "report_hashes": {
            role: record["sha256"] for role, record in records.items() if "sha256" in record
        },
        "validation": validation,
        "composition": composition,
        "resources": resources,
        "samplers": samplers,
        "plots": plots,
        "pending": pending,
        "sampler_policy": {
            "sampler": "PyMultiNest",
            "backend": "MultiNest",
            "ultranest_allowed": False,
            "two_seed_requirement": True,
            "max_iter": 0,
            "mpi_processes": 1,
            "max_threads": ROBERT_MAX_THREADS,
            "legacy_reference_max_threads": LEGACY_MAX_THREADS,
        },
        "provenance": {
            "source_manifest_primary_hashes_reused": True,
            "external_heavy_inputs_rehashed": False,
            "lightweight_report_and_plot_hashes_verified": True,
            "sampler_chain_files_read": False,
        },
        "git": _git_record(root),
        "command": command or " ".join(shlex.quote(argument) for argument in sys.argv),
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "thread_values": {name: int(os.environ[name]) for name in THREAD_VARIABLES},
        },
    }
    _atomic_write(output_path, manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-samplers",
        action="store_true",
        help="fail unless both two-seed PyMultiNest reports and H-minus notes are complete",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        manifest = build_validation_manifest(
            report_dir=arguments.report_dir,
            output_path=arguments.output,
            root=ROOT,
            require_samplers=arguments.require_samplers,
        )
    except ValidationFinalizerError as error:
        raise SystemExit(f"WASP-77Ab validation manifest not written: {error}") from error
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
