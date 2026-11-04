"""External supervision for configured, serial MultiNest runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

from robert_exoplanets.core import RobertConfigError

from ..status import append_retrieval_attempt_event, write_retrieval_status


CALLBACK_FAILURE_MARKER_ENV = "ROBERT_MULTINEST_CALLBACK_FAILURE_MARKER"
CALLBACK_SUPERVISED_ENV = "ROBERT_MULTINEST_CALLBACK_SUPERVISED"
_CALLBACK_MARKER_SCHEMA = 1
_MPI_LAUNCH_ENV_KEYS = (
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "PMIX_RANK",
    "PMIX_SIZE",
)


def write_callback_failure_marker(path: str | Path, payload: Mapping[str, object]) -> None:
    """Write one complete callback failure record for the supervising parent."""

    marker = Path(path)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    record = {"schema_version": _CALLBACK_MARKER_SCHEMA, **dict(payload)}
    try:
        temporary.write_text(
            json.dumps(record, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(marker)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def supervise_retrieval_command(
    command: Sequence[str],
    *,
    fallback_output_dir: str | Path,
    poll_interval_seconds: float = 0.1,
    terminate_timeout_seconds: float = 5.0,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run one fresh serial retrieval child and stop it after a callback marker.

    The marker path is unique for each call. The parent keeps MultiNest chain
    files and writes a failed status when a callback error appears.
    """

    if not command:
        raise RobertConfigError("supervised retrieval command must not be empty")
    if os.name != "posix":
        raise RobertConfigError(
            "callback supervision requires POSIX process termination"
        )
    if poll_interval_seconds <= 0.0:
        raise RobertConfigError("poll_interval_seconds must be positive")
    if terminate_timeout_seconds < 0.0:
        raise RobertConfigError("terminate_timeout_seconds must be non-negative")

    fallback_root = Path(fallback_output_dir).expanduser().resolve()
    fallback_root.mkdir(parents=True, exist_ok=True)
    child_environment = dict(os.environ if environment is None else environment)
    if _has_mpi_launcher_environment(child_environment):
        raise RobertConfigError(
            "callback supervision is for serial runs; do not launch it under MPI"
        )

    with tempfile.TemporaryDirectory(prefix="robert-multinest-supervisor-") as temp_dir:
        marker_path = Path(temp_dir) / "callback-failure.json"
        child_environment[CALLBACK_FAILURE_MARKER_ENV] = str(marker_path)
        child_environment[CALLBACK_SUPERVISED_ENV] = "1"
        child_environment["ROBERT_MPI_RANKS"] = "1"
        process = subprocess.Popen(
            list(command),
            env=child_environment,
            start_new_session=True,
        )
        try:
            while True:
                marker = _read_marker(marker_path)
                if marker is not None:
                    _stop_child(process, terminate_timeout_seconds)
                    _record_failure(marker, fallback_root, marker_path)
                    return 1

                return_code = process.poll()
                if return_code is not None:
                    # Read again after observing exit to cover a marker that
                    # appeared between the first check and process.poll().
                    marker = _read_marker(marker_path)
                    if marker is not None:
                        _record_failure(marker, fallback_root, marker_path)
                        return 1
                    if return_code == 70:
                        _record_marker_write_failure(fallback_root)
                        return 1
                    return int(return_code)
                time.sleep(poll_interval_seconds)
        except BaseException:
            if process.poll() is None:
                _stop_child(process, terminate_timeout_seconds)
            raise


def _has_mpi_launcher_environment(environment: Mapping[str, str]) -> bool:
    return any(name in environment for name in _MPI_LAUNCH_ENV_KEYS)


def _read_marker(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": _CALLBACK_MARKER_SCHEMA,
            "exception_type": "CallbackMarkerError",
            "message": f"callback failure marker could not be read: {exc}",
        }
    if not isinstance(value, dict) or value.get("schema_version") != _CALLBACK_MARKER_SCHEMA:
        return {
            "schema_version": _CALLBACK_MARKER_SCHEMA,
            "exception_type": "CallbackMarkerError",
            "message": "callback failure marker has an unsupported format",
        }
    return value


def _stop_child(process: subprocess.Popen[bytes], timeout: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _record_failure(
    marker: Mapping[str, object],
    fallback_root: Path,
    marker_path: Path,
) -> None:
    raw_root = marker.get("output_dir")
    output_root = Path(str(raw_root)).expanduser().resolve() if raw_root else fallback_root
    raw_base = marker.get("status_base")
    status_base = dict(raw_base) if isinstance(raw_base, Mapping) else _read_current_status(output_root)
    attempt_id = str(status_base.get("attempt_id") or f"callback-{os.getpid()}-{time.time_ns()}")
    raw_elapsed = marker.get("elapsed_seconds", 0.0)
    try:
        elapsed = max(float(raw_elapsed), 0.0)
    except (TypeError, ValueError, OverflowError):
        elapsed = 0.0
    failure = {
        **status_base,
        "state": "failed",
        "elapsed_seconds": elapsed,
        "exception_type": str(marker.get("exception_type", "CallbackError")),
        "message": str(marker.get("message", "MultiNest callback failed")),
        "callback_failure": True,
        "callback_rank": int(marker.get("rank", 0)),
        "multinest_resume_safe": False,
        "partial_chains_preserved": True,
    }
    if marker.get("traceback"):
        failure["callback_traceback"] = str(marker["traceback"])
    write_retrieval_status(output_root, failure)
    append_retrieval_attempt_event(
        output_root,
        {**failure, "event": "callback_failed", "attempt_id": attempt_id},
    )
    retained_marker = output_root / f"multinest_callback_failure_{attempt_id}.json"
    retained_marker.write_bytes(marker_path.read_bytes())


def _record_marker_write_failure(output_root: Path) -> None:
    status = _read_current_status(output_root)
    attempt_id = status.get("attempt_id")
    failure = {
        **status,
        "state": "failed",
        "exception_type": "CallbackMarkerWriteError",
        "message": "the MultiNest callback failure marker could not be written",
        "callback_failure": True,
        "multinest_resume_safe": False,
        "partial_chains_preserved": True,
    }
    write_retrieval_status(output_root, failure)
    append_retrieval_attempt_event(
        output_root,
        {**failure, "event": "callback_failed", "attempt_id": attempt_id},
    )


def _read_current_status(output_root: Path) -> dict[str, object]:
    path = output_root / "sampler_status.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


__all__ = [
    "CALLBACK_FAILURE_MARKER_ENV",
    "CALLBACK_SUPERVISED_ENV",
    "supervise_retrieval_command",
    "write_callback_failure_marker",
]
