"""External supervision for configured MultiNest runs."""

from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
from functools import lru_cache
import time
import uuid
from typing import Mapping, Sequence

from robert_exoplanets.core import RobertConfigError

from ..status import append_retrieval_attempt_event, write_retrieval_status


CALLBACK_FAILURE_MARKER_ENV = "ROBERT_MULTINEST_CALLBACK_FAILURE_MARKER"
CALLBACK_FAILURE_MARKER_DIR_ENV = "ROBERT_MULTINEST_CALLBACK_FAILURE_MARKER_DIR"
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
_MPI_RANKS_ENV = "ROBERT_MPI_RANKS"
_MPI_MARKER_DIRECTORY = ".multinest_callback_supervision"


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


def write_callback_rank_record(
    marker_dir: str | Path,
    record_type: str,
    rank: int,
    payload: Mapping[str, object] | None = None,
) -> Path:
    """Write one atomic, rank-specific record for the MPI supervisor."""

    if record_type not in {"ready", "failure", "complete"}:
        raise RobertConfigError(f"unsupported callback rank record: {record_type}")
    if isinstance(rank, bool) or int(rank) < 0:
        raise RobertConfigError("callback MPI rank must be a non-negative integer")
    identity = _current_process_identity()
    marker = Path(marker_dir) / (
        f"{record_type}-rank-{int(rank):05d}-pid-{identity['pid']}.json"
    )
    record = {
        **dict(payload or {}),
        "schema_version": _CALLBACK_MARKER_SCHEMA,
        "record_type": record_type,
        "rank": int(rank),
        **identity,
    }
    _write_json_atomically(marker, record)
    return marker


def _write_json_atomically(marker: Path, record: Mapping[str, object]) -> None:
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(record), sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(marker)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _current_process_identity() -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "process_group_id": os.getpgrp(),
        "session_id": os.getsid(0),
        "process_start_time": _linux_process_start_time(os.getpid()),
    }


def _linux_process_start_time(pid: int) -> str | None:
    """Return Linux /proc start time; None means this is not a Linux host."""

    info = _linux_process_info(pid)
    return None if info is None else info[0]


def _linux_process_info(pid: int) -> tuple[str, str] | None:
    if not Path("/proc/self/stat").is_file():
        return None
    try:
        stat_text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # The command name in /proc/PID/stat can contain spaces and parentheses.
    fields_after_comm = stat_text.rsplit(")", 1)[-1].split()
    # The first field is field 3 (state); starttime is field 22.
    if len(fields_after_comm) <= 19:
        return None
    return fields_after_comm[19], fields_after_comm[0]


@lru_cache(maxsize=1)
def _pidfd_operations():
    open_pidfd = getattr(os, "pidfd_open", None)
    send_signal = getattr(signal, "pidfd_send_signal", None)
    if callable(open_pidfd) and callable(send_signal):
        return open_pidfd, send_signal
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return None

    # Conda can build against headers that predate the Python wrappers.
    # Use the Linux x86-64 syscall ABI and retain PID identity verification.
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long

    def checked_syscall(number, *args):
        result = syscall(ctypes.c_long(number), *args)
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    def open_pidfd(pid, flags):
        return checked_syscall(434, ctypes.c_int(pid), ctypes.c_uint(flags))

    def send_signal(fd, sig, _info, flags):
        return checked_syscall(
            424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(flags)
        )

    return open_pidfd, send_signal


def _supports_linux_pidfds() -> bool:
    operations = _pidfd_operations()
    if operations is None or not Path("/proc/self/stat").is_file():
        return False
    open_pidfd, send_signal = operations
    try:
        fd = open_pidfd(os.getpid(), 0)
        try:
            send_signal(fd, 0, None, 0)
        finally:
            os.close(fd)
    except OSError:
        return False
    return True


def supervise_retrieval_command(
    command: Sequence[str],
    *,
    fallback_output_dir: str | Path,
    poll_interval_seconds: float = 0.1,
    terminate_timeout_seconds: float = 5.0,
    startup_timeout_seconds: float = 120.0,
    mpi_nprocs: int = 1,
    mpi_launcher: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run a fresh MultiNest child and stop it after a callback failure.

    Serial runs use one unique marker file. Opt-in MPI supervision is limited
    to one-node Slurm allocations with Linux process identities and an Open MPI
    launcher. It uses a fresh marker directory and rank-specific records.
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
    if startup_timeout_seconds <= 0.0:
        raise RobertConfigError("startup_timeout_seconds must be positive")
    if isinstance(mpi_nprocs, bool) or int(mpi_nprocs) < 1:
        raise RobertConfigError("mpi_nprocs must be a positive integer")
    rank_count = int(mpi_nprocs)

    fallback_root = Path(fallback_output_dir).expanduser().resolve()
    fallback_root.mkdir(parents=True, exist_ok=True)
    child_environment = dict(os.environ if environment is None else environment)
    if _has_mpi_launcher_environment(child_environment):
        raise RobertConfigError(
            "callback supervision must start outside an MPI world"
        )

    child_environment.pop(CALLBACK_FAILURE_MARKER_ENV, None)
    child_environment.pop(CALLBACK_FAILURE_MARKER_DIR_ENV, None)
    child_environment[CALLBACK_SUPERVISED_ENV] = "1"
    child_environment[_MPI_RANKS_ENV] = str(rank_count)

    if rank_count == 1:
        return _supervise_serial_command(
            command,
            fallback_root=fallback_root,
            child_environment=child_environment,
            poll_interval_seconds=poll_interval_seconds,
            terminate_timeout_seconds=terminate_timeout_seconds,
        )

    if mpi_launcher is None:
        raise RobertConfigError(
            "MPI callback supervision requires an explicit Open MPI launcher path"
        )
    _require_single_node_allocation(child_environment)
    if not _supports_linux_pidfds():
        raise RobertConfigError(
            "MPI callback supervision requires Linux /proc and pidfd signal support"
        )
    launcher = _validated_openmpi_launcher(mpi_launcher, child_environment)
    marker_parent = fallback_root / _MPI_MARKER_DIRECTORY
    marker_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker_dir = marker_parent / uuid.uuid4().hex
    marker_dir.mkdir(mode=0o700)
    child_environment[CALLBACK_FAILURE_MARKER_DIR_ENV] = str(marker_dir)
    supervised_command = [str(launcher), "-n", str(rank_count), *map(str, command)]
    process = subprocess.Popen(
        supervised_command,
        env=child_environment,
        start_new_session=True,
    )
    return _supervise_mpi_process(
        process,
        marker_dir=marker_dir,
        fallback_root=fallback_root,
        rank_count=rank_count,
        poll_interval_seconds=poll_interval_seconds,
        terminate_timeout_seconds=terminate_timeout_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
    )


def _supervise_serial_command(
    command: Sequence[str],
    *,
    fallback_root: Path,
    child_environment: dict[str, str],
    poll_interval_seconds: float,
    terminate_timeout_seconds: float,
) -> int:
    with tempfile.TemporaryDirectory(prefix="robert-multinest-supervisor-") as temp_dir:
        marker_path = Path(temp_dir) / "callback-failure.json"
        child_environment[CALLBACK_FAILURE_MARKER_ENV] = str(marker_path)
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


def _require_single_node_allocation(environment: Mapping[str, str]) -> None:
    value = environment.get("SLURM_NNODES") or environment.get("SLURM_JOB_NUM_NODES")
    if value is None:
        raise RobertConfigError(
            "MPI callback supervision requires a one-node Slurm allocation"
        )
    try:
        node_count = int(value)
    except ValueError as exc:
        raise RobertConfigError("Slurm node count must be an integer") from exc
    if node_count != 1:
        raise RobertConfigError(
            "MPI callback supervision supports one-node allocations only"
        )


def _validated_openmpi_launcher(
    launcher: str | Path, environment: Mapping[str, str]
) -> Path:
    raw = str(launcher)
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve() if candidate.is_file() else None
    if resolved is None:
        found = shutil.which(raw)
        resolved = Path(found).resolve() if found else None
    if resolved is None or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RobertConfigError(f"MPI launcher is not an executable file: {raw}")
    try:
        version = subprocess.run(
            [str(resolved), "--version"],
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RobertConfigError(f"could not validate Open MPI launcher: {raw}") from exc
    output = f"{version.stdout}\n{version.stderr}".lower()
    if version.returncode != 0 or "open mpi" not in output:
        raise RobertConfigError(
            f"MPI launcher is not a working Open MPI executable: {raw}"
        )
    return resolved


def _supervise_mpi_process(
    process: subprocess.Popen[bytes],
    *,
    marker_dir: Path,
    fallback_root: Path,
    rank_count: int,
    poll_interval_seconds: float,
    terminate_timeout_seconds: float,
    startup_timeout_seconds: float,
) -> int:
    started = time.monotonic()
    launcher_start_time = _linux_process_start_time(process.pid)
    missing_rank_since: dict[int, float] = {}
    try:
        while True:
            ready, ready_error = _read_rank_records(marker_dir, "ready")
            complete, complete_error = _read_rank_records(marker_dir, "complete")
            failures, failure_error = _read_rank_records(marker_dir, "failure")
            protocol_error = ready_error or complete_error or failure_error
            if protocol_error is not None:
                ready = _valid_rank_records(ready, rank_count)[0]
                cleanup_verified = _stop_mpi_world(
                    process,
                    ready,
                    rank_count,
                    launcher_start_time,
                    terminate_timeout_seconds,
                )
                _record_mpi_supervisor_failure(
                    fallback_root,
                    marker_dir,
                    f"invalid callback supervision marker: {protocol_error}",
                    cleanup_verified=cleanup_verified,
                    return_code=process.returncode,
                )
                return 1

            valid_ready, ready_problem = _valid_rank_records(ready, rank_count)
            if ready_problem is not None:
                cleanup_verified = _stop_mpi_world(
                    process,
                    valid_ready,
                    rank_count,
                    launcher_start_time,
                    terminate_timeout_seconds,
                )
                _record_mpi_supervisor_failure(
                    fallback_root,
                    marker_dir,
                    ready_problem,
                    cleanup_verified=cleanup_verified,
                    return_code=process.returncode,
                )
                return 1

            valid_complete, complete_problem = _valid_rank_records(
                complete, rank_count
            )
            if complete_problem is not None:
                cleanup_verified = _stop_mpi_world(
                    process,
                    valid_ready,
                    rank_count,
                    launcher_start_time,
                    terminate_timeout_seconds,
                )
                _record_mpi_supervisor_failure(
                    fallback_root,
                    marker_dir,
                    complete_problem,
                    cleanup_verified=cleanup_verified,
                    return_code=process.returncode,
                )
                return 1

            valid_failures, failure_problem = _valid_rank_records(
                failures, rank_count
            )
            if failure_problem is not None:
                cleanup_verified = _stop_mpi_world(
                    process,
                    valid_ready,
                    rank_count,
                    launcher_start_time,
                    terminate_timeout_seconds,
                )
                _record_mpi_supervisor_failure(
                    fallback_root,
                    marker_dir,
                    failure_problem,
                    cleanup_verified=cleanup_verified,
                    return_code=process.returncode,
                )
                return 1

            if valid_failures:
                cleanup_verified = _stop_mpi_world(
                    process,
                    valid_ready,
                    rank_count,
                    launcher_start_time,
                    terminate_timeout_seconds,
                )
                all_failures, _ = _read_rank_records(marker_dir, "failure")
                _record_mpi_callback_failure(
                    all_failures or valid_failures,
                    fallback_root,
                    marker_dir,
                    cleanup_verified=cleanup_verified,
                    expected_ranks=rank_count,
                )
                return 1

            return_code = process.poll()
            if return_code is not None:
                if return_code == 0 and len(valid_ready) == rank_count and len(
                    valid_complete
                ) == rank_count:
                    shutil.rmtree(marker_dir)
                    _remove_empty_marker_parent(marker_dir.parent)
                    return 0
                reason = (
                    "MPI child exited before every rank reported completion"
                    if return_code == 0
                    else f"Open MPI child exited with status {return_code}"
                )
                _record_mpi_supervisor_failure(
                    fallback_root,
                    marker_dir,
                    reason,
                    cleanup_verified=(
                        len(valid_ready) == rank_count
                        and _all_recorded_ranks_stopped(valid_ready)
                    ),
                    return_code=return_code,
                )
                return int(return_code) if return_code != 0 else 1

            elapsed = time.monotonic() - started
            if len(valid_ready) != rank_count and elapsed >= startup_timeout_seconds:
                cleanup_verified = _stop_mpi_world(
                    process,
                    valid_ready,
                    rank_count,
                    launcher_start_time,
                    terminate_timeout_seconds,
                )
                _record_mpi_supervisor_failure(
                    fallback_root,
                    marker_dir,
                    "MPI startup timed out before all ranks reported readiness",
                    cleanup_verified=cleanup_verified,
                    return_code=process.returncode,
                )
                return 1

            # Every rank writes readiness and then enters an MPI barrier before
            # calling MultiNest. A rank that exits without a completion record
            # is therefore a failed world, not a normal early return.
            if len(valid_ready) == rank_count and len(valid_complete) < rank_count:
                now = time.monotonic()
                completed_ranks = {int(record["rank"]) for record in valid_complete}
                for record in valid_ready:
                    rank = int(record["rank"])
                    if rank in completed_ranks:
                        continue
                    if not _rank_process_matches(record):
                        missing_rank_since.setdefault(rank, now)
                    else:
                        missing_rank_since.pop(rank, None)
                expired = [
                    rank
                    for rank, missing_since in missing_rank_since.items()
                    if now - missing_since >= 1.0
                ]
                if expired:
                    cleanup_verified = _stop_mpi_world(
                        process,
                        valid_ready,
                        rank_count,
                        launcher_start_time,
                        terminate_timeout_seconds,
                    )
                    _record_mpi_supervisor_failure(
                        fallback_root,
                        marker_dir,
                        "MPI rank(s) exited before completion: "
                        + ", ".join(str(rank) for rank in sorted(expired)),
                        cleanup_verified=cleanup_verified,
                        return_code=process.returncode,
                    )
                    return 1
            time.sleep(poll_interval_seconds)
    except BaseException:
        if process.poll() is None:
            ready, _ = _read_rank_records(marker_dir, "ready")
            valid_ready, _ = _valid_rank_records(ready, rank_count)
            _stop_mpi_world(
                process,
                valid_ready,
                rank_count,
                launcher_start_time,
                terminate_timeout_seconds,
            )
        raise


def _read_rank_records(
    marker_dir: Path, record_type: str
) -> tuple[list[dict[str, object]], str | None]:
    records: list[dict[str, object]] = []
    for path in sorted(marker_dir.glob(f"{record_type}-rank-*-pid-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return records, f"cannot read {path.name}: {exc}"
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != _CALLBACK_MARKER_SCHEMA
            or value.get("record_type") != record_type
        ):
            return records, f"unsupported record in {path.name}"
        records.append(value)
    return records, None


def _valid_rank_records(
    records: Sequence[Mapping[str, object]], expected_ranks: int
) -> tuple[list[dict[str, object]], str | None]:
    valid: list[dict[str, object]] = []
    seen: set[int] = set()
    local_host = socket.gethostname()
    for raw in records:
        record = dict(raw)
        try:
            rank = int(record["rank"])
            pid = int(record["pid"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return valid, "rank record has invalid rank or PID"
        if isinstance(record.get("rank"), bool) or isinstance(record.get("pid"), bool):
            return valid, "rank record has invalid rank or PID"
        if rank < 0 or rank >= expected_ranks or pid <= 0:
            return valid, "rank record is outside the configured MPI world"
        if rank in seen:
            return valid, f"duplicate MPI rank record for rank {rank}"
        if record.get("hostname") != local_host:
            return valid, "MPI callback supervision found a rank on another host"
        if not isinstance(record.get("process_start_time"), str):
            return valid, "rank record has no Linux process identity"
        try:
            int(record["process_group_id"])
            int(record["session_id"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return valid, "rank record has invalid process group identity"
        seen.add(rank)
        valid.append(record)
    return valid, None


def _stop_mpi_world(
    process: subprocess.Popen[bytes],
    ready_records: Sequence[Mapping[str, object]],
    expected_ranks: int,
    launcher_start_time: str | None,
    timeout: float,
) -> bool:
    """Stop the launcher group, then verify and stop recorded local ranks."""

    _signal_launcher_group(process, launcher_start_time, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal_launcher_group(process, launcher_start_time, signal.SIGKILL)
        process.wait()

    local_records = [
        record
        for record in ready_records
        if record.get("hostname") == socket.gethostname()
    ]
    cleanup_verified = (
        len(local_records) == len(ready_records) == expected_ranks
    )
    for record in local_records:
        if not _signal_recorded_rank(record, signal.SIGTERM):
            cleanup_verified = False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(not _rank_process_matches(record) for record in local_records):
            break
        time.sleep(0.05)

    for record in local_records:
        if not _signal_recorded_rank(record, signal.SIGKILL):
            cleanup_verified = False

    deadline = time.monotonic() + max(timeout, 0.1)
    while time.monotonic() < deadline:
        if all(not _rank_process_matches(record) for record in local_records):
            break
        time.sleep(0.05)
    if any(_rank_process_matches(record) for record in local_records):
        cleanup_verified = False
    return cleanup_verified


def _signal_launcher_group(
    process: subprocess.Popen[bytes], start_time: str | None, sig: int
) -> None:
    if (
        start_time is None
        or process.poll() is not None
        or _linux_process_start_time(process.pid) != start_time
    ):
        return
    try:
        os.killpg(process.pid, sig)
    except OSError:
        pass


def _signal_recorded_rank(record: Mapping[str, object], sig: int) -> bool:
    if record.get("hostname") != socket.gethostname():
        return False
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    start_time = str(record.get("process_start_time"))
    operations = _pidfd_operations()
    if operations is None:
        return False
    open_pidfd, send_signal = operations
    try:
        pidfd = open_pidfd(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    try:
        current = _linux_process_info(pid)
        if (
            current is None
            or current[1] == "Z"
            or current[0] != start_time
        ):
            return True
        send_signal(pidfd, sig, None, 0)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False
    finally:
        os.close(pidfd)


def _rank_process_matches(record: Mapping[str, object]) -> bool:
    """Check PID start identity before any rank signal or cleanup claim."""

    if record.get("hostname") != socket.gethostname():
        return False
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    current = _linux_process_info(pid)
    return (
        current is not None
        and current[1] != "Z"
        and current[0] == str(record.get("process_start_time"))
    )


def _all_recorded_ranks_stopped(records: Sequence[Mapping[str, object]]) -> bool:
    return all(not _rank_process_matches(record) for record in records)


def _record_mpi_callback_failure(
    markers: Sequence[Mapping[str, object]],
    fallback_root: Path,
    marker_dir: Path,
    *,
    cleanup_verified: bool,
    expected_ranks: int,
) -> None:
    output_root = fallback_root
    status_base = _read_current_status(output_root)
    attempt_id = str(
        status_base.get("attempt_id") or f"callback-{os.getpid()}-{time.time_ns()}"
    )
    failures = [dict(marker) for marker in markers]
    failures.sort(key=lambda marker: int(marker.get("rank", -1)))
    first = failures[0]
    ranks = sorted({int(marker.get("rank", -1)) for marker in failures})
    messages = [
        f"rank {marker.get('rank')}: {marker.get('exception_type', 'CallbackError')}: "
        f"{marker.get('message', 'callback failed')}"
        for marker in failures
    ]
    elapsed_values = []
    for marker in failures:
        try:
            elapsed_values.append(max(float(marker.get("elapsed_seconds", 0.0)), 0.0))
        except (TypeError, ValueError, OverflowError):
            pass
    failure = {
        **status_base,
        "state": "failed",
        "elapsed_seconds": max(elapsed_values, default=0.0),
        "exception_type": (
            str(first.get("exception_type", "CallbackError"))
            if len(failures) == 1
            else "MultiRankCallbackError"
        ),
        "message": "; ".join(messages),
        "callback_failure": True,
        "callback_rank": int(first.get("rank", -1)),
        "callback_failure_ranks": ranks,
        "callback_failures": failures,
        "callback_supervisor_cleanup_verified": cleanup_verified,
        "callback_marker_directory": str(marker_dir),
        "mpi_nprocs": expected_ranks,
        "multinest_resume_safe": False,
        "partial_chains_preserved": _has_chain_files(output_root),
    }
    if first.get("traceback"):
        failure["callback_traceback"] = str(first["traceback"])
    write_retrieval_status(output_root, failure)
    append_retrieval_attempt_event(
        output_root,
        {**failure, "event": "callback_failed", "attempt_id": attempt_id},
    )
    retained = output_root / f"multinest_callback_failure_{attempt_id}.json"
    _write_json_atomically(
        retained,
        {
            "schema_version": _CALLBACK_MARKER_SCHEMA,
            "attempt_id": attempt_id,
            "expected_ranks": expected_ranks,
            "cleanup_verified": cleanup_verified,
            "marker_directory": str(marker_dir),
            "failures": failures,
        },
    )


def _record_mpi_supervisor_failure(
    output_root: Path,
    marker_dir: Path,
    message: str,
    *,
    cleanup_verified: bool,
    return_code: int | None,
) -> None:
    status_base = _read_current_status(output_root)
    if status_base.get("state") not in {None, "running"}:
        # The child already wrote a terminal status. Keep its original error.
        return
    attempt_id = str(
        status_base.get("attempt_id") or f"mpi-supervisor-{os.getpid()}-{time.time_ns()}"
    )
    failure = {
        **status_base,
        "state": "failed",
        "exception_type": "MultiNestSupervisorError",
        "message": message,
        "supervisor_failure": True,
        "supervisor_return_code": return_code,
        "callback_supervisor_cleanup_verified": cleanup_verified,
        "callback_marker_directory": str(marker_dir),
        "multinest_resume_safe": False,
        "partial_chains_preserved": (output_root / "chains").is_dir(),
    }
    write_retrieval_status(output_root, failure)
    append_retrieval_attempt_event(
        output_root,
        {**failure, "event": "mpi_supervisor_failed", "attempt_id": attempt_id},
    )


def _remove_empty_marker_parent(marker_parent: Path) -> None:
    try:
        marker_parent.rmdir()
    except OSError:
        pass


def _has_chain_files(output_root: Path) -> bool:
    chains = output_root / "chains"
    return chains.is_dir() and any(chains.iterdir())


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
    "CALLBACK_FAILURE_MARKER_DIR_ENV",
    "CALLBACK_SUPERVISED_ENV",
    "supervise_retrieval_command",
    "write_callback_failure_marker",
    "write_callback_rank_record",
]
