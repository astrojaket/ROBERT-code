"""Persistent status records for long-running retrievals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from robert_exoplanets.core import RobertDataError

RETRIEVAL_STATUS_FILENAME = "sampler_status.json"
RETRIEVAL_ATTEMPTS_FILENAME = "run_attempts.jsonl"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_retrieval_status(
    output_dir: str | Path,
    status: Mapping[str, object],
) -> Path:
    """Atomically replace the human- and machine-readable run status."""

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RETRIEVAL_STATUS_FILENAME
    temporary = directory / f".{RETRIEVAL_STATUS_FILENAME}.{os.getpid()}.tmp"
    payload = {**dict(status), "updated_at_utc": utc_now()}
    try:
        temporary.write_text(
            json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        raise RobertDataError(f"failed to write retrieval status: {path}") from exc
    return path


def append_retrieval_attempt_event(
    output_dir: str | Path,
    event: Mapping[str, object],
) -> Path:
    """Append and flush one immutable event to the run-attempt journal."""

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RETRIEVAL_ATTEMPTS_FILENAME
    payload = {**dict(event), "recorded_at_utc": utc_now()}
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_value(payload), sort_keys=True, allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise RobertDataError(f"failed to append retrieval attempt event: {path}") from exc
    return path


def load_retrieval_status(output_dir: str | Path) -> dict[str, object]:
    """Load the best available status for an active or completed retrieval."""

    directory = Path(output_dir).expanduser()
    status = _read_json(directory / RETRIEVAL_STATUS_FILENAME)
    result = _read_json(directory / "result.json")
    found_artifact = bool(status or result)
    if result:
        status.setdefault("state", "converged" if result.get("converged") else "completed")
        status.setdefault("converged", bool(result.get("converged")))
        status.setdefault("message", result.get("message"))
        metadata = result.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("ncall") not in (None, ""):
            status.setdefault("ncall", int(metadata["ncall"]))
    hdf5_ncall = _read_ultranest_ncall(directory)
    if hdf5_ncall is not None:
        found_artifact = True
        status["ncall_checkpointed"] = hdf5_ncall
        status.setdefault("ncall", hdf5_ncall)
    events = _read_json_lines(directory / RETRIEVAL_ATTEMPTS_FILENAME)
    found_artifact = found_artifact or bool(events)
    if not found_artifact:
        raise RobertDataError(f"no retrieval status found under {directory}")
    status["attempt_count"] = sum(event.get("event") == "started" for event in events)
    status["output_dir"] = str(directory.resolve())
    updated = status.get("updated_at_utc")
    if isinstance(updated, str):
        try:
            timestamp = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            status["seconds_since_update"] = max(
                0.0,
                (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds(),
            )
        except ValueError:
            pass
    return status


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for inspecting a run directory."""

    parser = argparse.ArgumentParser(description="Inspect a ROBERT retrieval run directory.")
    parser.add_argument("output_dir")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    status = load_retrieval_status(args.output_dir)
    if args.as_json:
        print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
        return 0
    print(f"Output:       {status['output_dir']}")
    print(f"State:        {status.get('state', 'unknown')}")
    print(f"Attempts:     {status.get('attempt_count', 0)}")
    print(f"Calls:        {status.get('ncall', 'unknown')}")
    if status.get("ncall_this_attempt") is not None:
        print(f"This attempt: {status['ncall_this_attempt']}")
    if status.get("calls_per_second") is not None:
        print(f"Throughput:   {float(status['calls_per_second']):.3g} calls/s")
    if status.get("elapsed_seconds") is not None:
        print(f"Elapsed:      {float(status['elapsed_seconds']):.1f} s")
    if status.get("updated_at_utc"):
        print(f"Updated:      {status['updated_at_utc']}")
    if status.get("message"):
        print(f"Message:      {status['message']}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobertDataError(f"failed to read retrieval status file: {path}") from exc
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [dict(value) for line in lines if isinstance((value := json.loads(line)), Mapping)]
    except (OSError, json.JSONDecodeError) as exc:
        raise RobertDataError(f"failed to read retrieval attempt journal: {path}") from exc


def _read_ultranest_ncall(directory: Path) -> int | None:
    candidates = [directory / "results" / "points.hdf5"]
    candidates.extend(sorted(directory.glob("run*/results/points.hdf5"), reverse=True))
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        return None
    try:
        import h5py

        with h5py.File(existing, "r") as store:
            value = store.attrs.get("ncalls")
            return None if value is None else int(value)
    except (ImportError, OSError, ValueError, TypeError):
        return None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return repr(value)
    return number if math.isfinite(number) else None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RETRIEVAL_ATTEMPTS_FILENAME",
    "RETRIEVAL_STATUS_FILENAME",
    "append_retrieval_attempt_event",
    "load_retrieval_status",
    "main",
    "utc_now",
    "write_retrieval_status",
]
