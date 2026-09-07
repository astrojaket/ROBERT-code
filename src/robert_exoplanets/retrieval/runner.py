"""Retrieval orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import IO, Any, cast

from robert_exoplanets.core import RobertConfigError, RobertDataError

from .manifest import (
    RUN_MANIFEST_FILENAME,
    RunManifest,
    build_run_manifest,
    normalize_retrieval_method,
    normalize_multinest_resume,
    read_run_manifest,
    validate_resume_compatibility,
    write_run_manifest,
)
from .optimal_estimation import run_optimal_estimation
from .protocols import OptimalEstimationProblem, SamplerRetrievalProblem
from .predictions import capture_best_fit_prediction
from .results import (
    InferenceResult,
    RetrievalResult,
    build_retrieval_result,
    write_retrieval_result,
)
from .samplers import run_multinest


def run_retrieval(
    problem: SamplerRetrievalProblem,
    *,
    method: str = "multinest",
    output_dir: str | Path | None = None,
    seed: int | None = None,
    **kwargs: object,
) -> RetrievalResult:
    """Run and serialize a retrieval with a manifest written before inference."""

    normalized = normalize_retrieval_method(method)
    if normalized not in {"optimal_estimation", "multinest"}:
        raise ValueError(f"unsupported retrieval method: {method}")
    if output_dir is None:
        raise ValueError("output_dir is required for reproducible retrieval runs")

    output_path = Path(output_dir).expanduser()
    settings = {"method": normalized, **kwargs}
    if normalized == "multinest":
        settings["seed"] = seed
    current_manifest = build_run_manifest(
        problem,
        method=normalized,
        settings=settings,
        random_seed=seed if normalized == "multinest" else None,
    )
    rank, communicator = _mpi_context()
    lock: IO[str] | None = None
    lock_error: str | None = None
    if rank == 0:
        try:
            lock = _acquire_run_directory_lock(output_path)
        except (OSError, RobertConfigError) as exc:
            lock_error = str(exc)
    if communicator is not None:
        lock_error = communicator.bcast(lock_error, root=0)
    if lock_error is not None:
        raise RobertConfigError(lock_error)

    manifest = current_manifest
    try:
        manifest_error: str | None = None
        if rank == 0:
            try:
                manifest = _prepare_manifest(
                    current_manifest,
                    output_path,
                    resume=kwargs.get("resume", True),
                    is_nested=normalized == "multinest",
                )
            except (RobertConfigError, RobertDataError, OSError) as exc:
                manifest_error = str(exc)
        if communicator is not None:
            manifest_error = communicator.bcast(manifest_error, root=0)
        if manifest_error is not None:
            raise RobertConfigError(manifest_error)
        if communicator is not None:
            communicator.Barrier()

        inference_started = time.monotonic()
        inference_result: InferenceResult
        if normalized == "optimal_estimation":
            if not callable(getattr(problem, "gaussian_inputs_from_vector", None)):
                raise RobertConfigError(
                    "optimal estimation requires a problem implementing the "
                    "explicit gaussian_inputs_from_vector capability"
                )
            inference_result = run_optimal_estimation(
                cast(OptimalEstimationProblem, problem),
                **cast(Any, kwargs),
            )
        else:
            inference_result = run_multinest(
                problem,
                output_dir=output_path,
                seed=seed,
                **cast(Any, kwargs),
            )
        inference_elapsed = max(time.monotonic() - inference_started, 0.0)
        result = build_retrieval_result(
            inference_result, manifest=manifest, output_dir=output_path
        )
        result = replace(
            result,
            metadata={
                **dict(result.metadata),
                "inference_elapsed_seconds": f"{inference_elapsed:.6f}",
            },
        )
        write_error: str | None = None
        if rank == 0:
            try:
                artifact = capture_best_fit_prediction(
                    problem,
                    result.best_fit_parameters,
                    provenance={
                        "config_hash": manifest.config_hash,
                        "method": normalized,
                        "robert_version": manifest.robert_version,
                    },
                )
                write_retrieval_result(result, prediction_artifact=artifact)
            except Exception as exc:
                # Every rank must leave finalization together on a write failure.
                write_error = f"failed to write retrieval products: {type(exc).__name__}: {exc}"
        if communicator is not None:
            write_error = communicator.bcast(write_error, root=0)
        if write_error is not None:
            raise RobertDataError(write_error)
        return result
    finally:
        if rank == 0 and lock is not None:
            _release_run_directory_lock(lock)


def _prepare_manifest(
    current: RunManifest,
    output_path: Path,
    *,
    resume: object,
    is_nested: bool,
) -> RunManifest:
    """Preserve the original manifest and journal subsequent attempts."""

    manifest_path = output_path / RUN_MANIFEST_FILENAME
    resume_existing = False
    if is_nested:
        resume_existing = normalize_multinest_resume(resume)
    if resume_existing and manifest_path.exists():
        original = read_run_manifest(manifest_path)
        validate_resume_compatibility(original, current)
        active = original
    else:
        write_run_manifest(current, output_path)
        active = current
    _write_attempt_manifest(
        current, output_path, original_config_hash=active.config_hash
    )
    return active


def _write_attempt_manifest(
    manifest: RunManifest,
    output_path: Path,
    *,
    original_config_hash: str,
) -> Path:
    attempts = output_path / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = attempts / f"attempt-{stamp}-{os.getpid()}.json"
    payload = {
        **manifest.to_mapping(),
        "original_config_hash": original_config_hash,
    }
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RobertDataError(
            f"failed to write retrieval attempt manifest: {path}"
        ) from exc
    return path


def _acquire_run_directory_lock(output_path: Path) -> IO[str]:
    """Acquire a non-blocking process lock for one retrieval directory."""

    import fcntl

    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / ".robert-run.lock"
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.seek(0)
        owner = stream.read().strip() or "owner details unavailable"
        stream.close()
        raise RobertConfigError(
            f"retrieval directory is already in use: {output_path} ({owner})"
        ) from exc
    stream.seek(0)
    stream.truncate()
    stream.write(
        json.dumps(
            {
                "hostname": os.uname().nodename,
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
    )
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def _release_run_directory_lock(stream: IO[str]) -> None:
    import fcntl

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _mpi_context():
    """Return rank and communicator only for an active multi-process MPI run."""

    if not any(
        name in os.environ
        for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "SLURM_NTASKS")
    ):
        return 0, None
    try:
        from mpi4py import MPI
    except Exception:
        return 0, None
    communicator = MPI.COMM_WORLD
    if int(communicator.Get_size()) <= 1:
        return 0, None
    return int(communicator.Get_rank()), communicator


__all__ = ["RetrievalResult", "run_retrieval"]
