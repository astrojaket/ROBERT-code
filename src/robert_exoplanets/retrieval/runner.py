"""Retrieval orchestration and legacy stub workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import RobertConfigError, RobertDataError
from robert_exoplanets.instruments import Observation

from .config import RetrievalConfig
from .model import EmissionModel
from .manifest import (
    RUN_MANIFEST_FILENAME,
    RunManifest,
    build_run_manifest,
    read_run_manifest,
    write_run_manifest,
)
from .optimal_estimation import run_optimal_estimation
from .problem import RetrievalProblem
from .results import RetrievalResult, build_retrieval_result, write_retrieval_result
from .samplers import run_ultranest


@dataclass(frozen=True)
class StubRetrievalResult:
    """Result object returned by the stub retrieval runner."""

    config: RetrievalConfig
    model_name: str
    best_fit_parameters: dict[str, float]
    model_flux: NDArray[np.float64]
    log_likelihood: float
    converged: bool
    message: str


def run_stub_retrieval(
    observation: Observation,
    config: RetrievalConfig,
    model: EmissionModel | None = None,
) -> StubRetrievalResult:
    """Run a deterministic placeholder retrieval.

    This function wires together the intended end-to-end retrieval flow while
    avoiding real physics and sampling. It computes a weighted mean baseline and
    evaluates the placeholder emission model at that baseline.
    """

    observation.validate()
    active_model = model or EmissionModel()

    weights = 1.0 / np.square(observation.uncertainty)
    baseline = float(np.average(observation.flux, weights=weights))
    best_fit_parameters = {
        "baseline": baseline,
        "slope": 0.0,
    }
    model_flux = active_model.evaluate(observation.wavelength, best_fit_parameters)
    residual = observation.flux - model_flux
    log_likelihood = float(-0.5 * np.sum(np.square(residual / observation.uncertainty)))

    return StubRetrievalResult(
        config=config,
        model_name=active_model.name,
        best_fit_parameters=best_fit_parameters,
        model_flux=model_flux,
        log_likelihood=log_likelihood,
        converged=False,
        message="Stub retrieval completed; no physical sampler has been run.",
    )


def run_retrieval(
    problem: RetrievalProblem,
    *,
    method: str = "optimal_estimation",
    output_dir: str | Path | None = None,
    seed: int | None = None,
    **kwargs: object,
) -> RetrievalResult:
    """Run and serialize a retrieval with a manifest written before inference."""

    normalized = method.strip().lower().replace("-", "_")
    if normalized in {"optimal_estimation", "oe"}:
        normalized = "optimal_estimation"
    elif normalized in {"ultranest", "nested", "nested_sampling"}:
        normalized = "ultranest"
    else:
        raise ValueError(f"unsupported retrieval method: {method}")
    if output_dir is None:
        raise ValueError("output_dir is required for reproducible retrieval runs")

    output_path = Path(output_dir).expanduser()
    settings = {"method": normalized, **kwargs}
    if normalized == "ultranest":
        settings["seed"] = seed
    current_manifest = build_run_manifest(
        problem,
        method=normalized,
        settings=settings,
        random_seed=seed if normalized == "ultranest" else None,
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
                    resume=str(kwargs.get("resume", "overwrite")),
                    is_ultranest=normalized == "ultranest",
                )
            except (RobertConfigError, RobertDataError, OSError) as exc:
                manifest_error = str(exc)
        if communicator is not None:
            manifest_error = communicator.bcast(manifest_error, root=0)
        if manifest_error is not None:
            raise RobertConfigError(manifest_error)
        if communicator is not None:
            communicator.Barrier()

        if normalized == "optimal_estimation":
            inference_result = run_optimal_estimation(problem, **kwargs)
        else:
            inference_result = run_ultranest(problem, output_dir=output_path, seed=seed, **kwargs)
        result = build_retrieval_result(inference_result, manifest=manifest, output_dir=output_path)
        if rank == 0:
            write_retrieval_result(result)
        if communicator is not None:
            communicator.Barrier()
        return result
    finally:
        if rank == 0 and lock is not None:
            _release_run_directory_lock(lock)


def _prepare_manifest(
    current: RunManifest,
    output_path: Path,
    *,
    resume: str,
    is_ultranest: bool,
) -> RunManifest:
    """Preserve the original manifest and journal subsequent attempts."""

    manifest_path = output_path / RUN_MANIFEST_FILENAME
    resume_existing = is_ultranest and resume in {"resume", "resume-similar"}
    if resume_existing and manifest_path.exists():
        original = read_run_manifest(manifest_path)
        _validate_resume_compatibility(original, current)
        active = original
    else:
        write_run_manifest(current, output_path)
        active = current
    _write_attempt_manifest(current, output_path, original_config_hash=active.config_hash)
    return active


def _validate_resume_compatibility(original: RunManifest, current: RunManifest) -> None:
    fields = (
        "problem_name",
        "method",
        "parameter_names",
        "parameter_priors",
        "likelihood",
        "problem_metadata",
        "opacity_identifiers",
        "random_seed",
    )
    changed = [name for name in fields if getattr(original, name) != getattr(current, name)]
    original_floor = original.settings.get("invalid_loglike_floor")
    current_floor = current.settings.get("invalid_loglike_floor")
    if original_floor != current_floor:
        changed.append("invalid_loglike_floor")
    if changed:
        raise RobertConfigError(
            "cannot resume because the scientific run definition changed: "
            + ", ".join(changed)
            + ". Use a new output directory or resume='overwrite'."
        )


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
        raise RobertDataError(f"failed to write retrieval attempt manifest: {path}") from exc
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


__all__ = ["RetrievalResult", "StubRetrievalResult", "run_retrieval", "run_stub_retrieval"]
