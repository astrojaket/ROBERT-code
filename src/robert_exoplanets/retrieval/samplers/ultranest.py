"""UltraNest adapter for ROBERT retrieval problems."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np

from robert_exoplanets.core import RobertConfigError

from ..problem import RetrievalProblem
from ..status import append_retrieval_attempt_event, write_retrieval_status
from .base import (
    NestedSamplerResult,
    validate_mpi_world_size as _validate_mpi_world_size,
)


def run_ultranest(
    problem: RetrievalProblem,
    *,
    output_dir: str | Path,
    min_num_live_points: int = 400,
    max_ncalls: int | None = None,
    dlogz: float = 0.5,
    resume: str = "resume",
    show_status: bool = True,
    mpi_nprocs: int | None = None,
    seed: int | None = None,
    invalid_loglike_floor: float = -1.0e100,
    **run_kwargs: Any,
) -> NestedSamplerResult:
    """Run UltraNest for a ROBERT retrieval problem.

    MPI is controlled by launching the Python process with `mpiexec` or
    `mpirun`, for example `mpirun -np 3 python script.py`. When supplied,
    `mpi_nprocs` is checked against `MPI.COMM_WORLD` before UltraNest opens a
    checkpoint, then recorded as metadata.
    """

    _validate_mpi_world_size(mpi_nprocs)

    try:
        import ultranest
    except ImportError as exc:
        raise RobertConfigError(
            "UltraNest is not installed. Install with `python -m pip install ultranest mpi4py`."
        ) from exc

    if seed is not None and int(seed) < 0:
        raise RobertConfigError("UltraNest seed must be non-negative")
    invalid_floor = float(invalid_loglike_floor)
    if not np.isfinite(invalid_floor) or invalid_floor >= 0.0:
        raise RobertConfigError("UltraNest invalid_loglike_floor must be finite and negative")

    def finite_loglike(vector):
        value = problem.log_likelihood_from_vector(vector)
        return invalid_floor if not np.isfinite(value) else float(value)

    log_dir = Path(output_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    random_state = np.random.get_state()
    is_primary = _is_primary_process()
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    attempt_id = _attempt_id()
    ncall_start = 0
    status_base: dict[str, object] = {
        "attempt_id": attempt_id,
        "state": "starting",
        "started_at_utc": started_at,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "mpi_nprocs": mpi_nprocs,
        "resume": resume,
        "min_num_live_points": int(min_num_live_points),
        "max_ncalls": max_ncalls,
        "dlogz": float(dlogz),
        "seed": seed,
    }
    try:
        if seed is not None:
            np.random.seed(int(seed))
        sampler = ultranest.ReactiveNestedSampler(
            list(problem.parameter_names),
            finite_loglike,
            transform=problem.prior_transform,
            log_dir=str(log_dir),
            resume=resume,
        )
        ncall_start = int(getattr(sampler, "ncall", 0))
        status_base.update({"state": "running", "ncall_start": ncall_start})
        if is_primary:
            append_retrieval_attempt_event(
                log_dir,
                {**status_base, "event": "started", "ncall": ncall_start},
            )
            write_retrieval_status(
                log_dir,
                {
                    **status_base,
                    "ncall": ncall_start,
                    "ncall_this_attempt": 0,
                    "elapsed_seconds": 0.0,
                },
            )
        requested_callback = run_kwargs.pop("viz_callback", "auto")
        display_callback = _display_callback(
            requested_callback,
            show_status=bool(show_status),
        )

        def status_callback(points, info, region, transformLayer, region_fresh=False):
            if is_primary:
                ncall = int(info.get("ncall", getattr(sampler, "ncall", ncall_start)))
                elapsed = max(time.monotonic() - started_monotonic, 0.0)
                attempt_calls = max(ncall - ncall_start, 0)
                live_loglike = np.asarray(points.get("logl", ()), dtype=float)
                best_live = float(np.max(live_loglike)) if live_loglike.size else None
                write_retrieval_status(
                    log_dir,
                    {
                        **status_base,
                        "state": "running",
                        "iteration": info.get("it"),
                        "ncall": ncall,
                        "ncall_this_attempt": attempt_calls,
                        "calls_per_second": attempt_calls / elapsed if elapsed else None,
                        "elapsed_seconds": elapsed,
                        "best_live_log_likelihood": best_live,
                        "log_evidence": info.get("logz"),
                        "remaining_log_evidence": info.get("logz_remain"),
                        "log_volume": info.get("logvol"),
                    },
                )
            if display_callback is not None:
                display_callback(points, info, region, transformLayer, region_fresh)

        result = sampler.run(
            min_num_live_points=int(min_num_live_points),
            max_ncalls=max_ncalls,
            dlogz=float(dlogz),
            show_status=bool(show_status),
            viz_callback=status_callback,
            **run_kwargs,
        )
        elapsed = max(time.monotonic() - started_monotonic, 0.0)
        ncall = int(result.get("ncall", getattr(sampler, "ncall", ncall_start)))
        attempt_calls = max(ncall - ncall_start, 0)
        reached_call_limit = max_ncalls is not None and ncall >= int(max_ncalls)
        state = "call_limit_reached" if reached_call_limit else "converged"
        if is_primary:
            final_status = {
                **status_base,
                "state": state,
                "ncall": ncall,
                "ncall_this_attempt": attempt_calls,
                "calls_per_second": attempt_calls / elapsed if elapsed else None,
                "elapsed_seconds": elapsed,
                "converged": not reached_call_limit,
                "message": (
                    "maximum likelihood-call limit reached"
                    if reached_call_limit
                    else "converged"
                ),
            }
            write_retrieval_status(log_dir, final_status)
            append_retrieval_attempt_event(log_dir, {**final_status, "event": "finished"})
    except BaseException as exc:
        if is_primary:
            elapsed = max(time.monotonic() - started_monotonic, 0.0)
            ncall = int(getattr(locals().get("sampler"), "ncall", ncall_start))
            failure_status = {
                **status_base,
                "state": (
                    "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
                ),
                "ncall": ncall,
                "ncall_this_attempt": max(ncall - ncall_start, 0),
                "elapsed_seconds": elapsed,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            write_retrieval_status(log_dir, failure_status)
            append_retrieval_attempt_event(log_dir, {**failure_status, "event": "failed"})
        raise
    finally:
        np.random.set_state(random_state)
    return _result_from_ultranest(
        problem,
        result,
        log_dir=log_dir,
        mpi_nprocs=mpi_nprocs,
        max_ncalls=max_ncalls,
        seed=seed,
        invalid_loglike_floor=invalid_floor,
        attempt_id=attempt_id,
        resume=resume,
        ncall_start=ncall_start,
        elapsed_seconds=max(time.monotonic() - started_monotonic, 0.0),
    )


def _result_from_ultranest(
    problem: RetrievalProblem,
    result: dict[str, Any],
    *,
    log_dir: Path,
    mpi_nprocs: int | None,
    max_ncalls: int | None = None,
    seed: int | None = None,
    invalid_loglike_floor: float = -1.0e100,
    attempt_id: str | None = None,
    resume: str | None = None,
    ncall_start: int = 0,
    elapsed_seconds: float | None = None,
) -> NestedSamplerResult:
    weighted = result.get("weighted_samples", {})
    points = np.asarray(weighted.get("points", result.get("samples", np.empty((0, problem.ndim)))), dtype=float)
    weights = weighted.get("weights")
    weights_array = None if weights is None else np.asarray(weights, dtype=float)
    loglike = weighted.get("logl", result.get("maximum_likelihood", {}).get("logl", np.empty(points.shape[0])))
    loglike_array = np.asarray(loglike, dtype=float)
    if loglike_array.ndim == 0:
        loglike_array = np.full(points.shape[0], float(loglike_array), dtype=float)
    best = result.get("maximum_likelihood", {})
    best_point = best.get("point")
    if best_point is not None:
        best_fit = problem.parameter_mapping(np.asarray(best_point, dtype=float))
    elif points.size:
        best_fit = problem.parameter_mapping(points[int(np.argmax(loglike_array))])
    else:
        best_fit = {}
    ncall = None if result.get("ncall") is None else int(result["ncall"])
    reached_call_limit = max_ncalls is not None and ncall is not None and ncall >= int(max_ncalls)
    converged = bool(points.size) and not reached_call_limit
    message = "converged" if converged else "maximum likelihood-call limit reached"
    ncall_this_attempt = None if ncall is None else max(ncall - int(ncall_start), 0)
    return NestedSamplerResult(
        method="ultranest",
        parameter_names=problem.parameter_names,
        samples=points,
        log_likelihood=loglike_array,
        weights=weights_array,
        log_evidence=None if result.get("logz") is None else float(result["logz"]),
        log_evidence_error=None if result.get("logzerr") is None else float(result["logzerr"]),
        best_fit_parameters=best_fit,
        metadata={
            "problem": problem.name,
            "log_dir": str(log_dir),
            "mpi_nprocs": "" if mpi_nprocs is None else str(int(mpi_nprocs)),
            "ncall": "" if ncall is None else str(ncall),
            "seed": "" if seed is None else str(int(seed)),
            "invalid_loglike_floor": f"{invalid_loglike_floor:.17g}",
            "attempt_id": "" if attempt_id is None else attempt_id,
            "resume": "" if resume is None else str(resume),
            "ncall_start": str(int(ncall_start)),
            "ncall_this_attempt": (
                "" if ncall_this_attempt is None else str(ncall_this_attempt)
            ),
            "elapsed_seconds": (
                "" if elapsed_seconds is None else f"{float(elapsed_seconds):.6f}"
            ),
        },
        converged=converged,
        message=message,
    )


def _display_callback(callback: object, *, show_status: bool):
    if callback in (None, False):
        return None
    if callback == "auto":
        if not show_status:
            return None
        from ultranest.integrator import get_default_viz_callback

        return get_default_viz_callback()
    if not callable(callback):
        raise RobertConfigError("UltraNest viz_callback must be callable, 'auto', or None")
    return callback


def _is_primary_process() -> bool:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"):
        if name in os.environ:
            try:
                return int(os.environ[name]) == 0
            except ValueError:
                continue
    return True


def _attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    job = os.environ.get("SLURM_JOB_ID", "local")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    return f"{timestamp}-{job}-{restart}-{os.getpid()}"
