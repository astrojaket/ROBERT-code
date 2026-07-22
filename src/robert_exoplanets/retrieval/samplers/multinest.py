"""PyMultiNest adapter for ROBERT retrieval problems."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np

from robert_exoplanets.core import RobertConfigError

from ..problem import RetrievalProblem
from ..status import append_retrieval_attempt_event, write_retrieval_status
from .base import NestedSamplerResult, validate_mpi_world_size


MULTINEST_MAX_SEED = 30080


def run_multinest(
    problem: RetrievalProblem,
    *,
    output_dir: str | Path,
    n_live_points: int = 400,
    evidence_tolerance: float = 0.5,
    sampling_efficiency: float = 0.8,
    max_iter: int = 0,
    resume: bool = True,
    verbose: bool = True,
    mpi_nprocs: int | None = None,
    seed: int | None = None,
    invalid_loglike_floor: float = -1.0e100,
    importance_nested_sampling: bool = True,
    multimodal: bool = True,
    n_iter_before_update: int = 100,
    **run_kwargs: Any,
) -> NestedSamplerResult:
    """Run the compiled MultiNest library through PyMultiNest.

    ``max_iter=0`` means unlimited, matching MultiNest. MPI is selected by
    launching this function under ``mpirun``; ``mpi_nprocs`` verifies that all
    requested ranks joined the same communicator before shared files are opened.
    """

    validate_mpi_world_size(mpi_nprocs)
    live_points = _positive_integer(n_live_points, "n_live_points")
    update_interval = _positive_integer(n_iter_before_update, "n_iter_before_update")
    iteration_limit = int(max_iter)
    if isinstance(max_iter, bool) or iteration_limit < 0:
        raise RobertConfigError("max_iter must be a non-negative integer")
    tolerance = _positive_float(evidence_tolerance, "evidence_tolerance")
    efficiency = float(sampling_efficiency)
    if not np.isfinite(efficiency) or efficiency <= 0.0 or efficiency > 1.0:
        raise RobertConfigError("sampling_efficiency must be in (0, 1]")
    if seed is not None and not 0 <= int(seed) <= MULTINEST_MAX_SEED:
        raise RobertConfigError(
            f"MultiNest seed must lie in [0, {MULTINEST_MAX_SEED}] for the "
            "legacy Fortran random-number generator"
        )
    invalid_floor = float(invalid_loglike_floor)
    if not np.isfinite(invalid_floor) or invalid_floor >= 0.0:
        raise RobertConfigError("MultiNest invalid_loglike_floor must be finite and negative")

    try:
        import pymultinest
    except (ImportError, OSError, SystemExit) as exc:
        raise RobertConfigError(
            "PyMultiNest or its compiled MultiNest library is unavailable. "
            "Install both from conda-forge with `conda install multinest pymultinest`."
        ) from exc

    root = Path(output_dir).expanduser().resolve()
    chains = root / "chains"
    chains.mkdir(parents=True, exist_ok=True)
    # MultiNest retains a legacy fixed-length Fortran filename buffer. Running
    # from the output directory keeps the native prefix short even when the
    # user-selected absolute run path is long.
    basename = "chains/1-"
    absolute_basename = str(chains / "1-")
    primary = _is_primary_process()
    if primary:
        Path(f"{absolute_basename}params.json").write_text(
            json.dumps(list(problem.parameter_names), indent=2), encoding="utf-8"
        )
    _mpi_barrier()

    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    attempt_id = _attempt_id()
    status_base: dict[str, object] = {
        "attempt_id": attempt_id,
        "state": "running",
        "started_at_utc": started_at,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "mpi_nprocs": mpi_nprocs,
        "resume": bool(resume),
        "n_live_points": live_points,
        "max_iter": iteration_limit,
        "evidence_tolerance": tolerance,
        "sampling_efficiency": efficiency,
        "seed": seed,
    }
    if primary:
        append_retrieval_attempt_event(root, {**status_base, "event": "started"})
        write_retrieval_status(
            root, {**status_base, "elapsed_seconds": 0.0, "n_samples": 0}
        )

    def prior(cube, ndim, nparams) -> None:
        del nparams
        transformed = problem.prior_transform(
            np.fromiter((cube[index] for index in range(ndim)), dtype=float, count=ndim)
        )
        for index, value in enumerate(transformed):
            cube[index] = float(value)

    def loglike(cube, ndim, nparams, lnew) -> float:
        del nparams, lnew
        vector = np.fromiter(
            (cube[index] for index in range(ndim)), dtype=float, count=ndim
        )
        value = problem.log_likelihood_from_vector(vector)
        return invalid_floor if not np.isfinite(value) else float(value)

    requested_callback = run_kwargs.pop("dump_callback", None)
    if requested_callback is not None and not callable(requested_callback):
        raise RobertConfigError("MultiNest dump_callback must be callable or None")
    if "outputfiles_basename" in run_kwargs:
        raise RobertConfigError(
            "MultiNest outputfiles_basename is managed by ROBERT's output_dir"
        )

    def dump_callback(
        n_samples,
        n_live,
        n_params,
        live_points_array,
        posterior,
        parameter_constraints,
        max_loglike,
        log_evidence,
        log_evidence_error,
        context,
    ) -> None:
        if primary:
            write_retrieval_status(
                root,
                {
                    **status_base,
                    "elapsed_seconds": max(time.monotonic() - started_monotonic, 0.0),
                    "n_samples": int(n_samples),
                    "n_live": int(n_live),
                    "best_live_log_likelihood": float(max_loglike),
                    "log_evidence": float(log_evidence),
                    "log_evidence_error": float(log_evidence_error),
                },
            )
        if requested_callback is not None:
            requested_callback(
                n_samples,
                n_live,
                n_params,
                live_points_array,
                posterior,
                parameter_constraints,
                max_loglike,
                log_evidence,
                log_evidence_error,
                context,
            )

    try:
        with _working_directory(root):
            pymultinest.run(
                loglike,
                prior,
                problem.ndim,
                outputfiles_basename=basename,
                resume=bool(resume),
                verbose=bool(verbose),
                n_live_points=live_points,
                evidence_tolerance=tolerance,
                sampling_efficiency=efficiency,
                max_iter=iteration_limit,
                seed=-1 if seed is None else int(seed),
                log_zero=invalid_floor,
                importance_nested_sampling=bool(importance_nested_sampling),
                multimodal=bool(multimodal),
                n_iter_before_update=update_interval,
                dump_callback=dump_callback,
                **run_kwargs,
            )
            _mpi_barrier()
            analyzer = pymultinest.Analyzer(
                n_params=problem.ndim,
                outputfiles_basename=basename,
                verbose=bool(verbose and primary),
            )
            result = _result_from_analyzer(
                problem,
                analyzer,
                output_dir=root,
                basename=absolute_basename,
                mpi_nprocs=mpi_nprocs,
                seed=seed,
                resume=bool(resume),
                max_iter=iteration_limit,
                invalid_loglike_floor=invalid_floor,
                attempt_id=attempt_id,
                elapsed_seconds=max(time.monotonic() - started_monotonic, 0.0),
            )
        if primary:
            final_status = {
                **status_base,
                "state": "converged" if result.converged else "iteration_limit_reached",
                "elapsed_seconds": max(time.monotonic() - started_monotonic, 0.0),
                "converged": result.converged,
                "message": result.message,
                "log_evidence": result.log_evidence,
                "log_evidence_error": result.log_evidence_error,
                "n_samples": int(result.samples.shape[0]),
            }
            write_retrieval_status(root, final_status)
            append_retrieval_attempt_event(root, {**final_status, "event": "finished"})
        return result
    except BaseException as exc:
        if primary:
            failure_status = {
                **status_base,
                "state": (
                    "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
                ),
                "elapsed_seconds": max(time.monotonic() - started_monotonic, 0.0),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            write_retrieval_status(root, failure_status)
            append_retrieval_attempt_event(root, {**failure_status, "event": "failed"})
        raise


def _result_from_analyzer(
    problem: RetrievalProblem,
    analyzer: object,
    *,
    output_dir: Path,
    basename: str,
    mpi_nprocs: int | None,
    seed: int | None,
    resume: bool,
    max_iter: int,
    invalid_loglike_floor: float,
    attempt_id: str,
    elapsed_seconds: float,
) -> NestedSamplerResult:
    data = np.asarray(analyzer.get_data(), dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] != problem.ndim + 2:
        raise RobertConfigError(
            "PyMultiNest posterior has an unexpected number of columns: "
            f"expected {problem.ndim + 2}, received {data.shape[1]}"
        )
    weights = data[:, 0]
    log_likelihood = -0.5 * data[:, 1]
    samples = data[:, 2:]
    stats = analyzer.get_stats()
    best = analyzer.get_best_fit()
    best_fit = problem.parameter_mapping(np.asarray(best["parameters"], dtype=float))
    converged = bool(samples.size) and max_iter == 0
    message = (
        "converged"
        if converged
        else "maximum iteration limit configured; convergence is not guaranteed"
    )
    return NestedSamplerResult(
        method="multinest",
        parameter_names=problem.parameter_names,
        samples=samples,
        log_likelihood=log_likelihood,
        weights=weights,
        log_evidence=float(stats["global evidence"]),
        log_evidence_error=float(stats["global evidence error"]),
        best_fit_parameters=best_fit,
        metadata={
            "problem": problem.name,
            "output_dir": str(output_dir),
            "outputfiles_basename": basename,
            "mpi_nprocs": "" if mpi_nprocs is None else str(int(mpi_nprocs)),
            "seed": "" if seed is None else str(int(seed)),
            "resume": str(bool(resume)),
            "max_iter": str(int(max_iter)),
            "invalid_loglike_floor": f"{invalid_loglike_floor:.17g}",
            "attempt_id": attempt_id,
            "elapsed_seconds": f"{float(elapsed_seconds):.6f}",
        },
        converged=converged,
        message=message,
    )


def _positive_integer(value: object, name: str) -> int:
    converted = int(value)
    if isinstance(value, bool) or converted < 1:
        raise RobertConfigError(f"{name} must be a positive integer")
    return converted


def _positive_float(value: object, name: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise RobertConfigError(f"{name} must be finite and positive")
    return converted


def _mpi_barrier() -> None:
    try:
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
    except ImportError:
        return


@contextmanager
def _working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _is_primary_process() -> bool:
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank()) == 0
    except ImportError:
        return True


def _attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    job = os.environ.get("SLURM_JOB_ID", "local")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    return f"{timestamp}-{job}-{restart}-{os.getpid()}"


__all__ = ["MULTINEST_MAX_SEED", "run_multinest"]
