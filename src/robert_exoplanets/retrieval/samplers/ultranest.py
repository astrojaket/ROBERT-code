"""UltraNest adapter for ROBERT retrieval problems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robert_exoplanets.core import RobertConfigError

from ..problem import RetrievalProblem
from .base import NestedSamplerResult


def run_ultranest(
    problem: RetrievalProblem,
    *,
    output_dir: str | Path,
    min_num_live_points: int = 400,
    max_ncalls: int | None = None,
    dlogz: float = 0.5,
    resume: str = "overwrite",
    show_status: bool = True,
    mpi_nprocs: int | None = None,
    **run_kwargs: Any,
) -> NestedSamplerResult:
    """Run UltraNest for a ROBERT retrieval problem.

    MPI is controlled by launching the Python process with `mpiexec`, for
    example `/opt/homebrew/bin/mpiexec -n 3 python script.py`. The optional
    `mpi_nprocs` argument is recorded only as metadata.
    """

    try:
        import ultranest
    except ImportError as exc:
        raise RobertConfigError(
            "UltraNest is not installed. Install with `python -m pip install ultranest mpi4py`."
        ) from exc

    log_dir = Path(output_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    sampler = ultranest.ReactiveNestedSampler(
        list(problem.parameter_names),
        problem.log_likelihood_from_vector,
        transform=problem.prior_transform,
        log_dir=str(log_dir),
        resume=resume,
    )
    result = sampler.run(
        min_num_live_points=int(min_num_live_points),
        max_ncalls=max_ncalls,
        dlogz=float(dlogz),
        show_status=bool(show_status),
        **run_kwargs,
    )
    return _result_from_ultranest(problem, result, log_dir=log_dir, mpi_nprocs=mpi_nprocs)


def _result_from_ultranest(
    problem: RetrievalProblem,
    result: dict[str, Any],
    *,
    log_dir: Path,
    mpi_nprocs: int | None,
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
        },
    )
