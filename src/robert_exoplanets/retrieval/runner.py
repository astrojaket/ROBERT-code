"""Retrieval orchestration and legacy stub workflow."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.instruments import Observation

from .config import RetrievalConfig
from .model import EmissionModel
from .manifest import build_run_manifest, write_run_manifest
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
    manifest = build_run_manifest(
        problem,
        method=normalized,
        settings=settings,
        random_seed=seed if normalized == "ultranest" else None,
    )
    rank, communicator = _mpi_context()
    if rank == 0:
        write_run_manifest(manifest, output_path)
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
