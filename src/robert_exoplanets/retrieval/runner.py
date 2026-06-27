"""Retrieval orchestration stubs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.instruments import Observation

from .config import RetrievalConfig
from .model import EmissionModel


@dataclass(frozen=True)
class RetrievalResult:
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
) -> RetrievalResult:
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

    return RetrievalResult(
        config=config,
        model_name=active_model.name,
        best_fit_parameters=best_fit_parameters,
        model_flux=model_flux,
        log_likelihood=log_likelihood,
        converged=False,
        message="Stub retrieval completed; no physical sampler has been run.",
    )
