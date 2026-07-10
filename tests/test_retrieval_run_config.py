"""Tests for Python-first retrieval-run configuration."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    Observation,
    OptimalEstimationRunConfig,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalRunConfig,
    Spectrum,
    UltraNestRunConfig,
    UniformPrior,
    build_retrieval_problem,
    run_configured_retrieval,
)
from robert_exoplanets.core import RobertConfigError


class _ConfiguredModel:
    required_parameters = ("level",)
    manifest_metadata = {"model": "configured-test"}
    opacity_identifiers = {"H2O": "checksum"}

    def __init__(self, observation: Observation) -> None:
        self.observation = observation

    def __call__(self, parameters):
        return Spectrum(
            spectral_grid=self.observation.spectral_grid,
            values=np.full(self.observation.n_points, parameters["level"]),
            unit="eclipse_depth",
            observable="eclipse_depth",
        )


def _observation() -> Observation:
    return Observation(
        wavelength=np.array([2.0, 3.0]),
        flux=np.array([1.0, 1.0]),
        uncertainty=np.array([0.1, 0.1]),
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
    )


def _config(tmp_path) -> RetrievalRunConfig:
    observation = _observation()
    return RetrievalRunConfig(
        name="configured-retrieval",
        observation=observation,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("level", UniformPrior(0.0, 2.0)),)
        ),
        forward_model=_ConfiguredModel(observation),
        inference=OptimalEstimationRunConfig(max_iterations=2),
        output_dir=tmp_path,
        metadata={"interface": "python"},
    )


def test_run_config_builds_problem_and_collects_model_provenance(tmp_path) -> None:
    config = _config(tmp_path)

    problem = build_retrieval_problem(config)

    assert problem.name == "configured-retrieval"
    assert problem.metadata == {"model": "configured-test", "interface": "python"}
    assert problem.opacity_identifiers == {"H2O": "checksum"}
    assert problem.log_likelihood_from_vector([1.0]) == pytest.approx(0.0)


def test_run_config_executes_and_serializes_optimal_estimation(tmp_path) -> None:
    result = run_configured_retrieval(_config(tmp_path))

    assert result.method == "optimal_estimation"
    assert result.best_fit_parameters["level"] == pytest.approx(1.0)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "result.json").exists()


def test_run_config_rejects_missing_model_parameters(tmp_path) -> None:
    observation = _observation()
    with pytest.raises(RobertConfigError, match="missing forward-model parameters"):
        RetrievalRunConfig(
            name="invalid",
            observation=observation,
            parameters=RetrievalParameterSet(
                (RetrievalParameter("other", UniformPrior(0.0, 1.0)),)
            ),
            forward_model=_ConfiguredModel(observation),
            inference=OptimalEstimationRunConfig(),
            output_dir=tmp_path,
        )


def test_inference_configs_validate_and_expose_runner_settings() -> None:
    nested = UltraNestRunConfig(
        min_num_live_points=40,
        max_ncalls=10000,
        dlogz=1.5,
        seed=42,
    )

    assert nested.method == "ultranest"
    assert nested.kwargs()["min_num_live_points"] == 40
    assert nested.seed == 42
    with pytest.raises(RobertConfigError, match="positive"):
        OptimalEstimationRunConfig(max_iterations=0)
    with pytest.raises(RobertConfigError, match="reserved"):
        UltraNestRunConfig(extra_run_kwargs={"dlogz": 0.1})
