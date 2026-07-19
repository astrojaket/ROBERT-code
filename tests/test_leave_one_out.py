"""Tests for pointwise Gaussian likelihoods and PSIS-LOO diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    GaussianLikelihood,
    MultiDatasetRetrievalProblem,
    Observation,
    ObservationCollection,
    ObservationDataset,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    compare_psis_leave_one_out,
    psis_leave_one_out,
    postprocess_retrieval_output,
    run_psis_leave_one_out,
    write_leave_one_out_result,
)
from robert_exoplanets.io.task_config import LeaveOneOutConfig


pytest.importorskip("arviz")


def _problem(*, masked: bool = False) -> RetrievalProblem:
    observation = Observation.from_arrays(
        wavelength=[1.0, 1.5, 2.0, 2.5],
        flux=[1.00, 1.08, 0.94, 1.02],
        uncertainty=[0.10, 0.12, 0.09, 0.11],
        mask=None if not masked else [True, False, True, True],
        instrument="synthetic",
    )

    def forward(parameters):
        return Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, parameters["level"]),
            unit=observation.flux_unit,
            observable=observation.observable,
        )

    return RetrievalProblem(
        name="constant-spectrum",
        observation=observation,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("level", UniformPrior(0.5, 1.5)),)
        ),
        forward_model=forward,
        likelihood=GaussianLikelihood(include_normalization=True),
    )


def test_pointwise_likelihood_sums_to_scalar_likelihood() -> None:
    problem = _problem(masked=True)
    vector = np.array([1.0])

    pointwise = problem.pointwise_log_likelihood_from_vector(vector)

    assert pointwise.shape == (3,)
    assert np.sum(pointwise) == pytest.approx(
        problem.log_likelihood_from_vector(vector)
    )
    assert not pointwise.flags.writeable


def test_psis_loo_reports_pointwise_scores_and_model_difference() -> None:
    rng = np.random.default_rng(12)
    matrix = rng.normal(-2.0, 0.15, size=(300, 4))
    first = psis_leave_one_out(
        matrix,
        model_name="reference",
        observation_ids=("a", "b", "c", "d"),
        pareto_k_threshold=0.7,
    )
    degraded = matrix.copy()
    degraded[:, 1] -= 1.0
    second = psis_leave_one_out(
        degraded,
        model_name="degraded",
        observation_ids=("a", "b", "c", "d"),
        pareto_k_threshold=0.7,
    )

    comparison = compare_psis_leave_one_out(first, second)

    assert first.pointwise_elpd.shape == (4,)
    assert first.pareto_k.shape == (4,)
    assert np.sum(first.pointwise_elpd) == pytest.approx(first.elpd_loo)
    assert comparison.delta_elpd == pytest.approx(1.0)
    np.testing.assert_allclose(comparison.pointwise_delta, [0.0, 1.0, 0.0, 0.0])
    assert comparison.standard_error == pytest.approx(1.0)


def test_retrieval_loo_resamples_weighted_nested_posterior_reproducibly(
    tmp_path: Path,
) -> None:
    problem = _problem(masked=True)
    samples = np.linspace(0.85, 1.15, 120)[:, np.newaxis]
    weights = np.linspace(1.0, 3.0, samples.shape[0])

    first = run_psis_leave_one_out(
        problem,
        samples,
        weights=weights,
        max_posterior_draws=80,
        seed=42,
    )
    second = run_psis_leave_one_out(
        problem,
        samples,
        weights=weights,
        max_posterior_draws=80,
        seed=42,
    )
    summary_path, arrays_path = write_leave_one_out_result(first, tmp_path)

    assert first.posterior_draws == 80
    assert first.source_samples == 120
    assert first.posterior_resampled
    assert first.observation_ids == ("synthetic[0]", "synthetic[2]", "synthetic[3]")
    np.testing.assert_allclose(first.pointwise_elpd, second.pointwise_elpd)
    np.testing.assert_array_equal(
        first.selected_source_indices,
        second.selected_source_indices,
    )
    assert summary_path.is_file()
    assert arrays_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["method"] == "psis-loo"
    assert payload["number_observations"] == 3
    assert "pointwise_log_likelihood" in np.load(arrays_path)


def test_leave_one_out_config_validates_draw_count() -> None:
    configured = LeaveOneOutConfig(
        enabled=True,
        max_posterior_draws=500,
        seed=7,
        pareto_k_threshold=0.7,
    )
    assert configured.enabled
    assert configured.max_posterior_draws == 500
    with pytest.raises(ValueError, match="at least 20"):
        LeaveOneOutConfig(max_posterior_draws=19)


def test_retrieval_postprocessing_can_write_optional_loo_products(
    tmp_path: Path,
) -> None:
    observation = Observation.from_arrays(
        [1.0, 1.5, 2.0],
        [1.0, 1.05, 0.98],
        [0.1, 0.1, 0.1],
        instrument="synthetic",
    )
    observations = ObservationCollection(
        (ObservationDataset("synthetic", observation),)
    )

    def forward(parameters):
        return {
            "synthetic": Spectrum.from_arrays(
                observation.wavelength,
                np.full(observation.n_points, parameters["level"]),
                unit=observation.flux_unit,
                observable=observation.observable,
            )
        }

    problem = MultiDatasetRetrievalProblem(
        name="postprocessed-loo",
        observations=observations,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("level", UniformPrior(0.5, 1.5)),)
        ),
        forward_model=forward,
    )
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "parameter_names": ["level"],
                "best_fit_parameters": {"level": 1.0},
                "best_fit_log_likelihood": 0.0,
                "method": "multinest",
                "converged": True,
                "message": "finished",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    samples = np.linspace(0.85, 1.15, 100)[:, np.newaxis]
    np.savez(
        result_dir / "result_arrays.npz",
        samples=samples,
        weights=np.ones(samples.shape[0]),
        log_likelihood=np.zeros(samples.shape[0]),
    )
    plot_dir = tmp_path / "plots"

    diagnostics = postprocess_retrieval_output(
        problem,
        result_dir,
        plot_dir=plot_dir,
        leave_one_out=True,
        loo_max_posterior_draws=60,
        loo_seed=5,
    )

    assert diagnostics["leave_one_out"]["status"] == "complete"
    assert (plot_dir / "leave_one_out.json").is_file()
    assert (plot_dir / "leave_one_out_arrays.npz").is_file()
    assert (plot_dir / "leave_one_out.png").is_file()
