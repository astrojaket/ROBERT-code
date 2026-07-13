"""Tests for staged optimal-estimation and nested-sampling workflows."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets import (
    NestedSamplerResult,
    Observation,
    OptimalEstimationResult,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    nested_posterior_oe_prior,
    refine_priors_from_optimal_estimation,
    run_nested_sampling_then_oe,
    run_oe_then_nested_sampling,
)


def _problem(*, detailed: bool = False) -> RetrievalProblem:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[1.0, 1.2, 1.4],
        uncertainty=[0.05, 0.05, 0.05],
    )
    parameters = [
        RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),
        RetrievalParameter("slope", UniformPrior(-1.0, 1.0)),
    ]
    if detailed:
        parameters.append(RetrievalParameter("curvature", UniformPrior(-2.0, 2.0)))
    parameter_set = RetrievalParameterSet(tuple(parameters))

    def forward(values):
        x = observation.wavelength - 2.0
        model = values["baseline"] + values["slope"] * x
        if detailed:
            model = model + values["curvature"] * np.square(x)
        return Spectrum.from_arrays(observation.wavelength, model)

    return RetrievalProblem(
        name="detailed" if detailed else "simple",
        observation=observation,
        parameters=parameter_set,
        forward_model=forward,
    )


def _oe_result() -> OptimalEstimationResult:
    return OptimalEstimationResult(
        parameter_names=("baseline", "slope"),
        state_vector=np.array([1.2, 0.2]),
        covariance=np.diag([0.01, 0.0025]),
        averaging_kernel=np.eye(2),
        cost=1.0,
        log_likelihood=-0.5,
        n_iterations=2,
        converged=True,
        message="converged",
    )


def _nested_result(*, converged: bool = True) -> NestedSamplerResult:
    samples = np.array([[1.0, 0.1], [1.2, 0.2], [1.4, 0.3]])
    return NestedSamplerResult(
        method="ultranest",
        parameter_names=("baseline", "slope"),
        samples=samples,
        log_likelihood=np.array([-2.0, -0.5, -1.0]),
        weights=np.array([0.2, 0.6, 0.2]),
        best_fit_parameters={"baseline": 1.2, "slope": 0.2},
        converged=converged,
        message="converged" if converged else "call limit reached",
    )


def test_oe_covariance_refines_priors_within_original_bounds() -> None:
    refined = refine_priors_from_optimal_estimation(
        _problem().parameters,
        _oe_result(),
        prior_sigma=2.0,
        minimum_prior_fraction=0.1,
    )

    assert refined.bounds[0] == pytest.approx((1.0, 1.4))
    assert refined.bounds[1] == pytest.approx((0.1, 0.3))


def test_nested_posterior_maps_shared_and_new_oe_parameters() -> None:
    state, covariance, transferred = nested_posterior_oe_prior(
        _nested_result(),
        _problem(detailed=True),
        state_overrides={"curvature": 0.15},
    )

    np.testing.assert_allclose(state, [1.2, 0.2, 0.15])
    assert transferred == {"baseline": 1.2, "slope": 0.2, "curvature": 0.15}
    assert covariance.shape == (3, 3)
    assert covariance[0, 1] > 0.0
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_oe_then_nested_sampling_runs_both_stages(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(problem, *, method, output_dir, **kwargs):
        calls.append((problem, method, output_dir, kwargs))
        if method == "optimal_estimation":
            return SimpleNamespace(
                converged=True,
                message="converged",
                inference_result=_oe_result(),
                best_fit_parameters=_oe_result().best_fit_parameters,
            )
        return SimpleNamespace(converged=True, best_fit_parameters={})

    monkeypatch.setattr("robert_exoplanets.retrieval.hybrid.run_retrieval", fake_run)
    result = run_oe_then_nested_sampling(
        _problem(),
        output_dir=tmp_path,
        prior_sigma=2.0,
        nested_kwargs={"min_num_live_points": 20},
        seed=3,
    )

    assert [call[1] for call in calls] == ["optimal_estimation", "ultranest"]
    assert result.refined_problem.parameters.bounds[0] == pytest.approx((1.0, 1.4))
    assert (tmp_path / "hybrid_handoff.json").is_file()


def test_nested_then_oe_uses_best_fit_as_prior_state(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(problem, *, method, output_dir, **kwargs):
        calls.append((method, kwargs))
        if method == "ultranest":
            return _nested_result()
        return SimpleNamespace(converged=True, best_fit_parameters={})

    monkeypatch.setattr("robert_exoplanets.retrieval.hybrid.run_retrieval", fake_run)
    result = run_nested_sampling_then_oe(
        _problem(),
        oe_problem=_problem(detailed=True),
        oe_state_overrides={"curvature": 0.25},
        output_dir=tmp_path,
    )

    np.testing.assert_allclose(calls[1][1]["prior_state"], [1.2, 0.2, 0.25])
    np.testing.assert_allclose(calls[1][1]["initial_state"], [1.2, 0.2, 0.25])
    assert result.transferred_prior_state["curvature"] == pytest.approx(0.25)


def test_nested_then_oe_requires_convergence_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "robert_exoplanets.retrieval.hybrid.run_retrieval",
        lambda *args, **kwargs: _nested_result(converged=False),
    )

    with pytest.raises(Exception, match="has not converged"):
        run_nested_sampling_then_oe(_problem(), output_dir=tmp_path)
