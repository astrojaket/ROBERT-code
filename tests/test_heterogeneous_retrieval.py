"""Focused tests for heterogeneous HRS/LRS retrieval composition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.instruments import Observation
from robert_exoplanets.likelihoods import GaussianLikelihood
from robert_exoplanets.retrieval import (
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
    build_run_manifest,
)


class _PreparedHRSLike:
    """Small embedded-observation likelihood with the prepared call shape."""

    name = "prepared-hrs-test"

    def __init__(self) -> None:
        self.observation = object()

    def loglike(self, prediction: object, parameters: object = None) -> float:
        if not isinstance(parameters, dict):
            raise AssertionError("prepared HRS likelihood must receive parameters")
        return -float(prediction)


class _NaNLike:
    def loglike(self, prediction: object, parameters: object = None) -> float:
        return float("nan")


def _observation() -> Observation:
    return Observation.from_arrays(
        wavelength=[2.0, 2.1],
        flux=[1.0, 2.0],
        uncertainty=[0.5, 0.5],
        wavelength_unit="micron",
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
    )


def _spectrum() -> Spectrum:
    return Spectrum.from_arrays(
        wavelength=[2.0, 2.1],
        values=[1.5, 1.5],
        unit="eclipse_depth",
        observable="eclipse_depth",
        wavelength_unit="micron",
    )


def _parameters() -> RetrievalParameterSet:
    return RetrievalParameterSet(
        (
            RetrievalParameter("temperature", UniformPrior(0.0, 10.0)),
        )
    )


def _likelihood() -> HeterogeneousLikelihood:
    lrs = HeterogeneousLikelihoodComponent(
        name="lrs",
        prediction_key="lrs",
        likelihood=GaussianLikelihood(offset_parameter=None),
        observation=_observation(),
        metadata={"resolution": "low"},
    )
    hrs = HeterogeneousLikelihoodComponent(
        name="hrs",
        prediction_key="hrs",
        likelihood=_PreparedHRSLike(),
        metadata={"resolution": "high", "prepared": "true"},
    )
    return HeterogeneousLikelihood((hrs, lrs), name="wasp77-joint")


def test_components_use_exact_call_shapes_and_preserve_order() -> None:
    likelihood = _likelihood()

    terms = likelihood.loglike_by_component(
        {"hrs": 2.5, "lrs": _spectrum()},
        {"temperature": 5.0},
    )

    assert tuple(terms) == ("hrs", "lrs")
    assert terms["hrs"] == pytest.approx(-2.5)
    assert terms["lrs"] == pytest.approx(-1.0)
    assert likelihood.loglike({"hrs": 2.5, "lrs": _spectrum()}, {"temperature": 5.0}) == pytest.approx(
        -3.5
    )
    assert likelihood.component_names == ("hrs", "lrs")
    assert likelihood.prediction_keys == ("hrs", "lrs")


def test_unique_names_and_prediction_keys_are_required() -> None:
    prepared = _PreparedHRSLike()
    with pytest.raises(RobertValidationError, match="names must be unique"):
        HeterogeneousLikelihood(
            (
                HeterogeneousLikelihoodComponent("same", "a", prepared),
                HeterogeneousLikelihoodComponent("same", "b", prepared),
            )
        )
    with pytest.raises(RobertValidationError, match="prediction keys must be unique"):
        HeterogeneousLikelihood(
            (
                HeterogeneousLikelihoodComponent("a", "same", prepared),
                HeterogeneousLikelihoodComponent("b", "same", prepared),
            )
        )


def test_missing_prediction_is_strict_for_terms_and_invalid_for_scalar() -> None:
    likelihood = _likelihood()

    with pytest.raises(RobertValidationError, match="missing component key 'lrs'"):
        likelihood.loglike_by_component({"hrs": 2.5}, {})
    assert likelihood.loglike({"hrs": 2.5}, {}) == float("-inf")


def test_nonfinite_component_is_not_summed_as_a_valid_likelihood() -> None:
    component = HeterogeneousLikelihoodComponent("hrs", "hrs", _NaNLike())
    likelihood = HeterogeneousLikelihood((component,), invalid_model_loglike=-99.0)

    with pytest.raises(RobertValidationError, match="non-finite"):
        likelihood.loglike_by_component({"hrs": 1.0}, {})
    assert likelihood.loglike({"hrs": 1.0}, {}) == -99.0


def test_problem_mirrors_sampler_api_and_prior_transform() -> None:
    likelihood = _likelihood()

    def forward(parameters: dict[str, float]) -> dict[str, object]:
        return {"hrs": 2.5, "lrs": _spectrum()}

    problem = HeterogeneousRetrievalProblem(
        name="wasp77-joint",
        parameters=_parameters(),
        forward_model=forward,
        likelihood=likelihood,
        invalid_loglike=-123.0,
        metadata={"target": "WASP-77Ab"},
        opacity_identifiers={"co": "lbl-co-test"},
    )

    assert problem.parameter_names == ("temperature",)
    assert problem.ndim == 1
    np.testing.assert_allclose(problem.prior_transform([0.25]), [2.5])
    assert problem.parameter_mapping([2.5]) == {"temperature": 2.5}
    assert problem.log_likelihood_by_component_from_vector([2.5]) == {
        "hrs": pytest.approx(-2.5),
        "lrs": pytest.approx(-1.0),
    }
    assert problem.log_likelihood_from_vector([2.5]) == pytest.approx(-3.5)
    assert problem.log_prior_from_vector([2.5]) == pytest.approx(-np.log(10.0))
    assert problem.log_posterior_from_vector([2.5]) == pytest.approx(
        -3.5 - np.log(10.0)
    )


def test_problem_returns_invalid_floor_for_forward_model_errors() -> None:
    def invalid_forward(parameters: dict[str, float]) -> dict[str, object]:
        raise RobertValidationError("model outside domain")

    problem = HeterogeneousRetrievalProblem(
        name="invalid",
        parameters=_parameters(),
        forward_model=invalid_forward,
        likelihood=_likelihood(),
        invalid_loglike=-321.0,
    )

    assert problem.log_likelihood_from_vector([2.5]) == -321.0
    assert problem.log_likelihood_by_component_from_vector([2.5]) == {
        "hrs": -321.0,
        "lrs": -321.0,
    }
    assert problem.log_posterior_from_vector([2.5]) == pytest.approx(
        -321.0 - np.log(10.0)
    )


def test_components_and_problem_metadata_are_immutable() -> None:
    likelihood = _likelihood()
    component = likelihood.components[0]
    with pytest.raises(TypeError):
        component.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        component.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        likelihood.components = ()  # type: ignore[misc]

    problem = HeterogeneousRetrievalProblem(
        name="immutable",
        parameters=_parameters(),
        forward_model=lambda parameters: {"hrs": 1.0, "lrs": _spectrum()},
        likelihood=likelihood,
        metadata={"target": "WASP-77Ab"},
        opacity_identifiers={"co": "lbl-co-test"},
    )
    with pytest.raises(TypeError):
        problem.metadata["target"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.opacity_identifiers["co"] = "other"  # type: ignore[index]


def test_manifest_records_prepared_and_explicit_likelihood_components() -> None:
    problem = HeterogeneousRetrievalProblem(
        name="manifest-components",
        parameters=_parameters(),
        forward_model=lambda parameters: {"hrs": 1.0, "lrs": _spectrum()},
        likelihood=_likelihood(),
        metadata={"target": "WASP-77Ab"},
    )

    manifest = build_run_manifest(
        problem,
        method="multinest",
        settings={"n_live_points": 64, "max_iter": 0},
        random_seed=24680,
    )

    components = manifest.likelihood["components"]
    assert [item["name"] for item in components] == ["hrs", "lrs"]
    assert [item["observation_dispatch"] for item in components] == [
        "prepared",
        "explicit",
    ]
    assert components[0]["likelihood"]["type"] == "_PreparedHRSLike"
    assert components[1]["likelihood"]["type"] == "GaussianLikelihood"
