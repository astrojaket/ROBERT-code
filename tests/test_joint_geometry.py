"""Tests for shared-prior composition of independent forward geometries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import pytest

from robert_exoplanets.core import RobertConfigError, RobertValidationError, Spectrum
from robert_exoplanets.instruments import (
    Observation,
    TimeResolvedHighResolutionObservation,
)
from robert_exoplanets.likelihoods import (
    CorrelatedGaussianLikelihood,
    GaussianLikelihood,
    PositiveSemidefiniteCovariance,
    TimeResolvedHighResolutionLikelihood,
)
from robert_exoplanets.retrieval import (
    JointGeometryComponent,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
    build_joint_geometry_problem,
)
from robert_exoplanets.retrieval.manifest import (
    build_run_manifest,
    resume_compatibility_changes,
    validate_resume_compatibility,
)


@dataclass
class _LinearSpectrumModel:
    """Small parameterized model that records its local inputs."""

    required_parameters: tuple[str, ...]
    unit: str
    observable: str
    wavelength_micron: float
    opacity_identifiers: Mapping[str, str] = field(default_factory=dict)
    calls: list[dict[str, float]] = field(default_factory=list)

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        values = dict(parameters)
        self.calls.append(values)
        value = values["temperature"] + values["log_gas"]
        return Spectrum.from_arrays(
            wavelength=[self.wavelength_micron],
            values=[value],
            unit=self.unit,
            observable=self.observable,
            wavelength_unit="micron",
        )


def _observation(
    *,
    unit: str,
    observable: str,
    wavelength_micron: float,
    value: float,
    uncertainty: float,
) -> Observation:
    return Observation.from_arrays(
        wavelength=[wavelength_micron],
        flux=[value],
        uncertainty=[uncertainty],
        wavelength_unit="micron",
        flux_unit=unit,
        observable=observable,
        instrument=observable,
    )


def _parameter_set(*names: str) -> RetrievalParameterSet:
    priors = {
        "temperature": UniformPrior(200.0, 800.0),
        "transmission_gas": UniformPrior(-8.0, -2.0),
        "emission_gas": UniformPrior(-7.0, -1.0),
        "transmission_offset": UniformPrior(-1.0, 1.0),
        "emission_offset": UniformPrior(-2.0, 2.0),
    }
    return RetrievalParameterSet(
        tuple(RetrievalParameter(name, priors[name]) for name in names)
    )


def _components() -> tuple[JointGeometryComponent, JointGeometryComponent]:
    transmission_model = _LinearSpectrumModel(
        required_parameters=("temperature", "log_gas"),
        unit="transit_depth",
        observable="transit_depth",
        wavelength_micron=1.0,
        opacity_identifiers={"CO2": "transmission-test"},
    )
    emission_model = _LinearSpectrumModel(
        required_parameters=("temperature", "log_gas"),
        unit="eclipse_depth",
        observable="eclipse_depth",
        wavelength_micron=15.0,
        opacity_identifiers={"CO2": "emission-test"},
    )
    transmission = JointGeometryComponent(
        name="transmission",
        forward_model=transmission_model,
        required_parameters=("temperature", "log_gas"),
        likelihood_parameters=("offset",),
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "transmission_gas",
            "offset": "transmission_offset",
        },
        likelihood=GaussianLikelihood(
            include_normalization=True,
            offset_parameter="offset",
            jitter_parameter=None,
        ),
        observation=_observation(
            unit="transit_depth",
            observable="transit_depth",
            wavelength_micron=1.0,
            value=500.0,
            uncertainty=2.0,
        ),
        metadata={"geometry": "transmission"},
    )
    emission = JointGeometryComponent(
        name="emission",
        forward_model=emission_model,
        required_parameters=("temperature", "log_gas"),
        likelihood_parameters=("offset",),
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "emission_gas",
            "offset": "emission_offset",
        },
        likelihood=GaussianLikelihood(
            include_normalization=True,
            offset_parameter="offset",
            jitter_parameter=None,
        ),
        observation=_observation(
            unit="eclipse_depth",
            observable="eclipse_depth",
            wavelength_micron=15.0,
            value=500.0,
            uncertainty=3.0,
        ),
        metadata={"geometry": "emission"},
    )
    return transmission, emission


def test_joint_problem_shares_one_prior_and_keeps_component_states_local() -> None:
    components = _components()
    parameters = _parameter_set(
        "temperature",
        "transmission_gas",
        "emission_gas",
        "transmission_offset",
        "emission_offset",
    )
    problem = build_joint_geometry_problem(
        name="synthetic-joint",
        components=components,
        parameters=parameters,
        metadata={"prior_scenario": "broad_baseline"},
    )

    assert problem.parameter_names == parameters.names
    assert problem.ndim == 5
    assert problem.metadata["prior_scenario"] == "broad_baseline"
    assert problem.opacity_identifiers == {
        "transmission:CO2": "transmission-test",
        "emission:CO2": "emission-test",
    }
    assert problem.likelihood.components[0].metadata["parameter_bindings"] == (
        "temperature=temperature;log_gas=transmission_gas;"
        "offset=transmission_offset"
    )
    assert problem.opacity_identifiers == {
        "transmission:CO2": "transmission-test",
        "emission:CO2": "emission-test",
    }
    assert problem.likelihood.components[0].metadata["parameter_bindings"] == (
        "temperature=temperature;log_gas=transmission_gas;"
        "offset=transmission_offset"
    )
    vector = parameters.midpoint_vector()
    mapping = problem.parameter_mapping(vector)
    predictions = problem.predict(mapping)

    assert tuple(predictions) == ("transmission", "emission")
    transmission_model = components[0].forward_model
    emission_model = components[1].forward_model
    assert isinstance(transmission_model, _LinearSpectrumModel)
    assert isinstance(emission_model, _LinearSpectrumModel)
    assert transmission_model.calls == [
        {"temperature": 500.0, "log_gas": -5.0}
    ]
    assert emission_model.calls == [
        {"temperature": 500.0, "log_gas": -4.0}
    ]

    # A shared temperature has one prior density contribution in the joint set.
    expected_log_prior = sum(
        parameter.prior.log_probability(value)
        for parameter, value in zip(parameters.parameters, vector, strict=True)
    )
    assert problem.log_prior_from_vector(vector) == pytest.approx(expected_log_prior)

    terms = problem.log_likelihood_by_component_from_vector(vector)
    assert tuple(terms) == ("transmission", "emission")
    assert problem.log_likelihood_from_vector(vector) == pytest.approx(sum(terms.values()))


def test_joint_problem_supports_an_explicit_component_and_prior_subset() -> None:
    transmission, _ = _components()
    subset_parameters = _parameter_set(
        "temperature",
        "transmission_gas",
        "transmission_offset",
    )
    problem = build_joint_geometry_problem(
        name="transmission-only-subset",
        components=(transmission,),
        parameters=subset_parameters,
    )

    assert problem.parameter_names == subset_parameters.names
    assert problem.ndim == 3
    assert tuple(problem.predict(subset_parameters.vector_to_mapping([500.0, -5.0, 0.0]))) == (
        "transmission",
    )


def test_components_can_keep_separate_limb_and_dayside_temperatures() -> None:
    transmission_model = _LinearSpectrumModel(
        required_parameters=("temperature", "log_gas"),
        unit="transit_depth",
        observable="transit_depth",
        wavelength_micron=1.0,
        opacity_identifiers={"CO2": "transmission-test"},
    )
    emission_model = _LinearSpectrumModel(
        required_parameters=("temperature", "log_gas"),
        unit="eclipse_depth",
        observable="eclipse_depth",
        wavelength_micron=15.0,
        opacity_identifiers={"CO2": "emission-test"},
    )
    no_nuisance = GaussianLikelihood(offset_parameter=None, jitter_parameter=None)
    transmission = JointGeometryComponent(
        name="transmission",
        forward_model=transmission_model,
        likelihood=no_nuisance,
        required_parameters=("temperature", "log_gas"),
        parameter_bindings={
            "temperature": "limb_temperature",
            "log_gas": "shared_log_gas",
        },
    )
    emission = JointGeometryComponent(
        name="emission",
        forward_model=emission_model,
        likelihood=no_nuisance,
        required_parameters=("temperature", "log_gas"),
        parameter_bindings={
            "temperature": "dayside_temperature",
            "log_gas": "shared_log_gas",
        },
    )
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("limb_temperature", UniformPrior(300.0, 500.0)),
            RetrievalParameter("dayside_temperature", UniformPrior(800.0, 1200.0)),
            RetrievalParameter("shared_log_gas", UniformPrior(-8.0, -2.0)),
        )
    )
    problem = build_joint_geometry_problem(
        name="separate-temperature-joint",
        components=(transmission, emission),
        parameters=parameters,
    )

    predictions = problem.predict(
        {
            "limb_temperature": 400.0,
            "dayside_temperature": 1000.0,
            "shared_log_gas": -5.0,
        }
    )
    assert set(predictions) == {"transmission", "emission"}
    assert transmission_model.calls == [
        {"temperature": 400.0, "log_gas": -5.0}
    ]
    assert emission_model.calls == [
        {"temperature": 1000.0, "log_gas": -5.0}
    ]


def test_component_requires_explicit_complete_local_bindings() -> None:
    model = _LinearSpectrumModel(
        required_parameters=("temperature", "log_gas"),
        unit="transit_depth",
        observable="transit_depth",
        wavelength_micron=1.0,
    )
    with pytest.raises(RobertValidationError, match="missing bindings for log_gas"):
        JointGeometryComponent(
            name="transmission",
            forward_model=model,
            likelihood=GaussianLikelihood(offset_parameter=None, jitter_parameter=None),
            required_parameters=("temperature", "log_gas"),
            parameter_bindings={"temperature": "temperature"},
        )

    with pytest.raises(RobertValidationError, match="must match its forward model"):
        JointGeometryComponent(
            name="transmission",
            forward_model=model,
            likelihood=GaussianLikelihood(offset_parameter=None, jitter_parameter=None),
            required_parameters=("temperature",),
            parameter_bindings={"temperature": "temperature"},
        )

    with pytest.raises(RobertValidationError, match="must be listed in likelihood_parameters"):
        JointGeometryComponent(
            name="transmission",
            forward_model=model,
            likelihood=GaussianLikelihood(jitter_parameter=None),
            required_parameters=("temperature", "log_gas"),
            parameter_bindings={
                "temperature": "temperature",
                "log_gas": "log_gas",
            },
        )

    with pytest.raises(RobertValidationError, match="must be a sequence of names"):
        JointGeometryComponent(
            name="bad-required-parameter-shape",
            forward_model=model,
            likelihood=GaussianLikelihood(offset_parameter=None, jitter_parameter=None),
            required_parameters="temperature",  # type: ignore[arg-type]
            parameter_bindings={"temperature": "temperature"},
        )

    component = JointGeometryComponent(
        name="valid",
        forward_model=model,
        likelihood=GaussianLikelihood(offset_parameter=None, jitter_parameter=None),
        required_parameters=("temperature", "log_gas"),
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "log_gas",
        },
    )
    with pytest.raises(RobertValidationError, match="no binding for local parameter"):
        component.local_parameters({"temperature": 500.0}, names=("not_bound",))


def test_builder_rejects_unknown_and_unused_global_prior_parameters() -> None:
    transmission, _ = _components()
    unknown = JointGeometryComponent(
        name="transmission-unknown",
        forward_model=transmission.forward_model,
        likelihood=transmission.likelihood,
        required_parameters=transmission.required_parameters,
        likelihood_parameters=transmission.likelihood_parameters,
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "missing_gas_prior",
            "offset": "transmission_offset",
        },
        observation=transmission.observation,
    )
    with pytest.raises(RobertValidationError, match="absent from the prior set"):
        build_joint_geometry_problem(
            name="unknown-binding",
            components=(unknown,),
            parameters=_parameter_set(
                "temperature",
                "transmission_offset",
            ),
        )

    with pytest.raises(RobertValidationError, match="parameters unused"):
        build_joint_geometry_problem(
            name="unused-prior",
            components=(transmission,),
            parameters=_parameter_set(
                "temperature",
                "transmission_gas",
                "transmission_offset",
                "emission_gas",
            ),
        )


def test_joint_likelihood_maps_only_declared_component_parameters() -> None:
    transmission, _ = _components()
    problem = build_joint_geometry_problem(
        name="likelihood-binding",
        components=(transmission,),
        parameters=_parameter_set(
            "temperature",
            "transmission_gas",
            "transmission_offset",
        ),
    )
    vector = np.asarray([500.0, -5.0, 0.0])
    terms = problem.log_likelihood_by_component_from_vector(vector)

    assert set(terms) == {"transmission"}
    assert np.isfinite(terms["transmission"])
    shifted_offset_terms = problem.log_likelihood_by_component_from_vector(
        np.asarray([500.0, -5.0, 0.75])
    )
    assert shifted_offset_terms["transmission"] != terms["transmission"]


class _PreparedLikelihood:
    """Likelihood using the heterogeneous prepared-prediction call shape."""

    name = "prepared-test"

    def __init__(self, observation: Observation | None = None) -> None:
        self.observation = observation
        self.parameters: list[dict[str, float]] = []

    def loglike(self, prediction: object, parameters: Mapping[str, float]) -> float:
        self.parameters.append(dict(parameters))
        assert isinstance(prediction, Spectrum)
        return -float(prediction.values[0]) + float(parameters["offset"])


def test_prepared_likelihood_receives_bound_local_parameters() -> None:
    model = _LinearSpectrumModel(
        required_parameters=("temperature", "log_gas"),
        unit="transit_depth",
        observable="transit_depth",
        wavelength_micron=1.0,
    )
    likelihood = _PreparedLikelihood()
    component = JointGeometryComponent(
        name="prepared",
        forward_model=model,
        likelihood=likelihood,
        required_parameters=("temperature", "log_gas"),
        likelihood_parameters=("offset",),
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "log_gas",
            "offset": "prepared_offset",
        },
    )
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("temperature", UniformPrior(400.0, 600.0)),
            RetrievalParameter("log_gas", UniformPrior(-8.0, -2.0)),
            RetrievalParameter("prepared_offset", UniformPrior(-1.0, 1.0)),
        )
    )
    problem = build_joint_geometry_problem(
        name="prepared-likelihood-joint",
        components=(component,),
        parameters=parameters,
    )

    result = problem.log_likelihood_by_component_from_vector(
        np.asarray([500.0, -5.0, 0.5])
    )
    assert result["prepared"] == pytest.approx(-494.5)
    assert likelihood.parameters == [{"offset": 0.5}]


def test_invalid_forward_and_duplicate_components_fail_cleanly() -> None:
    class InvalidModel:
        required_parameters = ("temperature",)

        def __call__(self, parameters: Mapping[str, float]) -> object:
            raise RobertValidationError("out of physical domain")

    component = JointGeometryComponent(
        name="invalid",
        forward_model=InvalidModel(),
        likelihood=GaussianLikelihood(offset_parameter=None, jitter_parameter=None),
        required_parameters=("temperature",),
        parameter_bindings={"temperature": "temperature"},
        observation=_observation(
            unit="eclipse_depth",
            observable="eclipse_depth",
            wavelength_micron=15.0,
            value=1.0,
            uncertainty=0.1,
        ),
    )
    parameters = RetrievalParameterSet(
        (RetrievalParameter("temperature", UniformPrior(100.0, 1000.0)),)
    )
    problem = build_joint_geometry_problem(
        name="invalid-forward",
        components=(component,),
        parameters=parameters,
        invalid_loglike=-123.0,
    )
    assert problem.log_likelihood_from_vector([500.0]) == -123.0
    assert problem.log_likelihood_by_component_from_vector([500.0]) == {
        "invalid": -123.0
    }

    with pytest.raises(RobertValidationError, match="component names must be unique"):
        build_joint_geometry_problem(
            name="duplicate",
            components=(component, component),
            parameters=parameters,
        )

    with pytest.raises(RobertValidationError, match="must be JointGeometryComponent"):
        build_joint_geometry_problem(
            name="invalid-component-type",
            components=(object(),),  # type: ignore[arg-type]
            parameters=parameters,
        )


_AlternateGaussianLikelihood = type(
    "GaussianLikelihood",
    (GaussianLikelihood,),
    {"__module__": "tests.alternate_gaussian"},
)


def _manifest_joint_problem(
    *,
    include_normalization: bool = True,
    invalid_model_loglike: float = float("-inf"),
    nuisance_role: str = "offset",
    likelihood_type: type[GaussianLikelihood] = GaussianLikelihood,
    offset_binding: str = "transmission_offset",
):
    transmission, _ = _components()
    likelihood = likelihood_type(
        include_normalization=include_normalization,
        invalid_model_loglike=invalid_model_loglike,
        offset_parameter="offset" if nuisance_role == "offset" else None,
        jitter_parameter="offset" if nuisance_role == "jitter" else None,
    )
    component = JointGeometryComponent(
        name=transmission.name,
        forward_model=transmission.forward_model,
        likelihood=likelihood,
        required_parameters=transmission.required_parameters,
        likelihood_parameters=("offset",),
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "transmission_gas",
            "offset": offset_binding,
        },
        observation=transmission.observation,
        metadata=transmission.metadata,
    )
    return build_joint_geometry_problem(
        name="manifest-joint",
        components=(component,),
        parameters=_parameter_set(
            "temperature",
            "transmission_gas",
            offset_binding,
        ),
    )


def _joint_manifest(problem):
    return build_run_manifest(
        problem,
        method="multinest",
        settings={"n_live_points": 32},
        random_seed=1234,
    )


def test_joint_manifest_records_wrapped_likelihood_and_is_resume_compatible() -> None:
    original = _joint_manifest(_manifest_joint_problem())
    unchanged = _joint_manifest(_manifest_joint_problem())

    component_likelihood = original.likelihood["components"][0]["likelihood"]
    assert component_likelihood["wrapped_likelihood"] == {
        "type": "GaussianLikelihood",
        "name": "independent-gaussian",
        "include_normalization": True,
        "coordinate_rtol": 1.0e-12,
        "coordinate_atol": 0.0,
        "uncertainty_scale": 1.0,
        "uncertainty_scale_parameter": None,
        "offset_parameter": "offset",
        "jitter_parameter": None,
        "qualified_type": "robert_exoplanets.likelihoods.gaussian.GaussianLikelihood",
        "invalid_model_loglike": "-inf",
    }
    assert component_likelihood["likelihood_parameters"] == ["offset"]
    assert component_likelihood["parameter_bindings"] == {
        "temperature": "temperature",
        "log_gas": "transmission_gas",
        "offset": "transmission_offset",
    }
    assert unchanged.config_hash == original.config_hash
    assert resume_compatibility_changes(original, unchanged) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"include_normalization": False},
        {"invalid_model_loglike": -1.0e100},
        {"nuisance_role": "jitter"},
        {"offset_binding": "emission_offset"},
    ],
    ids=("normalization", "invalid-model-floor", "nuisance-role", "binding"),
)
def test_joint_manifest_changes_block_checkpoint_resume(changes) -> None:
    original = _joint_manifest(_manifest_joint_problem())
    changed = _joint_manifest(_manifest_joint_problem(**changes))

    assert original.config_hash != changed.config_hash
    assert "likelihood" in resume_compatibility_changes(original, changed)
    with pytest.raises(RobertConfigError, match="likelihood"):
        validate_resume_compatibility(original, changed)


def test_joint_manifest_distinguishes_same_named_wrapped_likelihood_types() -> None:
    original = _joint_manifest(_manifest_joint_problem())
    alternate = _joint_manifest(
        _manifest_joint_problem(likelihood_type=_AlternateGaussianLikelihood)
    )
    original_likelihood = original.likelihood["components"][0]["likelihood"]
    alternate_likelihood = alternate.likelihood["components"][0]["likelihood"]
    original_underlying = original_likelihood["wrapped_likelihood"]
    alternate_underlying = alternate_likelihood["wrapped_likelihood"]

    assert original_underlying["type"] == alternate_underlying["type"]
    assert (
        original_underlying["qualified_type"]
        != alternate_underlying["qualified_type"]
    )
    assert original.config_hash != alternate.config_hash
    assert "likelihood" in resume_compatibility_changes(original, alternate)
    with pytest.raises(RobertConfigError, match="likelihood"):
        validate_resume_compatibility(original, alternate)


def test_joint_manifest_records_covariance_and_rejects_changed_noise_model() -> None:
    transmission, _ = _components()

    def problem(covariance):
        component = JointGeometryComponent(
            name="correlated-transmission",
            forward_model=transmission.forward_model,
            likelihood=CorrelatedGaussianLikelihood(covariance=covariance),
            required_parameters=transmission.required_parameters,
            parameter_bindings={
                "temperature": "temperature",
                "log_gas": "transmission_gas",
            },
            observation=transmission.observation,
        )
        return build_joint_geometry_problem(
            name="manifest-correlated-joint",
            components=(component,),
            parameters=_parameter_set("temperature", "transmission_gas"),
        )

    original = _joint_manifest(problem([[1.0]]))
    changed = _joint_manifest(problem([[2.0]]))
    original_covariance = original.likelihood["components"][0]["likelihood"][
        "wrapped_likelihood"
    ]["covariance_identity"]
    changed_covariance = changed.likelihood["components"][0]["likelihood"][
        "wrapped_likelihood"
    ]["covariance_identity"]

    assert original_covariance["operator_type"].endswith("DenseCovariance")
    assert original_covariance["matrix"]["shape"] == [1, 1]
    assert original_covariance["matrix"]["dtype"] == np.dtype(np.float64).str
    assert original_covariance["matrix"]["sha256"] != changed_covariance[
        "matrix"
    ]["sha256"]
    assert "likelihood" in resume_compatibility_changes(original, changed)
    with pytest.raises(RobertConfigError, match="likelihood"):
        validate_resume_compatibility(original, changed)


def test_joint_manifest_records_psd_covariance_tolerances() -> None:
    transmission, _ = _components()

    def manifest(support_rtol: float):
        component = JointGeometryComponent(
            name="psd-transmission",
            forward_model=transmission.forward_model,
            likelihood=CorrelatedGaussianLikelihood(
                covariance=PositiveSemidefiniteCovariance(
                    [[1.0]], support_rtol=support_rtol
                )
            ),
            required_parameters=transmission.required_parameters,
            parameter_bindings={
                "temperature": "temperature",
                "log_gas": "transmission_gas",
            },
            observation=transmission.observation,
        )
        return _joint_manifest(
            build_joint_geometry_problem(
                name="manifest-psd-joint",
                components=(component,),
                parameters=_parameter_set("temperature", "transmission_gas"),
            )
        )

    original = manifest(1.0e-10)
    changed = manifest(1.0e-8)
    original_covariance = original.likelihood["components"][0]["likelihood"][
        "wrapped_likelihood"
    ]["covariance_identity"]
    changed_covariance = changed.likelihood["components"][0]["likelihood"][
        "wrapped_likelihood"
    ]["covariance_identity"]

    assert original_covariance["rank_rtol"] == changed_covariance["rank_rtol"]
    assert original_covariance["support_rtol"] == 1.0e-10
    assert changed_covariance["support_rtol"] == 1.0e-8
    assert "likelihood" in resume_compatibility_changes(original, changed)


class _OpaqueCovarianceOperator:
    size = 1

    def solve(self, vector):
        return np.asarray(vector, dtype=float)

    def log_determinant(self) -> float:
        return 0.0

    def diagonal(self) -> np.ndarray:
        return np.ones(1)

    def subset(self, mask):
        return self


def test_joint_manifest_rejects_covariance_without_explicit_identity() -> None:
    transmission, _ = _components()
    component = JointGeometryComponent(
        name="opaque-covariance",
        forward_model=transmission.forward_model,
        likelihood=CorrelatedGaussianLikelihood(_OpaqueCovarianceOperator()),
        required_parameters=transmission.required_parameters,
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "transmission_gas",
        },
        observation=transmission.observation,
    )
    problem = build_joint_geometry_problem(
        name="opaque-covariance-joint",
        components=(component,),
        parameters=_parameter_set("temperature", "transmission_gas"),
    )

    with pytest.raises(RobertConfigError, match="custom covariance operator"):
        _joint_manifest(problem)


def _prepared_observation_problem(value: float):
    transmission, _ = _components()
    likelihood = _PreparedLikelihood(
        _observation(
            unit="transit_depth",
            observable="transit_depth",
            wavelength_micron=1.0,
            value=value,
            uncertainty=2.0,
        )
    )
    component = JointGeometryComponent(
        name="prepared-transmission",
        forward_model=transmission.forward_model,
        likelihood=likelihood,
        required_parameters=transmission.required_parameters,
        likelihood_parameters=("offset",),
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "transmission_gas",
            "offset": "transmission_offset",
        },
    )
    return build_joint_geometry_problem(
        name="prepared-observation-joint",
        components=(component,),
        parameters=_parameter_set(
            "temperature", "transmission_gas", "transmission_offset"
        ),
    )


def test_joint_manifest_fingerprints_embedded_prepared_observation() -> None:
    original = _joint_manifest(_prepared_observation_problem(500.0))
    changed = _joint_manifest(_prepared_observation_problem(501.0))

    assert original.observation_identity != changed.observation_identity
    assert "observation_identity" in resume_compatibility_changes(original, changed)
    with pytest.raises(RobertConfigError, match="observation_identity"):
        validate_resume_compatibility(original, changed)


def _time_resolved_role_problem(*, dVsys_parameter: str):
    model, _ = _components()
    observation = TimeResolvedHighResolutionObservation.from_arrays(
        order_wavelengths=[[2.29, 2.30, 2.31, 2.32]],
        flux=np.ones((1, 2, 4)),
        phase=[0.0, 0.25],
        fixed_velocity_km_s=[0.0, 0.0],
        time_bjd=[2_460_000.0, 2_460_000.1],
    )
    likelihood = TimeResolvedHighResolutionLikelihood(
        kp_parameter="local_kp",
        dVsys_parameter=dVsys_parameter,
        dphi_parameter="local_phase",
    )
    component = JointGeometryComponent(
        name="time-resolved-emission",
        forward_model=model.forward_model,
        likelihood=likelihood,
        required_parameters=model.required_parameters,
        likelihood_parameters=("local_kp", "local_sys", "local_phase"),
        parameter_bindings={
            "temperature": "temperature",
            "log_gas": "transmission_gas",
            "local_kp": "global_kp",
            "local_sys": "global_sys",
            "local_phase": "global_phase",
        },
        observation=observation,
    )
    bounds = {
        "temperature": (200.0, 800.0),
        "transmission_gas": (-8.0, -2.0),
        "global_kp": (-100.0, 100.0),
        "global_sys": (-100.0, 100.0),
        "global_phase": (-0.5, 0.5),
    }
    parameters = RetrievalParameterSet(
        tuple(
            RetrievalParameter(name, UniformPrior(*bounds[name]))
            for name in bounds
        )
    )
    return build_joint_geometry_problem(
        name="time-resolved-role-joint",
        components=(component,),
        parameters=parameters,
    )


def test_joint_manifest_records_time_resolved_velocity_roles() -> None:
    original = _joint_manifest(_time_resolved_role_problem(dVsys_parameter="local_sys"))
    changed = _joint_manifest(_time_resolved_role_problem(dVsys_parameter="local_kp"))
    original_component = original.likelihood["components"][0]["likelihood"]
    changed_component = changed.likelihood["components"][0]["likelihood"]

    assert original_component["parameter_bindings"] == changed_component[
        "parameter_bindings"
    ]
    assert original_component["likelihood_parameters"] == changed_component[
        "likelihood_parameters"
    ]
    assert original_component["wrapped_likelihood"]["dVsys_parameter"] == "local_sys"
    assert changed_component["wrapped_likelihood"]["dVsys_parameter"] == "local_kp"
    assert "likelihood" in resume_compatibility_changes(original, changed)
    with pytest.raises(RobertConfigError, match="likelihood"):
        validate_resume_compatibility(original, changed)


def test_old_wrapper_only_joint_manifest_cannot_resume() -> None:
    current = _joint_manifest(_manifest_joint_problem())
    old_likelihood = dict(current.likelihood)
    old_components = [dict(item) for item in old_likelihood["components"]]
    old_components[0]["likelihood"] = {
        "type": "_BoundLikelihood",
        "name": "independent-gaussian",
    }
    old_likelihood["components"] = old_components
    old_manifest = replace(current, likelihood=old_likelihood)

    assert "likelihood" in resume_compatibility_changes(old_manifest, current)
    with pytest.raises(RobertConfigError, match="likelihood"):
        validate_resume_compatibility(old_manifest, current)
