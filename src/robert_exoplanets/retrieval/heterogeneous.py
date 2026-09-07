"""Heterogeneous likelihood composition for joint retrievals.

This module keeps data preparation and likelihood semantics owned by each
component.  In particular, a prepared high-resolution likelihood can retain
its observation and covariance/PCA projection internally while a conventional
likelihood receives its observation explicitly.  The composition layer only
dispatches the two supported call signatures and adds the resulting scalar
log likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

from .priors import RetrievalParameterSet


class HeterogeneousLikelihoodEvaluator(Protocol):
    """Structural contract for a likelihood component."""

    def loglike(self, prediction: object, *args: object) -> float:
        """Evaluate the component log likelihood."""


HeterogeneousForwardEvaluator = Callable[
    [Mapping[str, float]], Mapping[str, object]
]

_MODEL_ERRORS = (
    RobertError,
    ValueError,
    FloatingPointError,
    OverflowError,
    TypeError,
)


def _nonempty_text(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise RobertValidationError(f"{label} must not be empty")
    return text


def _invalid_loglike(value: object, label: str) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise RobertValidationError(f"{label} must be finite or -inf") from error
    if np.isnan(number) or number == float("inf"):
        raise RobertValidationError(f"{label} must be finite or -inf")
    return number


@dataclass(frozen=True)
class HeterogeneousLikelihoodComponent:
    """One named likelihood and its prediction key.

    When ``observation`` is supplied, ``likelihood.loglike`` is called as
    ``loglike(prediction, observation, parameters)``.  When it is ``None``,
    the likelihood is treated as prepared and called as
    ``loglike(prediction, parameters)``.  This explicit distinction prevents
    prepared HRS covariance or PCA terms from being flattened into a diagonal
    Gaussian likelihood.
    """

    name: str
    prediction_key: str
    likelihood: HeterogeneousLikelihoodEvaluator
    observation: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "component name"))
        object.__setattr__(
            self,
            "prediction_key",
            _nonempty_text(self.prediction_key, "component prediction key"),
        )
        if not callable(getattr(self.likelihood, "loglike", None)):
            raise RobertValidationError(
                f"likelihood component '{self.name}' must expose callable loglike"
            )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class HeterogeneousLikelihood:
    """Ordered sum of independently prepared or explicit-data likelihoods."""

    components: tuple[HeterogeneousLikelihoodComponent, ...]
    name: str = "heterogeneous-likelihood"
    invalid_model_loglike: float = float("-inf")
    _components_by_name: Mapping[str, HeterogeneousLikelihoodComponent] = field(
        init=False,
        repr=False,
    )
    _components_by_key: Mapping[str, HeterogeneousLikelihoodComponent] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise RobertValidationError(
                "heterogeneous likelihood must contain at least one component"
            )
        names: dict[str, HeterogeneousLikelihoodComponent] = {}
        prediction_keys: dict[str, HeterogeneousLikelihoodComponent] = {}
        for component in components:
            if not isinstance(component, HeterogeneousLikelihoodComponent):
                raise RobertValidationError(
                    "heterogeneous likelihood components must be "
                    "HeterogeneousLikelihoodComponent instances"
                )
            if component.name in names:
                raise RobertValidationError(
                    f"heterogeneous likelihood component names must be unique: "
                    f"'{component.name}'"
                )
            if component.prediction_key in prediction_keys:
                raise RobertValidationError(
                    "heterogeneous likelihood prediction keys must be unique: "
                    f"'{component.prediction_key}'"
                )
            names[component.name] = component
            prediction_keys[component.prediction_key] = component

        object.__setattr__(self, "components", components)
        object.__setattr__(self, "name", _nonempty_text(self.name, "likelihood name"))
        object.__setattr__(
            self,
            "invalid_model_loglike",
            _invalid_loglike(self.invalid_model_loglike, "invalid_model_loglike"),
        )
        object.__setattr__(self, "_components_by_name", immutable_mapping(names))
        object.__setattr__(
            self,
            "_components_by_key",
            immutable_mapping(prediction_keys),
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in evaluation order."""

        return tuple(component.name for component in self.components)

    @property
    def prediction_keys(self) -> tuple[str, ...]:
        """Prediction keys in evaluation order."""

        return tuple(component.prediction_key for component in self.components)

    def _strict_loglike_by_component(
        self,
        prediction: Mapping[str, object],
        parameters: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if not isinstance(prediction, Mapping):
            raise RobertValidationError(
                "heterogeneous forward prediction must be a named mapping"
            )

        values: dict[str, float] = {}
        for component in self.components:
            if component.prediction_key not in prediction:
                raise RobertValidationError(
                    "heterogeneous forward prediction is missing component key "
                    f"'{component.prediction_key}'"
                )
            if component.observation is None:
                result = component.likelihood.loglike(
                    prediction[component.prediction_key],
                    parameters,
                )
            else:
                result = component.likelihood.loglike(
                    prediction[component.prediction_key],
                    component.observation,
                    parameters,
                )
            try:
                numeric_result = float(result)
            except (TypeError, ValueError, OverflowError) as error:
                raise RobertValidationError(
                    f"likelihood component '{component.name}' returned a non-scalar "
                    "log likelihood"
                ) from error
            if not np.isfinite(numeric_result):
                raise RobertValidationError(
                    f"likelihood component '{component.name}' returned a "
                    "non-finite log likelihood"
                )
            values[component.name] = numeric_result
        return values

    def loglike_by_component(
        self,
        prediction: Mapping[str, object],
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return exact finite log likelihoods in component order.

        The method is intentionally strict.  A missing prediction or a
        non-finite component result raises a ROBERT validation error.  The
        scalar :meth:`loglike` method converts such model failures to the
        configured invalid-model floor, as required by sampler callbacks.
        """

        return self._strict_loglike_by_component(prediction, parameters)

    def loglike(
        self,
        prediction: Mapping[str, object],
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Return the finite sum of all component log likelihoods."""

        try:
            values = self._strict_loglike_by_component(prediction, parameters)
            total = float(np.sum(np.fromiter(values.values(), dtype=float)))
        except _MODEL_ERRORS:
            return float(self.invalid_model_loglike)
        if not np.isfinite(total):
            return float(self.invalid_model_loglike)
        return total


@dataclass(frozen=True)
class HeterogeneousRetrievalProblem:
    """Sampler-facing retrieval problem for heterogeneous predictions."""

    name: str
    parameters: RetrievalParameterSet
    forward_model: HeterogeneousForwardEvaluator
    likelihood: HeterogeneousLikelihood
    invalid_loglike: float = float("-inf")
    metadata: Mapping[str, str] = field(default_factory=dict)
    opacity_identifiers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "retrieval problem name"))
        if not isinstance(self.parameters, RetrievalParameterSet):
            raise RobertValidationError(
                "heterogeneous retrieval parameters must be a RetrievalParameterSet"
            )
        if not callable(self.forward_model):
            raise RobertValidationError("heterogeneous forward_model must be callable")
        if not isinstance(self.likelihood, HeterogeneousLikelihood):
            raise RobertValidationError(
                "heterogeneous retrieval likelihood must be a HeterogeneousLikelihood"
            )
        object.__setattr__(
            self,
            "invalid_loglike",
            _invalid_loglike(self.invalid_loglike, "invalid_loglike"),
        )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(
            self,
            "opacity_identifiers",
            immutable_mapping(self.opacity_identifiers),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names in sampler vector order."""

        return self.parameters.names

    @property
    def ndim(self) -> int:
        """Number of retrieval parameters."""

        return self.parameters.ndim

    def prior_transform(self, cube: ArrayLike) -> NDArray[np.float64]:
        """Map a unit-cube vector into physical parameter space."""

        return self.parameters.transform(cube)

    def parameter_mapping(self, vector: ArrayLike) -> dict[str, float]:
        """Convert a sampler vector into named parameters."""

        return self.parameters.vector_to_mapping(vector)

    def predict(
        self,
        parameters: Mapping[str, float] | ArrayLike,
    ) -> Mapping[str, object]:
        """Evaluate the forward model and require a named prediction mapping."""

        if isinstance(parameters, Mapping):
            parameter_values = {
                str(key): float(value) for key, value in parameters.items()
            }
        else:
            parameter_values = self.parameter_mapping(parameters)
        prediction = self.forward_model(parameter_values)
        if not isinstance(prediction, Mapping):
            raise RobertValidationError(
                "heterogeneous forward_model must return a named mapping"
            )
        return prediction

    def _invalid_component_values(self) -> dict[str, float]:
        return {
            component.name: float(self.invalid_loglike)
            for component in self.likelihood.components
        }

    def log_likelihood_by_component_from_vector(
        self,
        vector: ArrayLike,
    ) -> dict[str, float]:
        """Evaluate named component terms, returning invalid terms on model errors."""

        try:
            parameters = self.parameter_mapping(vector)
            prediction = self.predict(parameters)
            return self.likelihood.loglike_by_component(prediction, parameters)
        except _MODEL_ERRORS:
            return self._invalid_component_values()

    def log_likelihood_from_vector(self, vector: ArrayLike) -> float:
        """Evaluate the summed likelihood for a sampler vector."""

        try:
            parameters = self.parameter_mapping(vector)
            prediction = self.predict(parameters)
            loglike = self.likelihood.loglike(prediction, parameters)
        except _MODEL_ERRORS:
            return float(self.invalid_loglike)
        if not np.isfinite(loglike):
            return float(self.invalid_loglike)
        return float(loglike)

    def log_prior_from_vector(self, vector: ArrayLike) -> float:
        """Evaluate the summed log prior density."""

        try:
            return self.parameters.log_prior_from_vector(vector)
        except RobertValidationError:
            return float("-inf")

    def log_posterior_from_vector(self, vector: ArrayLike) -> float:
        """Evaluate log prior plus log likelihood for a sampler vector."""

        log_prior = self.log_prior_from_vector(vector)
        if not np.isfinite(log_prior):
            return float("-inf")
        log_likelihood = self.log_likelihood_from_vector(vector)
        if not np.isfinite(log_likelihood):
            return float("-inf")
        return float(log_prior + log_likelihood)


__all__ = [
    "HeterogeneousForwardEvaluator",
    "HeterogeneousLikelihood",
    "HeterogeneousLikelihoodComponent",
    "HeterogeneousRetrievalProblem",
]
