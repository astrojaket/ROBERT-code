"""Sampler-independent retrieval problem definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertError, RobertValidationError, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import Observation
from robert_exoplanets.likelihoods import GaussianLikelihood

from .priors import RetrievalParameterSet


class ObservedSpectrumPrediction(Protocol):
    """Structural contract for forward outputs carrying an observed spectrum."""

    observed_spectrum: Spectrum


ForwardPrediction = Spectrum | ObservedSpectrumPrediction
ForwardEvaluator = Callable[[Mapping[str, float]], ForwardPrediction]


@dataclass(frozen=True)
class RetrievalProblem:
    """Bundle a forward evaluator, observation, likelihood, and priors."""

    name: str
    observation: Observation
    parameters: RetrievalParameterSet
    forward_model: ForwardEvaluator
    likelihood: GaussianLikelihood = field(default_factory=GaussianLikelihood)
    invalid_loglike: float = float("-inf")
    metadata: Mapping[str, str] = field(default_factory=dict)
    opacity_identifiers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("retrieval problem name must not be empty")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(self, "opacity_identifiers", immutable_mapping(self.opacity_identifiers))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names in vector order."""

        return self.parameters.names

    @property
    def ndim(self) -> int:
        """Number of retrieval parameters."""

        return self.parameters.ndim

    def prior_transform(self, cube: ArrayLike) -> NDArray[np.float64]:
        """Map a unit-cube vector into the retrieval parameter vector."""

        return self.parameters.transform(cube)

    def parameter_mapping(self, vector: ArrayLike) -> dict[str, float]:
        """Convert a parameter vector into a mapping."""

        return self.parameters.vector_to_mapping(vector)

    def predict(self, parameters: Mapping[str, float] | ArrayLike) -> ForwardPrediction:
        """Evaluate the forward model for a mapping or vector."""

        if isinstance(parameters, Mapping):
            parameter_values = {str(key): float(value) for key, value in parameters.items()}
        else:
            parameter_values = self.parameter_mapping(parameters)
        return self.forward_model(parameter_values)

    def model_spectrum(self, parameters: Mapping[str, float] | ArrayLike) -> Spectrum:
        """Return the observed-grid model spectrum."""

        prediction = self.predict(parameters)
        if isinstance(prediction, Spectrum):
            return prediction
        observed_spectrum = getattr(prediction, "observed_spectrum", None)
        if isinstance(observed_spectrum, Spectrum):
            return observed_spectrum
        raise RobertValidationError("forward model must return a Spectrum or expose observed_spectrum")

    def model_values_from_vector(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Return model values for a vector in retrieval order."""

        spectrum = self.model_spectrum(vector)
        values = np.array(spectrum.values, dtype=float, copy=True)
        values.setflags(write=False)
        return values

    def gaussian_inputs_from_vector(
        self,
        vector: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return masked Gaussian model, data, and uncertainty for a vector."""

        parameter_values = self.parameter_mapping(vector)
        prediction = self.predict(parameter_values)
        model, data, uncertainty = self.likelihood.effective_inputs(
            prediction,
            self.observation,
            parameter_values,
        )
        return (
            np.asarray(model, dtype=float),
            np.asarray(data, dtype=float),
            np.asarray(uncertainty, dtype=float),
        )

    def log_likelihood_from_vector(self, vector: ArrayLike) -> float:
        """Evaluate the log likelihood for a retrieval vector."""

        try:
            parameters = self.parameter_mapping(vector)
            prediction = self.predict(parameters)
            loglike = self.likelihood.loglike(prediction, self.observation, parameters)
        except (RobertError, ValueError, FloatingPointError, OverflowError):
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
        """Evaluate log likelihood plus log prior density."""

        log_prior = self.log_prior_from_vector(vector)
        if not np.isfinite(log_prior):
            return float("-inf")
        log_likelihood = self.log_likelihood_from_vector(vector)
        if not np.isfinite(log_likelihood):
            return float("-inf")
        return float(log_prior + log_likelihood)
