"""Sampler-independent retrieval problem for multiple named datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertError, RobertValidationError, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import ObservationCollection
from robert_exoplanets.likelihoods import MultiDatasetGaussianLikelihood

from .priors import RetrievalParameterSet

MultiDatasetEvaluator = Callable[[Mapping[str, float]], object]


@dataclass(frozen=True)
class MultiDatasetRetrievalProblem:
    """Share atmospheric parameters while retaining per-dataset likelihoods."""

    name: str
    observations: ObservationCollection
    parameters: RetrievalParameterSet
    forward_model: MultiDatasetEvaluator
    likelihood: MultiDatasetGaussianLikelihood = field(
        default_factory=MultiDatasetGaussianLikelihood
    )
    invalid_loglike: float = float("-inf")
    metadata: Mapping[str, str] = field(default_factory=dict)
    opacity_identifiers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError(
                "multi-dataset retrieval problem name must not be empty"
            )
        nuisance_parameters = {
            parameter
            for dataset in self.observations.datasets
            for parameter in (
                dataset.offset_parameter,
                dataset.jitter_parameter,
                dataset.uncertainty_scale_parameter,
            )
            if parameter is not None
        }
        missing = sorted(nuisance_parameters - set(self.parameters.names))
        if missing:
            raise RobertValidationError(
                "retrieval parameter set is missing dataset nuisance parameters: "
                + ", ".join(missing)
            )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(
            self, "opacity_identifiers", immutable_mapping(self.opacity_identifiers)
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.parameters.names

    @property
    def ndim(self) -> int:
        return self.parameters.ndim

    def prior_transform(self, cube: ArrayLike) -> NDArray[np.float64]:
        return self.parameters.transform(cube)

    def parameter_mapping(self, vector: ArrayLike) -> dict[str, float]:
        return self.parameters.vector_to_mapping(vector)

    def predict(self, parameters: Mapping[str, float] | ArrayLike) -> object:
        if isinstance(parameters, Mapping):
            values = {str(key): float(value) for key, value in parameters.items()}
        else:
            values = self.parameter_mapping(parameters)
        return self.forward_model(values)

    def model_spectra(
        self,
        parameters: Mapping[str, float] | ArrayLike,
    ) -> Mapping[str, Spectrum]:
        prediction = self.predict(parameters)
        spectra = (
            prediction
            if isinstance(prediction, Mapping)
            else getattr(prediction, "spectra", None)
        )
        if not isinstance(spectra, Mapping) or any(
            not isinstance(value, Spectrum) for value in spectra.values()
        ):
            raise RobertValidationError(
                "multi-dataset forward model must expose named spectra"
            )
        return spectra

    def gaussian_inputs_from_vector(
        self,
        vector: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return deterministic flattened Gaussian arrays for optimal estimation."""

        parameters = self.parameter_mapping(vector)
        inputs = self.likelihood.effective_inputs_by_dataset(
            self.predict(parameters),
            self.observations,
            parameters,
        )
        ordered = [inputs[dataset.name] for dataset in self.observations.datasets]
        return tuple(
            np.concatenate([dataset_inputs[index] for dataset_inputs in ordered])
            for index in range(3)
        )

    def log_likelihood_from_vector(self, vector: ArrayLike) -> float:
        try:
            parameters = self.parameter_mapping(vector)
            prediction = self.forward_model(parameters)
            loglike = self.likelihood.loglike(prediction, self.observations, parameters)
        except (RobertError, ValueError, FloatingPointError, OverflowError):
            return float(self.invalid_loglike)
        if not np.isfinite(loglike):
            return float(self.invalid_loglike)
        return float(loglike)

    def log_prior_from_vector(self, vector: ArrayLike) -> float:
        try:
            return self.parameters.log_prior_from_vector(vector)
        except RobertValidationError:
            return float("-inf")

    def log_posterior_from_vector(self, vector: ArrayLike) -> float:
        log_prior = self.log_prior_from_vector(vector)
        if not np.isfinite(log_prior):
            return float("-inf")
        log_likelihood = self.log_likelihood_from_vector(vector)
        if not np.isfinite(log_likelihood):
            return float("-inf")
        return float(log_prior + log_likelihood)


__all__ = ["MultiDatasetRetrievalProblem"]
