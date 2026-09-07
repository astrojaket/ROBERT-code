"""Independent, dataset-specific likelihoods for one joint retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import Observation, ObservationCollection


class SingleDatasetLikelihood(Protocol):
    """Structural contract for one spectrum likelihood."""

    name: str

    def loglike(
        self,
        prediction: object,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float: ...

    def effective_inputs(
        self,
        prediction: object,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def pointwise_loglike(
        self,
        prediction: object,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> np.ndarray: ...


class MultiDatasetLikelihood(Protocol):
    """Structural contract used by ``MultiDatasetRetrievalProblem``."""

    name: str

    def loglike(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> float: ...

    def effective_inputs_by_dataset(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]: ...

    def pointwise_loglike(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> np.ndarray: ...


def _prediction_mapping(prediction: Any) -> Mapping[str, Spectrum]:
    if isinstance(prediction, Mapping):
        spectra = prediction
    else:
        spectra = getattr(prediction, "spectra", None)
    if not isinstance(spectra, Mapping):
        raise RobertValidationError(
            "multi-dataset prediction must be a spectrum mapping or expose spectra"
        )
    if any(not isinstance(value, Spectrum) for value in spectra.values()):
        raise RobertValidationError("every multi-dataset prediction must be a Spectrum")
    return spectra


@dataclass(frozen=True)
class MixedMultiDatasetLikelihood:
    """Sum independent likelihood types across named data sets.

    A typical combined-resolution retrieval uses ``GaussianLikelihood`` for
    calibrated low-resolution data and a prepared continuum or
    cross-correlation likelihood for high-resolution data.
    """

    likelihoods: Mapping[str, SingleDatasetLikelihood]
    name: str = "mixed-multi-dataset-independent"
    invalid_model_loglike: float = float("-inf")
    _likelihoods: Mapping[str, SingleDatasetLikelihood] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("multi-dataset likelihood name must not be empty")
        normalized = {
            str(dataset_name).strip(): likelihood
            for dataset_name, likelihood in self.likelihoods.items()
        }
        if not normalized or any(not dataset_name for dataset_name in normalized):
            raise RobertValidationError(
                "mixed likelihoods require non-empty dataset names"
            )
        for likelihood in normalized.values():
            for method in ("loglike", "effective_inputs", "pointwise_loglike"):
                if not callable(getattr(likelihood, method, None)):
                    raise RobertValidationError(
                        f"every dataset likelihood must implement {method}"
                    )
            if not str(getattr(likelihood, "name", "")).strip():
                raise RobertValidationError(
                    "every dataset likelihood must have a non-empty name"
                )
        invalid = float(self.invalid_model_loglike)
        if np.isnan(invalid) or invalid == np.inf:
            raise RobertValidationError(
                "invalid_model_loglike must be finite or negative infinity"
            )
        immutable = immutable_mapping(normalized)
        object.__setattr__(self, "likelihoods", immutable)
        object.__setattr__(self, "_likelihoods", immutable)
        object.__setattr__(self, "invalid_model_loglike", invalid)

    @property
    def supports_optimal_estimation(self) -> bool:
        """Return whether every child has the independent OE contract."""

        return all(
            bool(getattr(likelihood, "supports_optimal_estimation", False))
            for likelihood in self.likelihoods.values()
        )

    def _validated_inputs(
        self,
        prediction: object,
        observations: ObservationCollection,
    ) -> Mapping[str, Spectrum]:
        spectra = _prediction_mapping(prediction)
        expected = set(observations.names)
        if set(spectra) != expected:
            raise RobertValidationError(
                "prediction dataset names must match observation collection"
            )
        if set(self.likelihoods) != expected:
            raise RobertValidationError(
                "likelihood dataset names must match observation collection"
            )
        return spectra

    def loglike(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Return the sum of independent named likelihoods."""

        spectra = self._validated_inputs(prediction, observations)
        total = 0.0
        for dataset in observations.datasets:
            value = self.likelihoods[dataset.name].loglike(
                spectra[dataset.name],
                dataset.observation,
                parameters,
            )
            if not np.isfinite(value):
                return float(self.invalid_model_loglike)
            total += float(value)
        return float(total)

    def effective_inputs_by_dataset(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return the exact effective arrays used by each likelihood."""

        spectra = self._validated_inputs(prediction, observations)
        output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for dataset in observations.datasets:
            values = self.likelihoods[dataset.name].effective_inputs(
                spectra[dataset.name],
                dataset.observation,
                parameters,
            )
            if len(values) != 3:
                raise RobertValidationError(
                    "dataset effective inputs must contain model, data, and uncertainty"
                )
            output[dataset.name] = tuple(
                np.asarray(value, dtype=float) for value in values
            )  # type: ignore[assignment]
        return output

    @staticmethod
    def _has_covariance_interface(likelihood: SingleDatasetLikelihood) -> bool:
        """Return whether a likelihood needs a non-diagonal statistic hook."""

        return bool(
            callable(getattr(likelihood, "effective_covariance", None))
            or getattr(likelihood, "covariance", None) is not None
        )

    @staticmethod
    def _validated_effective_inputs(
        likelihood: SingleDatasetLikelihood,
        prediction: Spectrum,
        observation: Observation,
        parameters: Mapping[str, float] | None,
        dataset_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return validated arrays for the independent fallback statistic."""

        values = tuple(likelihood.effective_inputs(prediction, observation, parameters))
        if len(values) != 3:
            raise RobertValidationError(
                f"effective inputs for dataset {dataset_name!r} must contain "
                "model, data, and uncertainty"
            )
        model, data, uncertainty = (np.asarray(value, dtype=float) for value in values)
        if (
            model.ndim != 1
            or data.shape != model.shape
            or uncertainty.shape != model.shape
            or model.size == 0
        ):
            raise RobertValidationError(
                f"effective inputs for dataset {dataset_name!r} must be "
                "non-empty matching vectors"
            )
        if (
            not np.all(np.isfinite(model))
            or not np.all(np.isfinite(data))
            or not np.all(np.isfinite(uncertainty))
            or np.any(uncertainty <= 0.0)
        ):
            raise RobertValidationError(
                f"effective inputs for dataset {dataset_name!r} must be finite "
                "with positive uncertainty"
            )
        return model, data, uncertainty

    @staticmethod
    def _validated_chi_square(value: object, dataset_name: str) -> float:
        """Validate one exact non-negative dataset chi-square."""

        if isinstance(value, (bool, np.bool_)):
            raise RobertValidationError(
                f"chi-square for dataset {dataset_name!r} must be finite and non-negative"
            )
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise RobertValidationError(
                f"chi-square for dataset {dataset_name!r} must be finite and non-negative"
            ) from error
        if not np.isfinite(result) or result < 0.0:
            raise RobertValidationError(
                f"chi-square for dataset {dataset_name!r} must be finite and non-negative"
            )
        return result

    @staticmethod
    def _validated_rank(value: object, dataset_name: str) -> int:
        """Validate one non-negative integer residual rank."""

        if isinstance(value, (bool, np.bool_)):
            raise RobertValidationError(
                f"effective residual rank for dataset {dataset_name!r} "
                "must be a non-negative integer"
            )
        try:
            rank = int(value)
        except (TypeError, ValueError) as error:
            raise RobertValidationError(
                f"effective residual rank for dataset {dataset_name!r} "
                "must be a non-negative integer"
            ) from error
        if rank != value or rank < 0:
            raise RobertValidationError(
                f"effective residual rank for dataset {dataset_name!r} "
                "must be a non-negative integer"
            )
        return rank

    def chi_square_by_dataset(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return exact chi-square values while retaining dataset identity.

        A child likelihood can provide a ``chi_square`` method when its
        residuals are correlated or analytically profiled.  The independent
        effective-input calculation is used only for children without a
        covariance interface.  This prevents a covariance diagonal from being
        mistaken for the full likelihood.
        """

        spectra = self._validated_inputs(prediction, observations)
        output: dict[str, float] = {}
        for dataset in observations.datasets:
            likelihood = self.likelihoods[dataset.name]
            exact_method = getattr(likelihood, "chi_square", None)
            if callable(exact_method):
                value = exact_method(
                    spectra[dataset.name],
                    dataset.observation,
                    parameters,
                )
                output[dataset.name] = self._validated_chi_square(value, dataset.name)
                continue
            if self._has_covariance_interface(likelihood):
                raise RobertValidationError(
                    f"dataset likelihood {dataset.name!r} exposes covariance but "
                    "does not provide chi_square; diagonal effective inputs are "
                    "not an exact fallback"
                )
            model, data, uncertainty = self._validated_effective_inputs(
                likelihood,
                spectra[dataset.name],
                dataset.observation,
                parameters,
                dataset.name,
            )
            residual = (data - model) / uncertainty
            value = float(np.dot(residual, residual))
            output[dataset.name] = self._validated_chi_square(value, dataset.name)
        return output

    def effective_residual_rank_by_dataset(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, int]:
        """Return retained residual dimensions by dataset."""

        spectra = self._validated_inputs(prediction, observations)
        output: dict[str, int] = {}
        for dataset in observations.datasets:
            likelihood = self.likelihoods[dataset.name]
            exact_method = getattr(likelihood, "effective_residual_rank", None)
            if callable(exact_method):
                value = exact_method(
                    dataset.observation,
                )
                output[dataset.name] = self._validated_rank(value, dataset.name)
                continue
            if self._has_covariance_interface(likelihood):
                raise RobertValidationError(
                    f"dataset likelihood {dataset.name!r} exposes covariance but "
                    "does not provide effective_residual_rank"
                )
            model, _, _ = self._validated_effective_inputs(
                likelihood,
                spectra[dataset.name],
                dataset.observation,
                parameters,
                dataset.name,
            )
            output[dataset.name] = int(model.size)
        return output

    def pointwise_loglike_by_dataset(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return diagnostic terms while retaining data-set identity."""

        spectra = self._validated_inputs(prediction, observations)
        return {
            dataset.name: np.asarray(
                self.likelihoods[dataset.name].pointwise_loglike(
                    spectra[dataset.name],
                    dataset.observation,
                    parameters,
                ),
                dtype=float,
            )
            for dataset in observations.datasets
        }

    def pointwise_loglike(
        self,
        prediction: object,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return all diagnostic terms in observation-collection order."""

        by_dataset = self.pointwise_loglike_by_dataset(
            prediction,
            observations,
            parameters,
        )
        output = np.concatenate(
            [by_dataset[dataset.name] for dataset in observations.datasets]
        )
        output.setflags(write=False)
        return output


__all__ = [
    "MixedMultiDatasetLikelihood",
    "MultiDatasetLikelihood",
    "SingleDatasetLikelihood",
]
