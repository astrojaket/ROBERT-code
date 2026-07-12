"""Independent Gaussian likelihood across named spectral datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.instruments import ObservationCollection

from .gaussian import GaussianLikelihood


def _prediction_mapping(prediction: Any) -> Mapping[str, Spectrum]:
    if isinstance(prediction, Mapping):
        spectra = prediction
    else:
        spectra = getattr(prediction, "spectra", None)
    if not isinstance(spectra, Mapping):
        raise RobertValidationError("multi-dataset prediction must be a spectrum mapping or expose spectra")
    if any(not isinstance(value, Spectrum) for value in spectra.values()):
        raise RobertValidationError("every multi-dataset prediction must be a Spectrum")
    return spectra


@dataclass(frozen=True)
class MultiDatasetGaussianLikelihood:
    """Sum independent Gaussian terms with dataset-specific offset/jitter."""

    name: str = "multi-dataset-independent-gaussian"
    include_normalization: bool = False
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0

    def loglike(
        self,
        prediction: Any,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        spectra = _prediction_mapping(prediction)
        if set(spectra) != set(observations.names):
            raise RobertValidationError("prediction dataset names must match observation collection")
        total = 0.0
        for dataset in observations.datasets:
            likelihood = GaussianLikelihood(
                include_normalization=self.include_normalization,
                offset_parameter=dataset.offset_parameter,
                jitter_parameter=dataset.jitter_parameter,
                invalid_model_loglike=self.invalid_model_loglike,
                coordinate_rtol=self.coordinate_rtol,
                coordinate_atol=self.coordinate_atol,
            )
            value = likelihood.loglike(
                spectra[dataset.name],
                dataset.observation,
                parameters,
            )
            if not np.isfinite(value):
                return float(self.invalid_model_loglike)
            total += value
        return float(total)

    def effective_inputs_by_dataset(
        self,
        prediction: Any,
        observations: ObservationCollection,
        parameters: Mapping[str, float] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return masked Gaussian arrays without erasing dataset identity."""

        spectra = _prediction_mapping(prediction)
        if set(spectra) != set(observations.names):
            raise RobertValidationError("prediction dataset names must match observation collection")
        output = {}
        for dataset in observations.datasets:
            likelihood = GaussianLikelihood(
                include_normalization=self.include_normalization,
                offset_parameter=dataset.offset_parameter,
                jitter_parameter=dataset.jitter_parameter,
                invalid_model_loglike=self.invalid_model_loglike,
                coordinate_rtol=self.coordinate_rtol,
                coordinate_atol=self.coordinate_atol,
            )
            output[dataset.name] = likelihood.effective_inputs(
                spectra[dataset.name],
                dataset.observation,
                parameters,
            )
        return output


__all__ = ["MultiDatasetGaussianLikelihood"]
