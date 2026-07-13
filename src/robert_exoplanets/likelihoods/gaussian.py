"""Independent Gaussian likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.instruments import Observation


def _prediction_spectrum(prediction: Spectrum | Any) -> Spectrum:
    if isinstance(prediction, Spectrum):
        return prediction
    observed_spectrum = getattr(prediction, "observed_spectrum", None)
    if isinstance(observed_spectrum, Spectrum):
        return observed_spectrum
    raise RobertValidationError(
        "prediction must be a Spectrum or expose observed_spectrum"
    )


@dataclass(frozen=True)
class GaussianLikelihood:
    """Independent Gaussian log likelihood with optional offset and jitter."""

    name: str = "independent-gaussian"
    include_normalization: bool = False
    offset_parameter: str | None = "offset"
    jitter_parameter: str | None = "jitter"
    uncertainty_scale_parameter: str | None = None
    uncertainty_scale: float = 1.0
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0

    def __post_init__(self) -> None:
        if self.coordinate_rtol < 0.0 or not np.isfinite(self.coordinate_rtol):
            raise RobertValidationError(
                "coordinate_rtol must be finite and non-negative"
            )
        if self.coordinate_atol < 0.0 or not np.isfinite(self.coordinate_atol):
            raise RobertValidationError(
                "coordinate_atol must be finite and non-negative"
            )
        if not np.isfinite(self.uncertainty_scale) or self.uncertainty_scale <= 0.0:
            raise RobertValidationError("uncertainty_scale must be finite and positive")

    def loglike(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate the log likelihood for one observed spectrum."""

        selected_model, selected_data, selected_uncertainty = self.effective_inputs(
            prediction,
            observation,
            parameters,
        )
        if not np.all(np.isfinite(selected_model)):
            return float(self.invalid_model_loglike)

        variance = np.square(selected_uncertainty)
        residual = selected_data - selected_model
        loglike = -0.5 * np.sum(np.square(residual) / variance)
        if self.include_normalization:
            loglike -= 0.5 * np.sum(np.log(2.0 * np.pi * variance))
        return float(loglike)

    def effective_inputs(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return masked model, data, and effective uncertainty arrays.

        This is the shared Gaussian data contract used by likelihood and
        optimal-estimation calculations. Model predictions must already be on
        the observation spectral grid; resampling belongs in the instrument
        response.
        """

        spectrum = _prediction_spectrum(prediction)
        parameter_values = {} if parameters is None else parameters

        if spectrum.values.shape != observation.flux.shape:
            raise RobertValidationError("prediction and observation shapes must match")
        if spectrum.spectral_grid.unit != observation.wavelength_unit:
            raise RobertValidationError(
                "prediction and observation wavelength units must match"
            )
        if not np.allclose(
            spectrum.spectral_grid.values,
            observation.wavelength,
            rtol=self.coordinate_rtol,
            atol=self.coordinate_atol,
        ):
            raise RobertValidationError(
                "prediction must be evaluated on the observation wavelength grid"
            )
        if spectrum.unit != observation.flux_unit:
            raise RobertValidationError("prediction and observation units must match")
        if spectrum.observable != observation.observable:
            raise RobertValidationError(
                "prediction and observation observables must match"
            )

        model_values = np.array(spectrum.values, dtype=float, copy=True)
        if self.offset_parameter and self.offset_parameter in parameter_values:
            offset = float(parameter_values[self.offset_parameter])
            if not np.isfinite(offset):
                raise RobertValidationError("offset parameter must be finite")
            model_values += offset

        uncertainty = np.array(observation.uncertainty, dtype=float, copy=True)
        scale = float(self.uncertainty_scale)
        if (
            self.uncertainty_scale_parameter
            and self.uncertainty_scale_parameter in parameter_values
        ):
            scale *= float(parameter_values[self.uncertainty_scale_parameter])
        if not np.isfinite(scale) or scale <= 0.0:
            raise RobertValidationError(
                "uncertainty scale parameter must produce a finite positive scale"
            )
        uncertainty *= scale
        if self.jitter_parameter and self.jitter_parameter in parameter_values:
            jitter = float(parameter_values[self.jitter_parameter])
            if not np.isfinite(jitter) or jitter < 0.0:
                raise RobertValidationError(
                    "jitter parameter must be finite and non-negative"
                )
            uncertainty = np.sqrt(np.square(uncertainty) + jitter**2)

        valid = np.ones(observation.n_points, dtype=bool)
        if observation.mask is not None:
            valid = np.array(observation.mask, dtype=bool, copy=True)
        if not np.any(valid):
            raise RobertValidationError(
                "likelihood mask excludes all observation points"
            )

        selected_model = np.array(model_values[valid], dtype=float, copy=True)
        selected_data = np.array(observation.flux[valid], dtype=float, copy=True)
        selected_uncertainty = np.array(uncertainty[valid], dtype=float, copy=True)
        for array in (selected_model, selected_data, selected_uncertainty):
            array.setflags(write=False)
        return selected_model, selected_data, selected_uncertainty
