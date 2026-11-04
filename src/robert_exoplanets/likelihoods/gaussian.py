"""Independent Gaussian likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, cast

import numpy as np

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.instruments import Observation

from .correlated import CorrelatedGaussianLikelihood, DenseCovariance


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

    # OE checks the observation before it uses the independent effective-input
    # contract. Covariance-backed observations are rejected there.
    supports_optimal_estimation: ClassVar[bool] = True
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

        correlated = self._correlated_likelihood(observation, parameters)
        if correlated is not None:
            return correlated.loglike(prediction, observation, parameters)

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

    def pointwise_loglike(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return one independent Gaussian log-likelihood term per valid datum.

        The returned terms use the same mask, nuisance parameters, uncertainty
        scaling, and optional normalization as :meth:`loglike`. Their sum is
        therefore the scalar likelihood for every finite model prediction.
        """

        correlated = self._correlated_likelihood(observation, parameters)
        if correlated is not None:
            return correlated.pointwise_loglike(prediction, observation, parameters)

        selected_model, selected_data, selected_uncertainty = self.effective_inputs(
            prediction,
            observation,
            parameters,
        )
        if not np.all(np.isfinite(selected_model)):
            output = np.full(
                selected_model.shape, self.invalid_model_loglike, dtype=float
            )
            output.setflags(write=False)
            return output

        variance = np.square(selected_uncertainty)
        residual = selected_data - selected_model
        terms = -0.5 * np.square(residual) / variance
        if self.include_normalization:
            terms -= 0.5 * np.log(2.0 * np.pi * variance)
        output = np.array(terms, dtype=float, copy=True)
        output.setflags(write=False)
        return output

    def chi_square(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Return the exact independent-Gaussian residual statistic."""

        correlated = self._correlated_likelihood(observation, parameters)
        if correlated is not None:
            return correlated.chi_square(prediction, observation, parameters)

        model, data, uncertainty = self.effective_inputs(
            prediction,
            observation,
            parameters,
        )
        if not np.all(np.isfinite(model)):
            raise RobertValidationError("chi-square requires a finite model")
        residual = (data - model) / uncertainty
        value = float(np.dot(residual, residual))
        if not np.isfinite(value) or value < 0.0:
            raise RobertValidationError("chi-square must be finite and non-negative")
        return value

    def effective_residual_rank(
        self,
        observation: Observation,
    ) -> int:
        """Return the number of retained independent residual dimensions.

        Every unmasked datum contributes one residual dimension for an
        independent Gaussian likelihood.
        """

        valid = (
            np.ones(observation.n_points, dtype=bool)
            if observation.mask is None
            else np.asarray(observation.mask, dtype=bool)
        )
        count = int(np.count_nonzero(valid))
        if count <= 0:
            raise RobertValidationError(
                "likelihood mask excludes all observation points"
            )
        return count

    def effective_inputs(
        self,
        prediction: Spectrum | Any,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return masked model, data, and effective uncertainty arrays.

        For covariance-backed observations, the returned uncertainty is the
        marginal standard deviation. This is useful for residual plots, but it
        does not represent the full covariance. Optimal estimation rejects
        covariance-backed observations. Model predictions must already be on
        the observation spectral grid; resampling belongs in the instrument
        response.
        """

        spectrum = _prediction_spectrum(prediction)
        parameter_values = {} if parameters is None else parameters
        correlated = self._correlated_likelihood(observation, parameters)
        if correlated is not None:
            return correlated.effective_inputs(prediction, observation, parameters)

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
        scale, jitter = self._noise_scale_and_jitter(parameter_values)
        uncertainty *= scale
        if jitter > 0.0:
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

    def _noise_scale_and_jitter(
        self,
        parameters: Mapping[str, float] | None,
    ) -> tuple[float, float]:
        """Return the current uncertainty scale and additive jitter."""

        parameter_values = {} if parameters is None else parameters
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
        jitter = 0.0
        if self.jitter_parameter and self.jitter_parameter in parameter_values:
            jitter = float(parameter_values[self.jitter_parameter])
            if not np.isfinite(jitter) or jitter < 0.0:
                raise RobertValidationError(
                    "jitter parameter must be finite and non-negative"
                )
        return scale, jitter

    def _correlated_likelihood(
        self,
        observation: Observation,
        parameters: Mapping[str, float] | None,
    ) -> CorrelatedGaussianLikelihood | None:
        """Build the existing correlated likelihood for covariance-backed data."""

        if observation.covariance is None:
            return None
        scale, jitter = self._noise_scale_and_jitter(parameters)
        base_covariance = observation.covariance
        if jitter == 0.0:
            covariance = (
                base_covariance
                if scale == 1.0
                else np.asarray(base_covariance, dtype=float) * scale**2
            )
            cholesky = observation.covariance_cholesky
            if cholesky is None:
                raise RobertValidationError(
                    "observation covariance is missing its validated Cholesky factor"
                )
            if scale != 1.0:
                cholesky = np.asarray(cholesky, dtype=float) * scale
            operator = DenseCovariance._from_precomputed_cholesky(
                covariance,
                cholesky,
                name="observation-covariance",
            )
        else:
            covariance = np.asarray(base_covariance, dtype=float) * scale**2
            covariance = np.array(covariance, dtype=float, copy=True)
            diagonal = np.diag_indices_from(covariance)
            covariance[diagonal] += jitter**2
            operator = DenseCovariance(
                covariance,
                name="scaled-observation-covariance-with-jitter",
            )
        return CorrelatedGaussianLikelihood(
            covariance=cast(Any, operator),
            include_normalization=self.include_normalization,
            offset_parameter=self.offset_parameter,
            invalid_model_loglike=self.invalid_model_loglike,
            coordinate_rtol=self.coordinate_rtol,
            coordinate_atol=self.coordinate_atol,
            metadata={"source": "observation.covariance"},
        )
