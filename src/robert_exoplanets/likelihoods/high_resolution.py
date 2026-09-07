"""Likelihoods for high-resolution spectral data.

The classes in this module use the same one-dimensional contract as
``GaussianLikelihood``: a prediction is a :class:`~robert_exoplanets.core.Spectrum`
on the observation grid and the data are an
:class:`~robert_exoplanets.instruments.Observation`.

High-resolution spectra often contain an unknown smooth blaze or continuum.
``PreparedPolynomialContinuumProjection`` removes that component with a
fixed, weighted linear operator.  The operator is prepared once from the
observation and can then be applied to both the data and every trial model.
It stores a thin QR basis, not a dense ``N`` by ``N`` projection matrix.

The profiled continuum likelihood uses

``chi2 = min_a (d - m - X a)^T W (d - m - X a)``

where ``X`` is the polynomial design matrix and ``W`` is the diagonal
inverse-variance matrix.  The implementation evaluates this expression as
the squared norm of the projected, whitened residual.

The cross-correlation likelihood is a documented template-shape statistic
for detrended spectra.  After the optional fixed projection, let
``d_w = d / sigma`` and ``m_w = m / sigma``.  It defines

``rho = (d_w dot m_w) / sqrt((d_w dot d_w) (m_w dot m_w))``

and profiles a single template amplitude ``a``.  The residual statistic is

``chi2_profile = d_w dot d_w - (d_w dot m_w)^2 / (m_w dot m_w)``

(``a_hat = (d_w dot m_w) / (m_w dot m_w)``).  The returned value is
``-0.5 * chi2_profile`` plus the optional fixed Gaussian normalization.
This is a shape likelihood, not a replacement for a calibrated flux
likelihood.  It requires a non-zero data and template norm, and it uses only
unmasked points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.instruments import Observation


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
Prediction = Spectrum | Any


def _readonly_float_array(values: ArrayLike, name: str, *, ndim: int = 1) -> FloatArray:
    """Return a finite, copied, read-only floating-point array."""

    array = np.array(values, dtype=float, copy=True)
    if array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_int_array(values: ArrayLike, name: str) -> IntArray:
    """Return a copied, read-only one-dimensional integer array."""

    array = np.array(values, dtype=np.int64, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


def _positive_float(value: float, name: str) -> float:
    """Validate one finite positive scalar."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be finite and positive")
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite and positive") from error
    if not np.isfinite(normalised) or normalised <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return normalised


def _nonnegative_float(value: float, name: str) -> float:
    """Validate one finite non-negative scalar."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be finite and non-negative")
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite and non-negative") from error
    if not np.isfinite(normalised) or normalised < 0.0:
        raise RobertValidationError(f"{name} must be finite and non-negative")
    return normalised


def _degree(value: int, name: str = "degree") -> int:
    """Validate one non-negative integer polynomial degree."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be a non-negative integer")
    try:
        degree = int(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be a non-negative integer") from error
    if degree != value or degree < 0:
        raise RobertValidationError(f"{name} must be a non-negative integer")
    return degree


def _invalid_loglike(value: float) -> float:
    """Validate the deterministic invalid-model likelihood floor."""

    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError("invalid_model_loglike must be finite or -inf") from error
    if np.isnan(normalised) or normalised == np.inf:
        raise RobertValidationError("invalid_model_loglike must be finite or -inf")
    return normalised


def _validate_coordinate_tolerances(coordinate_rtol: float, coordinate_atol: float) -> tuple[float, float]:
    """Validate coordinate comparison tolerances."""

    rtol = _nonnegative_float(coordinate_rtol, "coordinate_rtol")
    atol = _nonnegative_float(coordinate_atol, "coordinate_atol")
    return rtol, atol


def _prediction_spectrum(prediction: Prediction) -> Spectrum:
    """Extract a spectrum from a direct spectrum or a forward prediction."""

    if isinstance(prediction, Spectrum):
        return prediction
    observed_spectrum = getattr(prediction, "observed_spectrum", None)
    if isinstance(observed_spectrum, Spectrum):
        return observed_spectrum
    raise RobertValidationError(
        "prediction must be a Spectrum or expose observed_spectrum"
    )


def _observation_valid_mask(observation: Observation) -> BoolArray:
    """Return the observation mask, or an all-valid mask when none is set."""

    if observation.mask is None:
        valid = np.ones(observation.n_points, dtype=bool)
    else:
        valid = np.array(observation.mask, dtype=bool, copy=True)
    if not np.any(valid):
        raise RobertValidationError("likelihood mask excludes all observation points")
    valid.setflags(write=False)
    return valid


def _validated_inputs(
    prediction: Prediction,
    observation: Observation,
    *,
    coordinate_rtol: float,
    coordinate_atol: float,
) -> tuple[Spectrum, FloatArray, FloatArray, FloatArray, BoolArray]:
    """Validate the Spectrum/Observation contract and return copied arrays."""

    spectrum = _prediction_spectrum(prediction)
    if spectrum.values.shape != observation.flux.shape:
        raise RobertValidationError("prediction and observation shapes must match")
    if spectrum.spectral_grid.unit != observation.wavelength_unit:
        raise RobertValidationError("prediction and observation wavelength units must match")
    if not np.allclose(
        spectrum.spectral_grid.values,
        observation.wavelength,
        rtol=coordinate_rtol,
        atol=coordinate_atol,
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

    model = np.array(spectrum.values, dtype=float, copy=True)
    data = np.array(observation.flux, dtype=float, copy=True)
    uncertainty = np.array(observation.uncertainty, dtype=float, copy=True)
    valid = _observation_valid_mask(observation)
    for array in (model, data, uncertainty):
        array.setflags(write=False)
    return spectrum, model, data, uncertainty, valid


def _readonly_result(values: ArrayLike) -> FloatArray:
    """Copy a result to a read-only floating-point array."""

    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _parameter_float(
    parameters: Mapping[str, float],
    name: str | None,
    default: float,
    label: str,
) -> float:
    """Read and validate one optional runtime scalar parameter."""

    value: Any = default
    if name is not None and name in parameters:
        value = parameters[name]
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{label} must be finite") from error
    if not np.isfinite(normalised):
        raise RobertValidationError(f"{label} must be finite")
    return normalised


def _parameter_positive_float(
    parameters: Mapping[str, float],
    name: str | None,
    default: float,
    label: str,
) -> float:
    """Read and validate one optional runtime positive scalar parameter."""

    value = _parameter_float(parameters, name, default, label)
    if value <= 0.0:
        raise RobertValidationError(f"{label} must be finite and positive")
    return value


def _normalization_term(uncertainty: FloatArray) -> float:
    """Return the independent-Gaussian normalization for selected data."""

    variance = np.square(uncertainty)
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * variance)))


@dataclass(frozen=True)
class CalibratedDirectFluxLikelihood:
    """Independent Gaussian likelihood for calibrated direct fluxes.

    The model entering the residual is

    ``model_calibrated = calibration_scale * model + calibration_offset``.

    ``calibration_scale`` and ``calibration_offset`` are fixed values.  The
    corresponding ``*_parameter`` fields can replace either value at runtime
    from the parameter mapping.  The observation mask is applied before the
    weighted residual is evaluated.  This class is therefore compatible with
    the existing ``Spectrum``/``Observation`` contract and can be used as the
    likelihood in a retrieval problem.

    ``include_normalization=False`` keeps the same convention as
    :class:`~robert_exoplanets.likelihoods.gaussian.GaussianLikelihood`.
    """

    name: str = "calibrated-direct-flux"
    include_normalization: bool = False
    calibration_scale: float = 1.0
    calibration_offset: float = 0.0
    calibration_scale_parameter: str | None = None
    calibration_offset_parameter: str | None = None
    # Short aliases are useful when sharing a calibration manifest with the
    # existing GaussianLikelihood.  They are resolved once at construction.
    scale: float | None = None
    offset: float | None = None
    scale_parameter: str | None = None
    offset_parameter: str | None = None
    uncertainty_scale: float = 1.0
    uncertainty_scale_parameter: str | None = None
    jitter_parameter: str | None = None
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("likelihood name must not be empty")
        scale = _positive_float(self.calibration_scale, "calibration_scale")
        offset = float(self.calibration_offset)
        if not np.isfinite(offset):
            raise RobertValidationError("calibration_offset must be finite")
        if self.scale is not None:
            if self.calibration_scale != 1.0:
                raise RobertValidationError(
                    "scale and calibration_scale must not both be configured"
                )
            scale = _positive_float(self.scale, "scale")
        if self.offset is not None:
            if self.calibration_offset != 0.0:
                raise RobertValidationError(
                    "offset and calibration_offset must not both be configured"
                )
            offset = float(self.offset)
            if not np.isfinite(offset):
                raise RobertValidationError("offset must be finite")

        if self.calibration_scale_parameter is not None and self.scale_parameter is not None:
            raise RobertValidationError(
                "scale_parameter and calibration_scale_parameter must not both be configured"
            )
        if self.calibration_offset_parameter is not None and self.offset_parameter is not None:
            raise RobertValidationError(
                "offset_parameter and calibration_offset_parameter must not both be configured"
            )
        for parameter_name, label in (
            (self.calibration_scale_parameter, "calibration_scale_parameter"),
            (self.calibration_offset_parameter, "calibration_offset_parameter"),
            (self.scale_parameter, "scale_parameter"),
            (self.offset_parameter, "offset_parameter"),
            (self.uncertainty_scale_parameter, "uncertainty_scale_parameter"),
            (self.jitter_parameter, "jitter_parameter"),
        ):
            if parameter_name is not None and not parameter_name:
                raise RobertValidationError(f"{label} must not be empty")

        uncertainty_scale = _positive_float(self.uncertainty_scale, "uncertainty_scale")
        rtol, atol = _validate_coordinate_tolerances(
            self.coordinate_rtol,
            self.coordinate_atol,
        )
        object.__setattr__(self, "calibration_scale", scale)
        object.__setattr__(self, "calibration_offset", offset)
        object.__setattr__(self, "uncertainty_scale", uncertainty_scale)
        object.__setattr__(self, "invalid_model_loglike", _invalid_loglike(self.invalid_model_loglike))
        object.__setattr__(self, "coordinate_rtol", rtol)
        object.__setattr__(self, "coordinate_atol", atol)

    def _effective_inputs(
        self,
        prediction: Prediction,
        observation: Observation,
        parameters: Mapping[str, float] | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return masked calibrated model, data, and uncertainty arrays."""

        _, model, data, uncertainty, valid = _validated_inputs(
            prediction,
            observation,
            coordinate_rtol=self.coordinate_rtol,
            coordinate_atol=self.coordinate_atol,
        )
        parameter_values = {} if parameters is None else parameters
        scale_parameter = self.calibration_scale_parameter or self.scale_parameter
        offset_parameter = self.calibration_offset_parameter or self.offset_parameter
        scale = _parameter_positive_float(
            parameter_values,
            scale_parameter,
            self.calibration_scale,
            "calibration scale",
        )
        offset = _parameter_float(
            parameter_values,
            offset_parameter,
            self.calibration_offset,
            "calibration offset",
        )
        calibrated_model = scale * model + offset

        effective_uncertainty_scale = _parameter_positive_float(
            parameter_values,
            self.uncertainty_scale_parameter,
            self.uncertainty_scale,
            "uncertainty scale",
        )
        effective_uncertainty = uncertainty * effective_uncertainty_scale
        if self.jitter_parameter is not None and self.jitter_parameter in parameter_values:
            jitter = _parameter_float(
                parameter_values,
                self.jitter_parameter,
                0.0,
                "jitter",
            )
            if jitter < 0.0:
                raise RobertValidationError("jitter must be finite and non-negative")
            effective_uncertainty = np.sqrt(
                np.square(effective_uncertainty) + jitter**2
            )
        if not np.all(np.isfinite(effective_uncertainty)) or np.any(effective_uncertainty <= 0.0):
            raise RobertValidationError("effective uncertainty must be finite and positive")

        selected_model = _readonly_result(calibrated_model[valid])
        selected_data = _readonly_result(data[valid])
        selected_uncertainty = _readonly_result(effective_uncertainty[valid])
        return selected_model, selected_data, selected_uncertainty

    def effective_inputs(
        self,
        prediction: Prediction,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return masked calibrated arrays used by :meth:`loglike`."""

        return self._effective_inputs(prediction, observation, parameters)

    def loglike(
        self,
        prediction: Prediction,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate the calibrated direct-flux log likelihood."""

        model, data, uncertainty = self._effective_inputs(
            prediction,
            observation,
            parameters,
        )
        if not np.all(np.isfinite(model)):
            return float(self.invalid_model_loglike)
        residual = data - model
        loglike = -0.5 * np.sum(np.square(residual / uncertainty))
        if self.include_normalization:
            loglike += _normalization_term(uncertainty)
        return float(loglike)

    def pointwise_loglike(
        self,
        prediction: Prediction,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return one direct-flux Gaussian term per unmasked datum."""

        model, data, uncertainty = self._effective_inputs(
            prediction,
            observation,
            parameters,
        )
        if not np.all(np.isfinite(model)):
            result = np.full(model.shape, self.invalid_model_loglike, dtype=float)
            result.setflags(write=False)
            return result
        terms = -0.5 * np.square((data - model) / uncertainty)
        if self.include_normalization:
            terms -= 0.5 * np.log(2.0 * np.pi * np.square(uncertainty))
        return _readonly_result(terms)


@dataclass(frozen=True)
class PreparedPolynomialContinuumProjection:
    """Prepared weighted polynomial projection for repeated model evaluations.

    ``apply(values)`` returns the selected, whitened residual after removing
    the weighted least-squares polynomial component.  If ``y`` is an input
    vector, the operation is

    ``(I - Q Q^T) diag(1 / sigma) y``

    where ``Q`` is the thin QR basis of the weighted polynomial design matrix.
    It is a linear operation.  Applying it to data and model separately gives
    exactly the same result as profiling their difference against the same
    polynomial continuum.  The output contains only unmasked points because
    masked points do not enter the likelihood.
    """

    observation: Observation
    degree: int
    normalized_wavelength: FloatArray
    design_matrix: FloatArray
    orthonormal_basis: FloatArray
    triangular_factor: FloatArray
    inverse_uncertainty: FloatArray
    valid_indices: IntArray
    rank: int
    rank_rtol: float = 1.0e-12
    name: str = "weighted-polynomial-continuum-projection"

    def __post_init__(self) -> None:
        degree = _degree(self.degree)
        rank_rtol = _positive_float(self.rank_rtol, "rank_rtol")
        if not self.name:
            raise RobertValidationError("projection name must not be empty")
        normalized_wavelength = _readonly_float_array(
            self.normalized_wavelength,
            "normalized_wavelength",
        )
        design_matrix = _readonly_float_array(
            self.design_matrix,
            "design_matrix",
            ndim=2,
        )
        orthonormal_basis = _readonly_float_array(
            self.orthonormal_basis,
            "orthonormal_basis",
            ndim=2,
        )
        triangular_factor = _readonly_float_array(
            self.triangular_factor,
            "triangular_factor",
            ndim=2,
        )
        inverse_uncertainty = _readonly_float_array(
            self.inverse_uncertainty,
            "inverse_uncertainty",
        )
        valid_indices = _readonly_int_array(self.valid_indices, "valid_indices")
        n_valid = valid_indices.size
        n_basis = degree + 1
        if n_valid < n_basis:
            raise RobertValidationError(
                "polynomial continuum design has fewer valid points than coefficients"
            )
        if normalized_wavelength.shape != (n_valid,):
            raise RobertValidationError(
                "normalized_wavelength must match the number of valid points"
            )
        if design_matrix.shape != (n_valid, n_basis):
            raise RobertValidationError(
                "design_matrix shape must match valid points and polynomial degree"
            )
        if orthonormal_basis.shape != (n_valid, n_basis):
            raise RobertValidationError(
                "orthonormal_basis shape must match valid points and polynomial degree"
            )
        if triangular_factor.shape != (n_basis, n_basis):
            raise RobertValidationError(
                "triangular_factor shape must match polynomial degree"
            )
        if inverse_uncertainty.shape != (n_valid,):
            raise RobertValidationError(
                "inverse_uncertainty must match the number of valid points"
            )
        if self.rank != n_basis:
            raise RobertValidationError("polynomial continuum design matrix is rank deficient")
        if np.any(valid_indices < 0) or np.any(valid_indices >= self.observation.n_points):
            raise RobertValidationError("valid_indices exceed the observation")
        if np.any(np.diff(valid_indices) <= 0):
            raise RobertValidationError("valid_indices must be strictly increasing")
        if np.any(inverse_uncertainty <= 0.0):
            raise RobertValidationError("inverse_uncertainty must be positive")

        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(self, "normalized_wavelength", normalized_wavelength)
        object.__setattr__(self, "design_matrix", design_matrix)
        object.__setattr__(self, "orthonormal_basis", orthonormal_basis)
        object.__setattr__(self, "triangular_factor", triangular_factor)
        object.__setattr__(self, "inverse_uncertainty", inverse_uncertainty)
        object.__setattr__(self, "valid_indices", valid_indices)

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        degree: int = 2,
        *,
        rank_rtol: float = 1.0e-12,
        name: str = "weighted-polynomial-continuum-projection",
    ) -> "PreparedPolynomialContinuumProjection":
        """Prepare a weighted polynomial operator from one observation.

        Wavelengths are mapped to ``[-1, 1]`` before the polynomial is built.
        This reduces conditioning problems for large wavelength coordinates.
        ``rank_rtol`` is applied to the diagonal of the QR factor.  A rank
        failure is raised before a retrieval can start.
        """

        degree_value = _degree(degree)
        rank_tolerance = _positive_float(rank_rtol, "rank_rtol")
        valid = _observation_valid_mask(observation)
        valid_indices = np.flatnonzero(valid).astype(np.int64)
        if valid_indices.size < degree_value + 1:
            raise RobertValidationError(
                "polynomial continuum design has fewer valid points than coefficients"
            )

        wavelength = np.asarray(observation.wavelength[valid], dtype=float)
        if wavelength.size == 1:
            normalized = np.zeros(1, dtype=float)
        else:
            lower = float(np.min(wavelength))
            upper = float(np.max(wavelength))
            span = upper - lower
            if not np.isfinite(span) or span <= 0.0:
                raise RobertValidationError(
                    "polynomial continuum wavelength coordinates must span a positive range"
                )
            normalized = 2.0 * (wavelength - 0.5 * (lower + upper)) / span

        powers = np.arange(degree_value + 1, dtype=float)
        design = normalized[:, np.newaxis] ** powers[np.newaxis, :]
        uncertainty = np.asarray(observation.uncertainty[valid], dtype=float)
        inverse_uncertainty = 1.0 / uncertainty
        weighted_design = design * inverse_uncertainty[:, np.newaxis]
        basis, triangular = np.linalg.qr(weighted_design, mode="reduced")
        diagonal = np.abs(np.diag(triangular))
        scale = float(np.max(diagonal)) if diagonal.size else 0.0
        if (
            not np.isfinite(scale)
            or scale <= 0.0
            or np.any(~np.isfinite(diagonal))
            or np.any(diagonal <= rank_tolerance * scale)
        ):
            raise RobertValidationError("polynomial continuum design matrix is rank deficient")

        return cls(
            observation=observation,
            degree=degree_value,
            normalized_wavelength=_readonly_float_array(normalized, "normalized_wavelength"),
            design_matrix=_readonly_float_array(design, "design_matrix", ndim=2),
            orthonormal_basis=_readonly_float_array(basis, "orthonormal_basis", ndim=2),
            triangular_factor=_readonly_float_array(triangular, "triangular_factor", ndim=2),
            inverse_uncertainty=_readonly_float_array(
                inverse_uncertainty,
                "inverse_uncertainty",
            ),
            valid_indices=_readonly_int_array(valid_indices, "valid_indices"),
            rank=degree_value + 1,
            rank_rtol=rank_tolerance,
            name=name,
        )

    @property
    def n_valid(self) -> int:
        """Number of points retained by the observation mask."""

        return int(self.valid_indices.size)

    @property
    def n_coefficients(self) -> int:
        """Number of profiled polynomial coefficients."""

        return int(self.degree + 1)

    def _validated_values(self, values: ArrayLike, name: str = "values") -> FloatArray:
        """Validate one full-grid vector for application of the operator."""

        array = np.array(values, dtype=float, copy=True)
        if array.ndim != 1 or array.shape != (self.observation.n_points,):
            raise RobertValidationError(
                f"{name} must be one-dimensional and match the observation grid"
            )
        if not np.all(np.isfinite(array)):
            raise RobertValidationError(f"{name} must contain only finite values")
        return array

    def apply(self, values: ArrayLike) -> FloatArray:
        """Apply the fixed whitened continuum projection to one vector."""

        array = self._validated_values(values)
        selected = array[self.valid_indices] * self.inverse_uncertainty
        projected = selected - self.orthonormal_basis @ (
            self.orthonormal_basis.T @ selected
        )
        return _readonly_result(projected)

    def project(self, values: ArrayLike) -> FloatArray:
        """Alias for :meth:`apply`."""

        return self.apply(values)

    def apply_spectrum(self, spectrum: Spectrum) -> FloatArray:
        """Validate and project a Spectrum on the prepared observation grid."""

        _validated_inputs(
            spectrum,
            self.observation,
            coordinate_rtol=1.0e-12,
            coordinate_atol=0.0,
        )
        return self.apply(spectrum.values)

    def continuum_coefficients(self, values: ArrayLike) -> FloatArray:
        """Return weighted-least-squares coefficients in normalized coordinates."""

        array = self._validated_values(values)
        weighted_values = array[self.valid_indices] * self.inverse_uncertainty
        coefficients = np.linalg.solve(
            self.triangular_factor,
            self.orthonormal_basis.T @ weighted_values,
        )
        return _readonly_result(coefficients)

    def residual(self, data: ArrayLike, model: ArrayLike) -> FloatArray:
        """Return the projected data-minus-model residual."""

        return _readonly_result(self.apply(data) - self.apply(model))

    def chi_square(self, data: ArrayLike, model: ArrayLike) -> float:
        """Return the exact squared norm after continuum projection."""

        residual = self.residual(data, model)
        value = float(np.dot(residual, residual))
        if not np.isfinite(value) or value < 0.0:
            raise RobertValidationError("chi-square must be finite and non-negative")
        return value

    def effective_residual_rank(self, observation: Observation | None = None) -> int:
        """Return the retained residual dimension after projection."""

        del observation
        return int(self.n_valid - self.rank)


@dataclass(frozen=True)
class PolynomialContinuumProjection:
    """Configuration object for a reusable weighted polynomial projection."""

    degree: int = 2
    rank_rtol: float = 1.0e-12
    name: str = "weighted-polynomial-continuum-projection"

    def __post_init__(self) -> None:
        object.__setattr__(self, "degree", _degree(self.degree))
        object.__setattr__(self, "rank_rtol", _positive_float(self.rank_rtol, "rank_rtol"))
        if not self.name:
            raise RobertValidationError("projection name must not be empty")

    def prepare(self, observation: Observation) -> PreparedPolynomialContinuumProjection:
        """Prepare the fixed operator for one observation."""

        return PreparedPolynomialContinuumProjection.from_observation(
            observation,
            degree=self.degree,
            rank_rtol=self.rank_rtol,
            name=self.name,
        )


def prepare_polynomial_continuum_projection(
    observation: Observation,
    degree: int = 2,
    *,
    rank_rtol: float = 1.0e-12,
) -> PreparedPolynomialContinuumProjection:
    """Prepare a weighted polynomial projection for repeated evaluations."""

    return PreparedPolynomialContinuumProjection.from_observation(
        observation,
        degree=degree,
        rank_rtol=rank_rtol,
    )


prepare_linear_projection = prepare_polynomial_continuum_projection


def _same_projection_observation(
    projection: PreparedPolynomialContinuumProjection,
    observation: Observation,
    *,
    coordinate_rtol: float,
    coordinate_atol: float,
) -> None:
    """Ensure a prepared operator is safe to reuse for an observation."""

    reference = projection.observation
    if observation.n_points != reference.n_points:
        raise RobertValidationError(
            "prepared projection and observation have different point counts"
        )
    if observation.wavelength_unit != reference.wavelength_unit:
        raise RobertValidationError(
            "prepared projection and observation wavelength units must match"
        )
    if not np.allclose(
        observation.wavelength,
        reference.wavelength,
        rtol=coordinate_rtol,
        atol=coordinate_atol,
    ):
        raise RobertValidationError(
            "prepared projection and observation wavelength grids must match"
        )
    if not np.array_equal(
        np.ones(observation.n_points, dtype=bool)
        if observation.mask is None
        else observation.mask,
        np.ones(reference.n_points, dtype=bool)
        if reference.mask is None
        else reference.mask,
    ):
        raise RobertValidationError("prepared projection and observation masks must match")
    if not np.array_equal(observation.uncertainty, reference.uncertainty):
        raise RobertValidationError(
            "prepared projection and observation uncertainties must match"
        )
    if observation.flux_unit != reference.flux_unit or observation.observable != reference.observable:
        raise RobertValidationError(
            "prepared projection and observation flux contracts must match"
        )


@dataclass(frozen=True)
class PreparedPolynomialContinuumLikelihood:
    """Prepared analytical polynomial-continuum likelihood.

    Prepare this object once before a retrieval.  The ``loglike`` method may
    then be called with each trial Spectrum.  The optional normalization is
    the original independent Gaussian constant based on the unmasked data;
    this is a profiled likelihood, not a marginal likelihood with a prior on
    continuum coefficients.
    """

    observation: Observation
    projection: PreparedPolynomialContinuumProjection
    include_normalization: bool = False
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0
    name: str = "profiled-polynomial-continuum"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("likelihood name must not be empty")
        rtol, atol = _validate_coordinate_tolerances(
            self.coordinate_rtol,
            self.coordinate_atol,
        )
        _same_projection_observation(
            self.projection,
            self.observation,
            coordinate_rtol=rtol,
            coordinate_atol=atol,
        )
        object.__setattr__(self, "invalid_model_loglike", _invalid_loglike(self.invalid_model_loglike))
        object.__setattr__(self, "coordinate_rtol", rtol)
        object.__setattr__(self, "coordinate_atol", atol)

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        degree: int = 2,
        *,
        include_normalization: bool = False,
        rank_rtol: float = 1.0e-12,
        invalid_model_loglike: float = float("-inf"),
    ) -> "PreparedPolynomialContinuumLikelihood":
        """Prepare a continuum likelihood and its projection in one call."""

        projection = prepare_polynomial_continuum_projection(
            observation,
            degree=degree,
            rank_rtol=rank_rtol,
        )
        return cls(
            observation=observation,
            projection=projection,
            include_normalization=include_normalization,
            invalid_model_loglike=invalid_model_loglike,
        )

    def _resolve_observation(self, observation: Observation | None) -> Observation:
        """Resolve an optional compatibility observation."""

        selected = self.observation if observation is None else observation
        _same_projection_observation(
            self.projection,
            selected,
            coordinate_rtol=self.coordinate_rtol,
            coordinate_atol=self.coordinate_atol,
        )
        return selected

    def _projected_inputs(
        self,
        prediction: Prediction,
        observation: Observation | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return projected model, data, and original selected uncertainty."""

        selected_observation = self._resolve_observation(observation)
        _, model, data, uncertainty, _ = _validated_inputs(
            prediction,
            selected_observation,
            coordinate_rtol=self.coordinate_rtol,
            coordinate_atol=self.coordinate_atol,
        )
        return (
            self.projection.apply(model),
            self.projection.apply(data),
            _readonly_result(uncertainty[self.projection.valid_indices]),
        )

    def effective_inputs(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return projected model, data, and unit uncertainty arrays.

        The projection already whitens by the observation uncertainty.  The
        returned uncertainty is therefore one for every retained point.  The
        original uncertainties remain available through the prepared
        observation and are used for optional normalization.
        """

        del parameters
        model, data, _ = self._projected_inputs(prediction, observation)
        ones = np.ones(model.shape, dtype=float)
        ones.setflags(write=False)
        return model, data, ones

    def loglike(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate the analytically profiled continuum log likelihood."""

        del parameters
        model, data, uncertainty = self._projected_inputs(prediction, observation)
        if not np.all(np.isfinite(model)):
            return float(self.invalid_model_loglike)
        residual = data - model
        loglike = -0.5 * np.dot(residual, residual)
        if self.include_normalization:
            loglike += _normalization_term(uncertainty)
        return float(loglike)

    def chi_square(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Return the exact chi-square after profiling the polynomial."""

        del parameters
        model, data, _ = self._projected_inputs(prediction, observation)
        if not np.all(np.isfinite(model)):
            raise RobertValidationError("chi-square requires a finite model")
        residual = data - model
        value = float(np.dot(residual, residual))
        if not np.isfinite(value) or value < 0.0:
            raise RobertValidationError("chi-square must be finite and non-negative")
        return value

    def effective_residual_rank(
        self,
        observation: Observation | None = None,
    ) -> int:
        """Return the retained residual dimension after polynomial profiling."""

        self._resolve_observation(observation)
        return int(self.projection.n_valid - self.projection.rank)

    def pointwise_loglike(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return projected residual contributions whose sum is ``loglike``.

        These terms are diagnostic contributions only.  Projection makes the
        transformed noise correlated, so they must not be treated as
        independent data points in a pointwise residual analysis.
        """

        del parameters
        model, data, uncertainty = self._projected_inputs(prediction, observation)
        if not np.all(np.isfinite(model)):
            result = np.full(model.shape, self.invalid_model_loglike, dtype=float)
            result.setflags(write=False)
            return result
        terms = -0.5 * np.square(data - model)
        if self.include_normalization:
            terms += _normalization_term(uncertainty) / terms.size
        return _readonly_result(terms)


@dataclass(frozen=True)
class PolynomialContinuumLikelihood:
    """Configuration for an analytically profiled polynomial continuum."""

    degree: int = 2
    include_normalization: bool = False
    rank_rtol: float = 1.0e-12
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0
    name: str = "profiled-polynomial-continuum"

    def __post_init__(self) -> None:
        object.__setattr__(self, "degree", _degree(self.degree))
        object.__setattr__(self, "rank_rtol", _positive_float(self.rank_rtol, "rank_rtol"))
        object.__setattr__(self, "invalid_model_loglike", _invalid_loglike(self.invalid_model_loglike))
        rtol, atol = _validate_coordinate_tolerances(
            self.coordinate_rtol,
            self.coordinate_atol,
        )
        object.__setattr__(self, "coordinate_rtol", rtol)
        object.__setattr__(self, "coordinate_atol", atol)
        if not self.name:
            raise RobertValidationError("likelihood name must not be empty")

    def prepare(self, observation: Observation) -> PreparedPolynomialContinuumLikelihood:
        """Prepare the projection and likelihood outside the retrieval loop."""

        projection = prepare_polynomial_continuum_projection(
            observation,
            degree=self.degree,
            rank_rtol=self.rank_rtol,
        )
        return PreparedPolynomialContinuumLikelihood(
            observation=observation,
            projection=projection,
            include_normalization=self.include_normalization,
            invalid_model_loglike=self.invalid_model_loglike,
            coordinate_rtol=self.coordinate_rtol,
            coordinate_atol=self.coordinate_atol,
            name=self.name,
        )

    def loglike(
        self,
        prediction: Prediction,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate once, preparing a temporary operator for compatibility.

        Retrieval code should call :meth:`prepare` once and reuse the returned
        object.  This convenience method keeps the same call shape as the
        existing likelihood classes.
        """

        return self.prepare(observation).loglike(prediction, parameters=parameters)

    def pointwise_loglike(
        self,
        prediction: Prediction,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return diagnostic projected contributions for one evaluation."""

        return self.prepare(observation).pointwise_loglike(
            prediction,
            parameters=parameters,
        )


@dataclass(frozen=True)
class PreparedCrossCorrelationLikelihood:
    """Prepared high-resolution cross-correlation likelihood.

    ``projection`` is normally a degree-zero or low-degree continuum
    projection prepared from the same observation.  Set it to ``None`` when
    the input spectra are already detrended and only uncertainty weighting is
    required.  The observation mask is always honoured.
    """

    observation: Observation
    projection: PreparedPolynomialContinuumProjection | None = None
    include_normalization: bool = False
    profile_amplitude: bool = True
    require_positive_amplitude: bool = False
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0
    name: str = "high-resolution-cross-correlation"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("likelihood name must not be empty")
        rtol, atol = _validate_coordinate_tolerances(
            self.coordinate_rtol,
            self.coordinate_atol,
        )
        if self.projection is not None:
            _same_projection_observation(
                self.projection,
                self.observation,
                coordinate_rtol=rtol,
                coordinate_atol=atol,
            )
        object.__setattr__(self, "invalid_model_loglike", _invalid_loglike(self.invalid_model_loglike))
        object.__setattr__(self, "coordinate_rtol", rtol)
        object.__setattr__(self, "coordinate_atol", atol)

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        detrend_degree: int | None = 0,
        include_normalization: bool = False,
        profile_amplitude: bool = True,
        require_positive_amplitude: bool = False,
        rank_rtol: float = 1.0e-12,
        invalid_model_loglike: float = float("-inf"),
    ) -> "PreparedCrossCorrelationLikelihood":
        """Prepare a cross-correlation likelihood for one observation.

        ``detrend_degree=None`` means that the supplied data and models are
        already detrended.  A non-negative degree prepares a fixed weighted
        polynomial projection and applies it to both vectors.
        """

        projection = None
        if detrend_degree is not None:
            projection = prepare_polynomial_continuum_projection(
                observation,
                degree=detrend_degree,
                rank_rtol=rank_rtol,
            )
        return cls(
            observation=observation,
            projection=projection,
            include_normalization=include_normalization,
            profile_amplitude=profile_amplitude,
            require_positive_amplitude=require_positive_amplitude,
            invalid_model_loglike=invalid_model_loglike,
        )

    def _resolve_observation(self, observation: Observation | None) -> Observation:
        """Resolve and validate an optional compatibility observation."""

        selected = self.observation if observation is None else observation
        if self.projection is not None:
            _same_projection_observation(
                self.projection,
                selected,
                coordinate_rtol=self.coordinate_rtol,
                coordinate_atol=self.coordinate_atol,
            )
        return selected

    def _vectors(
        self,
        prediction: Prediction,
        observation: Observation | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return whitened data, whitened model, and selected uncertainties."""

        selected_observation = self._resolve_observation(observation)
        _, model, data, uncertainty, valid = _validated_inputs(
            prediction,
            selected_observation,
            coordinate_rtol=self.coordinate_rtol,
            coordinate_atol=self.coordinate_atol,
        )
        if self.projection is None:
            data_vector = data[valid] / uncertainty[valid]
            model_vector = model[valid] / uncertainty[valid]
            selected_uncertainty = uncertainty[valid]
        else:
            data_vector = self.projection.apply(data)
            model_vector = self.projection.apply(model)
            selected_uncertainty = uncertainty[self.projection.valid_indices]
        return (
            _readonly_result(data_vector),
            _readonly_result(model_vector),
            _readonly_result(selected_uncertainty),
        )

    def correlation(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
    ) -> float:
        """Return the normalized weighted correlation coefficient ``rho``."""

        data, model, _ = self._vectors(prediction, observation)
        data_norm = float(np.dot(data, data))
        model_norm = float(np.dot(model, model))
        if not np.isfinite(data_norm) or not np.isfinite(model_norm) or data_norm <= 0.0 or model_norm <= 0.0:
            return float("nan")
        value = float(np.dot(data, model) / np.sqrt(data_norm * model_norm))
        return float(np.clip(value, -1.0, 1.0))

    def profiled_amplitude(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
    ) -> float:
        """Return the weighted least-squares template amplitude ``a_hat``."""

        data, model, _ = self._vectors(prediction, observation)
        model_norm = float(np.dot(model, model))
        if not np.isfinite(model_norm) or model_norm <= 0.0:
            return float("nan")
        amplitude = float(np.dot(data, model) / model_norm)
        if self.require_positive_amplitude and amplitude < 0.0:
            return 0.0
        return amplitude

    def effective_inputs(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the profiled whitened model, data, and unit uncertainty.

        These arrays reproduce :meth:`loglike` exactly. This makes the
        prepared likelihood safe for the shared optimal-estimation contract.
        """

        del parameters
        data, model, _ = self._vectors(prediction, observation)
        if self.profile_amplitude:
            model_norm = float(np.dot(model, model))
            if not np.isfinite(model_norm) or model_norm <= 0.0:
                raise RobertValidationError(
                    "cross-correlation template norm must be finite and positive"
                )
            amplitude = float(np.dot(data, model) / model_norm)
            if self.require_positive_amplitude and amplitude < 0.0:
                amplitude = 0.0
            model = _readonly_result(amplitude * model)
        ones = np.ones(data.shape, dtype=float)
        ones.setflags(write=False)
        return model, data, ones

    def loglike(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate the cross-correlation shape likelihood."""

        del parameters
        data, model, uncertainty = self._vectors(prediction, observation)
        if not np.all(np.isfinite(model)):
            return float(self.invalid_model_loglike)
        data_norm = float(np.dot(data, data))
        model_norm = float(np.dot(model, model))
        cross = float(np.dot(data, model))
        if (
            not np.isfinite(data_norm)
            or not np.isfinite(model_norm)
            or not np.isfinite(cross)
            or data_norm <= 0.0
            or model_norm <= 0.0
        ):
            return float(self.invalid_model_loglike)

        if self.profile_amplitude:
            amplitude = cross / model_norm
            if self.require_positive_amplitude and amplitude < 0.0:
                amplitude = 0.0
            residual = data - amplitude * model
        else:
            residual = data - model
        chi2 = float(np.dot(residual, residual))
        if not np.isfinite(chi2):
            return float(self.invalid_model_loglike)
        # Round-off can make a perfect match produce a tiny negative value in
        # the equivalent d2 - c2/m2 expression.  The residual form above is
        # non-negative by construction, with this guard for numerical noise.
        chi2 = max(0.0, chi2)
        loglike = -0.5 * chi2
        if self.include_normalization:
            loglike += _normalization_term(uncertainty)
        return float(loglike)

    def chi_square(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Return the exact residual statistic used by the CCF likelihood."""

        del parameters
        data, model, _ = self._vectors(prediction, observation)
        if not np.all(np.isfinite(model)):
            raise RobertValidationError("chi-square requires a finite model")
        model_norm = float(np.dot(model, model))
        if not np.isfinite(model_norm) or model_norm <= 0.0:
            raise RobertValidationError(
                "cross-correlation template norm must be finite and positive"
            )
        if self.profile_amplitude:
            amplitude = float(np.dot(data, model) / model_norm)
            if self.require_positive_amplitude and amplitude < 0.0:
                amplitude = 0.0
            residual = data - amplitude * model
        else:
            residual = data - model
        value = float(np.dot(residual, residual))
        if not np.isfinite(value) or value < 0.0:
            raise RobertValidationError("chi-square must be finite and non-negative")
        return value

    def effective_residual_rank(
        self,
        observation: Observation | None = None,
    ) -> int:
        """Return the retained residual dimension after profiling."""

        selected_observation = self._resolve_observation(observation)
        if self.projection is None:
            if selected_observation.mask is None:
                n_valid = selected_observation.n_points
            else:
                n_valid = int(np.count_nonzero(selected_observation.mask))
            if n_valid <= 0:
                raise RobertValidationError(
                    "likelihood mask excludes all observation points"
                )
            projection_rank = 0
        else:
            n_valid = self.projection.n_valid
            projection_rank = self.projection.rank
        amplitude_rank = 1 if self.profile_amplitude else 0
        retained = n_valid - projection_rank - amplitude_rank
        if retained < 0:
            raise RobertValidationError(
                "profiled cross-correlation rank exceeds active points"
            )
        return int(retained)

    def pointwise_loglike(
        self,
        prediction: Prediction,
        observation: Observation | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return diagnostic residual contributions for the profiled template."""

        del parameters
        data, model, uncertainty = self._vectors(prediction, observation)
        if not np.all(np.isfinite(model)):
            result = np.full(model.shape, self.invalid_model_loglike, dtype=float)
            result.setflags(write=False)
            return result
        if self.profile_amplitude:
            model_norm = float(np.dot(model, model))
            if not np.isfinite(model_norm) or model_norm <= 0.0:
                result = np.full(model.shape, self.invalid_model_loglike, dtype=float)
                result.setflags(write=False)
                return result
            amplitude = float(np.dot(data, model) / model_norm)
            if self.require_positive_amplitude and amplitude < 0.0:
                amplitude = 0.0
        else:
            amplitude = 1.0
        terms = -0.5 * np.square(data - amplitude * model)
        if self.include_normalization:
            terms += _normalization_term(uncertainty) / terms.size
        return _readonly_result(terms)


@dataclass(frozen=True)
class CrossCorrelationLikelihood:
    """Configuration for a prepared high-resolution cross-correlation likelihood.

    By default a weighted constant continuum is removed.  Use
    ``detrend_degree=None`` when both the observations and the forward model
    already contain the same external detrending.  The ``profile_amplitude``
    option controls whether a template amplitude is fitted analytically; the
    default is ``True`` and makes the score invariant to positive template
    rescaling.  The Gaussian normalization, when requested, is a fixed data
    term and does not turn this shape statistic into a calibrated flux model.
    """

    detrend_degree: int | None = 0
    include_normalization: bool = False
    profile_amplitude: bool = True
    require_positive_amplitude: bool = False
    rank_rtol: float = 1.0e-12
    invalid_model_loglike: float = float("-inf")
    coordinate_rtol: float = 1.0e-12
    coordinate_atol: float = 0.0
    name: str = "high-resolution-cross-correlation"

    def __post_init__(self) -> None:
        if self.detrend_degree is not None:
            object.__setattr__(self, "detrend_degree", _degree(self.detrend_degree, "detrend_degree"))
        object.__setattr__(self, "rank_rtol", _positive_float(self.rank_rtol, "rank_rtol"))
        object.__setattr__(self, "invalid_model_loglike", _invalid_loglike(self.invalid_model_loglike))
        rtol, atol = _validate_coordinate_tolerances(
            self.coordinate_rtol,
            self.coordinate_atol,
        )
        object.__setattr__(self, "coordinate_rtol", rtol)
        object.__setattr__(self, "coordinate_atol", atol)
        if not self.name:
            raise RobertValidationError("likelihood name must not be empty")

    def prepare(self, observation: Observation) -> PreparedCrossCorrelationLikelihood:
        """Prepare the mask, continuum operator, and data contract once."""

        projection = None
        if self.detrend_degree is not None:
            projection = prepare_polynomial_continuum_projection(
                observation,
                degree=self.detrend_degree,
                rank_rtol=self.rank_rtol,
            )
        return PreparedCrossCorrelationLikelihood(
            observation=observation,
            projection=projection,
            include_normalization=self.include_normalization,
            profile_amplitude=self.profile_amplitude,
            require_positive_amplitude=self.require_positive_amplitude,
            invalid_model_loglike=self.invalid_model_loglike,
            coordinate_rtol=self.coordinate_rtol,
            coordinate_atol=self.coordinate_atol,
            name=self.name,
        )

    def loglike(
        self,
        prediction: Prediction,
        observation: Observation,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate once, preparing a temporary operator for compatibility."""

        return self.prepare(observation).loglike(prediction, parameters=parameters)

    def correlation(
        self,
        prediction: Prediction,
        observation: Observation,
    ) -> float:
        """Return the normalized weighted correlation for one evaluation."""

        return self.prepare(observation).correlation(prediction)


# Discoverable aliases for callers using different terminology.
DirectFluxLikelihood = CalibratedDirectFluxLikelihood
HighResolutionDirectFluxLikelihood = CalibratedDirectFluxLikelihood
WeightedPolynomialContinuumLikelihood = PolynomialContinuumLikelihood
ProfiledPolynomialContinuumLikelihood = PolynomialContinuumLikelihood
PreparedWeightedPolynomialContinuumLikelihood = PreparedPolynomialContinuumLikelihood
PreparedProfiledPolynomialContinuumLikelihood = PreparedPolynomialContinuumLikelihood
PreparedLinearProjection = PreparedPolynomialContinuumProjection
WeightedPolynomialProjection = PolynomialContinuumProjection
LinearProjectionOperator = PolynomialContinuumProjection


__all__ = [
    "CalibratedDirectFluxLikelihood",
    "CrossCorrelationLikelihood",
    "DirectFluxLikelihood",
    "HighResolutionDirectFluxLikelihood",
    "LinearProjectionOperator",
    "PolynomialContinuumLikelihood",
    "PolynomialContinuumProjection",
    "PreparedCrossCorrelationLikelihood",
    "PreparedLinearProjection",
    "PreparedPolynomialContinuumLikelihood",
    "PreparedPolynomialContinuumProjection",
    "PreparedProfiledPolynomialContinuumLikelihood",
    "PreparedWeightedPolynomialContinuumLikelihood",
    "ProfiledPolynomialContinuumLikelihood",
    "WeightedPolynomialContinuumLikelihood",
    "WeightedPolynomialProjection",
    "prepare_linear_projection",
    "prepare_polynomial_continuum_projection",
]
