"""Preparation helpers for real, order-resolved high-resolution spectra.

This module owns instrument and data preparation only.  It does not fit
stellar, telluric, or other time-dependent systematics.  A named order binds
one :class:`~robert_exoplanets.instruments.Observation` to its velocity-frame
metadata, order-specific line-spread function, pixel bins, mask, and optional
fixed linear data/model filter.

The preparation object is immutable after construction.  Its response and
filter operators can therefore be built once and reused for every model
evaluation in a retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping

from .high_resolution import (
    GaussianHighResolutionResponse,
    HighResolutionResponseChain,
    PixelIntegrationResponse,
    PreparedHighResolutionResponseChain,
    RelativisticDopplerResponse,
    RotationalBroadeningResponse,
)
from .observation import Observation, infer_wavelength_bin_edges


_SPEED_OF_LIGHT_KM_S = 299_792.458


def _finite_float(value: float, name: str) -> float:
    """Validate one finite scalar."""

    if isinstance(value, (bool, np.bool_)):
        raise RobertValidationError(f"{name} must be finite")
    try:
        normalised = float(value)
    except (TypeError, ValueError) as error:
        raise RobertValidationError(f"{name} must be finite") from error
    if not np.isfinite(normalised):
        raise RobertValidationError(f"{name} must be finite")
    return normalised


def _positive_float(value: float, name: str) -> float:
    """Validate one finite positive scalar."""

    normalised = _finite_float(value, name)
    if normalised <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return normalised


def _nonnegative_float(value: float, name: str) -> float:
    """Validate one finite non-negative scalar."""

    normalised = _finite_float(value, name)
    if normalised < 0.0:
        raise RobertValidationError(f"{name} must be finite and non-negative")
    return normalised


def _velocity(value: float, name: str) -> float:
    """Validate one physical velocity in km/s."""

    normalised = _finite_float(value, name)
    if abs(normalised) >= _SPEED_OF_LIGHT_KM_S:
        raise RobertValidationError(
            f"{name} must have absolute value smaller than the speed of light"
        )
    return normalised


def _readonly_bool_mask(values: ArrayLike, name: str, size: int) -> NDArray[np.bool_]:
    """Validate and freeze one boolean mask."""

    mask = np.array(values, dtype=bool, copy=True)
    if mask.ndim != 1 or mask.size != size:
        raise RobertValidationError(f"{name} must be one-dimensional with length {size}")
    mask.setflags(write=False)
    return mask


def _readonly_float_vector(values: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    """Validate and freeze one finite vector."""

    vector = np.array(values, dtype=float, copy=True)
    if vector.ndim != 1 or vector.size != size:
        raise RobertValidationError(f"{name} must be one-dimensional with length {size}")
    if not np.all(np.isfinite(vector)):
        raise RobertValidationError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _validated_values(values: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    """Validate one vector used by a linear filter."""

    vector = np.array(values, dtype=float, copy=True)
    if vector.ndim != 1 or vector.size != size:
        raise RobertValidationError(f"{name} must be one-dimensional with length {size}")
    if not np.all(np.isfinite(vector)):
        raise RobertValidationError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _validated_covariance(
    covariance: ArrayLike | Any,
    name: str,
    size: int,
) -> NDArray[np.float64]:
    """Validate one full input covariance matrix.

    Covariance propagation does not require a positive-definite matrix.  A
    positive-definite check belongs to the likelihood that consumes the
    result.  This helper checks the shape, finite values, and symmetry needed
    by the linear transform.
    """

    values = getattr(covariance, "matrix", covariance)
    matrix = np.array(values, dtype=float, copy=True)
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise RobertValidationError(
            f"{name} must be a square matrix with shape ({size}, {size})"
        )
    if not np.all(np.isfinite(matrix)):
        raise RobertValidationError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14):
        raise RobertValidationError(f"{name} must be symmetric")
    matrix.setflags(write=False)
    return matrix


def _propagate_covariance(
    effective_matrix: NDArray[np.float64],
    covariance: ArrayLike | Any,
    n_input: int,
) -> NDArray[np.float64]:
    """Return ``F C F.T`` for one prepared linear operator."""

    matrix = _validated_covariance(covariance, "covariance", n_input)
    propagated = np.asarray(effective_matrix @ matrix @ effective_matrix.T, dtype=float)
    propagated = 0.5 * (propagated + propagated.T)
    propagated.setflags(write=False)
    return propagated


@dataclass(frozen=True)
class VelocityBookkeeping:
    """Fixed velocity-frame metadata for one high-resolution order.

    Positive velocity means that the model is shifted to longer wavelengths.
    The two supplied terms use the same sign convention.  This class records
    the fixed barycentric and systemic terms only; orbital or fitted velocity
    physics belongs to the retrieval parameter model.
    """

    barycentric_velocity_km_s: float = 0.0
    systemic_velocity_km_s: float = 0.0
    reference_frame: str = "observer"

    def __post_init__(self) -> None:
        barycentric = _velocity(
            self.barycentric_velocity_km_s,
            "barycentric_velocity_km_s",
        )
        systemic = _velocity(
            self.systemic_velocity_km_s,
            "systemic_velocity_km_s",
        )
        frame = str(self.reference_frame).strip()
        if not frame:
            raise RobertValidationError("reference_frame must not be empty")
        if abs(barycentric + systemic) >= _SPEED_OF_LIGHT_KM_S:
            raise RobertValidationError(
                "barycentric plus systemic velocity must be smaller than the speed of light"
            )
        object.__setattr__(self, "barycentric_velocity_km_s", barycentric)
        object.__setattr__(self, "systemic_velocity_km_s", systemic)
        object.__setattr__(self, "reference_frame", frame)

    @property
    def fixed_velocity_km_s(self) -> float:
        """Return the fixed barycentric plus systemic velocity."""

        return float(
            self.barycentric_velocity_km_s + self.systemic_velocity_km_s
        )

    @property
    def total_velocity_km_s(self) -> float:
        """Alias for :attr:`fixed_velocity_km_s`."""

        return self.fixed_velocity_km_s

    def with_additional_velocity(self, velocity_km_s: float) -> float:
        """Add a fitted or orbital velocity using the documented sign."""

        additional = _velocity(velocity_km_s, "additional_velocity_km_s")
        total = self.fixed_velocity_km_s + additional
        if abs(total) >= _SPEED_OF_LIGHT_KM_S:
            raise RobertValidationError(
                "total velocity must be smaller than the speed of light"
            )
        return float(total)


@dataclass(frozen=True)
class LinearDataModelFilter:
    """One fixed linear operator applied identically to data and model.

    ``matrix`` has shape ``(n_output, n_input)``.  Invalid input pixels are
    set to zero through ``input_mask`` before the matrix multiplication.  The
    same effective matrix is used by :meth:`apply_data_model`, so filtering
    cannot silently differ between observed data and a trial model.

    The operator is deliberately agnostic about how the matrix was made.  A
    caller can use it for a published PCA/SysRem projection, a fixed
    high-pass matrix, or another documented linear operation.  Telluric and
    stellar fitting are outside this class.
    """

    matrix: ArrayLike
    name: str = "fixed-linear-data-model-filter"
    input_mask: ArrayLike | None = None
    output_wavelength: ArrayLike | None = None
    output_wavelength_unit: str | None = None

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise RobertValidationError("linear filter matrix must be two-dimensional and non-empty")
        if not np.all(np.isfinite(matrix)):
            raise RobertValidationError("linear filter matrix must contain only finite values")
        if not str(self.name).strip():
            raise RobertValidationError("linear filter name must not be empty")
        input_mask = None
        if self.input_mask is not None:
            input_mask = _readonly_bool_mask(
                self.input_mask,
                "linear filter input_mask",
                matrix.shape[1],
            )
        output_wavelength = None
        if self.output_wavelength is not None:
            output_wavelength = _readonly_float_vector(
                self.output_wavelength,
                "linear filter output_wavelength",
                matrix.shape[0],
            )
            if np.any(output_wavelength <= 0.0):
                raise RobertValidationError(
                    "linear filter output_wavelength must be positive"
                )
            if output_wavelength.size > 1 and not (
                np.all(np.diff(output_wavelength) > 0.0)
                or np.all(np.diff(output_wavelength) < 0.0)
            ):
                raise RobertValidationError(
                    "linear filter output_wavelength must be strictly monotonic"
                )
        output_unit = self.output_wavelength_unit
        if output_unit is not None:
            output_unit = str(output_unit).strip()
            if not output_unit:
                raise RobertValidationError(
                    "linear filter output_wavelength_unit must not be empty"
                )
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "input_mask", input_mask)
        object.__setattr__(self, "output_wavelength", output_wavelength)
        object.__setattr__(self, "output_wavelength_unit", output_unit)
        object.__setattr__(self, "name", str(self.name).strip())

    @property
    def n_input(self) -> int:
        """Number of input pixels."""

        return int(self.matrix.shape[1])

    @property
    def n_output(self) -> int:
        """Number of output pixels or filtered components."""

        return int(self.matrix.shape[0])

    def _effective_matrix(
        self,
        input_mask: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return the fixed matrix after applying one optional mask."""

        if input_mask is None:
            if self.input_mask is None:
                mask = np.ones(self.n_input, dtype=bool)
            else:
                mask = np.array(self.input_mask, dtype=bool, copy=True)
        else:
            mask = _readonly_bool_mask(input_mask, "linear filter input_mask", self.n_input)
            if self.input_mask is not None:
                mask = np.asarray(mask & self.input_mask, dtype=bool)
        effective = np.array(self.matrix, dtype=float, copy=True)
        effective[:, ~mask] = 0.0
        effective.setflags(write=False)
        mask.setflags(write=False)
        return effective, mask

    def prepare(
        self,
        observation_or_mask: Observation | ArrayLike | None = None,
    ) -> "PreparedLinearDataModelFilter":
        """Bind the operator to an observation mask once."""

        if observation_or_mask is None:
            mask = self.input_mask
            if mask is None:
                mask = np.ones(self.n_input, dtype=bool)
        elif isinstance(observation_or_mask, Observation):
            if observation_or_mask.n_points != self.n_input:
                raise RobertValidationError(
                    "observation size must match the linear filter input size"
                )
            mask = (
                np.ones(self.n_input, dtype=bool)
                if observation_or_mask.mask is None
                else np.asarray(observation_or_mask.mask, dtype=bool)
            )
            if self.input_mask is not None:
                mask = np.asarray(mask & self.input_mask, dtype=bool)
        else:
            mask = observation_or_mask
        effective, valid_input_mask = self._effective_matrix(mask)
        output_mask = np.any(np.abs(effective) > 0.0, axis=1)
        output_mask.setflags(write=False)
        return PreparedLinearDataModelFilter(
            operator=self,
            effective_matrix=effective,
            valid_input_mask=valid_input_mask,
            output_mask=output_mask,
            observation=(
                observation_or_mask
                if isinstance(observation_or_mask, Observation)
                else None
            ),
        )

    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        """Apply the fixed operator to one vector."""

        vector = _validated_values(values, "linear filter values", self.n_input)
        effective, _ = self._effective_matrix()
        result = np.asarray(effective @ vector, dtype=float)
        result.setflags(write=False)
        return result

    def apply_data_model(
        self,
        data: ArrayLike,
        model: ArrayLike,
        input_mask: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Apply exactly one effective matrix to data and model."""

        data_vector = _validated_values(data, "linear filter data", self.n_input)
        model_vector = _validated_values(model, "linear filter model", self.n_input)
        effective, _ = self._effective_matrix(input_mask)
        filtered_data = np.asarray(effective @ data_vector, dtype=float)
        filtered_model = np.asarray(effective @ model_vector, dtype=float)
        output_mask = np.any(np.abs(effective) > 0.0, axis=1)
        for array in (filtered_data, filtered_model, output_mask):
            array.setflags(write=False)
        return filtered_data, filtered_model, output_mask

    def propagate_covariance(
        self,
        covariance: ArrayLike | Any,
        input_mask: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Propagate a full covariance through this filter.

        The returned matrix is exactly ``F C F.T``.  ``F`` includes this
        filter's configured mask and the optional observation mask.  The
        output can be passed to :class:`CorrelatedGaussianLikelihood`; that
        likelihood performs the positive-definite check and solve.
        """

        effective, _ = self._effective_matrix(input_mask)
        return _propagate_covariance(effective, covariance, self.n_input)

    def _output_grid(self, input_grid: SpectralGrid) -> SpectralGrid:
        """Build the output grid for a filtered spectrum."""

        if self.output_wavelength is None:
            if self.n_output != input_grid.size:
                raise RobertValidationError(
                    "linear filter output_wavelength is required when output size changes"
                )
            values = input_grid.values
            unit = input_grid.unit
        else:
            values = self.output_wavelength
            unit = self.output_wavelength_unit or input_grid.unit
        bin_edges = (
            input_grid.bin_edges
            if self.output_wavelength is None and self.n_output == input_grid.size
            else None
        )
        return SpectralGrid(
            values=values,
            unit=unit,
            role="filtered",
            name=self.name,
            bin_edges=bin_edges,
            metadata={"filter": self.name},
        )

    def apply_spectrum(self, spectrum: Spectrum) -> Spectrum:
        """Apply the fixed operator to a spectrum."""

        grid = self._output_grid(spectrum.spectral_grid)
        values = self.apply(spectrum.values)
        return Spectrum(
            spectral_grid=grid,
            values=values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            metadata={**dict(spectrum.metadata), "linear_filter": self.name},
        )

    def apply_to_observation(self, observation: Observation) -> Observation:
        """Apply the operator to data and propagate diagonal uncertainties.

        A linear transform generally creates correlated output noise.  This
        method provides the propagated diagonal as a convenience.  Use the
        correlated-noise likelihood with the full transformed covariance when
        the off-diagonal terms matter.
        """

        if observation.n_points != self.n_input:
            raise RobertValidationError(
                "observation size must match the linear filter input size"
            )
        effective, _ = self._effective_matrix(observation.mask)
        flux = np.asarray(effective @ observation.flux, dtype=float)
        uncertainty = np.sqrt(
            np.asarray(effective**2 @ np.square(observation.uncertainty), dtype=float)
        )
        output_mask = np.any(np.abs(effective) > 0.0, axis=1)
        uncertainty[~output_mask] = 1.0
        if self.output_wavelength is None:
            if self.n_output != observation.n_points:
                raise RobertValidationError(
                    "linear filter output_wavelength is required when output size changes"
                )
            wavelength = observation.wavelength
            wavelength_unit = observation.wavelength_unit
        else:
            wavelength = self.output_wavelength
            wavelength_unit = self.output_wavelength_unit or observation.wavelength_unit
        return Observation.from_arrays(
            wavelength=wavelength,
            flux=flux,
            uncertainty=uncertainty,
            wavelength_unit=wavelength_unit,
            flux_unit=observation.flux_unit,
            observable=observation.observable,
            instrument=observation.instrument,
            mask=output_mask,
        )


@dataclass(frozen=True)
class PreparedLinearDataModelFilter:
    """Observation-bound form of :class:`LinearDataModelFilter`.

    When prepared from an :class:`Observation`, the input wavelength grid and
    value contract are retained.  Spectrum application then rejects a
    same-length spectrum on different coordinates.
    """

    operator: LinearDataModelFilter
    effective_matrix: ArrayLike
    valid_input_mask: ArrayLike
    output_mask: ArrayLike
    observation: Observation | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        effective = np.array(self.effective_matrix, dtype=float, copy=True)
        if effective.shape != self.operator.matrix.shape:
            raise RobertValidationError("prepared filter matrix shape does not match operator")
        if not np.all(np.isfinite(effective)):
            raise RobertValidationError("prepared filter matrix must be finite")
        valid_input = _readonly_bool_mask(
            self.valid_input_mask,
            "prepared filter valid_input_mask",
            self.operator.n_input,
        )
        output_mask = _readonly_bool_mask(
            self.output_mask,
            "prepared filter output_mask",
            self.operator.n_output,
        )
        observation = self.observation
        if observation is not None:
            if not isinstance(observation, Observation):
                raise RobertValidationError(
                    "prepared filter observation must be an Observation or None"
                )
            if observation.n_points != self.operator.n_input:
                raise RobertValidationError(
                    "prepared filter observation size must match the operator input"
                )
        effective.setflags(write=False)
        object.__setattr__(self, "effective_matrix", effective)
        object.__setattr__(self, "valid_input_mask", valid_input)
        object.__setattr__(self, "output_mask", output_mask)
        object.__setattr__(self, "observation", observation)

    @property
    def name(self) -> str:
        """Filter name."""

        return self.operator.name

    @property
    def n_input(self) -> int:
        """Number of input pixels."""

        return self.operator.n_input

    @property
    def n_output(self) -> int:
        """Number of output pixels or filtered components."""

        return self.operator.n_output

    def _vector(self, values: ArrayLike, name: str) -> NDArray[np.float64]:
        return _validated_values(values, name, self.n_input)

    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        """Apply the observation-bound operator."""

        result = np.asarray(
            self.effective_matrix @ self._vector(values, "prepared filter values"),
            dtype=float,
        )
        result.setflags(write=False)
        return result

    def apply_data_model(
        self,
        data: ArrayLike,
        model: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Apply the same observation-bound operator to data and model."""

        filtered_data = np.asarray(
            self.effective_matrix @ self._vector(data, "prepared filter data"),
            dtype=float,
        )
        filtered_model = np.asarray(
            self.effective_matrix @ self._vector(model, "prepared filter model"),
            dtype=float,
        )
        output_mask = np.array(self.output_mask, dtype=bool, copy=True)
        for array in (filtered_data, filtered_model, output_mask):
            array.setflags(write=False)
        return filtered_data, filtered_model, output_mask

    def propagate_covariance(
        self,
        covariance: ArrayLike | Any,
    ) -> NDArray[np.float64]:
        """Propagate a full covariance through the prepared filter.

        This uses the exact observation-bound matrix, including all masked
        input columns.  It returns ``F C F.T`` without forming an inverse.
        """

        return _propagate_covariance(
            self.effective_matrix,
            covariance,
            self.n_input,
        )

    def apply_spectrum(self, spectrum: Spectrum) -> Spectrum:
        """Apply the observation-bound filter to one spectrum."""

        if self.observation is not None:
            observation_grid = self.observation.spectral_grid
            spectrum_grid = spectrum.spectral_grid
            if spectrum_grid.unit != observation_grid.unit:
                raise RobertValidationError(
                    "prepared filter spectrum and observation wavelength units must match"
                )
            if not np.array_equal(spectrum_grid.values, observation_grid.values):
                raise RobertValidationError(
                    "prepared filter spectrum coordinates must match the bound observation"
                )
            if observation_grid.bin_edges is None:
                if spectrum_grid.bin_edges is not None:
                    raise RobertValidationError(
                        "prepared filter spectrum bin edges must match the bound observation"
                    )
            elif spectrum_grid.bin_edges is None or not np.array_equal(
                spectrum_grid.bin_edges,
                observation_grid.bin_edges,
            ):
                raise RobertValidationError(
                    "prepared filter spectrum bin edges must match the bound observation"
                )
            if spectrum.unit != self.observation.flux_unit:
                raise RobertValidationError(
                    "prepared filter spectrum and observation value units must match"
                )
            if spectrum.observable != self.observation.observable:
                raise RobertValidationError(
                    "prepared filter spectrum and observation observables must match"
                )
        grid = self.operator._output_grid(spectrum.spectral_grid)
        values = np.asarray(self.effective_matrix @ spectrum.values, dtype=float)
        values.setflags(write=False)
        return Spectrum(
            spectral_grid=grid,
            values=values,
            unit=spectrum.unit,
            observable=spectrum.observable,
            metadata={**dict(spectrum.metadata), "linear_filter": self.operator.name},
        )


def propagate_covariance_through_filter(
    covariance: ArrayLike | Any,
    filter_operator: LinearDataModelFilter | PreparedLinearDataModelFilter,
    input_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Propagate a full covariance through a linear data/model filter.

    Use a prepared filter when an observation mask is already bound.  For an
    unprepared filter, ``input_mask`` is optional and is applied before the
    matrix product.  The result is ``F C F.T`` and is returned as a read-only
    dense array.
    """

    if isinstance(filter_operator, PreparedLinearDataModelFilter):
        if input_mask is not None:
            raise RobertValidationError(
                "input_mask cannot be supplied for a prepared linear filter"
            )
        return filter_operator.propagate_covariance(covariance)
    if isinstance(filter_operator, LinearDataModelFilter):
        return filter_operator.propagate_covariance(covariance, input_mask)
    raise RobertValidationError(
        "filter_operator must be a LinearDataModelFilter or PreparedLinearDataModelFilter"
    )


@dataclass(frozen=True)
class HighResolutionOrder:
    """One named high-resolution spectral order and its response settings."""

    name: str
    observation: Observation
    order_number: int | None = None
    barycentric_velocity_km_s: float = 0.0
    systemic_velocity_km_s: float = 0.0
    lsf_resolving_power: float = 100_000.0
    lsf_kernel_support_sigma: float = 4.0
    projected_rotation_km_s: float = 0.0
    limb_darkening: float = 0.0
    apply_velocity_shift: bool = True
    pixel_integration: bool = True
    infer_pixel_edges: bool = True
    mask: ArrayLike | None = None
    data_model_filter: LinearDataModelFilter | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise RobertValidationError("high-resolution order name must not be empty")
        if not isinstance(self.observation, Observation):
            raise RobertValidationError("high-resolution order observation must be an Observation")
        order_number = self.order_number
        if order_number is not None:
            if isinstance(order_number, (bool, np.bool_)) or int(order_number) != order_number:
                raise RobertValidationError("order_number must be an integer or None")
            order_number = int(order_number)
        barycentric = _velocity(
            self.barycentric_velocity_km_s,
            "barycentric_velocity_km_s",
        )
        systemic = _velocity(self.systemic_velocity_km_s, "systemic_velocity_km_s")
        if abs(barycentric + systemic) >= _SPEED_OF_LIGHT_KM_S:
            raise RobertValidationError(
                "barycentric plus systemic velocity must be smaller than the speed of light"
            )
        resolving_power = _positive_float(
            self.lsf_resolving_power,
            "lsf_resolving_power",
        )
        support = _positive_float(
            self.lsf_kernel_support_sigma,
            "lsf_kernel_support_sigma",
        )
        rotation = _nonnegative_float(
            self.projected_rotation_km_s,
            "projected_rotation_km_s",
        )
        limb_darkening = _finite_float(self.limb_darkening, "limb_darkening")
        if not 0.0 <= limb_darkening <= 1.0:
            raise RobertValidationError("limb_darkening must be between zero and one")
        if not isinstance(self.apply_velocity_shift, (bool, np.bool_)):
            raise RobertValidationError("apply_velocity_shift must be boolean")
        if not isinstance(self.pixel_integration, (bool, np.bool_)):
            raise RobertValidationError("pixel_integration must be boolean")
        if not isinstance(self.infer_pixel_edges, (bool, np.bool_)):
            raise RobertValidationError("infer_pixel_edges must be boolean")
        mask = None
        if self.mask is not None:
            mask = _readonly_bool_mask(self.mask, "high-resolution order mask", self.observation.n_points)
        if self.data_model_filter is not None and not isinstance(
            self.data_model_filter,
            LinearDataModelFilter,
        ):
            raise RobertValidationError(
                "data_model_filter must be a LinearDataModelFilter or None"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "order_number", order_number)
        object.__setattr__(self, "barycentric_velocity_km_s", barycentric)
        object.__setattr__(self, "systemic_velocity_km_s", systemic)
        object.__setattr__(self, "lsf_resolving_power", resolving_power)
        object.__setattr__(self, "lsf_kernel_support_sigma", support)
        object.__setattr__(self, "projected_rotation_km_s", rotation)
        object.__setattr__(self, "limb_darkening", limb_darkening)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def order_id(self) -> str:
        """Alias for the stable order name."""

        return self.name

    @property
    def velocity(self) -> VelocityBookkeeping:
        """Return the fixed velocity bookkeeping object."""

        return VelocityBookkeeping(
            barycentric_velocity_km_s=self.barycentric_velocity_km_s,
            systemic_velocity_km_s=self.systemic_velocity_km_s,
        )

    @property
    def fixed_velocity_km_s(self) -> float:
        """Return barycentric plus systemic velocity."""

        return self.velocity.fixed_velocity_km_s

    @property
    def total_velocity_km_s(self) -> float:
        """Alias for the fixed velocity."""

        return self.fixed_velocity_km_s

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Return the observation and order mask intersection."""

        valid = (
            np.ones(self.observation.n_points, dtype=bool)
            if self.observation.mask is None
            else np.asarray(self.observation.mask, dtype=bool).copy()
        )
        if self.mask is not None:
            valid &= self.mask
        valid.setflags(write=False)
        return valid

    @property
    def response_observation(self) -> Observation:
        """Return an observation with combined mask and usable pixel bins.

        If the input contains only pixel centres, contiguous bin edges are
        inferred by default.  The inference is a geometric instrument-data
        convenience.  Real detector edges should be supplied when available.
        """

        edges = self.observation.wavelength_bin_edges
        if edges is None and self.pixel_integration and self.infer_pixel_edges:
            if self.observation.n_points >= 2:
                edges = infer_wavelength_bin_edges(self.observation.wavelength)
        needs_copy = (
            self.observation.mask is None
            or self.mask is not None
            or edges is not self.observation.wavelength_bin_edges
        )
        if not needs_copy:
            return self.observation
        return Observation.from_arrays(
            wavelength=self.observation.wavelength,
            flux=self.observation.flux,
            uncertainty=self.observation.uncertainty,
            wavelength_unit=self.observation.wavelength_unit,
            flux_unit=self.observation.flux_unit,
            observable=self.observation.observable,
            instrument=self.observation.instrument,
            mask=self.valid_mask,
            wavelength_bin_edges=edges,
        )

    def response_chain_for_velocity(
        self,
        additional_velocity_km_s: float = 0.0,
    ) -> HighResolutionResponseChain:
        """Build a response chain for one additional runtime velocity.

        ``additional_velocity_km_s`` is normally a fitted orbital or
        calibration term.  It is added to the fixed barycentric and systemic
        terms before one Doppler stage is prepared.  Positive velocity shifts
        the model to longer wavelengths.  The ``apply_velocity_shift`` flag
        disables the complete Doppler stage, including the runtime term.
        """

        total_velocity = self.velocity.with_additional_velocity(
            additional_velocity_km_s
        )
        stages: list[object] = []
        if self.apply_velocity_shift and total_velocity != 0.0:
            stages.append(RelativisticDopplerResponse(total_velocity))
        if self.projected_rotation_km_s > 0.0:
            stages.append(
                RotationalBroadeningResponse(
                    projected_velocity_km_s=self.projected_rotation_km_s,
                    limb_darkening=self.limb_darkening,
                )
            )
        stages.append(
            GaussianHighResolutionResponse(
                resolving_power=self.lsf_resolving_power,
                kernel_support=self.lsf_kernel_support_sigma,
            )
        )
        response_observation = self.response_observation
        if self.pixel_integration and response_observation.wavelength_bin_edges is not None:
            stages.append(PixelIntegrationResponse())
        return HighResolutionResponseChain(
            stages=stages,
            name=f"{self.name}-response",
        )

    def response_chain(self) -> HighResolutionResponseChain:
        """Build the fixed physical response specification for this order."""

        return self.response_chain_for_velocity(0.0)

    def prepare_with_additional_velocity(
        self,
        native_grid: SpectralGrid,
        additional_velocity_km_s: float = 0.0,
    ) -> "PreparedHighResolutionOrder":
        """Prepare this order for one runtime velocity state.

        The fixed barycentric and systemic terms remain in the response.  A
        retrieval can call this method for each proposed velocity and reuse
        the returned prepared order for all model evaluations at that state.
        """

        if not isinstance(native_grid, SpectralGrid):
            raise RobertValidationError("native_grid must be a SpectralGrid")
        response_observation = self.response_observation
        response = self.response_chain_for_velocity(
            additional_velocity_km_s
        ).prepare(response_observation, native_grid)
        prepared_filter = None
        if self.data_model_filter is not None:
            prepared_filter = self.data_model_filter.prepare(response_observation)
        return PreparedHighResolutionOrder(
            order=self,
            native_grid=native_grid,
            response=response,
            data_model_filter=prepared_filter,
        )

    def prepare_runtime_velocity(
        self,
        native_grid: SpectralGrid,
        additional_velocity_km_s: float = 0.0,
    ) -> "PreparedHighResolutionOrder":
        """Alias for :meth:`prepare_with_additional_velocity`."""

        return self.prepare_with_additional_velocity(
            native_grid,
            additional_velocity_km_s,
        )

    def prepare(self, native_grid: SpectralGrid) -> "PreparedHighResolutionOrder":
        """Prepare this order's response and fixed data/model filter."""

        return self.prepare_with_additional_velocity(native_grid, 0.0)


@dataclass(frozen=True)
class PreparedHighResolutionOrder:
    """Prepared response and filter for one named order."""

    order: HighResolutionOrder
    native_grid: SpectralGrid
    response: PreparedHighResolutionResponseChain
    data_model_filter: PreparedLinearDataModelFilter | None = None

    @property
    def name(self) -> str:
        """Stable order name."""

        return self.order.name

    @property
    def observation(self) -> Observation:
        """Observation bound to the response."""

        return self.order.response_observation

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Return output-valid pixels after response and filtering."""

        if self.data_model_filter is None:
            return self.order.valid_mask
        return self.data_model_filter.output_mask

    def observe(self, spectrum: Spectrum) -> Spectrum:
        """Apply the order response and then the fixed data/model filter."""

        observed = self.response.observe(spectrum)
        if self.data_model_filter is None:
            return observed
        return self.data_model_filter.apply_spectrum(observed)

    def apply_data_model(
        self,
        data: ArrayLike,
        model: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Apply the one fixed filter to matching data and model vectors."""

        if self.data_model_filter is None:
            data_vector = _validated_values(data, "order data", self.observation.n_points)
            model_vector = _validated_values(model, "order model", self.observation.n_points)
            valid = np.array(self.order.valid_mask, dtype=bool, copy=True)
            for array in (data_vector, model_vector, valid):
                array.setflags(write=False)
            return data_vector, model_vector, valid
        return self.data_model_filter.apply_data_model(data, model)

    def filtered_observation(self) -> Observation:
        """Return the data container after the fixed filter, if configured."""

        if self.data_model_filter is None:
            return self.observation
        return self.data_model_filter.operator.apply_to_observation(self.observation)


@dataclass(frozen=True)
class HighResolutionObservation:
    """A named collection of high-resolution spectral orders.

    Order identity, masks, and response metadata are retained.  This
    container does not create a covariance between orders.  Summing separate
    per-order likelihoods therefore assumes independent order noise; a
    cross-order covariance needs one joint observation vector and covariance
    operator.
    """

    orders: tuple[HighResolutionOrder, ...]
    name: str = "high-resolution-observation"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        orders = tuple(self.orders)
        if not orders:
            raise RobertValidationError(
                "high-resolution observation requires at least one order"
            )
        if any(not isinstance(order, HighResolutionOrder) for order in orders):
            raise RobertValidationError(
                "high-resolution observation orders must be HighResolutionOrder values"
            )
        names = tuple(order.name for order in orders)
        if len(set(names)) != len(names):
            raise RobertValidationError("high-resolution order names must be unique")
        if not str(self.name).strip():
            raise RobertValidationError("high-resolution observation name must not be empty")
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @classmethod
    def from_mapping(
        cls,
        observations: Mapping[str, Observation],
        *,
        order_numbers: Mapping[str, int] | None = None,
        barycentric_velocities_km_s: Mapping[str, float] | None = None,
        systemic_velocities_km_s: Mapping[str, float] | None = None,
        masks: Mapping[str, ArrayLike] | None = None,
        lsf_resolving_powers: Mapping[str, float] | None = None,
        lsf_kernel_support_sigma: float = 4.0,
        projected_rotation_km_s: float = 0.0,
        limb_darkening: float = 0.0,
        apply_velocity_shift: bool = True,
        pixel_integration: bool = True,
        infer_pixel_edges: bool = True,
        data_model_filters: Mapping[str, LinearDataModelFilter] | None = None,
        name: str = "high-resolution-observation",
        metadata: Mapping[str, str] | None = None,
    ) -> "HighResolutionObservation":
        """Build named orders from an observation mapping."""

        if not observations:
            raise RobertValidationError("observations must contain at least one order")
        order_numbers = {} if order_numbers is None else order_numbers
        barycentric = (
            {} if barycentric_velocities_km_s is None else barycentric_velocities_km_s
        )
        systemic = {} if systemic_velocities_km_s is None else systemic_velocities_km_s
        masks = {} if masks is None else masks
        resolving = {} if lsf_resolving_powers is None else lsf_resolving_powers
        filters = {} if data_model_filters is None else data_model_filters
        unknown = (
            set(order_numbers)
            | set(barycentric)
            | set(systemic)
            | set(masks)
            | set(resolving)
            | set(filters)
        ) - set(observations)
        if unknown:
            raise RobertValidationError(
                "order metadata keys must match named observations"
            )
        orders = tuple(
            HighResolutionOrder(
                name=order_name,
                observation=observation,
                order_number=order_numbers.get(order_name),
                barycentric_velocity_km_s=barycentric.get(order_name, 0.0),
                systemic_velocity_km_s=systemic.get(order_name, 0.0),
                lsf_resolving_power=resolving.get(order_name, 100_000.0),
                lsf_kernel_support_sigma=lsf_kernel_support_sigma,
                projected_rotation_km_s=projected_rotation_km_s,
                limb_darkening=limb_darkening,
                apply_velocity_shift=apply_velocity_shift,
                pixel_integration=pixel_integration,
                infer_pixel_edges=infer_pixel_edges,
                mask=masks.get(order_name),
                data_model_filter=filters.get(order_name),
            )
            for order_name, observation in observations.items()
        )
        if any(not isinstance(observation, Observation) for observation in observations.values()):
            raise RobertValidationError("observations must map names to Observation values")
        return cls(orders=orders, name=name, metadata={} if metadata is None else metadata)

    @classmethod
    def from_orders(
        cls,
        orders: Iterable[HighResolutionOrder],
        *,
        name: str = "high-resolution-observation",
        metadata: Mapping[str, str] | None = None,
    ) -> "HighResolutionObservation":
        """Build a collection from prepared order specifications."""

        return cls(
            orders=tuple(orders),
            name=name,
            metadata={} if metadata is None else metadata,
        )

    @property
    def order_names(self) -> tuple[str, ...]:
        """Return names in deterministic input order."""

        return tuple(order.name for order in self.orders)

    @property
    def n_orders(self) -> int:
        """Number of named orders."""

        return len(self.orders)

    @property
    def n_points(self) -> int:
        """Total number of order pixels."""

        return sum(order.observation.n_points for order in self.orders)

    def order(self, name: str) -> HighResolutionOrder:
        """Return one order by name."""

        key = str(name).strip()
        for order in self.orders:
            if order.name == key:
                return order
        raise RobertValidationError(f"unknown high-resolution order: {name}")

    def prepare(
        self,
        native_grids: Mapping[str, SpectralGrid],
        additional_velocities_km_s: Mapping[str, float] | None = None,
    ) -> "PreparedHighResolutionObservation":
        """Prepare each order's response and filter once.

        ``additional_velocities_km_s`` supplies an optional runtime velocity
        for each named order.  Omitted values are zero.  Fixed barycentric
        and systemic terms are always retained by the order preparation.
        """

        if set(native_grids) != set(self.order_names):
            raise RobertValidationError(
                "native grid names must match high-resolution order names"
            )
        additional = (
            {} if additional_velocities_km_s is None else additional_velocities_km_s
        )
        if not set(additional).issubset(self.order_names):
            raise RobertValidationError(
                "additional velocity names must match high-resolution order names"
            )
        prepared = {
            order.name: order.prepare_with_additional_velocity(
                native_grids[order.name],
                additional.get(order.name, 0.0),
            )
            for order in self.orders
        }
        return PreparedHighResolutionObservation(
            observation=self,
            orders=prepared,
        )


@dataclass(frozen=True)
class PreparedHighResolutionObservation:
    """Prepared named-order response collection for retrieval loops."""

    observation: HighResolutionObservation
    orders: Mapping[str, PreparedHighResolutionOrder]

    def __post_init__(self) -> None:
        normalised = {str(name).strip(): value for name, value in self.orders.items()}
        expected = set(self.observation.order_names)
        if set(normalised) != expected:
            raise RobertValidationError(
                "prepared order names must match high-resolution observation names"
            )
        if any(not isinstance(value, PreparedHighResolutionOrder) for value in normalised.values()):
            raise RobertValidationError(
                "prepared orders must be PreparedHighResolutionOrder values"
            )
        object.__setattr__(self, "orders", immutable_mapping(normalised))

    @property
    def order_names(self) -> tuple[str, ...]:
        """Return names in source observation order."""

        return self.observation.order_names

    @property
    def n_orders(self) -> int:
        """Number of prepared orders."""

        return self.observation.n_orders

    def order(self, name: str) -> PreparedHighResolutionOrder:
        """Return one prepared order by name."""

        key = str(name).strip()
        try:
            return self.orders[key]
        except KeyError as error:
            raise RobertValidationError(f"unknown prepared high-resolution order: {name}") from error

    def observe(
        self,
        spectra: Mapping[str, Spectrum],
    ) -> Mapping[str, Spectrum]:
        """Apply each named order response to matching native spectra."""

        if set(spectra) != set(self.order_names):
            raise RobertValidationError(
                "native spectrum names must match prepared high-resolution order names"
            )
        if any(not isinstance(spectrum, Spectrum) for spectrum in spectra.values()):
            raise RobertValidationError("native spectra must be Spectrum values")
        return immutable_mapping(
            {
                name: self.orders[name].observe(spectra[name])
                for name in self.order_names
            }
        )

    def filtered_observations(self) -> Mapping[str, Observation]:
        """Return each data order after its fixed data/model filter."""

        return immutable_mapping(
            {
                name: self.orders[name].filtered_observation()
                for name in self.order_names
            }
        )

    def apply_data_model(
        self,
        data_by_order: Mapping[str, ArrayLike],
        model_by_order: Mapping[str, ArrayLike],
    ) -> Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]]:
        """Apply the fixed order filter to data/model pairs."""

        if set(data_by_order) != set(self.order_names) or set(model_by_order) != set(self.order_names):
            raise RobertValidationError(
                "data and model order names must match prepared order names"
            )
        return immutable_mapping(
            {
                name: self.orders[name].apply_data_model(
                    data_by_order[name],
                    model_by_order[name],
                )
                for name in self.order_names
            }
        )

    __call__ = observe


# Friendly aliases for code that uses “spectral order” terminology.
SpectralOrder = HighResolutionOrder
HighResolutionObservationOrder = HighResolutionOrder
PreparedSpectralOrder = PreparedHighResolutionOrder
HighResolutionObservationCollection = HighResolutionObservation
PreparedHighResolutionObservationCollection = PreparedHighResolutionObservation


__all__ = [
    "HighResolutionObservation",
    "HighResolutionObservationCollection",
    "HighResolutionObservationOrder",
    "HighResolutionOrder",
    "LinearDataModelFilter",
    "PreparedHighResolutionObservation",
    "PreparedHighResolutionObservationCollection",
    "PreparedHighResolutionOrder",
    "PreparedLinearDataModelFilter",
    "PreparedSpectralOrder",
    "propagate_covariance_through_filter",
    "SpectralOrder",
    "VelocityBookkeeping",
]
