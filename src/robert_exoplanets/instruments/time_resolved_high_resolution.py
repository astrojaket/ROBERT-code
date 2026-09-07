"""Time-resolved, order-resolved high-resolution observations.

This module contains the small data contract used by the Smith et al. (2024)
WASP-77Ab IGRINS comparison.  A cube is stored as
``(order, frame, pixel)`` and its wavelength coordinates as
``(order, pixel)``.  The class is deliberately independent of a radiative
transfer model: the velocity terms are observation metadata and the model
uses volume mixing ratios elsewhere in ROBERT.

The arrays are copied and made read-only during construction.  This is
important for retrievals because a cached PCA operator must not change when a
caller changes an input array after preparation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _readonly_float_array(
    values: ArrayLike,
    name: str,
    *,
    ndim: int | None = None,
) -> FloatArray:
    """Copy one finite floating-point array and freeze it."""

    array = np.array(values, dtype=float, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_bool_array(values: ArrayLike, name: str) -> BoolArray:
    """Copy one boolean array and freeze it."""

    array = np.array(values, dtype=bool, copy=True)
    if array.size == 0:
        raise RobertValidationError(f"{name} must not be empty")
    array.setflags(write=False)
    return array


def _finite_velocity_array(values: ArrayLike, name: str) -> FloatArray:
    """Validate velocities in km/s without imposing an orbital model."""

    array = _readonly_float_array(values, name, ndim=1)
    if np.any(np.abs(array) >= 299_792.458):
        raise RobertValidationError(
            f"{name} must have absolute values below the speed of light"
        )
    return array


def _strictly_monotonic_rows(values: FloatArray, name: str) -> None:
    """Require every order wavelength row to be strictly monotonic."""

    if values.ndim != 2:
        raise RobertValidationError(f"{name} must be two-dimensional")
    if values.shape[1] < 2:
        raise RobertValidationError(f"{name} must contain at least two pixels per order")
    differences = np.diff(values, axis=1)
    increasing = np.all(differences > 0.0, axis=1)
    decreasing = np.all(differences < 0.0, axis=1)
    if not np.all(increasing | decreasing):
        raise RobertValidationError(
            f"{name} must be strictly monotonic within every order"
        )


def _broadcast_mask(
    values: ArrayLike | None,
    shape: tuple[int, int, int],
    name: str,
) -> BoolArray:
    """Return a full cube mask from a 2-D or 3-D input mask."""

    if values is None:
        output = np.ones(shape, dtype=bool)
    else:
        array = np.array(values, dtype=bool, copy=True)
        if array.shape == shape:
            output = array
        elif array.shape == shape[::2]:
            output = np.broadcast_to(array[:, None, :], shape).copy()
        else:
            raise RobertValidationError(
                f"{name} must have shape {shape} or {(shape[0], shape[2])}"
            )
    return output


@dataclass(frozen=True)
class HighResolutionEmissionTemplate:
    """High-resolution planet and stellar fluxes on one wavelength grid.

    ``planet_flux`` and ``stellar_flux`` are fluxes in any common units.  The
    likelihood uses their ratio.  ``flux_ratio_scale`` is an explicit
    geometric or calibration factor (for example ``(Rp/Rs)**2`` for a
    surface-flux template); no mass fraction or other abundance conversion is
    performed here.
    """

    wavelength: FloatArray
    planet_flux: FloatArray
    stellar_flux: FloatArray
    wavelength_unit: str = "micron"
    name: str = "high-resolution-emission-template"
    flux_ratio_scale: float = 1.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = _readonly_float_array(self.wavelength, "template wavelength", ndim=1)
        planet_flux = _readonly_float_array(self.planet_flux, "template planet_flux", ndim=1)
        stellar_flux = _readonly_float_array(self.stellar_flux, "template stellar_flux", ndim=1)
        if not (wavelength.shape == planet_flux.shape == stellar_flux.shape):
            raise RobertValidationError(
                "template wavelength, planet_flux, and stellar_flux must have matching shapes"
            )
        if wavelength.size < 2:
            raise RobertValidationError("template requires at least two wavelength points")
        difference = np.diff(wavelength)
        if not (np.all(difference > 0.0) or np.all(difference < 0.0)):
            raise RobertValidationError("template wavelength must be strictly monotonic")
        if np.any(wavelength <= 0.0):
            raise RobertValidationError("template wavelength must be positive")
        if np.any(stellar_flux == 0.0):
            raise RobertValidationError("template stellar_flux must be non-zero")
        try:
            flux_ratio_scale = float(self.flux_ratio_scale)
        except (TypeError, ValueError) as error:
            raise RobertValidationError(
                "template flux_ratio_scale must be finite and positive"
            ) from error
        if not np.isfinite(flux_ratio_scale) or flux_ratio_scale <= 0.0:
            raise RobertValidationError(
                "template flux_ratio_scale must be finite and positive"
            )
        unit = str(self.wavelength_unit).strip()
        name = str(self.name).strip()
        if not unit:
            raise RobertValidationError("template wavelength_unit must not be empty")
        if not name:
            raise RobertValidationError("template name must not be empty")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "planet_flux", planet_flux)
        object.__setattr__(self, "stellar_flux", stellar_flux)
        object.__setattr__(self, "wavelength_unit", unit)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "flux_ratio_scale", flux_ratio_scale)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def flux_ratio(self) -> FloatArray:
        """Return the planet-to-star flux ratio on the template grid."""

        ratio = np.asarray(self.planet_flux / self.stellar_flux, dtype=float)
        if not np.all(np.isfinite(ratio)):
            raise RobertValidationError("template planet-to-star flux ratio is not finite")
        ratio.setflags(write=False)
        return ratio

    @property
    def wavelengths(self) -> FloatArray:
        """Plural alias for :attr:`wavelength`."""

        return self.wavelength

    @property
    def planet_to_star_flux_ratio(self) -> FloatArray:
        """Alias for :attr:`flux_ratio`."""

        return self.flux_ratio

    @property
    def scaled_flux_ratio(self) -> FloatArray:
        """Return the ratio after the explicit geometric scale factor."""

        ratio = np.asarray(self.flux_ratio * self.flux_ratio_scale, dtype=float)
        ratio.setflags(write=False)
        return ratio

    @property
    def fp(self) -> FloatArray:
        """Compact alias for the planet flux."""

        return self.planet_flux

    @property
    def fs(self) -> FloatArray:
        """Compact alias for the stellar flux."""

        return self.stellar_flux

    def interpolate_flux_ratio(
        self,
        wavelength: ArrayLike,
        velocity_km_s: float = 0.0,
        doppler_mode: str = "smith_nonrelativistic",
    ) -> FloatArray:
        """Interpolate the ratio at observed wavelengths for one velocity.

        Positive velocity is redshift: a rest wavelength ``lambda`` is seen at
        ``lambda * (1 + beta)`` to first order.  The Smith/POSEIDON
        implementation instead samples the source grid at
        ``wavelength * (1 - beta)``; that exact non-relativistic mapping is the
        default here.  Set ``doppler_mode="relativistic"`` only for a
        separately labelled diagnostic.  Values outside the template support
        are returned as ``NaN`` so a likelihood cannot hide an uncovered model
        point through edge extrapolation.
        """

        values = _readonly_float_array(wavelength, "interpolation wavelength", ndim=1)
        velocity = float(velocity_km_s)
        if not np.isfinite(velocity) or abs(velocity) >= 299_792.458:
            raise RobertValidationError("velocity_km_s must be finite and below light speed")
        beta = velocity / 299_792.458
        mode = str(doppler_mode).strip().lower()
        if mode in {"smith", "smith_nonrelativistic", "nonrelativistic"}:
            source_wavelength = values * (1.0 - beta)
        elif mode in {"relativistic", "relativistic_doppler"}:
            doppler = float(np.sqrt((1.0 + beta) / (1.0 - beta)))
            source_wavelength = values / doppler
        else:
            raise RobertValidationError(
                "doppler_mode must be smith_nonrelativistic or relativistic"
            )
        grid = self.wavelength
        ratio = self.flux_ratio
        if grid[0] > grid[-1]:
            grid = grid[::-1]
            ratio = ratio[::-1]
        result = np.interp(
            source_wavelength,
            grid,
            ratio,
            left=np.nan,
            right=np.nan,
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class TimeResolvedHighResolutionObservation:
    """Immutable order/frame/pixel high-resolution observation cube.

    The public cube convention is ``flux[order, frame, pixel]`` and
    ``order_wavelengths[order, pixel]``.  ``mask`` is stored in the full cube
    shape and combines optional order, frame, and pixel masks.  A loader may
    enforce a study-specific shape; this generic container also accepts small
    arrays so formula tests can use a hand-sized observation.
    """

    order_wavelengths: FloatArray
    flux: FloatArray
    phase: FloatArray
    fixed_velocity_km_s: FloatArray
    time_bjd: FloatArray
    mask: BoolArray | None = None
    order_mask: BoolArray | None = None
    frame_mask: BoolArray | None = None
    airmass: FloatArray | None = None
    humidity_percent: FloatArray | None = None
    median_snr: FloatArray | None = None
    wavelength_unit: str = "micron"
    flux_unit: str = "detector_counts"
    observable: str = "normalized_high_resolution_flux"
    instrument: str = "Gemini South/IGRINS"
    name: str = "time-resolved-high-resolution-observation"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelengths = _readonly_float_array(
            self.order_wavelengths,
            "order_wavelengths",
            ndim=2,
        )
        flux = _readonly_float_array(self.flux, "flux", ndim=3)
        phase = _readonly_float_array(self.phase, "phase", ndim=1)
        fixed_velocity = _finite_velocity_array(
            self.fixed_velocity_km_s,
            "fixed_velocity_km_s",
        )
        time_bjd = _readonly_float_array(self.time_bjd, "time_bjd", ndim=1)
        if flux.shape[0] != wavelengths.shape[0] or flux.shape[2] != wavelengths.shape[1]:
            raise RobertValidationError(
                "flux must have shape (n_orders, n_frames, n_pixels) matching order_wavelengths"
            )
        n_frames = flux.shape[1]
        if not (phase.size == fixed_velocity.size == time_bjd.size == n_frames):
            raise RobertValidationError(
                "phase, fixed_velocity_km_s, and time_bjd must match the frame dimension"
            )
        if np.any(wavelengths <= 0.0):
            raise RobertValidationError("order_wavelengths must be positive")
        _strictly_monotonic_rows(wavelengths, "order_wavelengths")

        shape = tuple(int(value) for value in flux.shape)
        cube_mask = _broadcast_mask(self.mask, shape, "mask")

        order_mask = np.ones(shape[0], dtype=bool)
        if self.order_mask is not None:
            order_mask = np.array(self.order_mask, dtype=bool, copy=True)
            if order_mask.shape != (shape[0],):
                raise RobertValidationError(
                    f"order_mask must have shape {(shape[0],)}"
                )
        frame_mask = np.ones(shape[1], dtype=bool)
        if self.frame_mask is not None:
            frame_mask = np.array(self.frame_mask, dtype=bool, copy=True)
            if frame_mask.shape != (shape[1],):
                raise RobertValidationError(
                    f"frame_mask must have shape {(shape[1],)}"
                )
        cube_mask &= order_mask[:, None, None]
        cube_mask &= frame_mask[None, :, None]
        cube_mask.setflags(write=False)
        order_mask.setflags(write=False)
        frame_mask.setflags(write=False)

        optional_arrays: dict[str, FloatArray | None] = {}
        for name, values in (
            ("airmass", self.airmass),
            ("humidity_percent", self.humidity_percent),
            ("median_snr", self.median_snr),
        ):
            if values is None:
                optional_arrays[name] = None
                continue
            optional = _readonly_float_array(values, name, ndim=1)
            if optional.size != n_frames:
                raise RobertValidationError(
                    f"{name} must have length {n_frames}"
                )
            optional_arrays[name] = optional

        for field_name, value in (
            ("order_wavelengths", wavelengths),
            ("flux", flux),
            ("phase", phase),
            ("fixed_velocity_km_s", fixed_velocity),
            ("time_bjd", time_bjd),
            ("mask", cube_mask),
            ("order_mask", order_mask),
            ("frame_mask", frame_mask),
        ):
            object.__setattr__(self, field_name, value)
        for field_name, value in optional_arrays.items():
            object.__setattr__(self, field_name, value)

        wavelength_unit = str(self.wavelength_unit).strip()
        flux_unit = str(self.flux_unit).strip()
        observable = str(self.observable).strip()
        instrument = str(self.instrument).strip()
        name = str(self.name).strip()
        if not all((wavelength_unit, flux_unit, observable, instrument, name)):
            raise RobertValidationError(
                "wavelength_unit, flux_unit, observable, instrument, and name must not be empty"
            )
        object.__setattr__(self, "wavelength_unit", wavelength_unit)
        object.__setattr__(self, "flux_unit", flux_unit)
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @classmethod
    def from_arrays(
        cls,
        order_wavelengths: ArrayLike,
        flux: ArrayLike,
        phase: ArrayLike,
        fixed_velocity_km_s: ArrayLike,
        time_bjd: ArrayLike,
        **kwargs: object,
    ) -> "TimeResolvedHighResolutionObservation":
        """Construct an observation from array-like values."""

        return cls(
            order_wavelengths=np.asarray(order_wavelengths, dtype=float),
            flux=np.asarray(flux, dtype=float),
            phase=np.asarray(phase, dtype=float),
            fixed_velocity_km_s=np.asarray(fixed_velocity_km_s, dtype=float),
            time_bjd=np.asarray(time_bjd, dtype=float),
            **kwargs,
        )

    @property
    def wavelengths(self) -> FloatArray:
        """Alias for the order wavelength grids."""

        return self.order_wavelengths

    @property
    def n_orders(self) -> int:
        """Number of spectral orders."""

        return int(self.flux.shape[0])

    @property
    def n_frames(self) -> int:
        """Number of time frames."""

        return int(self.flux.shape[1])

    @property
    def n_pixels(self) -> int:
        """Number of pixels per order."""

        return int(self.flux.shape[2])

    @property
    def n_points(self) -> int:
        """Number of active order/frame/pixel samples."""

        return int(np.count_nonzero(self.mask))

    @property
    def rv_km_s(self) -> FloatArray:
        """Alias for the fixed per-frame RV term."""

        return self.fixed_velocity_km_s

    @property
    def fixed_velocity(self) -> FloatArray:
        """Alias for :attr:`fixed_velocity_km_s`."""

        return self.fixed_velocity_km_s

    @property
    def observed_flux(self) -> FloatArray:
        """Alias for the detector flux cube."""

        return self.flux

    @property
    def valid_mask(self) -> BoolArray:
        """Alias for the combined full-cube mask."""

        return self.mask

    def order_wavelength(self, order: int) -> FloatArray:
        """Return one order wavelength grid by integer index."""

        index = int(order)
        if index < 0 or index >= self.n_orders:
            raise RobertValidationError(f"order index {order} is outside the observation")
        return self.order_wavelengths[index]


# Discoverable aliases used by study-specific callers.
Smith2024HighResolutionObservation = TimeResolvedHighResolutionObservation
Smith2024HRSObservation = TimeResolvedHighResolutionObservation
HighResolutionTimeSeries = TimeResolvedHighResolutionObservation
Smith2024EmissionTemplate = HighResolutionEmissionTemplate


__all__ = [
    "HighResolutionEmissionTemplate",
    "HighResolutionTimeSeries",
    "Smith2024EmissionTemplate",
    "Smith2024HRSObservation",
    "Smith2024HighResolutionObservation",
    "TimeResolvedHighResolutionObservation",
]
