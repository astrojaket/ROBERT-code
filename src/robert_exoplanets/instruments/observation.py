"""Observation containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core.exceptions import RobertValidationError


def _readonly_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Observation:
    """One-dimensional observed spectrum and uncertainties."""

    wavelength: NDArray[np.float64]
    flux: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    wavelength_unit: str = "micron"
    flux_unit: str = "eclipse_depth"
    observable: str = "eclipse_depth"
    instrument: str | None = None
    mask: NDArray[np.bool_] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        wavelength: ArrayLike,
        flux: ArrayLike,
        uncertainty: ArrayLike,
        wavelength_unit: str = "micron",
        flux_unit: str = "eclipse_depth",
        observable: str = "eclipse_depth",
        instrument: str | None = None,
        mask: ArrayLike | None = None,
    ) -> "Observation":
        """Build an observation from array-like inputs and validate it."""

        return cls(
            wavelength=np.asarray(wavelength, dtype=float),
            flux=np.asarray(flux, dtype=float),
            uncertainty=np.asarray(uncertainty, dtype=float),
            wavelength_unit=wavelength_unit,
            flux_unit=flux_unit,
            observable=observable,
            instrument=instrument,
            mask=None if mask is None else np.asarray(mask, dtype=bool),
        )

    def __post_init__(self) -> None:
        wavelength = _readonly_array(self.wavelength, "wavelength")
        flux = _readonly_array(self.flux, "flux")
        uncertainty = _readonly_array(self.uncertainty, "uncertainty")
        if not (wavelength.shape == flux.shape == uncertainty.shape):
            raise RobertValidationError("wavelength, flux, and uncertainty must have matching shapes")
        if np.any(uncertainty <= 0):
            raise RobertValidationError("uncertainty values must be positive")
        if wavelength.size > 1 and not (np.all(np.diff(wavelength) > 0) or np.all(np.diff(wavelength) < 0)):
            raise RobertValidationError("wavelength values must be strictly monotonic")
        if not self.wavelength_unit:
            raise RobertValidationError("wavelength_unit must not be empty")
        if not self.flux_unit:
            raise RobertValidationError("flux_unit must not be empty")
        if not self.observable:
            raise RobertValidationError("observable must not be empty")

        mask = None
        if self.mask is not None:
            mask = np.array(self.mask, dtype=bool, copy=True)
            if mask.shape != wavelength.shape:
                raise RobertValidationError("mask must match wavelength shape")
            mask.setflags(write=False)

        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_points(self) -> int:
        """Number of observed spectral points."""

        return int(self.wavelength.size)

    @property
    def value(self) -> NDArray[np.float64]:
        """Alias for the observed quantity values."""

        return self.flux

    def validate(self) -> None:
        """Preserve the v0.1 validation method as a no-op for compatibility."""

        return None
