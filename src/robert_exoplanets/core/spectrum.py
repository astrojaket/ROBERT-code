"""Spectrum containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import RobertValidationError
from .grids import SpectralGrid
from ._immutability import immutable_mapping


@dataclass(frozen=True)
class Spectrum:
    """Spectral values on a spectral grid."""

    spectral_grid: SpectralGrid
    values: NDArray[np.float64]
    unit: str
    observable: str
    uncertainty: NDArray[np.float64] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        if values.ndim != 1:
            raise RobertValidationError("spectrum values must be one-dimensional")
        if values.shape != self.spectral_grid.values.shape:
            raise RobertValidationError("spectrum values must match spectral grid shape")
        if not np.all(np.isfinite(values)):
            raise RobertValidationError("spectrum values must be finite")
        if not self.unit:
            raise RobertValidationError("spectrum unit must not be empty")
        if not self.observable:
            raise RobertValidationError("spectrum observable must not be empty")
        values.setflags(write=False)

        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = np.array(self.uncertainty, dtype=float, copy=True)
            if uncertainty.shape != values.shape:
                raise RobertValidationError("spectrum uncertainty must match values shape")
            if np.any(uncertainty <= 0) or not np.all(np.isfinite(uncertainty)):
                raise RobertValidationError("spectrum uncertainty must be finite and positive")
            uncertainty.setflags(write=False)

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @classmethod
    def from_arrays(
        cls,
        wavelength: ArrayLike,
        values: ArrayLike,
        unit: str,
        observable: str,
        wavelength_unit: str = "micron",
        uncertainty: ArrayLike | None = None,
    ) -> "Spectrum":
        """Build a spectrum from coordinate and value arrays."""

        grid = SpectralGrid.from_array(wavelength, unit=wavelength_unit)
        uncertainty_array = None if uncertainty is None else np.asarray(uncertainty, dtype=float)
        return cls(
            spectral_grid=grid,
            values=np.asarray(values, dtype=float),
            unit=unit,
            observable=observable,
            uncertainty=uncertainty_array,
        )
