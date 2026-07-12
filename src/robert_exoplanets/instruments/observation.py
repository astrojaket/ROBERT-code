"""Observation containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.core.exceptions import RobertValidationError
from robert_exoplanets.core.grids import SpectralGrid


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
    wavelength_bin_edges: NDArray[np.float64] | None = None
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
        wavelength_bin_edges: ArrayLike | None = None,
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
            wavelength_bin_edges=(
                None if wavelength_bin_edges is None else np.asarray(wavelength_bin_edges, dtype=float)
            ),
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

        wavelength_bin_edges = None
        if self.wavelength_bin_edges is not None:
            wavelength_bin_edges = _readonly_array(self.wavelength_bin_edges, "wavelength_bin_edges")
            if wavelength_bin_edges.size != wavelength.size + 1:
                raise RobertValidationError("wavelength_bin_edges must contain n_points + 1 values")
            edge_difference = np.diff(wavelength_bin_edges)
            if not (np.all(edge_difference > 0.0) or np.all(edge_difference < 0.0)):
                raise RobertValidationError("wavelength_bin_edges must be strictly monotonic")
            if wavelength.size > 1:
                wavelength_orientation = np.sign(np.diff(wavelength)[0])
                if np.sign(edge_difference[0]) != wavelength_orientation:
                    raise RobertValidationError("wavelength_bin_edges must have the same orientation as wavelength")
            lower = np.minimum(wavelength_bin_edges[:-1], wavelength_bin_edges[1:])
            upper = np.maximum(wavelength_bin_edges[:-1], wavelength_bin_edges[1:])
            if np.any(wavelength <= lower) or np.any(wavelength >= upper):
                raise RobertValidationError("each wavelength must lie strictly inside its bin edges")

        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "wavelength_bin_edges", wavelength_bin_edges)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def n_points(self) -> int:
        """Number of observed spectral points."""

        return int(self.wavelength.size)

    @property
    def value(self) -> NDArray[np.float64]:
        """Alias for the observed quantity values."""

        return self.flux

    @property
    def spectral_grid(self) -> SpectralGrid:
        """Observed spectral grid, including bin edges when available."""

        return SpectralGrid(
            values=self.wavelength,
            unit=self.wavelength_unit,
            name=self.instrument,
            role="observed",
            bin_edges=self.wavelength_bin_edges,
            metadata={"source": "observation"},
        )

def infer_wavelength_bin_edges(wavelength: ArrayLike) -> NDArray[np.float64]:
    """Infer contiguous bin edges halfway between monotonic bin centres."""

    values = _readonly_array(wavelength, "wavelength")
    if values.size < 2:
        raise RobertValidationError("at least two wavelengths are required to infer bin edges")
    difference = np.diff(values)
    if not (np.all(difference > 0.0) or np.all(difference < 0.0)):
        raise RobertValidationError("wavelength values must be strictly monotonic")
    inner = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * difference[0]
    last = values[-1] + 0.5 * difference[-1]
    edges = np.concatenate(([first], inner, [last]))
    if np.any(edges <= 0.0):
        raise RobertValidationError("inferred wavelength bin edges must be positive")
    edges.setflags(write=False)
    return edges
