"""Grid containers for pressure and spectral coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import RobertValidationError


def _readonly_1d_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a validated immutable one-dimensional float array."""

    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _is_strictly_monotonic(array: NDArray[np.float64]) -> bool:
    diff = np.diff(array)
    return bool(np.all(diff > 0) or np.all(diff < 0))


@dataclass(frozen=True)
class PressureGrid:
    """Atmospheric pressure grid with explicit layer edges and centers.

    Pressures must be positive and strictly monotonic. Centers must have one
    fewer element than edges.
    """

    edges: NDArray[np.float64]
    centers: NDArray[np.float64]
    unit: str = "bar"
    name: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edges = _readonly_1d_float_array(self.edges, "pressure edges")
        centers = _readonly_1d_float_array(self.centers, "pressure centers")

        if edges.size < 2:
            raise RobertValidationError("pressure edges must contain at least two values")
        if centers.size != edges.size - 1:
            raise RobertValidationError("pressure centers must have len(edges) - 1 values")
        if np.any(edges <= 0) or np.any(centers <= 0):
            raise RobertValidationError("pressure values must be positive")
        if not _is_strictly_monotonic(edges):
            raise RobertValidationError("pressure edges must be strictly monotonic")
        if not _is_strictly_monotonic(centers):
            raise RobertValidationError("pressure centers must be strictly monotonic")
        if not self.unit:
            raise RobertValidationError("pressure unit must not be empty")

        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_layers(self) -> int:
        """Number of atmospheric layers."""

        return int(self.centers.size)

    @property
    def orientation(self) -> str:
        """Pressure ordering: `increasing` or `decreasing`."""

        return "increasing" if self.edges[-1] > self.edges[0] else "decreasing"

    @classmethod
    def logspace(
        cls,
        min_pressure: float,
        max_pressure: float,
        n_layers: int,
        unit: str = "bar",
        name: str | None = None,
    ) -> "PressureGrid":
        """Build a pressure grid with log-spaced edges and geometric centers."""

        if min_pressure <= 0 or max_pressure <= 0:
            raise RobertValidationError("pressure bounds must be positive")
        if n_layers < 1:
            raise RobertValidationError("n_layers must be at least 1")

        edges = np.logspace(np.log10(min_pressure), np.log10(max_pressure), n_layers + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        return cls(edges=edges, centers=centers, unit=unit, name=name)


@dataclass(frozen=True)
class SpectralGrid:
    """One-dimensional spectral coordinate grid."""

    values: NDArray[np.float64]
    unit: str = "micron"
    name: str | None = None
    role: str = "native"
    bin_edges: NDArray[np.float64] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly_1d_float_array(self.values, "spectral grid values")
        if values.size < 1:
            raise RobertValidationError("spectral grid must contain at least one value")
        if values.size > 1 and not _is_strictly_monotonic(values):
            raise RobertValidationError("spectral grid values must be strictly monotonic")
        if not self.unit:
            raise RobertValidationError("spectral grid unit must not be empty")
        if not self.role:
            raise RobertValidationError("spectral grid role must not be empty")

        bin_edges = None
        if self.bin_edges is not None:
            bin_edges = _readonly_1d_float_array(self.bin_edges, "spectral bin edges")
            if bin_edges.size != values.size + 1:
                raise RobertValidationError("spectral bin edges must have len(values) + 1 values")
            if not _is_strictly_monotonic(bin_edges):
                raise RobertValidationError("spectral bin edges must be strictly monotonic")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bin_edges", bin_edges)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def size(self) -> int:
        """Number of spectral samples."""

        return int(self.values.size)

    @classmethod
    def from_array(
        cls,
        values: ArrayLike,
        unit: str = "micron",
        name: str | None = None,
        role: str = "native",
    ) -> "SpectralGrid":
        """Build a spectral grid from array-like values."""

        return cls(values=np.asarray(values, dtype=float), unit=unit, name=name, role=role)
