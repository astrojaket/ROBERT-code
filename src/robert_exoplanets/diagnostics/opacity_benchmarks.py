"""Numerical comparison helpers for opacity benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from robert_exoplanets.core import RobertValidationError


@dataclass(frozen=True)
class OpacityComparisonResult:
    """Summary of candidate-vs-reference opacity agreement."""

    name: str
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    max_absolute_difference: float
    max_relative_difference: float
    median_relative_difference: float
    worst_index: tuple[int, ...]
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("opacity comparison name must not be empty")
        if len(self.shape) != len(self.axis_names):
            raise RobertValidationError("axis_names must match opacity comparison dimensionality")
        if len(self.shape) != len(self.worst_index):
            raise RobertValidationError("worst_index must match opacity comparison dimensionality")
        if any(size < 1 for size in self.shape):
            raise RobertValidationError("opacity comparison shape values must be positive")
        for value_name in (
            "max_absolute_difference",
            "max_relative_difference",
            "median_relative_difference",
            "absolute_tolerance",
            "relative_tolerance",
        ):
            value = float(getattr(self, value_name))
            if not np.isfinite(value) or value < 0.0:
                raise RobertValidationError(f"{value_name} must be finite and non-negative")
        object.__setattr__(self, "shape", tuple(int(item) for item in self.shape))
        object.__setattr__(self, "axis_names", tuple(str(item) for item in self.axis_names))
        object.__setattr__(self, "worst_index", tuple(int(item) for item in self.worst_index))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_mapping(self) -> dict[str, object]:
        """Return JSON-serializable benchmark summary."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "axis_names": list(self.axis_names),
            "max_absolute_difference": self.max_absolute_difference,
            "max_relative_difference": self.max_relative_difference,
            "median_relative_difference": self.median_relative_difference,
            "worst_index": list(self.worst_index),
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }


def compare_opacity_arrays(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    name: str = "opacity-comparison",
    axis_names: tuple[str, ...] | None = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 1.0e-6,
    relative_floor: float = 1.0e-300,
    metadata: Mapping[str, str] | None = None,
) -> OpacityComparisonResult:
    """Compare two opacity arrays on the same grid.

    The arrays may represent absorption cross sections, CIA coefficients, or
    correlated-k coefficients. The caller owns axis order and should pass
    descriptive `axis_names`, for example `("pressure", "temperature",
    "wavelength", "g_ordinate")`.
    """

    candidate_array = np.asarray(candidate, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if candidate_array.shape != reference_array.shape:
        raise RobertValidationError("candidate and reference opacity arrays must have the same shape")
    if candidate_array.size == 0:
        raise RobertValidationError("opacity arrays must contain at least one value")
    if not np.all(np.isfinite(candidate_array)) or not np.all(np.isfinite(reference_array)):
        raise RobertValidationError("opacity arrays must contain only finite values")
    if np.any(candidate_array < 0.0) or np.any(reference_array < 0.0):
        raise RobertValidationError("opacity arrays must be non-negative")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise RobertValidationError("opacity comparison tolerances must be non-negative")
    if relative_floor <= 0.0 or not np.isfinite(relative_floor):
        raise RobertValidationError("relative_floor must be finite and positive")

    shape = tuple(int(item) for item in candidate_array.shape)
    if axis_names is None:
        axis_names = tuple(f"axis_{index}" for index in range(candidate_array.ndim))
    if len(axis_names) != candidate_array.ndim:
        raise RobertValidationError("axis_names must match opacity array dimensionality")

    absolute_difference = np.abs(candidate_array - reference_array)
    denominator = np.maximum(np.abs(reference_array), relative_floor)
    relative_difference = absolute_difference / denominator
    worst_flat_index = int(np.argmax(relative_difference))
    worst_index = tuple(int(item) for item in np.unravel_index(worst_flat_index, shape))
    threshold = absolute_tolerance + relative_tolerance * denominator
    passed = bool(np.all(absolute_difference <= threshold))

    return OpacityComparisonResult(
        name=name,
        shape=shape,
        axis_names=tuple(axis_names),
        max_absolute_difference=float(np.max(absolute_difference)),
        max_relative_difference=float(np.max(relative_difference)),
        median_relative_difference=float(np.median(relative_difference)),
        worst_index=worst_index,
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
        passed=passed,
        metadata={} if metadata is None else metadata,
    )
