"""Shared interpolation primitives for tabulated opacity providers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _brackets(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    upper = np.searchsorted(grid, values, side="right")
    upper = np.clip(upper, 1, grid.size - 1).astype(np.int64)
    lower = upper - 1
    weight = (values - grid[lower]) / (grid[upper] - grid[lower])
    exact_low = values == grid[0]
    exact_high = values == grid[-1]
    lower[exact_low] = upper[exact_low] = 0
    lower[exact_high] = upper[exact_high] = grid.size - 1
    weight[exact_low | exact_high] = 0.0
    return lower, upper, weight
