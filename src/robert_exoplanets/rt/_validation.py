"""Shared validation helpers for reference RT components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.opacity import (
    pressure_values_in_unit,
    spectral_grid_values_in_unit,
)

if TYPE_CHECKING:
    from .optical_depth import GasOpticalDepth


def _validate_contribution_grid_match(
    gas_optical_depth: "GasOpticalDepth",
    contribution: object,
) -> None:
    if hasattr(contribution, "spectral_grid"):
        contribution_wavelength = spectral_grid_values_in_unit(
            getattr(contribution, "spectral_grid"),
            "micron",
        )
        gas_wavelength = spectral_grid_values_in_unit(
            gas_optical_depth.spectral_grid, "micron"
        )
        if contribution_wavelength.shape != gas_wavelength.shape or not np.allclose(
            contribution_wavelength,
            gas_wavelength,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RobertValidationError(
                "additional optical-depth spectral grid must match gas grid"
            )
    if hasattr(contribution, "pressure_grid"):
        contribution_pressure = pressure_values_in_unit(
            getattr(contribution, "pressure_grid").centers,
            getattr(contribution, "pressure_grid").unit,
            "pa",
        )
        gas_pressure = pressure_values_in_unit(
            gas_optical_depth.pressure_grid.centers,
            gas_optical_depth.pressure_grid.unit,
            "pa",
        )
        if contribution_pressure.shape != gas_pressure.shape or not np.allclose(
            contribution_pressure,
            gas_pressure,
            rtol=1.0e-10,
            atol=0.0,
        ):
            raise RobertValidationError(
                "additional optical-depth pressure grid must match gas grid"
            )


def _gravity_profile(values: float | ArrayLike, n_layers: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim == 0:
        array = np.full(n_layers, float(array), dtype=float)
    if array.ndim != 1:
        raise RobertValidationError("gravity_m_s2 must be scalar or one-dimensional")
    if array.shape != (n_layers,):
        raise RobertValidationError("gravity_m_s2 must match pressure grid layers")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise RobertValidationError("gravity_m_s2 values must be finite and positive")
    array.setflags(write=False)
    return array
