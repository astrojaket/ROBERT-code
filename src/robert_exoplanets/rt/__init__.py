"""Radiative-transfer-facing reference helpers."""

from .emission import (
    ClearSkyEmissionResult,
    disk_average_quadrature,
    solve_clear_sky_emission,
)
from .optical_depth import GasOpticalDepth, assemble_gas_optical_depth

__all__ = [
    "ClearSkyEmissionResult",
    "GasOpticalDepth",
    "assemble_gas_optical_depth",
    "disk_average_quadrature",
    "solve_clear_sky_emission",
]
