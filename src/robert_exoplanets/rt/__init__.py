"""Radiative-transfer-facing reference helpers."""

from .optical_depth import GasOpticalDepth, assemble_gas_optical_depth

__all__ = [
    "GasOpticalDepth",
    "assemble_gas_optical_depth",
]
