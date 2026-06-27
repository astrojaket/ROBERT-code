"""Core domain objects and utilities for ROBERT."""

from .exceptions import (
    RobertConfigError,
    RobertCoverageError,
    RobertDataError,
    RobertError,
    RobertValidationError,
)
from .grids import PressureGrid, SpectralGrid
from .logging import get_logger
from .spectrum import Spectrum

__all__ = [
    "PressureGrid",
    "RobertConfigError",
    "RobertCoverageError",
    "RobertDataError",
    "RobertError",
    "RobertValidationError",
    "SpectralGrid",
    "Spectrum",
    "get_logger",
]
