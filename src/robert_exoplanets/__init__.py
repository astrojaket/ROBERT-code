"""ROBERT: a foundation for JWST exoplanet emission retrievals."""

from .bodies import Planet, Star
from .core import PressureGrid, SpectralGrid, Spectrum
from .instruments import Observation
from .io import RobertConfig
from .retrieval import (
    EmissionModel,
    RetrievalConfig,
    RetrievalResult,
    run_stub_retrieval,
)

__all__ = [
    "EmissionModel",
    "Observation",
    "Planet",
    "PressureGrid",
    "RobertConfig",
    "RetrievalConfig",
    "RetrievalResult",
    "SpectralGrid",
    "Spectrum",
    "Star",
    "run_stub_retrieval",
]

__version__ = "0.2.0"
