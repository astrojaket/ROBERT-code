"""ROBERT: a foundation for JWST exoplanet emission retrievals."""

from .atmosphere import (
    AtmosphereBuilder,
    AtmosphereState,
    ConstantChemistry,
    IsothermalTemperatureProfile,
)
from .bodies import Planet, Star
from .core import PressureGrid, SpectralGrid, Spectrum
from .diagnostics import (
    blackbody_eclipse_depth,
    blackbody_eclipse_depth_spectrum,
    planck_radiance_wavelength,
)
from .forward import ForwardModel, ModelPrediction, PlaceholderEmissionBackend
from .instruments import LinearObservationResponse, Observation, PreparedObservationResponse
from .io import RobertConfig
from .likelihoods import GaussianLikelihood
from .opacity import CoverageReport, EvaluatedOpacity, FixtureOpacityProvider, PreparedOpacity
from .retrieval import (
    EmissionModel,
    RetrievalConfig,
    RetrievalResult,
    run_stub_retrieval,
)

__all__ = [
    "AtmosphereBuilder",
    "AtmosphereState",
    "ConstantChemistry",
    "CoverageReport",
    "EmissionModel",
    "EvaluatedOpacity",
    "FixtureOpacityProvider",
    "ForwardModel",
    "GaussianLikelihood",
    "IsothermalTemperatureProfile",
    "LinearObservationResponse",
    "ModelPrediction",
    "Observation",
    "Planet",
    "PlaceholderEmissionBackend",
    "PressureGrid",
    "PreparedObservationResponse",
    "PreparedOpacity",
    "RobertConfig",
    "RetrievalConfig",
    "RetrievalResult",
    "SpectralGrid",
    "Spectrum",
    "Star",
    "blackbody_eclipse_depth",
    "blackbody_eclipse_depth_spectrum",
    "planck_radiance_wavelength",
    "run_stub_retrieval",
]

__version__ = "0.3.0"
