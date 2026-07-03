"""ROBERT: a foundation for JWST exoplanet emission retrievals."""

from .atmosphere import (
    AtmosphereBuilder,
    AtmosphereState,
    BackgroundGasMixture,
    ChemistryModel,
    ConstantChemistry,
    FreeChemistry,
    IsothermalTemperatureProfile,
    MadhusudhanSeager2009TemperatureProfile,
    ParmentierGuillot2014TemperatureProfile,
    SplineTemperatureProfile,
    TabulatedTemperatureProfile,
    TemperatureProfile,
)
from .bodies import Planet, Star
from .core import PressureGrid, SpectralGrid, Spectrum
from .diagnostics import (
    EmissionBenchmark,
    blackbody_eclipse_depth,
    blackbody_eclipse_depth_spectrum,
    load_emission_benchmark_csv,
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
    "BackgroundGasMixture",
    "ChemistryModel",
    "ConstantChemistry",
    "CoverageReport",
    "EmissionModel",
    "EmissionBenchmark",
    "EvaluatedOpacity",
    "FixtureOpacityProvider",
    "ForwardModel",
    "FreeChemistry",
    "GaussianLikelihood",
    "IsothermalTemperatureProfile",
    "LinearObservationResponse",
    "MadhusudhanSeager2009TemperatureProfile",
    "ModelPrediction",
    "Observation",
    "ParmentierGuillot2014TemperatureProfile",
    "Planet",
    "PlaceholderEmissionBackend",
    "PressureGrid",
    "PreparedObservationResponse",
    "PreparedOpacity",
    "RobertConfig",
    "RetrievalConfig",
    "RetrievalResult",
    "SplineTemperatureProfile",
    "SpectralGrid",
    "Spectrum",
    "Star",
    "TabulatedTemperatureProfile",
    "TemperatureProfile",
    "blackbody_eclipse_depth",
    "blackbody_eclipse_depth_spectrum",
    "load_emission_benchmark_csv",
    "planck_radiance_wavelength",
    "run_stub_retrieval",
]

__version__ = "0.3.0"
