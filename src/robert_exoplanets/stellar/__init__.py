"""Stellar-spectrum preparation and transit light source effects."""

from .contamination import (
    StellarContaminationModel,
    StellarContaminationResult,
    StellarHeterogeneity,
    StellarHeterogeneityDefinition,
    prepare_stellar_contamination_model,
)

from .spectra import (
    BlackbodyStellarSpectrumModel,
    PhoenixStellarSpectrumModel,
    StellarSpectrumModel,
    prepare_stellar_spectrum,
)

__all__ = [
    "BlackbodyStellarSpectrumModel",
    "PhoenixStellarSpectrumModel",
    "StellarSpectrumModel",
    "StellarContaminationModel",
    "StellarContaminationResult",
    "StellarHeterogeneity",
    "StellarHeterogeneityDefinition",
    "prepare_stellar_contamination_model",
    "prepare_stellar_spectrum",
]
