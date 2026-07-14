"""Stellar-spectrum preparation for forward models and future TLSE components."""

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
    "prepare_stellar_spectrum",
]
