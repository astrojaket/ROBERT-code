"""Diagnostics and reference calculations."""

from .blackbody import (
    blackbody_eclipse_depth,
    blackbody_eclipse_depth_spectrum,
    planck_radiance_wavelength,
)

__all__ = [
    "blackbody_eclipse_depth",
    "blackbody_eclipse_depth_spectrum",
    "planck_radiance_wavelength",
]
