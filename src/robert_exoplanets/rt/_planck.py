"""Shared Planck radiance helper for reference RT components."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError

PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
MICRON_TO_METER = 1.0e-6


def _planck_radiance_wavelength(
    wavelength_micron: ArrayLike,
    temperature_k: float,
) -> NDArray[np.float64]:
    wavelength = _positive_wavelength_micron(wavelength_micron)
    temperature = _positive_float(temperature_k, "temperature_k")
    wavelength_m = wavelength * MICRON_TO_METER
    exponent = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / (
        wavelength_m * BOLTZMANN_CONSTANT_J_K * temperature
    )
    with np.errstate(over="ignore", invalid="ignore"):
        radiance = (
            2.0
            * PLANCK_CONSTANT_J_S
            * SPEED_OF_LIGHT_M_S**2
            / (np.power(wavelength_m, 5) * np.expm1(exponent))
        )
    if not np.all(np.isfinite(radiance)) or np.any(radiance < 0.0):
        raise RobertValidationError("Planck source calculation produced invalid values")
    radiance.setflags(write=False)
    return radiance


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return number


def _positive_wavelength_micron(wavelength_micron: ArrayLike) -> NDArray[np.float64]:
    wavelength = _readonly_1d(wavelength_micron, "wavelength_micron")
    if np.any(wavelength <= 0.0):
        raise RobertValidationError("wavelength_micron values must be positive")
    return wavelength


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
