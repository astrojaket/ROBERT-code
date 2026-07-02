"""Blackbody reference calculations for emission sanity checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum

_PLANCK_CONSTANT_J_S = 6.62607015e-34
_SPEED_OF_LIGHT_M_S = 299_792_458.0
_BOLTZMANN_CONSTANT_J_K = 1.380649e-23
_MICRON_TO_METER = 1.0e-6


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return number


def _readonly_wavelength_micron(wavelength_micron: ArrayLike) -> NDArray[np.float64]:
    wavelength = np.array(wavelength_micron, dtype=float, copy=True)
    if wavelength.ndim != 1:
        raise RobertValidationError("wavelength_micron must be one-dimensional")
    if wavelength.size == 0:
        raise RobertValidationError("wavelength_micron must contain at least one value")
    if not np.all(np.isfinite(wavelength)):
        raise RobertValidationError("wavelength_micron must contain only finite values")
    if np.any(wavelength <= 0.0):
        raise RobertValidationError("wavelength_micron values must be positive")
    wavelength.setflags(write=False)
    return wavelength


def planck_radiance_wavelength(
    wavelength_micron: ArrayLike,
    temperature_k: float,
) -> NDArray[np.float64]:
    """Return spectral radiance from Planck's law.

    Parameters
    ----------
    wavelength_micron:
        Wavelength samples in micron.
    temperature_k:
        Blackbody temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Spectral radiance per wavelength in SI units, W m^-3 sr^-1.
    """

    wavelength = _readonly_wavelength_micron(wavelength_micron)
    temperature = _positive_float(temperature_k, "temperature_k")
    wavelength_m = wavelength * _MICRON_TO_METER

    exponent = (
        _PLANCK_CONSTANT_J_S
        * _SPEED_OF_LIGHT_M_S
        / (wavelength_m * _BOLTZMANN_CONSTANT_J_K * temperature)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        radiance = (
            2.0
            * _PLANCK_CONSTANT_J_S
            * _SPEED_OF_LIGHT_M_S**2
            / (np.power(wavelength_m, 5) * np.expm1(exponent))
        )

    if not np.all(np.isfinite(radiance)) or np.any(radiance < 0.0):
        raise RobertValidationError("blackbody radiance calculation produced invalid values")
    radiance.setflags(write=False)
    return radiance


def blackbody_eclipse_depth(
    wavelength_micron: ArrayLike,
    planet_temperature_k: float,
    star_temperature_k: float,
    planet_radius_m: float,
    star_radius_m: float,
) -> NDArray[np.float64]:
    """Estimate secondary-eclipse depth from two blackbodies.

    The estimate is `(B_planet / B_star) * (R_planet / R_star)^2`. It is a
    diagnostic reference curve only; it does not include atmospheric opacity,
    stellar models, instrument response, or radiative transfer.
    """

    planet_radius = _positive_float(planet_radius_m, "planet_radius_m")
    star_radius = _positive_float(star_radius_m, "star_radius_m")
    planet_radiance = planck_radiance_wavelength(wavelength_micron, planet_temperature_k)
    star_radiance = planck_radiance_wavelength(wavelength_micron, star_temperature_k)
    if np.any(star_radiance <= 0.0):
        raise RobertValidationError("stellar blackbody radiance must be positive")

    depth = (planet_radiance / star_radiance) * (planet_radius / star_radius) ** 2
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise RobertValidationError("blackbody eclipse depth calculation produced invalid values")
    depth.setflags(write=False)
    return depth


def blackbody_eclipse_depth_spectrum(
    wavelength_micron: ArrayLike,
    planet_temperature_k: float,
    star_temperature_k: float,
    planet_radius_m: float,
    star_radius_m: float,
) -> Spectrum:
    """Return the blackbody eclipse-depth estimate as a ROBERT spectrum."""

    wavelength = _readonly_wavelength_micron(wavelength_micron)
    depth = blackbody_eclipse_depth(
        wavelength,
        planet_temperature_k=planet_temperature_k,
        star_temperature_k=star_temperature_k,
        planet_radius_m=planet_radius_m,
        star_radius_m=star_radius_m,
    )
    return Spectrum(
        spectral_grid=SpectralGrid.from_array(wavelength, role="reference"),
        values=depth,
        unit="eclipse_depth",
        observable="eclipse_depth",
        metadata={
            "reference": "blackbody",
            "planet_temperature_k": f"{float(planet_temperature_k):.6g}",
            "star_temperature_k": f"{float(star_temperature_k):.6g}",
        },
    )
