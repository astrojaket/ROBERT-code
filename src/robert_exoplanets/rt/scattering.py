"""Scattering source-function helpers for reference radiative transfer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, SpectralGrid
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import spectral_grid_values_in_unit

from .geometry import DiscGeometry

PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
MICRON_TO_METER = 1.0e-6


@dataclass(frozen=True)
class DirectStellarBeam:
    """Direct stellar beam integrated over the stellar solid angle at the planet."""

    spectral_grid: SpectralGrid
    values: ArrayLike
    unit: str = "W m^-3"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly_1d(self.values, "direct stellar beam values")
        if values.shape != self.spectral_grid.values.shape:
            raise RobertValidationError("direct stellar beam values must match spectral grid")
        if np.any(values < 0.0):
            raise RobertValidationError("direct stellar beam values must be non-negative")
        if not self.unit:
            raise RobertValidationError("direct stellar beam unit must not be empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @classmethod
    def blackbody(
        cls,
        spectral_grid: SpectralGrid,
        *,
        star_temperature_k: float,
        star_radius_m: float,
        semi_major_axis_m: float,
    ) -> "DirectStellarBeam":
        """Create a diluted blackbody stellar beam at the planet."""

        temperature = _positive_float(star_temperature_k, "star_temperature_k")
        star_radius = _positive_float(star_radius_m, "star_radius_m")
        orbital_distance = _positive_float(semi_major_axis_m, "semi_major_axis_m")
        if star_radius >= orbital_distance:
            raise RobertValidationError("star_radius_m must be smaller than semi_major_axis_m")
        angular_radius = np.arcsin(star_radius / orbital_distance)
        solid_angle_sr = 2.0 * np.pi * (1.0 - np.cos(angular_radius))
        wavelength = spectral_grid_values_in_unit(spectral_grid, "micron")
        stellar_radiance = _planck_radiance_wavelength(wavelength, temperature)
        return cls(
            spectral_grid=spectral_grid,
            values=stellar_radiance * solid_angle_sr,
            metadata={
                "stellar_model": "blackbody",
                "star_temperature_k": f"{temperature:.12g}",
                "star_radius_m": f"{star_radius:.12g}",
                "semi_major_axis_m": f"{orbital_distance:.12g}",
                "stellar_solid_angle_sr": f"{solid_angle_sr:.12g}",
            },
        )

    def values_on(self, spectral_grid: SpectralGrid) -> NDArray[np.float64]:
        """Return beam values on a requested spectral grid."""

        source_wavelength = spectral_grid_values_in_unit(self.spectral_grid, "micron")
        target_wavelength = spectral_grid_values_in_unit(spectral_grid, "micron")
        if source_wavelength.shape == target_wavelength.shape and np.allclose(
            source_wavelength,
            target_wavelength,
            rtol=1.0e-12,
            atol=0.0,
        ):
            return self.values
        if source_wavelength.size < 2:
            raise RobertValidationError("direct stellar beam cannot be interpolated from one point")
        if not np.all(np.diff(source_wavelength) > 0.0):
            raise RobertValidationError("direct stellar beam spectral grid must be increasing for interpolation")
        if (
            float(np.min(target_wavelength)) < source_wavelength[0]
            or float(np.max(target_wavelength)) > source_wavelength[-1]
        ):
            raise RobertValidationError("direct stellar beam does not cover the requested spectral grid")
        values = np.interp(target_wavelength, source_wavelength, self.values)
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class SingleScatteringSource:
    """First-order direct-beam scattering source-function treatment."""

    stellar_beam: DirectStellarBeam
    phase_function: str = "rayleigh"
    name: str = "single_scattering_direct_beam"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase_function = self.phase_function.strip().lower()
        if phase_function not in {"rayleigh", "isotropic"}:
            raise RobertValidationError("phase_function must be 'rayleigh' or 'isotropic'")
        if not self.name:
            raise RobertValidationError("single-scattering source name must not be empty")
        object.__setattr__(self, "phase_function", phase_function)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def phase_function_values(self, geometry: DiscGeometry) -> NDArray[np.float64]:
        """Return normalized phase-function values for every geometry point."""

        if self.phase_function == "isotropic":
            values = np.ones(geometry.n_points, dtype=float)
        else:
            scattering_angle = geometry.stellar_azimuth_deg
            if not np.all(np.isfinite(scattering_angle)):
                raise RobertValidationError(
                    "Rayleigh single scattering requires finite geometry stellar_azimuth_deg values"
                )
            values = rayleigh_phase_function(scattering_angle)
        values.setflags(write=False)
        return values


def rayleigh_phase_function(scattering_angle_deg: ArrayLike) -> NDArray[np.float64]:
    """Return the Rayleigh phase function normalized so its sphere average is one."""

    angle = _readonly_1d(scattering_angle_deg, "scattering_angle_deg")
    cos_angle = np.cos(np.radians(angle))
    phase = 0.75 * (1.0 + cos_angle**2)
    phase.setflags(write=False)
    return phase


def isotropic_phase_function(scattering_angle_deg: ArrayLike) -> NDArray[np.float64]:
    """Return the isotropic phase function normalized to one."""

    angle = _readonly_1d(scattering_angle_deg, "scattering_angle_deg")
    phase = np.ones_like(angle)
    phase.setflags(write=False)
    return phase


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
