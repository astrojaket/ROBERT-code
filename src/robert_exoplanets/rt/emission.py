"""Clear-sky thermal-emission reference solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.opacity import spectral_grid_values_in_unit

from .optical_depth import GasOpticalDepth

PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
MICRON_TO_METER = 1.0e-6


@dataclass(frozen=True)
class ClearSkyEmissionResult:
    """Output and diagnostics from the clear-sky thermal-emission solver."""

    gas_optical_depth: GasOpticalDepth
    radiance: Spectrum
    layer_source_function: ArrayLike
    layer_contribution_radiance: ArrayLike
    bottom_contribution_radiance: ArrayLike
    emission_angle_cosines: ArrayLike
    emission_angle_weights: ArrayLike
    eclipse_depth: Spectrum | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_layers = self.gas_optical_depth.atmosphere.n_layers
        n_spectral = self.radiance.spectral_grid.size
        source = _readonly_array(
            self.layer_source_function,
            "layer_source_function",
            (n_layers, n_spectral),
        )
        contribution = _readonly_array(
            self.layer_contribution_radiance,
            "layer_contribution_radiance",
            (n_layers, n_spectral),
        )
        bottom = _readonly_1d(
            self.bottom_contribution_radiance,
            "bottom_contribution_radiance",
        )
        if bottom.shape != (n_spectral,):
            raise RobertValidationError("bottom_contribution_radiance must match spectral grid")
        if np.any(source < 0.0) or np.any(contribution < 0.0) or np.any(bottom < 0.0):
            raise RobertValidationError("clear-sky emission diagnostics must be non-negative")
        mu, weights = _validate_emission_angle_quadrature(
            self.emission_angle_cosines,
            self.emission_angle_weights,
        )
        if self.eclipse_depth is not None:
            if self.eclipse_depth.spectral_grid.values.shape != self.radiance.spectral_grid.values.shape:
                raise RobertValidationError("eclipse depth and radiance grids must have matching shapes")
            if not np.allclose(
                self.eclipse_depth.spectral_grid.values,
                self.radiance.spectral_grid.values,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise RobertValidationError("eclipse depth and radiance grids must match")

        object.__setattr__(self, "layer_source_function", source)
        object.__setattr__(self, "layer_contribution_radiance", contribution)
        object.__setattr__(self, "bottom_contribution_radiance", bottom)
        object.__setattr__(self, "emission_angle_cosines", mu)
        object.__setattr__(self, "emission_angle_weights", weights)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def normalized_layer_contribution(self) -> NDArray[np.float64]:
        """Return layer emission contribution normalized independently by wavelength."""

        total = np.sum(self.layer_contribution_radiance, axis=0, keepdims=True)
        normalized = np.divide(
            self.layer_contribution_radiance,
            total,
            out=np.zeros_like(self.layer_contribution_radiance),
            where=total > 0.0,
        )
        normalized.setflags(write=False)
        return normalized


def solve_clear_sky_emission(
    gas_optical_depth: GasOpticalDepth,
    *,
    emission_angle_cosines: ArrayLike | None = None,
    emission_angle_weights: ArrayLike | None = None,
    bottom_boundary: str = "blackbody",
    planet_radius_m: float | None = None,
    star_radius_m: float | None = None,
    star_temperature_k: float | None = None,
) -> ClearSkyEmissionResult:
    """Solve thermal emission for a gas-only, non-scattering atmosphere.

    The solver integrates layer source functions through the gas optical depth.
    It is a readable NumPy reference implementation and intentionally excludes
    CIA, Rayleigh, clouds, aerosols, and scattering source terms.
    """

    bottom_mode = bottom_boundary.strip().lower()
    if bottom_mode not in {"blackbody", "none"}:
        raise RobertValidationError("bottom_boundary must be 'blackbody' or 'none'")

    if emission_angle_cosines is None and emission_angle_weights is None:
        mu, mu_weights = _normal_emission_quadrature()
    elif emission_angle_cosines is not None and emission_angle_weights is not None:
        mu, mu_weights = _validate_emission_angle_quadrature(
            emission_angle_cosines,
            emission_angle_weights,
        )
    else:
        raise RobertValidationError(
            "emission_angle_cosines and emission_angle_weights must be provided together"
        )

    wavelength = spectral_grid_values_in_unit(gas_optical_depth.spectral_grid, "micron")
    output_grid = SpectralGrid.from_array(
        wavelength,
        unit="micron",
        role="rt_native",
        name=gas_optical_depth.spectral_grid.name,
    )
    order = _top_to_bottom_order(gas_optical_depth)
    source = _layer_planck_source(wavelength, gas_optical_depth.atmosphere.temperature)
    source_ordered = source[order]
    tau_ordered = gas_optical_depth.total_tau[order]

    layer_contribution_ordered = np.zeros(
        (gas_optical_depth.atmosphere.n_layers, wavelength.size),
        dtype=float,
    )
    bottom_contribution = np.zeros(wavelength.size, dtype=float)

    for mu_value, mu_weight in zip(mu, mu_weights):
        slant_tau = tau_ordered / mu_value
        cumulative_before = _exclusive_cumulative(slant_tau)
        transmission_before = np.exp(-cumulative_before)
        layer_escape = transmission_before * (-np.expm1(-slant_tau))
        layer_radiance_by_g = source_ordered[:, :, None] * layer_escape
        layer_contribution_ordered += (
            np.sum(layer_radiance_by_g * gas_optical_depth.g_weights[None, None, :], axis=-1)
            * mu_weight
        )
        if bottom_mode == "blackbody":
            total_transmission = np.exp(-np.sum(slant_tau, axis=0))
            deepest_temperature = float(
                gas_optical_depth.atmosphere.temperature[
                    int(np.argmax(gas_optical_depth.pressure_grid.centers))
                ]
            )
            bottom_source = _planck_radiance_wavelength(wavelength, deepest_temperature)
            bottom_contribution += (
                np.sum(total_transmission * gas_optical_depth.g_weights[None, :], axis=-1)
                * bottom_source
                * mu_weight
            )

    layer_contribution = _restore_layer_order(layer_contribution_ordered, order)
    radiance_values = np.sum(layer_contribution, axis=0) + bottom_contribution
    radiance_values.setflags(write=False)
    radiance = Spectrum(
        spectral_grid=output_grid,
        values=radiance_values,
        unit="W m^-3 sr^-1",
        observable="spectral_radiance",
        metadata={
            "rt_solver": "clear_sky_numpy_reference",
            "bottom_boundary": bottom_mode,
            "scattering_treatment": "none",
        },
    )

    eclipse_depth = None
    if star_temperature_k is not None or planet_radius_m is not None or star_radius_m is not None:
        if star_temperature_k is None or planet_radius_m is None or star_radius_m is None:
            raise RobertValidationError(
                "star_temperature_k, planet_radius_m, and star_radius_m are all required for eclipse depth"
            )
        stellar_radiance = _planck_radiance_wavelength(wavelength, float(star_temperature_k))
        planet_radius = _positive_float(planet_radius_m, "planet_radius_m")
        star_radius = _positive_float(star_radius_m, "star_radius_m")
        depth = (radiance_values / stellar_radiance) * (planet_radius / star_radius) ** 2
        if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
            raise RobertValidationError("clear-sky eclipse-depth calculation produced invalid values")
        depth.setflags(write=False)
        eclipse_depth = Spectrum(
            spectral_grid=output_grid,
            values=depth,
            unit="eclipse_depth",
            observable="eclipse_depth",
            metadata={
                "rt_solver": "clear_sky_numpy_reference",
                "stellar_model": "blackbody",
                "scattering_treatment": "none",
            },
        )

    return ClearSkyEmissionResult(
        gas_optical_depth=gas_optical_depth,
        radiance=radiance,
        eclipse_depth=eclipse_depth,
        layer_source_function=source,
        layer_contribution_radiance=layer_contribution,
        bottom_contribution_radiance=bottom_contribution,
        emission_angle_cosines=mu,
        emission_angle_weights=mu_weights,
        metadata={
            "rt_solver": "clear_sky_numpy_reference",
            "bottom_boundary": bottom_mode,
            "scattering_treatment": "none",
            "opacity_sources": "gas_correlated_k",
        },
    )


def disk_average_quadrature(n_mu: int = 4) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return quadrature for disk-averaged thermal emission.

    The weights integrate `2 * integral_0^1 I(mu) * mu dmu` and therefore sum
    to one for a constant specific intensity.
    """

    n = int(n_mu)
    if n < 1:
        raise RobertValidationError("n_mu must be at least one")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mu = 0.5 * (nodes + 1.0)
    integral_weights = 0.5 * weights
    disk_weights = 2.0 * mu * integral_weights
    mu.setflags(write=False)
    disk_weights.setflags(write=False)
    return mu, disk_weights


def _normal_emission_quadrature() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mu = np.array([1.0], dtype=float)
    weights = np.array([1.0], dtype=float)
    mu.setflags(write=False)
    weights.setflags(write=False)
    return mu, weights


def _validate_emission_angle_quadrature(
    mu_values: ArrayLike,
    weights: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mu = _readonly_1d(mu_values, "emission_angle_cosines")
    quadrature_weights = _readonly_1d(weights, "emission_angle_weights")
    if mu.shape != quadrature_weights.shape:
        raise RobertValidationError("emission angle cosines and weights must have the same shape")
    if np.any(mu <= 0.0) or np.any(mu > 1.0):
        raise RobertValidationError("emission angle cosines must be in the interval (0, 1]")
    if np.any(quadrature_weights < 0.0):
        raise RobertValidationError("emission angle weights must be non-negative")
    total_weight = float(np.sum(quadrature_weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise RobertValidationError("emission angle weights must have a finite positive sum")
    normalized = np.array(quadrature_weights / total_weight, dtype=float, copy=True)
    normalized.setflags(write=False)
    return mu, normalized


def _layer_planck_source(
    wavelength_micron: NDArray[np.float64],
    temperature_k: NDArray[np.float64],
) -> NDArray[np.float64]:
    source = np.vstack(
        [_planck_radiance_wavelength(wavelength_micron, float(temperature)) for temperature in temperature_k]
    )
    source.setflags(write=False)
    return source


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


def _exclusive_cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    output = np.zeros_like(values)
    if values.shape[0] > 1:
        output[1:] = np.cumsum(values[:-1], axis=0)
    return output


def _top_to_bottom_order(gas_optical_depth: GasOpticalDepth) -> NDArray[np.int64]:
    pressure = np.asarray(gas_optical_depth.pressure_grid.centers, dtype=float)
    return np.argsort(pressure).astype(np.int64)


def _restore_layer_order(
    values_in_top_to_bottom_order: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    restored = np.empty_like(values_in_top_to_bottom_order)
    restored[order] = values_in_top_to_bottom_order
    restored.setflags(write=False)
    return restored


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


def _readonly_array(
    values: ArrayLike,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != shape:
        raise RobertValidationError(f"{name} has incorrect shape")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
