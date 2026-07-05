"""Clear-sky thermal-emission reference solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.opacity import spectral_grid_values_in_unit

from .geometry import (
    DiscGeometry,
    gauss_legendre_disk_geometry,
    geometry_from_emission_angles,
    normal_emission_geometry,
)
from .optical_depth import GasOpticalDepth
from .scattering import SingleScatteringSource

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
    geometry: DiscGeometry | None = None
    point_radiance: ArrayLike | None = None
    point_layer_contribution_radiance: ArrayLike | None = None
    point_bottom_contribution_radiance: ArrayLike | None = None
    scattering_layer_contribution_radiance: ArrayLike | None = None
    point_scattering_source_function: ArrayLike | None = None
    point_scattering_contribution_radiance: ArrayLike | None = None
    total_optical_depth: ArrayLike | None = None
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
        if self.total_optical_depth is None:
            total_tau = self.gas_optical_depth.total_tau
        else:
            total_tau = _readonly_array(
                self.total_optical_depth,
                "total_optical_depth",
                self.gas_optical_depth.total_tau.shape,
            )
            if np.any(total_tau < 0.0):
                raise RobertValidationError("total_optical_depth must be non-negative")
        mu, weights = _validate_emission_angle_quadrature(
            self.emission_angle_cosines,
            self.emission_angle_weights,
        )
        geometry = self.geometry
        if geometry is None:
            geometry = geometry_from_emission_angles(
                mu,
                weights,
                name="result_emission_quadrature",
                quadrature="result_mu",
            )
        else:
            if geometry.emission_angle_cosines.shape != mu.shape:
                raise RobertValidationError("geometry and emission angle quadrature must have matching shapes")
            if not np.allclose(geometry.emission_angle_cosines, mu, rtol=1.0e-12, atol=0.0):
                raise RobertValidationError("geometry emission angles must match emission_angle_cosines")
            if not np.allclose(geometry.emission_angle_weights, weights, rtol=1.0e-12, atol=0.0):
                raise RobertValidationError("geometry weights must match emission_angle_weights")

        point_radiance = None
        if self.point_radiance is not None:
            point_radiance = _readonly_array(
                self.point_radiance,
                "point_radiance",
                (mu.size, n_spectral),
            )
            if np.any(point_radiance < 0.0):
                raise RobertValidationError("point_radiance must be non-negative")
        point_layer_contribution = None
        if self.point_layer_contribution_radiance is not None:
            point_layer_contribution = _readonly_array(
                self.point_layer_contribution_radiance,
                "point_layer_contribution_radiance",
                (mu.size, n_layers, n_spectral),
            )
            if np.any(point_layer_contribution < 0.0):
                raise RobertValidationError("point_layer_contribution_radiance must be non-negative")
        point_bottom_contribution = None
        if self.point_bottom_contribution_radiance is not None:
            point_bottom_contribution = _readonly_array(
                self.point_bottom_contribution_radiance,
                "point_bottom_contribution_radiance",
                (mu.size, n_spectral),
            )
            if np.any(point_bottom_contribution < 0.0):
                raise RobertValidationError("point_bottom_contribution_radiance must be non-negative")

        scattering_layer_contribution = None
        if self.scattering_layer_contribution_radiance is not None:
            scattering_layer_contribution = _readonly_array(
                self.scattering_layer_contribution_radiance,
                "scattering_layer_contribution_radiance",
                (n_layers, n_spectral),
            )
            if np.any(scattering_layer_contribution < 0.0):
                raise RobertValidationError("scattering_layer_contribution_radiance must be non-negative")
        point_scattering_source = None
        if self.point_scattering_source_function is not None:
            point_scattering_source = _readonly_array(
                self.point_scattering_source_function,
                "point_scattering_source_function",
                (mu.size, n_layers, n_spectral),
            )
            if np.any(point_scattering_source < 0.0):
                raise RobertValidationError("point_scattering_source_function must be non-negative")
        point_scattering_contribution = None
        if self.point_scattering_contribution_radiance is not None:
            point_scattering_contribution = _readonly_array(
                self.point_scattering_contribution_radiance,
                "point_scattering_contribution_radiance",
                (mu.size, n_layers, n_spectral),
            )
            if np.any(point_scattering_contribution < 0.0):
                raise RobertValidationError("point_scattering_contribution_radiance must be non-negative")

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
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "point_radiance", point_radiance)
        object.__setattr__(self, "point_layer_contribution_radiance", point_layer_contribution)
        object.__setattr__(self, "point_bottom_contribution_radiance", point_bottom_contribution)
        object.__setattr__(self, "scattering_layer_contribution_radiance", scattering_layer_contribution)
        object.__setattr__(self, "point_scattering_source_function", point_scattering_source)
        object.__setattr__(self, "point_scattering_contribution_radiance", point_scattering_contribution)
        object.__setattr__(self, "total_optical_depth", total_tau)
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
    geometry: DiscGeometry | None = None,
    bottom_boundary: str = "blackbody",
    additional_optical_depths: Sequence[object] | None = None,
    scattering_source: SingleScatteringSource | None = None,
    planet_radius_m: float | None = None,
    star_radius_m: float | None = None,
    star_temperature_k: float | None = None,
) -> ClearSkyEmissionResult:
    """Solve thermal emission for a clear atmosphere.

    The solver integrates Planck layer source functions through gas optical
    depth plus any additional extinction optical depths. When a
    ``SingleScatteringSource`` is supplied, scattering optical depths in
    ``additional_optical_depths`` add a first-order direct-beam scattering
    source term; otherwise scattering remains extinction-only.
    """

    bottom_mode = bottom_boundary.strip().lower()
    if bottom_mode not in {"blackbody", "none"}:
        raise RobertValidationError("bottom_boundary must be 'blackbody' or 'none'")

    emission_geometry = _resolve_emission_geometry(
        geometry,
        emission_angle_cosines,
        emission_angle_weights,
    )
    mu = emission_geometry.emission_angle_cosines
    mu_weights = emission_geometry.emission_angle_weights

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
    (
        total_tau,
        opacity_sources,
        scattering_treatment,
        scattering_tau,
        scattering_sources,
    ) = _total_optical_depth(
        gas_optical_depth,
        additional_optical_depths,
    )
    tau_ordered = total_tau[order]
    scattering_tau_ordered = scattering_tau[order]
    scattering_beam = None
    scattering_phase = None
    stellar_mu = None
    if scattering_source is not None:
        if not scattering_sources or not np.any(scattering_tau > 0.0):
            raise RobertValidationError(
                "scattering_source requires at least one positive scattering optical-depth contribution"
            )
        stellar_mu = emission_geometry.stellar_mu
        if not np.all(np.isfinite(stellar_mu)):
            raise RobertValidationError("scattering_source requires finite geometry stellar_mu values")
        scattering_beam = scattering_source.stellar_beam.values_on(output_grid)
        scattering_phase = scattering_source.phase_function_values(emission_geometry)
        scattering_treatment = "single_scattering_direct_beam"

    point_layer_contribution_ordered = np.zeros(
        (mu.size, gas_optical_depth.atmosphere.n_layers, wavelength.size),
        dtype=float,
    )
    point_bottom_contribution = np.zeros((mu.size, wavelength.size), dtype=float)
    point_scattering_source_ordered = None
    point_scattering_contribution_ordered = None
    if scattering_source is not None:
        point_scattering_source_ordered = np.zeros_like(point_layer_contribution_ordered)
        point_scattering_contribution_ordered = np.zeros_like(point_layer_contribution_ordered)
    bottom_source = None
    if bottom_mode == "blackbody":
        deepest_temperature = float(
            gas_optical_depth.atmosphere.temperature[
                int(np.argmax(gas_optical_depth.pressure_grid.centers))
            ]
        )
        bottom_source = _planck_radiance_wavelength(wavelength, deepest_temperature)

    for point_index, mu_value in enumerate(mu):
        slant_tau = tau_ordered / mu_value
        cumulative_before = _exclusive_cumulative(slant_tau)
        transmission_before = np.exp(-cumulative_before)
        layer_escape = transmission_before * (-np.expm1(-slant_tau))
        layer_radiance_by_g = source_ordered[:, :, None] * layer_escape
        point_layer_contribution_ordered[point_index] = np.sum(
            layer_radiance_by_g * gas_optical_depth.g_weights[None, None, :],
            axis=-1,
        )
        if scattering_source is not None:
            if (
                scattering_beam is None
                or scattering_phase is None
                or stellar_mu is None
                or point_scattering_source_ordered is None
                or point_scattering_contribution_ordered is None
            ):
                raise RobertValidationError("single-scattering source was not initialized")
            mu0 = float(stellar_mu[point_index])
            if mu0 > 0.0:
                incoming_midpoint_tau = (cumulative_before + 0.5 * tau_ordered) / mu0
                incoming_transmission = np.exp(-incoming_midpoint_tau)
                single_scattering_albedo = np.divide(
                    scattering_tau_ordered,
                    tau_ordered,
                    out=np.zeros_like(tau_ordered),
                    where=tau_ordered > 0.0,
                )
                scattering_source_by_g = (
                    scattering_beam[None, :, None]
                    * (float(scattering_phase[point_index]) / (4.0 * np.pi))
                    * single_scattering_albedo
                    * incoming_transmission
                )
                scattering_contribution_by_g = scattering_source_by_g * layer_escape
                point_scattering_source_ordered[point_index] = np.sum(
                    scattering_source_by_g * gas_optical_depth.g_weights[None, None, :],
                    axis=-1,
                )
                point_scattering_contribution_ordered[point_index] = np.sum(
                    scattering_contribution_by_g * gas_optical_depth.g_weights[None, None, :],
                    axis=-1,
                )
                point_layer_contribution_ordered[point_index] += point_scattering_contribution_ordered[
                    point_index
                ]
        if bottom_mode == "blackbody":
            total_transmission = np.exp(-np.sum(slant_tau, axis=0))
            point_bottom_contribution[point_index] = (
                np.sum(total_transmission * gas_optical_depth.g_weights[None, :], axis=-1)
                * bottom_source
            )

    layer_contribution_ordered = np.tensordot(mu_weights, point_layer_contribution_ordered, axes=(0, 0))
    bottom_contribution = np.tensordot(mu_weights, point_bottom_contribution, axes=(0, 0))
    scattering_layer_contribution = None
    point_scattering_source = None
    point_scattering_contribution = None
    if point_scattering_contribution_ordered is not None and point_scattering_source_ordered is not None:
        scattering_layer_contribution_ordered = np.tensordot(
            mu_weights,
            point_scattering_contribution_ordered,
            axes=(0, 0),
        )
        scattering_layer_contribution = _restore_layer_order(scattering_layer_contribution_ordered, order)
        point_scattering_source = _restore_point_layer_order(point_scattering_source_ordered, order)
        point_scattering_contribution = _restore_point_layer_order(
            point_scattering_contribution_ordered,
            order,
        )
    layer_contribution = _restore_layer_order(layer_contribution_ordered, order)
    point_layer_contribution = _restore_point_layer_order(point_layer_contribution_ordered, order)
    point_radiance = np.sum(point_layer_contribution, axis=1) + point_bottom_contribution
    point_radiance.setflags(write=False)
    point_bottom_contribution.setflags(write=False)
    radiance_values = np.sum(layer_contribution, axis=0) + bottom_contribution
    radiance_values.setflags(write=False)
    common_metadata = {
        "rt_solver": "clear_sky_numpy_reference",
        "bottom_boundary": bottom_mode,
        "source_function": "thermal_planck"
        if scattering_source is None
        else "thermal_planck_plus_single_scattering",
        "scattering_treatment": scattering_treatment,
        "scattering_source_function": "not_included"
        if scattering_source is None
        else scattering_source.name,
        "opacity_sources": "+".join(opacity_sources),
        "geometry": emission_geometry.name,
        "geometry_quadrature": emission_geometry.quadrature,
        "geometry_n_points": str(emission_geometry.n_points),
    }
    if emission_geometry.phase_angle_deg is not None:
        common_metadata["phase_angle_deg"] = f"{emission_geometry.phase_angle_deg:.12g}"
    if scattering_source is not None:
        common_metadata["scattering_phase_function"] = scattering_source.phase_function
        common_metadata["scattering_sources"] = "+".join(scattering_sources)
    radiance = Spectrum(
        spectral_grid=output_grid,
        values=radiance_values,
        unit="W m^-3 sr^-1",
        observable="spectral_radiance",
        metadata=common_metadata,
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
        eclipse_metadata = dict(common_metadata)
        eclipse_metadata["stellar_model"] = "blackbody"
        eclipse_depth = Spectrum(
            spectral_grid=output_grid,
            values=depth,
            unit="eclipse_depth",
            observable="eclipse_depth",
            metadata=eclipse_metadata,
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
        geometry=emission_geometry,
        point_radiance=point_radiance,
        point_layer_contribution_radiance=point_layer_contribution,
        point_bottom_contribution_radiance=point_bottom_contribution,
        scattering_layer_contribution_radiance=scattering_layer_contribution,
        point_scattering_source_function=point_scattering_source,
        point_scattering_contribution_radiance=point_scattering_contribution,
        total_optical_depth=total_tau,
        metadata=common_metadata,
    )


def disk_average_quadrature(n_mu: int = 4) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return quadrature for disk-averaged thermal emission.

    The weights integrate `2 * integral_0^1 I(mu) * mu dmu` and therefore sum
    to one for a constant specific intensity.
    """

    geometry = gauss_legendre_disk_geometry(n_mu)
    return geometry.emission_angle_cosines, geometry.emission_angle_weights


def _normal_emission_quadrature() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    geometry = normal_emission_geometry()
    return geometry.emission_angle_cosines, geometry.emission_angle_weights


def _resolve_emission_geometry(
    geometry: DiscGeometry | None,
    emission_angle_cosines: ArrayLike | None,
    emission_angle_weights: ArrayLike | None,
) -> DiscGeometry:
    if geometry is not None:
        if emission_angle_cosines is not None or emission_angle_weights is not None:
            raise RobertValidationError("geometry cannot be combined with emission angle quadrature inputs")
        return geometry
    if emission_angle_cosines is None and emission_angle_weights is None:
        return normal_emission_geometry()
    if emission_angle_cosines is not None and emission_angle_weights is not None:
        return geometry_from_emission_angles(
            emission_angle_cosines,
            emission_angle_weights,
            name="emission_angle_quadrature",
            quadrature="custom_mu",
        )
    raise RobertValidationError("emission_angle_cosines and emission_angle_weights must be provided together")


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


def _total_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    additional_optical_depths: Sequence[object] | None,
) -> tuple[NDArray[np.float64], tuple[str, ...], str, NDArray[np.float64], tuple[str, ...]]:
    total_tau = np.array(gas_optical_depth.total_tau, dtype=float, copy=True)
    scattering_tau = np.zeros_like(total_tau)
    sources = ["gas_correlated_k"]
    scattering_sources = []
    has_scattering_extinction = False
    if additional_optical_depths is not None:
        for contribution in additional_optical_depths:
            if contribution is None:
                continue
            name = str(getattr(contribution, "name", "additional_extinction"))
            kind = str(getattr(contribution, "kind", "extinction"))
            tau_values = getattr(contribution, "tau", contribution)
            tau = np.array(tau_values, dtype=float, copy=True)
            if tau.shape == total_tau.shape[:2]:
                tau_for_g = tau[:, :, None]
            elif tau.shape == total_tau.shape:
                tau_for_g = tau
            else:
                raise RobertValidationError(
                    "additional optical depths must have shape layer x wavelength "
                    "or layer x wavelength x g"
                )
            if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
                raise RobertValidationError("additional optical depths must be finite and non-negative")
            total_tau += tau_for_g
            sources.append(name)
            if "scattering" in kind.lower():
                has_scattering_extinction = True
                scattering_sources.append(name)
                scattering_tau += tau_for_g

    if not np.all(np.isfinite(total_tau)) or np.any(total_tau < 0.0):
        raise RobertValidationError("total optical depth must be finite and non-negative")
    total_tau.setflags(write=False)
    scattering_tau.setflags(write=False)
    scattering_treatment = "extinction_only_no_scattering_source" if has_scattering_extinction else "none"
    return total_tau, tuple(sources), scattering_treatment, scattering_tau, tuple(scattering_sources)


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


def _restore_point_layer_order(
    values_in_top_to_bottom_order: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    restored = np.empty_like(values_in_top_to_bottom_order)
    restored[:, order, :] = values_in_top_to_bottom_order
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
