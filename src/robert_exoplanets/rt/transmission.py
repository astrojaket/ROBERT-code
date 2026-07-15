"""Absorption-dominated transmission through spherical atmospheric shells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import (
    pressure_values_in_unit,
    spectral_grid_values_in_unit,
)

from .optical_depth import GasOpticalDepth
from .path_geometry import HydrostaticPathGeometry


@dataclass(frozen=True)
class AbsorptionTransmissionResult:
    """Transit spectrum and spherical-annulus diagnostics."""

    transit_depth: Spectrum
    effective_radius_m: ArrayLike
    impact_radius_edges_m: ArrayLike
    annulus_area_contribution_m2: ArrayLike
    path_geometry: HydrostaticPathGeometry | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nspectral = self.transit_depth.spectral_grid.size
        radius = _readonly_array(self.effective_radius_m, "effective_radius_m", (nspectral,))
        edges = np.asarray(self.impact_radius_edges_m, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise RobertValidationError("impact_radius_edges_m must contain at least two edges")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise RobertValidationError("impact_radius_edges_m must be finite and increasing")
        contribution = _readonly_array(
            self.annulus_area_contribution_m2,
            "annulus_area_contribution_m2",
            (edges.size - 1, nspectral),
        )
        if np.any(radius <= 0.0) or np.any(contribution < 0.0):
            raise RobertValidationError("transmission radii and area contributions must be physical")
        edges = np.array(edges, copy=True)
        edges.setflags(write=False)
        object.__setattr__(self, "effective_radius_m", radius)
        object.__setattr__(self, "impact_radius_edges_m", edges)
        object.__setattr__(self, "annulus_area_contribution_m2", contribution)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


def solve_absorption_transmission(
    gas_optical_depth: GasOpticalDepth,
    path_geometry: HydrostaticPathGeometry,
    *,
    star_radius_m: float,
    additional_optical_depths: Sequence[object] | None = None,
    impact_quadrature_order: int = 8,
) -> AbsorptionTransmissionResult:
    """Calculate an absorption/extinction transit spectrum in spherical geometry.

    Each atmospheric layer is a constant-property spherical shell. Exact shell
    chord lengths are integrated over impact parameter with Gauss--Legendre
    quadrature. Correlated-k transmission is integrated over g before annulus
    area integration. Scattered photons returning to the observing beam,
    refraction, stellar limb darkening, and finite-stellar-disc effects are not
    included.
    """

    _validate_path_geometry_match(gas_optical_depth, path_geometry)
    star_radius = _positive_float(star_radius_m, "star_radius_m")
    if (
        isinstance(impact_quadrature_order, bool)
        or int(impact_quadrature_order) != impact_quadrature_order
        or int(impact_quadrature_order) < 2
    ):
        raise RobertValidationError(
            "impact_quadrature_order must be an integer of at least two"
        )
    impact_quadrature_order = int(impact_quadrature_order)

    tau = np.array(gas_optical_depth.total_tau, dtype=float, copy=True)
    opacity_sources = ["gas"]
    for contribution in additional_optical_depths or ():
        _validate_contribution_grid_match(gas_optical_depth, contribution)
        values = np.asarray(getattr(contribution, "tau", contribution), dtype=float)
        name = str(getattr(contribution, "name", type(contribution).__name__))
        if values.shape == tau.shape[:2]:
            values = values[:, :, None]
        if values.shape == tau.shape[:2] + (1,):
            values = np.broadcast_to(values, tau.shape)
        if values.shape != tau.shape:
            raise RobertValidationError(
                f"additional optical depth {name!r} must match layer, spectral, and g axes"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RobertValidationError(f"additional optical depth {name!r} must be non-negative")
        tau += values
        opacity_sources.append(name)

    weights = np.asarray(gas_optical_depth.g_weights, dtype=float)
    weights = weights / np.sum(weights)
    shell_inner = np.minimum(
        path_geometry.edge_radius_m[:-1],
        path_geometry.edge_radius_m[1:],
    )
    shell_outer = np.maximum(
        path_geometry.edge_radius_m[:-1],
        path_geometry.edge_radius_m[1:],
    )
    shell_thickness = shell_outer - shell_inner
    if np.any(shell_thickness <= 0.0):
        raise RobertValidationError("transmission shells must have positive thickness")
    impact_edges = np.unique(np.concatenate((shell_inner, shell_outer)))
    impact_edges.sort()
    base_radius = float(impact_edges[0])

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(impact_quadrature_order)
    annulus_contribution = np.zeros((impact_edges.size - 1, tau.shape[1]), dtype=float)
    for annulus_index, (lower, upper) in enumerate(
        zip(impact_edges[:-1], impact_edges[1:], strict=True)
    ):
        half_width = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        impact_radius = midpoint + half_width * nodes
        chord = _full_shell_chord_lengths(impact_radius, shell_inner, shell_outer)
        path_factor = chord / shell_thickness[None, :]
        slant_tau = np.einsum("ql,lsw->qsw", path_factor, tau, optimize=True)
        transmission = np.sum(np.exp(-slant_tau) * weights[None, None, :], axis=-1)
        transmission = np.clip(transmission, 0.0, 1.0)
        integrand = 2.0 * impact_radius[:, None] * (1.0 - transmission)
        annulus_contribution[annulus_index] = half_width * np.sum(
            quadrature_weights[:, None] * integrand,
            axis=0,
        )

    effective_radius_squared = base_radius**2 + np.sum(annulus_contribution, axis=0)
    effective_radius = np.sqrt(np.maximum(effective_radius_squared, base_radius**2))
    transit_depth_values = effective_radius_squared / star_radius**2
    if not np.all(np.isfinite(transit_depth_values)):
        raise RobertValidationError("transmission solve produced non-finite transit depths")

    wavelength = np.asarray(gas_optical_depth.spectral_grid.values, dtype=float)
    output_grid = SpectralGrid.from_array(
        wavelength,
        unit=gas_optical_depth.spectral_grid.unit,
        role="rt_native",
        name=gas_optical_depth.spectral_grid.name,
    )
    metadata = {
        "rt_solver": "spherical_shell_absorption_transmission",
        "scattering_treatment": "extinction_only_no_scattered_light_return",
        "refraction": "not_included",
        "stellar_limb_darkening": "not_included",
        "impact_quadrature_order": str(impact_quadrature_order),
        "base_radius_m": f"{base_radius:.17g}",
        "top_radius_m": f"{float(impact_edges[-1]):.17g}",
        "star_radius_m": f"{star_radius:.17g}",
        "opacity_sources": "+".join(opacity_sources),
        "top_layer_max_vertical_tau": (
            f"{float(np.max(tau[_top_layer_index(gas_optical_depth)])):.17g}"
        ),
        "bottom_layer_min_vertical_tau": (
            f"{float(np.min(tau[_bottom_layer_index(gas_optical_depth)])):.17g}"
        ),
    }
    transit_depth = Spectrum(
        spectral_grid=output_grid,
        values=transit_depth_values,
        unit="transit_depth",
        observable="transit_depth",
        metadata=metadata,
    )
    return AbsorptionTransmissionResult(
        transit_depth=transit_depth,
        effective_radius_m=effective_radius,
        impact_radius_edges_m=impact_edges,
        annulus_area_contribution_m2=annulus_contribution,
        path_geometry=path_geometry,
        metadata=metadata,
    )


def _top_layer_index(gas_optical_depth: GasOpticalDepth) -> int:
    pressure = pressure_values_in_unit(
        gas_optical_depth.pressure_grid.centers,
        gas_optical_depth.pressure_grid.unit,
        "pa",
    )
    return int(np.argmin(pressure))


def _bottom_layer_index(gas_optical_depth: GasOpticalDepth) -> int:
    pressure = pressure_values_in_unit(
        gas_optical_depth.pressure_grid.centers,
        gas_optical_depth.pressure_grid.unit,
        "pa",
    )
    return int(np.argmax(pressure))


def _validate_path_geometry_match(
    gas_optical_depth: GasOpticalDepth,
    path_geometry: HydrostaticPathGeometry,
) -> None:
    gas_edges = pressure_values_in_unit(
        gas_optical_depth.pressure_grid.edges,
        gas_optical_depth.pressure_grid.unit,
        "pa",
    )
    path_edges = pressure_values_in_unit(
        path_geometry.pressure_grid.edges,
        path_geometry.pressure_grid.unit,
        "pa",
    )
    if gas_edges.shape != path_edges.shape or not np.allclose(
        gas_edges,
        path_edges,
        rtol=1.0e-10,
        atol=0.0,
    ):
        raise RobertValidationError(
            "path geometry pressure grid must match gas optical-depth pressure grid"
        )


def _validate_contribution_grid_match(
    gas_optical_depth: GasOpticalDepth,
    contribution: object,
) -> None:
    if hasattr(contribution, "spectral_grid"):
        contribution_wavelength = spectral_grid_values_in_unit(
            getattr(contribution, "spectral_grid"),
            "micron",
        )
        gas_wavelength = spectral_grid_values_in_unit(
            gas_optical_depth.spectral_grid,
            "micron",
        )
        if contribution_wavelength.shape != gas_wavelength.shape or not np.allclose(
            contribution_wavelength,
            gas_wavelength,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RobertValidationError(
                "additional optical-depth spectral grid must match gas grid"
            )
    if hasattr(contribution, "pressure_grid"):
        contribution_pressure = pressure_values_in_unit(
            getattr(contribution, "pressure_grid").centers,
            getattr(contribution, "pressure_grid").unit,
            "pa",
        )
        gas_pressure = pressure_values_in_unit(
            gas_optical_depth.pressure_grid.centers,
            gas_optical_depth.pressure_grid.unit,
            "pa",
        )
        if contribution_pressure.shape != gas_pressure.shape or not np.allclose(
            contribution_pressure,
            gas_pressure,
            rtol=1.0e-10,
            atol=0.0,
        ):
            raise RobertValidationError(
                "additional optical-depth pressure grid must match gas grid"
            )


def _full_shell_chord_lengths(
    impact_radius: NDArray[np.float64],
    shell_inner: NDArray[np.float64],
    shell_outer: NDArray[np.float64],
) -> NDArray[np.float64]:
    impact_squared = impact_radius[:, None] ** 2
    outer = np.sqrt(np.maximum(shell_outer[None, :] ** 2 - impact_squared, 0.0))
    inner = np.sqrt(np.maximum(shell_inner[None, :] ** 2 - impact_squared, 0.0))
    chord = 2.0 * (outer - inner)
    return np.where(impact_radius[:, None] < shell_outer[None, :], chord, 0.0)


def _readonly_array(
    values: ArrayLike,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must be finite with shape {shape}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return number
