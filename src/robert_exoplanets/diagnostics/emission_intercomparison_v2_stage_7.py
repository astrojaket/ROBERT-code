"""Frozen absorbing-cloud contract for emission intercomparison V2 Stage 7.

This module defines cloud placement and extinction only.  It does not provide
scattering physics: every Stage-7 single-scattering albedo is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError


CLOUD_OPTICAL_DEPTHS = (0.1, 1.0, 10.0, 100.0)
CLOUD_TOP_PRESSURES_BAR = (1.0e-3, 1.0e-2, 1.0e-1)
EXTINCTION_SLOPES = (-4.0, -2.0, 0.0, 2.0)
REFERENCE_WAVELENGTH_MICRON = 5.0
CLOUD_BOTTOM_PRESSURE_BAR = 100.0
MEMORY_LIMIT_FRACTION = 0.60
WALL_TIME_LIMIT_S = 7200.0
ASSEMBLY_OVERHEAD_FACTOR = 1.20


@dataclass(frozen=True)
class CloudDefinition:
    """One immutable absorbing-cloud definition.

    Parameters use bar, micron, and dimensionless vertical optical depth.
    ``optical_depth_at_reference`` is the complete column optical depth from
    the continuous cloud top to the frozen 100-bar cloud bottom.
    """

    label: str
    kind: str
    optical_depth_at_reference: float
    cloud_top_pressure_bar: float
    extinction_slope: float
    single_scattering_albedo: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"clear", "power_law_deck", "archived_tabulated"}:
            raise RobertValidationError(f"unsupported Stage-7 cloud kind: {self.kind}")
        if not self.label:
            raise RobertValidationError("cloud label must not be empty")
        if self.single_scattering_albedo != 0.0:
            raise RobertValidationError("Stage 7 requires exact omega0=0")


def fractional_log_pressure_weights(
    pressure_edges_bar: ArrayLike,
    cloud_top_pressure_bar: float,
    *,
    cloud_bottom_pressure_bar: float = CLOUD_BOTTOM_PRESSURE_BAR,
) -> NDArray[np.float64]:
    """Return fractional cell placement for uniform ``d tau / d log(P)``.

    Cloud boundaries are continuous.  An intersected boundary cell receives
    its exact fractional log-pressure overlap; no cloud-top snapping occurs.
    """

    edges = np.asarray(pressure_edges_bar, dtype=float)
    top = float(cloud_top_pressure_bar)
    bottom = float(cloud_bottom_pressure_bar)
    if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)):
        raise RobertValidationError("pressure edges must be a finite one-dimensional grid")
    if np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise RobertValidationError("pressure edges must be positive and increasing")
    if not np.isfinite(top) or not np.isfinite(bottom) or not 0.0 < top < bottom:
        raise RobertValidationError("cloud top and bottom must be finite with 0 < top < bottom")
    if top < edges[0] or bottom > edges[-1]:
        raise RobertValidationError("cloud boundaries must lie within the pressure grid")

    lower = np.maximum(edges[:-1], top)
    upper = np.minimum(edges[1:], bottom)
    overlap = np.where(upper > lower, np.log(upper / lower), 0.0)
    normalization = np.log(bottom / top)
    weights = overlap / normalization
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2.0e-15):
        raise RobertValidationError("fractional cloud placement does not integrate to unity")
    return weights


def power_law_cloud_tau(
    pressure_edges_bar: ArrayLike,
    wavelength_micron: ArrayLike,
    *,
    optical_depth_at_reference: float,
    cloud_top_pressure_bar: float,
    extinction_slope: float,
    reference_wavelength_micron: float = REFERENCE_WAVELENGTH_MICRON,
) -> NDArray[np.float64]:
    """Evaluate a pure-absorption deck on pressure cells and wavelengths.

    The sign convention is
    ``tau(lambda)=tau_ref*(lambda/reference_wavelength)**slope``.  Negative
    slopes therefore strengthen extinction toward shorter wavelengths.
    """

    wavelength = np.asarray(wavelength_micron, dtype=float)
    tau_reference = float(optical_depth_at_reference)
    slope = float(extinction_slope)
    reference = float(reference_wavelength_micron)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise RobertValidationError("wavelength must be finite, positive, and one-dimensional")
    if not np.isfinite(tau_reference) or tau_reference < 0.0:
        raise RobertValidationError("reference optical depth must be finite and non-negative")
    if not np.isfinite(slope) or not np.isfinite(reference) or reference <= 0.0:
        raise RobertValidationError("spectral slope and reference wavelength are invalid")
    vertical = fractional_log_pressure_weights(
        pressure_edges_bar, cloud_top_pressure_bar
    )
    spectral = tau_reference * (wavelength / reference) ** slope
    return vertical[:, None] * spectral[None, :]


def _interpolate_extinction_row(
    source_wavelength_micron: NDArray[np.float64],
    source_values: NDArray[np.float64],
    target_wavelength_micron: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate positive extinction log-linearly without epsilon floors."""

    output = np.empty(target_wavelength_micron.size, dtype=float)
    below = target_wavelength_micron <= source_wavelength_micron[0]
    above = target_wavelength_micron >= source_wavelength_micron[-1]
    output[below] = source_values[0]
    output[above] = source_values[-1]
    interior = ~(below | above)
    for target_index in np.flatnonzero(interior):
        wavelength = target_wavelength_micron[target_index]
        right = int(np.searchsorted(source_wavelength_micron, wavelength, side="right"))
        left = right - 1
        fraction = (
            np.log(wavelength / source_wavelength_micron[left])
            / np.log(source_wavelength_micron[right] / source_wavelength_micron[left])
        )
        left_value = source_values[left]
        right_value = source_values[right]
        if left_value > 0.0 and right_value > 0.0:
            output[target_index] = np.exp(
                (1.0 - fraction) * np.log(left_value)
                + fraction * np.log(right_value)
            )
        else:
            output[target_index] = (
                (1.0 - fraction) * left_value + fraction * right_value
            )
    return output


def regrid_tabulated_extinction_tau(
    source_pressure_edges_bar: ArrayLike,
    source_wavelength_micron: ArrayLike,
    source_extinction_tau: ArrayLike,
    target_pressure_edges_bar: ArrayLike,
    target_wavelength_micron: ArrayLike,
) -> NDArray[np.float64]:
    """Conservatively remap the archived physical extinction field.

    Pressure remapping assumes uniform ``d tau / d log(P)`` within each source
    cell and therefore conserves integrated optical depth.  Positive spectral
    values are log-linearly interpolated.  Exact zeros remain exact zeros and
    no epsilon is introduced.  Wavelengths outside the archived 1--12 micron
    range use constant endpoint extension.
    """

    source_edges = np.asarray(source_pressure_edges_bar, dtype=float)
    source_wavelength = np.asarray(source_wavelength_micron, dtype=float)
    source_tau = np.asarray(source_extinction_tau, dtype=float)
    target_edges = np.asarray(target_pressure_edges_bar, dtype=float)
    target_wavelength = np.asarray(target_wavelength_micron, dtype=float)
    for label, values in (
        ("source pressure", source_edges),
        ("target pressure", target_edges),
        ("source wavelength", source_wavelength),
        ("target wavelength", target_wavelength),
    ):
        if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise RobertValidationError(f"{label} must be finite and positive")
        if np.any(np.diff(values) <= 0.0):
            raise RobertValidationError(f"{label} must be strictly increasing")
    if source_tau.shape != (source_edges.size - 1, source_wavelength.size):
        raise RobertValidationError("source extinction must have pressure-cell by wavelength shape")
    if np.any(~np.isfinite(source_tau)) or np.any(source_tau < 0.0):
        raise RobertValidationError("source extinction must be finite and non-negative")
    if target_edges[0] > source_edges[0] or target_edges[-1] < source_edges[-1]:
        raise RobertValidationError("target pressure grid must cover the archived field")

    source_log = np.log(source_edges)
    target_log = np.log(target_edges)
    pressure_remapped = np.zeros((target_edges.size - 1, source_wavelength.size))
    for source_index in range(source_edges.size - 1):
        width = source_log[source_index + 1] - source_log[source_index]
        low = np.maximum(target_log[:-1], source_log[source_index])
        high = np.minimum(target_log[1:], source_log[source_index + 1])
        overlap_fraction = np.clip(high - low, 0.0, None) / width
        pressure_remapped += overlap_fraction[:, None] * source_tau[source_index]

    output = np.zeros((target_edges.size - 1, target_wavelength.size))
    for layer_index, values in enumerate(pressure_remapped):
        if np.any(values > 0.0):
            output[layer_index] = _interpolate_extinction_row(
                source_wavelength, values, target_wavelength
            )
    return output


def cloud_definitions() -> tuple[CloudDefinition, ...]:
    """Return clear, 48 parametric, and one archived cloud definitions."""

    definitions = [
        CloudDefinition("clear", "clear", 0.0, np.nan, 0.0)
    ]
    for tau_reference in CLOUD_OPTICAL_DEPTHS:
        for top_pressure in CLOUD_TOP_PRESSURES_BAR:
            for slope in EXTINCTION_SLOPES:
                definitions.append(
                    CloudDefinition(
                        label=(
                            f"deck_tau{tau_reference:g}_top{top_pressure * 1e3:g}"
                            f"mbar_slope{slope:+g}"
                        ),
                        kind="power_law_deck",
                        optical_depth_at_reference=tau_reference,
                        cloud_top_pressure_bar=top_pressure,
                        extinction_slope=slope,
                    )
                )
    definitions.append(
        CloudDefinition(
            "archived_virga_mie_extinction",
            "archived_tabulated",
            np.nan,
            np.nan,
            np.nan,
        )
    )
    return tuple(definitions)


def build_cloud_extinction_matrix(
    pressure_edges_bar: ArrayLike,
    wavelength_micron: ArrayLike,
    *,
    archived_pressure_edges_bar: ArrayLike,
    archived_wavelength_micron: ArrayLike,
    archived_extinction_tau: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate every frozen Stage-7 cloud definition on a target grid."""

    edges = np.asarray(pressure_edges_bar, dtype=float)
    wavelength = np.asarray(wavelength_micron, dtype=float)
    definitions = cloud_definitions()
    matrix = np.zeros((len(definitions), edges.size - 1, wavelength.size))
    for index, definition in enumerate(definitions[1:-1], start=1):
        matrix[index] = power_law_cloud_tau(
            edges,
            wavelength,
            optical_depth_at_reference=definition.optical_depth_at_reference,
            cloud_top_pressure_bar=definition.cloud_top_pressure_bar,
            extinction_slope=definition.extinction_slope,
        )
    matrix[-1] = regrid_tabulated_extinction_tau(
        archived_pressure_edges_bar,
        archived_wavelength_micron,
        archived_extinction_tau,
        edges,
        wavelength,
    )
    return matrix


def pilot_resource_decision(
    timings: Mapping[str, Mapping[str, float]],
    *,
    pilot_case_count: int,
    peak_process_tree_rss_bytes: int,
    available_memory_bytes: int,
    profile_count_including_control: int = 3,
    cloud_case_count: int = 50,
    resolution_count: int = 3,
) -> dict[str, float | int | bool | str]:
    """Project the complete matrix from frozen cold/warm pilot measurements."""

    if pilot_case_count <= 0 or available_memory_bytes <= 0:
        raise RobertValidationError("pilot case count and available memory must be positive")
    projected_worker_wall = 0.0
    for label, values in timings.items():
        cold = float(values["cold_wall_time_s"])
        warm = float(values["warm_wall_time_s"])
        if cold < 0.0 or warm < 0.0 or not np.isfinite(cold + warm):
            raise RobertValidationError(f"invalid pilot timing for {label}")
        setup = max(cold - warm, 0.0)
        warm_per_case = warm / pilot_case_count
        projected_worker_wall += resolution_count * (
            setup + warm_per_case * profile_count_including_control * cloud_case_count
        )
    projected_total = projected_worker_wall * ASSEMBLY_OVERHEAD_FACTOR
    memory_fraction = peak_process_tree_rss_bytes / available_memory_bytes
    return {
        "pilot_case_count_per_framework_track": pilot_case_count,
        "profile_count_including_isothermal_control": profile_count_including_control,
        "cloud_case_count_per_profile": cloud_case_count,
        "resolution_count": resolution_count,
        "projection_method": (
            "sum over framework/track of three resolution workers times cold-minus-warm "
            "setup plus warm per-case time times three profiles and 50 clouds; multiply "
            "by frozen 1.20 plot/checksum/report assembly overhead"
        ),
        "projected_worker_wall_time_s": projected_worker_wall,
        "assembly_overhead_factor": ASSEMBLY_OVERHEAD_FACTOR,
        "projected_complete_wall_time_s": projected_total,
        "wall_time_limit_s": WALL_TIME_LIMIT_S,
        "peak_process_tree_rss_bytes": int(peak_process_tree_rss_bytes),
        "available_memory_bytes_at_decision": int(available_memory_bytes),
        "peak_rss_fraction_of_available": memory_fraction,
        "memory_limit_fraction_of_available": MEMORY_LIMIT_FRACTION,
        "wall_time_safe": bool(projected_total <= WALL_TIME_LIMIT_S),
        "memory_safe": bool(memory_fraction <= MEMORY_LIMIT_FRACTION),
        "continue_full_matrix": bool(
            projected_total <= WALL_TIME_LIMIT_S
            and memory_fraction <= MEMORY_LIMIT_FRACTION
        ),
    }


__all__ = [
    "ASSEMBLY_OVERHEAD_FACTOR",
    "CLOUD_BOTTOM_PRESSURE_BAR",
    "CLOUD_OPTICAL_DEPTHS",
    "CLOUD_TOP_PRESSURES_BAR",
    "CloudDefinition",
    "EXTINCTION_SLOPES",
    "MEMORY_LIMIT_FRACTION",
    "REFERENCE_WAVELENGTH_MICRON",
    "WALL_TIME_LIMIT_S",
    "build_cloud_extinction_matrix",
    "cloud_definitions",
    "fractional_log_pressure_weights",
    "pilot_resource_decision",
    "power_law_cloud_tau",
    "regrid_tabulated_extinction_tau",
]
