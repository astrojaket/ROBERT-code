"""Hydrostatic radius and path-geometry helpers for emission RT."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import pressure_values_in_unit

BOLTZMANN_CONSTANT_J_K = 1.380649e-23
ATOMIC_MASS_KG = 1.66053906660e-27


@dataclass(frozen=True)
class HydrostaticPathGeometry:
    """Hydrostatic layer radii and spherical shell path factors."""

    pressure_grid: PressureGrid
    reference_radius_m: float
    reference_pressure_pa: float
    gravity_m_s2: ArrayLike
    scale_height_m: ArrayLike
    edge_radius_m: ArrayLike
    center_radius_m: ArrayLike
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_layers = self.pressure_grid.n_layers
        reference_radius = _positive_float(self.reference_radius_m, "reference_radius_m")
        reference_pressure = _positive_float(self.reference_pressure_pa, "reference_pressure_pa")
        gravity = _readonly_1d(self.gravity_m_s2, "gravity_m_s2")
        scale_height = _readonly_1d(self.scale_height_m, "scale_height_m")
        edge_radius = _readonly_1d(self.edge_radius_m, "edge_radius_m")
        center_radius = _readonly_1d(self.center_radius_m, "center_radius_m")
        if gravity.shape != (n_layers,):
            raise RobertValidationError("gravity_m_s2 must match pressure grid layers")
        if scale_height.shape != (n_layers,):
            raise RobertValidationError("scale_height_m must match pressure grid layers")
        if edge_radius.shape != (n_layers + 1,):
            raise RobertValidationError("edge_radius_m must match pressure grid edges")
        if center_radius.shape != (n_layers,):
            raise RobertValidationError("center_radius_m must match pressure grid layers")
        if np.any(gravity <= 0.0):
            raise RobertValidationError("gravity_m_s2 values must be positive")
        if np.any(scale_height <= 0.0):
            raise RobertValidationError("scale_height_m values must be positive")
        if np.any(edge_radius <= 0.0) or np.any(center_radius <= 0.0):
            raise RobertValidationError("hydrostatic radii must be positive")
        if not np.isclose(
            self.radius_at_pressure(reference_pressure),
            reference_radius,
            rtol=1.0e-10,
            atol=max(1.0e-3, reference_radius * 1.0e-12),
        ):
            raise RobertValidationError("reference radius and pressure are inconsistent with edge radii")

        object.__setattr__(self, "reference_radius_m", reference_radius)
        object.__setattr__(self, "reference_pressure_pa", reference_pressure)
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "scale_height_m", scale_height)
        object.__setattr__(self, "edge_radius_m", edge_radius)
        object.__setattr__(self, "center_radius_m", center_radius)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def top_radius_m(self) -> float:
        """Radius at the lowest-pressure edge."""

        pressure = pressure_values_in_unit(self.pressure_grid.edges, self.pressure_grid.unit, "pa")
        return float(self.edge_radius_m[int(np.argmin(pressure))])

    @property
    def bottom_radius_m(self) -> float:
        """Radius at the highest-pressure edge."""

        pressure = pressure_values_in_unit(self.pressure_grid.edges, self.pressure_grid.unit, "pa")
        return float(self.edge_radius_m[int(np.argmax(pressure))])

    @property
    def layer_thickness_m(self) -> NDArray[np.float64]:
        """Geometric layer thicknesses in pressure-grid layer order."""

        thickness = np.abs(np.diff(self.edge_radius_m))
        thickness.setflags(write=False)
        return thickness

    def radius_at_pressure(self, pressure_pa: float) -> float:
        """Return hydrostatic radius at one pressure in Pa."""

        pressure = _positive_float(pressure_pa, "pressure_pa")
        edge_pressure = pressure_values_in_unit(self.pressure_grid.edges, self.pressure_grid.unit, "pa")
        layer_index = _layer_index_for_pressure(pressure, edge_pressure)
        radius = self.edge_radius_m[layer_index] - self.scale_height_m[layer_index] * (
            np.log(pressure) - np.log(edge_pressure[layer_index])
        )
        return float(radius)

    def emission_path_factors(self, emission_mu: ArrayLike) -> NDArray[np.float64]:
        """Return spherical path-length factors for each emission angle and layer.

        The output has shape `(point, layer)` in the pressure-grid layer order.
        Each factor is the shell chord length divided by the local radial layer
        thickness, so multiplying vertical optical depth by this factor gives a
        constant-property spherical slant optical depth.
        """

        mu = _readonly_1d(emission_mu, "emission_mu")
        if np.any(mu <= 0.0) or np.any(mu > 1.0):
            raise RobertValidationError("emission_mu must be in the interval (0, 1]")
        factors = np.zeros((mu.size, self.pressure_grid.n_layers), dtype=float)
        top_radius = self.top_radius_m
        shell_inner = np.minimum(self.edge_radius_m[:-1], self.edge_radius_m[1:])
        shell_outer = np.maximum(self.edge_radius_m[:-1], self.edge_radius_m[1:])
        thickness = shell_outer - shell_inner
        for point_index, mu_value in enumerate(mu):
            impact_parameter = top_radius * np.sqrt(max(0.0, 1.0 - float(mu_value) ** 2))
            factors[point_index] = _shell_path_lengths(
                impact_parameter,
                shell_inner,
                shell_outer,
            ) / thickness
        factors.setflags(write=False)
        return factors

    def bottom_visible(self, emission_mu: ArrayLike) -> NDArray[np.bool_]:
        """Return whether a ray reaches the deepest atmospheric boundary."""

        mu = _readonly_1d(emission_mu, "emission_mu")
        top_radius = self.top_radius_m
        impact_parameter = top_radius * np.sqrt(np.maximum(0.0, 1.0 - mu**2))
        return impact_parameter <= self.bottom_radius_m


def hydrostatic_path_geometry(
    atmosphere: AtmosphereState,
    *,
    gravity_m_s2: float | ArrayLike,
    reference_radius_m: float,
    reference_pressure: float,
    reference_pressure_unit: str = "bar",
) -> HydrostaticPathGeometry:
    """Build hydrostatic radius/path geometry from an atmosphere state.

    Layer temperatures, mean molecular weights, and gravity are treated as
    constant inside each pressure layer. The radius is anchored by requiring
    `reference_radius_m` at `reference_pressure`.
    """

    gravity = _gravity_profile(gravity_m_s2, atmosphere.n_layers)
    if atmosphere.mean_molecular_weight_unit.strip().lower() not in {
        "amu",
        "atomic_mass_unit",
        "atomic_mass_units",
        "u",
    }:
        raise RobertValidationError("hydrostatic path geometry requires mean molecular weight in amu")
    reference_pressure_pa = pressure_values_in_unit(
        np.array([reference_pressure], dtype=float),
        reference_pressure_unit,
        "pa",
    )[0]
    reference_radius = _positive_float(reference_radius_m, "reference_radius_m")
    reference_pressure_pa = _positive_float(reference_pressure_pa, "reference_pressure")
    particle_mass_kg = np.asarray(atmosphere.mean_molecular_weight, dtype=float) * ATOMIC_MASS_KG
    scale_height = BOLTZMANN_CONSTANT_J_K * atmosphere.temperature / (particle_mass_kg * gravity)
    if not np.all(np.isfinite(scale_height)) or np.any(scale_height <= 0.0):
        raise RobertValidationError("hydrostatic scale heights must be finite and positive")

    edge_pressure_pa = pressure_values_in_unit(atmosphere.pressure_grid.edges, atmosphere.pressure_grid.unit, "pa")
    _require_pressure_in_grid(reference_pressure_pa, edge_pressure_pa)
    center_pressure_pa = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "pa",
    )
    relative_edge_radius = np.zeros(atmosphere.n_layers + 1, dtype=float)
    for layer_index in range(atmosphere.n_layers):
        relative_edge_radius[layer_index + 1] = relative_edge_radius[layer_index] - scale_height[
            layer_index
        ] * np.log(edge_pressure_pa[layer_index + 1] / edge_pressure_pa[layer_index])
    relative_center_radius = np.array(
        [
            relative_edge_radius[layer_index]
            - scale_height[layer_index] * np.log(center_pressure_pa[layer_index] / edge_pressure_pa[layer_index])
            for layer_index in range(atmosphere.n_layers)
        ],
        dtype=float,
    )
    reference_relative_radius = _radius_at_pressure(
        reference_pressure_pa,
        edge_pressure_pa,
        relative_edge_radius,
        scale_height,
    )
    radius_offset = reference_radius - reference_relative_radius
    edge_radius = relative_edge_radius + radius_offset
    center_radius = relative_center_radius + radius_offset
    if np.any(edge_radius <= 0.0) or np.any(center_radius <= 0.0):
        raise RobertValidationError("hydrostatic radius grid produced non-positive radii")

    return HydrostaticPathGeometry(
        pressure_grid=atmosphere.pressure_grid,
        reference_radius_m=reference_radius,
        reference_pressure_pa=reference_pressure_pa,
        gravity_m_s2=gravity,
        scale_height_m=scale_height,
        edge_radius_m=edge_radius,
        center_radius_m=center_radius,
        metadata={
            "path_model": "hydrostatic_spherical_shell",
            "gravity_model": "supplied_layer_profile",
            "reference_radius_m": f"{reference_radius:.12g}",
            "reference_pressure_pa": f"{reference_pressure_pa:.12g}",
            "reference_pressure_unit": reference_pressure_unit,
        },
    )


def inverse_square_hydrostatic_path_geometry(
    atmosphere: AtmosphereState,
    *,
    reference_radius_m: float,
    reference_pressure: float,
    reference_gravity_m_s2: float,
    reference_pressure_unit: str = "bar",
    convergence_tolerance: float = 1.0e-12,
    max_iterations: int = 50,
) -> HydrostaticPathGeometry:
    """Build a self-consistent hydrostatic geometry with ``g(r) = GM / r^2``.

    Gravity is represented by one value at each layer centre, matching the
    constant-property layer convention used by gas-column and shell-path
    calculations. The layer gravity and radii are iterated to a fixed point so
    the returned profile can be used consistently by both opacity assembly and
    transmission geometry.
    """

    reference_radius = _positive_float(reference_radius_m, "reference_radius_m")
    reference_gravity = _positive_float(
        reference_gravity_m_s2,
        "reference_gravity_m_s2",
    )
    tolerance = float(convergence_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise RobertValidationError(
            "convergence_tolerance must be finite and positive"
        )
    if (
        isinstance(max_iterations, bool)
        or int(max_iterations) != max_iterations
        or int(max_iterations) < 1
    ):
        raise RobertValidationError("max_iterations must be a positive integer")

    gravity = np.full(atmosphere.n_layers, reference_gravity, dtype=float)
    geometry = None
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        geometry = hydrostatic_path_geometry(
            atmosphere,
            gravity_m_s2=gravity,
            reference_radius_m=reference_radius,
            reference_pressure=reference_pressure,
            reference_pressure_unit=reference_pressure_unit,
        )
        updated_gravity = reference_gravity * (
            reference_radius / geometry.center_radius_m
        ) ** 2
        if np.allclose(
            updated_gravity,
            gravity,
            rtol=tolerance,
            atol=0.0,
        ):
            gravity = updated_gravity
            break
        gravity = updated_gravity
    else:
        raise RobertValidationError(
            "inverse-square hydrostatic geometry did not converge"
        )

    geometry = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=gravity,
        reference_radius_m=reference_radius,
        reference_pressure=reference_pressure,
        reference_pressure_unit=reference_pressure_unit,
    )
    residual = reference_gravity * (
        reference_radius / geometry.center_radius_m
    ) ** 2
    if not np.allclose(residual, gravity, rtol=tolerance * 10.0, atol=0.0):
        raise RobertValidationError(
            "inverse-square hydrostatic geometry failed its final consistency check"
        )
    return replace(
        geometry,
        metadata={
            **dict(geometry.metadata),
            "gravity_model": "inverse_square_layer_center_fixed_point",
            "reference_gravity_m_s2": f"{reference_gravity:.12g}",
            "gravitational_parameter_m3_s2": (
                f"{reference_gravity * reference_radius**2:.12g}"
            ),
            "gravity_iterations": str(iterations),
            "gravity_convergence_tolerance": f"{tolerance:.12g}",
        },
    )


def _radius_at_pressure(
    pressure_pa: float,
    edge_pressure_pa: NDArray[np.float64],
    edge_radius_m: NDArray[np.float64],
    scale_height_m: NDArray[np.float64],
) -> float:
    layer_index = _layer_index_for_pressure(pressure_pa, edge_pressure_pa)
    return float(
        edge_radius_m[layer_index]
        - scale_height_m[layer_index] * (np.log(pressure_pa) - np.log(edge_pressure_pa[layer_index]))
    )


def _layer_index_for_pressure(pressure_pa: float, edge_pressure_pa: NDArray[np.float64]) -> int:
    log_pressure = np.log(pressure_pa)
    log_edges = np.log(edge_pressure_pa)
    for layer_index in range(edge_pressure_pa.size - 1):
        lower = min(log_edges[layer_index], log_edges[layer_index + 1])
        upper = max(log_edges[layer_index], log_edges[layer_index + 1])
        if lower <= log_pressure <= upper:
            return layer_index
    raise RobertValidationError("pressure lies outside the hydrostatic pressure grid")


def _require_pressure_in_grid(
    pressure_pa: float,
    edge_pressure_pa: NDArray[np.float64],
) -> None:
    lower = float(np.min(edge_pressure_pa))
    upper = float(np.max(edge_pressure_pa))
    if pressure_pa < lower or pressure_pa > upper:
        raise RobertValidationError(
            "reference pressure must lie within the atmospheric pressure-grid edges"
        )


def _shell_path_lengths(
    impact_parameter_m: float,
    shell_inner_radius_m: NDArray[np.float64],
    shell_outer_radius_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    outer = np.sqrt(np.maximum(shell_outer_radius_m**2 - impact_parameter_m**2, 0.0))
    inner = np.sqrt(np.maximum(shell_inner_radius_m**2 - impact_parameter_m**2, 0.0))
    path = np.where(impact_parameter_m < shell_outer_radius_m, outer - inner, 0.0)
    path = np.maximum(path, 0.0)
    return path


def _gravity_profile(values: float | ArrayLike, n_layers: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim == 0:
        array = np.full(n_layers, float(array), dtype=float)
    if array.ndim != 1:
        raise RobertValidationError("gravity_m_s2 must be scalar or one-dimensional")
    if array.shape != (n_layers,):
        raise RobertValidationError("gravity_m_s2 must match pressure grid layers")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise RobertValidationError("gravity_m_s2 values must be finite and positive")
    array.setflags(write=False)
    return array


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return number


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
