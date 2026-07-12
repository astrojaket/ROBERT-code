"""Gas optical-depth assembly for RT reference calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, RobertValidationError, SpectralGrid
from robert_exoplanets.opacity import (
    EvaluatedOpacity,
    OpacitySamplingProvider,
    PreparedOpacitySampling,
    pressure_values_in_unit,
)

from .random_overlap import (
    fused_random_overlap_backend_name,
    fused_random_overlap_kcoeff,
    random_overlap_species_tau,
)

ATOMIC_MASS_KG = 1.66053906660e-27


@dataclass(frozen=True)
class GasOpticalDepth:
    """Gas optical depth on layer, spectral, and opacity quadrature/sample axes.

    The native `total_tau` array has shape `(layer, wavelength, g_or_sample)`
    and `species_tau` has shape
    `(species, layer, wavelength, g_or_sample)`. Opacity sampling uses a
    singleton compatibility axis because each wavelength is a physical sample.
    """

    atmosphere: AtmosphereState
    opacity: EvaluatedOpacity
    species: tuple[str, ...]
    gravity_m_s2: ArrayLike
    layer_pressure_thickness_pa: ArrayLike
    layer_column_density_molecules_m2: ArrayLike
    species_column_density_molecules_m2: ArrayLike
    species_tau: ArrayLike | None
    total_tau: ArrayLike
    unit: str = "dimensionless"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = tuple(str(item) for item in self.species)
        if not species or any(not item for item in species):
            raise RobertValidationError("gas optical-depth species must be non-empty")
        if species != self.opacity.prepared.species:
            raise RobertValidationError("gas optical-depth species must match evaluated opacity")

        n_species = len(species)
        n_layers = self.atmosphere.n_layers
        n_spectral = self.opacity.prepared.spectral_grid.size
        n_g = self.opacity.prepared.g_weights.size

        gravity = _readonly_1d(self.gravity_m_s2, "gravity_m_s2")
        layer_delta_p = _readonly_1d(
            self.layer_pressure_thickness_pa,
            "layer_pressure_thickness_pa",
        )
        layer_column = _readonly_1d(
            self.layer_column_density_molecules_m2,
            "layer_column_density_molecules_m2",
        )
        species_column = _readonly_array(
            self.species_column_density_molecules_m2,
            "species_column_density_molecules_m2",
            (n_species, n_layers),
        )
        species_tau = None
        if self.species_tau is not None:
            species_tau = _readonly_array(
                self.species_tau,
                "species_tau",
                (n_species, n_layers, n_spectral, n_g),
            )
        total_tau = _readonly_array(
            self.total_tau,
            "total_tau",
            (n_layers, n_spectral, n_g),
        )

        if gravity.shape != (n_layers,):
            raise RobertValidationError("gravity_m_s2 must match pressure grid layers")
        if layer_delta_p.shape != (n_layers,):
            raise RobertValidationError("layer_pressure_thickness_pa must match pressure grid layers")
        if layer_column.shape != (n_layers,):
            raise RobertValidationError("layer_column_density_molecules_m2 must match pressure grid layers")
        if np.any(gravity <= 0.0):
            raise RobertValidationError("gravity_m_s2 values must be positive")
        if np.any(layer_delta_p <= 0.0):
            raise RobertValidationError("layer pressure thicknesses must be positive")
        if np.any(layer_column <= 0.0):
            raise RobertValidationError("layer column densities must be positive")
        if np.any(species_column < 0.0):
            raise RobertValidationError("species column densities must be non-negative")
        if (species_tau is not None and np.any(species_tau < 0.0)) or np.any(total_tau < 0.0):
            raise RobertValidationError("gas optical depths must be non-negative")
        metadata = dict(self.metadata)
        gas_combination = metadata.get("gas_combination", "sum_by_g")
        if gas_combination not in {"sum_by_g", "random_overlap"}:
            raise RobertValidationError("gas_combination must be 'sum_by_g' or 'random_overlap'")
        if species_tau is not None and gas_combination == "sum_by_g" and not np.allclose(
            np.sum(species_tau, axis=0),
            total_tau,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise RobertValidationError("total_tau must equal the sum of species_tau")
        if not self.unit:
            raise RobertValidationError("gas optical-depth unit must not be empty")

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "layer_pressure_thickness_pa", layer_delta_p)
        object.__setattr__(self, "layer_column_density_molecules_m2", layer_column)
        object.__setattr__(self, "species_column_density_molecules_m2", species_column)
        object.__setattr__(self, "species_tau", species_tau)
        object.__setattr__(self, "total_tau", total_tau)
        object.__setattr__(self, "metadata", metadata)

    @property
    def pressure_grid(self) -> PressureGrid:
        """Atmospheric pressure grid."""

        return self.atmosphere.pressure_grid

    @property
    def spectral_grid(self) -> SpectralGrid:
        """Spectral grid used by the evaluated opacity."""

        return self.opacity.prepared.spectral_grid

    @property
    def g_weights(self) -> NDArray[np.float64]:
        """Correlated-k quadrature weights."""

        return self.opacity.prepared.g_weights

    def cumulative_tau_from_top(self) -> NDArray[np.float64]:
        """Return gas optical depth from the top of atmosphere through each layer.

        The returned array has the same layer order as `total_tau` even when the
        pressure grid is stored from high to low pressure.
        """

        order = _top_to_bottom_order(self.pressure_grid)
        cumulative = np.cumsum(self.total_tau[order], axis=0)
        return _restore_layer_order(cumulative, order)

    def tau_above_layer(self) -> NDArray[np.float64]:
        """Return gas optical depth above each layer, excluding that layer."""

        order = _top_to_bottom_order(self.pressure_grid)
        sorted_tau = self.total_tau[order]
        above = np.zeros_like(sorted_tau)
        if sorted_tau.shape[0] > 1:
            above[1:] = np.cumsum(sorted_tau[:-1], axis=0)
        return _restore_layer_order(above, order)

    def g_weighted_layer_tau(self) -> NDArray[np.float64]:
        """Return g-weighted mean layer optical depth, shape `(layer, wavelength)`."""

        return _readonly_result(np.sum(self.total_tau * self.g_weights[None, None, :], axis=-1))

    def g_weighted_cumulative_tau_from_top(self) -> NDArray[np.float64]:
        """Return g-weighted mean cumulative optical depth from the top."""

        cumulative = self.cumulative_tau_from_top()
        return _readonly_result(np.sum(cumulative * self.g_weights[None, None, :], axis=-1))

    def band_transmission_to_space(self) -> NDArray[np.float64]:
        """Return g-weighted transmission from the top through each layer."""

        cumulative = self.cumulative_tau_from_top()
        transmission = np.sum(
            np.exp(-cumulative) * self.g_weights[None, None, :],
            axis=-1,
        )
        return _readonly_result(transmission)

    def layer_transmission_weighting(
        self,
        *,
        integrate_g: bool = True,
        normalize: bool = False,
    ) -> NDArray[np.float64]:
        """Return a layer escape-weighting diagnostic derived from optical depth.

        This is a pre-RT diagnostic: it measures
        `exp(-tau_above) * (1 - exp(-tau_layer))`. For true thermal-emission
        contribution functions, a later emission solver will multiply this
        escape weighting by the layer source function.
        """

        tau_above = self.tau_above_layer()
        weighting = np.exp(-tau_above) * (-np.expm1(-self.total_tau))
        if integrate_g:
            weighting = np.sum(weighting * self.g_weights[None, None, :], axis=-1)
        if normalize:
            weighting = _normalize_over_layers(weighting)
        return _readonly_result(weighting)


def assemble_gas_optical_depth(
    atmosphere: AtmosphereState,
    opacity: EvaluatedOpacity,
    *,
    gravity_m_s2: float | ArrayLike,
    gas_combination: str = "sum_by_g",
    retain_species_tau: bool = True,
) -> GasOpticalDepth:
    """Assemble gas optical depth from evaluated correlated-k coefficients.

    ROBERT currently assumes hydrostatic, plane-parallel layers and VMR
    composition. The layer column density is:

    `N = delta_pressure / (mean_molecular_weight * atomic_mass * gravity)`.
    """

    _validate_pressure_grid_match(atmosphere.pressure_grid, opacity.prepared.pressure_grid)
    _validate_composition_convention(atmosphere.composition_convention)
    _validate_mean_molecular_weight_unit(atmosphere.mean_molecular_weight_unit)

    species = opacity.prepared.species
    missing = tuple(item for item in species if item not in atmosphere.composition)
    if missing:
        raise RobertValidationError(f"atmosphere is missing opacity species: {', '.join(missing)}")

    gravity = _gravity_profile(gravity_m_s2, atmosphere.n_layers)
    layer_delta_p_pa = _pressure_layer_thickness_pa(atmosphere.pressure_grid)
    particle_mass_kg = atmosphere.mean_molecular_weight * ATOMIC_MASS_KG
    layer_column_density = layer_delta_p_pa / (particle_mass_kg * gravity)

    vmr = np.stack([atmosphere.composition[item] for item in species], axis=0)
    species_column_density = vmr * layer_column_density[None, :]
    kcoeff_m2_per_molecule = _kcoeff_m2_per_molecule(opacity.kcoeff, opacity.unit)
    requested_combination = _gas_combination_mode(gas_combination)
    opacity_mode = str(opacity.prepared.metadata.get("opacity_mode", "correlated_k"))
    combination = (
        "sum_by_g" if opacity_mode == "opacity_sampling" else requested_combination
    )
    if retain_species_tau:
        kcoeff_m2_per_molecule = _kcoeff_m2_per_molecule(
            opacity.kcoeff, opacity.unit
        )
        species_tau = (
            kcoeff_m2_per_molecule
            * species_column_density[:, :, None, None]
        )
        if combination == "sum_by_g":
            total_tau = np.sum(species_tau, axis=0)
        else:
            total_tau = random_overlap_species_tau(
                species_tau, opacity.prepared.g_weights
            )
    else:
        species_tau = None
        unit_scale = _opacity_unit_scale_m2(opacity.unit)
        if combination == "random_overlap":
            total_tau = fused_random_overlap_kcoeff(
                opacity.kcoeff,
                species_column_density,
                opacity.prepared.g_weights,
                unit_scale_m2=unit_scale,
            )
        else:
            total_tau = np.einsum(
                "slwg,sl->lwg",
                opacity.kcoeff,
                species_column_density * unit_scale,
                optimize=True,
            )

    if (
        species_tau is not None and not np.all(np.isfinite(species_tau))
    ) or not np.all(np.isfinite(total_tau)):
        raise RobertValidationError("assembled gas optical depth must be finite")

    return GasOpticalDepth(
        atmosphere=atmosphere,
        opacity=opacity,
        species=species,
        gravity_m_s2=gravity,
        layer_pressure_thickness_pa=layer_delta_p_pa,
        layer_column_density_molecules_m2=layer_column_density,
        species_column_density_molecules_m2=species_column_density,
        species_tau=species_tau,
        total_tau=total_tau,
        metadata={
            "opacity_mode": opacity_mode,
            "opacity_unit": opacity.unit,
            "column_model": "hydrostatic_plane_parallel",
            "gas_combination": combination,
            "requested_gas_combination": requested_combination,
            "species_tau_diagnostics": (
                "enabled" if species_tau is not None else "disabled"
            ),
            "assembly_backend": (
                "species_resolved_reference"
                if species_tau is not None
                else fused_random_overlap_backend_name()
                if combination == "random_overlap"
                else "fused_direct_sum"
            ),
        },
    )


def assemble_opacity_sampling_gas_optical_depth(
    atmosphere: AtmosphereState,
    provider: OpacitySamplingProvider,
    prepared: PreparedOpacitySampling,
    *,
    gravity_m_s2: float | ArrayLike,
) -> GasOpticalDepth:
    """Fuse sampled-opacity interpolation, VMR mixing, and tau assembly."""

    _validate_pressure_grid_match(atmosphere.pressure_grid, prepared.pressure_grid)
    _validate_composition_convention(atmosphere.composition_convention)
    _validate_mean_molecular_weight_unit(atmosphere.mean_molecular_weight_unit)
    missing = tuple(name for name in prepared.species if name not in atmosphere.composition)
    if missing:
        raise RobertValidationError(
            f"atmosphere is missing opacity species: {', '.join(missing)}"
        )
    gravity = _gravity_profile(gravity_m_s2, atmosphere.n_layers)
    layer_delta_p_pa = _pressure_layer_thickness_pa(atmosphere.pressure_grid)
    particle_mass_kg = atmosphere.mean_molecular_weight * ATOMIC_MASS_KG
    layer_column_density = layer_delta_p_pa / (particle_mass_kg * gravity)
    vmr = np.stack(
        [atmosphere.composition[name] for name in prepared.species], axis=0
    )
    species_column_density = vmr * layer_column_density[None, :]
    mixture = provider.evaluate_mixture(atmosphere, prepared)
    cross_section_m2 = _kcoeff_m2_per_molecule(
        mixture.cross_section, mixture.unit
    )
    total_tau = cross_section_m2 * layer_column_density[:, None]
    total_tau = total_tau[:, :, None]
    if not np.all(np.isfinite(total_tau)):
        raise RobertValidationError("assembled opacity-sampling optical depth must be finite")
    return GasOpticalDepth(
        atmosphere=atmosphere,
        opacity=mixture,
        species=prepared.species,
        gravity_m_s2=gravity,
        layer_pressure_thickness_pa=layer_delta_p_pa,
        layer_column_density_molecules_m2=layer_column_density,
        species_column_density_molecules_m2=species_column_density,
        species_tau=None,
        total_tau=total_tau,
        metadata={
            "opacity_mode": "opacity_sampling",
            "maturity": "beta",
            "opacity_unit": mixture.unit,
            "column_model": "hydrostatic_plane_parallel",
            "gas_combination": "sum_by_g",
            "requested_gas_combination": "fused_direct_sum",
            "species_tau_diagnostics": "disabled",
        },
    )


def _validate_pressure_grid_match(left: PressureGrid, right: PressureGrid) -> None:
    left_edges = pressure_values_in_unit(left.edges, left.unit, "pa")
    right_edges = pressure_values_in_unit(right.edges, right.unit, "pa")
    left_centers = pressure_values_in_unit(left.centers, left.unit, "pa")
    right_centers = pressure_values_in_unit(right.centers, right.unit, "pa")
    if left_edges.shape != right_edges.shape or left_centers.shape != right_centers.shape:
        raise RobertValidationError("atmosphere and opacity pressure grids must match")
    if not np.allclose(left_edges, right_edges, rtol=1.0e-10, atol=0.0) or not np.allclose(
        left_centers,
        right_centers,
        rtol=1.0e-10,
        atol=0.0,
    ):
        raise RobertValidationError("atmosphere and opacity pressure grids must match")


def _validate_composition_convention(convention: str) -> None:
    normalized = convention.strip().lower()
    if normalized not in {"volume_mixing_ratio", "vmr"}:
        raise RobertValidationError("gas optical-depth assembly currently requires VMR composition")


def _validate_mean_molecular_weight_unit(unit: str) -> None:
    normalized = unit.strip().lower().replace(" ", "_")
    if normalized not in {"amu", "atomic_mass_unit", "atomic_mass_units", "u"}:
        raise RobertValidationError("gas optical-depth assembly currently requires MMW in amu")


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


def _pressure_layer_thickness_pa(pressure_grid: PressureGrid) -> NDArray[np.float64]:
    edges = pressure_values_in_unit(pressure_grid.edges, pressure_grid.unit, "pa")
    thickness = np.abs(np.diff(edges))
    if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0.0):
        raise RobertValidationError("pressure layer thicknesses must be finite and positive")
    thickness.setflags(write=False)
    return thickness


def _kcoeff_m2_per_molecule(
    values: NDArray[np.float64],
    unit: str,
) -> NDArray[np.float64]:
    normalized = unit.strip().lower().replace(" ", "")
    kcoeff = np.array(values, dtype=float, copy=True)
    if normalized in {"cm^2/molecule", "cm2/molecule", "cm^2molecule^-1", "cm2molecule-1"}:
        kcoeff *= 1.0e-4
    elif normalized in {"m^2/molecule", "m2/molecule", "m^2molecule^-1", "m2molecule-1"}:
        pass
    else:
        raise RobertValidationError(f"unsupported opacity unit for gas optical depth: {unit}")
    kcoeff.setflags(write=False)
    return kcoeff


def _opacity_unit_scale_m2(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {
        "cm^2/molecule",
        "cm2/molecule",
        "cm^2molecule^-1",
        "cm2molecule-1",
    }:
        return 1.0e-4
    if normalized in {
        "m^2/molecule",
        "m2/molecule",
        "m^2molecule^-1",
        "m2molecule-1",
    }:
        return 1.0
    raise RobertValidationError(
        f"unsupported opacity unit for gas optical depth: {unit}"
    )


def _gas_combination_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"sum_by_g", "sum", "same_g_sum"}:
        return "sum_by_g"
    if normalized in {"random_overlap", "randomoverlap", "noverlap"}:
        return "random_overlap"
    raise RobertValidationError("gas_combination must be 'sum_by_g' or 'random_overlap'")


def _top_to_bottom_order(pressure_grid: PressureGrid) -> NDArray[np.int64]:
    pressure = pressure_values_in_unit(pressure_grid.centers, pressure_grid.unit, "pa")
    return np.argsort(pressure).astype(np.int64)


def _restore_layer_order(
    values_in_top_to_bottom_order: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    restored = np.empty_like(values_in_top_to_bottom_order)
    restored[order] = values_in_top_to_bottom_order
    restored.setflags(write=False)
    return restored


def _normalize_over_layers(values: NDArray[np.float64]) -> NDArray[np.float64]:
    layer_sum = np.sum(values, axis=0, keepdims=True)
    normalized = np.divide(
        values,
        layer_sum,
        out=np.zeros_like(values),
        where=layer_sum > 0.0,
    )
    return normalized


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
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


def _readonly_result(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result
