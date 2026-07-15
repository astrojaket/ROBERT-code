"""Shared parameterized cloud models for emission and transmission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import pressure_values_in_unit
from robert_exoplanets.rt import (
    CloudOpticalProperties,
    GasOpticalDepth,
    RefractiveIndexSpectrum,
    grey_cloud_deck,
    lognormal_mie_optics,
    mie_cloud_from_mass_fraction,
    power_law_haze_from_mass_extinction,
    refractive_index_from_parameters,
)


@runtime_checkable
class ParameterizedCloudModel(Protocol):
    """Geometry-independent cloud optical-property parameterization."""

    @property
    def required_parameters(self) -> tuple[str, ...]:
        """Names of retrieval parameters consumed by the cloud model."""

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        """Serializable description of the cloud parameterization."""

    @property
    def multiple_scattering_backend(self) -> str:
        """Emission multiple-scattering backend requested by the cloud."""

    def evaluate(
        self,
        gas_optical_depth: GasOpticalDepth,
        parameters: Mapping[str, float],
    ) -> tuple[CloudOpticalProperties, ...]:
        """Evaluate cloud properties on an atmosphere and spectral grid."""


@dataclass(frozen=True)
class ParameterizedDeckHazeCloudModel:
    """Shared finite grey-deck plus well-mixed power-law haze model.

    The grey deck is specified by its top pressure and integrated vertical
    extinction optical depth. The haze is specified by a bulk-atmosphere mass
    extinction coefficient at a reference wavelength and a wavelength power
    law. Both components return the same optical-property objects to emission
    and transmission; only the downstream radiative-transfer geometry differs.
    """

    log10_cloud_top_pressure_bar_parameter: str = "log_cloud_top_pressure_bar"
    log10_cloud_optical_depth_parameter: str = "log_cloud_optical_depth"
    log10_haze_mass_extinction_parameter: str = "log_haze_mass_extinction"
    haze_slope_parameter: str = "haze_slope"
    haze_reference_wavelength_micron: float = 1.0
    deck_single_scattering_albedo: float = 0.0
    deck_asymmetry_factor: float = 0.0
    haze_single_scattering_albedo: float = 1.0
    haze_asymmetry_factor: float = 0.0
    multiple_scattering_backend: str = "sh4"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameter_fields = (
            "log10_cloud_top_pressure_bar_parameter",
            "log10_cloud_optical_depth_parameter",
            "log10_haze_mass_extinction_parameter",
            "haze_slope_parameter",
        )
        parameters = tuple(str(getattr(self, name)).strip() for name in parameter_fields)
        if any(not parameter for parameter in parameters):
            raise RobertValidationError("cloud parameter names must not be empty")
        if len(set(parameters)) != len(parameters):
            raise RobertValidationError("cloud parameter names must be unique")

        reference_wavelength = float(self.haze_reference_wavelength_micron)
        if not np.isfinite(reference_wavelength) or reference_wavelength <= 0.0:
            raise RobertValidationError(
                "haze_reference_wavelength_micron must be finite and positive"
            )
        for field_name in (
            "deck_single_scattering_albedo",
            "haze_single_scattering_albedo",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RobertValidationError(f"{field_name} must lie in [0, 1]")
            object.__setattr__(self, field_name, value)
        for field_name in ("deck_asymmetry_factor", "haze_asymmetry_factor"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise RobertValidationError(f"{field_name} must lie in [-1, 1]")
            object.__setattr__(self, field_name, value)

        backend = self.multiple_scattering_backend.strip().lower()
        if backend not in {"none", "two_stream", "toon_hemispheric_mean", "sh4", "p3"}:
            raise RobertValidationError(
                "multiple_scattering_backend must be 'none', 'two_stream', or 'sh4'"
            )
        for field_name, value in zip(parameter_fields, parameters, strict=True):
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "haze_reference_wavelength_micron", reference_wavelength)
        object.__setattr__(self, "multiple_scattering_backend", backend)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (
            self.log10_cloud_top_pressure_bar_parameter,
            self.log10_cloud_optical_depth_parameter,
            self.log10_haze_mass_extinction_parameter,
            self.haze_slope_parameter,
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                "cloud_model": "grey_deck_power_law_haze",
                "cloud_geometry_independent": "true",
                "cloud_log10_top_pressure_parameter": (
                    self.log10_cloud_top_pressure_bar_parameter
                ),
                "cloud_log10_optical_depth_parameter": (
                    self.log10_cloud_optical_depth_parameter
                ),
                "haze_log10_mass_extinction_parameter": (
                    self.log10_haze_mass_extinction_parameter
                ),
                "haze_slope_parameter": self.haze_slope_parameter,
                "haze_reference_wavelength_micron": (
                    f"{self.haze_reference_wavelength_micron:.17g}"
                ),
                "deck_single_scattering_albedo": (
                    f"{self.deck_single_scattering_albedo:.17g}"
                ),
                "deck_asymmetry_factor": f"{self.deck_asymmetry_factor:.17g}",
                "haze_single_scattering_albedo": (
                    f"{self.haze_single_scattering_albedo:.17g}"
                ),
                "haze_asymmetry_factor": f"{self.haze_asymmetry_factor:.17g}",
                "cloud_multiple_scattering_backend": self.multiple_scattering_backend,
                **dict(self.metadata),
            }
        )

    def evaluate(
        self,
        gas_optical_depth: GasOpticalDepth,
        parameters: Mapping[str, float],
    ) -> tuple[CloudOpticalProperties, CloudOpticalProperties]:
        """Evaluate deck and haze optical properties on the prepared RT grid."""

        values = {}
        for name in self.required_parameters:
            if name not in parameters:
                raise RobertValidationError(f"cloud parameter is missing: {name}")
            value = float(parameters[name])
            if not np.isfinite(value):
                raise RobertValidationError(f"cloud parameter {name!r} must be finite")
            values[name] = value

        cloud_top_pressure_bar = 10.0 ** values[
            self.log10_cloud_top_pressure_bar_parameter
        ]
        cloud_optical_depth = 10.0 ** values[
            self.log10_cloud_optical_depth_parameter
        ]
        haze_mass_extinction = 10.0 ** values[
            self.log10_haze_mass_extinction_parameter
        ]
        haze_slope = values[self.haze_slope_parameter]
        if not all(
            np.isfinite(value)
            for value in (
                cloud_top_pressure_bar,
                cloud_optical_depth,
                haze_mass_extinction,
            )
        ):
            raise RobertValidationError("log10 cloud parameters overflowed")

        deck = grey_cloud_deck(
            gas_optical_depth.pressure_grid,
            gas_optical_depth.spectral_grid,
            cloud_top_pressure=cloud_top_pressure_bar,
            cloud_top_pressure_unit="bar",
            optical_depth=cloud_optical_depth,
            single_scattering_albedo=self.deck_single_scattering_albedo,
            asymmetry_factor=self.deck_asymmetry_factor,
        )
        haze = power_law_haze_from_mass_extinction(
            gas_optical_depth,
            mass_extinction_at_reference_cm2_g=haze_mass_extinction,
            reference_wavelength_micron=self.haze_reference_wavelength_micron,
            slope=haze_slope,
            single_scattering_albedo=self.haze_single_scattering_albedo,
            asymmetry_factor=self.haze_asymmetry_factor,
        )
        return deck, haze


@dataclass(frozen=True)
class ParameterizedMieCloudModel:
    """Shared lognormal homogeneous-sphere Mie particle cloud.

    The refractive index can be fixed from a material catalogue or retrieved
    as nodal real ``n`` and log10 imaginary ``k`` values. Evaluation returns
    geometry-independent layer optical properties consumed unchanged by
    emission and transmission.
    """

    refractive_index_wavelength_micron: tuple[float, ...]
    real_index_parameter_names: tuple[str, ...]
    log10_imaginary_index_parameter_names: tuple[str, ...]
    log10_condensate_mass_fraction_parameter: str
    log10_effective_radius_micron_parameter: str
    particle_density_kg_m3: float
    geometric_stddev: float = 1.0
    geometric_stddev_parameter: str | None = None
    log10_cloud_top_pressure_bar_parameter: str | None = None
    log10_cloud_base_pressure_bar_parameter: str | None = None
    quadrature_points: int = 1
    refractive_index_extrapolation: str = "raise"
    multiple_scattering_backend: str = "sh4"
    fixed_refractive_index: RefractiveIndexSpectrum | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = tuple(float(value) for value in self.refractive_index_wavelength_micron)
        real_names = tuple(str(value).strip() for value in self.real_index_parameter_names)
        imaginary_names = tuple(
            str(value).strip() for value in self.log10_imaginary_index_parameter_names
        )
        if self.fixed_refractive_index is None:
            if not wavelength or any(not np.isfinite(value) or value <= 0.0 for value in wavelength):
                raise RobertValidationError(
                    "refractive-index wavelength nodes must be finite and positive"
                )
            if any(right <= left for left, right in zip(wavelength[:-1], wavelength[1:], strict=True)):
                raise RobertValidationError(
                    "refractive-index wavelength nodes must be strictly increasing"
                )
            if len(real_names) != len(wavelength) or len(imaginary_names) != len(wavelength):
                raise RobertValidationError(
                    "refractive-index parameter names must match wavelength nodes"
                )
        elif wavelength or real_names or imaginary_names:
            raise RobertValidationError(
                "fixed_refractive_index cannot be combined with retrieved n/k nodes"
            )
        scalar_names = (
            str(self.log10_condensate_mass_fraction_parameter).strip(),
            str(self.log10_effective_radius_micron_parameter).strip(),
        )
        optional_names = tuple(
            None if value is None else str(value).strip()
            for value in (
                self.geometric_stddev_parameter,
                self.log10_cloud_top_pressure_bar_parameter,
                self.log10_cloud_base_pressure_bar_parameter,
            )
        )
        names = (
            *real_names,
            *imaginary_names,
            *scalar_names,
            *(value for value in optional_names if value is not None),
        )
        if any(not name for name in names) or len(names) != len(set(names)):
            raise RobertValidationError(
                "cloud retrieval parameter names must be non-empty and unique"
            )
        density = float(self.particle_density_kg_m3)
        width = float(self.geometric_stddev)
        points = int(self.quadrature_points)
        extrapolation = str(self.refractive_index_extrapolation).strip().lower()
        backend = str(self.multiple_scattering_backend).strip().lower()
        if not np.isfinite(density) or density <= 0.0:
            raise RobertValidationError("particle_density_kg_m3 must be finite and positive")
        if not np.isfinite(width) or width < 1.0:
            raise RobertValidationError("geometric_stddev must be finite and at least one")
        if points < 1:
            raise RobertValidationError("quadrature_points must be positive")
        if extrapolation not in {"raise", "clip"}:
            raise RobertValidationError(
                "refractive_index_extrapolation must be 'raise' or 'clip'"
            )
        if backend not in {"two_stream", "toon_hemispheric_mean", "sh4", "p3"}:
            raise RobertValidationError("Mie cloud requires a multiple-scattering backend")
        object.__setattr__(self, "refractive_index_wavelength_micron", wavelength)
        object.__setattr__(self, "real_index_parameter_names", real_names)
        object.__setattr__(self, "log10_imaginary_index_parameter_names", imaginary_names)
        object.__setattr__(self, "log10_condensate_mass_fraction_parameter", scalar_names[0])
        object.__setattr__(self, "log10_effective_radius_micron_parameter", scalar_names[1])
        object.__setattr__(self, "geometric_stddev_parameter", optional_names[0])
        object.__setattr__(self, "log10_cloud_top_pressure_bar_parameter", optional_names[1])
        object.__setattr__(self, "log10_cloud_base_pressure_bar_parameter", optional_names[2])
        object.__setattr__(self, "particle_density_kg_m3", density)
        object.__setattr__(self, "geometric_stddev", width)
        object.__setattr__(self, "quadrature_points", points)
        object.__setattr__(self, "refractive_index_extrapolation", extrapolation)
        object.__setattr__(self, "multiple_scattering_backend", backend)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        parameters = [
            self.log10_condensate_mass_fraction_parameter,
            self.log10_effective_radius_micron_parameter,
        ]
        if self.fixed_refractive_index is None:
            parameters[:0] = [
                *self.real_index_parameter_names,
                *self.log10_imaginary_index_parameter_names,
            ]
        parameters.extend(
            value
            for value in (
                self.geometric_stddev_parameter,
                self.log10_cloud_top_pressure_bar_parameter,
                self.log10_cloud_base_pressure_bar_parameter,
            )
            if value is not None
        )
        return tuple(parameters)

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                "cloud_model": "lognormal_homogeneous_sphere_mie",
                "cloud_geometry_independent": "true",
                "cloud_refractive_index_mode": (
                    "fixed_tabulated_n_k"
                    if self.fixed_refractive_index is not None
                    else "retrieved_nodal_n_log10_k"
                ),
                "cloud_particle_density_kg_m3": f"{self.particle_density_kg_m3:.17g}",
                "cloud_geometric_stddev": f"{self.geometric_stddev:.17g}",
                "cloud_quadrature_points": str(self.quadrature_points),
                "cloud_multiple_scattering_backend": self.multiple_scattering_backend,
                "cloud_phase_function_closure": "exact_mie_legendre_moments_through_l4",
                **dict(self.metadata),
            }
        )

    def evaluate(
        self,
        gas_optical_depth: GasOpticalDepth,
        parameters: Mapping[str, float],
    ) -> tuple[CloudOpticalProperties]:
        values = {}
        for name in self.required_parameters:
            if name not in parameters:
                raise RobertValidationError(f"cloud parameter is missing: {name}")
            value = float(parameters[name])
            if not np.isfinite(value):
                raise RobertValidationError(f"cloud parameter {name!r} must be finite")
            values[name] = value
        index = self.fixed_refractive_index
        if index is None:
            index = refractive_index_from_parameters(
                self.refractive_index_wavelength_micron,
                values,
                real_parameter_names=self.real_index_parameter_names,
                log10_imaginary_parameter_names=self.log10_imaginary_index_parameter_names,
            )
        particle_optics = lognormal_mie_optics(
            index,
            gas_optical_depth.spectral_grid,
            effective_radius_micron=(
                10.0 ** values[self.log10_effective_radius_micron_parameter]
            ),
            geometric_stddev=(
                self.geometric_stddev
                if self.geometric_stddev_parameter is None
                else values[self.geometric_stddev_parameter]
            ),
            particle_density_kg_m3=self.particle_density_kg_m3,
            quadrature_points=self.quadrature_points,
            extrapolation=self.refractive_index_extrapolation,
        )
        pressure_bar = pressure_values_in_unit(
            gas_optical_depth.pressure_grid.centers,
            gas_optical_depth.pressure_grid.unit,
            "bar",
        )
        active = np.ones(gas_optical_depth.pressure_grid.n_layers, dtype=bool)
        top_pressure = None
        base_pressure = None
        if self.log10_cloud_top_pressure_bar_parameter is not None:
            top_pressure = 10.0 ** values[self.log10_cloud_top_pressure_bar_parameter]
            active &= pressure_bar >= top_pressure
        if self.log10_cloud_base_pressure_bar_parameter is not None:
            base_pressure = 10.0 ** values[self.log10_cloud_base_pressure_bar_parameter]
            active &= pressure_bar <= base_pressure
        if top_pressure is not None and base_pressure is not None and top_pressure > base_pressure:
            raise RobertValidationError("cloud top pressure must not exceed cloud base pressure")
        mass_fraction = np.where(
            active,
            10.0 ** values[self.log10_condensate_mass_fraction_parameter],
            0.0,
        )
        return (
            mie_cloud_from_mass_fraction(
                gas_optical_depth,
                particle_optics,
                condensate_mass_fraction=mass_fraction,
            ),
        )


__all__ = [
    "ParameterizedCloudModel",
    "ParameterizedDeckHazeCloudModel",
    "ParameterizedMieCloudModel",
]
