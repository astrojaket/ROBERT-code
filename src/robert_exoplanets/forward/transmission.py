"""Parameterized one-dimensional transmission forward model."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from robert_exoplanets.atmosphere import AtmosphereBuilder, AtmosphereState
from robert_exoplanets.bodies import Planet, Star
from robert_exoplanets.core import (
    PressureGrid,
    RobertValidationError,
    SpectralGrid,
    Spectrum,
)
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import (
    OpacityProvider,
    PreparedOpacity,
    pressure_values_in_unit,
)
from robert_exoplanets.rt import (
    AbsorptionTransmissionResult,
    CiaTable,
    hydrostatic_path_geometry,
    inverse_square_hydrostatic_path_geometry,
    solve_absorption_transmission,
)

from ._atmospheric import (
    evaluate_additional_optical_depths,
    evaluate_gas_optical_depth,
)
from .clouds import ParameterizedCloudModel

GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11


@dataclass(frozen=True)
class ParameterizedTransmissionModelConfig:
    """Physical and numerical choices for a parameterized transmission model.

    ``planet.radius_m`` is interpreted as the reference radius at
    ``reference_pressure_bar``. An optional radius-scale parameter retrieves a
    multiplicative correction to that anchor. ``inverse_square`` is a
    constrained variable-gravity model; ``constant`` retains one gravity value
    throughout the atmosphere.
    """

    opacity_species: tuple[str, ...]
    reference_pressure_bar: float
    radius_scale_parameter: str | None = None
    gravity_model: str = "inverse_square"
    include_rayleigh: bool = True
    cia_normal_hydrogen: bool = True
    cia_temperature_extrapolation: str = "clip"
    cia_spectral_extrapolation: str = "zero"
    gas_combination: str = "random_overlap"
    impact_quadrature_order: int = 8
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = tuple(str(item).strip() for item in self.opacity_species)
        if not species or any(not item for item in species):
            raise RobertValidationError(
                "opacity_species must contain non-empty names"
            )
        if len(set(species)) != len(species):
            raise RobertValidationError("opacity_species must not contain duplicates")
        reference_pressure = float(self.reference_pressure_bar)
        if not np.isfinite(reference_pressure) or reference_pressure <= 0.0:
            raise RobertValidationError(
                "reference_pressure_bar must be finite and positive"
            )
        radius_parameter = self.radius_scale_parameter
        if radius_parameter is not None:
            radius_parameter = str(radius_parameter).strip()
            if not radius_parameter:
                raise RobertValidationError(
                    "radius_scale_parameter must be non-empty when provided"
                )
        gravity_model = self.gravity_model.strip().lower().replace("-", "_")
        if gravity_model not in {"constant", "inverse_square"}:
            raise RobertValidationError(
                "gravity_model must be 'constant' or 'inverse_square'"
            )
        if self.gas_combination not in {"sum_by_g", "random_overlap"}:
            raise RobertValidationError(
                "gas_combination must be 'sum_by_g' or 'random_overlap'"
            )
        if self.cia_temperature_extrapolation not in {"raise", "clip"}:
            raise RobertValidationError(
                "CIA temperature extrapolation must be 'raise' or 'clip'"
            )
        if self.cia_spectral_extrapolation not in {"raise", "zero"}:
            raise RobertValidationError(
                "CIA spectral extrapolation must be 'raise' or 'zero'"
            )
        if (
            isinstance(self.impact_quadrature_order, bool)
            or int(self.impact_quadrature_order) != self.impact_quadrature_order
            or int(self.impact_quadrature_order) < 2
        ):
            raise RobertValidationError(
                "impact_quadrature_order must be an integer of at least two"
            )
        object.__setattr__(self, "opacity_species", species)
        object.__setattr__(self, "reference_pressure_bar", reference_pressure)
        object.__setattr__(self, "radius_scale_parameter", radius_parameter)
        object.__setattr__(self, "gravity_model", gravity_model)
        object.__setattr__(
            self,
            "impact_quadrature_order",
            int(self.impact_quadrature_order),
        )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ParameterizedTransmissionForwardModel:
    """Transmission model with runtime temperature and chemistry profiles.

    The model prepares opacity once. Each call builds the atmospheric state,
    constructs a radius-pressure geometry, evaluates gas/CIA/Rayleigh optical
    depth, and returns a transit-depth spectrum.
    """

    planet: Planet
    star: Star
    spectral_grid: SpectralGrid
    atmosphere_builder: AtmosphereBuilder
    opacity_provider: OpacityProvider
    config: ParameterizedTransmissionModelConfig
    cia_table: CiaTable | tuple[CiaTable, ...] | None = None
    cloud_model: ParameterizedCloudModel | None = None
    prepared_opacity: PreparedOpacity = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.planet.radius_m is None:
            raise RobertValidationError(
                "parameterized transmission requires planet.radius_m"
            )
        if self.star.radius_m is None:
            raise RobertValidationError(
                "parameterized transmission requires star.radius_m"
            )
        pressure_edges_bar = pressure_values_in_unit(
            self.pressure_grid.edges,
            self.pressure_grid.unit,
            "bar",
        )
        if not (
            float(np.min(pressure_edges_bar))
            <= self.config.reference_pressure_bar
            <= float(np.max(pressure_edges_bar))
        ):
            raise RobertValidationError(
                "transmission reference pressure must lie within the pressure grid"
            )
        missing_opacity = tuple(
            species
            for species in self.config.opacity_species
            if species not in self.opacity_provider.tables
        )
        if missing_opacity:
            raise RobertValidationError(
                "opacity provider is missing configured species: "
                + ", ".join(missing_opacity)
            )
        missing_chemistry = tuple(
            species
            for species in self.config.opacity_species
            if species not in self.atmosphere_builder.species
        )
        if missing_chemistry:
            raise RobertValidationError(
                "chemistry model is missing opacity species: "
                + ", ".join(missing_chemistry)
            )
        if len(set(self.required_parameters)) != len(self.required_parameters):
            raise RobertValidationError(
                "temperature, chemistry, and radius parameter names must be unique"
            )
        prepared = self.opacity_provider.prepare(
            self.spectral_grid,
            self.pressure_grid,
            species=self.config.opacity_species,
        )
        object.__setattr__(self, "prepared_opacity", prepared)

    @property
    def pressure_grid(self) -> PressureGrid:
        return self.atmosphere_builder.pressure_grid

    @property
    def cia_tables(self) -> tuple[CiaTable, ...]:
        if self.cia_table is None:
            return ()
        if isinstance(self.cia_table, CiaTable):
            return (self.cia_table,)
        return tuple(self.cia_table)

    @property
    def required_parameters(self) -> tuple[str, ...]:
        parameters = [
            *self.atmosphere_builder.temperature_profile.required_parameters(),
            *self.atmosphere_builder.chemistry_model.required_parameters(),
        ]
        if self.atmosphere_builder.mean_molecular_weight_model is not None:
            parameters.extend(
                self.atmosphere_builder.mean_molecular_weight_model.required_parameters()
            )
        if self.config.radius_scale_parameter is not None:
            parameters.append(self.config.radius_scale_parameter)
        if self.cloud_model is not None:
            parameters.extend(self.cloud_model.required_parameters)
        return tuple(parameters)

    @property
    def opacity_identifiers(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                species: str(
                    self.opacity_provider.tables[species].metadata.get(
                        "checksum_sha256",
                        "",
                    )
                )
                for species in self.config.opacity_species
            }
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                "forward_model": "parameterized_transmission",
                "planet_name": self.planet.name,
                "planet_reference_radius_m": f"{self.planet.radius_m:.17g}",
                "reference_pressure_bar": (
                    f"{self.config.reference_pressure_bar:.17g}"
                ),
                "gravity_model": self.config.gravity_model,
                "gravity_source": (
                    "planet_mass"
                    if self.planet.mass_kg is not None
                    else "planet_reference_gravity"
                ),
                "star_name": self.star.name,
                "star_radius_m": f"{self.star.radius_m:.17g}",
                "opacity_species": ",".join(self.config.opacity_species),
                "opacity_provider": self.opacity_provider.name,
                "opacity_cache_key": self.prepared_opacity.cache_key,
                "temperature_profile": (
                    self.atmosphere_builder.temperature_profile.name
                ),
                "chemistry_model": self.atmosphere_builder.chemistry_model.name,
                "required_parameters": ",".join(self.required_parameters),
                "gas_combination": self.config.gas_combination,
                "include_rayleigh": str(self.config.include_rayleigh).lower(),
                "include_cia": str(bool(self.cia_tables)).lower(),
                "impact_quadrature_order": str(
                    self.config.impact_quadrature_order
                ),
                "radius_scale_parameter": (
                    self.config.radius_scale_parameter or ""
                ),
                "pressure_grid_layers": str(self.pressure_grid.n_layers),
                "pressure_grid_sha256": _array_signature(
                    self.pressure_grid.edges,
                    self.pressure_grid.centers,
                    labels=(self.pressure_grid.unit,),
                ),
                "spectral_grid_sha256": _array_signature(
                    self.spectral_grid.values,
                    *(
                        ()
                        if self.spectral_grid.bin_edges is None
                        else (self.spectral_grid.bin_edges,)
                    ),
                    labels=(self.spectral_grid.unit,),
                ),
                **(
                    {}
                    if self.cloud_model is None
                    else dict(self.cloud_model.manifest_metadata)
                ),
                **dict(self.config.metadata),
            }
        )

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        return self.evaluate_result(parameters).transit_depth

    def evaluate_result(
        self,
        parameters: Mapping[str, float],
    ) -> AbsorptionTransmissionResult:
        """Return transit depth together with annulus diagnostics."""

        parameter_values = self.validated_parameters(parameters)
        atmosphere = self.atmosphere_builder.build(parameter_values)
        return self.evaluate_atmosphere(atmosphere, parameter_values)

    def validated_parameters(
        self,
        parameters: Mapping[str, float],
    ) -> Mapping[str, float]:
        missing = tuple(
            name for name in self.required_parameters if name not in parameters
        )
        if missing:
            raise RobertValidationError(
                "parameterized transmission parameters are missing: "
                + ", ".join(missing)
            )
        values = {name: float(parameters[name]) for name in self.required_parameters}
        if not all(np.isfinite(value) for value in values.values()):
            raise RobertValidationError(
                "parameterized transmission parameters must be finite"
            )
        return immutable_mapping(values)

    def evaluate_atmosphere(
        self,
        atmosphere: AtmosphereState,
        parameters: Mapping[str, float],
    ) -> AbsorptionTransmissionResult:
        """Evaluate a precomputed atmosphere through transmission RT."""

        parameter_values = self.validated_parameters(parameters)
        _validate_pressure_grid(atmosphere.pressure_grid, self.pressure_grid)
        radius_scale = (
            1.0
            if self.config.radius_scale_parameter is None
            else parameter_values[self.config.radius_scale_parameter]
        )
        if radius_scale <= 0.0:
            raise RobertValidationError("radius scale must be positive")
        reference_radius = float(self.planet.radius_m) * radius_scale
        reference_gravity = self._reference_gravity(reference_radius)
        if self.config.gravity_model == "inverse_square":
            path_geometry = inverse_square_hydrostatic_path_geometry(
                atmosphere,
                reference_radius_m=reference_radius,
                reference_pressure=self.config.reference_pressure_bar,
                reference_gravity_m_s2=reference_gravity,
                reference_pressure_unit="bar",
            )
        else:
            path_geometry = hydrostatic_path_geometry(
                atmosphere,
                gravity_m_s2=reference_gravity,
                reference_radius_m=reference_radius,
                reference_pressure=self.config.reference_pressure_bar,
                reference_pressure_unit="bar",
            )
        gas_optical_depth = evaluate_gas_optical_depth(
            self.opacity_provider,
            self.prepared_opacity,
            atmosphere,
            gravity_m_s2=path_geometry.gravity_m_s2,
            gas_combination=self.config.gas_combination,
            retain_species_tau=False,
        )
        additional_optical_depths = evaluate_additional_optical_depths(
            gas_optical_depth,
            cia_tables=self.cia_tables,
            include_rayleigh=self.config.include_rayleigh,
            cia_normal_hydrogen=self.config.cia_normal_hydrogen,
            cia_temperature_extrapolation=self.config.cia_temperature_extrapolation,
            cia_spectral_extrapolation=self.config.cia_spectral_extrapolation,
            cloud_model=self.cloud_model,
            parameters=parameter_values,
        )
        result = solve_absorption_transmission(
            gas_optical_depth,
            path_geometry,
            star_radius_m=float(self.star.radius_m),
            additional_optical_depths=additional_optical_depths,
            impact_quadrature_order=self.config.impact_quadrature_order,
        )
        forward_metadata = {
            **dict(result.metadata),
            "forward_model": "parameterized_transmission",
            "gravity_model": self.config.gravity_model,
            "reference_pressure_bar": (
                f"{self.config.reference_pressure_bar:.17g}"
            ),
            "reference_gravity_m_s2": f"{reference_gravity:.17g}",
        }
        transit_depth = Spectrum(
            spectral_grid=result.transit_depth.spectral_grid,
            values=result.transit_depth.values,
            unit=result.transit_depth.unit,
            observable=result.transit_depth.observable,
            metadata=forward_metadata,
        )
        return AbsorptionTransmissionResult(
            transit_depth=transit_depth,
            effective_radius_m=result.effective_radius_m,
            impact_radius_edges_m=result.impact_radius_edges_m,
            annulus_area_contribution_m2=result.annulus_area_contribution_m2,
            path_geometry=result.path_geometry,
            metadata=forward_metadata,
        )

    def _reference_gravity(self, reference_radius_m: float) -> float:
        if self.planet.mass_kg is not None:
            return float(
                GRAVITATIONAL_CONSTANT_M3_KG_S2
                * self.planet.mass_kg
                / reference_radius_m**2
            )
        if self.planet.gravity_m_s2 is None:
            raise RobertValidationError(
                "transmission requires planet mass or reference gravity"
            )
        return float(self.planet.gravity_m_s2)


def _validate_pressure_grid(actual: PressureGrid, expected: PressureGrid) -> None:
    if actual is expected:
        return
    if (
        actual.unit != expected.unit
        or not np.array_equal(actual.edges, expected.edges)
        or not np.array_equal(actual.centers, expected.centers)
    ):
        raise RobertValidationError(
            "shared atmosphere pressure grid must match the prepared transmission model"
        )


def _array_signature(*arrays: ArrayLike, labels: tuple[str, ...] = ()) -> str:
    digest = sha256()
    for label in labels:
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\0")
    for values in arrays:
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "ParameterizedTransmissionForwardModel",
    "ParameterizedTransmissionModelConfig",
]
