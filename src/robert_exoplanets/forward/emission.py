"""Reusable parameterized clear-sky emission forward model."""

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
    OpacitySamplingProvider,
    OpacityProvider,
    PreparedOpacity,
    PreparedOpacitySampling,
    pressure_values_in_unit,
)
from robert_exoplanets.rt import (
    CiaTable,
    DiscGeometry,
    RefractiveIndexSpectrum,
    assemble_gas_optical_depth,
    assemble_opacity_sampling_gas_optical_depth,
    cia_optical_depth,
    gauss_legendre_disk_geometry,
    grey_cloud_from_mass_extinction,
    lognormal_mie_optics,
    mie_cloud_from_mass_fraction,
    rayleigh_scattering_optical_depth,
    refractive_index_from_parameters,
    solve_clear_sky_emission,
    solve_clear_sky_emission_spectrum,
    sh4_spectrum_backend_name,
    thermal_integration_backend_name,
)

GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11


@dataclass(frozen=True)
class ClearSkyEmissionModelConfig:
    """Typed physical and numerical choices for a clear-sky emission model.

    Trace-gas parameters represent constant-with-altitude log10 volume mixing
    ratios. Temperature and radius parameters are optional uniform offset and
    multiplicative scale terms respectively.
    """

    opacity_species: tuple[str, ...]
    log_vmr_parameters: Mapping[str, str]
    temperature_offset_parameter: str | None = "temperature_offset"
    radius_scale_parameter: str | None = "radius_scale"
    mean_molecular_weight: float = 2.3
    hydrogen_fraction_of_background: float = 0.84
    helium_fraction_of_background: float = 0.16
    include_rayleigh: bool = True
    gas_combination: str = "random_overlap"
    thermal_integration_backend: str = "auto"
    compute_diagnostics: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = tuple(str(item).strip() for item in self.opacity_species)
        if not species or any(not item for item in species):
            raise RobertValidationError("opacity_species must contain non-empty names")
        if len(set(species)) != len(species):
            raise RobertValidationError("opacity_species must not contain duplicates")
        parameters = {
            str(key).strip(): str(value).strip()
            for key, value in self.log_vmr_parameters.items()
        }
        if set(parameters) != set(species):
            raise RobertValidationError(
                "log_vmr_parameters keys must match opacity_species"
            )
        if any(not value for value in parameters.values()) or len(
            set(parameters.values())
        ) != len(parameters):
            raise RobertValidationError(
                "log VMR parameter names must be non-empty and unique"
            )
        for field_name in ("temperature_offset_parameter", "radius_scale_parameter"):
            value = getattr(self, field_name)
            if value is not None and not str(value).strip():
                raise RobertValidationError(
                    f"{field_name} must be non-empty when provided"
                )
        reserved = {
            value
            for value in (
                self.temperature_offset_parameter,
                self.radius_scale_parameter,
            )
            if value is not None
        }
        if reserved.intersection(parameters.values()):
            raise RobertValidationError(
                "temperature/radius parameter names must differ from abundance parameters"
            )
        mmw = float(self.mean_molecular_weight)
        h2_fraction = float(self.hydrogen_fraction_of_background)
        he_fraction = float(self.helium_fraction_of_background)
        if not np.isfinite(mmw) or mmw <= 0.0:
            raise RobertValidationError(
                "mean_molecular_weight must be finite and positive"
            )
        if not np.isfinite(h2_fraction) or not np.isfinite(he_fraction):
            raise RobertValidationError("background gas fractions must be finite")
        if (
            h2_fraction < 0.0
            or he_fraction < 0.0
            or not np.isclose(
                h2_fraction + he_fraction,
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise RobertValidationError(
                "hydrogen and helium background fractions must be non-negative and sum to one"
            )
        if self.gas_combination not in {"sum_by_g", "random_overlap"}:
            raise RobertValidationError(
                "gas_combination must be 'sum_by_g' or 'random_overlap'"
            )
        backend = thermal_integration_backend_name(self.thermal_integration_backend)
        if not isinstance(self.compute_diagnostics, bool):
            raise RobertValidationError("compute_diagnostics must be a boolean")
        object.__setattr__(self, "opacity_species", species)
        object.__setattr__(self, "log_vmr_parameters", immutable_mapping(parameters))
        object.__setattr__(self, "mean_molecular_weight", mmw)
        object.__setattr__(self, "hydrogen_fraction_of_background", h2_fraction)
        object.__setattr__(self, "helium_fraction_of_background", he_fraction)
        object.__setattr__(self, "thermal_integration_backend", backend)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        """Parameter names required by the configured model."""

        parameters = [
            self.log_vmr_parameters[species] for species in self.opacity_species
        ]
        if self.temperature_offset_parameter is not None:
            parameters.append(self.temperature_offset_parameter)
        if self.radius_scale_parameter is not None:
            parameters.append(self.radius_scale_parameter)
        return tuple(parameters)


@dataclass(frozen=True)
class ParameterizedClearSkyEmissionModelConfig:
    """RT choices for a model driven by atmosphere parameterization objects."""

    opacity_species: tuple[str, ...]
    radius_scale_parameter: str | None = None
    include_rayleigh: bool = True
    cia_normal_hydrogen: bool = True
    cia_temperature_extrapolation: str = "clip"
    cia_spectral_extrapolation: str = "zero"
    gas_combination: str = "random_overlap"
    thermal_integration_backend: str = "auto"
    compute_diagnostics: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = tuple(str(item).strip() for item in self.opacity_species)
        if not species or any(not item for item in species):
            raise RobertValidationError("opacity_species must contain non-empty names")
        if len(set(species)) != len(species):
            raise RobertValidationError("opacity_species must not contain duplicates")
        radius_parameter = self.radius_scale_parameter
        if radius_parameter is not None:
            radius_parameter = str(radius_parameter).strip()
            if not radius_parameter:
                raise RobertValidationError(
                    "radius_scale_parameter must be non-empty when provided"
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
        if not isinstance(self.compute_diagnostics, bool):
            raise RobertValidationError("compute_diagnostics must be a boolean")
        object.__setattr__(self, "opacity_species", species)
        object.__setattr__(self, "radius_scale_parameter", radius_parameter)
        object.__setattr__(
            self,
            "thermal_integration_backend",
            thermal_integration_backend_name(self.thermal_integration_backend),
        )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ClearSkyEmissionForwardModel:
    """Parameterized correlated-k clear-sky emission model.

    Opacity must already be prepared on ``spectral_grid`` (for example using
    exo_k binning). The model performs no file discovery or spectral resampling
    inside likelihood calls.
    """

    planet: Planet
    star: Star
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    base_temperature_K: ArrayLike
    opacity_provider: OpacityProvider
    config: ClearSkyEmissionModelConfig
    geometry: DiscGeometry | None = None
    prepared_opacity: PreparedOpacity = field(init=False, repr=False)
    gravity_m_s2: float = field(init=False)

    def __post_init__(self) -> None:
        if self.planet.radius_m is None:
            raise RobertValidationError("clear-sky emission requires planet.radius_m")
        if self.star.radius_m is None or self.star.effective_temperature_k is None:
            raise RobertValidationError(
                "clear-sky emission requires star.radius_m and star.effective_temperature_k"
            )
        temperature = np.array(self.base_temperature_K, dtype=float, copy=True)
        if temperature.shape != (self.pressure_grid.n_layers,):
            raise RobertValidationError(
                "base_temperature_K must match pressure-grid layers"
            )
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise RobertValidationError(
                "base_temperature_K must be finite and positive"
            )
        missing = tuple(
            species
            for species in self.config.opacity_species
            if species not in self.opacity_provider.tables
        )
        if missing:
            raise RobertValidationError(
                f"opacity provider is missing configured species: {', '.join(missing)}"
            )
        gravity = self._planet_gravity()
        geometry = self.geometry or gauss_legendre_disk_geometry(4)
        prepared = self.opacity_provider.prepare(
            self.spectral_grid,
            self.pressure_grid,
            species=self.config.opacity_species,
        )
        temperature.setflags(write=False)
        object.__setattr__(self, "base_temperature_K", temperature)
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "prepared_opacity", prepared)

    @property
    def required_parameters(self) -> tuple[str, ...]:
        """Parameter names required by this forward model."""

        return self.config.required_parameters

    @property
    def opacity_identifiers(self) -> Mapping[str, str]:
        """Immutable source-opacity checksums by species."""

        return immutable_mapping(
            {
                species: str(
                    self.opacity_provider.tables[species].metadata.get(
                        "checksum_sha256", ""
                    )
                )
                for species in self.config.opacity_species
            }
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        """Flat model metadata suitable for a retrieval run manifest."""

        return immutable_mapping(
            {
                "forward_model": "clear_sky_emission",
                "planet_name": self.planet.name,
                "planet_radius_m": f"{self.planet.radius_m:.17g}",
                "planet_gravity_m_s2": f"{self.gravity_m_s2:.17g}",
                "planet_mass_kg": ""
                if self.planet.mass_kg is None
                else f"{self.planet.mass_kg:.17g}",
                "star_name": self.star.name,
                "star_radius_m": f"{self.star.radius_m:.17g}",
                "star_effective_temperature_k": f"{self.star.effective_temperature_k:.17g}",
                "opacity_species": ",".join(self.config.opacity_species),
                "log_vmr_parameters": ",".join(
                    f"{species}:{self.config.log_vmr_parameters[species]}"
                    for species in self.config.opacity_species
                ),
                "opacity_provider": self.opacity_provider.name,
                "opacity_cache_key": self.prepared_opacity.cache_key,
                "gas_combination": self.config.gas_combination,
                "include_rayleigh": str(bool(self.config.include_rayleigh)).lower(),
                "thermal_integration_backend": self.config.thermal_integration_backend,
                "compute_diagnostics": str(self.config.compute_diagnostics).lower(),
                "geometry": self.geometry.name,
                "geometry_points": str(self.geometry.n_points),
                "mean_molecular_weight": f"{self.config.mean_molecular_weight:.12g}",
                "hydrogen_fraction_of_background": (
                    f"{self.config.hydrogen_fraction_of_background:.12g}"
                ),
                "helium_fraction_of_background": f"{self.config.helium_fraction_of_background:.12g}",
                "temperature_offset_parameter": self.config.temperature_offset_parameter
                or "",
                "radius_scale_parameter": self.config.radius_scale_parameter or "",
                "pressure_grid_layers": str(self.pressure_grid.n_layers),
                "pressure_grid_min": f"{np.min(self.pressure_grid.centers):.17g}",
                "pressure_grid_max": f"{np.max(self.pressure_grid.centers):.17g}",
                "pressure_grid_unit": self.pressure_grid.unit,
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
                "base_temperature_sha256": _array_signature(
                    self.base_temperature_K,
                    labels=("K",),
                ),
                **dict(self.config.metadata),
            }
        )

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        """Evaluate an observed-grid eclipse-depth spectrum."""

        missing = tuple(
            name for name in self.required_parameters if name not in parameters
        )
        if missing:
            raise RobertValidationError(
                f"clear-sky emission parameters are missing: {', '.join(missing)}"
            )
        parameter_values = {
            name: float(parameters[name]) for name in self.required_parameters
        }
        if not all(np.isfinite(value) for value in parameter_values.values()):
            raise RobertValidationError("clear-sky emission parameters must be finite")
        trace_abundances = {
            species: float(
                np.power(
                    10.0, parameter_values[self.config.log_vmr_parameters[species]]
                )
            )
            for species in self.config.opacity_species
        }
        trace_sum = float(sum(trace_abundances.values()))
        if not np.isfinite(trace_sum) or trace_sum >= 1.0:
            raise RobertValidationError(
                "trace-gas volume mixing ratios must have a finite sum below one"
            )
        background = 1.0 - trace_sum
        composition = {
            species: np.full(self.pressure_grid.n_layers, abundance)
            for species, abundance in trace_abundances.items()
        }
        composition.update(
            {
                "H2": np.full(
                    self.pressure_grid.n_layers,
                    self.config.hydrogen_fraction_of_background * background,
                ),
                "He": np.full(
                    self.pressure_grid.n_layers,
                    self.config.helium_fraction_of_background * background,
                ),
            }
        )
        temperature_offset = (
            0.0
            if self.config.temperature_offset_parameter is None
            else parameter_values[self.config.temperature_offset_parameter]
        )
        radius_scale = (
            1.0
            if self.config.radius_scale_parameter is None
            else parameter_values[self.config.radius_scale_parameter]
        )
        if radius_scale <= 0.0:
            raise RobertValidationError("radius scale must be positive")
        atmosphere = AtmosphereState(
            pressure_grid=self.pressure_grid,
            temperature=self.base_temperature_K + temperature_offset,
            composition=composition,
            mean_molecular_weight=np.full(
                self.pressure_grid.n_layers,
                self.config.mean_molecular_weight,
            ),
            metadata={"forward_model": "clear_sky_emission"},
        )
        gas_optical_depth = _evaluate_gas_optical_depth(
            self.opacity_provider,
            self.prepared_opacity,
            atmosphere,
            gravity_m_s2=self.gravity_m_s2,
            gas_combination=self.config.gas_combination,
            retain_species_tau=self.config.compute_diagnostics,
        )
        additional_optical_depths = []
        if self.config.include_rayleigh:
            additional_optical_depths.append(
                rayleigh_scattering_optical_depth(gas_optical_depth)
            )
        if not self.config.compute_diagnostics:
            return solve_clear_sky_emission_spectrum(
                gas_optical_depth,
                geometry=self.geometry,
                additional_optical_depths=additional_optical_depths,
                planet_radius_m=self.planet.radius_m * radius_scale,
                star_radius_m=self.star.radius_m,
                star_temperature_k=self.star.effective_temperature_k,
                thermal_integration_backend=self.config.thermal_integration_backend,
            )
        result = solve_clear_sky_emission(
            gas_optical_depth,
            geometry=self.geometry,
            additional_optical_depths=additional_optical_depths,
            planet_radius_m=self.planet.radius_m * radius_scale,
            star_radius_m=self.star.radius_m,
            star_temperature_k=self.star.effective_temperature_k,
            thermal_integration_backend=self.config.thermal_integration_backend,
        )
        if result.eclipse_depth is None:
            raise RobertValidationError(
                "clear-sky emission solver did not return eclipse depth"
            )
        return result.eclipse_depth

    def _planet_gravity(self) -> float:
        if self.planet.gravity_m_s2 is not None:
            return float(self.planet.gravity_m_s2)
        if self.planet.mass_kg is None or self.planet.radius_m is None:
            raise RobertValidationError(
                "planet mass and radius are required to derive gravity"
            )
        return float(
            GRAVITATIONAL_CONSTANT_M3_KG_S2
            * self.planet.mass_kg
            / self.planet.radius_m**2
        )


@dataclass(frozen=True)
class ParameterizedClearSkyEmissionForwardModel:
    """Clear-sky emission model with runtime temperature and chemistry profiles."""

    planet: Planet
    star: Star
    spectral_grid: SpectralGrid
    atmosphere_builder: AtmosphereBuilder
    opacity_provider: OpacityProvider
    config: ParameterizedClearSkyEmissionModelConfig
    cia_table: CiaTable | tuple[CiaTable, ...] | None = None
    geometry: DiscGeometry | None = None
    prepared_opacity: PreparedOpacity = field(init=False, repr=False)
    gravity_m_s2: float = field(init=False)

    def __post_init__(self) -> None:
        if self.planet.radius_m is None:
            raise RobertValidationError(
                "parameterized emission requires planet.radius_m"
            )
        if self.star.radius_m is None or self.star.effective_temperature_k is None:
            raise RobertValidationError(
                "parameterized emission requires star.radius_m and star.effective_temperature_k"
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
        required = self.required_parameters
        if len(set(required)) != len(required):
            raise RobertValidationError(
                "temperature, chemistry, and radius parameter names must be unique"
            )
        gravity = _planet_gravity(self.planet)
        geometry = self.geometry or gauss_legendre_disk_geometry(4)
        prepared = self.opacity_provider.prepare(
            self.spectral_grid,
            self.pressure_grid,
            species=self.config.opacity_species,
        )
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "prepared_opacity", prepared)

    @property
    def pressure_grid(self) -> PressureGrid:
        return self.atmosphere_builder.pressure_grid

    @property
    def cia_tables(self) -> tuple[CiaTable, ...]:
        """Return every configured CIA source as an additive tuple."""

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
        if self.config.radius_scale_parameter is not None:
            parameters.append(self.config.radius_scale_parameter)
        return tuple(parameters)

    @property
    def opacity_identifiers(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                species: str(
                    self.opacity_provider.tables[species].metadata.get(
                        "checksum_sha256", ""
                    )
                )
                for species in self.config.opacity_species
            }
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        mmw_model = self.atmosphere_builder.mean_molecular_weight_model
        chemistry_metadata = getattr(
            self.atmosphere_builder.chemistry_model, "metadata", {}
        )
        return immutable_mapping(
            {
                "forward_model": "parameterized_clear_sky_emission",
                "planet_name": self.planet.name,
                "planet_radius_m": f"{self.planet.radius_m:.17g}",
                "planet_gravity_m_s2": f"{self.gravity_m_s2:.17g}",
                "star_name": self.star.name,
                "star_radius_m": f"{self.star.radius_m:.17g}",
                "star_effective_temperature_k": f"{self.star.effective_temperature_k:.17g}",
                "opacity_species": ",".join(self.config.opacity_species),
                "opacity_provider": self.opacity_provider.name,
                "opacity_cache_key": self.prepared_opacity.cache_key,
                "temperature_profile": self.atmosphere_builder.temperature_profile.name,
                "chemistry_model": self.atmosphere_builder.chemistry_model.name,
                "chemistry_species": ",".join(
                    self.atmosphere_builder.chemistry_model.species
                ),
                "mean_molecular_weight_model": (
                    "fixed_scalar"
                    if mmw_model is None
                    else type(mmw_model).__qualname__
                ),
                "required_parameters": ",".join(self.required_parameters),
                "gas_combination": self.config.gas_combination,
                "include_rayleigh": str(bool(self.config.include_rayleigh)).lower(),
                "include_cia": str(bool(self.cia_tables)).lower(),
                "cia_source": ",".join(
                    str(table.metadata.get("source_project", ""))
                    for table in self.cia_tables
                ),
                "cia_checksum_sha256": ",".join(
                    str(table.metadata.get("checksum_sha256", ""))
                    for table in self.cia_tables
                ),
                "cia_normal_hydrogen": str(
                    bool(self.config.cia_normal_hydrogen)
                ).lower(),
                "thermal_integration_backend": self.config.thermal_integration_backend,
                "compute_diagnostics": str(self.config.compute_diagnostics).lower(),
                "geometry": self.geometry.name,
                "geometry_points": str(self.geometry.n_points),
                "radius_scale_parameter": self.config.radius_scale_parameter or "",
                "pressure_grid_layers": str(self.pressure_grid.n_layers),
                "pressure_grid_min": f"{np.min(self.pressure_grid.centers):.17g}",
                "pressure_grid_max": f"{np.max(self.pressure_grid.centers):.17g}",
                "pressure_grid_unit": self.pressure_grid.unit,
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
                **{
                    f"chemistry_{key}": str(value)
                    for key, value in chemistry_metadata.items()
                },
                **dict(self.config.metadata),
            }
        )

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        parameter_values = self.validated_parameters(parameters)
        atmosphere = self.atmosphere_builder.build(parameter_values)
        return self.evaluate_atmosphere(atmosphere, parameter_values)

    def validated_parameters(
        self,
        parameters: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Return the finite model parameters required by this prepared model."""

        missing = tuple(
            name for name in self.required_parameters if name not in parameters
        )
        if missing:
            raise RobertValidationError(
                "parameterized emission parameters are missing: " + ", ".join(missing)
            )
        parameter_values = {
            name: float(parameters[name]) for name in self.required_parameters
        }
        if not all(np.isfinite(value) for value in parameter_values.values()):
            raise RobertValidationError(
                "parameterized emission parameters must be finite"
            )
        return immutable_mapping(parameter_values)

    def evaluate_atmosphere(
        self,
        atmosphere: AtmosphereState,
        parameters: Mapping[str, float],
    ) -> Spectrum:
        """Evaluate opacity and RT for an already-computed atmospheric state.

        This preserves the mode-specific correlated-k preparation while
        allowing several spectral models to share temperature and chemistry.
        """

        parameter_values = self.validated_parameters(parameters)
        if atmosphere.pressure_grid is not self.pressure_grid:
            same_grid = (
                atmosphere.pressure_grid.unit == self.pressure_grid.unit
                and np.array_equal(
                    atmosphere.pressure_grid.edges,
                    self.pressure_grid.edges,
                )
                and np.array_equal(
                    atmosphere.pressure_grid.centers,
                    self.pressure_grid.centers,
                )
            )
            if not same_grid:
                raise RobertValidationError(
                    "shared atmosphere pressure grid must match the prepared emission model"
                )
        gas_optical_depth = _evaluate_gas_optical_depth(
            self.opacity_provider,
            self.prepared_opacity,
            atmosphere,
            gravity_m_s2=self.gravity_m_s2,
            gas_combination=self.config.gas_combination,
            retain_species_tau=self.config.compute_diagnostics,
        )
        additional_optical_depths = []
        for table in self.cia_tables:
            additional_optical_depths.append(
                cia_optical_depth(
                    gas_optical_depth,
                    table,
                    normal_hydrogen=self.config.cia_normal_hydrogen,
                    temperature_extrapolation=self.config.cia_temperature_extrapolation,
                    spectral_extrapolation=self.config.cia_spectral_extrapolation,
                )
            )
        if self.config.include_rayleigh:
            additional_optical_depths.append(
                rayleigh_scattering_optical_depth(gas_optical_depth)
            )
        radius_scale = (
            1.0
            if self.config.radius_scale_parameter is None
            else parameter_values[self.config.radius_scale_parameter]
        )
        if radius_scale <= 0.0:
            raise RobertValidationError("radius scale must be positive")
        if not self.config.compute_diagnostics:
            return solve_clear_sky_emission_spectrum(
                gas_optical_depth,
                geometry=self.geometry,
                additional_optical_depths=additional_optical_depths,
                planet_radius_m=self.planet.radius_m * radius_scale,
                star_radius_m=self.star.radius_m,
                star_temperature_k=self.star.effective_temperature_k,
                thermal_integration_backend=self.config.thermal_integration_backend,
            )
        result = solve_clear_sky_emission(
            gas_optical_depth,
            geometry=self.geometry,
            additional_optical_depths=additional_optical_depths,
            planet_radius_m=self.planet.radius_m * radius_scale,
            star_radius_m=self.star.radius_m,
            star_temperature_k=self.star.effective_temperature_k,
            thermal_integration_backend=self.config.thermal_integration_backend,
        )
        if result.eclipse_depth is None:
            raise RobertValidationError(
                "clear-sky emission solver did not return eclipse depth"
            )
        return result.eclipse_depth


@dataclass(frozen=True)
class GreyScatteringCloudConfig:
    """Region-specific uniform gray cloud-opacity retrieval parameter."""

    log10_mass_extinction_parameter: str
    single_scattering_albedo: float = 1.0
    asymmetry_factor: float = 0.0
    multiple_scattering_backend: str = "sh4"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameter = str(self.log10_mass_extinction_parameter).strip()
        if not parameter:
            raise RobertValidationError(
                "log10_mass_extinction_parameter must not be empty"
            )
        omega = float(self.single_scattering_albedo)
        asymmetry = float(self.asymmetry_factor)
        if not np.isfinite(omega) or not 0.0 <= omega <= 1.0:
            raise RobertValidationError("single_scattering_albedo must lie in [0, 1]")
        if not np.isfinite(asymmetry) or not -1.0 <= asymmetry <= 1.0:
            raise RobertValidationError("asymmetry_factor must lie in [-1, 1]")
        backend = self.multiple_scattering_backend.strip().lower()
        if backend not in {"two_stream", "toon_hemispheric_mean", "sh4", "p3"}:
            raise RobertValidationError(
                "gray scattering cloud requires a multiple-scattering backend"
            )
        object.__setattr__(self, "log10_mass_extinction_parameter", parameter)
        object.__setattr__(self, "single_scattering_albedo", omega)
        object.__setattr__(self, "asymmetry_factor", asymmetry)
        object.__setattr__(self, "multiple_scattering_backend", backend)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ParameterizedGreyCloudEmissionForwardModel(
    ParameterizedClearSkyEmissionForwardModel
):
    """Parameterized emission column with a uniform gray scattering opacity."""

    cloud: GreyScatteringCloudConfig = field(
        default_factory=lambda: GreyScatteringCloudConfig("log_cloud_mass_extinction")
    )

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (
            *super().required_parameters,
            self.cloud.log10_mass_extinction_parameter,
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                **dict(super().manifest_metadata),
                "forward_model": "parameterized_grey_cloud_emission",
                "cloud_model": "uniform_mass_extinction",
                "cloud_log10_mass_extinction_parameter": (
                    self.cloud.log10_mass_extinction_parameter
                ),
                "cloud_mass_extinction_unit": "cm2/g_bulk_atmosphere",
                "cloud_single_scattering_albedo": (
                    f"{self.cloud.single_scattering_albedo:.17g}"
                ),
                "cloud_asymmetry_factor": f"{self.cloud.asymmetry_factor:.17g}",
                "cloud_multiple_scattering_backend": self.cloud.multiple_scattering_backend,
                "cloud_spectrum_only": str(
                    not self.config.compute_diagnostics
                    and self.cloud.multiple_scattering_backend in {"sh4", "p3"}
                ).lower(),
                "cloud_sh4_spectrum_backend": (
                    sh4_spectrum_backend_name(self.config.thermal_integration_backend)
                    if self.cloud.multiple_scattering_backend in {"sh4", "p3"}
                    else ""
                ),
                **dict(self.cloud.metadata),
            }
        )

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        missing = tuple(
            name for name in self.required_parameters if name not in parameters
        )
        if missing:
            raise RobertValidationError(
                "parameterized cloudy emission parameters are missing: "
                + ", ".join(missing)
            )
        parameter_values = {
            name: float(parameters[name]) for name in self.required_parameters
        }
        if not all(np.isfinite(value) for value in parameter_values.values()):
            raise RobertValidationError(
                "parameterized cloudy emission parameters must be finite"
            )
        atmosphere = self.atmosphere_builder.build(parameter_values)
        gas_optical_depth = _evaluate_gas_optical_depth(
            self.opacity_provider,
            self.prepared_opacity,
            atmosphere,
            gravity_m_s2=self.gravity_m_s2,
            gas_combination=self.config.gas_combination,
            retain_species_tau=self.config.compute_diagnostics,
        )
        additional_optical_depths = []
        for table in self.cia_tables:
            additional_optical_depths.append(
                cia_optical_depth(
                    gas_optical_depth,
                    table,
                    normal_hydrogen=self.config.cia_normal_hydrogen,
                    temperature_extrapolation=self.config.cia_temperature_extrapolation,
                    spectral_extrapolation=self.config.cia_spectral_extrapolation,
                )
            )
        if self.config.include_rayleigh:
            additional_optical_depths.append(
                rayleigh_scattering_optical_depth(gas_optical_depth)
            )
        cloud_opacity = float(
            np.power(10.0, parameter_values[self.cloud.log10_mass_extinction_parameter])
        )
        if not np.isfinite(cloud_opacity):
            raise RobertValidationError("gray cloud mass extinction overflowed")
        additional_optical_depths.append(
            grey_cloud_from_mass_extinction(
                gas_optical_depth,
                mass_extinction_cm2_g=cloud_opacity,
                single_scattering_albedo=self.cloud.single_scattering_albedo,
                asymmetry_factor=self.cloud.asymmetry_factor,
            )
        )
        radius_scale = (
            1.0
            if self.config.radius_scale_parameter is None
            else parameter_values[self.config.radius_scale_parameter]
        )
        if radius_scale <= 0.0:
            raise RobertValidationError("radius scale must be positive")
        if (
            not self.config.compute_diagnostics
            and self.cloud.multiple_scattering_backend in {"sh4", "p3"}
        ):
            return solve_clear_sky_emission_spectrum(
                gas_optical_depth,
                geometry=self.geometry,
                additional_optical_depths=additional_optical_depths,
                multiple_scattering_backend=self.cloud.multiple_scattering_backend,
                planet_radius_m=self.planet.radius_m * radius_scale,
                star_radius_m=self.star.radius_m,
                star_temperature_k=self.star.effective_temperature_k,
                thermal_integration_backend=self.config.thermal_integration_backend,
            )
        result = solve_clear_sky_emission(
            gas_optical_depth,
            geometry=self.geometry,
            additional_optical_depths=additional_optical_depths,
            multiple_scattering_backend=self.cloud.multiple_scattering_backend,
            planet_radius_m=self.planet.radius_m * radius_scale,
            star_radius_m=self.star.radius_m,
            star_temperature_k=self.star.effective_temperature_k,
            thermal_integration_backend=self.config.thermal_integration_backend,
        )
        if result.eclipse_depth is None:
            raise RobertValidationError(
                "cloudy emission solver did not return eclipse depth"
            )
        return result.eclipse_depth


@dataclass(frozen=True)
class RefractiveIndexCloudConfig:
    """Cloud-type-agnostic Mie cloud retrieval parameters.

    The refractive index is represented by nodal real ``n`` and log10
    imaginary ``k`` parameters. Particle composition is not assumed. Material
    catalogues are used after retrieval for comparison, or may provide fixed
    indices through the lower-level Mie API.
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
        wavelength = tuple(
            float(item) for item in self.refractive_index_wavelength_micron
        )
        real_names = tuple(
            str(item).strip() for item in self.real_index_parameter_names
        )
        imaginary_names = tuple(
            str(item).strip() for item in self.log10_imaginary_index_parameter_names
        )
        fixed_index = self.fixed_refractive_index
        if fixed_index is None:
            if not wavelength or any(
                not np.isfinite(item) or item <= 0.0 for item in wavelength
            ):
                raise RobertValidationError(
                    "refractive-index wavelength nodes must be finite and positive"
                )
            if any(
                right <= left
                for left, right in zip(wavelength[:-1], wavelength[1:], strict=True)
            ):
                raise RobertValidationError(
                    "refractive-index wavelength nodes must be strictly increasing"
                )
            if len(real_names) != len(wavelength) or len(imaginary_names) != len(
                wavelength
            ):
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
            None if item is None else str(item).strip()
            for item in (
                self.geometric_stddev_parameter,
                self.log10_cloud_top_pressure_bar_parameter,
                self.log10_cloud_base_pressure_bar_parameter,
            )
        )
        all_names = (
            *real_names,
            *imaginary_names,
            *scalar_names,
            *(item for item in optional_names if item),
        )
        if any(not item for item in all_names) or len(set(all_names)) != len(all_names):
            raise RobertValidationError(
                "cloud retrieval parameter names must be non-empty and unique"
            )
        density = float(self.particle_density_kg_m3)
        width = float(self.geometric_stddev)
        if not np.isfinite(density) or density <= 0.0:
            raise RobertValidationError(
                "particle_density_kg_m3 must be finite and positive"
            )
        if not np.isfinite(width) or width < 1.0:
            raise RobertValidationError(
                "geometric_stddev must be finite and at least one"
            )
        points = int(self.quadrature_points)
        if points < 1:
            raise RobertValidationError("quadrature_points must be positive")
        extrapolation = str(self.refractive_index_extrapolation).strip().lower()
        if extrapolation not in {"raise", "clip"}:
            raise RobertValidationError(
                "refractive_index_extrapolation must be 'raise' or 'clip'"
            )
        backend = str(self.multiple_scattering_backend).strip().lower()
        if backend not in {"two_stream", "toon_hemispheric_mean", "sh4", "p3"}:
            raise RobertValidationError(
                "refractive-index cloud requires a multiple-scattering backend"
            )
        object.__setattr__(self, "refractive_index_wavelength_micron", wavelength)
        object.__setattr__(self, "real_index_parameter_names", real_names)
        object.__setattr__(
            self, "log10_imaginary_index_parameter_names", imaginary_names
        )
        object.__setattr__(
            self, "log10_condensate_mass_fraction_parameter", scalar_names[0]
        )
        object.__setattr__(
            self, "log10_effective_radius_micron_parameter", scalar_names[1]
        )
        object.__setattr__(self, "geometric_stddev_parameter", optional_names[0])
        object.__setattr__(
            self, "log10_cloud_top_pressure_bar_parameter", optional_names[1]
        )
        object.__setattr__(
            self, "log10_cloud_base_pressure_bar_parameter", optional_names[2]
        )
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
            item
            for item in (
                self.geometric_stddev_parameter,
                self.log10_cloud_top_pressure_bar_parameter,
                self.log10_cloud_base_pressure_bar_parameter,
            )
            if item is not None
        )
        return tuple(parameters)


@dataclass(frozen=True)
class ParameterizedRefractiveIndexCloudEmissionForwardModel(
    ParameterizedClearSkyEmissionForwardModel
):
    """Emission column with directly retrieved complex cloud refractive index."""

    cloud: RefractiveIndexCloudConfig = field(
        default_factory=lambda: RefractiveIndexCloudConfig(
            refractive_index_wavelength_micron=(1.0, 10.0),
            real_index_parameter_names=("cloud_n_0", "cloud_n_1"),
            log10_imaginary_index_parameter_names=("cloud_logk_0", "cloud_logk_1"),
            log10_condensate_mass_fraction_parameter="log_cloud_mass_fraction",
            log10_effective_radius_micron_parameter="log_cloud_radius_micron",
            particle_density_kg_m3=3000.0,
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(set(self.required_parameters)) != len(self.required_parameters):
            raise RobertValidationError(
                "cloud and atmosphere retrieval parameter names must be unique"
            )

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (*super().required_parameters, *self.cloud.required_parameters)

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                **dict(super().manifest_metadata),
                "forward_model": "parameterized_refractive_index_cloud_emission",
                "cloud_model": "lognormal_homogeneous_sphere_mie",
                "cloud_refractive_index_parameterization": (
                    "nodal_n_log10_k"
                    if self.cloud.fixed_refractive_index is None
                    else "tabulated_n_k"
                ),
                "cloud_refractive_index_wavelength_micron": ",".join(
                    f"{item:.17g}"
                    for item in self.cloud.refractive_index_wavelength_micron
                ),
                "cloud_real_index_parameters": ",".join(
                    self.cloud.real_index_parameter_names
                ),
                "cloud_log10_imaginary_index_parameters": ",".join(
                    self.cloud.log10_imaginary_index_parameter_names
                ),
                "cloud_particle_density_kg_m3": f"{self.cloud.particle_density_kg_m3:.17g}",
                "cloud_geometric_stddev": f"{self.cloud.geometric_stddev:.17g}",
                "cloud_quadrature_points": str(self.cloud.quadrature_points),
                "cloud_multiple_scattering_backend": self.cloud.multiple_scattering_backend,
                "cloud_spectrum_only": str(
                    not self.config.compute_diagnostics
                    and self.cloud.multiple_scattering_backend in {"sh4", "p3"}
                ).lower(),
                "cloud_sh4_spectrum_backend": (
                    sh4_spectrum_backend_name(self.config.thermal_integration_backend)
                    if self.cloud.multiple_scattering_backend in {"sh4", "p3"}
                    else ""
                ),
                "cloud_phase_function_closure": "exact_mie_legendre_moments_through_l4",
                "cloud_refractive_index_mode": (
                    "retrieved_nodal_n_log10_k"
                    if self.cloud.fixed_refractive_index is None
                    else "fixed_tabulated_n_k"
                ),
                "cloud_refractive_index_name": (
                    "retrieved refractive index"
                    if self.cloud.fixed_refractive_index is None
                    else self.cloud.fixed_refractive_index.name
                ),
                **dict(self.cloud.metadata),
            }
        )

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        missing = tuple(
            name for name in self.required_parameters if name not in parameters
        )
        if missing:
            raise RobertValidationError(
                "parameterized refractive-index cloud parameters are missing: "
                + ", ".join(missing)
            )
        parameter_values = {
            name: float(parameters[name]) for name in self.required_parameters
        }
        if not all(np.isfinite(value) for value in parameter_values.values()):
            raise RobertValidationError(
                "parameterized refractive-index cloud parameters must be finite"
            )
        atmosphere = self.atmosphere_builder.build(parameter_values)
        gas_optical_depth = _evaluate_gas_optical_depth(
            self.opacity_provider,
            self.prepared_opacity,
            atmosphere,
            gravity_m_s2=self.gravity_m_s2,
            gas_combination=self.config.gas_combination,
            retain_species_tau=self.config.compute_diagnostics,
        )
        additional_optical_depths = [
            cia_optical_depth(
                gas_optical_depth,
                table,
                normal_hydrogen=self.config.cia_normal_hydrogen,
                temperature_extrapolation=self.config.cia_temperature_extrapolation,
                spectral_extrapolation=self.config.cia_spectral_extrapolation,
            )
            for table in self.cia_tables
        ]
        if self.config.include_rayleigh:
            additional_optical_depths.append(
                rayleigh_scattering_optical_depth(gas_optical_depth)
            )

        index = self.cloud.fixed_refractive_index
        if index is None:
            index = refractive_index_from_parameters(
                self.cloud.refractive_index_wavelength_micron,
                parameter_values,
                real_parameter_names=self.cloud.real_index_parameter_names,
                log10_imaginary_parameter_names=self.cloud.log10_imaginary_index_parameter_names,
            )
        radius = float(
            np.power(
                10.0,
                parameter_values[self.cloud.log10_effective_radius_micron_parameter],
            )
        )
        width = (
            self.cloud.geometric_stddev
            if self.cloud.geometric_stddev_parameter is None
            else parameter_values[self.cloud.geometric_stddev_parameter]
        )
        particle_optics = lognormal_mie_optics(
            index,
            self.spectral_grid,
            effective_radius_micron=radius,
            geometric_stddev=width,
            particle_density_kg_m3=self.cloud.particle_density_kg_m3,
            quadrature_points=self.cloud.quadrature_points,
            extrapolation=self.cloud.refractive_index_extrapolation,
        )
        mass_fraction_value = float(
            np.power(
                10.0,
                parameter_values[self.cloud.log10_condensate_mass_fraction_parameter],
            )
        )
        pressure_bar = pressure_values_in_unit(
            self.pressure_grid.centers, self.pressure_grid.unit, "bar"
        )
        active = np.ones(self.pressure_grid.n_layers, dtype=bool)
        top_pressure = None
        base_pressure = None
        if self.cloud.log10_cloud_top_pressure_bar_parameter is not None:
            top_pressure = float(
                np.power(
                    10.0,
                    parameter_values[self.cloud.log10_cloud_top_pressure_bar_parameter],
                )
            )
            active &= pressure_bar >= top_pressure
        if self.cloud.log10_cloud_base_pressure_bar_parameter is not None:
            base_pressure = float(
                np.power(
                    10.0,
                    parameter_values[
                        self.cloud.log10_cloud_base_pressure_bar_parameter
                    ],
                )
            )
            active &= pressure_bar <= base_pressure
        if (
            top_pressure is not None
            and base_pressure is not None
            and top_pressure > base_pressure
        ):
            raise RobertValidationError(
                "cloud top pressure must not exceed cloud base pressure"
            )
        mass_fraction = np.where(active, mass_fraction_value, 0.0)
        additional_optical_depths.append(
            mie_cloud_from_mass_fraction(
                gas_optical_depth,
                particle_optics,
                condensate_mass_fraction=mass_fraction,
            )
        )

        radius_scale = (
            1.0
            if self.config.radius_scale_parameter is None
            else parameter_values[self.config.radius_scale_parameter]
        )
        if radius_scale <= 0.0:
            raise RobertValidationError("radius scale must be positive")
        if (
            not self.config.compute_diagnostics
            and self.cloud.multiple_scattering_backend in {"sh4", "p3"}
        ):
            return solve_clear_sky_emission_spectrum(
                gas_optical_depth,
                geometry=self.geometry,
                additional_optical_depths=additional_optical_depths,
                multiple_scattering_backend=self.cloud.multiple_scattering_backend,
                planet_radius_m=self.planet.radius_m * radius_scale,
                star_radius_m=self.star.radius_m,
                star_temperature_k=self.star.effective_temperature_k,
                thermal_integration_backend=self.config.thermal_integration_backend,
            )
        result = solve_clear_sky_emission(
            gas_optical_depth,
            geometry=self.geometry,
            additional_optical_depths=additional_optical_depths,
            multiple_scattering_backend=self.cloud.multiple_scattering_backend,
            planet_radius_m=self.planet.radius_m * radius_scale,
            star_radius_m=self.star.radius_m,
            star_temperature_k=self.star.effective_temperature_k,
            thermal_integration_backend=self.config.thermal_integration_backend,
        )
        if result.eclipse_depth is None:
            raise RobertValidationError(
                "refractive-index cloud solver did not return eclipse depth"
            )
        return result.eclipse_depth


def _planet_gravity(planet: Planet) -> float:
    if planet.gravity_m_s2 is not None:
        return float(planet.gravity_m_s2)
    if planet.mass_kg is None or planet.radius_m is None:
        raise RobertValidationError(
            "planet mass and radius are required to derive gravity"
        )
    return float(GRAVITATIONAL_CONSTANT_M3_KG_S2 * planet.mass_kg / planet.radius_m**2)


def _evaluate_gas_optical_depth(
    provider: OpacityProvider,
    prepared: PreparedOpacity,
    atmosphere: AtmosphereState,
    *,
    gravity_m_s2: float,
    gas_combination: str,
    retain_species_tau: bool = True,
):
    if isinstance(provider, OpacitySamplingProvider):
        if not isinstance(prepared, PreparedOpacitySampling):
            raise RobertValidationError(
                "opacity-sampling provider requires prepared opacity sampling"
            )
        return assemble_opacity_sampling_gas_optical_depth(
            atmosphere,
            provider,
            prepared,
            gravity_m_s2=gravity_m_s2,
        )
    evaluated = provider.evaluate(atmosphere, prepared)
    return assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=gravity_m_s2,
        gas_combination=gas_combination,
        retain_species_tau=retain_species_tau,
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
    "ClearSkyEmissionForwardModel",
    "ClearSkyEmissionModelConfig",
    "GreyScatteringCloudConfig",
    "ParameterizedClearSkyEmissionForwardModel",
    "ParameterizedGreyCloudEmissionForwardModel",
    "ParameterizedRefractiveIndexCloudEmissionForwardModel",
    "ParameterizedClearSkyEmissionModelConfig",
    "RefractiveIndexCloudConfig",
]
