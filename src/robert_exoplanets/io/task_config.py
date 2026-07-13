"""Strict YAML configuration for user-facing ROBERT tasks."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class ConfigModel(BaseModel):
    """Base for configuration sections: immutable and typo-intolerant."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(ConfigModel):
    name: str = Field(min_length=1)
    description: str = ""


class PlanetConfig(ConfigModel):
    name: str = Field(min_length=1)
    radius_m: PositiveFloat
    mass_kg: PositiveFloat | None = None
    gravity_m_s2: PositiveFloat | None = None

    @model_validator(mode="after")
    def require_mass_or_gravity(self) -> "PlanetConfig":
        if self.mass_kg is None and self.gravity_m_s2 is None:
            raise ValueError("provide planet.mass_kg or planet.gravity_m_s2")
        return self


class StarConfig(ConfigModel):
    name: str = Field(min_length=1)
    radius_m: PositiveFloat
    effective_temperature_k: PositiveFloat


class BodiesConfig(ConfigModel):
    planet: PlanetConfig
    star: StarConfig


class ObservationsConfig(ConfigModel):
    loader: Literal["schlawin2024_wasp69b", "wiser2025_wasp80b"]
    path: Path
    datasets: tuple[str, ...] = Field(min_length=1)
    verify_checksum: bool = True
    miri_offset_parameter: str | None = None


class PressureConfig(ConfigModel):
    bottom_bar: PositiveFloat = 100.0
    top_bar: PositiveFloat = 1.0e-6
    layers: PositiveInt = 80

    @model_validator(mode="after")
    def validate_order(self) -> "PressureConfig":
        if self.bottom_bar <= self.top_bar:
            raise ValueError("pressure.bottom_bar must exceed pressure.top_bar")
        return self


class ParmentierGuillotTemperatureConfig(ConfigModel):
    model: Literal["parmentier_guillot_2014"]
    internal_temperature_k: float = Field(default=100.0, ge=0.0)


class IsothermalTemperatureConfig(ConfigModel):
    model: Literal["isothermal"]
    temperature_k: PositiveFloat | None = None
    parameter_name: str = "temperature"


class TabulatedTemperatureConfig(ConfigModel):
    model: Literal["tabulated"]
    profile_path: Path
    pressure_column: str = "pressure_bar"
    temperature_column: str = "temperature_K"
    pressure_unit: str = "bar"
    extrapolation: Literal["raise", "clip"] = "raise"


TemperatureConfig = Annotated[
    ParmentierGuillotTemperatureConfig
    | IsothermalTemperatureConfig
    | TabulatedTemperatureConfig,
    Field(discriminator="model"),
]


class ChemistrySpeciesConfig(ConfigModel):
    label: str = Field(min_length=1)
    fastchem_name: str = Field(min_length=1)


class ChemistryConfig(ConfigModel):
    model: Literal["fastchem_equilibrium"]
    fastchem_path: Path
    species: tuple[ChemistrySpeciesConfig, ...] = Field(min_length=1)
    metallicity_parameter: str = "metallicity"
    carbon_to_oxygen_parameter: str = "CtoO"


class AtmosphereConfig(ConfigModel):
    pressure: PressureConfig
    temperature: TemperatureConfig
    chemistry: ChemistryConfig


class ClearCloudConfig(ConfigModel):
    model: Literal["none"] = "none"


class MieCatalogCloudConfig(ConfigModel):
    """Fixed laboratory optical constants with retrieved cloud structure."""

    model: Literal["mie_catalog"]
    optical_constants_path: Path
    material: str = Field(min_length=1)
    particle_density_kg_m3: PositiveFloat = 3200.0
    geometric_stddev: PositiveFloat = 1.0
    quadrature_points: PositiveInt = 1
    log10_mass_fraction_parameter: str = "log_cloud_mass_fraction"
    log10_radius_micron_parameter: str = "log_cloud_radius_micron"
    log10_top_pressure_bar_parameter: str = "log_cloud_top_pressure_bar"
    log10_base_pressure_bar_parameter: str = "log_cloud_base_pressure_bar"
    multiple_scattering_backend: Literal["sh4"] = "sh4"


class MieDirectNkCloudConfig(ConfigModel):
    """Retrieved refractive-index nodes with retrieved cloud structure."""

    model: Literal["mie_direct_nk"]
    refractive_index_wavelength_micron: tuple[PositiveFloat, ...] = (
        2.4, 4.0, 5.5, 7.0, 9.0, 12.0,
    )
    real_index_parameter_names: tuple[str, ...] = (
        "cloud_n_0", "cloud_n_1", "cloud_n_2", "cloud_n_3", "cloud_n_4", "cloud_n_5",
    )
    log10_imaginary_index_parameter_names: tuple[str, ...] = (
        "cloud_logk_0", "cloud_logk_1", "cloud_logk_2", "cloud_logk_3", "cloud_logk_4", "cloud_logk_5",
    )
    particle_density_kg_m3: PositiveFloat = 3200.0
    geometric_stddev: PositiveFloat = 1.0
    quadrature_points: PositiveInt = 1
    log10_mass_fraction_parameter: str = "log_cloud_mass_fraction"
    log10_radius_micron_parameter: str = "log_cloud_radius_micron"
    log10_top_pressure_bar_parameter: str = "log_cloud_top_pressure_bar"
    log10_base_pressure_bar_parameter: str = "log_cloud_base_pressure_bar"
    multiple_scattering_backend: Literal["sh4"] = "sh4"

    @model_validator(mode="after")
    def validate_node_names(self) -> "MieDirectNkCloudConfig":
        count = len(self.refractive_index_wavelength_micron)
        if len(self.real_index_parameter_names) != count:
            raise ValueError("real_index_parameter_names must match refractive-index nodes")
        if len(self.log10_imaginary_index_parameter_names) != count:
            raise ValueError("log10_imaginary_index_parameter_names must match refractive-index nodes")
        return self


CloudsConfig = Annotated[
    ClearCloudConfig | MieCatalogCloudConfig | MieDirectNkCloudConfig,
    Field(discriminator="model"),
]


class OpacityBinningConfig(ConfigModel):
    num: PositiveInt = 300
    use_rebin: bool = False
    remove_zeros: bool = True


class OpacityConfig(ConfigModel):
    format: Literal["exomol_kta"]
    path: Path
    resolution: str = Field(pattern=r"^R[1-9][0-9]*$")
    species: tuple[str, ...] = Field(min_length=1)
    cache_directory: Path
    binning: OpacityBinningConfig = OpacityBinningConfig()


class GeometryConfig(ConfigModel):
    model: Literal["normal_emission", "gauss_legendre_disk"] = "normal_emission"
    points: PositiveInt = 4


class RadiativeTransferConfig(ConfigModel):
    model: Literal["clear_emission"] = "clear_emission"
    geometry: GeometryConfig = GeometryConfig()
    include_rayleigh: bool = True
    gas_combination: Literal["random_overlap", "equivalent_extinction"] = "random_overlap"
    thermal_integration_backend: Literal["auto", "numpy", "numba"] = "auto"


class PriorConfig(ConfigModel):
    type: Literal["uniform", "log_uniform"]
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "PriorConfig":
        if self.upper <= self.lower:
            raise ValueError("prior upper must exceed lower")
        if self.type == "log_uniform" and self.lower <= 0.0:
            raise ValueError("log_uniform prior lower must be positive")
        return self


class ParameterConfig(ConfigModel):
    name: str = Field(min_length=1)
    prior: PriorConfig
    label: str | None = None
    unit: str | None = None
    value: float | None = None


class LikelihoodConfig(ConfigModel):
    model: Literal["gaussian"] = "gaussian"
    include_normalization: bool = True


class SamplerConfig(ConfigModel):
    engine: Literal["ultranest"] = "ultranest"
    live_points: PositiveInt = 400
    max_calls: PositiveInt | None = 200_000
    dlogz: PositiveFloat = 0.5
    resume: Literal["resume", "resume-similar", "overwrite", "subfolder"] = "resume"
    show_status: bool = True
    seed: int | None = Field(default=None, ge=0)


class OutputsConfig(ConfigModel):
    directory: Path


class RuntimeConfig(ConfigModel):
    mpi_processes: PositiveInt | Literal["auto"] = "auto"
    scratch_directory: Path


class HousekeepingConfig(ConfigModel):
    """Machine- and project-specific locations for a user-facing YAML file."""

    observations_directory: Path
    fastchem_directory: Path
    k_table_directory: Path
    opacity_cache_directory: Path
    output_directory: Path
    scratch_directory: Path
    optical_constants_directory: Path | None = None


class TaskConfig(ConfigModel):
    """Complete schema-versioned retrieval/forward-model configuration."""

    schema_version: Literal[1]
    run: RunConfig
    bodies: BodiesConfig
    observations: ObservationsConfig
    atmosphere: AtmosphereConfig
    clouds: CloudsConfig = ClearCloudConfig()
    opacity: OpacityConfig
    radiative_transfer: RadiativeTransferConfig
    likelihood: LikelihoodConfig = LikelihoodConfig()
    parameters: tuple[ParameterConfig, ...] = Field(min_length=1)
    sampler: SamplerConfig = SamplerConfig()
    outputs: OutputsConfig
    runtime: RuntimeConfig
    housekeeping: HousekeepingConfig | None = None

    @model_validator(mode="after")
    def validate_cross_references(self) -> "TaskConfig":
        opacity = self.opacity.species
        chemistry = tuple(item.label for item in self.atmosphere.chemistry.species)
        if len(set(opacity)) != len(opacity):
            raise ValueError("opacity.species contains duplicates")
        if len(set(chemistry)) != len(chemistry):
            raise ValueError("atmosphere.chemistry.species labels contain duplicates")
        missing = sorted(set(opacity) - set(chemistry))
        if missing:
            raise ValueError("opacity species missing from chemistry labels: " + ", ".join(missing))
        names = tuple(item.name for item in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        required = {
            self.atmosphere.chemistry.metallicity_parameter,
            self.atmosphere.chemistry.carbon_to_oxygen_parameter,
        }
        temperature = self.atmosphere.temperature
        if temperature.model == "parmentier_guillot_2014":
            required.update({"kappa_IR", "gamma1", "gamma2", "T_irr", "alpha"})
        elif temperature.model == "isothermal" and temperature.temperature_k is None:
            required.add(temperature.parameter_name)
        missing_parameters = sorted(required - set(names))
        if missing_parameters:
            raise ValueError("required model parameters are missing: " + ", ".join(missing_parameters))
        clouds = self.clouds
        if clouds.model != "none":
            cloud_parameters = {
                clouds.log10_mass_fraction_parameter,
                clouds.log10_radius_micron_parameter,
                clouds.log10_top_pressure_bar_parameter,
                clouds.log10_base_pressure_bar_parameter,
            }
            if clouds.model == "mie_direct_nk":
                cloud_parameters.update(clouds.real_index_parameter_names)
                cloud_parameters.update(clouds.log10_imaginary_index_parameter_names)
            missing_cloud_parameters = sorted(cloud_parameters - set(names))
            if missing_cloud_parameters:
                raise ValueError(
                    "required cloud parameters are missing: "
                    + ", ".join(missing_cloud_parameters)
                )
        return self


def load_task_config(path: str | Path) -> TaskConfig:
    """Load YAML, resolve its paths relative to the file, and validate it."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = _load_yaml_mapping(source)
    _apply_housekeeping_paths(raw)
    for keys in (
        ("observations", "path"),
        ("atmosphere", "chemistry", "fastchem_path"),
        ("atmosphere", "temperature", "profile_path"),
        ("opacity", "path"),
        ("opacity", "cache_directory"),
        ("clouds", "optical_constants_path"),
        ("housekeeping", "observations_directory"),
        ("housekeeping", "fastchem_directory"),
        ("housekeeping", "k_table_directory"),
        ("housekeeping", "opacity_cache_directory"),
        ("housekeeping", "output_directory"),
        ("housekeeping", "scratch_directory"),
        ("housekeeping", "optical_constants_directory"),
        ("outputs", "directory"),
        ("runtime", "scratch_directory"),
    ):
        section = raw
        for key in keys[:-1]:
            if not isinstance(section, dict) or key not in section:
                break
            section = section[key]
        else:
            if not isinstance(section, dict) or keys[-1] not in section:
                continue
            value = Path(section[keys[-1]]).expanduser()
            section[keys[-1]] = str(value if value.is_absolute() else source.parent / value)
    return TaskConfig.model_validate(raw)


def _apply_housekeeping_paths(raw: dict) -> None:
    """Fill internal path fields from one readable user-facing path block."""

    housekeeping = raw.get("housekeeping")
    if housekeeping is None:
        return
    if not isinstance(housekeeping, dict):
        raise ValueError("housekeeping must be a YAML mapping")
    mappings = (
        (("observations", "path"), "observations_directory"),
        (("atmosphere", "chemistry", "fastchem_path"), "fastchem_directory"),
        (("opacity", "path"), "k_table_directory"),
        (("opacity", "cache_directory"), "opacity_cache_directory"),
        (("outputs", "directory"), "output_directory"),
        (("runtime", "scratch_directory"), "scratch_directory"),
    )
    for keys, source_key in mappings:
        if source_key not in housekeeping:
            continue
        section = raw
        for key in keys[:-1]:
            if not isinstance(section, dict):
                raise ValueError("configuration sections must be YAML mappings")
            section = section.setdefault(key, {})
        if not isinstance(section, dict):
            raise ValueError("configuration sections must be YAML mappings")
        section.setdefault(keys[-1], housekeeping[source_key])
    clouds = raw.get("clouds")
    if (
        isinstance(clouds, dict)
        and clouds.get("model") == "mie_catalog"
        and "optical_constants_directory" in housekeeping
    ):
        clouds.setdefault("optical_constants_path", housekeeping["optical_constants_directory"])


def _load_yaml_mapping(source: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    """Load one YAML mapping, resolving an optional relative ``extends`` file."""

    if source in ancestors:
        chain = " -> ".join(str(path) for path in (*ancestors, source))
        raise ValueError(f"configuration extends cycle: {chain}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ROBERT configuration must be a YAML mapping")
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError("configuration extends must be a non-empty YAML path")
    parent = Path(extends).expanduser()
    parent = (source.parent / parent).resolve() if not parent.is_absolute() else parent.resolve()
    if not parent.is_file():
        raise FileNotFoundError(parent)
    return _deep_merge(_load_yaml_mapping(parent, (*ancestors, source)), raw)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge mappings recursively; lists and scalar values replace wholesale."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # Discriminated configuration sections (for example clouds.model)
            # cannot retain fields belonging to a different variant.
            if value.get("model") and value.get("model") != merged[key].get("model"):
                merged[key] = value
            else:
                merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def initialize_task_directories(config: TaskConfig) -> tuple[Path, ...]:
    """Create every writable directory selected by a task configuration."""

    scratch = config.runtime.scratch_directory
    directories = (
        config.outputs.directory,
        config.opacity.cache_directory,
        config.opacity.cache_directory / config.opacity.resolution,
        scratch,
        scratch / "numba",
        scratch / "matplotlib",
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(scratch / "numba")
    os.environ["MPLCONFIGDIR"] = str(scratch / "matplotlib")
    return directories


__all__ = ["TaskConfig", "initialize_task_directories", "load_task_config"]
