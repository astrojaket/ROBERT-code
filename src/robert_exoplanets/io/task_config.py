"""Strict YAML configuration for user-facing ROBERT tasks."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


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
    log_g_cgs: float
    metallicity_dex: float
    spectrum_model: Literal["phoenix", "blackbody"] = "phoenix"


class BodiesConfig(ConfigModel):
    planet: PlanetConfig
    star: StarConfig


class DatasetNuisanceConfig(ConfigModel):
    """Calibration and uncertainty controls for one named dataset."""

    offset_parameter: str | None = None
    uncertainty_scale: PositiveFloat = 1.0
    uncertainty_scale_parameter: str | None = None
    jitter_parameter: str | None = None


class ObservationsConfig(ConfigModel):
    loader: Literal[
        "robert_npz",
        "bello_arufe2025_l9859b",
        "schlawin2024_wasp69b",
        "wiser2025_wasp80b",
    ]
    path: Path
    datasets: tuple[str, ...] = Field(min_length=1)
    verify_checksum: bool = True
    miri_offset_parameter: str | None = None
    dataset_options: dict[str, DatasetNuisanceConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_loader_options(self) -> "ObservationsConfig":
        if self.loader == "robert_npz" and len(self.datasets) != 1:
            raise ValueError("robert_npz observations require exactly one dataset name")
        return self


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
    kappa_ir_parameter: str = Field(default="kappa_IR", min_length=1)
    gamma1_parameter: str = Field(default="gamma1", min_length=1)
    gamma2_parameter: str = Field(default="gamma2", min_length=1)
    irradiation_temperature_parameter: str = Field(default="T_irr", min_length=1)
    alpha_parameter: str = Field(default="alpha", min_length=1)

    @model_validator(mode="after")
    def validate_parameter_names(self) -> "ParmentierGuillotTemperatureConfig":
        names = (
            self.kappa_ir_parameter,
            self.gamma1_parameter,
            self.gamma2_parameter,
            self.irradiation_temperature_parameter,
            self.alpha_parameter,
        )
        if len(set(names)) != len(names):
            raise ValueError("Parmentier-Guillot parameter names must be unique")
        return self


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


class MadhusudhanSeagerTemperatureConfig(ConfigModel):
    model: Literal["madhusudhan_seager_2009"]
    pressure_unit: str = "bar"
    reference_pressure: PositiveFloat | None = None
    p1_parameter: str = "P1"
    p2_parameter: str = "P2"
    p3_parameter: str = "P3"
    t0_parameter: str = "T0"
    alpha1_parameter: str = "alpha1"
    alpha2_parameter: str = "alpha2"


class SplineTemperatureConfig(ConfigModel):
    model: Literal["spline"]
    knot_pressure: tuple[PositiveFloat, ...] = Field(min_length=2)
    knot_temperature_k: tuple[PositiveFloat, ...] | None = None
    parameter_names: tuple[str, ...] | None = None
    pressure_unit: str = "bar"
    extrapolation: Literal["raise", "clip"] = "raise"

    @model_validator(mode="after")
    def validate_knots(self) -> "SplineTemperatureConfig":
        count = len(self.knot_pressure)
        if len(set(self.knot_pressure)) != count:
            raise ValueError("spline knot_pressure values must be unique")
        if (
            self.knot_temperature_k is not None
            and len(self.knot_temperature_k) != count
        ):
            raise ValueError("spline knot_temperature_k must match knot_pressure")
        if self.parameter_names is not None and len(self.parameter_names) != count:
            raise ValueError("spline parameter_names must match knot_pressure")
        if self.knot_temperature_k is not None and self.parameter_names is not None:
            raise ValueError(
                "spline accepts fixed knot temperatures or parameter names, not both"
            )
        return self


TemperatureConfig = Annotated[
    ParmentierGuillotTemperatureConfig
    | IsothermalTemperatureConfig
    | TabulatedTemperatureConfig
    | MadhusudhanSeagerTemperatureConfig
    | SplineTemperatureConfig,
    Field(discriminator="model"),
]


class ChemistrySpeciesConfig(ConfigModel):
    label: str = Field(min_length=1)
    fastchem_name: str = Field(min_length=1)


class FastChemConfig(ConfigModel):
    model: Literal["fastchem_equilibrium"]
    fastchem_path: Path
    species: tuple[ChemistrySpeciesConfig, ...] = Field(min_length=1)
    metallicity_parameter: str = "metallicity"
    carbon_to_oxygen_parameter: str = "CtoO"
    constant_log10_vmr_parameters: dict[str, str] | None = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_constant_overrides(self) -> "FastChemConfig":
        labels = {item.label for item in self.species}
        overrides = self.constant_log10_vmr_parameters or {}
        unknown = sorted(set(overrides) - labels)
        if unknown:
            raise ValueError(
                "constant_log10_vmr_parameters contains unknown species: "
                + ", ".join(unknown)
            )
        parameters = tuple(overrides.values())
        if any(not name for name in parameters):
            raise ValueError("constant log10 VMR parameter names must not be empty")
        if len(set(parameters)) != len(parameters):
            raise ValueError("constant log10 VMR parameter names must be unique")
        return self


class FreeChemistryConfig(ConfigModel):
    model: Literal["free"]
    species: tuple[str, ...] = Field(min_length=1)
    parameter_names: dict[str, str] = Field(default_factory=dict)
    fixed_mixing_ratios: dict[str, float] = Field(default_factory=dict)
    parameter_mode: Literal["linear", "log10"] = "log10"
    background_species: tuple[str, ...] = ("H2", "He")
    background_fractions: tuple[PositiveFloat, ...] | None = None
    fill_background: bool = True
    excess_policy: Literal["raise", "normalize"] = "raise"
    phantom_species: str | None = None
    phantom_mean_molecular_weight_parameter: str | None = None

    @model_validator(mode="after")
    def validate_species(self) -> "FreeChemistryConfig":
        if len(set(self.species)) != len(self.species):
            raise ValueError("free chemistry species must be unique")
        unknown = (set(self.parameter_names) | set(self.fixed_mixing_ratios)) - set(
            self.species
        )
        if unknown:
            raise ValueError(
                "free chemistry mappings contain unknown species: "
                + ", ".join(sorted(unknown))
            )
        if self.background_fractions is not None and len(
            self.background_fractions
        ) != len(self.background_species):
            raise ValueError("background_fractions must match background_species")
        if self.fill_background and not self.background_species:
            raise ValueError("fill_background requires background_species")
        if set(self.species).intersection(self.background_species):
            raise ValueError(
                "free chemistry species must not overlap background_species"
            )
        phantom_values = (
            self.phantom_species,
            self.phantom_mean_molecular_weight_parameter,
        )
        if (phantom_values[0] is None) != (phantom_values[1] is None):
            raise ValueError(
                "phantom_species and phantom_mean_molecular_weight_parameter "
                "must be configured together"
            )
        if self.phantom_species is not None:
            if not self.fill_background:
                raise ValueError("phantom chemistry requires fill_background=true")
            if self.background_species != (self.phantom_species,):
                raise ValueError(
                    "phantom_species must be the sole background species"
                )
            if not self.phantom_mean_molecular_weight_parameter:
                raise ValueError(
                    "phantom_mean_molecular_weight_parameter must be non-empty"
                )
        return self


ChemistryConfig = Annotated[
    FastChemConfig | FreeChemistryConfig,
    Field(discriminator="model"),
]


class AtmosphereConfig(ConfigModel):
    pressure: PressureConfig
    temperature: TemperatureConfig
    chemistry: ChemistryConfig


class CloudFreeConfig(ConfigModel):
    model: Literal["none"] = "none"


# Backward-compatible alias for configuration code written before the
# cloud-free terminology was standardized.
ClearCloudConfig = CloudFreeConfig


class DeckHazeCloudConfig(ConfigModel):
    """Shared grey-deck and well-mixed power-law haze parameterization."""

    model: Literal["deck_haze"]
    log10_cloud_top_pressure_bar_parameter: str = Field(
        default="log_cloud_top_pressure_bar",
        min_length=1,
    )
    log10_cloud_optical_depth_parameter: str = Field(
        default="log_cloud_optical_depth",
        min_length=1,
    )
    log10_haze_mass_extinction_parameter: str = Field(
        default="log_haze_mass_extinction",
        min_length=1,
    )
    haze_slope_parameter: str = Field(default="haze_slope", min_length=1)
    haze_reference_wavelength_micron: PositiveFloat = 1.0
    deck_single_scattering_albedo: float = Field(default=0.0, ge=0.0, le=1.0)
    deck_asymmetry_factor: float = Field(default=0.0, ge=-1.0, le=1.0)
    haze_single_scattering_albedo: float = Field(default=1.0, ge=0.0, le=1.0)
    haze_asymmetry_factor: float = Field(default=0.0, ge=-1.0, le=1.0)
    multiple_scattering_backend: Literal[
        "none", "two_stream", "toon_hemispheric_mean", "sh4", "p3"
    ] = "sh4"

    @model_validator(mode="after")
    def validate_parameter_names(self) -> "DeckHazeCloudConfig":
        names = (
            self.log10_cloud_top_pressure_bar_parameter,
            self.log10_cloud_optical_depth_parameter,
            self.log10_haze_mass_extinction_parameter,
            self.haze_slope_parameter,
        )
        if len(set(names)) != len(names):
            raise ValueError("deck+haze cloud parameter names must be unique")
        return self


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
        2.4,
        4.0,
        5.5,
        7.0,
        9.0,
        12.0,
    )
    real_index_parameter_names: tuple[str, ...] = (
        "cloud_n_0",
        "cloud_n_1",
        "cloud_n_2",
        "cloud_n_3",
        "cloud_n_4",
        "cloud_n_5",
    )
    log10_imaginary_index_parameter_names: tuple[str, ...] = (
        "cloud_logk_0",
        "cloud_logk_1",
        "cloud_logk_2",
        "cloud_logk_3",
        "cloud_logk_4",
        "cloud_logk_5",
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
            raise ValueError(
                "real_index_parameter_names must match refractive-index nodes"
            )
        if len(self.log10_imaginary_index_parameter_names) != count:
            raise ValueError(
                "log10_imaginary_index_parameter_names must match refractive-index nodes"
            )
        return self


CloudsConfig = Annotated[
    CloudFreeConfig
    | DeckHazeCloudConfig
    | MieCatalogCloudConfig
    | MieDirectNkCloudConfig,
    Field(discriminator="model"),
]


class RegionalAtmosphereOverrideConfig(ConfigModel):
    """Optional regional replacements for the shared atmospheric defaults."""

    pressure: PressureConfig | None = None
    temperature: TemperatureConfig | None = None
    chemistry: ChemistryConfig | None = None


class RegionOverrideConfig(ConfigModel):
    """Overrides applied to one projected-disk emission region."""

    atmosphere: RegionalAtmosphereOverrideConfig = RegionalAtmosphereOverrideConfig()
    clouds: CloudsConfig | None = None


class OneRegionDiskEmissionConfig(ConfigModel):
    model: Literal["one_region"] = "one_region"


class DilutedDiskEmissionConfig(ConfigModel):
    model: Literal["diluted_one_region", "diluted"]
    dilution_parameter: str = Field(default="dayside_dilution", min_length=1)


class TwoRegionDiskEmissionConfig(ConfigModel):
    model: Literal["two_region", "2tp"]
    hot_fraction_parameter: str = Field(default="hot_area_fraction", min_length=1)
    hot_region: RegionOverrideConfig = RegionOverrideConfig()
    cold_region: RegionOverrideConfig


DiskEmissionConfig = Annotated[
    OneRegionDiskEmissionConfig
    | DilutedDiskEmissionConfig
    | TwoRegionDiskEmissionConfig,
    Field(discriminator="model"),
]


class ResolvedRegionConfig(ConfigModel):
    """Fully resolved configuration for one physical emission column."""

    name: str = Field(min_length=1)
    atmosphere: AtmosphereConfig
    clouds: CloudsConfig


class OpacityBinningConfig(ConfigModel):
    num: PositiveInt = 300
    use_rebin: bool = False
    remove_zeros: bool = True
    g_points: PositiveInt = 8


class OpacityConfig(ConfigModel):
    format: Literal["exomol_kta", "exomol_cross_section_hdf"]
    path: Path
    resolution: str = Field(pattern=r"^R[1-9][0-9]*$")
    species: tuple[str, ...] = Field(min_length=1)
    cache_directory: Path
    binning: OpacityBinningConfig = OpacityBinningConfig()


class GeometryConfig(ConfigModel):
    model: Literal["normal_emission", "gauss_legendre_disk"] = "normal_emission"
    points: PositiveInt = 4


class RadiativeTransferConfig(ConfigModel):
    model: Literal["emission", "transmission"] = "emission"
    geometry: GeometryConfig = GeometryConfig()
    include_rayleigh: bool = True
    gas_combination: Literal[
        "sum_by_g", "random_overlap", "equivalent_extinction"
    ] = (
        "random_overlap"
    )
    thermal_integration_backend: Literal["auto", "numpy", "numba"] = "auto"
    reference_pressure_bar: PositiveFloat = 1.0
    radius_scale_parameter: str | None = None
    gravity_model: Literal["constant", "inverse_square"] = "inverse_square"
    impact_quadrature_order: int = Field(default=8, ge=2)

    @model_validator(mode="after")
    def validate_model_options(self) -> "RadiativeTransferConfig":
        if self.model == "transmission" and self.gas_combination == "equivalent_extinction":
            raise ValueError(
                "transmission gas_combination must be 'sum_by_g' or 'random_overlap'"
            )
        if self.radius_scale_parameter is not None and not self.radius_scale_parameter:
            raise ValueError("radius_scale_parameter must be non-empty")
        return self


class PriorConfig(ConfigModel):
    type: Literal["uniform", "log_uniform", "centered_log_ratio"]
    lower: float
    upper: float
    group: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "PriorConfig":
        if self.upper <= self.lower:
            raise ValueError("prior upper must exceed lower")
        if self.type == "log_uniform" and self.lower <= 0.0:
            raise ValueError("log_uniform prior lower must be positive")
        if self.type == "centered_log_ratio":
            if self.lower >= 0.0 or self.upper != 0.0:
                raise ValueError(
                    "centered_log_ratio prior requires lower < 0 and upper = 0"
                )
            if self.group is not None and not self.group.strip():
                raise ValueError("centered_log_ratio prior group must be non-empty")
        elif self.group is not None:
            raise ValueError("prior group is only valid for centered_log_ratio")
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
    engine: Literal[
        "ultranest",
        "multinest",
        "optimal_estimation",
        "optimal_estimation_to_ultranest",
        "optimal_estimation_to_multinest",
    ] = "ultranest"
    live_points: PositiveInt = 400
    max_calls: PositiveInt | None = None
    multinest_max_iterations: NonNegativeInt = 0
    dlogz: PositiveFloat = 0.5
    sampling_efficiency: float = Field(default=0.8, gt=0.0, le=1.0)
    importance_nested_sampling: bool = True
    multimodal: bool = True
    iterations_before_update: PositiveInt = 100
    resume: Literal["resume", "resume-similar", "overwrite", "subfolder"] = "resume"
    show_status: bool = True
    seed: int | None = Field(default=None, ge=0)
    invalid_loglike_floor: float = Field(default=-1.0e100, lt=0.0)
    oe_max_iterations: PositiveInt = 8
    oe_convergence_tolerance: PositiveFloat = 1.0e-4
    oe_finite_difference_fraction: PositiveFloat = 1.0e-4
    oe_damping: float = Field(default=0.0, ge=0.0)
    oe_temperature_prior_sigma_k: PositiveFloat | None = None
    oe_temperature_correlation_length_dex: PositiveFloat | None = None
    prior_sigma: PositiveFloat = 4.0
    minimum_prior_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    require_oe_convergence: bool = True

    @model_validator(mode="after")
    def validate_engine_settings(self) -> "SamplerConfig":
        if "multinest" in self.engine and self.resume not in {"resume", "overwrite"}:
            raise ValueError(
                "MultiNest resume must be 'resume' or 'overwrite'"
            )
        temperature_prior_values = (
            self.oe_temperature_prior_sigma_k,
            self.oe_temperature_correlation_length_dex,
        )
        if (temperature_prior_values[0] is None) != (
            temperature_prior_values[1] is None
        ):
            raise ValueError(
                "OE temperature prior sigma and correlation length must be "
                "configured together"
            )
        return self


class OutputsConfig(ConfigModel):
    directory: Path


class LeaveOneOutConfig(ConfigModel):
    """Optional PSIS-LOO retrieval diagnostic controls."""

    enabled: bool = False
    max_posterior_draws: PositiveInt = 2_000
    seed: NonNegativeInt = 0
    pareto_k_threshold: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_draw_count(self) -> "LeaveOneOutConfig":
        if self.max_posterior_draws < 20:
            raise ValueError("plotting.leave_one_out.max_posterior_draws must be at least 20")
        return self


class PlottingConfig(ConfigModel):
    """Optional automatic and manual post-processing controls."""

    enabled: bool = False
    retrieval: bool = True
    forward: bool = True
    style: str = Field(default="default", min_length=1)
    image_format: Literal["png", "pdf", "svg"] = "png"
    dpi: PositiveInt = 180
    max_posterior_samples: PositiveInt = 20_000
    posterior_predictive_samples: PositiveInt = 200
    posterior_predictive_seed: NonNegativeInt = 0
    corner_max_parameters: PositiveInt = 20
    dataset_colors: dict[str, str] = Field(default_factory=dict)
    parameter_labels: dict[str, str] = Field(default_factory=dict)
    leave_one_out: LeaveOneOutConfig = LeaveOneOutConfig()

    @model_validator(mode="after")
    def validate_plot_mappings(self) -> "PlottingConfig":
        for mapping_name, mapping in (
            ("dataset_colors", self.dataset_colors),
            ("parameter_labels", self.parameter_labels),
        ):
            invalid = [key for key, value in mapping.items() if not key or not value]
            if invalid:
                raise ValueError(
                    f"plotting.{mapping_name} keys and values must be non-empty"
                )
        return self


class RuntimeConfig(ConfigModel):
    mpi_processes: PositiveInt | Literal["auto"] = "auto"
    scratch_directory: Path


class HousekeepingConfig(ConfigModel):
    """Machine- and project-specific paths kept in one visible YAML block."""

    project_directory: Path | None = None
    observations_directory: Path | None = None
    fastchem_directory: Path | None = None
    k_table_directory: Path | None = None
    opacity_cache_directory: Path | None = None
    output_directory: Path | None = None
    scratch_directory: Path | None = None
    optical_constants_directory: Path | None = None


class TaskConfig(ConfigModel):
    """Complete schema-versioned retrieval/forward-model configuration."""

    schema_version: Literal[2]
    paths: HousekeepingConfig | None = None
    run: RunConfig
    bodies: BodiesConfig
    observations: ObservationsConfig
    atmosphere: AtmosphereConfig
    clouds: CloudsConfig = CloudFreeConfig()
    disk_emission: DiskEmissionConfig = OneRegionDiskEmissionConfig()
    opacity: OpacityConfig
    radiative_transfer: RadiativeTransferConfig
    likelihood: LikelihoodConfig = LikelihoodConfig()
    parameters: tuple[ParameterConfig, ...] = Field(min_length=1)
    sampler: SamplerConfig = SamplerConfig()
    outputs: OutputsConfig
    plotting: PlottingConfig = PlottingConfig()
    runtime: RuntimeConfig
    # Legacy name retained so existing configurations continue to load. New
    # YAML should use the top-level ``paths`` block.
    housekeeping: HousekeepingConfig | None = None

    @model_validator(mode="after")
    def validate_cross_references(self) -> "TaskConfig":
        if self.paths is not None and self.housekeeping is not None:
            raise ValueError("configure paths or legacy housekeeping, not both")
        opacity = self.opacity.species
        if len(set(opacity)) != len(opacity):
            raise ValueError("opacity.species contains duplicates")
        regions = configured_regions(self)
        for region in regions:
            chemistry = _chemistry_species(region.atmosphere.chemistry)
            if len(set(chemistry)) != len(chemistry):
                raise ValueError(
                    f"{region.name} chemistry species labels contain duplicates"
                )
            missing = sorted(set(opacity) - set(chemistry))
            if missing:
                raise ValueError(
                    f"opacity species missing from {region.name} chemistry labels: "
                    + ", ".join(missing)
                )
        names = tuple(item.name for item in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        required: set[str] = set()
        for region in regions:
            required.update(_required_chemistry_parameters(region.atmosphere.chemistry))
            required.update(_required_temperature_parameters(region.atmosphere.temperature))
            required.update(_required_cloud_parameters(region.clouds))
        disk_mode = _disk_emission_mode(self.disk_emission)
        if disk_mode == "diluted_one_region":
            required.add(self.disk_emission.dilution_parameter)
            fraction_parameter = self.disk_emission.dilution_parameter
        elif disk_mode == "two_region":
            required.add(self.disk_emission.hot_fraction_parameter)
            fraction_parameter = self.disk_emission.hot_fraction_parameter
        else:
            fraction_parameter = None
        if fraction_parameter is not None:
            parameter_lookup = {item.name: item for item in self.parameters}
            configured_fraction = parameter_lookup.get(fraction_parameter)
            if configured_fraction is not None and (
                configured_fraction.prior.lower < 0.0
                or configured_fraction.prior.upper > 1.0
            ):
                raise ValueError(
                    f"{fraction_parameter} prior must lie within [0, 1]"
                )
        if self.sampler.oe_temperature_prior_sigma_k is not None:
            if not self.sampler.engine.startswith("optimal_estimation"):
                raise ValueError(
                    "correlated OE temperature priors require an optimal-estimation engine"
                )
            if disk_mode != "one_region":
                raise ValueError(
                    "correlated OE temperature priors currently require one_region disk emission"
                )
            temperature = regions[0].atmosphere.temperature
            if (
                temperature.model != "spline"
                or temperature.knot_temperature_k is not None
            ):
                raise ValueError(
                    "correlated OE temperature priors require a retrieved spline profile"
                )
        unknown_options = sorted(
            set(self.observations.dataset_options) - set(self.observations.datasets)
        )
        if unknown_options:
            raise ValueError(
                "dataset_options names are not selected observations: "
                + ", ".join(unknown_options)
            )
        for option in self.observations.dataset_options.values():
            required.update(
                parameter
                for parameter in (
                    option.offset_parameter,
                    option.uncertainty_scale_parameter,
                    option.jitter_parameter,
                )
                if parameter is not None
            )
        if self.observations.miri_offset_parameter is not None:
            required.add(self.observations.miri_offset_parameter)
        radiative_transfer = self.radiative_transfer
        if radiative_transfer.model == "transmission" and disk_mode != "one_region":
            raise ValueError(
                "diluted and two-region disk models require radiative_transfer.model=emission"
            )
        if radiative_transfer.model == "transmission":
            pressure = regions[0].atmosphere.pressure
            if not (
                pressure.top_bar
                <= radiative_transfer.reference_pressure_bar
                <= pressure.bottom_bar
            ):
                raise ValueError(
                    "transmission reference_pressure_bar must lie within the pressure grid"
                )
            if radiative_transfer.radius_scale_parameter is not None:
                required.add(radiative_transfer.radius_scale_parameter)
        clr_parameters = {
            item.name: item.prior
            for item in self.parameters
            if item.prior.type == "centered_log_ratio"
        }
        if clr_parameters:
            matched_clr_parameters: set[str] = set()
            for region in regions:
                chemistry_config = region.atmosphere.chemistry
                if chemistry_config.model != "free":
                    continue
                chemistry_parameters = _retrieved_free_chemistry_parameters(
                    chemistry_config
                )
                regional_clr = chemistry_parameters.intersection(clr_parameters)
                if not regional_clr:
                    continue
                if chemistry_config.parameter_mode != "log10":
                    raise ValueError(
                        "centered_log_ratio priors require chemistry parameter_mode=log10"
                    )
                if not chemistry_config.fill_background:
                    raise ValueError(
                        "centered_log_ratio priors require a background closure category"
                    )
                if regional_clr != chemistry_parameters:
                    raise ValueError(
                        "all and only retrieved free-chemistry abundances must use "
                        "centered_log_ratio priors within each region"
                    )
                groups = {
                    clr_parameters[name].group or "composition"
                    for name in regional_clr
                }
                if len(groups) != 1:
                    raise ValueError(
                        "free-chemistry centered_log_ratio priors must share one group per region"
                    )
                matched_clr_parameters.update(regional_clr)
            if matched_clr_parameters != set(clr_parameters):
                raise ValueError(
                    "centered_log_ratio priors require matching free chemistry parameters"
                )
            if self.sampler.engine == "optimal_estimation" or self.sampler.engine.startswith(
                "optimal_estimation_to_"
            ):
                raise ValueError(
                    "centered_log_ratio priors currently require direct nested sampling"
                )
        missing_parameters = sorted(required - set(names))
        if missing_parameters:
            raise ValueError(
                "required model parameters are missing: "
                + ", ".join(missing_parameters)
            )
        return self


def _disk_emission_mode(config: DiskEmissionConfig) -> str:
    aliases = {
        "one_region": "one_region",
        "diluted": "diluted_one_region",
        "diluted_one_region": "diluted_one_region",
        "2tp": "two_region",
        "two_region": "two_region",
    }
    return aliases[config.model]


def configured_regions(config: TaskConfig) -> tuple[ResolvedRegionConfig, ...]:
    """Resolve regional overrides against the top-level atmospheric defaults."""

    mode = _disk_emission_mode(config.disk_emission)
    if mode != "two_region":
        return (
            ResolvedRegionConfig(
                name="primary",
                atmosphere=config.atmosphere,
                clouds=config.clouds,
            ),
        )
    disk = config.disk_emission
    return (
        _resolve_region("hot", config.atmosphere, config.clouds, disk.hot_region),
        _resolve_region("cold", config.atmosphere, config.clouds, disk.cold_region),
    )


def _resolve_region(
    name: str,
    base_atmosphere: AtmosphereConfig,
    base_clouds: CloudsConfig,
    override: RegionOverrideConfig,
) -> ResolvedRegionConfig:
    atmosphere_override = override.atmosphere
    atmosphere = AtmosphereConfig(
        pressure=atmosphere_override.pressure or base_atmosphere.pressure,
        temperature=atmosphere_override.temperature or base_atmosphere.temperature,
        chemistry=atmosphere_override.chemistry or base_atmosphere.chemistry,
    )
    return ResolvedRegionConfig(
        name=name,
        atmosphere=atmosphere,
        clouds=override.clouds or base_clouds,
    )


def _chemistry_species(config: ChemistryConfig) -> tuple[str, ...]:
    if config.model == "fastchem_equilibrium":
        return tuple(item.label for item in config.species)
    return config.species + config.background_species


def _required_chemistry_parameters(config: ChemistryConfig) -> set[str]:
    if config.model == "fastchem_equilibrium":
        return {
            config.metallicity_parameter,
            config.carbon_to_oxygen_parameter,
            *(config.constant_log10_vmr_parameters or {}).values(),
        }
    required = _retrieved_free_chemistry_parameters(config)
    if config.phantom_mean_molecular_weight_parameter is not None:
        required.add(config.phantom_mean_molecular_weight_parameter)
    return required


def _retrieved_free_chemistry_parameters(config: FreeChemistryConfig) -> set[str]:
    return {
        config.parameter_names.get(species, species)
        for species in config.species
        if species not in config.fixed_mixing_ratios
    }


def _required_temperature_parameters(config: TemperatureConfig) -> set[str]:
    if config.model == "parmentier_guillot_2014":
        return {
            config.kappa_ir_parameter,
            config.gamma1_parameter,
            config.gamma2_parameter,
            config.irradiation_temperature_parameter,
            config.alpha_parameter,
        }
    if config.model == "isothermal":
        return set() if config.temperature_k is not None else {config.parameter_name}
    if config.model == "madhusudhan_seager_2009":
        return {
            config.p1_parameter,
            config.p2_parameter,
            config.p3_parameter,
            config.t0_parameter,
            config.alpha1_parameter,
            config.alpha2_parameter,
        }
    if config.model == "spline" and config.knot_temperature_k is None:
        return set(
            config.parameter_names
            or tuple(
                f"temperature_{index}"
                for index in range(len(config.knot_pressure))
            )
        )
    return set()


def _required_cloud_parameters(config: CloudsConfig) -> set[str]:
    if config.model == "deck_haze":
        return {
            config.log10_cloud_top_pressure_bar_parameter,
            config.log10_cloud_optical_depth_parameter,
            config.log10_haze_mass_extinction_parameter,
            config.haze_slope_parameter,
        }
    if config.model == "none":
        return set()
    required = {
        config.log10_mass_fraction_parameter,
        config.log10_radius_micron_parameter,
        config.log10_top_pressure_bar_parameter,
        config.log10_base_pressure_bar_parameter,
    }
    if config.model == "mie_direct_nk":
        required.update(config.real_index_parameter_names)
        required.update(config.log10_imaginary_index_parameter_names)
    return required


def load_task_config(path: str | Path) -> TaskConfig:
    """Load YAML, resolve its paths relative to the file, and validate it."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = _load_yaml_mapping(source)
    _apply_configured_paths(raw)
    for keys in (
        ("observations", "path"),
        ("atmosphere", "chemistry", "fastchem_path"),
        ("atmosphere", "temperature", "profile_path"),
        ("opacity", "path"),
        ("opacity", "cache_directory"),
        ("clouds", "optical_constants_path"),
        ("disk_emission", "hot_region", "atmosphere", "chemistry", "fastchem_path"),
        ("disk_emission", "hot_region", "atmosphere", "temperature", "profile_path"),
        ("disk_emission", "hot_region", "clouds", "optical_constants_path"),
        ("disk_emission", "cold_region", "atmosphere", "chemistry", "fastchem_path"),
        ("disk_emission", "cold_region", "atmosphere", "temperature", "profile_path"),
        ("disk_emission", "cold_region", "clouds", "optical_constants_path"),
        ("paths", "project_directory"),
        ("paths", "observations_directory"),
        ("paths", "fastchem_directory"),
        ("paths", "k_table_directory"),
        ("paths", "opacity_cache_directory"),
        ("paths", "output_directory"),
        ("paths", "scratch_directory"),
        ("paths", "optical_constants_directory"),
        ("housekeeping", "project_directory"),
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
            if section[keys[-1]] is None:
                continue
            value = _expanded_path(section[keys[-1]])
            section[keys[-1]] = str(
                value if value.is_absolute() else source.parent / value
            )
    return TaskConfig.model_validate(raw)


def _expanded_path(value: object) -> Path:
    text = os.path.expandvars(str(value))
    if "$" in text:
        raise ValueError(f"path contains an undefined environment variable: {value}")
    return Path(text).expanduser()


def _apply_configured_paths(raw: dict) -> None:
    """Fill component paths and local writable defaults from the top path block."""

    if raw.get("paths") is not None and raw.get("housekeeping") is not None:
        raise ValueError("configure paths or legacy housekeeping, not both")
    path_config = raw.get("paths", raw.get("housekeeping", {}))
    if path_config is None:
        path_config = {}
    if not isinstance(path_config, dict):
        raise ValueError("paths must be a YAML mapping")
    project_directory = path_config.get("project_directory") or "."
    derived = {
        "opacity_cache_directory": str(
            Path(str(project_directory)) / "opacity_cache"
        ),
        "output_directory": str(Path(str(project_directory)) / "outputs"),
        "scratch_directory": str(Path(str(project_directory)) / "scratch"),
    }
    for key, value in derived.items():
        if path_config.get(key) is None:
            path_config[key] = value
    if raw.get("paths") is not None:
        raw["paths"] = path_config
    elif raw.get("housekeeping") is not None:
        raw["housekeeping"] = path_config
    mappings = (
        (("observations", "path"), "observations_directory"),
        (("opacity", "path"), "k_table_directory"),
        (("opacity", "cache_directory"), "opacity_cache_directory"),
        (("outputs", "directory"), "output_directory"),
        (("runtime", "scratch_directory"), "scratch_directory"),
    )
    for keys, source_key in mappings:
        if path_config.get(source_key) is None:
            continue
        section = raw
        for key in keys[:-1]:
            if not isinstance(section, dict):
                raise ValueError("configuration sections must be YAML mappings")
            section = section.setdefault(key, {})
        if not isinstance(section, dict):
            raise ValueError("configuration sections must be YAML mappings")
        section.setdefault(keys[-1], path_config[source_key])
    _fill_regional_input_paths(raw, path_config)


def _fill_regional_input_paths(raw: dict, path_config: dict) -> None:
    atmosphere_cloud_pairs = [
        (raw.get("atmosphere", {}), raw.get("clouds")),
    ]
    disk = raw.get("disk_emission", {})
    if isinstance(disk, dict):
        for name in ("hot_region", "cold_region"):
            region = disk.get(name, {})
            if isinstance(region, dict):
                atmosphere_cloud_pairs.append(
                    (region.get("atmosphere", {}), region.get("clouds"))
                )
    for atmosphere, clouds in atmosphere_cloud_pairs:
        if isinstance(atmosphere, dict):
            chemistry = atmosphere.get("chemistry", {})
            if (
                isinstance(chemistry, dict)
                and chemistry.get("model") == "fastchem_equilibrium"
                and path_config.get("fastchem_directory") is not None
            ):
                chemistry.setdefault("fastchem_path", path_config["fastchem_directory"])
        if (
            isinstance(clouds, dict)
            and clouds.get("model") == "mie_catalog"
            and path_config.get("optical_constants_directory") is not None
        ):
            clouds.setdefault(
                "optical_constants_path", path_config["optical_constants_directory"]
            )


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
    parent = (
        (source.parent / parent).resolve()
        if not parent.is_absolute()
        else parent.resolve()
    )
    if parent == source:
        raise ValueError(
            f"configuration extends itself: {source} declares extends: {extends!r}; "
            "a self-contained run configuration must not contain an extends entry"
        )
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
