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


class StellarContaminationRegionConfig(ConfigModel):
    """One fixed-temperature active region in the visible stellar disk."""

    name: str = Field(min_length=1)
    kind: Literal["spot", "facula", "heterogeneity"]
    temperature_k: PositiveFloat
    covering_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    covering_fraction_parameter: str | None = None
    log_g_cgs: float | None = None
    metallicity_dex: float | None = None

    @model_validator(mode="after")
    def require_one_fraction_source(self) -> "StellarContaminationRegionConfig":
        if (self.covering_fraction is None) == (
            self.covering_fraction_parameter is None
        ):
            raise ValueError(
                "provide exactly one of covering_fraction or covering_fraction_parameter"
            )
        if (
            self.covering_fraction_parameter is not None
            and not self.covering_fraction_parameter.strip()
        ):
            raise ValueError("covering_fraction_parameter must be non-empty")
        return self


class StellarContaminationConfig(ConfigModel):
    """Rackham/POSEIDON stellar-contamination transform configuration."""

    model: Literal["poseidon_rackham"] = "poseidon_rackham"
    regions: tuple[StellarContaminationRegionConfig, ...] = ()
    transit_chord_temperature_k: PositiveFloat | None = None
    transit_chord_log_g_cgs: float | None = None
    transit_chord_metallicity_dex: float | None = None

    @model_validator(mode="after")
    def validate_regions(self) -> "StellarContaminationConfig":
        names = tuple(region.name for region in self.regions)
        if len(set(names)) != len(names):
            raise ValueError("stellar contamination region names must be unique")
        parameters = tuple(
            region.covering_fraction_parameter
            for region in self.regions
            if region.covering_fraction_parameter is not None
        )
        if len(set(parameters)) != len(parameters):
            raise ValueError(
                "stellar covering-fraction parameter names must be unique"
            )
        fixed_total = sum(
            region.covering_fraction or 0.0 for region in self.regions
        )
        if fixed_total > 1.0:
            raise ValueError(
                "fixed stellar heterogeneity fractions must sum to at most one"
            )
        if self.transit_chord_temperature_k is None and (
            self.transit_chord_log_g_cgs is not None
            or self.transit_chord_metallicity_dex is not None
        ):
            raise ValueError(
                "transit chord log_g/metallicity require transit_chord_temperature_k"
            )
        return self


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
    prior_sigma: PositiveFloat = 4.0
    minimum_prior_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    require_oe_convergence: bool = True

    @model_validator(mode="after")
    def validate_engine_settings(self) -> "SamplerConfig":
        if "multinest" in self.engine and self.resume not in {"resume", "overwrite"}:
            raise ValueError(
                "MultiNest resume must be 'resume' or 'overwrite'"
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

    schema_version: Literal[2]
    run: RunConfig
    bodies: BodiesConfig
    stellar_contamination: StellarContaminationConfig | None = None
    observations: ObservationsConfig
    atmosphere: AtmosphereConfig
    clouds: CloudsConfig = CloudFreeConfig()
    opacity: OpacityConfig
    radiative_transfer: RadiativeTransferConfig
    likelihood: LikelihoodConfig = LikelihoodConfig()
    parameters: tuple[ParameterConfig, ...] = Field(min_length=1)
    sampler: SamplerConfig = SamplerConfig()
    outputs: OutputsConfig
    plotting: PlottingConfig = PlottingConfig()
    runtime: RuntimeConfig
    housekeeping: HousekeepingConfig | None = None

    @model_validator(mode="after")
    def validate_cross_references(self) -> "TaskConfig":
        opacity = self.opacity.species
        chemistry_config = self.atmosphere.chemistry
        chemistry = (
            tuple(item.label for item in chemistry_config.species)
            if chemistry_config.model == "fastchem_equilibrium"
            else chemistry_config.species + chemistry_config.background_species
        )
        if len(set(opacity)) != len(opacity):
            raise ValueError("opacity.species contains duplicates")
        if len(set(chemistry)) != len(chemistry):
            raise ValueError("atmosphere.chemistry.species labels contain duplicates")
        missing = sorted(set(opacity) - set(chemistry))
        if missing:
            raise ValueError(
                "opacity species missing from chemistry labels: " + ", ".join(missing)
            )
        names = tuple(item.name for item in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        required: set[str] = set()
        stellar_contamination = self.stellar_contamination
        if stellar_contamination is not None:
            if self.radiative_transfer.model != "transmission":
                raise ValueError(
                    "stellar_contamination is only supported for transmission"
                )
            photosphere_temperature = self.bodies.star.effective_temperature_k
            for region in stellar_contamination.regions:
                if region.kind == "spot" and region.temperature_k >= photosphere_temperature:
                    raise ValueError(
                        "stellar spot temperature must be cooler than the photosphere"
                    )
                if region.kind == "facula" and region.temperature_k <= photosphere_temperature:
                    raise ValueError(
                        "stellar facula temperature must be hotter than the photosphere"
                    )
                if region.covering_fraction_parameter is not None:
                    required.add(region.covering_fraction_parameter)
        if chemistry_config.model == "fastchem_equilibrium":
            required.update(
                {
                    chemistry_config.metallicity_parameter,
                    chemistry_config.carbon_to_oxygen_parameter,
                }
            )
            required.update(
                (chemistry_config.constant_log10_vmr_parameters or {}).values()
            )
        else:
            required.update(
                chemistry_config.parameter_names.get(species, species)
                for species in chemistry_config.species
                if species not in chemistry_config.fixed_mixing_ratios
            )
            if chemistry_config.phantom_mean_molecular_weight_parameter is not None:
                required.add(
                    chemistry_config.phantom_mean_molecular_weight_parameter
                )
        temperature = self.atmosphere.temperature
        if temperature.model == "parmentier_guillot_2014":
            required.update({"kappa_IR", "gamma1", "gamma2", "T_irr", "alpha"})
        elif temperature.model == "isothermal" and temperature.temperature_k is None:
            required.add(temperature.parameter_name)
        elif temperature.model == "madhusudhan_seager_2009":
            required.update(
                {
                    temperature.p1_parameter,
                    temperature.p2_parameter,
                    temperature.p3_parameter,
                    temperature.t0_parameter,
                    temperature.alpha1_parameter,
                    temperature.alpha2_parameter,
                }
            )
        elif temperature.model == "spline" and temperature.knot_temperature_k is None:
            required.update(
                temperature.parameter_names
                or tuple(
                    f"temperature_{index}"
                    for index in range(len(temperature.knot_pressure))
                )
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
        if radiative_transfer.model == "transmission":
            if not (
                self.atmosphere.pressure.top_bar
                <= radiative_transfer.reference_pressure_bar
                <= self.atmosphere.pressure.bottom_bar
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
            if chemistry_config.model != "free":
                raise ValueError(
                    "centered_log_ratio priors require free chemistry"
                )
            if chemistry_config.parameter_mode != "log10":
                raise ValueError(
                    "centered_log_ratio priors require chemistry parameter_mode=log10"
                )
            if not chemistry_config.fill_background:
                raise ValueError(
                    "centered_log_ratio priors require a background closure category"
                )
            chemistry_parameters = {
                chemistry_config.parameter_names.get(species, species)
                for species in chemistry_config.species
                if species not in chemistry_config.fixed_mixing_ratios
            }
            if set(clr_parameters) != chemistry_parameters:
                raise ValueError(
                    "all and only retrieved free-chemistry abundances must use "
                    "centered_log_ratio priors"
                )
            groups = {
                prior.group or "composition" for prior in clr_parameters.values()
            }
            if len(groups) != 1:
                raise ValueError(
                    "free-chemistry centered_log_ratio priors must share one group"
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
        if stellar_contamination is not None:
            parameter_configs = {item.name: item for item in self.parameters}
            maximum_fraction = sum(
                region.covering_fraction or 0.0
                for region in stellar_contamination.regions
            )
            for region in stellar_contamination.regions:
                parameter_name = region.covering_fraction_parameter
                if parameter_name is None:
                    continue
                prior = parameter_configs[parameter_name].prior
                if prior.type != "uniform" or prior.lower < 0.0 or prior.upper > 1.0:
                    raise ValueError(
                        "stellar covering-fraction parameters require uniform priors within [0, 1]"
                    )
                maximum_fraction += prior.upper
            if maximum_fraction > 1.0:
                raise ValueError(
                    "stellar covering-fraction prior upper bounds and fixed fractions must sum to at most one"
                )
        clouds = self.clouds
        if clouds.model == "deck_haze":
            cloud_parameters = {
                clouds.log10_cloud_top_pressure_bar_parameter,
                clouds.log10_cloud_optical_depth_parameter,
                clouds.log10_haze_mass_extinction_parameter,
                clouds.haze_slope_parameter,
            }
        elif clouds.model != "none":
            cloud_parameters = {
                clouds.log10_mass_fraction_parameter,
                clouds.log10_radius_micron_parameter,
                clouds.log10_top_pressure_bar_parameter,
                clouds.log10_base_pressure_bar_parameter,
            }
            if clouds.model == "mie_direct_nk":
                cloud_parameters.update(clouds.real_index_parameter_names)
                cloud_parameters.update(clouds.log10_imaginary_index_parameter_names)
        else:
            cloud_parameters = set()
        if cloud_parameters:
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
            section[keys[-1]] = str(
                value if value.is_absolute() else source.parent / value
            )
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
    chemistry = raw.get("atmosphere", {}).get("chemistry", {})
    if (
        isinstance(chemistry, dict)
        and chemistry.get("model") == "fastchem_equilibrium"
        and "fastchem_directory" in housekeeping
    ):
        chemistry.setdefault("fastchem_path", housekeeping["fastchem_directory"])
    clouds = raw.get("clouds")
    if (
        isinstance(clouds, dict)
        and clouds.get("model") == "mie_catalog"
        and "optical_constants_directory" in housekeeping
    ):
        clouds.setdefault(
            "optical_constants_path", housekeeping["optical_constants_directory"]
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
