"""Typed Python configuration and factories for emission forward models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from robert_exoplanets.atmosphere import (
    AtmosphereBuilder,
    ChemistryModel,
    MeanMolecularWeightModel,
    TemperatureProfile,
)
from robert_exoplanets.bodies import Planet, Star
from robert_exoplanets.core import (
    PressureGrid,
    RobertConfigError,
    SpectralGrid,
)
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import (
    CorrelatedKOpacityProvider,
    LineByLineOpacityProvider,
    OpacityProvider,
    OpacitySamplingProvider,
)
from robert_exoplanets.rt import CiaTable, DiscGeometry

from ._construction import prepare_opacity_inputs, pressure_grid_from_opacity
from .emission import (
    EmissionForwardModel,
    EmissionModelConfig,
    ParameterizedEmissionForwardModel,
    ParameterizedEmissionModelConfig,
)
from .clouds import ParameterizedCloudModel
from .multi_dataset import MultiDatasetEmissionForwardModel, PreparedSpectrumResponse
from .transmission import (
    ParameterizedTransmissionForwardModel,
    ParameterizedTransmissionModelConfig,
)


@dataclass(frozen=True)
class ExoKOpacitySource:
    """Python configuration for molecular correlated-k opacity inputs.

    Use ``directory`` to discover ExoMol ``.kta`` files by filename, or use
    ``paths`` to select any individual table format supported by ``exo_k``.
    Exactly one source style must be supplied.
    """

    species: tuple[str, ...]
    directory: str | Path | None = None
    paths: Mapping[str, str | Path] = field(default_factory=dict)
    resolution: str | int | None = None
    filename_pattern: str = "*.kta"
    name: str = "exomol-correlated-k"
    interpolation: str = "log_pressure_temperature_log_k"
    nonfinite_policy: str = "floor"
    nonfinite_fill_value: float = 1.0e-300
    remove_zeros: bool = True
    zero_deltalog_min_value: float = 10.0

    def __post_init__(self) -> None:
        species = tuple(str(item).strip() for item in self.species)
        if not species or any(not item for item in species):
            raise RobertConfigError("opacity species must contain non-empty names")
        if len({item.casefold() for item in species}) != len(species):
            raise RobertConfigError("opacity species must not contain duplicates")

        directory = (
            None if self.directory is None else Path(self.directory).expanduser()
        )
        paths = {
            str(key).strip(): Path(value).expanduser()
            for key, value in self.paths.items()
        }
        if (directory is None) == (not paths):
            raise RobertConfigError(
                "provide exactly one of opacity directory or opacity paths"
            )
        if paths and set(paths) != set(species):
            raise RobertConfigError("opacity path keys must match configured species")
        if any(not key for key in paths):
            raise RobertConfigError("opacity path species names must not be empty")
        resolution = None
        if self.resolution is not None:
            value = str(self.resolution).strip().upper()
            if value.startswith("R"):
                value = value[1:]
            if not value.isdigit() or int(value) <= 0:
                raise RobertConfigError(
                    "opacity resolution must be a positive integer or 'R<integer>'"
                )
            resolution = f"R{int(value)}"
        if resolution is not None and directory is None:
            raise RobertConfigError(
                "opacity resolution is only valid with an opacity directory"
            )
        if not str(self.filename_pattern).strip():
            raise RobertConfigError("opacity filename_pattern must not be empty")
        if not str(self.name).strip():
            raise RobertConfigError("opacity source name must not be empty")
        fill_value = float(self.nonfinite_fill_value)
        zero_delta = float(self.zero_deltalog_min_value)
        if not np.isfinite(fill_value) or fill_value <= 0.0:
            raise RobertConfigError(
                "opacity nonfinite_fill_value must be finite and positive"
            )
        if not np.isfinite(zero_delta) or zero_delta <= 0.0:
            raise RobertConfigError(
                "zero_deltalog_min_value must be finite and positive"
            )

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "paths", immutable_mapping(paths))
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "nonfinite_fill_value", fill_value)
        object.__setattr__(self, "zero_deltalog_min_value", zero_delta)

    def load(self) -> CorrelatedKOpacityProvider:
        """Load the configured native correlated-k provider."""

        if self.directory is not None:
            return CorrelatedKOpacityProvider.from_exomol_kta_directory(
                self.directory,
                species=self.species,
                resolution=self.resolution,
                filename_pattern=self.filename_pattern,
                name=self.name,
                interpolation=self.interpolation,
                nonfinite_policy=self.nonfinite_policy,
                nonfinite_fill_value=self.nonfinite_fill_value,
            )
        return CorrelatedKOpacityProvider.from_exok_paths(
            self.paths,
            name=self.name,
            interpolation=self.interpolation,
            nonfinite_policy=self.nonfinite_policy,
            nonfinite_fill_value=self.nonfinite_fill_value,
            remove_zeros=self.remove_zeros,
            zero_deltalog_min_value=self.zero_deltalog_min_value,
        )


@dataclass(frozen=True)
class ExoMolOpacitySamplingSource:
    """Python configuration for ExoMolOP cross-section opacity sampling."""

    species: tuple[str, ...]
    paths: Mapping[str, str | Path]
    name: str = "exomol-opacity-sampling"
    interpolation: str = "log_pressure_temperature_log_xsec"
    checksum: bool = True

    def __post_init__(self) -> None:
        species = tuple(str(item).strip() for item in self.species)
        paths = {
            str(key).strip(): Path(value).expanduser()
            for key, value in self.paths.items()
        }
        if not species or any(not item for item in species):
            raise RobertConfigError("opacity species must contain non-empty names")
        if len(set(species)) != len(species):
            raise RobertConfigError("opacity species must not contain duplicates")
        if set(paths) != set(species):
            raise RobertConfigError("opacity path keys must match configured species")
        if not str(self.name).strip():
            raise RobertConfigError("opacity source name must not be empty")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "paths", immutable_mapping(paths))

    def load(self) -> OpacitySamplingProvider:
        """Load the configured ExoMol opacity-sampling provider."""

        return OpacitySamplingProvider.from_exomol_paths(
            self.paths,
            name=self.name,
            interpolation=self.interpolation,
            checksum=self.checksum,
        )


@dataclass(frozen=True)
class LineByLineOpacitySource:
    """Typed source configuration for native line-by-line opacity.

    The source keeps the physical wavelength samples.  ``wavelength_bounds``
    is an explicit safety window for normal forward-model preparation.  The
    optional ``native_sampling_stride`` retains every ``stride``-th native
    sample.  This is a speed option only.  It is marked as requiring a later
    convergence test at the final observation resolution.
    """

    species: tuple[str, ...]
    paths: Mapping[str, str | Path]
    name: str = "line-by-line-native"
    interpolation: str = "log_pressure_temperature_log_xsec"
    checksum: bool = True
    cross_section_floor: float = 1.0e-300
    max_memory_bytes: int | None = 1 * 1024**3
    wavelength_bounds_micron: tuple[float, float] | None = None
    native_sampling_stride: int = 1
    max_cached_slices: int = 2

    def __post_init__(self) -> None:
        species = tuple(str(item).strip() for item in self.species)
        paths = {
            str(key).strip(): Path(value).expanduser()
            for key, value in self.paths.items()
        }
        if not species or any(not item for item in species):
            raise RobertConfigError("line-by-line species must contain non-empty names")
        if len(set(species)) != len(species):
            raise RobertConfigError("line-by-line species must not contain duplicates")
        if set(paths) != set(species):
            raise RobertConfigError(
                "line-by-line opacity path keys must match configured species"
            )
        if any(not key for key in paths):
            raise RobertConfigError("line-by-line opacity path names must not be empty")
        if not str(self.name).strip():
            raise RobertConfigError("line-by-line opacity source name must not be empty")
        if self.interpolation not in {
            "log_pressure_temperature_log_xsec",
            "log_pressure_temperature_log_xsec_clip",
        }:
            raise RobertConfigError(
                "line-by-line interpolation must be log-pressure/temperature/log-xsec"
            )
        floor = float(self.cross_section_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise RobertConfigError(
                "line-by-line cross_section_floor must be finite and positive"
            )
        if self.max_memory_bytes is not None:
            if isinstance(self.max_memory_bytes, bool) or not isinstance(
                self.max_memory_bytes, (int, np.integer)
            ) or int(self.max_memory_bytes) < 1:
                raise RobertConfigError(
                    "line-by-line max_memory_bytes must be a positive integer or None"
                )
            max_memory_bytes = int(self.max_memory_bytes)
        else:
            max_memory_bytes = None
        if isinstance(self.native_sampling_stride, bool) or not isinstance(
            self.native_sampling_stride, (int, np.integer)
        ) or int(self.native_sampling_stride) < 1:
            raise RobertConfigError(
                "line-by-line native_sampling_stride must be a positive integer"
            )
        stride = int(self.native_sampling_stride)
        if isinstance(self.max_cached_slices, bool) or not isinstance(
            self.max_cached_slices, (int, np.integer)
        ) or int(self.max_cached_slices) < 0:
            raise RobertConfigError(
                "line-by-line max_cached_slices must be a non-negative integer"
            )
        bounds = _validate_line_by_line_window(self.wavelength_bounds_micron)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "paths", immutable_mapping(paths))
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "cross_section_floor", floor)
        object.__setattr__(self, "max_memory_bytes", max_memory_bytes)
        object.__setattr__(self, "wavelength_bounds_micron", bounds)
        object.__setattr__(self, "native_sampling_stride", stride)
        object.__setattr__(self, "max_cached_slices", int(self.max_cached_slices))

    def load(self) -> LineByLineOpacityProvider:
        """Load the configured HDF5 or NPZ line-by-line provider."""

        return LineByLineOpacityProvider.from_paths(
            self.paths,
            name=self.name,
            interpolation=self.interpolation,
            checksum=self.checksum,
            cross_section_floor=self.cross_section_floor,
            max_memory_bytes=self.max_memory_bytes,
            wavelength_bounds_micron=self.wavelength_bounds_micron,
            native_sampling_stride=self.native_sampling_stride,
            max_cached_slices=self.max_cached_slices,
        )

    def native_spectral_grid(self) -> SpectralGrid:
        """Return the configured exact native wavelength grid."""

        return self.load().native_spectral_grid()


@dataclass(frozen=True)
class ExoKTableBinning:
    """Configuration for exo_k correlated-k binning and recompression."""

    num: int = 300
    use_rebin: bool = False
    remove_zeros: bool = True
    zero_deltalog_min_value: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.num, bool) or int(self.num) != self.num or int(self.num) < 1:
            raise RobertConfigError("exo_k binning num must be a positive integer")
        zero_delta = float(self.zero_deltalog_min_value)
        if not np.isfinite(zero_delta) or zero_delta <= 0.0:
            raise RobertConfigError(
                "zero_deltalog_min_value must be finite and positive"
            )
        object.__setattr__(self, "num", int(self.num))
        object.__setattr__(self, "zero_deltalog_min_value", zero_delta)

    def apply(
        self,
        provider: CorrelatedKOpacityProvider,
        spectral_grid: SpectralGrid,
    ) -> CorrelatedKOpacityProvider:
        """Bin and recompress a provider onto an observed spectral grid."""

        if spectral_grid.bin_edges is None:
            raise RobertConfigError(
                "exo_k spectral binning requires spectral-grid bin edges"
            )
        return provider.bin_to_spectral_grid(
            spectral_grid,
            num=self.num,
            use_rebin=self.use_rebin,
            remove_zeros=self.remove_zeros,
            zero_deltalog_min_value=self.zero_deltalog_min_value,
        )


@dataclass(frozen=True)
class EmissionFactoryConfig:
    """Complete typed Python configuration for a cloud-free emission model.

    ``pressure_grid=None`` selects the native pressure centers of the first
    configured opacity species. Set ``opacity_binning=None`` only when a
    pre-built provider already lies on the requested spectral grid.
    """

    planet: Planet
    star: Star
    temperature_profile: TemperatureProfile
    opacity_source: (
        ExoKOpacitySource
        | ExoMolOpacitySamplingSource
        | LineByLineOpacitySource
        | OpacityProvider
    )
    model: EmissionModelConfig
    pressure_grid: PressureGrid | None = None
    temperature_parameters: Mapping[str, float] = field(default_factory=dict)
    opacity_binning: ExoKTableBinning | None = field(default_factory=ExoKTableBinning)
    geometry: DiscGeometry | None = None

    def __post_init__(self) -> None:
        parameters: dict[str, float] = {}
        for name, value in self.temperature_parameters.items():
            normalized_name = str(name).strip()
            if not normalized_name:
                raise RobertConfigError("temperature parameter names must not be empty")
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise RobertConfigError("temperature parameters must be finite")
            parameters[normalized_name] = numeric_value

        if not callable(
            getattr(self.temperature_profile, "evaluate", None)
        ) or not callable(
            getattr(self.temperature_profile, "required_parameters", None)
        ):
            raise RobertConfigError(
                "temperature_profile must implement the TemperatureProfile protocol"
            )
        required = tuple(self.temperature_profile.required_parameters())
        missing = tuple(name for name in required if name not in parameters)
        unexpected = tuple(name for name in parameters if name not in required)
        if missing:
            raise RobertConfigError(
                f"temperature parameters are missing: {', '.join(missing)}"
            )
        if unexpected:
            raise RobertConfigError(
                f"unexpected temperature parameters: {', '.join(unexpected)}"
            )

        source_species = self.opacity_source.species
        missing_species = tuple(
            species
            for species in self.model.opacity_species
            if species not in source_species
        )
        if missing_species:
            raise RobertConfigError(
                "opacity source is missing model species: " + ", ".join(missing_species)
            )
        object.__setattr__(
            self, "temperature_parameters", immutable_mapping(parameters)
        )


@dataclass(frozen=True)
class ParameterizedEmissionFactoryConfig:
    """Python configuration for runtime P–T and chemistry parameterizations."""

    planet: Planet
    star: Star
    temperature_profile: TemperatureProfile
    chemistry_model: ChemistryModel
    opacity_source: (
        ExoKOpacitySource
        | ExoMolOpacitySamplingSource
        | LineByLineOpacitySource
        | OpacityProvider
    )
    model: ParameterizedEmissionModelConfig
    cia_table: CiaTable | tuple[CiaTable, ...] | None = None
    pressure_grid: PressureGrid | None = None
    mean_molecular_weight: float = 2.3
    mean_molecular_weight_model: MeanMolecularWeightModel | None = None
    opacity_free_species: tuple[str, ...] = ()
    opacity_binning: ExoKTableBinning | None = field(default_factory=ExoKTableBinning)
    geometry: DiscGeometry | None = None
    cloud_model: ParameterizedCloudModel | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.temperature_profile, "temperature_profile"),
            (self.chemistry_model, "chemistry_model"),
        ):
            if not callable(getattr(value, "evaluate", None)) or not callable(
                getattr(value, "required_parameters", None)
            ):
                raise RobertConfigError(
                    f"{label} does not implement its ROBERT protocol"
                )
        source_species = self.opacity_source.species
        missing_source = tuple(
            species
            for species in self.model.opacity_species
            if species not in source_species
        )
        if missing_source:
            raise RobertConfigError(
                "opacity source is missing model species: " + ", ".join(missing_source)
            )
        missing_chemistry = tuple(
            species
            for species in self.model.opacity_species
            if species not in self.chemistry_model.species
        )
        if missing_chemistry:
            raise RobertConfigError(
                "chemistry model is missing opacity species: "
                + ", ".join(missing_chemistry)
            )
        mmw = float(self.mean_molecular_weight)
        if not np.isfinite(mmw) or mmw <= 0.0:
            raise RobertConfigError("mean_molecular_weight must be finite and positive")
        object.__setattr__(self, "mean_molecular_weight", mmw)


@dataclass(frozen=True)
class ParameterizedTransmissionFactoryConfig:
    """Python configuration for runtime transmission parameterizations."""

    planet: Planet
    star: Star
    temperature_profile: TemperatureProfile
    chemistry_model: ChemistryModel
    opacity_source: (
        ExoKOpacitySource
        | ExoMolOpacitySamplingSource
        | LineByLineOpacitySource
        | OpacityProvider
    )
    model: ParameterizedTransmissionModelConfig
    cia_table: CiaTable | tuple[CiaTable, ...] | None = None
    pressure_grid: PressureGrid | None = None
    mean_molecular_weight: float = 2.3
    mean_molecular_weight_model: MeanMolecularWeightModel | None = None
    opacity_free_species: tuple[str, ...] = ()
    opacity_binning: ExoKTableBinning | None = field(default_factory=ExoKTableBinning)
    cloud_model: ParameterizedCloudModel | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.temperature_profile, "temperature_profile"),
            (self.chemistry_model, "chemistry_model"),
        ):
            if not callable(getattr(value, "evaluate", None)) or not callable(
                getattr(value, "required_parameters", None)
            ):
                raise RobertConfigError(
                    f"{label} does not implement its ROBERT protocol"
                )
        source_species = self.opacity_source.species
        missing_source = tuple(
            species
            for species in self.model.opacity_species
            if species not in source_species
        )
        if missing_source:
            raise RobertConfigError(
                "opacity source is missing model species: "
                + ", ".join(missing_source)
            )
        missing_chemistry = tuple(
            species
            for species in self.model.opacity_species
            if species not in self.chemistry_model.species
        )
        if missing_chemistry:
            raise RobertConfigError(
                "chemistry model is missing opacity species: "
                + ", ".join(missing_chemistry)
            )
        mmw = float(self.mean_molecular_weight)
        if not np.isfinite(mmw) or mmw <= 0.0:
            raise RobertConfigError(
                "mean_molecular_weight must be finite and positive"
            )
        object.__setattr__(self, "mean_molecular_weight", mmw)


def _validate_line_by_line_window(
    value: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RobertConfigError(
            "line-by-line wavelength_bounds_micron must contain two finite, "
            "positive, increasing values"
        ) from exc
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or lower >= upper
    ):
        raise RobertConfigError(
            "line-by-line wavelength_bounds_micron must contain two finite, "
            "positive, increasing values"
        )
    return (lower, upper)


def _uses_exok_binning(config: object) -> bool:
    source = getattr(config, "opacity_source")
    binning = getattr(config, "opacity_binning")
    return binning is not None and not isinstance(
        source,
        (
            ExoMolOpacitySamplingSource,
            LineByLineOpacitySource,
            OpacitySamplingProvider,
            LineByLineOpacityProvider,
        ),
    )


def build_emission_model(
    config: EmissionFactoryConfig,
    *,
    spectral_grid: SpectralGrid,
) -> EmissionForwardModel:
    """Construct and prepare a cloud-free emission model from Python config."""

    if not isinstance(config, EmissionFactoryConfig):
        raise RobertConfigError("config must be a EmissionFactoryConfig")
    prepared = prepare_opacity_inputs(
        config.opacity_source,
        spectral_grid=spectral_grid,
        pressure_grid=config.pressure_grid,
        species=config.model.opacity_species[0],
        opacity_binning=config.opacity_binning,
    )
    pressure_grid = prepared.pressure_grid
    provider = prepared.provider
    base_temperature = config.temperature_profile.evaluate(
        config.temperature_parameters,
        pressure_grid,
    )
    model_config = replace(
        config.model,
        metadata={
            **dict(config.model.metadata),
            **_factory_manifest_metadata(config),
        },
    )
    return EmissionForwardModel(
        planet=config.planet,
        star=config.star,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        base_temperature_K=base_temperature,
        opacity_provider=provider,
        config=model_config,
        geometry=config.geometry,
    )


def _build_atmosphere_builder(
    config: ParameterizedEmissionFactoryConfig
    | ParameterizedTransmissionFactoryConfig,
    pressure_grid: PressureGrid,
) -> AtmosphereBuilder:
    """Assemble the shared runtime atmosphere builder for a prepared grid."""

    return AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=config.temperature_profile,
        chemistry_model=config.chemistry_model,
        mean_molecular_weight=config.mean_molecular_weight,
        mean_molecular_weight_model=config.mean_molecular_weight_model,
        opacity_free_species=config.opacity_free_species,
    )


def build_parameterized_emission_model(
    config: ParameterizedEmissionFactoryConfig,
    *,
    spectral_grid: SpectralGrid,
) -> ParameterizedEmissionForwardModel:
    """Construct a prepared model with runtime P–T and chemistry evaluation."""

    if not isinstance(config, ParameterizedEmissionFactoryConfig):
        raise RobertConfigError(
            "config must be a ParameterizedEmissionFactoryConfig"
        )
    prepared = prepare_opacity_inputs(
        config.opacity_source,
        spectral_grid=spectral_grid,
        pressure_grid=config.pressure_grid,
        species=config.model.opacity_species[0],
        opacity_binning=config.opacity_binning,
    )
    pressure_grid = prepared.pressure_grid
    provider = prepared.provider
    atmosphere_builder = _build_atmosphere_builder(config, pressure_grid)
    model_config = replace(
        config.model,
        metadata={
            **dict(config.model.metadata),
            **_parameterized_factory_manifest_metadata(config),
        },
    )
    return ParameterizedEmissionForwardModel(
        planet=config.planet,
        star=config.star,
        spectral_grid=spectral_grid,
        atmosphere_builder=atmosphere_builder,
        opacity_provider=provider,
        config=model_config,
        cia_table=config.cia_table,
        geometry=config.geometry,
        cloud_model=config.cloud_model,
    )


def build_parameterized_transmission_model(
    config: ParameterizedTransmissionFactoryConfig,
    *,
    spectral_grid: SpectralGrid,
) -> ParameterizedTransmissionForwardModel:
    """Construct a prepared model with runtime transmission physics."""

    if not isinstance(config, ParameterizedTransmissionFactoryConfig):
        raise RobertConfigError(
            "config must be a ParameterizedTransmissionFactoryConfig"
        )
    prepared = prepare_opacity_inputs(
        config.opacity_source,
        spectral_grid=spectral_grid,
        pressure_grid=config.pressure_grid,
        species=config.model.opacity_species[0],
        opacity_binning=config.opacity_binning,
    )
    pressure_grid = prepared.pressure_grid
    provider = prepared.provider
    atmosphere_builder = _build_atmosphere_builder(config, pressure_grid)
    model_config = replace(
        config.model,
        metadata={
            **dict(config.model.metadata),
            **_parameterized_factory_manifest_metadata(config),
        },
    )
    return ParameterizedTransmissionForwardModel(
        planet=config.planet,
        star=config.star,
        spectral_grid=spectral_grid,
        atmosphere_builder=atmosphere_builder,
        opacity_provider=provider,
        config=model_config,
        cia_table=config.cia_table,
        cloud_model=config.cloud_model,
    )


def build_multi_dataset_emission_model(
    configs: Mapping[str, ParameterizedEmissionFactoryConfig],
    *,
    spectral_grids: Mapping[str, SpectralGrid],
    responses: Mapping[str, PreparedSpectrumResponse] | None = None,
) -> MultiDatasetEmissionForwardModel:
    """Build the default shared-atmosphere multi-instrument emission model.

    Atmospheric components must be the same objects in every mode config.
    Opacity providers and spectral grids remain mode-specific.
    """

    named_configs = {str(name): config for name, config in configs.items()}
    named_grids = {str(name): grid for name, grid in spectral_grids.items()}
    if not named_configs or any(not name for name in named_configs):
        raise RobertConfigError(
            "multi-dataset emission configs require non-empty names"
        )
    if set(named_configs) != set(named_grids):
        raise RobertConfigError(
            "multi-dataset emission config and spectral-grid names must match"
        )
    reference = next(iter(named_configs.values()))
    for name, config in named_configs.items():
        _validate_shared_atmosphere_config(reference, config, name=name)

    models: dict[str, ParameterizedEmissionForwardModel] = {}
    shared_builder = None
    for name, config in named_configs.items():
        model = build_parameterized_emission_model(
            config,
            spectral_grid=named_grids[name],
        )
        if shared_builder is None:
            shared_builder = model.atmosphere_builder
        else:
            model = replace(model, atmosphere_builder=shared_builder)
        models[name] = model
    return MultiDatasetEmissionForwardModel(
        models,
        responses={} if responses is None else responses,
    )


def _validate_shared_atmosphere_config(
    reference: ParameterizedEmissionFactoryConfig,
    candidate: ParameterizedEmissionFactoryConfig,
    *,
    name: str,
) -> None:
    shared_objects = (
        (
            "temperature_profile",
            reference.temperature_profile,
            candidate.temperature_profile,
        ),
        ("chemistry_model", reference.chemistry_model, candidate.chemistry_model),
        (
            "mean_molecular_weight_model",
            reference.mean_molecular_weight_model,
            candidate.mean_molecular_weight_model,
        ),
        ("pressure_grid", reference.pressure_grid, candidate.pressure_grid),
        ("cia_table", reference.cia_table, candidate.cia_table),
        ("geometry", reference.geometry, candidate.geometry),
        ("cloud_model", reference.cloud_model, candidate.cloud_model),
    )
    different = [label for label, left, right in shared_objects if left is not right]
    if different:
        raise RobertConfigError(
            f"multi-dataset mode {name!r} must share atmospheric objects: "
            + ", ".join(different)
        )
    if candidate.planet != reference.planet or candidate.star != reference.star:
        raise RobertConfigError(
            f"multi-dataset mode {name!r} must use the same planet and star"
        )
    if candidate.mean_molecular_weight != reference.mean_molecular_weight:
        raise RobertConfigError(
            f"multi-dataset mode {name!r} must use the same mean molecular weight"
        )
    if candidate.model != reference.model:
        raise RobertConfigError(
            f"multi-dataset mode {name!r} must use the same emission model configuration"
        )


def _factory_manifest_metadata(config: EmissionFactoryConfig) -> dict[str, str]:
    profile_name = getattr(config.temperature_profile, "name", "")
    metadata = {
        "factory_configuration_interface": "typed_python",
        "factory_temperature_profile_type": type(
            config.temperature_profile
        ).__qualname__,
        "factory_temperature_profile_name": str(profile_name),
        "factory_temperature_parameters": ",".join(
            f"{name}:{value:.17g}"
            for name, value in config.temperature_parameters.items()
        ),
        "factory_pressure_grid_source": (
            "explicit" if config.pressure_grid is not None else "opacity_centers"
        ),
    }
    if isinstance(config.opacity_source, ExoKOpacitySource):
        metadata["factory_opacity_source_type"] = (
            "exomol_kta_directory"
            if config.opacity_source.directory is not None
            else "exo_k_paths"
        )
        metadata["factory_opacity_directory"] = (
            ""
            if config.opacity_source.directory is None
            else str(config.opacity_source.directory)
        )
        metadata["factory_opacity_paths"] = ",".join(
            f"{species}:{path}" for species, path in config.opacity_source.paths.items()
        )
        metadata["factory_opacity_filename_pattern"] = (
            config.opacity_source.filename_pattern
        )
        metadata["factory_opacity_resolution"] = (
            "" if config.opacity_source.resolution is None else config.opacity_source.resolution
        )
    elif isinstance(config.opacity_source, ExoMolOpacitySamplingSource):
        metadata["factory_opacity_source_type"] = "exomol_opacity_sampling"
        metadata["factory_opacity_paths"] = ",".join(
            f"{species}:{path}" for species, path in config.opacity_source.paths.items()
        )
    elif isinstance(config.opacity_source, LineByLineOpacitySource):
        metadata["factory_opacity_source_type"] = "line_by_line"
        metadata["factory_opacity_paths"] = ",".join(
            f"{species}:{path}" for species, path in config.opacity_source.paths.items()
        )
        metadata["factory_opacity_wavelength_window_micron"] = (
            ""
            if config.opacity_source.wavelength_bounds_micron is None
            else ":".join(
                f"{value:.17g}"
                for value in config.opacity_source.wavelength_bounds_micron
            )
        )
        metadata["factory_opacity_native_sampling_stride"] = str(
            config.opacity_source.native_sampling_stride
        )
        metadata["factory_opacity_native_sampling_accuracy"] = (
            "requires_convergence_validation"
        )
        metadata["factory_opacity_native_sampling_validation"] = "required"
        metadata["factory_opacity_max_cached_slices"] = str(
            config.opacity_source.max_cached_slices
        )
    else:
        metadata["factory_opacity_source_type"] = "provider"
        metadata["factory_opacity_provider"] = config.opacity_source.name
    if not _uses_exok_binning(config):
        metadata["factory_exo_k_binning"] = "disabled"
    else:
        metadata.update(
            {
                "factory_exo_k_binning": "enabled",
                "factory_exo_k_binning_num": str(config.opacity_binning.num),
                "factory_exo_k_use_rebin": str(
                    config.opacity_binning.use_rebin
                ).lower(),
                "factory_exo_k_remove_zeros": str(
                    config.opacity_binning.remove_zeros
                ).lower(),
                "factory_exo_k_zero_deltalog_min_value": (
                    f"{config.opacity_binning.zero_deltalog_min_value:.17g}"
                ),
            }
        )
    return metadata


def _parameterized_factory_manifest_metadata(
    config: ParameterizedEmissionFactoryConfig
    | ParameterizedTransmissionFactoryConfig,
) -> dict[str, str]:
    metadata = {
        "factory_configuration_interface": "typed_python",
        "factory_parameterization": "runtime_temperature_and_chemistry",
        "factory_temperature_profile_type": type(
            config.temperature_profile
        ).__qualname__,
        "factory_chemistry_model_type": type(config.chemistry_model).__qualname__,
        "factory_pressure_grid_source": (
            "explicit" if config.pressure_grid is not None else "opacity_centers"
        ),
        "factory_opacity_free_species": ",".join(config.opacity_free_species),
    }
    if isinstance(config.opacity_source, ExoKOpacitySource):
        metadata["factory_opacity_source_type"] = (
            "exomol_kta_directory"
            if config.opacity_source.directory is not None
            else "exo_k_paths"
        )
        metadata["factory_opacity_directory"] = (
            ""
            if config.opacity_source.directory is None
            else str(config.opacity_source.directory)
        )
        metadata["factory_opacity_filename_pattern"] = (
            config.opacity_source.filename_pattern
        )
        metadata["factory_opacity_resolution"] = (
            "" if config.opacity_source.resolution is None else config.opacity_source.resolution
        )
    elif isinstance(config.opacity_source, ExoMolOpacitySamplingSource):
        metadata["factory_opacity_source_type"] = "exomol_opacity_sampling"
        metadata["factory_opacity_paths"] = ",".join(
            f"{species}:{path}" for species, path in config.opacity_source.paths.items()
        )
    elif isinstance(config.opacity_source, LineByLineOpacitySource):
        metadata["factory_opacity_source_type"] = "line_by_line"
        metadata["factory_opacity_paths"] = ",".join(
            f"{species}:{path}" for species, path in config.opacity_source.paths.items()
        )
        metadata["factory_opacity_wavelength_window_micron"] = (
            ""
            if config.opacity_source.wavelength_bounds_micron is None
            else ":".join(
                f"{value:.17g}"
                for value in config.opacity_source.wavelength_bounds_micron
            )
        )
        metadata["factory_opacity_native_sampling_stride"] = str(
            config.opacity_source.native_sampling_stride
        )
        metadata["factory_opacity_native_sampling_accuracy"] = (
            "requires_convergence_validation"
        )
        metadata["factory_opacity_native_sampling_validation"] = "required"
        metadata["factory_opacity_max_cached_slices"] = str(
            config.opacity_source.max_cached_slices
        )
    else:
        metadata["factory_opacity_source_type"] = "provider"
        metadata["factory_opacity_provider"] = config.opacity_source.name
    uses_binning = _uses_exok_binning(config)
    metadata["factory_exo_k_binning"] = "enabled" if uses_binning else "disabled"
    if uses_binning:
        metadata["factory_exo_k_binning_num"] = str(config.opacity_binning.num)
    return metadata


__all__ = [
    "EmissionFactoryConfig",
    "ExoKOpacitySource",
    "ExoKTableBinning",
    "LineByLineOpacitySource",
    "ParameterizedEmissionFactoryConfig",
    "build_emission_model",
    "build_multi_dataset_emission_model",
    "build_parameterized_emission_model",
    "pressure_grid_from_opacity",
]
