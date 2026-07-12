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
from robert_exoplanets.core import PressureGrid, RobertConfigError, RobertValidationError, SpectralGrid
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import CorrelatedKOpacityProvider
from robert_exoplanets.rt import CiaTable, DiscGeometry

from .emission import (
    ClearSkyEmissionForwardModel,
    ClearSkyEmissionModelConfig,
    ParameterizedClearSkyEmissionForwardModel,
    ParameterizedClearSkyEmissionModelConfig,
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

        directory = None if self.directory is None else Path(self.directory).expanduser()
        paths = {str(key).strip(): Path(value).expanduser() for key, value in self.paths.items()}
        if (directory is None) == (not paths):
            raise RobertConfigError("provide exactly one of opacity directory or opacity paths")
        if paths and set(paths) != set(species):
            raise RobertConfigError("opacity path keys must match configured species")
        if any(not key for key in paths):
            raise RobertConfigError("opacity path species names must not be empty")
        if not str(self.filename_pattern).strip():
            raise RobertConfigError("opacity filename_pattern must not be empty")
        if not str(self.name).strip():
            raise RobertConfigError("opacity source name must not be empty")
        fill_value = float(self.nonfinite_fill_value)
        zero_delta = float(self.zero_deltalog_min_value)
        if not np.isfinite(fill_value) or fill_value <= 0.0:
            raise RobertConfigError("opacity nonfinite_fill_value must be finite and positive")
        if not np.isfinite(zero_delta) or zero_delta <= 0.0:
            raise RobertConfigError("zero_deltalog_min_value must be finite and positive")

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "paths", immutable_mapping(paths))
        object.__setattr__(self, "nonfinite_fill_value", fill_value)
        object.__setattr__(self, "zero_deltalog_min_value", zero_delta)

    def load(self) -> CorrelatedKOpacityProvider:
        """Load the configured native correlated-k provider."""

        if self.directory is not None:
            return CorrelatedKOpacityProvider.from_exomol_kta_directory(
                self.directory,
                species=self.species,
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
            raise RobertConfigError("zero_deltalog_min_value must be finite and positive")
        object.__setattr__(self, "num", int(self.num))
        object.__setattr__(self, "zero_deltalog_min_value", zero_delta)

    def apply(
        self,
        provider: CorrelatedKOpacityProvider,
        spectral_grid: SpectralGrid,
    ) -> CorrelatedKOpacityProvider:
        """Bin and recompress a provider onto an observed spectral grid."""

        if spectral_grid.bin_edges is None:
            raise RobertConfigError("exo_k spectral binning requires spectral-grid bin edges")
        return provider.bin_to_spectral_grid(
            spectral_grid,
            num=self.num,
            use_rebin=self.use_rebin,
            remove_zeros=self.remove_zeros,
            zero_deltalog_min_value=self.zero_deltalog_min_value,
        )


@dataclass(frozen=True)
class ClearSkyEmissionFactoryConfig:
    """Complete typed Python configuration for a clear-sky emission model.

    ``pressure_grid=None`` selects the native pressure centers of the first
    configured opacity species. Set ``opacity_binning=None`` only when a
    pre-built provider already lies on the requested spectral grid.
    """

    planet: Planet
    star: Star
    temperature_profile: TemperatureProfile
    opacity_source: ExoKOpacitySource | CorrelatedKOpacityProvider
    model: ClearSkyEmissionModelConfig
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

        if not callable(getattr(self.temperature_profile, "evaluate", None)) or not callable(
            getattr(self.temperature_profile, "required_parameters", None)
        ):
            raise RobertConfigError("temperature_profile must implement the TemperatureProfile protocol")
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
            species for species in self.model.opacity_species if species not in source_species
        )
        if missing_species:
            raise RobertConfigError(
                "opacity source is missing model species: " + ", ".join(missing_species)
            )
        object.__setattr__(self, "temperature_parameters", immutable_mapping(parameters))


@dataclass(frozen=True)
class ParameterizedClearSkyEmissionFactoryConfig:
    """Python configuration for runtime P–T and chemistry parameterizations."""

    planet: Planet
    star: Star
    temperature_profile: TemperatureProfile
    chemistry_model: ChemistryModel
    opacity_source: ExoKOpacitySource | CorrelatedKOpacityProvider
    model: ParameterizedClearSkyEmissionModelConfig
    cia_table: CiaTable | tuple[CiaTable, ...] | None = None
    pressure_grid: PressureGrid | None = None
    mean_molecular_weight: float = 2.3
    mean_molecular_weight_model: MeanMolecularWeightModel | None = None
    opacity_binning: ExoKTableBinning | None = field(default_factory=ExoKTableBinning)
    geometry: DiscGeometry | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.temperature_profile, "temperature_profile"),
            (self.chemistry_model, "chemistry_model"),
        ):
            if not callable(getattr(value, "evaluate", None)) or not callable(
                getattr(value, "required_parameters", None)
            ):
                raise RobertConfigError(f"{label} does not implement its ROBERT protocol")
        source_species = self.opacity_source.species
        missing_source = tuple(
            species for species in self.model.opacity_species if species not in source_species
        )
        if missing_source:
            raise RobertConfigError(
                "opacity source is missing model species: " + ", ".join(missing_source)
            )
        missing_chemistry = tuple(
            species for species in self.model.opacity_species if species not in self.chemistry_model.species
        )
        if missing_chemistry:
            raise RobertConfigError(
                "chemistry model is missing opacity species: " + ", ".join(missing_chemistry)
            )
        mmw = float(self.mean_molecular_weight)
        if not np.isfinite(mmw) or mmw <= 0.0:
            raise RobertConfigError("mean_molecular_weight must be finite and positive")
        object.__setattr__(self, "mean_molecular_weight", mmw)


def build_clear_sky_emission_model(
    config: ClearSkyEmissionFactoryConfig,
    *,
    spectral_grid: SpectralGrid,
) -> ClearSkyEmissionForwardModel:
    """Construct and prepare a clear-sky emission model from Python config."""

    if not isinstance(config, ClearSkyEmissionFactoryConfig):
        raise RobertConfigError("config must be a ClearSkyEmissionFactoryConfig")
    native_provider = (
        config.opacity_source.load()
        if isinstance(config.opacity_source, ExoKOpacitySource)
        else config.opacity_source
    )
    pressure_grid = config.pressure_grid or pressure_grid_from_opacity(
        native_provider,
        species=config.model.opacity_species[0],
    )
    provider = (
        native_provider
        if config.opacity_binning is None
        else config.opacity_binning.apply(native_provider, spectral_grid)
    )
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
    return ClearSkyEmissionForwardModel(
        planet=config.planet,
        star=config.star,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        base_temperature_K=base_temperature,
        opacity_provider=provider,
        config=model_config,
        geometry=config.geometry,
    )


def build_parameterized_clear_sky_emission_model(
    config: ParameterizedClearSkyEmissionFactoryConfig,
    *,
    spectral_grid: SpectralGrid,
) -> ParameterizedClearSkyEmissionForwardModel:
    """Construct a prepared model with runtime P–T and chemistry evaluation."""

    if not isinstance(config, ParameterizedClearSkyEmissionFactoryConfig):
        raise RobertConfigError("config must be a ParameterizedClearSkyEmissionFactoryConfig")
    native_provider = (
        config.opacity_source.load()
        if isinstance(config.opacity_source, ExoKOpacitySource)
        else config.opacity_source
    )
    pressure_grid = config.pressure_grid or pressure_grid_from_opacity(
        native_provider,
        species=config.model.opacity_species[0],
    )
    provider = (
        native_provider
        if config.opacity_binning is None
        else config.opacity_binning.apply(native_provider, spectral_grid)
    )
    atmosphere_builder = AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=config.temperature_profile,
        chemistry_model=config.chemistry_model,
        mean_molecular_weight=config.mean_molecular_weight,
        mean_molecular_weight_model=config.mean_molecular_weight_model,
    )
    model_config = replace(
        config.model,
        metadata={
            **dict(config.model.metadata),
            **_parameterized_factory_manifest_metadata(config),
        },
    )
    return ParameterizedClearSkyEmissionForwardModel(
        planet=config.planet,
        star=config.star,
        spectral_grid=spectral_grid,
        atmosphere_builder=atmosphere_builder,
        opacity_provider=provider,
        config=model_config,
        cia_table=config.cia_table,
        geometry=config.geometry,
    )


def _factory_manifest_metadata(config: ClearSkyEmissionFactoryConfig) -> dict[str, str]:
    profile_name = getattr(config.temperature_profile, "name", "")
    metadata = {
        "factory_configuration_interface": "typed_python",
        "factory_temperature_profile_type": type(config.temperature_profile).__qualname__,
        "factory_temperature_profile_name": str(profile_name),
        "factory_temperature_parameters": ",".join(
            f"{name}:{value:.17g}" for name, value in config.temperature_parameters.items()
        ),
        "factory_pressure_grid_source": (
            "explicit" if config.pressure_grid is not None else "opacity_centers"
        ),
    }
    if isinstance(config.opacity_source, ExoKOpacitySource):
        metadata["factory_opacity_source_type"] = (
            "exomol_kta_directory" if config.opacity_source.directory is not None else "exo_k_paths"
        )
        metadata["factory_opacity_directory"] = (
            "" if config.opacity_source.directory is None else str(config.opacity_source.directory)
        )
        metadata["factory_opacity_paths"] = ",".join(
            f"{species}:{path}" for species, path in config.opacity_source.paths.items()
        )
        metadata["factory_opacity_filename_pattern"] = config.opacity_source.filename_pattern
    else:
        metadata["factory_opacity_source_type"] = "provider"
        metadata["factory_opacity_provider"] = config.opacity_source.name
    if config.opacity_binning is None:
        metadata["factory_exo_k_binning"] = "disabled"
    else:
        metadata.update(
            {
                "factory_exo_k_binning": "enabled",
                "factory_exo_k_binning_num": str(config.opacity_binning.num),
                "factory_exo_k_use_rebin": str(config.opacity_binning.use_rebin).lower(),
                "factory_exo_k_remove_zeros": str(config.opacity_binning.remove_zeros).lower(),
                "factory_exo_k_zero_deltalog_min_value": (
                    f"{config.opacity_binning.zero_deltalog_min_value:.17g}"
                ),
            }
        )
    return metadata


def _parameterized_factory_manifest_metadata(
    config: ParameterizedClearSkyEmissionFactoryConfig,
) -> dict[str, str]:
    metadata = {
        "factory_configuration_interface": "typed_python",
        "factory_parameterization": "runtime_temperature_and_chemistry",
        "factory_temperature_profile_type": type(config.temperature_profile).__qualname__,
        "factory_chemistry_model_type": type(config.chemistry_model).__qualname__,
        "factory_pressure_grid_source": (
            "explicit" if config.pressure_grid is not None else "opacity_centers"
        ),
    }
    if isinstance(config.opacity_source, ExoKOpacitySource):
        metadata["factory_opacity_source_type"] = (
            "exomol_kta_directory" if config.opacity_source.directory is not None else "exo_k_paths"
        )
        metadata["factory_opacity_directory"] = (
            "" if config.opacity_source.directory is None else str(config.opacity_source.directory)
        )
        metadata["factory_opacity_filename_pattern"] = config.opacity_source.filename_pattern
    else:
        metadata["factory_opacity_source_type"] = "provider"
        metadata["factory_opacity_provider"] = config.opacity_source.name
    metadata["factory_exo_k_binning"] = (
        "disabled" if config.opacity_binning is None else "enabled"
    )
    if config.opacity_binning is not None:
        metadata["factory_exo_k_binning_num"] = str(config.opacity_binning.num)
    return metadata


def pressure_grid_from_opacity(
    provider: CorrelatedKOpacityProvider,
    *,
    species: str | None = None,
    name: str | None = None,
) -> PressureGrid:
    """Construct layer edges around one opacity table's pressure centers."""

    selected_species = provider.species[0] if species is None else str(species)
    try:
        table = provider.tables[selected_species]
    except KeyError as exc:
        raise RobertConfigError(
            f"cannot derive pressure grid: opacity species {selected_species!r} is unavailable"
        ) from exc
    centers = np.asarray(table.pressure_bar, dtype=float)
    if centers.size < 2:
        raise RobertValidationError(
            "at least two opacity pressure points are required to infer layer edges; "
            "provide pressure_grid explicitly"
        )
    if np.any(centers <= 0.0) or not (
        np.all(np.diff(centers) > 0.0) or np.all(np.diff(centers) < 0.0)
    ):
        raise RobertValidationError("opacity pressure centers must be positive and monotonic")
    log_centers = np.log(centers)
    inner_edges = 0.5 * (log_centers[:-1] + log_centers[1:])
    first_edge = log_centers[0] - (inner_edges[0] - log_centers[0])
    last_edge = log_centers[-1] + (log_centers[-1] - inner_edges[-1])
    edges = np.exp(np.concatenate(([first_edge], inner_edges, [last_edge])))
    return PressureGrid(
        edges=edges,
        centers=centers,
        unit="bar",
        name=name or f"{selected_species} opacity pressure grid",
        metadata={"source": "correlated-k opacity", "species": selected_species},
    )


__all__ = [
    "ClearSkyEmissionFactoryConfig",
    "ExoKOpacitySource",
    "ExoKTableBinning",
    "ParameterizedClearSkyEmissionFactoryConfig",
    "build_clear_sky_emission_model",
    "build_parameterized_clear_sky_emission_model",
    "pressure_grid_from_opacity",
]
