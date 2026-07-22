"""Build and run ROBERT tasks from the strict user-facing configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
import os
from pathlib import Path
import shutil
import time
from typing import Mapping

import numpy as np
import yaml

from robert_exoplanets.atmosphere import (
    BackgroundGasMixture,
    CompositionMeanMolecularWeight,
    FastChemEquilibriumChemistry,
    FreeChemistry,
    IsothermalTemperatureProfile,
    MadhusudhanSeager2009TemperatureProfile,
    ParmentierGuillot2014TemperatureProfile,
    SplineTemperatureProfile,
    TabulatedTemperatureProfile,
)
from robert_exoplanets.bodies import Planet, Star
from robert_exoplanets.core import PressureGrid, RobertConfigError, SpectralGrid
from robert_exoplanets.forward import (
    DilutedEmissionModel,
    MultiDatasetDilutedEmissionModel,
    MultiDatasetTwoRegionEmissionModel,
    ParameterizedEmissionFactoryConfig,
    ParameterizedEmissionModelConfig,
    ParameterizedDeckHazeCloudModel,
    ParameterizedMieCloudModel,
    ParameterizedTransmissionFactoryConfig,
    ParameterizedTransmissionModelConfig,
    TwoRegionEmissionModel,
    build_multi_dataset_emission_model,
    build_parameterized_emission_model,
    build_parameterized_transmission_model,
)
from robert_exoplanets.instruments import ObservationCollection, ObservationDataset
from robert_exoplanets.likelihoods import MultiDatasetGaussianLikelihood
from robert_exoplanets.opacity import CorrelatedKOpacityProvider, CorrelatedKTable
from robert_exoplanets.retrieval import (
    CenteredLogRatioPrior,
    LogUniformPrior,
    MultiDatasetRetrievalProblem,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
    load_observation_npz,
    log_pressure_correlated_covariance,
    run_oe_then_nested_sampling,
    run_retrieval,
)
from robert_exoplanets.retrieval.manifest import build_run_manifest
from robert_exoplanets.rt import (
    gauss_legendre_disk_geometry,
    load_nemesispy_cia_table,
    normal_emission_geometry,
    OpticalConstantsCatalog,
)

from .task_config import (
    CloudsConfig,
    ChemistryConfig,
    ParameterConfig,
    ResolvedRegionConfig,
    TaskConfig,
    TemperatureConfig,
    configured_regions,
    initialize_task_directories,
)
from .l9859b import load_bello_arufe2025_l9859b
from .wasp69b import load_schlawin2024_wasp69b
from .wasp80b import load_wiser2025_wasp80b


@dataclass(frozen=True)
class NamedRegionalModels:
    """Evaluate one configured forward model for each observation dataset."""

    models: Mapping[str, object]

    def __call__(self, values: Mapping[str, float]) -> Mapping[str, object]:
        return {name: model(values) for name, model in self.models.items()}


def mpi_rank() -> int:
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except ImportError:
        return 0


def mpi_processes(config: TaskConfig) -> int:
    configured = config.runtime.mpi_processes
    return (
        int(os.environ.get("SLURM_NTASKS", "1")) if configured == "auto" else configured
    )


def describe_config(config: TaskConfig) -> str:
    """Return a concise preflight summary suitable for terminal inspection."""

    regions = configured_regions(config)
    atmosphere_description = "; ".join(
        f"{region.name}={region.atmosphere.temperature.model}/"
        f"{region.atmosphere.chemistry.model}/clouds:{region.clouds.model}"
        for region in regions
    )
    return "\n".join(
        [
            f"Run: {config.run.name}",
            f"Target: {config.bodies.planet.name} / {config.bodies.star.name}",
            f"Data: {config.observations.loader} [{', '.join(config.observations.datasets)}]",
            f"Disk emission: {config.disk_emission.model}",
            f"Atmosphere regions: {atmosphere_description}",
            f"Opacity: {config.opacity.resolution} [{', '.join(config.opacity.species)}]",
            (
                f"Transmission: reference={config.radiative_transfer.reference_pressure_bar:g} bar, "
                f"gravity={config.radiative_transfer.gravity_model}"
                if config.radiative_transfer.model == "transmission"
                else f"Geometry: {config.radiative_transfer.geometry.model}"
            ),
            f"Parameters: {', '.join(item.name for item in config.parameters)}",
            _sampler_description(config),
            (
                "Plotting: automatic"
                if config.plotting.enabled
                else "Plotting: manual/disabled"
            ),
            f"MPI processes: {mpi_processes(config)}",
            f"Output: {config.outputs.directory}",
        ]
    )


def load_observations(config: TaskConfig) -> ObservationCollection:
    if config.observations.loader == "robert_npz":
        name = config.observations.datasets[0]
        observation = load_observation_npz(config.observations.path)
        option = config.observations.dataset_options.get(name)
        dataset = ObservationDataset(name=name, observation=observation)
        if option is not None:
            dataset = replace(
                dataset,
                offset_parameter=option.offset_parameter,
                jitter_parameter=option.jitter_parameter,
                uncertainty_scale_parameter=option.uncertainty_scale_parameter,
                uncertainty_scale=option.uncertainty_scale,
            )
        return ObservationCollection(
            datasets=(dataset,),
            name=f"{name} ROBERT NPZ observation",
            metadata={"source_path": str(config.observations.path)},
        )
    loaders = {
        "bello_arufe2025_l9859b": load_bello_arufe2025_l9859b,
        "schlawin2024_wasp69b": load_schlawin2024_wasp69b,
        "wiser2025_wasp80b": load_wiser2025_wasp80b,
    }
    published = loaders[config.observations.loader](
        config.observations.path,
        verify_checksum=config.observations.verify_checksum,
        miri_offset_parameter=config.observations.miri_offset_parameter,
    )
    requested = config.observations.datasets
    available = {dataset.name: dataset for dataset in published.datasets}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            f"unknown observation datasets {missing}; available: {sorted(available)}"
        )
    datasets = []
    for name in requested:
        dataset = available[name]
        option = config.observations.dataset_options.get(name)
        if option is not None:
            dataset = replace(
                dataset,
                offset_parameter=option.offset_parameter,
                jitter_parameter=option.jitter_parameter,
                uncertainty_scale_parameter=option.uncertainty_scale_parameter,
                uncertainty_scale=option.uncertainty_scale,
            )
        datasets.append(dataset)
    return ObservationCollection(
        datasets=tuple(datasets),
        name=f"{published.name} (configured selection)",
        metadata={**dict(published.metadata), "selection": ",".join(requested)},
    )


def cache_directory(config: TaskConfig) -> Path:
    return config.opacity.cache_directory / config.opacity.resolution


def prepare_opacity(config: TaskConfig, observations: ObservationCollection) -> None:
    """Prepare selected molecular opacity on each observed spectral grid."""

    if config.opacity.format == "exomol_cross_section_hdf":
        _prepare_exomol_cross_section_opacity(config, observations)
        return

    cache = cache_directory(config)
    cache.mkdir(parents=True, exist_ok=True)
    binning = config.opacity.binning
    for species in config.opacity.species:
        provider = CorrelatedKOpacityProvider.from_exomol_kta_directory(
            config.opacity.path,
            species=(species,),
            resolution=config.opacity.resolution,
            name=f"ExoMol-{config.opacity.resolution}-{species}",
            interpolation="log_pressure_temperature_log_k_clip",
            nonfinite_policy="floor",
        )
        table = provider.tables[species]
        source = Path(str(table.metadata["source_path"]))
        source_sha = str(table.metadata["checksum_sha256"])
        for dataset in observations.datasets:
            target = cache / f"{dataset.name}_{species}.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    current = {"opacity_resolution", "binning_num"}.issubset(
                        saved.files
                    ) and (
                        str(saved["source_sha256"]) == source_sha
                        and str(saved["opacity_resolution"])
                        == config.opacity.resolution
                        and int(saved["binning_num"]) == binning.num
                    )
                if current:
                    continue
            binned = provider.bin_to_spectral_grid(
                dataset.observation.spectral_grid,
                num=binning.num,
                use_rebin=binning.use_rebin,
                remove_zeros=binning.remove_zeros,
            ).tables[species]
            np.savez_compressed(
                target,
                species=species,
                pressure_bar=binned.pressure_bar,
                temperature_K=binned.temperature_K,
                wavenumber_cm_inverse=binned.wavenumber_cm_inverse,
                wavelength_micron=binned.wavelength_micron,
                g_samples=binned.g_samples,
                g_weights=binned.g_weights,
                kcoeff=binned.kcoeff,
                unit=binned.unit,
                source_path=str(source.resolve()),
                source_sha256=source_sha,
                opacity_resolution=config.opacity.resolution,
                binning_num=binning.num,
                spectral_preparation="exo_k_bin_down_cp",
            )


def _prepare_exomol_cross_section_opacity(
    config: TaskConfig,
    observations: ObservationCollection,
) -> None:
    """Correlate real ExoMolOP cross sections within observation bins.

    Each source wavelength sample is sorted by cross section independently at
    every pressure and temperature. Gauss-Legendre nodes then sample that
    empirical cumulative distribution. This is a target-bin correlated-k
    preparation; the source cross sections themselves are never synthesized.
    """

    cache = cache_directory(config)
    cache.mkdir(parents=True, exist_ok=True)
    for species in config.opacity.species:
        source = config.opacity.path / f"{species}.h5"
        if not source.is_file():
            raise FileNotFoundError(
                f"ExoMol cross-section HDF is missing for {species}: {source}"
            )
        source_sha = _file_sha256(source)
        for dataset in observations.datasets:
            target = cache / f"{dataset.name}_{species}.npz"
            wavelength = np.asarray(
                dataset.observation.spectral_grid.values,
                dtype=float,
            )
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    current = (
                        "source_sha256" in saved.files
                        and str(saved["source_sha256"]) == source_sha
                        and str(saved.get("spectral_preparation", ""))
                        == "exomol_cross_section_wavelength_weighted_k"
                        and saved["g_samples"].size
                        == config.opacity.binning.g_points
                        and np.allclose(saved["wavelength_micron"], wavelength)
                    )
                if current:
                    continue
            table = CorrelatedKTable.from_exomol_cross_section_hdf(
                source,
                species=species,
                spectral_grid=dataset.observation.spectral_grid,
                g_points=config.opacity.binning.g_points,
                checksum=False,
            )
            np.savez_compressed(
                target,
                species=species,
                pressure_bar=table.pressure_bar,
                temperature_K=table.temperature_K,
                wavenumber_cm_inverse=table.wavenumber_cm_inverse,
                wavelength_micron=table.wavelength_micron,
                g_samples=table.g_samples,
                g_weights=table.g_weights,
                kcoeff=table.kcoeff,
                unit=table.unit,
                source_path=str(source.resolve()),
                source_sha256=source_sha,
                source_doi=table.metadata["doi"],
                source_line_list=table.metadata["line_list"],
                opacity_resolution=config.opacity.resolution,
                binning_num=config.opacity.binning.num,
                g_points=config.opacity.binning.g_points,
                spectral_preparation="exomol_cross_section_wavelength_weighted_k",
            )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cached_table(
    config: TaskConfig, dataset: str, species: str
) -> CorrelatedKTable:
    path = cache_directory(config) / f"{dataset}_{species}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"prepared opacity cache is missing: {path}\n"
            "Run `python run_retrieval.py --config CONFIG --prepare-opacity` first."
        )
    with np.load(path, allow_pickle=False) as saved:
        opacity_resolution = (
            str(saved["opacity_resolution"])
            if "opacity_resolution" in saved.files
            else config.opacity.resolution
        )
        spectral_preparation = (
            str(saved["spectral_preparation"])
            if "spectral_preparation" in saved.files
            else "legacy_prepared_cache"
        )
        source_doi = str(saved["source_doi"]) if "source_doi" in saved.files else ""
        source_line_list = (
            str(saved["source_line_list"])
            if "source_line_list" in saved.files
            else ""
        )
        return CorrelatedKTable(
            species=species,
            pressure_bar=saved["pressure_bar"],
            temperature_K=saved["temperature_K"],
            wavenumber_cm_inverse=saved["wavenumber_cm_inverse"],
            wavelength_micron=saved["wavelength_micron"],
            g_samples=saved["g_samples"],
            g_weights=saved["g_weights"],
            kcoeff=saved["kcoeff"],
            unit=str(saved["unit"]),
            metadata={
                "source_path": str(saved["source_path"]),
                "checksum_sha256": str(saved["source_sha256"]),
                "opacity_resolution": opacity_resolution,
                "spectral_preparation": spectral_preparation,
                "source_doi": source_doi,
                "source_line_list": source_line_list,
            },
        )


def _planet(config: TaskConfig) -> tuple[Planet, float]:
    item = config.bodies.planet
    planet = Planet(
        name=item.name,
        radius_m=item.radius_m,
        mass_kg=item.mass_kg,
        gravity_m_s2=item.gravity_m_s2,
    )
    gravity = item.gravity_m_s2
    if gravity is None:
        gravity = 6.67430e-11 * float(item.mass_kg) / item.radius_m**2
    return planet, float(gravity)


def _parameters(items: tuple[ParameterConfig, ...]) -> RetrievalParameterSet:
    parameters = []
    for item in items:
        if item.prior.type == "uniform":
            prior = UniformPrior(item.prior.lower, item.prior.upper)
        elif item.prior.type == "log_uniform":
            prior = LogUniformPrior(item.prior.lower, item.prior.upper)
        else:
            prior = CenteredLogRatioPrior(
                item.prior.lower,
                item.prior.upper,
                group=item.prior.group or "composition",
            )
        parameters.append(
            RetrievalParameter(
                item.name,
                prior,
                label=item.label,
                unit=item.unit,
            )
        )
    return RetrievalParameterSet(tuple(parameters))


def _temperature_profile(config: TemperatureConfig, *, gravity: float):
    if config.model == "parmentier_guillot_2014":
        return ParmentierGuillot2014TemperatureProfile(
            gravity=gravity,
            internal_temperature=config.internal_temperature_k,
            kappa_ir_parameter_name=config.kappa_ir_parameter,
            gamma1_parameter_name=config.gamma1_parameter,
            gamma2_parameter_name=config.gamma2_parameter,
            irradiation_temperature_parameter_name=(
                config.irradiation_temperature_parameter
            ),
            alpha_parameter_name=config.alpha_parameter,
        )
    if config.model == "isothermal":
        return IsothermalTemperatureProfile(
            temperature=config.temperature_k,
            parameter_name=config.parameter_name,
        )
    if config.model == "tabulated":
        return TabulatedTemperatureProfile.from_csv(
            config.profile_path,
            pressure_column=config.pressure_column,
            temperature_column=config.temperature_column,
            pressure_unit=config.pressure_unit,
            extrapolation=config.extrapolation,
        )
    if config.model == "madhusudhan_seager_2009":
        return MadhusudhanSeager2009TemperatureProfile(
            pressure_unit=config.pressure_unit,
            reference_pressure=config.reference_pressure,
            p1_parameter_name=config.p1_parameter,
            p2_parameter_name=config.p2_parameter,
            p3_parameter_name=config.p3_parameter,
            t0_parameter_name=config.t0_parameter,
            alpha1_parameter_name=config.alpha1_parameter,
            alpha2_parameter_name=config.alpha2_parameter,
        )
    return SplineTemperatureProfile(
        knot_pressure=np.asarray(config.knot_pressure, dtype=float),
        knot_temperature=(
            None
            if config.knot_temperature_k is None
            else np.asarray(config.knot_temperature_k, dtype=float)
        ),
        parameter_names=config.parameter_names,
        pressure_unit=config.pressure_unit,
        extrapolation=config.extrapolation,
    )


def _chemistry_components(config: ChemistryConfig):
    if config.model == "fastchem_equilibrium":
        chemistry = FastChemEquilibriumChemistry(
            fastchem_path=config.fastchem_path,
            fastchem_species=tuple(item.fastchem_name for item in config.species),
            labels=tuple(item.label for item in config.species),
            metallicity_parameter_name=config.metallicity_parameter,
            carbon_to_oxygen_parameter_name=config.carbon_to_oxygen_parameter,
            constant_log10_vmr_parameters=(
                config.constant_log10_vmr_parameters or {}
            ),
        )
        return (
            chemistry,
            CompositionMeanMolecularWeight(normalization="raw_sum"),
            (),
        )
    if config.fill_background:
        fractions = config.background_fractions
        background = BackgroundGasMixture(
            {name: 1.0 for name in config.background_species}
            if fractions is None
            else dict(zip(config.background_species, fractions, strict=True))
        )
    else:
        background = None
    chemistry = FreeChemistry(
        active_species=config.species,
        background=background,
        fixed_mixing_ratios=config.fixed_mixing_ratios,
        parameter_names=config.parameter_names,
        parameter_mode=config.parameter_mode,
        fill_background=config.fill_background,
        excess_policy=config.excess_policy,
    )
    mean_molecular_weight = CompositionMeanMolecularWeight(
        normalization="require" if config.fill_background else "normalize",
        molecular_mass_parameters=(
            {}
            if config.phantom_species is None
            else {
                config.phantom_species: config.phantom_mean_molecular_weight_parameter
            }
        ),
    )
    opacity_free_species = (
        () if config.phantom_species is None else (config.phantom_species,)
    )
    return chemistry, mean_molecular_weight, opacity_free_species


def _cloud_model(config: CloudsConfig):
    if config.model == "none":
        return None
    if config.model == "deck_haze":
        return ParameterizedDeckHazeCloudModel(
            log10_cloud_top_pressure_bar_parameter=(
                config.log10_cloud_top_pressure_bar_parameter
            ),
            log10_cloud_optical_depth_parameter=(
                config.log10_cloud_optical_depth_parameter
            ),
            log10_haze_mass_extinction_parameter=(
                config.log10_haze_mass_extinction_parameter
            ),
            haze_slope_parameter=config.haze_slope_parameter,
            haze_reference_wavelength_micron=(
                config.haze_reference_wavelength_micron
            ),
            deck_single_scattering_albedo=config.deck_single_scattering_albedo,
            deck_asymmetry_factor=config.deck_asymmetry_factor,
            haze_single_scattering_albedo=config.haze_single_scattering_albedo,
            haze_asymmetry_factor=config.haze_asymmetry_factor,
            multiple_scattering_backend=config.multiple_scattering_backend,
        )
    if config.model == "mie_catalog":
        wavelength = ()
        real_parameters = ()
        imaginary_parameters = ()
        fixed_index = OpticalConstantsCatalog(config.optical_constants_path).load(
            config.material
        )
    else:
        wavelength = config.refractive_index_wavelength_micron
        real_parameters = config.real_index_parameter_names
        imaginary_parameters = config.log10_imaginary_index_parameter_names
        fixed_index = None
    return ParameterizedMieCloudModel(
        refractive_index_wavelength_micron=wavelength,
        real_index_parameter_names=real_parameters,
        log10_imaginary_index_parameter_names=imaginary_parameters,
        fixed_refractive_index=fixed_index,
        log10_condensate_mass_fraction_parameter=config.log10_mass_fraction_parameter,
        log10_effective_radius_micron_parameter=config.log10_radius_micron_parameter,
        particle_density_kg_m3=config.particle_density_kg_m3,
        geometric_stddev=config.geometric_stddev,
        log10_cloud_top_pressure_bar_parameter=config.log10_top_pressure_bar_parameter,
        log10_cloud_base_pressure_bar_parameter=config.log10_base_pressure_bar_parameter,
        quadrature_points=config.quadrature_points,
        refractive_index_extrapolation="raise",
        multiple_scattering_backend=config.multiple_scattering_backend,
    )


def _pressure_grid(region: ResolvedRegionConfig, *, planet: Planet) -> PressureGrid:
    pressure = region.atmosphere.pressure
    return PressureGrid.from_log_centers(
        pressure.bottom_bar,
        pressure.top_bar,
        n_layers=pressure.layers,
        unit="bar",
        name=f"{planet.name} {region.name} configured pressure grid",
    )


def _opacity_providers(
    config: TaskConfig,
    observations: ObservationCollection,
    *,
    planet: Planet,
) -> dict[str, CorrelatedKOpacityProvider]:
    providers = {}
    for dataset in observations.datasets:
        tables = {
            species: _load_cached_table(config, dataset.name, species)
            for species in config.opacity.species
        }
        reference = next(iter(tables.values()))
        for species, table in tuple(tables.items()):
            if not np.allclose(
                table.g_samples, reference.g_samples, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(f"{species} uses a different correlated-k g grid")
            if not np.allclose(
                table.g_weights, reference.g_weights, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(f"{species} uses different correlated-k weights")
            tables[species] = replace(
                table,
                g_samples=reference.g_samples,
                g_weights=reference.g_weights,
            )
        providers[dataset.name] = CorrelatedKOpacityProvider(
            tables,
            name=f"{planet.name}-{dataset.name}-{config.opacity.resolution}-binned",
            interpolation="log_pressure_temperature_log_k_clip",
        )
    return providers


def build_problem(
    config: TaskConfig,
    observations: ObservationCollection | None = None,
) -> MultiDatasetRetrievalProblem:
    """Construct the typed regional retrieval problem selected by the YAML file."""

    observations = load_observations(config) if observations is None else observations
    planet, gravity = _planet(config)
    star_item = config.bodies.star
    star = Star(
        name=star_item.name,
        radius_m=star_item.radius_m,
        effective_temperature_k=star_item.effective_temperature_k,
        log_g_cgs=star_item.log_g_cgs,
        metallicity_dex=star_item.metallicity_dex,
    )
    geometry_item = config.radiative_transfer.geometry
    geometry = (
        normal_emission_geometry()
        if geometry_item.model == "normal_emission"
        else gauss_legendre_disk_geometry(geometry_item.points)
    )
    cia = load_nemesispy_cia_table()
    rt = config.radiative_transfer
    if rt.model == "emission":
        model_config = ParameterizedEmissionModelConfig(
            opacity_species=config.opacity.species,
            include_rayleigh=rt.include_rayleigh,
            gas_combination=rt.gas_combination,
            thermal_integration_backend=rt.thermal_integration_backend,
            stellar_spectrum_model=star_item.spectrum_model,
            metadata={"configured_geometry": geometry_item.model},
        )
    else:
        model_config = ParameterizedTransmissionModelConfig(
            opacity_species=config.opacity.species,
            reference_pressure_bar=rt.reference_pressure_bar,
            radius_scale_parameter=rt.radius_scale_parameter,
            gravity_model=rt.gravity_model,
            include_rayleigh=rt.include_rayleigh,
            gas_combination=rt.gas_combination,
            impact_quadrature_order=rt.impact_quadrature_order,
            metadata={"configured_model": "transmission"},
        )
    providers = _opacity_providers(config, observations, planet=planet)
    spectral_grids = {
        dataset.name: dataset.observation.spectral_grid
        for dataset in observations.datasets
    }
    regions = configured_regions(config)
    regional_models = {}
    regional_opacity_ids = {}
    for region in regions:
        temperature = _temperature_profile(
            region.atmosphere.temperature,
            gravity=gravity,
        )
        chemistry, mean_molecular_weight, opacity_free_species = (
            _chemistry_components(region.atmosphere.chemistry)
        )
        pressure = _pressure_grid(region, planet=planet)
        cloud = _cloud_model(region.clouds)
        if rt.model == "emission":
            factories = {
                dataset.name: ParameterizedEmissionFactoryConfig(
                    planet=planet,
                    star=star,
                    temperature_profile=temperature,
                    chemistry_model=chemistry,
                    mean_molecular_weight_model=mean_molecular_weight,
                    opacity_free_species=opacity_free_species,
                    pressure_grid=pressure,
                    cia_table=cia,
                    geometry=geometry,
                    opacity_source=providers[dataset.name],
                    opacity_binning=None,
                    model=model_config,
                    cloud_model=cloud,
                )
                for dataset in observations.datasets
            }
            regional_model = build_multi_dataset_emission_model(
                factories,
                spectral_grids=spectral_grids,
            )
            regional_opacity_ids[region.name] = {
                f"{dataset}:{key}": value
                for dataset, model in regional_model.models.items()
                for key, value in model.opacity_identifiers.items()
            }
        else:
            dataset_models = {}
            for dataset in observations.datasets:
                factory = ParameterizedTransmissionFactoryConfig(
                    planet=planet,
                    star=star,
                    temperature_profile=temperature,
                    chemistry_model=chemistry,
                    mean_molecular_weight_model=mean_molecular_weight,
                    opacity_free_species=opacity_free_species,
                    pressure_grid=pressure,
                    cia_table=cia,
                    opacity_source=providers[dataset.name],
                    opacity_binning=None,
                    model=model_config,
                    cloud_model=cloud,
                )
                dataset_models[dataset.name] = build_parameterized_transmission_model(
                    factory,
                    spectral_grid=dataset.observation.spectral_grid,
                )
            regional_model = NamedRegionalModels(dataset_models)
            regional_opacity_ids[region.name] = {
                f"{dataset}:{key}": value
                for dataset, model in dataset_models.items()
                for key, value in model.opacity_identifiers.items()
            }
        regional_models[region.name] = regional_model

    disk_mode = config.disk_emission.model
    if disk_mode in {"two_region", "2tp"}:
        forward_model = MultiDatasetTwoRegionEmissionModel(
            regional_models["hot"],
            regional_models["cold"],
            hot_fraction_parameter=config.disk_emission.hot_fraction_parameter,
        )
        opacity_ids = {
            f"{region}:{key}": value
            for region, identifiers in regional_opacity_ids.items()
            for key, value in identifiers.items()
        }
        canonical_disk_mode = "two_region"
    elif disk_mode in {"diluted", "diluted_one_region"}:
        forward_model = MultiDatasetDilutedEmissionModel(
            regional_models["primary"],
            dilution_parameter=config.disk_emission.dilution_parameter,
        )
        opacity_ids = regional_opacity_ids["primary"]
        canonical_disk_mode = "diluted_one_region"
    else:
        forward_model = regional_models["primary"]
        opacity_ids = regional_opacity_ids["primary"]
        canonical_disk_mode = "one_region"
    return MultiDatasetRetrievalProblem(
        name=config.run.name,
        observations=observations,
        parameters=_parameters(config.parameters),
        forward_model=forward_model,
        likelihood=MultiDatasetGaussianLikelihood(
            include_normalization=config.likelihood.include_normalization
        ),
        invalid_loglike=-1.0e100,
        metadata={
            "configuration_schema_version": str(config.schema_version),
            "opacity_resolution": config.opacity.resolution,
            "cloud_model": ",".join(
                f"{region.name}:{region.clouds.model}" for region in regions
            ),
            "disk_emission_model": canonical_disk_mode,
            "geometry": geometry_item.model,
            "radiative_transfer_model": rt.model,
        },
        opacity_identifiers=opacity_ids,
    )


def build_native_emission_model(
    config: TaskConfig,
    observations: ObservationCollection,
):
    """Build an exact native-opacity-grid model for retrieval diagnostics.

    Native diagnostic spectra are currently available for ExoMol correlated-k
    tables. Cross-section HDF inputs require a separate spectral correlation
    choice and therefore deliberately fall back to observation-grid plotting.
    """

    if (
        config.radiative_transfer.model != "emission"
        or config.opacity.format != "exomol_kta"
    ):
        return None
    provider = CorrelatedKOpacityProvider.from_exomol_kta_directory(
        config.opacity.path,
        species=config.opacity.species,
        resolution=config.opacity.resolution,
        name=f"native-{config.opacity.resolution}",
        interpolation="log_pressure_temperature_log_k_clip",
        nonfinite_policy="floor",
    )
    reference = provider.tables[config.opacity.species[0]]
    common_wavenumber = np.asarray(reference.wavenumber_cm_inverse, dtype=float)
    for species in config.opacity.species[1:]:
        common_wavenumber = np.intersect1d(
            common_wavenumber,
            provider.tables[species].wavenumber_cm_inverse,
        )
    observed_wavelength = np.concatenate(
        [dataset.observation.wavelength for dataset in observations.datasets]
    )
    native_wavelength = 10_000.0 / common_wavenumber
    selected = (
        (native_wavelength >= float(np.min(observed_wavelength)))
        & (native_wavelength <= float(np.max(observed_wavelength)))
    )
    native_wavelength = np.sort(native_wavelength[selected])
    if native_wavelength.size < 2:
        raise RobertConfigError(
            "native opacity grid does not cover the configured observations"
        )
    spectral_grid = SpectralGrid.from_array(
        native_wavelength,
        unit="micron",
        name=f"{config.opacity.resolution} native opacity diagnostic grid",
    )
    planet, gravity = _planet(config)
    star_item = config.bodies.star
    star = Star(
        name=star_item.name,
        radius_m=star_item.radius_m,
        effective_temperature_k=star_item.effective_temperature_k,
        log_g_cgs=star_item.log_g_cgs,
        metallicity_dex=star_item.metallicity_dex,
    )
    geometry_item = config.radiative_transfer.geometry
    geometry = (
        normal_emission_geometry()
        if geometry_item.model == "normal_emission"
        else gauss_legendre_disk_geometry(geometry_item.points)
    )
    rt = config.radiative_transfer
    model_config = ParameterizedEmissionModelConfig(
        opacity_species=config.opacity.species,
        include_rayleigh=rt.include_rayleigh,
        gas_combination=rt.gas_combination,
        thermal_integration_backend=rt.thermal_integration_backend,
        stellar_spectrum_model=star_item.spectrum_model,
        metadata={
            "configured_geometry": geometry_item.model,
            "diagnostic_grid": "native_opacity",
        },
    )
    cia = load_nemesispy_cia_table()
    regional_models = {}
    for region in configured_regions(config):
        chemistry, mean_molecular_weight, opacity_free_species = (
            _chemistry_components(region.atmosphere.chemistry)
        )
        factory = ParameterizedEmissionFactoryConfig(
            planet=planet,
            star=star,
            temperature_profile=_temperature_profile(
                region.atmosphere.temperature,
                gravity=gravity,
            ),
            chemistry_model=chemistry,
            mean_molecular_weight_model=mean_molecular_weight,
            opacity_free_species=opacity_free_species,
            pressure_grid=_pressure_grid(region, planet=planet),
            cia_table=cia,
            geometry=geometry,
            opacity_source=provider,
            opacity_binning=None,
            model=model_config,
            cloud_model=_cloud_model(region.clouds),
        )
        regional_models[region.name] = build_parameterized_emission_model(
            factory,
            spectral_grid=spectral_grid,
        )
    disk = config.disk_emission
    if disk.model in {"two_region", "2tp"}:
        return TwoRegionEmissionModel(
            regional_models["hot"],
            regional_models["cold"],
            hot_fraction_parameter=disk.hot_fraction_parameter,
        )
    if disk.model in {"diluted", "diluted_one_region"}:
        return DilutedEmissionModel(
            regional_models["primary"],
            dilution_parameter=disk.dilution_parameter,
        )
    return regional_models["primary"]


def smoke_evaluation(
    problem: MultiDatasetRetrievalProblem,
    state: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate a representative state and expose any hidden model error."""

    theta = (
        problem.prior_transform(np.full(problem.ndim, 0.5))
        if state is None
        else np.asarray(state, dtype=float)
    )
    started = time.perf_counter()
    value = problem.log_likelihood_from_vector(theta)
    elapsed = time.perf_counter() - started
    if not np.isfinite(value) or value <= problem.invalid_loglike:
        # log_likelihood_from_vector deliberately converts physical-domain
        # errors to the sampler's invalid floor. Re-evaluate through the
        # unguarded OE interface so setup-time smoke failures retain their
        # actionable original exception.
        problem.gaussian_inputs_from_vector(theta)
        raise RuntimeError(
            "smoke evaluation returned an invalid likelihood"
        )
    return {"elapsed_seconds": elapsed, "log_likelihood": float(value)}


def write_config_snapshot(config: TaskConfig, source: Path) -> None:
    if mpi_rank() != 0:
        return
    output = config.outputs.directory
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "input_config.yaml"
    if source.resolve() != destination.resolve():
        shutil.copyfile(source.resolve(), destination)
    normalized = config.model_dump(mode="json")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8"
    )


def run_retrieval_task(config: TaskConfig, source: Path):
    initialize_task_directories(config)
    observations = load_observations(config)
    problem = build_problem(config, observations)
    smoke = smoke_evaluation(problem)
    write_config_snapshot(config, source)
    if mpi_rank() == 0:
        (config.outputs.directory / "smoke_evaluation.json").write_text(
            dumps(smoke, indent=2), encoding="utf-8"
        )
    sampler = config.sampler
    engine = sampler.engine
    if engine == "optimal_estimation":
        result = run_retrieval(
            problem,
            method="optimal_estimation",
            output_dir=config.outputs.directory / "optimal_estimation",
            **_optimal_estimation_kwargs(config, problem),
        )
    elif engine in {"ultranest", "multinest"}:
        result = run_retrieval(
            problem,
            method=engine,
            output_dir=config.outputs.directory / engine,
            seed=sampler.seed,
            **_nested_sampler_kwargs(config, engine),
        )
    else:
        nested_method = "multinest" if engine.endswith("multinest") else "ultranest"
        result = run_oe_then_nested_sampling(
            problem,
            output_dir=config.outputs.directory,
            prior_sigma=sampler.prior_sigma,
            minimum_prior_fraction=sampler.minimum_prior_fraction,
            require_oe_convergence=sampler.require_oe_convergence,
            oe_kwargs=_optimal_estimation_kwargs(config, problem),
            nested_kwargs=_nested_sampler_kwargs(config, nested_method),
            nested_method=nested_method,
            seed=sampler.seed,
        )
    if mpi_rank() == 0 and config.plotting.enabled and config.plotting.retrieval:
        _postprocess_retrieval_outputs(config, problem)
    return result


def _optimal_estimation_kwargs(
    config: TaskConfig,
    problem: MultiDatasetRetrievalProblem,
) -> dict[str, object]:
    sampler = config.sampler
    settings: dict[str, object] = {
        "max_iterations": sampler.oe_max_iterations,
        "convergence_tolerance": sampler.oe_convergence_tolerance,
        "finite_difference_fraction": sampler.oe_finite_difference_fraction,
        "damping": sampler.oe_damping,
    }
    if sampler.oe_temperature_prior_sigma_k is not None:
        settings["prior_covariance"] = configured_temperature_prior_covariance(
            config, problem
        )[0]
    return settings


def configured_temperature_prior_covariance(
    config: TaskConfig,
    problem: MultiDatasetRetrievalProblem,
    covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert the configured log-pressure smoothing block into an OE prior."""

    sigma = config.sampler.oe_temperature_prior_sigma_k
    correlation_length = config.sampler.oe_temperature_correlation_length_dex
    if sigma is None or correlation_length is None:
        raise RobertConfigError(
            "OE temperature prior sigma and correlation length are not configured"
        )
    builder = getattr(problem.forward_model, "atmosphere_builder", None)
    profile = None if builder is None else builder.temperature_profile
    if not isinstance(profile, SplineTemperatureProfile):
        raise RobertConfigError(
            "correlated OE temperature priors require a retrieved spline profile"
        )
    if profile.knot_temperature is not None:
        raise RobertConfigError(
            "correlated OE temperature priors require retrieved temperatures"
        )
    names = tuple(profile.parameter_names or ())
    parameter_index = {
        name: index for index, name in enumerate(problem.parameter_names)
    }
    try:
        indices = np.asarray([parameter_index[name] for name in names], dtype=int)
    except KeyError as exc:
        raise RobertConfigError(
            f"OE problem is missing spline temperature parameter {exc.args[0]!r}"
        ) from exc
    if covariance is None:
        smoothed = np.diag(
            [
                parameter.approximate_standard_deviation**2
                for parameter in problem.parameters.parameters
            ]
        )
    else:
        smoothed = np.array(covariance, dtype=float, copy=True)
    expected_shape = (problem.ndim, problem.ndim)
    if smoothed.shape != expected_shape:
        raise RobertConfigError(
            f"OE prior covariance must have shape {expected_shape}"
        )
    temperature_covariance = log_pressure_correlated_covariance(
        profile.knot_pressure,
        standard_deviation=sigma,
        correlation_length_dex=correlation_length,
    )
    smoothed[np.ix_(indices, indices)] = temperature_covariance
    return smoothed, np.array(profile.knot_pressure, copy=True)


def _nested_sampler_kwargs(config: TaskConfig, method: str) -> dict[str, object]:
    sampler = config.sampler
    common: dict[str, object] = {
        "mpi_nprocs": mpi_processes(config),
        "invalid_loglike_floor": sampler.invalid_loglike_floor,
    }
    if method == "ultranest":
        return {
            **common,
            "min_num_live_points": sampler.live_points,
            "max_ncalls": sampler.max_calls,
            "dlogz": sampler.dlogz,
            "resume": sampler.resume,
            "show_status": sampler.show_status,
        }
    return {
        **common,
        "n_live_points": sampler.live_points,
        "max_iter": sampler.multinest_max_iterations,
        "evidence_tolerance": sampler.dlogz,
        "sampling_efficiency": sampler.sampling_efficiency,
        "resume": sampler.resume == "resume",
        "verbose": sampler.show_status,
        "importance_nested_sampling": sampler.importance_nested_sampling,
        "multimodal": sampler.multimodal,
        "n_iter_before_update": sampler.iterations_before_update,
    }


def _sampler_description(config: TaskConfig) -> str:
    sampler = config.sampler
    if sampler.engine == "optimal_estimation":
        description = (
            f"Inference: optimal_estimation, max_iterations={sampler.oe_max_iterations}"
        )
        if sampler.oe_temperature_prior_sigma_k is not None:
            description += (
                ", correlated_temperature_prior="
                f"{sampler.oe_temperature_prior_sigma_k:g} K/"
                f"{sampler.oe_temperature_correlation_length_dex:g} dex"
            )
        return description
    limits = (
        f"max_calls={sampler.max_calls}"
        if "ultranest" in sampler.engine
        else f"max_iterations={sampler.multinest_max_iterations or 'unlimited'}"
    )
    return f"Inference: {sampler.engine}, live_points={sampler.live_points}, {limits}"


def run_smoke_task(config: TaskConfig, source: Path) -> dict[str, float]:
    """Run one likelihood evaluation and persist its exact configuration."""

    initialize_task_directories(config)
    problem = build_problem(config, load_observations(config))
    smoke = smoke_evaluation(problem)
    if mpi_rank() == 0:
        _preflight_retrieval_manifests(config, problem)
    write_config_snapshot(config, source)
    if mpi_rank() == 0:
        (config.outputs.directory / "smoke_evaluation.json").write_text(
            dumps(smoke, indent=2), encoding="utf-8"
        )
    return smoke


def _preflight_retrieval_manifests(
    config: TaskConfig,
    problem: MultiDatasetRetrievalProblem,
) -> None:
    """Exercise manifest serialization for every configured inference phase."""

    engine = config.sampler.engine
    if engine == "optimal_estimation" or engine.startswith("optimal_estimation_to_"):
        oe_kwargs = _optimal_estimation_kwargs(config, problem)
        build_run_manifest(
            problem,
            method="optimal_estimation",
            settings={"method": "optimal_estimation", **oe_kwargs},
            random_seed=None,
        )
    if engine in {"ultranest", "multinest"} or engine.startswith(
        "optimal_estimation_to_"
    ):
        nested_method = "multinest" if engine.endswith("multinest") else "ultranest"
        nested_kwargs = _nested_sampler_kwargs(config, nested_method)
        build_run_manifest(
            problem,
            method=nested_method,
            settings={
                "method": nested_method,
                **nested_kwargs,
                "seed": config.sampler.seed,
            },
            random_seed=config.sampler.seed,
        )


def run_forward_task(config: TaskConfig, source: Path) -> Path:
    initialize_task_directories(config)
    observations = load_observations(config)
    problem = build_problem(config, observations)
    values = {}
    midpoint = problem.parameters.midpoint_vector()
    for configured, parameter, midpoint_value in zip(
        config.parameters,
        problem.parameters.parameters,
        midpoint,
        strict=True,
    ):
        values[parameter.name] = (
            float(midpoint_value) if configured.value is None else configured.value
        )
    spectra = problem.model_spectra(values)
    write_config_snapshot(config, source)
    target = config.outputs.directory / "forward_model.npz"
    np.savez_compressed(
        target,
        **{
            f"{name}_wavelength_micron": spectrum.spectral_grid.values
            for name, spectrum in spectra.items()
        },
        **{f"{name}_model": spectrum.values for name, spectrum in spectra.items()},
        **{f"parameter_{name}": value for name, value in values.items()},
    )
    if mpi_rank() == 0 and config.plotting.enabled and config.plotting.forward:
        from robert_exoplanets.postprocessing import postprocess_forward_output

        postprocess_forward_output(
            problem,
            target,
            plot_dir=config.outputs.directory / "plots" / "forward",
            dataset_colors=config.plotting.dataset_colors,
            style=config.plotting.style,
            image_format=config.plotting.image_format,
            dpi=config.plotting.dpi,
        )
        print(
            f"Forward plots written to {config.outputs.directory / 'plots' / 'forward'}",
            flush=True,
        )
    return target


def _postprocess_retrieval_outputs(
    config: TaskConfig,
    problem: MultiDatasetRetrievalProblem,
) -> None:
    from robert_exoplanets.postprocessing import (
        discover_retrieval_result_directories,
        postprocess_retrieval_output,
    )

    plot_root = config.outputs.directory / "plots"
    parameter_labels = {
        parameter.name: parameter.label
        for parameter in config.parameters
        if parameter.label is not None
    }
    parameter_labels.update(config.plotting.parameter_labels)
    native_spectrum_model = build_native_emission_model(
        config,
        problem.observations,
    )
    for result_dir in discover_retrieval_result_directories(config.outputs.directory):
        postprocess_retrieval_output(
            problem,
            result_dir,
            plot_dir=plot_root / result_dir.name,
            parameter_labels=parameter_labels,
            dataset_colors=config.plotting.dataset_colors,
            style=config.plotting.style,
            image_format=config.plotting.image_format,
            dpi=config.plotting.dpi,
            max_posterior_samples=config.plotting.max_posterior_samples,
            posterior_predictive_samples=(
                config.plotting.posterior_predictive_samples
            ),
            posterior_predictive_seed=config.plotting.posterior_predictive_seed,
            corner_max_parameters=config.plotting.corner_max_parameters,
            native_spectrum_model=native_spectrum_model,
            leave_one_out=config.plotting.leave_one_out.enabled,
            loo_max_posterior_draws=(
                config.plotting.leave_one_out.max_posterior_draws
            ),
            loo_seed=config.plotting.leave_one_out.seed,
            loo_pareto_k_threshold=(
                config.plotting.leave_one_out.pareto_k_threshold
            ),
        )
        print(
            f"Retrieval plots written to {plot_root / result_dir.name}",
            flush=True,
        )


__all__ = [
    "build_native_emission_model",
    "build_problem",
    "describe_config",
    "load_observations",
    "prepare_opacity",
    "run_forward_task",
    "run_retrieval_task",
    "run_smoke_task",
    "smoke_evaluation",
]
