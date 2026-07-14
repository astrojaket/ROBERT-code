"""Build and run ROBERT tasks from the strict user-facing configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from robert_exoplanets.core import PressureGrid
from robert_exoplanets.forward import (
    ParameterizedEmissionFactoryConfig,
    ParameterizedEmissionModelConfig,
    ParameterizedRefractiveIndexCloudEmissionForwardModel,
    RefractiveIndexCloudConfig,
    build_multi_dataset_emission_model,
    build_parameterized_emission_model,
)
from robert_exoplanets.instruments import ObservationCollection
from robert_exoplanets.likelihoods import MultiDatasetGaussianLikelihood
from robert_exoplanets.opacity import CorrelatedKOpacityProvider, CorrelatedKTable
from robert_exoplanets.retrieval import (
    LogUniformPrior,
    MultiDatasetRetrievalProblem,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
    run_oe_then_nested_sampling,
    run_retrieval,
)
from robert_exoplanets.rt import (
    gauss_legendre_disk_geometry,
    load_nemesispy_cia_table,
    normal_emission_geometry,
    OpticalConstantsCatalog,
)

from .task_config import ParameterConfig, TaskConfig, initialize_task_directories
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

    return "\n".join(
        [
            f"Run: {config.run.name}",
            f"Target: {config.bodies.planet.name} / {config.bodies.star.name}",
            f"Data: {config.observations.loader} [{', '.join(config.observations.datasets)}]",
            f"Atmosphere: {config.atmosphere.temperature.model}, {config.atmosphere.chemistry.model}, clouds={config.clouds.model}",
            f"Opacity: {config.opacity.resolution} [{', '.join(config.opacity.species)}]",
            f"Geometry: {config.radiative_transfer.geometry.model}",
            f"Parameters: {', '.join(item.name for item in config.parameters)}",
            _sampler_description(config),
            f"MPI processes: {mpi_processes(config)}",
            f"Output: {config.outputs.directory}",
        ]
    )


def load_observations(config: TaskConfig) -> ObservationCollection:
    loaders = {
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
    """Recompress the selected KTA species into each observed spectral grid."""

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
        prior_class = UniformPrior if item.prior.type == "uniform" else LogUniformPrior
        parameters.append(
            RetrievalParameter(
                item.name,
                prior_class(item.prior.lower, item.prior.upper),
                label=item.label,
                unit=item.unit,
            )
        )
    return RetrievalParameterSet(tuple(parameters))


def build_problem(
    config: TaskConfig,
    observations: ObservationCollection | None = None,
) -> MultiDatasetRetrievalProblem:
    """Construct the typed emission problem selected by the YAML file."""

    observations = load_observations(config) if observations is None else observations
    planet, gravity = _planet(config)
    star_item = config.bodies.star
    star = Star(
        name=star_item.name,
        radius_m=star_item.radius_m,
        effective_temperature_k=star_item.effective_temperature_k,
    )
    pressure_item = config.atmosphere.pressure
    pressure = PressureGrid.from_log_centers(
        pressure_item.bottom_bar,
        pressure_item.top_bar,
        n_layers=pressure_item.layers,
        unit="bar",
        name=f"{planet.name} configured pressure grid",
    )
    temperature_item = config.atmosphere.temperature
    if temperature_item.model == "parmentier_guillot_2014":
        temperature = ParmentierGuillot2014TemperatureProfile(
            gravity=gravity,
            internal_temperature=temperature_item.internal_temperature_k,
        )
    elif temperature_item.model == "isothermal":
        temperature = IsothermalTemperatureProfile(
            temperature=temperature_item.temperature_k,
            parameter_name=temperature_item.parameter_name,
        )
    elif temperature_item.model == "tabulated":
        temperature = TabulatedTemperatureProfile.from_csv(
            temperature_item.profile_path,
            pressure_column=temperature_item.pressure_column,
            temperature_column=temperature_item.temperature_column,
            pressure_unit=temperature_item.pressure_unit,
            extrapolation=temperature_item.extrapolation,
        )
    elif temperature_item.model == "madhusudhan_seager_2009":
        temperature = MadhusudhanSeager2009TemperatureProfile(
            pressure_unit=temperature_item.pressure_unit,
            reference_pressure=temperature_item.reference_pressure,
            p1_parameter_name=temperature_item.p1_parameter,
            p2_parameter_name=temperature_item.p2_parameter,
            p3_parameter_name=temperature_item.p3_parameter,
            t0_parameter_name=temperature_item.t0_parameter,
            alpha1_parameter_name=temperature_item.alpha1_parameter,
            alpha2_parameter_name=temperature_item.alpha2_parameter,
        )
    else:
        temperature = SplineTemperatureProfile(
            knot_pressure=np.asarray(temperature_item.knot_pressure, dtype=float),
            knot_temperature=(
                None
                if temperature_item.knot_temperature_k is None
                else np.asarray(temperature_item.knot_temperature_k, dtype=float)
            ),
            parameter_names=temperature_item.parameter_names,
            pressure_unit=temperature_item.pressure_unit,
            extrapolation=temperature_item.extrapolation,
        )
    chemistry_item = config.atmosphere.chemistry
    if chemistry_item.model == "fastchem_equilibrium":
        chemistry = FastChemEquilibriumChemistry(
            fastchem_path=chemistry_item.fastchem_path,
            fastchem_species=tuple(
                item.fastchem_name for item in chemistry_item.species
            ),
            labels=tuple(item.label for item in chemistry_item.species),
            metallicity_parameter_name=chemistry_item.metallicity_parameter,
            carbon_to_oxygen_parameter_name=chemistry_item.carbon_to_oxygen_parameter,
        )
        mean_molecular_weight = CompositionMeanMolecularWeight(normalization="raw_sum")
    else:
        if chemistry_item.fill_background:
            fractions = chemistry_item.background_fractions
            if fractions is None:
                background = BackgroundGasMixture(
                    {name: 1.0 for name in chemistry_item.background_species}
                )
            else:
                background = BackgroundGasMixture(
                    dict(zip(chemistry_item.background_species, fractions, strict=True))
                )
        else:
            background = None
        chemistry = FreeChemistry(
            active_species=chemistry_item.species,
            background=background,
            fixed_mixing_ratios=chemistry_item.fixed_mixing_ratios,
            parameter_names=chemistry_item.parameter_names,
            parameter_mode=chemistry_item.parameter_mode,
            fill_background=chemistry_item.fill_background,
            excess_policy=chemistry_item.excess_policy,
        )
        mean_molecular_weight = CompositionMeanMolecularWeight(
            normalization="require" if chemistry_item.fill_background else "normalize"
        )
    geometry_item = config.radiative_transfer.geometry
    geometry = (
        normal_emission_geometry()
        if geometry_item.model == "normal_emission"
        else gauss_legendre_disk_geometry(geometry_item.points)
    )
    cia = load_nemesispy_cia_table()
    rt = config.radiative_transfer
    model_config = ParameterizedEmissionModelConfig(
        opacity_species=config.opacity.species,
        include_rayleigh=rt.include_rayleigh,
        gas_combination=rt.gas_combination,
        thermal_integration_backend=rt.thermal_integration_backend,
        metadata={"configured_geometry": geometry_item.model},
    )
    configs = {}
    cloud_models = {}
    spectral_grids = {}
    clouds = config.clouds
    cloud = None
    if clouds.model == "mie_catalog":
        cloud = RefractiveIndexCloudConfig(
            refractive_index_wavelength_micron=(),
            real_index_parameter_names=(),
            log10_imaginary_index_parameter_names=(),
            fixed_refractive_index=OpticalConstantsCatalog(
                clouds.optical_constants_path
            ).load(clouds.material),
            log10_condensate_mass_fraction_parameter=clouds.log10_mass_fraction_parameter,
            log10_effective_radius_micron_parameter=clouds.log10_radius_micron_parameter,
            particle_density_kg_m3=clouds.particle_density_kg_m3,
            geometric_stddev=clouds.geometric_stddev,
            log10_cloud_top_pressure_bar_parameter=clouds.log10_top_pressure_bar_parameter,
            log10_cloud_base_pressure_bar_parameter=clouds.log10_base_pressure_bar_parameter,
            quadrature_points=clouds.quadrature_points,
            refractive_index_extrapolation="raise",
            multiple_scattering_backend=clouds.multiple_scattering_backend,
        )
    elif clouds.model == "mie_direct_nk":
        cloud = RefractiveIndexCloudConfig(
            refractive_index_wavelength_micron=clouds.refractive_index_wavelength_micron,
            real_index_parameter_names=clouds.real_index_parameter_names,
            log10_imaginary_index_parameter_names=clouds.log10_imaginary_index_parameter_names,
            log10_condensate_mass_fraction_parameter=clouds.log10_mass_fraction_parameter,
            log10_effective_radius_micron_parameter=clouds.log10_radius_micron_parameter,
            particle_density_kg_m3=clouds.particle_density_kg_m3,
            geometric_stddev=clouds.geometric_stddev,
            log10_cloud_top_pressure_bar_parameter=clouds.log10_top_pressure_bar_parameter,
            log10_cloud_base_pressure_bar_parameter=clouds.log10_base_pressure_bar_parameter,
            quadrature_points=clouds.quadrature_points,
            refractive_index_extrapolation="raise",
            multiple_scattering_backend=clouds.multiple_scattering_backend,
        )
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
                table, g_samples=reference.g_samples, g_weights=reference.g_weights
            )
        provider = CorrelatedKOpacityProvider(
            tables,
            name=f"{planet.name}-{dataset.name}-{config.opacity.resolution}-binned",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        factory = ParameterizedEmissionFactoryConfig(
            planet=planet,
            star=star,
            temperature_profile=temperature,
            chemistry_model=chemistry,
            mean_molecular_weight_model=mean_molecular_weight,
            pressure_grid=pressure,
            cia_table=cia,
            geometry=geometry,
            opacity_source=provider,
            opacity_binning=None,
            model=model_config,
        )
        spectral_grids[dataset.name] = dataset.observation.spectral_grid
        if cloud is None:
            configs[dataset.name] = factory
        else:
            base_model = build_parameterized_emission_model(
                factory,
                spectral_grid=dataset.observation.spectral_grid,
            )
            cloud_models[dataset.name] = (
                ParameterizedRefractiveIndexCloudEmissionForwardModel(
                    planet=base_model.planet,
                    star=base_model.star,
                    spectral_grid=base_model.spectral_grid,
                    atmosphere_builder=base_model.atmosphere_builder,
                    opacity_provider=base_model.opacity_provider,
                    config=base_model.config,
                    cia_table=base_model.cia_table,
                    geometry=base_model.geometry,
                    cloud=cloud,
                )
            )
    if cloud is None:
        forward_model = build_multi_dataset_emission_model(
            configs, spectral_grids=spectral_grids
        )
        opacity_ids = {
            f"{dataset}:{key}": value
            for dataset, model in forward_model.models.items()
            for key, value in model.opacity_identifiers.items()
        }
    else:
        forward_model = NamedRegionalModels(cloud_models)
        opacity_ids = {
            f"{dataset}:{key}": value
            for dataset, model in cloud_models.items()
            for key, value in model.opacity_identifiers.items()
        }
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
            "cloud_model": config.clouds.model,
            "geometry": geometry_item.model,
        },
        opacity_identifiers=opacity_ids,
    )


def smoke_evaluation(problem: MultiDatasetRetrievalProblem) -> dict[str, float]:
    theta = problem.prior_transform(np.full(problem.ndim, 0.5))
    started = time.perf_counter()
    value = problem.log_likelihood_from_vector(theta)
    elapsed = time.perf_counter() - started
    if not np.isfinite(value) or value <= problem.invalid_loglike:
        raise RuntimeError(
            "prior-midpoint smoke evaluation returned an invalid likelihood"
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
        return run_retrieval(
            problem,
            method="optimal_estimation",
            output_dir=config.outputs.directory / "optimal_estimation",
            **_optimal_estimation_kwargs(config),
        )
    if engine in {"ultranest", "multinest"}:
        return run_retrieval(
            problem,
            method=engine,
            output_dir=config.outputs.directory / engine,
            seed=sampler.seed,
            **_nested_sampler_kwargs(config, engine),
        )
    nested_method = "multinest" if engine.endswith("multinest") else "ultranest"
    return run_oe_then_nested_sampling(
        problem,
        output_dir=config.outputs.directory,
        prior_sigma=sampler.prior_sigma,
        minimum_prior_fraction=sampler.minimum_prior_fraction,
        require_oe_convergence=sampler.require_oe_convergence,
        oe_kwargs=_optimal_estimation_kwargs(config),
        nested_kwargs=_nested_sampler_kwargs(config, nested_method),
        nested_method=nested_method,
        seed=sampler.seed,
    )


def _optimal_estimation_kwargs(config: TaskConfig) -> dict[str, object]:
    sampler = config.sampler
    return {
        "max_iterations": sampler.oe_max_iterations,
        "convergence_tolerance": sampler.oe_convergence_tolerance,
        "finite_difference_fraction": sampler.oe_finite_difference_fraction,
        "damping": sampler.oe_damping,
    }


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
        return f"Inference: optimal_estimation, max_iterations={sampler.oe_max_iterations}"
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
    write_config_snapshot(config, source)
    if mpi_rank() == 0:
        (config.outputs.directory / "smoke_evaluation.json").write_text(
            dumps(smoke, indent=2), encoding="utf-8"
        )
    return smoke


def run_forward_task(config: TaskConfig, source: Path) -> Path:
    initialize_task_directories(config)
    observations = load_observations(config)
    problem = build_problem(config, observations)
    values = {}
    for configured, parameter in zip(
        config.parameters, problem.parameters.parameters, strict=True
    ):
        values[parameter.name] = (
            parameter.midpoint if configured.value is None else configured.value
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
    return target


__all__ = [
    "build_problem",
    "describe_config",
    "load_observations",
    "prepare_opacity",
    "run_forward_task",
    "run_retrieval_task",
    "run_smoke_task",
    "smoke_evaluation",
]
