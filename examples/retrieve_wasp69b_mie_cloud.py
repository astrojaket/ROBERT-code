"""Full-band one-region Mie-cloud retrieval for a configured target.

Two optical-constant modes share the same cloud mass/particle physics:

* ``catalog`` fixes n(lambda), k(lambda) to a selected laboratory material;
* ``direct-nk`` retrieves nodal n(lambda) and log10(k(lambda)).

The catalog run is the practical first inference and evidence baseline. The
direct-nk run is higher-dimensional and should follow once that baseline and
its prior sensitivity are understood.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from robert_exoplanets import (
    CompositionMeanMolecularWeight,
    CorrelatedKOpacityProvider,
    FastChemEquilibriumChemistry,
    LogUniformPrior,
    MultiDatasetGaussianLikelihood,
    MultiDatasetRetrievalProblem,
    OpticalConstantsCatalog,
    ParameterizedClearSkyEmissionFactoryConfig,
    ParameterizedClearSkyEmissionModelConfig,
    ParameterizedRefractiveIndexCloudEmissionForwardModel,
    ParmentierGuillot2014TemperatureProfile,
    PressureGrid,
    RefractiveIndexCloudConfig,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
    build_parameterized_clear_sky_emission_model,
    run_ultranest,
)
from robert_exoplanets.core import Spectrum

from retrieve_wasp69b_nircam_clear import (
    FASTCHEM,
    PLANET,
    PLANET_GRAVITY_M_S2,
    SPECIES,
    STAR,
    TARGET,
    TARGET_SLUG,
    _cia_tables,
    _load_table,
    prepare_opacity_cache,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "optical_constants" / "exo_skryer"
OUTPUT = Path(__file__).resolve().parent / "outputs" / f"{TARGET_SLUG}_mie_cloud"
DIRECT_NK_NODES_MICRON = (2.4, 4.0, 5.5, 7.0, 9.0, 12.0)


@dataclass(frozen=True)
class NamedRegionalModels:
    """Evaluate named cloud models while retaining dataset identity."""

    models: Mapping[str, object]

    def __call__(self, values: Mapping[str, float]) -> Mapping[str, Spectrum]:
        return {name: model(values) for name, model in self.models.items()}


def observations():
    """Return all four named NIRCam/MIRI datasets with a MIRI offset."""

    return TARGET.load_observations(miri_offset_parameter="miri_offset")


def retrieval_parameters(cloud_mode: str) -> RetrievalParameterSet:
    parameters = [
        RetrievalParameter(
            "metallicity", UniformPrior(0.0, 2.0), label="[M/H]", unit="dex"
        ),
        RetrievalParameter("CtoO", UniformPrior(0.05, 1.5), label="C/O"),
        RetrievalParameter("kappa_IR", LogUniformPrior(1.0e-5, 1.0), unit="m2 kg-1"),
        RetrievalParameter("gamma1", LogUniformPrior(1.0e-2, 100.0)),
        RetrievalParameter("gamma2", LogUniformPrior(1.0e-2, 100.0)),
        RetrievalParameter("T_irr", UniformPrior(800.0, 2200.0), unit="K"),
        RetrievalParameter("alpha", UniformPrior(0.0, 1.0)),
        RetrievalParameter("log_cloud_mass_fraction", UniformPrior(-12.0, -2.0)),
        RetrievalParameter("log_cloud_radius_micron", UniformPrior(-3.0, 1.0)),
        # Non-overlapping top/base priors eliminate geometrically invalid slabs.
        RetrievalParameter("log_cloud_top_pressure_bar", UniformPrior(-6.0, -2.0)),
        RetrievalParameter("log_cloud_base_pressure_bar", UniformPrior(-1.0, 2.0)),
        RetrievalParameter("miri_offset", UniformPrior(-500.0e-6, 500.0e-6)),
    ]
    if cloud_mode == "direct-nk":
        for index, wavelength in enumerate(DIRECT_NK_NODES_MICRON):
            parameters.append(
                RetrievalParameter(
                    f"cloud_n_{index}",
                    UniformPrior(1.0, 3.0),
                    metadata={"wavelength_micron": f"{wavelength:.17g}"},
                )
            )
        for index, wavelength in enumerate(DIRECT_NK_NODES_MICRON):
            parameters.append(
                RetrievalParameter(
                    f"cloud_logk_{index}",
                    UniformPrior(-6.0, 0.0),
                    metadata={"wavelength_micron": f"{wavelength:.17g}"},
                )
            )
    return RetrievalParameterSet(tuple(parameters))


def build_problem(
    *,
    cloud_mode: str,
    material: str,
    particle_density_kg_m3: float,
) -> MultiDatasetRetrievalProblem:
    selected_observations = observations()
    pressure = PressureGrid.from_log_centers(
        100.0,
        1.0e-6,
        n_layers=80,
        unit="bar",
        name=f"{PLANET.name} emission pressure grid",
    )
    fixed_index = None
    nodes: tuple[float, ...] = DIRECT_NK_NODES_MICRON
    real_names = tuple(f"cloud_n_{index}" for index in range(len(nodes)))
    imaginary_names = tuple(f"cloud_logk_{index}" for index in range(len(nodes)))
    if cloud_mode == "catalog":
        fixed_index = OpticalConstantsCatalog(CATALOG).load(material)
        nodes = ()
        real_names = ()
        imaginary_names = ()
    cloud = RefractiveIndexCloudConfig(
        refractive_index_wavelength_micron=nodes,
        real_index_parameter_names=real_names,
        log10_imaginary_index_parameter_names=imaginary_names,
        log10_condensate_mass_fraction_parameter="log_cloud_mass_fraction",
        log10_effective_radius_micron_parameter="log_cloud_radius_micron",
        particle_density_kg_m3=particle_density_kg_m3,
        # The first retrieval is monodisperse, matching the direct-n,k model
        # assumption and avoiding an unconverged size-distribution quadrature.
        geometric_stddev=1.0,
        log10_cloud_top_pressure_bar_parameter="log_cloud_top_pressure_bar",
        log10_cloud_base_pressure_bar_parameter="log_cloud_base_pressure_bar",
        quadrature_points=1,
        refractive_index_extrapolation="raise",
        multiple_scattering_backend="sh4",
        fixed_refractive_index=fixed_index,
        metadata={
            "target": PLANET.name,
            "material_selection": material
            if fixed_index is not None
            else "retrieved_n_k",
        },
    )
    cia = _cia_tables()
    models = {}
    opacity_ids = {}
    for dataset in selected_observations.datasets:
        tables = {species: _load_table(dataset.name, species) for species in SPECIES}
        canonical_g = tables["H2O"].g_samples
        canonical_weights = tables["H2O"].g_weights
        for species, table in tuple(tables.items()):
            if not np.allclose(table.g_samples, canonical_g, rtol=0.0, atol=1.0e-8):
                raise ValueError(
                    f"{species} uses a genuinely different correlated-k g grid"
                )
            if not np.allclose(
                table.g_weights, canonical_weights, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(
                    f"{species} uses genuinely different correlated-k weights"
                )
            tables[species] = replace(
                table,
                g_samples=canonical_g,
                g_weights=canonical_weights,
            )
        provider = CorrelatedKOpacityProvider(
            tables,
            name=f"{PLANET.name}-{dataset.name}-ExoMol-pRT-observation-binned",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        factory = ParameterizedClearSkyEmissionFactoryConfig(
            planet=PLANET,
            star=STAR,
            temperature_profile=ParmentierGuillot2014TemperatureProfile(
                gravity=PLANET_GRAVITY_M_S2,
                internal_temperature=100.0,
            ),
            chemistry_model=FastChemEquilibriumChemistry(
                fastchem_path=FASTCHEM,
                metadata={"element_abundances": "asplund_2009"},
            ),
            mean_molecular_weight_model=CompositionMeanMolecularWeight(
                normalization="raw_sum"
            ),
            pressure_grid=pressure,
            cia_table=cia,
            opacity_source=provider,
            opacity_binning=None,
            model=ParameterizedClearSkyEmissionModelConfig(
                opacity_species=SPECIES,
                include_rayleigh=True,
                gas_combination="random_overlap",
                thermal_integration_backend="auto",
                metadata={
                    "target": PLANET.name,
                    "dataset_selection": "NIRCam_plus_MIRI",
                    "dayside_geometry": "one_region_mie_cloud",
                },
            ),
        )
        clear = build_parameterized_clear_sky_emission_model(
            factory,
            spectral_grid=dataset.observation.spectral_grid,
        )
        model = ParameterizedRefractiveIndexCloudEmissionForwardModel(
            planet=clear.planet,
            star=clear.star,
            spectral_grid=clear.spectral_grid,
            atmosphere_builder=clear.atmosphere_builder,
            opacity_provider=clear.opacity_provider,
            config=clear.config,
            cia_table=clear.cia_table,
            geometry=clear.geometry,
            cloud=cloud,
        )
        models[dataset.name] = model
        opacity_ids.update(
            {
                f"{dataset.name}:{key}": value
                for key, value in model.opacity_identifiers.items()
            }
        )
    return MultiDatasetRetrievalProblem(
        name=f"{TARGET_SLUG}-fullband-one-region-mie-{cloud_mode}",
        observations=selected_observations,
        parameters=retrieval_parameters(cloud_mode),
        forward_model=NamedRegionalModels(models),
        likelihood=MultiDatasetGaussianLikelihood(include_normalization=True),
        invalid_loglike=-1.0e100,
        metadata={
            "comparison": f"{PLANET.name}_published_eclipse_spectrum_cloud",
            "cloud_mode": cloud_mode,
            "material": material if cloud_mode == "catalog" else "retrieved_n_k",
            "geometry": "one_region",
            "phase_function": "exact_Mie_moments_SH4_delta_M",
        },
        opacity_identifiers=opacity_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cloud-mode", choices=("catalog", "direct-nk"), default="catalog"
    )
    parser.add_argument("--material", default="MgSiO3")
    parser.add_argument("--particle-density-kg-m3", type=float, default=3200.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--live-points", type=int, default=200)
    parser.add_argument("--max-ncalls", type=int, default=100000)
    parser.add_argument("--mpi-processes", type=int, default=4)
    args = parser.parse_args()

    selected_observations = observations()
    if args.prepare_only:
        prepare_opacity_cache(selected_observations)
        return
    problem = build_problem(
        cloud_mode=args.cloud_mode,
        material=args.material,
        particle_density_kg_m3=args.particle_density_kg_m3,
    )
    if args.smoke_only:
        start = time.perf_counter()
        midpoint = problem.parameters.midpoint_vector()
        loglike = problem.log_likelihood_from_vector(midpoint)
        elapsed = time.perf_counter() - start
        report = {
            "problem": problem.name,
            "ndim": problem.ndim,
            "n_points": problem.observations.n_points,
            "finite_log_likelihood": bool(np.isfinite(loglike)),
            "log_likelihood": float(loglike),
            "elapsed_seconds": elapsed,
            "parameters": problem.parameter_mapping(midpoint),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"smoke_{args.cloud_mode}_{args.material}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    result = run_ultranest(
        problem,
        output_dir=args.output / "ultranest",
        min_num_live_points=args.live_points,
        max_ncalls=args.max_ncalls,
        dlogz=0.5,
        resume="resume",
        show_status=False,
        mpi_nprocs=args.mpi_processes,
        seed=20260712,
    )
    try:
        from mpi4py import MPI

        is_primary = MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        is_primary = True
    if is_primary:
        args.output.mkdir(parents=True, exist_ok=True)
        summary = {
            "problem": problem.name,
            "cloud_mode": args.cloud_mode,
            "material": args.material,
            "converged": result.converged,
            "message": result.message,
            "ncall": result.metadata.get("ncall"),
            "log_evidence": result.log_evidence,
            "log_evidence_error": result.log_evidence_error,
            "best_fit": dict(result.best_fit_parameters),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
