"""Shared NIRCam cloud-free builders and the current YAML entry point.

WASP-69b is the default configuration. Following ROBERT's default
multi-instrument workflow, opacity is recompressed into each mode's published
observation bins with exo_k before inference. The user supplies an ExoMol KTA
root and selects its model resolution explicitly. Use the strict YAML runner
for retrieval execution; the functions in this module remain reusable by
benchmarks that compare forward-model construction paths.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import os
from pathlib import Path
import runpy
import sys
import tempfile
from typing import Sequence

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
    CiaTable,
    CompositionMeanMolecularWeight,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    FastChemEquilibriumChemistry,
    LogUniformPrior,
    MultiDatasetGaussianLikelihood,
    MultiDatasetRetrievalProblem,
    ObservationCollection,
    ParameterizedEmissionFactoryConfig,
    ParameterizedEmissionModelConfig,
    ParmentierGuillot2014TemperatureProfile,
    PressureGrid,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
    build_multi_dataset_emission_model,
    load_nemesispy_cia_table,
)


def _target_configuration():
    module_name = os.environ.get("ROBERT_TARGET_CONFIG", "examples.wasp69b_target")
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        if "." not in module_name:
            raise
        return import_module(module_name.rsplit(".", 1)[-1])


TARGET = _target_configuration()
PLANET = TARGET.PLANET
STAR = TARGET.STAR
PLANET_GRAVITY_M_S2 = TARGET.PLANET_GRAVITY_M_S2
TARGET_SLUG = TARGET.TARGET_SLUG

ROOT = Path(__file__).resolve().parents[1]
DATA = TARGET.DATA_DIRECTORY
FASTCHEM = Path(
    os.environ.get(
        "ROBERT_FASTCHEM_DATA",
        ROOT / "data" / "chemistry" / "fastchem",
    )
).expanduser()
CACHE = TARGET.CACHE_DIRECTORY
SPECIES = ("H2O", "CO2", "CO", "CH4", "NH3", "HCN")
OPACITY_RESOLUTIONS = ("R1000", "R15000")
DEFAULT_OPACITY_RESOLUTION = "R1000"
DEFAULT_CONFIG = ROOT / "configurations" / "targets/WASP-69b/wasp69b_cloud_free_nircam_pg14_R1000.yaml"


def nircam_observations() -> ObservationCollection:
    full = TARGET.load_observations(miri_offset_parameter=None)
    return ObservationCollection(
        datasets=tuple(dataset for dataset in full.datasets if dataset.name != "lrs"),
        name=f"{PLANET.name} NIRCam-only published spectrum",
        metadata={**dict(full.metadata), "selection": "NIRCam only"},
    )


def opacity_cache_directory(resolution: str) -> Path:
    """Return the target cache directory for one explicit KTA resolution."""

    normalized = str(resolution).strip().upper()
    if normalized not in OPACITY_RESOLUTIONS:
        raise ValueError(
            f"resolution must be one of {', '.join(OPACITY_RESOLUTIONS)}"
        )
    return CACHE / normalized


def prepare_opacity_cache(
    observations: ObservationCollection,
    *,
    kta_path: str | Path,
    resolution: str = DEFAULT_OPACITY_RESOLUTION,
) -> None:
    """Recompress selected user-supplied KTA tables into published bins."""

    cache = opacity_cache_directory(resolution)
    cache.mkdir(parents=True, exist_ok=True)
    for species in SPECIES:
        provider = CorrelatedKOpacityProvider.from_exomol_kta_directory(
            kta_path,
            species=(species,),
            resolution=resolution,
            name=f"ExoMol-{resolution}-{species}",
            interpolation="log_pressure_temperature_log_k_clip",
            nonfinite_policy="floor",
        )
        table = provider.tables[species]
        source = Path(str(table.metadata["source_path"]))
        source_sha = str(table.metadata["checksum_sha256"])
        single_species_provider = CorrelatedKOpacityProvider(
            {species: table},
            interpolation="log_pressure_temperature_log_k_clip",
        )
        for dataset in observations.datasets:
            target = cache / f"{dataset.name}_{species}.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    if (
                        str(saved["source_sha256"]) == source_sha
                        and str(saved["opacity_resolution"]) == resolution
                    ):
                        continue
            binned = single_species_provider.bin_to_spectral_grid(
                dataset.observation.spectral_grid,
                num=300,
                use_rebin=False,
                remove_zeros=True,
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
                opacity_resolution=resolution,
                spectral_preparation="exo_k_bin_down_cp_num300",
            )


def _load_table(
    dataset: str,
    species: str,
    resolution: str = DEFAULT_OPACITY_RESOLUTION,
) -> CorrelatedKTable:
    path = opacity_cache_directory(resolution) / f"{dataset}_{species}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"prepared opacity cache is missing: {path}; run this command "
            "with --prepare-only first"
        )
    with np.load(path, allow_pickle=False) as saved:
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
                "opacity_resolution": str(saved["opacity_resolution"]),
                "spectral_preparation": str(saved["spectral_preparation"]),
            },
        )


def _cia_tables() -> CiaTable:
    """Load ROBERT's vendored H2-H2/H2-He CIA table."""

    return load_nemesispy_cia_table()


def parameters() -> RetrievalParameterSet:
    return RetrievalParameterSet(
        (
            RetrievalParameter(
                "metallicity",
                UniformPrior(-1.0, 2.0),
                label="log10(Z/Z_sun)",
                unit="dex",
            ),
            RetrievalParameter("CtoO", UniformPrior(0.0, 1.0), label="C/O"),
            RetrievalParameter(
                "kappa_IR", LogUniformPrior(1.0e-5, 1.0), unit="m2 kg-1"
            ),
            RetrievalParameter("gamma1", LogUniformPrior(1.0e-4, 100.0)),
            RetrievalParameter("gamma2", LogUniformPrior(1.0e-4, 100.0)),
            RetrievalParameter("T_irr", UniformPrior(800.0, 2200.0), unit="K"),
            RetrievalParameter("alpha", UniformPrior(0.0, 1.0)),
        )
    )


def build_problem(
    observations: ObservationCollection,
    *,
    opacity_resolution: str = DEFAULT_OPACITY_RESOLUTION,
) -> MultiDatasetRetrievalProblem:
    pressure = PressureGrid.from_log_centers(
        100.0,
        1.0e-6,
        n_layers=80,
        unit="bar",
        name=f"{PLANET.name} emission pressure grid",
    )
    cia = _cia_tables()
    temperature_profile = ParmentierGuillot2014TemperatureProfile(
        gravity=PLANET_GRAVITY_M_S2,
        internal_temperature=100.0,
    )
    chemistry_model = FastChemEquilibriumChemistry(
        fastchem_path=FASTCHEM,
        metadata={"element_abundances": "asplund_2009"},
    )
    mean_molecular_weight_model = CompositionMeanMolecularWeight(
        normalization="raw_sum"
    )
    model_config = ParameterizedEmissionModelConfig(
        opacity_species=SPECIES,
        include_rayleigh=True,
        gas_combination="random_overlap",
        thermal_integration_backend="auto",
        metadata={
            "target": PLANET.name,
            "dataset_selection": ",".join(observations.names),
            "dayside_geometry": "one_region_cloud_free",
            "paper_analogue": "not_exact_EGP_grid_reproduction",
        },
    )
    configs = {}
    spectral_grids = {}
    for dataset in observations.datasets:
        tables = {
            species: _load_table(dataset.name, species, opacity_resolution)
            for species in SPECIES
        }
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
            name=(
                f"{PLANET.name}-{dataset.name}-ExoMol-{opacity_resolution}-binned"
            ),
            interpolation="log_pressure_temperature_log_k_clip",
        )
        configs[dataset.name] = ParameterizedEmissionFactoryConfig(
            planet=PLANET,
            star=STAR,
            temperature_profile=temperature_profile,
            chemistry_model=chemistry_model,
            mean_molecular_weight_model=mean_molecular_weight_model,
            pressure_grid=pressure,
            cia_table=cia,
            opacity_source=provider,
            opacity_binning=None,
            model=model_config,
        )
        spectral_grids[dataset.name] = dataset.observation.spectral_grid
    forward_model = build_multi_dataset_emission_model(
        configs,
        spectral_grids=spectral_grids,
    )
    opacity_ids = {}
    for dataset_name, model in forward_model.models.items():
        opacity_ids.update(
            {
                f"{dataset_name}:{key}": value
                for key, value in model.opacity_identifiers.items()
            }
        )
    return MultiDatasetRetrievalProblem(
        name=f"{TARGET_SLUG}-nircam-cloud-free-one-region",
        observations=observations,
        parameters=parameters(),
        forward_model=forward_model,
        likelihood=MultiDatasetGaussianLikelihood(include_normalization=True),
        invalid_loglike=-1.0e100,
        metadata={
            "comparison": f"{PLANET.name}_published_eclipse_spectrum",
            "difference": "NIRCam-only equilibrium chemistry with PG14 analytic TP",
            "opacity_resolution": opacity_resolution,
        },
        opacity_identifiers=opacity_ids,
    )


def _run_current_configuration(
    default_config: Path = DEFAULT_CONFIG,
    argv: Sequence[str] | None = None,
) -> None:
    """Delegate execution to the repository's strict YAML retrieval runner."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(
        argument == "--config" or argument.startswith("--config=")
        for argument in arguments
    ):
        arguments = ["--config", str(default_config), *arguments]
    runner = ROOT / "run_retrieval.py"
    previous_argv = sys.argv
    sys.argv = [str(runner), *arguments]
    try:
        runpy.run_path(str(runner), run_name="__main__")
    finally:
        sys.argv = previous_argv


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected current YAML retrieval workflow."""

    _run_current_configuration(argv=argv)


if __name__ == "__main__":
    main()
