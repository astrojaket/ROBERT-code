"""Python run configuration for the HAT-P-32b FastChem/Madhu comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robert_exoplanets import (
    CompositionMeanMolecularWeight,
    CorrelatedKOpacityProvider,
    ExoKOpacitySource,
    ExoKTableBinning,
    FastChemEquilibriumChemistry,
    MadhusudhanSeager2009TemperatureProfile,
    OptimalEstimationRunConfig,
    ParameterizedClearSkyEmissionFactoryConfig,
    ParameterizedClearSkyEmissionModelConfig,
    Planet,
    PressureGrid,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalRunConfig,
    Star,
    UltraNestRunConfig,
    UniformPrior,
    build_parameterized_clear_sky_emission_model,
    load_emission_observation_npz,
    load_nemesispy_cia_table,
)

if __package__:
    from .hat_p_32b_config import (
        PLANET_MASS_KG,
        PLANET_RADIUS_M,
        STAR_RADIUS_M,
        STAR_TEMPERATURE_K,
    )
else:
    from hat_p_32b_config import (
        PLANET_MASS_KG,
        PLANET_RADIUS_M,
        STAR_RADIUS_M,
        STAR_TEMPERATURE_K,
    )

BUNDLE_ROOT = Path(__file__).resolve().parent / "data" / "hat_p_32b"
RESULTS_DIR = BUNDLE_ROOT / "reference"
OBSERVATION_NPZ = RESULTS_DIR / "quench_study_emission_G395H_spectra_band.npz"
REFERENCE_POSTERIOR_NPZ = RESULTS_DIR / "quench_study_emission_corner_data.npz"
OPACITY_ARCHIVE_DIR = BUNDLE_ROOT / "opacities"
FASTCHEM_PATH = BUNDLE_ROOT / "fastchem"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_fastchem_comparison"

OPACITY_SPECIES = ("H2O", "CO2", "CO", "CH4", "NH3", "HCN")
FASTCHEM_SPECIES = ("H2O1", "C1O2", "C1O1", "C1H4", "H3N1", "C1H1N1_1", "H2", "He")
FASTCHEM_LABELS = (*OPACITY_SPECIES, "H2", "He")


def retrieval_parameters() -> RetrievalParameterSet:
    """Return priors matching the saved comparison retrieval."""

    return RetrievalParameterSet(
        (
            RetrievalParameter("metallicity", UniformPrior(-1.0, 2.0), label="[M/H]", unit="dex"),
            RetrievalParameter("CtoO", UniformPrior(0.0, 1.0), label="C/O"),
            RetrievalParameter("P1", UniformPrior(-6.0, 2.0), unit="log10(bar)"),
            RetrievalParameter("P2", UniformPrior(-6.0, 2.0), unit="log10(bar)"),
            RetrievalParameter("P3", UniformPrior(-2.0, 2.0), unit="log10(bar)"),
            RetrievalParameter("T0", UniformPrior(300.0, 2000.0), unit="K"),
            RetrievalParameter("alpha1", UniformPrior(0.02, 2.0)),
            RetrievalParameter("alpha2", UniformPrior(0.02, 2.0)),
        )
    )


def reference_map_parameters() -> dict[str, float]:
    """Load the saved comparison MAP state by parameter name."""

    with np.load(REFERENCE_POSTERIOR_NPZ, allow_pickle=False) as archive:
        names = tuple(str(name) for name in archive["names"])
        values = np.asarray(archive["truths_map_raw"], dtype=float)
    return dict(zip(names, values, strict=True))


def make_model_config(
    *,
    kta_dir: str | Path | None = None,
    opacity_archive_dir: str | Path = OPACITY_ARCHIVE_DIR,
    fastchem_path: str | Path = FASTCHEM_PATH,
    exok_num: int = 300,
    include_rayleigh: bool = False,
    pressure_top_bar: float = 1.0e-6,
    pressure_bottom_bar: float = 100.0,
    n_layers: int = 100,
) -> ParameterizedClearSkyEmissionFactoryConfig:
    """Return the FastChem/Madhusudhan-Seager forward-model configuration."""

    if kta_dir is None:
        archive_root = Path(opacity_archive_dir).expanduser()
        opacity_source = CorrelatedKOpacityProvider.from_robert_archives(
            {
                species: archive_root / f"{species}.robert-opacity.npz"
                for species in OPACITY_SPECIES
            },
            name="HAT-P-32b-exo-k-binned",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        opacity_binning = None
    else:
        opacity_source = ExoKOpacitySource(
            species=OPACITY_SPECIES,
            directory=Path(kta_dir).expanduser(),
            filename_pattern="*_emission_R1000.kta",
            interpolation="log_pressure_temperature_log_k_clip",
            nonfinite_policy="floor",
        )
        opacity_binning = ExoKTableBinning(num=exok_num)

    return ParameterizedClearSkyEmissionFactoryConfig(
        planet=Planet(
            name="HAT-P-32b",
            radius_m=PLANET_RADIUS_M,
            mass_kg=PLANET_MASS_KG,
        ),
        star=Star(
            name="HAT-P-32",
            radius_m=STAR_RADIUS_M,
            effective_temperature_k=STAR_TEMPERATURE_K,
        ),
        temperature_profile=MadhusudhanSeager2009TemperatureProfile(
            pressure_unit="bar",
            reference_pressure=1.0e-6,
        ),
        chemistry_model=FastChemEquilibriumChemistry(
            fastchem_path=Path(fastchem_path).expanduser(),
            fastchem_species=FASTCHEM_SPECIES,
            labels=FASTCHEM_LABELS,
            metadata={"element_abundances": "asplund_2009"},
        ),
        mean_molecular_weight_model=CompositionMeanMolecularWeight(
            normalization="raw_sum"
        ),
        pressure_grid=PressureGrid.from_log_centers(
            pressure_bottom_bar,
            pressure_top_bar,
            n_layers=n_layers,
            unit="bar",
            name="HAT-P-32b NEMESIS comparison pressure grid",
        ),
        cia_table=load_nemesispy_cia_table(),
        opacity_source=opacity_source,
        opacity_binning=opacity_binning,
        model=ParameterizedClearSkyEmissionModelConfig(
            opacity_species=OPACITY_SPECIES,
            include_rayleigh=include_rayleigh,
            gas_combination="random_overlap",
            thermal_integration_backend="auto",
            metadata={
                "target": "HAT-P-32b",
                "comparison_reference": str(RESULTS_DIR),
                "comparison_physics": "FastChem+MadhusudhanSeager2009",
                "cia": "NemesisPy_v1.0.1_exocia_hitran12_200-3800K",
                "opacity_boundary_policy": "NemesisPy-compatible pressure-temperature clipping",
                "opacity_bundle": str(OPACITY_ARCHIVE_DIR),
            },
        ),
    )


def make_run_config(
    *,
    method: str = "ultranest",
    output_dir: str | Path = OUTPUT_DIR,
    live_points: int = 400,
    max_ncalls: int | None = 100000,
    dlogz: float = 0.5,
    resume: str = "resume",
    seed: int = 20260710,
    mpi_nprocs: int | None = None,
    pressure_top_bar: float = 1.0e-6,
    pressure_bottom_bar: float = 100.0,
    n_layers: int = 100,
) -> RetrievalRunConfig:
    """Build the complete comparison run from ordinary Python objects."""

    observation = load_emission_observation_npz(
        OBSERVATION_NPZ,
        instrument="JWST/NIRSpec G395H",
    )
    forward_model = build_parameterized_clear_sky_emission_model(
        make_model_config(
            pressure_top_bar=pressure_top_bar,
            pressure_bottom_bar=pressure_bottom_bar,
            n_layers=n_layers,
        ),
        spectral_grid=observation.spectral_grid,
    )
    normalized_method = method.strip().lower().replace("-", "_")
    if normalized_method in {"optimal_estimation", "oe"}:
        map_parameters = reference_map_parameters()
        initial_state = tuple(map_parameters[name] for name in retrieval_parameters().names)
        inference = OptimalEstimationRunConfig(
            initial_state=initial_state,
            max_iterations=4,
            damping=1.0,
        )
    elif normalized_method == "ultranest":
        inference = UltraNestRunConfig(
            min_num_live_points=live_points,
            max_ncalls=max_ncalls,
            dlogz=dlogz,
            resume=resume,
            seed=seed,
            mpi_nprocs=mpi_nprocs,
        )
    else:
        raise ValueError("method must be 'optimal_estimation' or 'ultranest'")
    return RetrievalRunConfig(
        name="hat-p-32b-fastchem-madhu-comparison",
        observation=observation,
        parameters=retrieval_parameters(),
        forward_model=forward_model,
        inference=inference,
        output_dir=output_dir,
        metadata={"configuration": "examples.hat_p_32b_fastchem_config"},
    )


__all__ = [
    "BUNDLE_ROOT",
    "FASTCHEM_PATH",
    "OBSERVATION_NPZ",
    "OPACITY_ARCHIVE_DIR",
    "OUTPUT_DIR",
    "REFERENCE_POSTERIOR_NPZ",
    "make_model_config",
    "make_run_config",
    "reference_map_parameters",
    "retrieval_parameters",
]
