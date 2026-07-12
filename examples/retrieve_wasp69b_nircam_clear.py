"""NIRCam-only clear one-region retrieval for WASP-69b.

This is a ROBERT validation analogue, not an exact reproduction of the paper's
full 2--12 micron CHIMERA/EGP-grid retrieval. Following ROBERT's default
multi-instrument retrieval workflow, opacity is recompressed into each mode's
published observation bins with exo_k before inference. Native-resolution
spectra are reserved for plotting and diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ParameterizedClearSkyEmissionFactoryConfig,
    ParameterizedClearSkyEmissionModelConfig,
    ParmentierGuillot2014TemperatureProfile,
    Planet,
    PressureGrid,
    RetrievalParameter,
    RetrievalParameterSet,
    Star,
    UniformPrior,
    build_multi_dataset_emission_model,
    load_schlawin2024_wasp69b,
    run_ultranest,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wasp69b_schlawin2024"
PRT_DATA = ROOT / "opacity_data" / "petitRADTRANS" / "input_data"
FASTCHEM = ROOT / "examples" / "data" / "hat_p_32b" / "fastchem"
CACHE = ROOT / "opacity_data" / "wasp69b_nircam_observation_bins"
OUTPUT = Path(__file__).resolve().parent / "outputs" / "wasp69b_nircam_clear"
SPECIES = ("H2O", "CO2", "CO", "CH4", "NH3", "HCN")
PATTERNS = {
    "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
    "CO2": "*UCL-4000*.ktable.petitRADTRANS.h5",
    "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
    "CH4": "*YT34to10*.ktable.petitRADTRANS.h5",
    "NH3": "*CoYuTe*.ktable.petitRADTRANS.h5",
    "HCN": "*Harris*.ktable.petitRADTRANS.h5",
}
RJUP_M = 7.1492e7
MJUP_KG = 1.89813e27
RSUN_M = 6.957e8
CALLS_PER_ATTEMPT = 5000


def nircam_observations() -> ObservationCollection:
    full = load_schlawin2024_wasp69b(DATA, miri_offset_parameter=None)
    return ObservationCollection(
        datasets=tuple(dataset for dataset in full.datasets if dataset.name != "lrs"),
        name="WASP-69b NIRCam-only Schlawin et al. 2024",
        metadata={**dict(full.metadata), "selection": "NIRCam only"},
    )


def prepare_opacity_cache(observations: ObservationCollection) -> None:
    """Recompress native pRT/ExoMol correlated-k data into published bins."""

    CACHE.mkdir(parents=True, exist_ok=True)
    for species in SPECIES:
        source = next(PRT_DATA.rglob(PATTERNS[species]))
        source_sha = _sha256(source)
        table = CorrelatedKTable.from_petitradtrans_hdf(source, species=species)
        provider = CorrelatedKOpacityProvider(
            {species: table},
            interpolation="log_pressure_temperature_log_k_clip",
        )
        for dataset in observations.datasets:
            target = CACHE / f"{dataset.name}_{species}.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    if str(saved["source_sha256"]) == source_sha:
                        continue
            binned = provider.bin_to_spectral_grid(
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
                spectral_preparation="exo_k_bin_down_cp_num300",
            )


def _load_table(dataset: str, species: str) -> CorrelatedKTable:
    path = CACHE / f"{dataset}_{species}.npz"
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
                "spectral_preparation": str(saved["spectral_preparation"]),
            },
        )


def _cia_tables() -> tuple[CiaTable, CiaTable]:
    h2_h2 = next(PRT_DATA.rglob("*H2--H2*.ciatable.petitRADTRANS.h5"))
    h2_he = next(PRT_DATA.rglob("*H2--He*.ciatable.petitRADTRANS.h5"))
    return (
        CiaTable.from_petitradtrans_hdf(h2_h2, collision_pair="H2-H2"),
        CiaTable.from_petitradtrans_hdf(h2_he, collision_pair="H2-He"),
    )


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


def build_problem(observations: ObservationCollection) -> MultiDatasetRetrievalProblem:
    gravity = 6.67430e-11 * (0.26 * MJUP_KG) / (1.06 * RJUP_M) ** 2
    pressure = PressureGrid.from_log_centers(
        100.0,
        1.0e-6,
        n_layers=80,
        unit="bar",
        name="WASP-69b emission pressure grid",
    )
    cia = _cia_tables()
    planet = Planet(
        name="WASP-69b",
        radius_m=1.06 * RJUP_M,
        mass_kg=0.26 * MJUP_KG,
    )
    star = Star(
        name="WASP-69",
        radius_m=0.813 * RSUN_M,
        effective_temperature_k=4750.0,
    )
    temperature_profile = ParmentierGuillot2014TemperatureProfile(
        gravity=gravity,
        internal_temperature=100.0,
    )
    chemistry_model = FastChemEquilibriumChemistry(
        fastchem_path=FASTCHEM,
        metadata={"element_abundances": "asplund_2009"},
    )
    mean_molecular_weight_model = CompositionMeanMolecularWeight(
        normalization="raw_sum"
    )
    model_config = ParameterizedClearSkyEmissionModelConfig(
        opacity_species=SPECIES,
        include_rayleigh=True,
        gas_combination="random_overlap",
        thermal_integration_backend="auto",
        metadata={
            "target": "WASP-69b",
            "dataset_selection": ",".join(observations.names),
            "dayside_geometry": "one_region_clear",
            "paper_analogue": "not_exact_EGP_grid_reproduction",
        },
    )
    configs = {}
    spectral_grids = {}
    for dataset in observations.datasets:
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
            name=f"WASP-69b-NIRCam-{dataset.name}-ExoMol-pRT-R1000-binned",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        configs[dataset.name] = ParameterizedClearSkyEmissionFactoryConfig(
            planet=planet,
            star=star,
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
        name="wasp69b-nircam-clear-one-region",
        observations=observations,
        parameters=parameters(),
        forward_model=forward_model,
        likelihood=MultiDatasetGaussianLikelihood(include_normalization=True),
        invalid_loglike=-1.0e100,
        metadata={
            "comparison": "Schlawin_et_al_2024_clear_one_region",
            "difference": "NIRCam-only and PG14 analytic TP instead of full-band EGP RCE grid",
        },
        opacity_identifiers=opacity_ids,
    )


def _write_products(
    problem: MultiDatasetRetrievalProblem,
    result,
    output: Path,
    *,
    live_points: int = 50,
    max_ncalls: int = 5000,
    mpi_processes: int = 2,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    weights = np.array(result.weights, dtype=float, copy=True)
    weights /= np.sum(weights)
    rng = np.random.default_rng(20260711)
    draw_indices = rng.choice(
        result.samples.shape[0], size=300, replace=True, p=weights
    )
    draws = result.samples[draw_indices]
    spectra = {name: [] for name in problem.observations.names}
    for draw in draws:
        prediction = problem.model_spectra(draw)
        for name in spectra:
            spectra[name].append(prediction[name].values)
    envelopes = {
        name: np.percentile(np.asarray(values), [16.0, 50.0, 84.0], axis=0)
        for name, values in spectra.items()
    }
    best = result.best_fit_parameters
    best_spectra = problem.model_spectra(best)
    chi2 = 0.0
    for dataset in problem.observations.datasets:
        residual = (
            dataset.observation.flux - best_spectra[dataset.name].values
        ) / dataset.observation.uncertainty
        chi2 += float(np.sum(residual**2))
    dof = problem.observations.n_points - problem.ndim
    quantiles = {}
    for index, name in enumerate(problem.parameter_names):
        order = np.argsort(result.samples[:, index])
        values = result.samples[order, index]
        cumulative = np.cumsum(weights[order])
        quantiles[name] = np.interp([0.16, 0.5, 0.84], cumulative, values).tolist()
    summary = {
        "selection": str(
            problem.metadata.get(
                "dataset_selection",
                "NIRCam only (F322W2, overlap average, F444W)",
            )
        ),
        "n_points": problem.observations.n_points,
        "n_parameters": problem.ndim,
        "live_points": live_points,
        "max_ncalls": max_ncalls,
        "mpi_processes": mpi_processes,
        "converged": result.converged,
        "message": result.message,
        "log_evidence": result.log_evidence,
        "log_evidence_error": result.log_evidence_error,
        "best_fit": dict(best),
        "posterior_16_50_84": quantiles,
        "chi_squared": chi2,
        "degrees_of_freedom": dof,
        "reduced_chi_squared": chi2 / dof,
        "paper_full_band_clear_one_region": {
            "metallicity_16_50_84": [1.28, 1.30, 1.32],
            "CtoO_16_50_84": [0.10, 0.11, 0.13],
            "reduced_chi_squared": 14.5,
            "log_evidence": -588,
        },
        "comparison_warning": (
            "Paper values use 2-12 micron CHIMERA plus interpolated EGP RCE profiles; "
            "this ROBERT run uses its recorded dataset selection and a PG14 analytic "
            "TP profile, so evidence values are not directly comparable."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "posterior_products.npz",
        samples=result.samples,
        weights=weights,
        log_likelihood=result.log_likelihood,
        **{f"{name}_q16": value[0] for name, value in envelopes.items()},
        **{f"{name}_q50": value[1] for name, value in envelopes.items()},
        **{f"{name}_q84": value[2] for name, value in envelopes.items()},
    )
    _plot(
        problem.observations,
        envelopes,
        best_spectra,
        summary,
        output / "spectrum_1sigma.png",
    )


def _plot(observations, envelopes, best_spectra, summary, path: Path) -> None:
    colors = {
        "f322w2": "#20639b",
        "avg": "#7a5195",
        "f444w": "#ef5675",
        "lrs": "#2ca25f",
    }
    fig, (axis, residual_axis) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for dataset in observations.datasets:
        name = dataset.name
        obs = dataset.observation
        q16, q50, q84 = envelopes[name]
        color = colors[name]
        axis.fill_between(
            obs.wavelength, 1.0e6 * q16, 1.0e6 * q84, color=color, alpha=0.22
        )
        axis.plot(obs.wavelength, 1.0e6 * q50, color=color, linewidth=1.5)
        axis.errorbar(
            obs.wavelength,
            1.0e6 * obs.flux,
            yerr=1.0e6 * obs.uncertainty,
            fmt=".",
            color=color,
            alpha=0.75,
            markersize=3,
            label=obs.instrument,
        )
        residual = (obs.flux - best_spectra[name].values) / obs.uncertainty
        residual_axis.plot(obs.wavelength, residual, ".", color=color, markersize=3)
    axis.set_ylabel("Eclipse depth (ppm)")
    axis.set_title("WASP-69b: ROBERT clear one-region retrieval")
    axis.text(
        0.02,
        0.97,
        "Shading: posterior 68% (1-sigma) spectrum envelope\n"
        f"ROBERT reduced chi-square = {summary['reduced_chi_squared']:.2f}\n"
        "Paper full 2-12 micron clear model: reduced chi-square = 14.5",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    axis.legend(loc="lower right", fontsize=8)
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.axhline(1.0, color="0.6", linewidth=0.6, linestyle="--")
    residual_axis.axhline(-1.0, color="0.6", linewidth=0.6, linestyle="--")
    residual_axis.set_ylabel("Residual / sigma")
    residual_axis.set_xlabel("Wavelength (micron)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _next_cumulative_call_limit(log_dir: Path) -> int:
    """Advance UltraNest's cumulative call ceiling in fixed-size resume chunks."""

    status_path = log_dir / "sampler_status.json"
    if not status_path.exists():
        return CALLS_PER_ATTEMPT
    status = json.loads(status_path.read_text(encoding="utf-8"))
    completed_calls = max(int(status.get("ncall", 0)), 0)
    return completed_calls + CALLS_PER_ATTEMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    observations = nircam_observations()
    if args.prepare_only:
        prepare_opacity_cache(observations)
        return
    problem = build_problem(observations)
    max_ncalls = _next_cumulative_call_limit(args.output / "ultranest")
    result = run_ultranest(
        problem,
        output_dir=args.output / "ultranest",
        min_num_live_points=50,
        max_ncalls=max_ncalls,
        dlogz=0.5,
        resume="resume",
        show_status=False,
        mpi_nprocs=2,
        seed=20260711,
    )
    try:
        from mpi4py import MPI

        is_primary = MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        is_primary = True
    if is_primary:
        _write_products(
            problem,
            result,
            args.output,
            live_points=50,
            max_ncalls=max_ncalls,
            mpi_processes=2,
        )


if __name__ == "__main__":
    main()
