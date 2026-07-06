"""Template retrieval using ROBERT's HAT-P-32b emission RT path."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    CorrelatedKOpacityProvider,
    PressureGrid,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    SpectralGrid,
    TabulatedTemperatureProfile,
    UniformPrior,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    load_emission_observation_npz,
    rayleigh_scattering_optical_depth,
    run_retrieval,
    solve_clear_sky_emission,
)

DEFAULT_OBSERVATION_NPZ = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "Retrieval_Results"
    / "HAT-P-32b"
    / "quench_study_emission_G395H_spectra_band.npz"
)
DEFAULT_PT_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "PTprofiles-Teq_1800-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_1.8-TiOVO_false-daysideavg-w_mu_area.csv"
)
DEFAULT_KTA_DIR = Path.home() / "Dropbox" / "PostDoc4" / "Emission_Example" / "HAT-P-32b" / "kta_temp"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_rt_retrieval"

R_SUN_M = 6.957e8
R_JUP_M = 7.1492e7
M_JUP_KG = 1.898e27
GRAVITATIONAL_CONSTANT = 6.67430e-11
DEFAULT_PLANET_RADIUS_M = 1.98 * R_JUP_M
DEFAULT_PLANET_MASS_KG = 0.68 * M_JUP_KG
DEFAULT_STAR_RADIUS_M = 1.32 * R_SUN_M
DEFAULT_STAR_TEMPERATURE_K = 6001.0
DEFAULT_GRAVITY_M_S2 = GRAVITATIONAL_CONSTANT * DEFAULT_PLANET_MASS_KG / DEFAULT_PLANET_RADIUS_M**2
RUNTIME_NONFINITE_FILL_VALUE = 1.0e-300


class HatP32bRtForwardModel:
    """Callable RT forward model for early retrieval tests."""

    def __init__(self, observation, *, pt_csv: Path, kta_dir: Path, include_rayleigh: bool = True) -> None:
        self.observation = observation
        self.include_rayleigh = include_rayleigh
        self.provider = CorrelatedKOpacityProvider.from_kta_paths(
            {"H2O": kta_dir / "H2O_emission_R1000.kta"},
            interpolation="log_pressure_temperature_log_k",
            nonfinite_policy="floor",
            nonfinite_fill_value=RUNTIME_NONFINITE_FILL_VALUE,
        )
        table = self.provider.tables["H2O"]
        self.pressure_grid = _pressure_grid_from_centers(table.pressure_bar)
        self.spectral_grid = SpectralGrid.from_array(
            table.wavelength_micron,
            unit="micron",
            role="opacity",
            name="HAT-P-32b native k-table grid",
        )
        profile = TabulatedTemperatureProfile.from_csv(pt_csv, name="HAT-P-32b retrieval PT")
        self.base_temperature = profile.evaluate({}, self.pressure_grid)
        self.prepared_opacity = self.provider.prepare(self.spectral_grid, self.pressure_grid, species=("H2O",))
        self.geometry = gauss_legendre_disk_geometry(4)

    def __call__(self, parameters: dict[str, float]):
        h2o = float(np.power(10.0, parameters["log_h2o"]))
        background = max(1.0 - h2o, 1.0e-12)
        composition = {
            "H2O": np.full(self.pressure_grid.n_layers, h2o),
            "H2": np.full(self.pressure_grid.n_layers, 0.84 * background),
            "He": np.full(self.pressure_grid.n_layers, 0.16 * background),
        }
        atmosphere = AtmosphereState(
            pressure_grid=self.pressure_grid,
            temperature=self.base_temperature + float(parameters["temperature_offset"]),
            composition=composition,
            mean_molecular_weight=np.full(self.pressure_grid.n_layers, 2.3),
        )
        evaluated = self.provider.evaluate(atmosphere, self.prepared_opacity)
        gas_tau = assemble_gas_optical_depth(
            atmosphere,
            evaluated,
            gravity_m_s2=DEFAULT_GRAVITY_M_S2,
            gas_combination="random_overlap",
        )
        additional_optical_depths = []
        if self.include_rayleigh:
            additional_optical_depths.append(rayleigh_scattering_optical_depth(gas_tau))
        result = solve_clear_sky_emission(
            gas_tau,
            geometry=self.geometry,
            additional_optical_depths=additional_optical_depths,
            planet_radius_m=DEFAULT_PLANET_RADIUS_M * float(parameters["radius_scale"]),
            star_radius_m=DEFAULT_STAR_RADIUS_M,
            star_temperature_k=DEFAULT_STAR_TEMPERATURE_K,
            thermal_integration_backend="auto",
        )
        if result.eclipse_depth is None:
            raise RuntimeError("RT result did not include eclipse depth")
        native_wavelength = np.array(result.eclipse_depth.spectral_grid.values, dtype=float, copy=True)
        order = np.argsort(native_wavelength)
        observed_values = np.interp(
            self.observation.wavelength,
            native_wavelength[order],
            result.eclipse_depth.values[order],
        )
        return result.eclipse_depth.__class__.from_arrays(
            self.observation.wavelength,
            observed_values,
            unit=result.eclipse_depth.unit,
            observable=result.eclipse_depth.observable,
            wavelength_unit=self.observation.wavelength_unit,
        )


def main() -> dict[str, object]:
    args = _parser().parse_args()
    rank, size = _mpi_rank_size()
    if args.method != "ultranest" and rank != 0:
        return {}

    observation = load_emission_observation_npz(Path(args.observation_npz), instrument="JWST/NIRSpec G395H")
    forward_model = HatP32bRtForwardModel(
        observation,
        pt_csv=Path(args.pt_csv).expanduser(),
        kta_dir=Path(args.kta_dir).expanduser(),
        include_rayleigh=not args.no_rayleigh,
    )
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("log_h2o", UniformPrior(*args.log_h2o_prior)),
            RetrievalParameter("temperature_offset", UniformPrior(*args.temperature_offset_prior), unit="K"),
            RetrievalParameter("radius_scale", UniformPrior(*args.radius_scale_prior)),
        )
    )
    problem = RetrievalProblem(
        name="hat-p-32b-rt-retrieval-template",
        observation=observation,
        parameters=parameters,
        forward_model=forward_model,
        metadata={"rt": "clear_sky_emission", "opacity": "H2O_correlated_k"},
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "optimal_estimation":
        result = run_retrieval(problem, method="optimal_estimation", max_iterations=args.max_iterations)
    else:
        result = run_retrieval(
            problem,
            method="ultranest",
            output_dir=output_dir / "ultranest",
            min_num_live_points=args.live_points,
            max_ncalls=args.max_ncalls,
            dlogz=args.dlogz,
            mpi_nprocs=size,
            show_status=(rank == 0),
        )
    if rank != 0:
        return {}
    model = problem.model_spectrum(result.best_fit_parameters)
    report = _report(problem, result, model, mpi_size=size)
    json_path = output_dir / f"hat_p_32b_rt_{args.method}_retrieval.json"
    plot_path = output_dir / f"hat_p_32b_rt_{args.method}_retrieval.png"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(plot_path, observation, model)
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    return report


def _report(problem: RetrievalProblem, result, model, *, mpi_size: int) -> dict[str, object]:
    residual = problem.observation.flux - model.values
    chi2 = float(np.sum(np.square(residual / problem.observation.uncertainty)))
    return {
        "problem": problem.name,
        "parameters": result.best_fit_parameters,
        "chi2": chi2,
        "reduced_chi2": chi2 / max(1, problem.observation.n_points - problem.ndim),
        "log_likelihood": _result_log_likelihood(result),
        "mpi_size": mpi_size,
    }


def _result_log_likelihood(result) -> float:
    values = np.asarray(result.log_likelihood, dtype=float)
    if values.ndim == 0:
        return float(values)
    return float(np.max(values))


def _plot(path: Path, observation, model) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.errorbar(
        observation.wavelength,
        observation.flux * 1.0e6,
        yerr=observation.uncertainty * 1.0e6,
        fmt=".",
        color="#222222",
        ecolor="#999999",
        elinewidth=0.8,
        markersize=4,
        label="Observed G395H",
    )
    ax.plot(model.spectral_grid.values, model.values * 1.0e6, color="#f58518", linewidth=1.7, label="ROBERT RT")
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Eclipse depth [ppm]")
    ax.set_title("HAT-P-32b RT Retrieval Template")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _pressure_grid_from_centers(centers: np.ndarray) -> PressureGrid:
    values = np.array(centers, dtype=float, copy=True)
    log_centers = np.log(values)
    inner_edges = 0.5 * (log_centers[:-1] + log_centers[1:])
    first_edge = log_centers[0] - (inner_edges[0] - log_centers[0])
    last_edge = log_centers[-1] + (log_centers[-1] - inner_edges[-1])
    edges = np.exp(np.concatenate(([first_edge], inner_edges, [last_edge])))
    return PressureGrid(edges=edges, centers=values, unit="bar", name="HAT-P-32b k-table pressure")


def _mpi_rank_size() -> tuple[int, int]:
    if not any(name in os.environ for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "SLURM_NTASKS")):
        return 0, 1
    try:
        from mpi4py import MPI
    except Exception:
        return 0, 1
    communicator = MPI.COMM_WORLD
    return int(communicator.Get_rank()), int(communicator.Get_size())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-npz", default=str(DEFAULT_OBSERVATION_NPZ))
    parser.add_argument("--pt-csv", default=str(DEFAULT_PT_CSV))
    parser.add_argument("--kta-dir", default=str(DEFAULT_KTA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--method", choices=("optimal_estimation", "ultranest"), default="optimal_estimation")
    parser.add_argument("--log-h2o-prior", nargs=2, type=float, default=(-8.0, -1.0))
    parser.add_argument("--temperature-offset-prior", nargs=2, type=float, default=(-400.0, 400.0))
    parser.add_argument("--radius-scale-prior", nargs=2, type=float, default=(0.8, 1.2))
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--live-points", type=int, default=60)
    parser.add_argument("--max-ncalls", type=int, default=600)
    parser.add_argument("--dlogz", type=float, default=1.0)
    parser.add_argument("--no-rayleigh", action="store_true")
    return parser


if __name__ == "__main__":
    main()
