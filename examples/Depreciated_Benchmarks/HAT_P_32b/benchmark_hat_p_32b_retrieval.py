"""Run a lightweight HAT-P-32b retrieval plumbing benchmark."""

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
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    load_emission_observation_npz,
    run_retrieval,
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
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_retrieval"


def main() -> dict[str, object]:
    """Run the benchmark and return a JSON-serializable report."""

    args = _parser().parse_args()
    rank, size = _mpi_rank_size()
    if args.method != "ultranest" and rank != 0:
        return {}

    observation_path = Path(args.observation_npz).expanduser()
    observation = load_emission_observation_npz(observation_path, instrument=args.instrument)
    problem = _polynomial_problem(
        observation,
        baseline_prior=tuple(args.baseline_prior),
        slope_prior=tuple(args.slope_prior),
        curvature_prior=tuple(args.curvature_prior),
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "optimal_estimation":
        result = run_retrieval(
            problem,
            method="optimal_estimation",
            output_dir=output_dir,
            max_iterations=args.max_iterations,
        )
    else:
        result = run_retrieval(
            problem,
            method="ultranest",
            output_dir=output_dir / "ultranest",
            seed=args.seed,
            min_num_live_points=args.live_points,
            max_ncalls=args.max_ncalls,
            dlogz=args.dlogz,
            mpi_nprocs=size,
            show_status=(rank == 0),
        )

    if rank != 0:
        return {}
    best_parameters = result.best_fit_parameters
    model = problem.model_spectrum(best_parameters)
    report = _report(args, observation_path, problem, result, model, mpi_size=size)
    json_path = output_dir / f"hat_p_32b_{args.method}_retrieval.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = output_dir / f"hat_p_32b_{args.method}_retrieval.png"
    _plot(plot_path, observation_path, observation, model, best_parameters)
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    return report


def _polynomial_problem(
    observation,
    *,
    baseline_prior: tuple[float, float],
    slope_prior: tuple[float, float],
    curvature_prior: tuple[float, float],
) -> RetrievalProblem:
    wavelength = np.array(observation.wavelength, dtype=float, copy=True)
    x = (wavelength - np.mean(wavelength)) / (np.max(wavelength) - np.min(wavelength))
    x2 = np.square(x) - np.mean(np.square(x))
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("baseline", UniformPrior(*baseline_prior), unit=observation.flux_unit),
            RetrievalParameter("slope", UniformPrior(*slope_prior), unit=f"{observation.flux_unit}/scaled_wavelength"),
            RetrievalParameter(
                "curvature",
                UniformPrior(*curvature_prior),
                unit=f"{observation.flux_unit}/scaled_wavelength^2",
            ),
        )
    )

    def forward_model(values: dict[str, float]) -> Spectrum:
        model = values["baseline"] + values["slope"] * x + values["curvature"] * x2
        return Spectrum.from_arrays(
            observation.wavelength,
            model,
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        )

    return RetrievalProblem(
        name="hat-p-32b-polynomial-retrieval-benchmark",
        observation=observation,
        parameters=parameters,
        forward_model=forward_model,
        metadata={"model": "polynomial_surrogate", "physics": "retrieval_plumbing_benchmark"},
    )


def _report(args, observation_path: Path, problem: RetrievalProblem, result, model: Spectrum, *, mpi_size: int) -> dict[str, object]:
    residual = problem.observation.flux - model.values
    chi2 = float(np.sum(np.square(residual / problem.observation.uncertainty)))
    report = {
        "method": args.method,
        "observation_npz": str(observation_path),
        "n_points": problem.observation.n_points,
        "parameters": dict(result.best_fit_parameters),
        "chi2": chi2,
        "reduced_chi2": chi2 / max(1, problem.observation.n_points - problem.ndim),
        "log_likelihood": _result_log_likelihood(result),
        "mpi_size": mpi_size,
    }
    if hasattr(result, "log_evidence"):
        report["log_evidence"] = result.log_evidence
        report["log_evidence_error"] = result.log_evidence_error
    if hasattr(result, "converged"):
        report["converged"] = bool(result.converged)
        report["message"] = result.message
    return report


def _result_log_likelihood(result) -> float:
    values = np.asarray(result.log_likelihood, dtype=float)
    if values.ndim == 0:
        return float(values)
    return float(np.max(values))


def _plot(path: Path, observation_path: Path, observation, model: Spectrum, parameters: dict[str, float]) -> None:
    with np.load(observation_path, allow_pickle=False) as archive:
        external_map = np.array(archive["MAP"], dtype=float, copy=True) if "MAP" in archive.files else None
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
    if external_map is not None:
        ax.plot(observation.wavelength, external_map * 1.0e6, color="#4c78a8", linewidth=1.4, label="Existing MAP")
    ax.plot(observation.wavelength, model.values * 1.0e6, color="#f58518", linewidth=1.8, label="ROBERT fit")
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Eclipse depth [ppm]")
    ax.set_title("HAT-P-32b Retrieval Smoke Benchmark")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    text = "\n".join(f"{key}={value:.4g}" for key, value in parameters.items())
    ax.text(0.02, 0.04, text, transform=ax.transAxes, fontsize=8, color="#333333")
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    parser.add_argument("--instrument", default="JWST/NIRSpec G395H")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--method", choices=("optimal_estimation", "ultranest"), default="optimal_estimation")
    parser.add_argument("--baseline-prior", nargs=2, type=float, default=(1.0e-3, 5.0e-3))
    parser.add_argument("--slope-prior", nargs=2, type=float, default=(-5.0e-3, 5.0e-3))
    parser.add_argument("--curvature-prior", nargs=2, type=float, default=(-5.0e-3, 5.0e-3))
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--live-points", type=int, default=80)
    parser.add_argument("--max-ncalls", type=int, default=2000)
    parser.add_argument("--dlogz", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main()
