"""Template retrieval using ROBERT's HAT-P-32b emission RT path."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache"))

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    ClearSkyEmissionForwardModel,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    UniformPrior,
    build_clear_sky_emission_model,
    load_emission_observation_npz,
    run_retrieval,
)

if __package__:
    from .hat_p_32b_config import (
        DEFAULT_KTA_DIR,
        DEFAULT_OBSERVATION_NPZ,
        DEFAULT_PT_CSV,
        make_hat_p_32b_model_config,
    )
else:
    from hat_p_32b_config import (
        DEFAULT_KTA_DIR,
        DEFAULT_OBSERVATION_NPZ,
        DEFAULT_PT_CSV,
        make_hat_p_32b_model_config,
    )

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_rt_retrieval"


def build_hat_p_32b_forward_model(
    observation,
    *,
    pt_csv: Path,
    kta_dir: Path,
    include_rayleigh: bool = True,
    exok_num: int = 300,
    opacity_species: tuple[str, ...] = ("H2O",),
) -> ClearSkyEmissionForwardModel:
    """Assemble HAT-P-32b inputs around ROBERT's public emission model."""

    config = make_hat_p_32b_model_config(
        pt_csv=pt_csv,
        kta_dir=kta_dir,
        opacity_species=opacity_species,
        include_rayleigh=include_rayleigh,
        exok_num=exok_num,
    )
    return build_clear_sky_emission_model(config, spectral_grid=observation.spectral_grid)


def main() -> dict[str, object]:
    args = _parser().parse_args()
    rank, size = _mpi_rank_size()
    if args.method != "ultranest" and rank != 0:
        return {}

    observation = load_emission_observation_npz(Path(args.observation_npz), instrument="JWST/NIRSpec G395H")
    forward_model = build_hat_p_32b_forward_model(
        observation,
        pt_csv=Path(args.pt_csv).expanduser(),
        kta_dir=Path(args.kta_dir).expanduser(),
        include_rayleigh=not args.no_rayleigh,
        exok_num=args.exok_num,
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
        metadata=dict(forward_model.manifest_metadata),
        opacity_identifiers=forward_model.opacity_identifiers,
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
        "parameters": dict(result.best_fit_parameters),
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
    parser.add_argument("--live-points", type=int, default=40)
    parser.add_argument("--max-ncalls", type=int, default=10000)
    parser.add_argument("--dlogz", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exok-num", type=int, default=300)
    parser.add_argument("--no-rayleigh", action="store_true")
    return parser


if __name__ == "__main__":
    main()
