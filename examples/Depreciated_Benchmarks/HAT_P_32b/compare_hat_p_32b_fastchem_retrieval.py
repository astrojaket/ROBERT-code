"""Compare ROBERT's FastChem/Madhu model with the saved HAT-P-32b result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from robert_exoplanets import build_retrieval_problem, run_configured_retrieval

if __package__:
    from .hat_p_32b_fastchem_config import (
        OBSERVATION_NPZ,
        OUTPUT_DIR,
        make_run_config,
        reference_map_parameters,
    )
else:
    from hat_p_32b_fastchem_config import (
        OBSERVATION_NPZ,
        OUTPUT_DIR,
        make_run_config,
        reference_map_parameters,
    )


def main() -> dict[str, object]:
    args = _parser().parse_args()
    rank, mpi_size = _mpi_rank_size()
    output_dir = Path(args.output_dir).expanduser()
    run_config = make_run_config(
        method=args.method,
        output_dir=output_dir,
        live_points=args.live_points,
        max_ncalls=args.max_ncalls,
        dlogz=args.dlogz,
        resume=args.resume,
        seed=args.seed,
        mpi_nprocs=args.mpi_nprocs or mpi_size,
        pressure_top_bar=args.pressure_top_bar,
        pressure_bottom_bar=args.pressure_bottom_bar,
        n_layers=args.layers,
    )
    problem = build_retrieval_problem(run_config)
    report: dict[str, object] = {}
    if rank == 0:
        reference_parameters = reference_map_parameters()
        robert_map = problem.model_spectrum(reference_parameters)
        observation = problem.observation
        reference_map = _reference_map_spectrum()
        residual_data = observation.flux - robert_map.values
        residual_reference = reference_map - robert_map.values
        report = {
            "parameters": reference_parameters,
            "n_points": observation.n_points,
            "robert_log_likelihood_at_reference_map": problem.likelihood.loglike(
                robert_map,
                observation,
                reference_parameters,
            ),
            "robert_chi_square_at_reference_map": float(
                np.sum(np.square(residual_data / observation.uncertainty))
            ),
            "robert_vs_reference_map_rms_ppm": float(
                np.sqrt(np.mean(np.square(residual_reference))) * 1.0e6
            ),
            "pressure_grid": {
                "top_bar": args.pressure_top_bar,
                "bottom_bar": args.pressure_bottom_bar,
                "n_layers": args.layers,
            },
            "cia": "NemesisPy v1.0.1 exocia_hitran12_200-3800K",
            "mpi_processes": mpi_size,
        }
    if args.run_retrieval:
        result = run_configured_retrieval(run_config)
        if rank == 0:
            report["retrieval"] = {
                "method": result.method,
                "converged": result.converged,
                "best_fit_parameters": dict(result.best_fit_parameters),
                "best_fit_log_likelihood": result.best_fit_log_likelihood,
            }
    if rank != 0:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reference_map_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    return report


def _reference_map_spectrum() -> np.ndarray:
    with np.load(OBSERVATION_NPZ, allow_pickle=False) as archive:
        return np.asarray(archive["MAP"], dtype=float)


def _mpi_rank_size() -> tuple[int, int]:
    if not any(
        name in os.environ
        for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "SLURM_NTASKS")
    ):
        return 0, 1
    try:
        from mpi4py import MPI
    except Exception:
        return 0, 1
    communicator = MPI.COMM_WORLD
    return int(communicator.Get_rank()), int(communicator.Get_size())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("optimal_estimation", "ultranest"), default="ultranest")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--run-retrieval", action="store_true")
    parser.add_argument("--live-points", type=int, default=400)
    parser.add_argument("--max-ncalls", type=int, default=100000)
    parser.add_argument("--dlogz", type=float, default=0.5)
    parser.add_argument(
        "--resume",
        choices=("resume", "resume-similar", "overwrite", "subfolder"),
        default="resume",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--mpi-nprocs", type=int)
    parser.add_argument("--pressure-top-bar", type=float, default=1.0e-6)
    parser.add_argument("--pressure-bottom-bar", type=float, default=100.0)
    parser.add_argument("--layers", type=int, default=100)
    return parser


if __name__ == "__main__":
    main()
