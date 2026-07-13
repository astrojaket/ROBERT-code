"""Run the optimized clear-emission retrieval on configured native modes.

This fresh run excludes the derived NIRCam overlap-average product and uses
F322W2, F444W, and MIRI/LRS as independent likelihood terms. It deliberately
uses a new output directory because its priors differ from earlier runs and
therefore cannot reuse their UltraNest checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time

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
    ObservationCollection,
    run_ultranest,
)

import retrieve_wasp69b_nircam_cloud_free as workflow

OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / f"{workflow.TARGET_SLUG}_cloud_free_native_modes_optimized_priors_v2"
)
RETAINED_MODES = ("f322w2", "f444w", "lrs")
LIVE_POINTS = 50
DLOGZ = 2.5
CALLS_PER_ATTEMPT = 10_000
SEED = 20260712


def native_mode_observations() -> ObservationCollection:
    """Return only independent published instrument modes."""

    published = workflow.TARGET.load_observations(miri_offset_parameter=None)
    observations = ObservationCollection(
        datasets=tuple(
            dataset for dataset in published.datasets if dataset.name in RETAINED_MODES
        ),
        name=f"{workflow.PLANET.name} native instrument modes",
        metadata={
            **dict(published.metadata),
            "selection": "F322W2, F444W, and LRS; overlap average excluded",
            "overlap_handling": "none",
        },
    )
    if observations.names != RETAINED_MODES:
        raise RuntimeError(
            f"expected native modes {RETAINED_MODES}, found {observations.names}"
        )
    return observations


def build_native_mode_problem(
    observations: ObservationCollection,
    *,
    opacity_resolution: str = workflow.DEFAULT_OPACITY_RESOLUTION,
):
    """Build the optimized retrieval problem with v2 prior provenance."""

    problem = workflow.build_problem(
        observations,
        opacity_resolution=opacity_resolution,
    )
    return replace(
        problem,
        name=f"{workflow.TARGET_SLUG}-cloud-free-native-modes-optimized-priors-v2",
        metadata={
            **dict(problem.metadata),
            "difference": ("native F322W2/F444W/LRS modes with PG14 analytic TP"),
            "dataset_selection": "F322W2,F444W,LRS",
            "overlap_average": "excluded",
            "dataset_manipulation": "none",
            "opacity_resolution": opacity_resolution,
            "metallicity_prior": "uniform_log10_Z_over_Zsun_-1_to_2",
            "carbon_to_oxygen_prior": "uniform_linear_0_to_1",
            "performance_path": (
                "cached_log_k_prepared_spectral_indices_streaming_random_overlap"
            ),
        },
    )


def _configuration(
    problem,
    *,
    max_ncalls: int,
    dlogz: float,
    fast_stop: bool,
    mpi_processes: int,
    kta_path: Path,
    opacity_resolution: str,
    smoke: dict[str, float],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "problem": problem.name,
        "datasets": list(problem.observations.names),
        "n_observations": problem.observations.n_points,
        "opacity": {
            "kta_path": str(kta_path.expanduser().resolve()),
            "resolution": opacity_resolution,
            "cache_directory": str(
                workflow.opacity_cache_directory(opacity_resolution).resolve()
            ),
            "species": list(workflow.SPECIES),
        },
        "opacity_identifiers": dict(problem.opacity_identifiers),
        "parameters": [
            {
                "name": parameter.name,
                "prior": type(parameter.prior).__name__,
                "lower": parameter.prior.lower,
                "upper": parameter.prior.upper,
                "unit": parameter.unit,
            }
            for parameter in problem.parameters.parameters
        ],
        "sampler": {
            "name": "UltraNest",
            "min_num_live_points": LIVE_POINTS,
            "max_ncalls": max_ncalls,
            "dlogz": dlogz,
            "fast_stop": fast_stop,
            "frac_remain": 1.0 if fast_stop else 0.01,
            "min_ess": 50 if fast_stop else 400,
            "dKL": 100.0 if fast_stop else 0.5,
            "Lepsilon": 100.0 if fast_stop else 0.001,
            "max_num_improvement_loops": 0 if fast_stop else -1,
            "resume": "resume",
            "mpi_processes": mpi_processes,
            "seed": SEED,
        },
        "smoke_evaluation": smoke,
        "metadata": dict(problem.metadata),
    }


def smoke_evaluation(problem) -> dict[str, float]:
    """Evaluate the prior midpoint twice before allocating sampler time."""

    theta = problem.prior_transform(np.full(problem.ndim, 0.5))
    started = time.perf_counter()
    log_likelihood = problem.log_likelihood_from_vector(theta)
    first_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    repeated_log_likelihood = problem.log_likelihood_from_vector(theta)
    warmed_elapsed = time.perf_counter() - started
    if not np.isfinite(log_likelihood) or log_likelihood <= problem.invalid_loglike:
        raise RuntimeError(
            "prior-midpoint smoke evaluation returned an invalid likelihood"
        )
    if repeated_log_likelihood != log_likelihood:
        raise RuntimeError("repeated prior-midpoint likelihood was not deterministic")
    return {
        "first_elapsed_seconds": first_elapsed,
        "warmed_elapsed_seconds": warmed_elapsed,
        "log_likelihood": log_likelihood,
    }


def _is_primary_process() -> bool:
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def _next_cumulative_call_limit(output: Path) -> int:
    """Advance the cumulative UltraNest ceiling by one resumable chunk."""

    status_path = output / "ultranest" / "sampler_status.json"
    if not status_path.exists():
        return CALLS_PER_ATTEMPT
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return max(int(status.get("ncall", 0)), 0) + CALLS_PER_ATTEMPT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configured_kta_path = os.environ.get("ROBERT_KTABLE_PATH")
    parser.add_argument(
        "--kta-path",
        type=Path,
        default=configured_kta_path,
        required=configured_kta_path is None,
        help="KTA root containing R1000/R15000, or the selected resolution directory",
    )
    parser.add_argument(
        "--opacity-resolution",
        choices=workflow.OPACITY_RESOLUTIONS,
        default=os.environ.get(
            "ROBERT_OPACITY_RESOLUTION", workflow.DEFAULT_OPACITY_RESOLUTION
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--max-ncalls",
        type=int,
        help="explicit cumulative UltraNest likelihood-call ceiling",
    )
    parser.add_argument(
        "--dlogz",
        type=float,
        default=DLOGZ,
        help=f"UltraNest remaining-evidence tolerance (default: {DLOGZ})",
    )
    parser.add_argument(
        "--fast-stop",
        action="store_true",
        help="exploratory early stop with relaxed remaining-integral, ESS, and KL targets",
    )
    parser.add_argument(
        "--mpi-processes",
        type=int,
        default=int(os.environ.get("SLURM_NTASKS", "1")),
        help="MPI process count recorded in run metadata",
    )
    args = parser.parse_args()
    if not np.isfinite(args.dlogz) or args.dlogz <= 0.0:
        parser.error("--dlogz must be finite and positive")
    if args.mpi_processes < 1:
        parser.error("--mpi-processes must be positive")

    observations = native_mode_observations()
    if args.prepare_only:
        workflow.prepare_opacity_cache(
            observations,
            kta_path=args.kta_path,
            resolution=args.opacity_resolution,
        )
        return

    problem = build_native_mode_problem(
        observations,
        opacity_resolution=args.opacity_resolution,
    )
    smoke = smoke_evaluation(problem)
    max_ncalls = (
        _next_cumulative_call_limit(args.output)
        if args.max_ncalls is None
        else args.max_ncalls
    )
    current_calls = max_ncalls - CALLS_PER_ATTEMPT
    status_path = args.output / "ultranest" / "sampler_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        current_calls = max(int(status.get("ncall", 0)), 0)
    if max_ncalls <= current_calls:
        parser.error(
            "--max-ncalls must be greater than the current cumulative call count "
            f"({current_calls})"
        )
    is_primary = _is_primary_process()
    if is_primary:
        configuration = _configuration(
            problem,
            max_ncalls=max_ncalls,
            dlogz=args.dlogz,
            fast_stop=args.fast_stop,
            mpi_processes=args.mpi_processes,
            kta_path=args.kta_path,
            opacity_resolution=args.opacity_resolution,
            smoke=smoke,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "run_configuration.json").write_text(
            json.dumps(configuration, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(configuration, indent=2))
    if args.smoke_only:
        return

    fast_stop_options = (
        {
            "frac_remain": 1.0,
            "min_ess": 50,
            "dKL": 100.0,
            "Lepsilon": 100.0,
            "max_num_improvement_loops": 0,
        }
        if args.fast_stop
        else {}
    )
    result = run_ultranest(
        problem,
        output_dir=args.output / "ultranest",
        min_num_live_points=LIVE_POINTS,
        max_ncalls=max_ncalls,
        dlogz=args.dlogz,
        resume="resume",
        show_status=True,
        mpi_nprocs=args.mpi_processes,
        seed=SEED,
        **fast_stop_options,
    )
    if is_primary and result.converged:
        workflow._write_products(
            problem,
            result,
            args.output,
            live_points=LIVE_POINTS,
            max_ncalls=max_ncalls,
            mpi_processes=args.mpi_processes,
        )
        if args.fast_stop:
            summary_path = args.output / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(
                {
                    "run_classification": "exploratory_forced_early_stop",
                    "scientific_convergence": False,
                    "sampling_warning": (
                        "UltraNest met deliberately relaxed stopping criteria; "
                        "posterior ESS must be checked before scientific interpretation."
                    ),
                }
            )
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    elif is_primary:
        print(
            "UltraNest reached this call-limit chunk without convergence; "
            "rerun the same command to resume with a larger cumulative limit."
        )


if __name__ == "__main__":
    main()
