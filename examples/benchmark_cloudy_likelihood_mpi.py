"""Measure strong scaling of ROBERT's full cloudy likelihood under MPI.

Each MPI rank constructs the same production-shaped WASP-69b problem, warms
its compiled kernels, and evaluates independent deterministic points from the
prior. The reported aggregate throughput uses the slowest rank's wall time,
so it includes load imbalance without including problem construction or JIT
warmup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)

import numpy as np

from retrieve_wasp69b_mie_cloud import build_problem


def _communicator():
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1
    communicator = MPI.COMM_WORLD
    return communicator, communicator.Get_rank(), communicator.Get_size()


def _parameter_vectors(problem, repeats: int) -> tuple[np.ndarray, ...]:
    vectors = []
    for repeat in range(repeats):
        unit = np.full(problem.ndim, 0.5)
        selected = repeat % problem.ndim
        unit[selected] = 0.4 if repeat % 2 == 0 else 0.6
        vectors.append(problem.prior_transform(unit))
    return tuple(vectors)


def run(
    *,
    cloud_mode: str,
    material: str,
    particle_density_kg_m3: float,
    opacity_resolution: str,
    warmups: int,
    repeats: int,
) -> dict[str, object] | None:
    communicator, rank, world_size = _communicator()
    setup_started = perf_counter()
    problem = build_problem(
        cloud_mode=cloud_mode,
        material=material,
        particle_density_kg_m3=particle_density_kg_m3,
        opacity_resolution=opacity_resolution,
    )
    setup_seconds = perf_counter() - setup_started
    vectors = _parameter_vectors(problem, max(warmups, repeats))
    for index in range(warmups):
        value = problem.log_likelihood_from_vector(vectors[index])
        if not np.isfinite(value):
            raise RuntimeError("cloudy likelihood warmup returned a non-finite value")

    if communicator is not None:
        communicator.Barrier()
    started = perf_counter()
    likelihoods = [
        float(problem.log_likelihood_from_vector(vectors[index]))
        for index in range(repeats)
    ]
    elapsed_seconds = perf_counter() - started
    if not np.all(np.isfinite(likelihoods)):
        raise RuntimeError("cloudy likelihood benchmark returned a non-finite value")
    local = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "setup_seconds": setup_seconds,
        "timed_seconds": elapsed_seconds,
        "mean_call_seconds": elapsed_seconds / repeats,
        "likelihood_min": min(likelihoods),
        "likelihood_max": max(likelihoods),
    }
    gathered = [local] if communicator is None else communicator.gather(local, root=0)
    if rank != 0:
        return None
    slowest_seconds = max(float(item["timed_seconds"]) for item in gathered)
    total_calls = world_size * repeats
    model = next(iter(problem.forward_model.models.values()))
    shared_atmosphere = all(
        item.atmosphere_builder is model.atmosphere_builder
        for item in problem.forward_model.models.values()
    )
    return {
        "benchmark": "WASP-69b cloudy likelihood MPI strong scaling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "omp_threads": os.environ["OMP_NUM_THREADS"],
            "numba_threads": os.environ["NUMBA_NUM_THREADS"],
        },
        "configuration": {
            "cloud_mode": cloud_mode,
            "material": material,
            "opacity_resolution": opacity_resolution,
            "n_parameters": problem.ndim,
            "n_observations": problem.observations.n_points,
            "n_datasets": len(problem.forward_model.models),
            "shared_atmosphere_state": shared_atmosphere,
            "warmups_per_rank": warmups,
            "timed_calls_per_rank": repeats,
        },
        "mpi_world_size": world_size,
        "total_timed_likelihood_calls": total_calls,
        "slowest_rank_seconds": slowest_seconds,
        "aggregate_calls_per_second": total_calls / slowest_seconds,
        "mean_seconds_per_likelihood": slowest_seconds / repeats,
        "ranks": gathered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud-mode", choices=("catalog", "direct-nk"), default="catalog"
    )
    parser.add_argument("--material", default="MgSiO3")
    parser.add_argument("--particle-density-kg-m3", type=float, default=3200.0)
    parser.add_argument("--opacity-resolution", default="R1000")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats must be positive")
    result = run(
        cloud_mode=args.cloud_mode,
        material=args.material,
        particle_density_kg_m3=args.particle_density_kg_m3,
        opacity_resolution=args.opacity_resolution,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    if result is None:
        return
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
