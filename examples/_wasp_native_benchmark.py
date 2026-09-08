"""Shared native-instrument builders for the WASP benchmark diagnostics."""

from __future__ import annotations

from dataclasses import replace
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

from robert_exoplanets import ObservationCollection

if __package__:
    from . import _wasp_emission_benchmark as workflow
else:
    import _wasp_emission_benchmark as workflow

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / f"{workflow.TARGET_SLUG}_cloud_free_native_modes_optimized_priors_v2"
)
RETAINED_MODES = ("f322w2", "f444w", "lrs")


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
