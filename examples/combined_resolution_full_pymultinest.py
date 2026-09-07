"""Run full combined-resolution PyMultiNest and native-stride checks.

The data use one shared four-parameter atmosphere. The high-resolution branch
uses the full native line-by-line grid for the reference retrieval. An optional
stride-2 fit uses the same noisy observations and reports the retrieval bias
against the stride-1 fit. A complete paired nested assessment records an
explicit production decision: it approves stride 2 only when every candidate
gate passes; otherwise it rejects stride 2 and retains stride 1. Incomplete or
failed sampler runs fail the workflow. The nested run is opt-in because it is a
scientific run, not a unit-test operation.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from typing import Mapping, Sequence

# Keep lightweight checks at the requested value, and cap accidental values at
# the project limit before NumPy, SciPy, HDF5, or PyMultiNest are imported.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
for _thread_variable in _THREAD_VARIABLES:
    _requested_threads = os.environ.get(_thread_variable)
    if _requested_threads is None:
        _requested_threads = "3"
    else:
        try:
            _requested_threads = str(min(3, max(1, int(_requested_threads))))
        except ValueError:
            _requested_threads = "3"
    os.environ[_thread_variable] = _requested_threads
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import combined_resolution_injection_recovery as combined  # noqa: E402
from robert_exoplanets import (  # noqa: E402
    GaussianLikelihood,
    MultiDatasetRetrievalProblem,
    run_retrieval,
)
from robert_exoplanets.likelihoods import (  # noqa: E402
    MixedMultiDatasetLikelihood,
    PolynomialContinuumLikelihood,
)


REPORT_PATH = ROOT / "docs" / "data" / "combined_resolution_full_pymultinest_20260831.json"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "combined_resolution_full_pymultinest"
MAX_MEMORY_BYTES = 2 * 1024**3
DEFAULT_MAX_MEMORY_BYTES = 1900 * 1024**2
DEFAULT_SEED = 24680
DEFAULT_PYMULTINEST_LIVE_POINTS = 64
DEFAULT_PYMULTINEST_MAX_ITER = 0
DEFAULT_PYMULTINEST_EVIDENCE_TOLERANCE = 0.5
DEFAULT_PYMULTINEST_SAMPLING_EFFICIENCY = 0.8
POSTERIOR_INTERVAL_QUANTILES = (0.05, 0.95)
EVIDENCE_SIGMA_MULTIPLIER = 2.0
REDUCED_CHI_SQUARE_BOUNDS = (0.5, 1.5)
ABSOLUTE_PARAMETER_TOLERANCES = {
    "temperature_K": 80.0,
    "log10_h2o_vmr": 0.25,
    "log10_co_vmr": 0.25,
    "radial_velocity_km_s": 1.0,
}
STRIDE_RMS_LIMIT = 0.002
STRIDE_MAX_LIMIT = 0.01
STRIDE_POSTERIOR_PARAMETER_SHIFT_TOLERANCES = ABSOLUTE_PARAMETER_TOLERANCES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=DEFAULT_MAX_MEMORY_BYTES / 1024**3,
        help="Strict process RSS guard; values must be below 2 GiB.",
    )
    parser.add_argument(
        "--run-pymultinest",
        action="store_true",
        help="Run two full four-parameter PyMultiNest retrievals.",
    )
    parser.add_argument(
        "--run-native-stride-comparison",
        action="store_true",
        help="Fit stride 2 against the stride-1 noisy observations.",
    )
    parser.add_argument(
        "--pymultinest-live-points",
        type=int,
        default=DEFAULT_PYMULTINEST_LIVE_POINTS,
    )
    parser.add_argument(
        "--pymultinest-max-iter",
        type=int,
        default=DEFAULT_PYMULTINEST_MAX_ITER,
        help="Use 0 for natural MultiNest convergence.",
    )
    parser.add_argument(
        "--pymultinest-evidence-tolerance",
        type=float,
        default=DEFAULT_PYMULTINEST_EVIDENCE_TOLERANCE,
    )
    parser.add_argument(
        "--pymultinest-sampling-efficiency",
        type=float,
        default=DEFAULT_PYMULTINEST_SAMPLING_EFFICIENCY,
    )
    return parser.parse_args()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _thread_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for name in _THREAD_VARIABLES:
        try:
            values[name] = int(os.environ.get(name, "3"))
        except ValueError:
            values[name] = 0
    return values


def _check_tables() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for species, path in {
        "H2O": combined.H2O_TABLE,
        "CO": combined.CO_TABLE,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing external LBL table: {path}")
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        if checksum != combined.EXPECTED_SHA256[species]:
            raise ValueError(f"unexpected {species} LBL checksum")
        records[species] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": checksum,
        }
    return records


def _likelihood(observations: object) -> MixedMultiDatasetLikelihood:
    datasets = getattr(observations, "datasets", ())
    observations_by_name = {dataset.name: dataset.observation for dataset in datasets}
    return MixedMultiDatasetLikelihood(
        {
            "low_resolution": GaussianLikelihood(
                offset_parameter=None,
                jitter_parameter=None,
            ),
            "high_resolution": PolynomialContinuumLikelihood(degree=1).prepare(
                observations_by_name["high_resolution"]
            ),
        }
    )


def _problem(
    model: object,
    observations: object,
    likelihood: MixedMultiDatasetLikelihood,
    *,
    native_sampling_stride: int,
) -> MultiDatasetRetrievalProblem:
    return MultiDatasetRetrievalProblem(
        name=(
            "combined-low-high-resolution-emission-"
            f"stride-{native_sampling_stride}"
        ),
        observations=observations,
        parameters=combined._parameters(),
        forward_model=model,
        likelihood=likelihood,
        metadata={
            "clear_atmosphere": "true",
            "low_resolution_opacity": "bundled_R100_correlated_k",
            "high_resolution_opacity": "petitRADTRANS_R1e6_line_by_line",
            "high_resolution_response": "Doppler+rotation+Gaussian_LSF+pixels",
            "native_sampling_stride": str(native_sampling_stride),
        },
        opacity_identifiers={
            "H2O_LBL_sha256": combined.EXPECTED_SHA256["H2O"],
            "CO_LBL_sha256": combined.EXPECTED_SHA256["CO"],
        },
    )


def _observations(
    truth_spectra: Mapping[str, object],
    *,
    seed: int,
    noise_scale: float,
) -> object:
    low = truth_spectra["low_resolution"]
    high = truth_spectra["high_resolution"]
    return combined.inject_spectrum_collection(
        truth_spectra,
        {
            "low_resolution": np.full(low.values.shape, 5.0e-6),
            "high_resolution": np.full(high.values.shape, 2.0e-6),
        },
        seed=seed,
        noise_scale=noise_scale,
        instruments={
            "low_resolution": "synthetic broadband R~30",
            "high_resolution": "synthetic K-band R=100000",
        },
        metadata={"validation_case": "combined_resolution_full_pymultinest"},
    )


def _run_oe(
    problem: MultiDatasetRetrievalProblem,
    *,
    output_dir: Path,
) -> object:
    return run_retrieval(
        problem,
        method="optimal_estimation",
        output_dir=output_dir,
        initial_state=[1450.0, -3.6, -3.4, 0.0],
        max_iterations=12,
        convergence_tolerance=2.0e-4,
        finite_difference_fraction=2.0e-3,
        damping=1.0e-8,
    )


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float] | None,
    quantiles: Sequence[float],
) -> np.ndarray:
    """Return central weighted quantiles for one posterior parameter."""

    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size == 0:
        raise ValueError("posterior values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(value_array)):
        raise ValueError("posterior values must be finite")
    if weights is None:
        weight_array = np.ones(value_array.size, dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != value_array.shape:
            raise ValueError("posterior weights must match posterior values")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("posterior weights must be finite and non-negative")
    total = float(np.sum(weight_array))
    if total <= 0.0:
        raise ValueError("posterior weights must have positive total weight")
    requested = np.asarray(quantiles, dtype=float)
    if requested.ndim != 1 or np.any(~np.isfinite(requested)) or np.any(
        (requested < 0.0) | (requested > 1.0)
    ):
        raise ValueError("posterior quantiles must be between zero and one")
    order = np.argsort(value_array, kind="mergesort")
    sorted_values = value_array[order]
    sorted_weights = weight_array[order]
    cumulative = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / total
    return np.interp(requested, cumulative, sorted_values, left=sorted_values[0], right=sorted_values[-1])


def _posterior_intervals(
    samples: np.ndarray,
    weights: np.ndarray | None,
    parameter_names: Sequence[str],
) -> dict[str, tuple[float, float]]:
    if samples.ndim != 2 or samples.shape[1] != len(parameter_names):
        raise ValueError("posterior samples must match the parameter names")
    return {
        name: tuple(
            float(value)
            for value in _weighted_quantile(
                samples[:, index], weights, POSTERIOR_INTERVAL_QUANTILES
            )
        )
        for index, name in enumerate(parameter_names)
    }


def _posterior_medians(
    samples: np.ndarray,
    weights: np.ndarray | None,
    parameter_names: Sequence[str],
) -> dict[str, float]:
    """Return deterministic weighted posterior centres for each parameter."""

    if samples.ndim != 2 or samples.shape[1] != len(parameter_names):
        raise ValueError("posterior samples must match the parameter names")
    return {
        name: float(
            _weighted_quantile(samples[:, index], weights, (0.5,))[0]
        )
        for index, name in enumerate(parameter_names)
    }


def _fit_statistics(
    problem: MultiDatasetRetrievalProblem,
    observations: object,
    spectra: Mapping[str, object],
    parameters: Mapping[str, float],
) -> dict[str, object]:
    """Compute fit statistics from the exact arrays used by the likelihood.

    A prepared high-resolution likelihood can remove a fitted continuum before
    it evaluates the residual.  A fresh Gaussian likelihood would therefore
    give a different statistic.  The effective point count is the length of
    each returned array.  When available, the likelihood supplies exact
    chi-square values and retained independent residual ranks.  The degrees
    of freedom are the sum of retained ranks minus the atmospheric parameter
    count.  The projection rank remains in the report as a compatibility
    diagnostic (raw effective points minus retained rank).
    """

    effective_inputs = problem.likelihood.effective_inputs_by_dataset(
        spectra,
        observations,
        parameters,
    )
    total = 0.0
    effective_point_count = 0
    raw_point_count = 0
    effective_point_count_by_dataset: dict[str, int] = {}
    removed_basis_rank_by_dataset: dict[str, int] = {}
    retained_residual_rank_by_dataset: dict[str, int] = {}
    chi_square_by_dataset: dict[str, float] = {}
    likelihoods = getattr(problem.likelihood, "likelihoods", {})
    exact_chi_square = getattr(problem.likelihood, "chi_square_by_dataset", None)
    exact_ranks = getattr(
        problem.likelihood,
        "effective_residual_rank_by_dataset",
        None,
    )
    exact_chi_values = (
        {
            str(name): float(value)
            for name, value in exact_chi_square(
                spectra,
                observations,
                parameters,
            ).items()
        }
        if callable(exact_chi_square)
        else None
    )
    exact_rank_values = (
        {
            str(name): int(value)
            for name, value in exact_ranks(
                spectra,
                observations,
                parameters,
            ).items()
        }
        if callable(exact_ranks)
        else None
    )
    for dataset in observations.datasets:
        raw_point_count += int(dataset.observation.n_points)
        try:
            model, data, uncertainty = effective_inputs[dataset.name]
        except KeyError as error:
            raise ValueError(
                f"effective inputs are missing dataset {dataset.name!r}"
            ) from error
        model = np.asarray(model, dtype=float)
        data = np.asarray(data, dtype=float)
        uncertainty = np.asarray(uncertainty, dtype=float)
        if model.ndim != 1 or data.ndim != 1 or uncertainty.ndim != 1:
            raise ValueError("effective likelihood inputs must be one-dimensional")
        if not (model.shape == data.shape == uncertainty.shape):
            raise ValueError("effective likelihood inputs must have matching shapes")
        if (
            not np.all(np.isfinite(model))
            or not np.all(np.isfinite(data))
            or not np.all(np.isfinite(uncertainty))
            or np.any(uncertainty <= 0.0)
        ):
            raise ValueError("effective likelihood inputs must be finite and positive")
        residual = (data - model) / uncertainty
        fallback_chi_square = float(np.sum(np.square(residual)))
        chi_square = (
            fallback_chi_square
            if exact_chi_values is None
            else exact_chi_values[dataset.name]
        )
        if not np.isfinite(chi_square) or chi_square < 0.0:
            raise ValueError(
                f"chi-square for dataset {dataset.name!r} must be finite and non-negative"
            )
        chi_square_by_dataset[dataset.name] = chi_square
        total += chi_square
        point_count = int(data.size)
        effective_point_count += point_count
        effective_point_count_by_dataset[dataset.name] = point_count

        if exact_rank_values is not None:
            rank = exact_rank_values[dataset.name]
        else:
            selected_likelihood = (
                likelihoods.get(dataset.name)
                if isinstance(likelihoods, Mapping)
                else None
            )
            projection = getattr(selected_likelihood, "projection", None)
            rank_value = getattr(projection, "rank", 0)
            try:
                removed_rank = int(rank_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"projection rank for dataset {dataset.name!r} must be an integer"
                ) from error
            if removed_rank < 0 or removed_rank > point_count:
                raise ValueError(
                    f"projection rank for dataset {dataset.name!r} is outside its "
                    "effective point count"
                )
            # Compatibility with the earlier report format.  The exact hook
            # below is preferred because it returns retained dimensions.
            rank = point_count - removed_rank
        if rank < 0 or rank > point_count:
            raise ValueError(
                f"effective residual rank for dataset {dataset.name!r} is outside its "
                "effective point count"
            )
        retained_residual_rank_by_dataset[dataset.name] = rank
        removed_basis_rank_by_dataset[dataset.name] = point_count - rank

    retained_residual_rank = sum(retained_residual_rank_by_dataset.values())
    removed_basis_rank = sum(removed_basis_rank_by_dataset.values())
    degrees_of_freedom = retained_residual_rank - problem.ndim
    if degrees_of_freedom <= 0:
        raise ValueError("fit statistics require positive residual degrees of freedom")
    reduced = total / degrees_of_freedom
    return {
        "chi_square": total,
        "reduced_chi_square": reduced,
        "n_active_points": effective_point_count,
        "effective_point_count": effective_point_count,
        "effective_point_count_by_dataset": effective_point_count_by_dataset,
        "chi_square_by_dataset": chi_square_by_dataset,
        "retained_residual_rank": retained_residual_rank,
        "retained_residual_rank_by_dataset": retained_residual_rank_by_dataset,
        "raw_point_count": raw_point_count,
        "removed_basis_rank": removed_basis_rank,
        "removed_basis_rank_by_dataset": removed_basis_rank_by_dataset,
        "degrees_of_freedom": degrees_of_freedom,
        "chi_square_passed": REDUCED_CHI_SQUARE_BOUNDS[0]
        <= reduced
        <= REDUCED_CHI_SQUARE_BOUNDS[1],
    }


def _evidence_agreement(
    evidences: Sequence[float],
    errors: Sequence[float | None],
    *,
    absolute_tolerance: float,
) -> dict[str, float | bool | None]:
    if len(evidences) != 2 or len(errors) != 2:
        return {
            "passed": False,
            "difference": None,
            "combined_reported_error": None,
            "allowed_difference": absolute_tolerance,
        }
    evidence_errors = np.asarray(
        [np.nan if error is None else float(error) for error in errors], dtype=float
    )
    if not np.all(np.isfinite(evidence_errors)):
        return {
            "passed": False,
            "difference": None,
            "combined_reported_error": None,
            "allowed_difference": absolute_tolerance,
        }
    difference = abs(float(evidences[0]) - float(evidences[1]))
    combined_error = float(np.hypot(evidence_errors[0], evidence_errors[1]))
    allowed = max(absolute_tolerance, EVIDENCE_SIGMA_MULTIPLIER * combined_error)
    return {
        "passed": difference <= allowed,
        "difference": difference,
        "combined_reported_error": combined_error,
        "allowed_difference": allowed,
    }


def _run_full_pymultinest(
    problem: MultiDatasetRetrievalProblem,
    observations: object,
    *,
    output_dir: Path,
    seed: int,
    n_live_points: int,
    max_iter: int,
    evidence_tolerance: float,
    sampling_efficiency: float,
) -> dict[str, object]:
    """Run two full four-parameter MultiNest retrievals and apply gates."""

    started = perf_counter()
    records: list[dict[str, object]] = []
    for run_index, run_seed in enumerate((int(seed), int(seed) + 1), start=1):
        run_started = perf_counter()
        try:
            result = run_retrieval(
                problem,
                method="pymultinest",
                output_dir=output_dir / f"seed_{run_seed}",
                seed=run_seed,
                n_live_points=n_live_points,
                max_iter=max_iter,
                evidence_tolerance=evidence_tolerance,
                sampling_efficiency=sampling_efficiency,
                resume=False,
                verbose=False,
                mpi_nprocs=1,
            )
            inference = result.inference_result
            samples = np.asarray(inference.samples, dtype=float)
            weights = (
                None
                if inference.weights is None
                else np.asarray(inference.weights, dtype=float)
            )
            intervals = _posterior_intervals(
                samples,
                weights,
                problem.parameter_names,
            )
            medians = _posterior_medians(
                samples,
                weights,
                problem.parameter_names,
            )
            truth_inside = all(
                lower <= float(combined.TRUTH[name]) <= upper
                for name, (lower, upper) in intervals.items()
            )
            best_fit = dict(result.best_fit_parameters)
            best_fit_errors = {
                name: abs(float(best_fit[name]) - float(combined.TRUTH[name]))
                for name in problem.parameter_names
            }
            fit_statistics = _fit_statistics(
                problem,
                observations,
                problem.model_spectra(best_fit),
                best_fit,
            )
            metadata = result.inference_result.metadata
            records.append(
                {
                    "run_index": run_index,
                    "seed": run_seed,
                    "status": "completed",
                    "converged": bool(result.converged),
                    "message": result.message,
                    "best_fit": best_fit,
                    "best_fit_absolute_errors": best_fit_errors,
                    "best_fit_truth_recovery": all(
                        best_fit_errors[name] <= ABSOLUTE_PARAMETER_TOLERANCES[name]
                        for name in problem.parameter_names
                    ),
                    "posterior_interval_quantiles": list(
                        POSTERIOR_INTERVAL_QUANTILES
                    ),
                    "posterior_medians": medians,
                    "posterior_intervals": {
                        name: list(interval)
                        for name, interval in intervals.items()
                    },
                    "truth_inside_posterior_intervals": truth_inside,
                    "fit_statistics": fit_statistics,
                    "log_evidence": result.log_evidence,
                    "log_evidence_error": result.log_evidence_error,
                    "likelihood_evaluations": int(
                        metadata.get("likelihood_evaluations", "0")
                    ),
                    "likelihood_callback_evaluations": int(
                        metadata.get("likelihood_callback_evaluations", "0")
                    ),
                    "peak_rss_bytes": _peak_rss_bytes(),
                    "elapsed_seconds": perf_counter() - run_started,
                }
            )
        except Exception as error:  # keep a complete, honest gate record
            records.append(
                {
                    "run_index": run_index,
                    "seed": run_seed,
                    "status": "failed",
                    "converged": False,
                    "message": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": perf_counter() - run_started,
                }
            )

    completed = [record for record in records if record["status"] == "completed"]
    converged = len(completed) == 2 and all(
        bool(record["converged"]) for record in completed
    )
    truth_inside = len(completed) == 2 and all(
        bool(record["truth_inside_posterior_intervals"]) for record in completed
    )
    best_fit_recovery = len(completed) == 2 and all(
        bool(record["best_fit_truth_recovery"]) for record in completed
    )
    chi_square_passed = len(completed) == 2 and all(
        bool(record["fit_statistics"]["chi_square_passed"])
        for record in completed
    )
    evidences = [
        float(record["log_evidence"])
        for record in completed
        if record.get("log_evidence") is not None
        and np.isfinite(float(record["log_evidence"]))
    ]
    errors = [
        None if record.get("log_evidence_error") is None else float(record["log_evidence_error"])
        for record in completed
    ]
    evidence = _evidence_agreement(
        evidences,
        errors,
        absolute_tolerance=evidence_tolerance,
    )
    gate = {
        "required_completed_runs": 2,
        "completed_runs": len(completed),
        "max_iter_zero": max_iter == 0,
        "natural_convergence": converged,
        "truth_inside_posterior_intervals": truth_inside,
        "best_fit_truth_recovery": best_fit_recovery,
        "chi_square": chi_square_passed,
        "evidence_agreement": evidence["passed"],
        "evidence": evidence,
    }
    gate["passed"] = bool(all(gate[key] for key in (
        "completed_runs",
        "max_iter_zero",
        "natural_convergence",
        "truth_inside_posterior_intervals",
        "best_fit_truth_recovery",
        "chi_square",
        "evidence_agreement",
    ))) and gate["completed_runs"] == 2
    return {
        "status": "pass" if gate["passed"] else "fail",
        "scope": "full four-parameter atmospheric posterior",
        "parameter_names": list(problem.parameter_names),
        "full_four_parameter_nested_sampling": "run",
        "posterior_interval_statement": "central 90 percent weighted posterior interval",
        "settings": {
            "n_live_points": n_live_points,
            "max_iter": max_iter,
            "evidence_tolerance": evidence_tolerance,
            "sampling_efficiency": sampling_efficiency,
            "resume": False,
            "verbose": False,
            "mpi_nprocs": 1,
            "seed_pair": [int(seed), int(seed) + 1],
        },
        "gate": gate,
        "runs": records,
        "limitation": (
            "This full posterior test uses the synthetic shared-atmosphere case. "
            "It does not validate a real target or replace independent pRT oracle checks."
        ),
        "elapsed_seconds": perf_counter() - started,
    }


def _stride_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("stride comparison spectra must share the observation grid")
    denominator = np.maximum(np.abs(reference), np.finfo(float).tiny)
    relative = candidate / denominator - 1.0
    return {
        "rms_relative_error": float(np.sqrt(np.mean(np.square(relative)))),
        "max_absolute_relative_error": float(np.max(np.abs(relative))),
        "median_signed_relative_error": float(np.median(relative)),
    }


def _compare_nested_stride_posteriors(
    stride_one_nested: Mapping[str, object],
    stride_two_nested: Mapping[str, object],
) -> dict[str, object]:
    """Compare paired stride-1 and stride-2 nested posterior summaries."""

    comparison_settings = stride_two_nested.get("settings", {})
    evidence_tolerance = DEFAULT_PYMULTINEST_EVIDENCE_TOLERANCE
    if isinstance(comparison_settings, Mapping):
        evidence_tolerance = float(
            comparison_settings.get(
                "evidence_tolerance",
                DEFAULT_PYMULTINEST_EVIDENCE_TOLERANCE,
            )
        )
    first_runs = {
        int(record["seed"]): record
        for record in stride_one_nested.get("runs", [])
        if isinstance(record, Mapping) and "seed" in record
    }
    second_runs = {
        int(record["seed"]): record
        for record in stride_two_nested.get("runs", [])
        if isinstance(record, Mapping) and "seed" in record
    }
    paired: list[dict[str, object]] = []
    posterior_passed = True
    evidence_passed = True
    for seed in sorted(set(first_runs) & set(second_runs)):
        first = first_runs[seed]
        second = second_runs[seed]
        first_medians = first.get("posterior_medians", {})
        second_medians = second.get("posterior_medians", {})
        first_intervals = first.get("posterior_intervals", {})
        second_intervals = second.get("posterior_intervals", {})
        median_shift: dict[str, float] = {}
        interval_endpoint_shift: dict[str, list[float]] = {}
        run_posterior_passed = True
        for name, tolerance in STRIDE_POSTERIOR_PARAMETER_SHIFT_TOLERANCES.items():
            try:
                shift = float(second_medians[name]) - float(first_medians[name])
                lower_shift = float(second_intervals[name][0]) - float(first_intervals[name][0])
                upper_shift = float(second_intervals[name][1]) - float(first_intervals[name][1])
            except (KeyError, TypeError, ValueError, IndexError):
                run_posterior_passed = False
                continue
            median_shift[name] = shift
            interval_endpoint_shift[name] = [lower_shift, upper_shift]
            run_posterior_passed = run_posterior_passed and abs(shift) <= tolerance
            run_posterior_passed = run_posterior_passed and abs(lower_shift) <= tolerance
            run_posterior_passed = run_posterior_passed and abs(upper_shift) <= tolerance

        log_evidence_difference: float | None = None
        absolute_log_evidence_difference: float | None = None
        evidence_allowed_difference: float | None = None
        try:
            first_evidence = float(first["log_evidence"])
            second_evidence = float(second["log_evidence"])
            first_error = float(first["log_evidence_error"])
            second_error = float(second["log_evidence_error"])
            log_evidence_difference = second_evidence - first_evidence
            absolute_log_evidence_difference = abs(log_evidence_difference)
            evidence_allowed_difference = max(
                evidence_tolerance,
                EVIDENCE_SIGMA_MULTIPLIER * float(np.hypot(first_error, second_error)),
            )
            run_evidence_passed = (
                absolute_log_evidence_difference <= evidence_allowed_difference
            )
        except (KeyError, TypeError, ValueError):
            run_evidence_passed = False
        posterior_passed = posterior_passed and run_posterior_passed
        evidence_passed = evidence_passed and run_evidence_passed
        paired.append(
            {
                "seed": seed,
                "stride_1": {
                    "posterior_medians": first_medians,
                    "posterior_intervals": first_intervals,
                    "log_evidence": first.get("log_evidence"),
                    "log_evidence_error": first.get("log_evidence_error"),
                    "likelihood_evaluations": first.get("likelihood_evaluations"),
                    "elapsed_seconds": first.get("elapsed_seconds"),
                    "peak_rss_bytes": first.get("peak_rss_bytes"),
                },
                "stride_2": {
                    "posterior_medians": second_medians,
                    "posterior_intervals": second_intervals,
                    "log_evidence": second.get("log_evidence"),
                    "log_evidence_error": second.get("log_evidence_error"),
                    "likelihood_evaluations": second.get("likelihood_evaluations"),
                    "elapsed_seconds": second.get("elapsed_seconds"),
                    "peak_rss_bytes": second.get("peak_rss_bytes"),
                },
                "posterior_median_shift_stride_2_minus_stride_1": median_shift,
                "posterior_interval_endpoint_shift_stride_2_minus_stride_1": interval_endpoint_shift,
                "log_evidence_stride_2_minus_stride_1": log_evidence_difference,
                "absolute_log_evidence_difference_stride_2_minus_stride_1": (
                    absolute_log_evidence_difference
                ),
                "evidence_difference_allowed": evidence_allowed_difference,
                "posterior_gate_passed": run_posterior_passed,
                "evidence_gate_passed": run_evidence_passed,
            }
        )

    paired_count = len(paired)
    stride_two_runs = [
        record
        for record in stride_two_nested.get("runs", [])
        if isinstance(record, Mapping) and record.get("status") == "completed"
    ]
    nested_gate = stride_two_nested.get("gate", {})
    natural_convergence = bool(
        isinstance(nested_gate, Mapping)
        and nested_gate.get("natural_convergence", False)
    )
    chi_square = bool(
        isinstance(nested_gate, Mapping) and nested_gate.get("chi_square", False)
    )
    gate = {
        "paired_seed_runs": paired_count == 2,
        "stride_2_completed_runs": len(stride_two_runs) == 2,
        "stride_1_natural_convergence": bool(
            stride_one_nested.get("gate", {}).get("natural_convergence", False)
            if isinstance(stride_one_nested.get("gate", {}), Mapping)
            else False
        ),
        "stride_2_natural_convergence": natural_convergence,
        "posterior_centres_and_intervals": paired_count == 2 and posterior_passed,
        "evidence_agreement_with_reported_errors": paired_count == 2 and evidence_passed,
        "stride_2_chi_square": chi_square,
    }
    gate["passed"] = bool(all(gate.values()))
    return {
        "status": "pass" if gate["passed"] else "fail",
        "comparison": "paired same-seed stride-1 versus stride-2 four-parameter PyMultiNest",
        "posterior_interval_statement": "central 90 percent weighted posterior interval",
        "evidence_tolerance": evidence_tolerance,
        "parameter_shift_tolerances": dict(STRIDE_POSTERIOR_PARAMETER_SHIFT_TOLERANCES),
        "paired_runs": paired,
        "gate": gate,
        "calls": {
            "stride_1": [
                record.get("likelihood_evaluations")
                for record in stride_one_nested.get("runs", [])
                if isinstance(record, Mapping)
            ],
            "stride_2": [
                record.get("likelihood_evaluations")
                for record in stride_two_nested.get("runs", [])
                if isinstance(record, Mapping)
            ],
        },
        "wall_time_seconds": {
            "stride_1": [
                record.get("elapsed_seconds")
                for record in stride_one_nested.get("runs", [])
                if isinstance(record, Mapping)
            ],
            "stride_2": [
                record.get("elapsed_seconds")
                for record in stride_two_nested.get("runs", [])
                if isinstance(record, Mapping)
            ],
        },
        "peak_rss_bytes": {
            "stride_1": [
                record.get("peak_rss_bytes")
                for record in stride_one_nested.get("runs", [])
                if isinstance(record, Mapping)
            ],
            "stride_2": [
                record.get("peak_rss_bytes")
                for record in stride_two_nested.get("runs", [])
                if isinstance(record, Mapping)
            ],
        },
        "limitation": (
            "This comparison measures retrieval changes caused by native stride. "
            "It does not establish a general stride-2 accuracy bound for other "
            "opacity tables or atmospheric states."
        ),
    }


def _production_stride_decision(
    stride_record: Mapping[str, object],
) -> dict[str, object]:
    """Choose the production stride after a complete paired assessment.

    The nested stride comparison remains a strict scientific diagnostic.  A
    failed candidate comparison therefore does not become a generic workflow
    failure when both sampler runs completed naturally.  In that case stride
    1 remains the approved production reference and stride 2 is explicitly
    rejected.  Missing or failed sampler runs remain an outer failure.
    """

    comparison = stride_record.get("nested_posterior_comparison")
    if not isinstance(comparison, Mapping):
        return {
            "status": "incomplete",
            "decision": "incomplete",
            "assessment_complete": False,
            "reference_stride": 1,
            "candidate_stride": 2,
            "production_stride": None,
            "reference_stride_decision": "not_assessed",
            "candidate_stride_decision": "not_assessed",
            "reason": "paired stride-1 versus stride-2 nested assessment is missing",
            "failed_checks": ["paired_nested_assessment_present"],
        }

    comparison_gate = comparison.get("gate")
    stride_gate = stride_record.get("gate")
    if not isinstance(comparison_gate, Mapping) or not isinstance(stride_gate, Mapping):
        return {
            "status": "incomplete",
            "decision": "incomplete",
            "assessment_complete": False,
            "reference_stride": 1,
            "candidate_stride": 2,
            "production_stride": None,
            "reference_stride_decision": "not_assessed",
            "candidate_stride_decision": "not_assessed",
            "reason": "paired stride assessment is missing its gate mapping",
            "failed_checks": ["paired_assessment_gate_present"],
        }

    paired_runs = comparison.get("paired_runs")
    paired_records_present = (
        isinstance(paired_runs, Sequence)
        and not isinstance(paired_runs, (str, bytes))
        and len(paired_runs) == 2
        and all(
            isinstance(record, Mapping)
            and isinstance(record.get("stride_1"), Mapping)
            and isinstance(record.get("stride_2"), Mapping)
            for record in paired_runs
        )
    )
    stride_two_nested = stride_record.get("stride_2_nested")
    stride_two_runs = (
        stride_two_nested.get("runs", [])
        if isinstance(stride_two_nested, Mapping)
        else []
    )
    sampler_runs_complete = (
        isinstance(stride_two_runs, Sequence)
        and not isinstance(stride_two_runs, (str, bytes))
        and len(stride_two_runs) == 2
        and all(
            isinstance(record, Mapping)
            and record.get("status") == "completed"
            and bool(record.get("converged", False))
            for record in stride_two_runs
        )
    )
    complete_checks = {
        "paired_seed_runs": comparison_gate.get("paired_seed_runs") is True,
        "paired_records_present": paired_records_present,
        "stride_2_completed_runs": comparison_gate.get("stride_2_completed_runs") is True,
        "stride_1_natural_convergence": comparison_gate.get(
            "stride_1_natural_convergence"
        )
        is True,
        "stride_2_natural_convergence": comparison_gate.get(
            "stride_2_natural_convergence"
        )
        is True,
        "stride_2_sampler_records": sampler_runs_complete,
    }
    assessment_complete = bool(all(complete_checks.values()))
    if not assessment_complete:
        return {
            "status": "incomplete",
            "decision": "incomplete",
            "assessment_complete": False,
            "reference_stride": 1,
            "candidate_stride": 2,
            "production_stride": None,
            "reference_stride_decision": "not_assessed",
            "candidate_stride_decision": "not_assessed",
            "reason": "paired stride assessment has incomplete or failed sampler runs",
            "failed_checks": [
                name for name, passed in complete_checks.items() if not passed
            ],
        }

    comparison_failures = [
        name
        for name, passed in comparison_gate.items()
        if name != "passed" and isinstance(passed, bool) and not passed
    ]
    if comparison_gate.get("passed") is False and not comparison_failures:
        comparison_failures.append("nested_comparison_gate")
    stride_failures = [
        name
        for name, passed in stride_gate.items()
        if name != "passed" and isinstance(passed, bool) and not passed
    ]
    failed_checks = sorted(set(comparison_failures + stride_failures))
    candidate_approved = not failed_checks
    if candidate_approved:
        return {
            "status": "approved",
            "decision": "approve",
            "assessment_complete": True,
            "reference_stride": 1,
            "candidate_stride": 2,
            "production_stride": 2,
            "reference_stride_decision": "approve",
            "candidate_stride_decision": "approve",
            "reason": "complete paired assessment passed all stride-2 gates",
            "failed_checks": [],
        }

    evidence_bias = (
        comparison_gate.get("evidence_agreement_with_reported_errors") is False
    )
    reason = (
        "stride-2 rejected because paired evidence agreement failed; "
        "stride-1 is the approved production stride"
        if evidence_bias
        else "stride-2 rejected because one or more complete candidate gates failed; "
        "stride-1 is the approved production stride"
    )
    return {
        "status": "rejected",
        "decision": "reject",
        "assessment_complete": True,
        "reference_stride": 1,
        "candidate_stride": 2,
        "production_stride": 1,
        "reference_stride_decision": "approve",
        "candidate_stride_decision": "reject",
        "reason": reason,
        "evidence_bias": evidence_bias,
        "failed_checks": failed_checks,
    }


def _workflow_status(
    *,
    run_pymultinest: bool,
    run_native_stride_comparison: bool,
    recovery_passed: bool,
    memory_passed: bool,
    cpu_passed: bool,
    pymultinest_passed: bool | None,
    production_stride_decision: Mapping[str, object],
) -> str:
    """Return the outer status while preserving strict diagnostic gates."""

    if run_pymultinest:
        native_proof_passed = all(
            (
                recovery_passed,
                memory_passed,
                cpu_passed,
                pymultinest_passed is True,
            )
        )
        if not native_proof_passed:
            return "fail"
        if not run_native_stride_comparison:
            return "pass_native_stride_1"
        assessment_is_complete = (
            production_stride_decision.get("assessment_complete") is True
            and production_stride_decision.get("decision") in {"approve", "reject"}
        )
        if not assessment_is_complete:
            return "fail"
        if production_stride_decision.get("decision") == "approve":
            return "pass_native_stride_1_stride_2_approved"
        return "pass_native_stride_1_stride_2_rejected"

    if run_native_stride_comparison:
        # A stride comparison without the native nested proof has no complete
        # paired candidate assessment.  Keep this as an explicit failure.
        return "fail"
    return (
        "pass_recovery_only_nested_not_run"
        if all((recovery_passed, memory_passed, cpu_passed))
        else "fail"
    )


def _run_stride_comparison(
    stride_one_problem: MultiDatasetRetrievalProblem,
    stride_one_truth: Mapping[str, object],
    observations: object,
    likelihood: MixedMultiDatasetLikelihood,
    stride_one_oe: object,
    *,
    output_dir: Path,
    stride_one_nested: Mapping[str, object] | None = None,
) -> dict[str, object]:
    started = perf_counter()
    stride_two_model, stride_two_preparation = combined._build_forward_model(
        native_sampling_stride=2
    )
    stride_two_problem = _problem(
        stride_two_model,
        observations,
        likelihood,
        native_sampling_stride=2,
    )
    stride_two_truth = stride_two_model(combined.TRUTH)
    stride_two_oe = _run_oe(stride_two_problem, output_dir=output_dir / "stride_2")
    stride_one_best = stride_one_oe.best_fit_parameters
    stride_two_best = stride_two_oe.best_fit_parameters
    bias_vs_stride_one = {
        name: float(stride_two_best[name] - stride_one_best[name])
        for name in stride_one_problem.parameter_names
    }
    error_vs_truth = {
        name: abs(float(stride_two_best[name]) - float(combined.TRUTH[name]))
        for name in stride_one_problem.parameter_names
    }
    stride_two_stats = _fit_statistics(
        stride_two_problem,
        observations,
        stride_two_problem.model_spectra(stride_two_best),
        stride_two_best,
    )
    spectral_metrics = _stride_metrics(
        np.asarray(stride_one_truth["high_resolution"].values, dtype=float),
        np.asarray(stride_two_truth["high_resolution"].values, dtype=float),
    )
    bias_passed = all(
        abs(bias_vs_stride_one[name]) <= ABSOLUTE_PARAMETER_TOLERANCES[name]
        for name in stride_one_problem.parameter_names
    )
    truth_recovery = all(
        error_vs_truth[name] <= ABSOLUTE_PARAMETER_TOLERANCES[name]
        for name in stride_one_problem.parameter_names
    )
    spectral_passed = (
        spectral_metrics["rms_relative_error"] <= STRIDE_RMS_LIMIT
        and spectral_metrics["max_absolute_relative_error"] <= STRIDE_MAX_LIMIT
    )
    gate = {
        "spectral_accuracy": spectral_passed,
        "retrieval_bias_vs_stride_1": bias_passed,
        "stride_2_truth_recovery": truth_recovery,
        "stride_2_chi_square": bool(stride_two_stats["chi_square_passed"]),
        "stride_2_oe_converged": bool(stride_two_oe.converged),
    }
    stride_two_nested: dict[str, object] | None = None
    nested_comparison: dict[str, object] | None = None
    if stride_one_nested is not None:
        settings = stride_one_nested.get("settings", {})
        if not isinstance(settings, Mapping):
            raise ValueError("stride-1 nested result is missing sampler settings")
        seed_pair = settings.get("seed_pair", ())
        if not isinstance(seed_pair, Sequence) or len(seed_pair) != 2:
            raise ValueError("stride-1 nested result is missing its two seeds")
        stride_two_nested = _run_full_pymultinest(
            stride_two_problem,
            observations,
            output_dir=output_dir / "pymultinest_stride_2",
            seed=int(seed_pair[0]),
            n_live_points=int(settings["n_live_points"]),
            max_iter=int(settings["max_iter"]),
            evidence_tolerance=float(settings["evidence_tolerance"]),
            sampling_efficiency=float(settings["sampling_efficiency"]),
        )
        stride_two_nested["scope"] = (
            "stride-2 full four-parameter atmospheric posterior"
        )
        nested_comparison = _compare_nested_stride_posteriors(
            stride_one_nested,
            stride_two_nested,
        )
        gate["nested_stride_2_comparison"] = bool(nested_comparison["gate"]["passed"])
    return {
        "status": "pass" if all(gate.values()) else "fail",
        "reference_stride": 1,
        "candidate_stride": 2,
        "spectral_metrics_at_injection": spectral_metrics,
        "spectral_limits": {
            "rms_relative_error_max": STRIDE_RMS_LIMIT,
            "max_absolute_relative_error_max": STRIDE_MAX_LIMIT,
        },
        "stride_1_best_fit": dict(stride_one_best),
        "stride_2_best_fit": dict(stride_two_best),
        "parameter_bias_stride_2_minus_stride_1": bias_vs_stride_one,
        "stride_2_absolute_error_vs_truth": error_vs_truth,
        "stride_2_fit_statistics": stride_two_stats,
        "stride_2_nested": stride_two_nested,
        "nested_posterior_comparison": nested_comparison,
        "stride_2_preparation": {
            key: value
            for key, value in stride_two_preparation.items()
            if key != "high_response"
        },
        "gate": gate,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> dict[str, object]:
    args = _parse_args()
    workflow_started = perf_counter()
    max_memory_bytes = int(float(args.max_memory_gib) * 1024**3)
    if max_memory_bytes <= 0 or max_memory_bytes >= MAX_MEMORY_BYTES:
        raise ValueError("max-memory-gib must be positive and below 2 GiB")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(args.noise_scale) or args.noise_scale < 0.0:
        raise ValueError("noise-scale must be finite and non-negative")
    if args.pymultinest_live_points < 50:
        raise ValueError("pymultinest-live-points must be at least 50")
    if args.pymultinest_max_iter != 0:
        raise ValueError("pymultinest-max-iter must be 0 for natural convergence")
    if (
        not np.isfinite(args.pymultinest_evidence_tolerance)
        or args.pymultinest_evidence_tolerance <= 0.0
    ):
        raise ValueError("pymultinest-evidence-tolerance must be finite and positive")
    if (
        not np.isfinite(args.pymultinest_sampling_efficiency)
        or not 0.0 < args.pymultinest_sampling_efficiency <= 1.0
    ):
        raise ValueError("pymultinest-sampling-efficiency must be in (0, 1]")

    table_inputs = _check_tables()
    build_started = perf_counter()
    stride_one_model, stride_one_preparation = combined._build_forward_model(
        native_sampling_stride=1
    )
    stride_one_truth = stride_one_model(combined.TRUTH)
    observations = _observations(
        stride_one_truth,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )
    likelihood = _likelihood(observations)
    stride_one_problem = _problem(
        stride_one_model,
        observations,
        likelihood,
        native_sampling_stride=1,
    )
    build_seconds = perf_counter() - build_started

    oe_started = perf_counter()
    stride_one_oe = _run_oe(
        stride_one_problem,
        output_dir=OUTPUT_DIR / "stride_1_optimal_estimation",
    )
    oe_seconds = perf_counter() - oe_started
    stride_one_best_spectra = stride_one_problem.model_spectra(
        stride_one_oe.best_fit_parameters
    )
    reduced_bounds, fit_gate_description = combined._reduced_chi_square_gate(
        args.noise_scale
    )
    recovery = combined.evaluate_multi_dataset_injection_recovery(
        case_name="combined-low-high-resolution-full-pymultinest",
        truth=combined.TRUTH,
        estimates=stride_one_oe.best_fit_parameters,
        absolute_tolerances=ABSOLUTE_PARAMETER_TOLERANCES,
        observations=observations,
        best_fit_spectra=stride_one_best_spectra,
        likelihood=likelihood,
        seed=args.seed,
        parameter_order=stride_one_problem.parameter_names,
        posterior_covariance=stride_one_oe.inference_result.covariance,
        inference_converged=stride_one_oe.converged,
        reduced_chi_square_bounds=reduced_bounds,
        metadata={
            "noise_scale": f"{args.noise_scale:.12g}",
            "likelihood": "Gaussian low resolution + projected high resolution",
            "fit_gate": fit_gate_description,
        },
    )

    if args.run_pymultinest:
        pymultinest_record = _run_full_pymultinest(
            stride_one_problem,
            observations,
            output_dir=OUTPUT_DIR / "pymultinest_full",
            seed=args.seed,
            n_live_points=args.pymultinest_live_points,
            max_iter=args.pymultinest_max_iter,
            evidence_tolerance=args.pymultinest_evidence_tolerance,
            sampling_efficiency=args.pymultinest_sampling_efficiency,
        )
    else:
        pymultinest_record = {
            "status": "not_run",
            "gate_passed": None,
            "scope": "full four-parameter atmospheric posterior",
            "full_four_parameter_nested_sampling": "not_run",
            "limitation": "Use --run-pymultinest for the two-seed full posterior proof.",
        }

    if args.run_native_stride_comparison:
        stride_record = _run_stride_comparison(
            stride_one_problem,
            stride_one_truth,
            observations,
            likelihood,
            stride_one_oe,
            output_dir=OUTPUT_DIR / "native_stride_comparison",
            stride_one_nested=(
                pymultinest_record if args.run_pymultinest else None
            ),
        )
    else:
        stride_record = {
            "status": "not_run",
            "gate_passed": None,
            "reference_stride": 1,
            "candidate_stride": 2,
            "limitation": "Use --run-native-stride-comparison for the retrieval-bias check.",
        }

    peak_rss = _peak_rss_bytes()
    thread_values = _thread_values()
    memory_passed = peak_rss < max_memory_bytes
    cpu_passed = all(1 <= value <= 3 for value in thread_values.values())
    recovery_passed = bool(recovery.passed)
    pymultinest_passed = (
        None
        if not args.run_pymultinest
        else bool(pymultinest_record["gate"]["passed"])
    )
    stride_passed = (
        None
        if not args.run_native_stride_comparison
        else all(stride_record["gate"].values())
    )
    if args.run_native_stride_comparison:
        production_stride_decision = _production_stride_decision(stride_record)
    else:
        production_stride_decision = {
            "status": "not_run",
            "decision": "not_run",
            "assessment_complete": False,
            "reference_stride": 1,
            "candidate_stride": 2,
            "production_stride": None,
            "reference_stride_decision": "not_assessed",
            "candidate_stride_decision": "not_assessed",
            "reason": "paired native-stride assessment was not requested",
            "failed_checks": ["paired_nested_assessment_not_requested"],
        }
    status = _workflow_status(
        run_pymultinest=args.run_pymultinest,
        run_native_stride_comparison=args.run_native_stride_comparison,
        recovery_passed=recovery_passed,
        memory_passed=memory_passed,
        cpu_passed=cpu_passed,
        pymultinest_passed=pymultinest_passed,
        production_stride_decision=production_stride_decision,
    )

    report: dict[str, object] = recovery.to_mapping()
    recovery_component_passed = bool(report.get("passed", False))
    overall_passed = status.startswith("pass")
    report.update(
        {
            "status": status,
            "passed": overall_passed,
            "recovery_component_passed": recovery_component_passed,
            "workflow": "full four-parameter combined-resolution validation",
            "command": " ".join(sys.argv),
            "recovery_passed": recovery_passed,
            "memory_guard_passed": memory_passed,
            "cpu_guard_passed": cpu_passed,
            "table_inputs": table_inputs,
            "truth": combined.TRUTH,
            "best_fit": dict(stride_one_oe.best_fit_parameters),
            "preparation": {
                key: value
                for key, value in stride_one_preparation.items()
                if key != "high_response"
            },
            "timings_seconds": {
                "model_and_truth": build_seconds,
                "optimal_estimation": oe_seconds,
                "total_workflow": perf_counter() - workflow_started,
            },
            "resources": {
                "thread_values": thread_values,
                "max_memory_bytes": max_memory_bytes,
                "measured_peak_rss_bytes": peak_rss,
                "memory_rule": "measured_peak_rss < max_memory_bytes < 2 GiB",
                "cpu_rule": "all numerical thread variables are between 1 and 3; PyMultiNest MPI size is 1",
                "opacity_memory_limit_bytes": combined.OPACITY_MEMORY_BYTES,
            },
            "pymultinest": pymultinest_record,
            "native_stride_comparison": stride_record,
            "native_stride_gate_passed": stride_passed,
            "production_stride_decision": production_stride_decision,
        }
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = REPORT_PATH.with_name(
        f".{REPORT_PATH.name}.{os.getpid()}.tmp"
    )
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(REPORT_PATH)
    print(json.dumps(report, indent=2, sort_keys=True))
    if status == "fail":
        raise SystemExit("full combined-resolution validation failed")
    return report


if __name__ == "__main__":
    main()
