"""Bayesian leave-one-out diagnostics for retrieval posterior samples.

The implementation follows Welbanks et al. (2023) and uses Pareto-smoothed
importance sampling (PSIS) through ArviZ. It evaluates each independent
Gaussian likelihood term at each posterior draw, avoiding one full retrieval
per spectral datum while retaining Pareto-k reliability diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertDataError, RobertValidationError


LEAVE_ONE_OUT_SCHEMA_VERSION = "1.0"
LEAVE_ONE_OUT_SUMMARY_FILENAME = "leave_one_out.json"
LEAVE_ONE_OUT_ARRAYS_FILENAME = "leave_one_out_arrays.npz"


class PointwiseRetrievalProblem(Protocol):
    """Structural contract required by :func:`run_psis_leave_one_out`."""

    name: str

    def pointwise_log_likelihood_from_vector(
        self,
        vector: ArrayLike,
    ) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class LeaveOneOutResult:
    """Portable PSIS-LOO summary and data-point-level diagnostics."""

    model_name: str
    elpd_loo: float
    standard_error: float
    effective_parameter_count: float
    pointwise_elpd: NDArray[np.float64]
    pareto_k: NDArray[np.float64]
    observation_ids: tuple[str, ...]
    dataset_names: tuple[str, ...]
    wavelength: NDArray[np.float64]
    wavelength_unit: str
    posterior_draws: int
    source_samples: int
    source_effective_sample_size: float
    posterior_resampled: bool
    pareto_k_threshold: float
    warning: bool
    likelihood_normalized: bool
    pointwise_log_likelihood: NDArray[np.float64]
    selected_source_indices: NDArray[np.int64]
    schema_version: str = LEAVE_ONE_OUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model_name or not self.schema_version:
            raise RobertValidationError("LOO model name and schema version are required")
        for name, value in (
            ("elpd_loo", self.elpd_loo),
            ("standard_error", self.standard_error),
            ("effective_parameter_count", self.effective_parameter_count),
            ("source_effective_sample_size", self.source_effective_sample_size),
            ("pareto_k_threshold", self.pareto_k_threshold),
        ):
            if not np.isfinite(value):
                raise RobertValidationError(f"LOO {name} must be finite")
        if self.standard_error < 0.0 or self.effective_parameter_count < 0.0:
            raise RobertValidationError("LOO uncertainty and effective parameters cannot be negative")
        if not 0.0 < self.pareto_k_threshold <= 1.0:
            raise RobertValidationError("pareto_k_threshold must lie in (0, 1]")
        if self.posterior_draws < 1 or self.source_samples < 1:
            raise RobertValidationError("LOO sample counts must be positive")

        pointwise = _readonly_vector(self.pointwise_elpd, "pointwise_elpd", finite=True)
        pareto = _readonly_vector(self.pareto_k, "pareto_k", finite=False)
        wavelength = _readonly_vector(self.wavelength, "wavelength", finite=True)
        log_likelihood = np.array(self.pointwise_log_likelihood, dtype=float, copy=True)
        selected_indices = np.array(self.selected_source_indices, dtype=np.int64, copy=True)
        size = pointwise.size
        if pareto.size != size or wavelength.size != size:
            raise RobertValidationError("LOO pointwise arrays must have matching lengths")
        if len(self.observation_ids) != size or len(self.dataset_names) != size:
            raise RobertValidationError("LOO observation metadata must match pointwise arrays")
        if log_likelihood.shape != (self.posterior_draws, size):
            raise RobertValidationError(
                "LOO log-likelihood matrix must match posterior draws and observations"
            )
        if not np.all(np.isfinite(log_likelihood)):
            raise RobertValidationError("LOO log-likelihood matrix must be finite")
        if selected_indices.shape != (self.posterior_draws,) or np.any(selected_indices < 0):
            raise RobertValidationError(
                "LOO selected source indices must match posterior draws and be non-negative"
            )
        if not self.wavelength_unit:
            raise RobertValidationError("LOO wavelength unit must not be empty")
        object.__setattr__(self, "pointwise_elpd", pointwise)
        object.__setattr__(self, "pareto_k", pareto)
        object.__setattr__(self, "wavelength", wavelength)
        log_likelihood.setflags(write=False)
        selected_indices.setflags(write=False)
        object.__setattr__(self, "pointwise_log_likelihood", log_likelihood)
        object.__setattr__(self, "selected_source_indices", selected_indices)
        object.__setattr__(self, "observation_ids", tuple(str(value) for value in self.observation_ids))
        object.__setattr__(self, "dataset_names", tuple(str(value) for value in self.dataset_names))

    @property
    def unreliable_indices(self) -> tuple[int, ...]:
        """Indices whose PSIS approximation requires an exact LOO refit."""

        return tuple(
            int(index)
            for index in np.flatnonzero(
                ~np.isfinite(self.pareto_k) | (self.pareto_k > self.pareto_k_threshold)
            )
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable summary including pointwise scores."""

        unreliable = set(self.unreliable_indices)
        points = []
        for index in range(self.pointwise_elpd.size):
            points.append(
                {
                    "index": index,
                    "observation_id": self.observation_ids[index],
                    "dataset": self.dataset_names[index],
                    "wavelength": float(self.wavelength[index]),
                    "pointwise_elpd": float(self.pointwise_elpd[index]),
                    "pareto_k": _json_float(self.pareto_k[index]),
                    "requires_exact_refit": index in unreliable,
                }
            )
        return {
            "schema_version": self.schema_version,
            "method": "psis-loo",
            "model_name": self.model_name,
            "elpd_loo": self.elpd_loo,
            "standard_error": self.standard_error,
            "effective_parameter_count": self.effective_parameter_count,
            "number_observations": self.pointwise_elpd.size,
            "posterior_draws": self.posterior_draws,
            "source_samples": self.source_samples,
            "source_effective_sample_size": self.source_effective_sample_size,
            "posterior_resampled": self.posterior_resampled,
            "likelihood_normalized": self.likelihood_normalized,
            "pareto_k_threshold": self.pareto_k_threshold,
            "warning": self.warning,
            "unreliable_point_count": len(unreliable),
            "wavelength_unit": self.wavelength_unit,
            "points": points,
            "interpretation": (
                "Higher ELPD indicates better expected out-of-sample predictive accuracy. "
                "Points marked requires_exact_refit must not be interpreted from PSIS alone. "
                + (
                    "The Gaussian normalization is included."
                    if self.likelihood_normalized
                    else "The Gaussian normalization is omitted, so absolute ELPD is shifted; "
                    "same-data model differences and Pareto-k diagnostics are unchanged."
                )
            ),
        }


@dataclass(frozen=True)
class LeaveOneOutComparison:
    """Pairwise ELPD difference; positive values favor ``first_model``."""

    first_model: str
    second_model: str
    delta_elpd: float
    standard_error: float
    pointwise_delta: NDArray[np.float64]
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.first_model or not self.second_model:
            raise RobertValidationError("LOO comparison model names are required")
        if not np.isfinite(self.delta_elpd) or not np.isfinite(self.standard_error):
            raise RobertValidationError("LOO comparison statistics must be finite")
        if self.standard_error < 0.0:
            raise RobertValidationError("LOO comparison standard error cannot be negative")
        delta = _readonly_vector(self.pointwise_delta, "pointwise_delta", finite=True)
        if len(self.observation_ids) != delta.size:
            raise RobertValidationError("LOO comparison metadata must match pointwise delta")
        object.__setattr__(self, "pointwise_delta", delta)
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))

    def to_mapping(self) -> dict[str, object]:
        return {
            "first_model": self.first_model,
            "second_model": self.second_model,
            "delta_elpd": self.delta_elpd,
            "standard_error": self.standard_error,
            "positive_favors": self.first_model,
            "pointwise_delta": self.pointwise_delta.tolist(),
            "observation_ids": list(self.observation_ids),
            "note": "The ELPD difference divided by its standard error is not a sigma significance.",
        }


def psis_leave_one_out(
    pointwise_log_likelihood: ArrayLike,
    *,
    model_name: str = "model",
    observation_ids: tuple[str, ...] | None = None,
    dataset_names: tuple[str, ...] | None = None,
    wavelength: ArrayLike | None = None,
    wavelength_unit: str = "point_index",
    pareto_k_threshold: float | None = None,
    source_samples: int | None = None,
    source_effective_sample_size: float | None = None,
    posterior_resampled: bool = False,
    likelihood_normalized: bool = True,
    selected_source_indices: ArrayLike | None = None,
) -> LeaveOneOutResult:
    """Compute PSIS-LOO from ``(posterior_draw, observation)`` log likelihoods.

    ``pareto_k_threshold=None`` uses ArviZ's sample-size-dependent threshold.
    Set it to ``0.7`` to reproduce the fixed threshold used by Welbanks et al.
    (2023). The posterior draws must be equally weighted.
    """

    matrix = np.asarray(pointwise_log_likelihood, dtype=float)
    if matrix.ndim != 2:
        raise RobertValidationError(
            "pointwise_log_likelihood must have shape (posterior_draw, observation)"
        )
    draws, observations = matrix.shape
    if draws < 20:
        raise RobertValidationError("PSIS-LOO requires at least 20 posterior draws")
    if observations < 1 or not np.all(np.isfinite(matrix)):
        raise RobertValidationError("pointwise log likelihoods must be finite and non-empty")

    try:
        import arviz as az
    except ImportError as exc:
        raise RobertConfigError(
            "PSIS-LOO requires ArviZ. Install ROBERT with the diagnostics extra: "
            "`conda run -n robert-exoplanets python -m pip install -e "
            "'.[diagnostics]'`."
        ) from exc

    likelihood_cube = matrix[np.newaxis, :, :]
    coordinates = {"observation_id": np.arange(observations)}
    dimensions = {"observation": ["observation_id"]}
    try:
        inference_data = az.from_dict(
            {"log_likelihood": {"observation": likelihood_cube}},
            sample_dims=["chain", "draw"],
            coords=coordinates,
            dims=dimensions,
        )
    except TypeError:
        # ArviZ 0.x used keyword groups instead of the nested mapping accepted
        # by the ArviZ 1.x DataTree API.
        inference_data = az.from_dict(
            log_likelihood={"observation": likelihood_cube},
            coords=coordinates,
            dims=dimensions,
        )
    loo = az.loo(inference_data, pointwise=True, reff=1.0)

    pointwise = np.asarray(_loo_attribute(loo, "elpd_i", "loo_i"), dtype=float).reshape(-1)
    pareto = np.asarray(_loo_attribute(loo, "pareto_k"), dtype=float).reshape(-1)
    default_threshold = float(
        getattr(loo, "good_k", min(1.0 - 1.0 / math.log10(draws), 0.7))
    )
    threshold = default_threshold if pareto_k_threshold is None else float(pareto_k_threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise RobertValidationError("pareto_k_threshold must lie in (0, 1]")

    ids = observation_ids or tuple(f"point[{index}]" for index in range(observations))
    datasets = dataset_names or tuple("observation" for _ in range(observations))
    wavelength_values = (
        np.arange(observations, dtype=float)
        if wavelength is None
        else np.asarray(wavelength, dtype=float)
    )
    if len(ids) != observations or len(datasets) != observations:
        raise RobertValidationError("LOO observation metadata has the wrong length")

    resolved_source_samples = draws if source_samples is None else int(source_samples)
    resolved_ess = (
        float(resolved_source_samples)
        if source_effective_sample_size is None
        else float(source_effective_sample_size)
    )
    resolved_indices = (
        np.arange(draws, dtype=np.int64)
        if selected_source_indices is None
        else np.asarray(selected_source_indices, dtype=np.int64)
    )
    warning = bool(np.any(~np.isfinite(pareto) | (pareto > threshold)))
    return LeaveOneOutResult(
        model_name=model_name,
        elpd_loo=float(_loo_attribute(loo, "elpd", "elpd_loo", "loo")),
        standard_error=float(_loo_attribute(loo, "se", "loo_se")),
        effective_parameter_count=float(_loo_attribute(loo, "p", "p_loo")),
        pointwise_elpd=pointwise,
        pareto_k=pareto,
        observation_ids=tuple(ids),
        dataset_names=tuple(datasets),
        wavelength=wavelength_values,
        wavelength_unit=wavelength_unit,
        posterior_draws=draws,
        source_samples=resolved_source_samples,
        source_effective_sample_size=resolved_ess,
        posterior_resampled=bool(posterior_resampled),
        pareto_k_threshold=threshold,
        warning=warning,
        likelihood_normalized=bool(likelihood_normalized),
        pointwise_log_likelihood=matrix,
        selected_source_indices=resolved_indices,
    )


def run_psis_leave_one_out(
    problem: PointwiseRetrievalProblem,
    samples: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    max_posterior_draws: int = 2_000,
    seed: int = 0,
    pareto_k_threshold: float | None = None,
) -> LeaveOneOutResult:
    """Evaluate a retrieval posterior and return its PSIS-LOO diagnostic.

    Weighted nested-sampler output is converted reproducibly to equal-weight
    posterior draws using systematic resampling before PSIS is applied.
    """

    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] < 1:
        raise RobertValidationError("posterior samples must be a non-empty 2D array")
    if not np.all(np.isfinite(sample_array)):
        raise RobertValidationError("posterior samples must be finite")
    if isinstance(max_posterior_draws, bool) or int(max_posterior_draws) < 20:
        raise RobertValidationError("max_posterior_draws must be at least 20")
    if isinstance(seed, bool) or int(seed) < 0:
        raise RobertValidationError("LOO seed must be a non-negative integer")

    draws, selected_indices, effective_sample_size, resampled = _equal_weight_draws(
        sample_array,
        weights,
        max_draws=int(max_posterior_draws),
        seed=int(seed),
    )
    rows = []
    expected_observations = None
    for draw in draws:
        terms = np.asarray(problem.pointwise_log_likelihood_from_vector(draw), dtype=float)
        if terms.ndim != 1 or not np.all(np.isfinite(terms)):
            raise RobertValidationError(
                "posterior draw produced invalid pointwise log-likelihood terms"
            )
        if expected_observations is None:
            expected_observations = terms.size
        elif terms.size != expected_observations:
            raise RobertValidationError(
                "pointwise log-likelihood length changed between posterior draws"
            )
        rows.append(terms)
    matrix = np.vstack(rows)
    ids, datasets, wavelength, wavelength_unit = _observation_metadata(problem)
    if len(ids) != matrix.shape[1]:
        raise RobertValidationError(
            "problem observation metadata does not match pointwise likelihood length"
        )
    result = psis_leave_one_out(
        matrix,
        model_name=problem.name,
        observation_ids=ids,
        dataset_names=datasets,
        wavelength=wavelength,
        wavelength_unit=wavelength_unit,
        pareto_k_threshold=pareto_k_threshold,
        source_samples=sample_array.shape[0],
        source_effective_sample_size=effective_sample_size,
        posterior_resampled=resampled,
        likelihood_normalized=bool(
            getattr(getattr(problem, "likelihood", None), "include_normalization", False)
        ),
        selected_source_indices=selected_indices,
    )
    return result


def compare_psis_leave_one_out(
    first: LeaveOneOutResult,
    second: LeaveOneOutResult,
) -> LeaveOneOutComparison:
    """Compare aligned models using the pointwise ELPD difference and its SE."""

    if first.observation_ids != second.observation_ids:
        raise RobertValidationError("LOO comparisons require aligned observation IDs")
    delta = first.pointwise_elpd - second.pointwise_elpd
    count = delta.size
    standard_error = (
        float(np.sqrt(count * np.var(delta, ddof=1))) if count > 1 else 0.0
    )
    return LeaveOneOutComparison(
        first_model=first.model_name,
        second_model=second.model_name,
        delta_elpd=float(np.sum(delta)),
        standard_error=standard_error,
        pointwise_delta=delta,
        observation_ids=first.observation_ids,
    )


def write_leave_one_out_result(
    result: LeaveOneOutResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Serialize a LOO summary plus lossless numerical arrays."""

    destination = Path(output_dir).expanduser()
    summary_path = destination / LEAVE_ONE_OUT_SUMMARY_FILENAME
    arrays_path = destination / LEAVE_ONE_OUT_ARRAYS_FILENAME
    try:
        destination.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(result.to_mapping(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        arrays: dict[str, Any] = {
            "pointwise_elpd": result.pointwise_elpd,
            "pareto_k": result.pareto_k,
            "wavelength": result.wavelength,
            "observation_ids": np.asarray(result.observation_ids),
            "dataset_names": np.asarray(result.dataset_names),
            "selected_source_indices": result.selected_source_indices,
            "pointwise_log_likelihood": result.pointwise_log_likelihood,
        }
        np.savez_compressed(arrays_path, **arrays)
    except OSError as exc:
        raise RobertDataError(f"failed to write LOO diagnostic under {destination}") from exc
    return summary_path, arrays_path


def plot_leave_one_out_result(
    result: LeaveOneOutResult,
    path: str | Path,
    *,
    style: str = "default",
    dpi: int = 180,
) -> Path:
    """Plot pointwise ELPD and Pareto-k reliability against wavelength."""

    import matplotlib.pyplot as plt

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context(style):
        figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
        for dataset in dict.fromkeys(result.dataset_names):
            selected = np.asarray(result.dataset_names) == dataset
            axes[0].scatter(
                result.wavelength[selected],
                result.pointwise_elpd[selected],
                s=18,
                label=dataset,
            )
            axes[1].scatter(
                result.wavelength[selected],
                result.pareto_k[selected],
                s=18,
                label=dataset,
            )
        axes[0].axhline(0.0, color="0.5", linewidth=0.8)
        axes[0].set_ylabel("Pointwise ELPD")
        axes[0].set_title(
            f"PSIS-LOO: ELPD = {result.elpd_loo:.2f} ± {result.standard_error:.2f}"
        )
        axes[1].axhline(
            result.pareto_k_threshold,
            color="#c44e52",
            linestyle="--",
            linewidth=1.0,
            label=f"reliability threshold ({result.pareto_k_threshold:.2f})",
        )
        axes[1].set_ylabel("Pareto k")
        axes[1].set_xlabel(
            "Observation index"
            if result.wavelength_unit == "point_index"
            else f"Wavelength ({result.wavelength_unit})"
        )
        axes[1].legend(loc="best", fontsize="small")
        figure.tight_layout()
        figure.savefig(target, dpi=dpi)
        plt.close(figure)
    return target


def _equal_weight_draws(
    samples: NDArray[np.float64],
    weights: ArrayLike | None,
    *,
    max_draws: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], float, bool]:
    count = min(samples.shape[0], max_draws)
    rng = np.random.default_rng(seed)
    if weights is None:
        effective_sample_size = float(samples.shape[0])
        if count == samples.shape[0]:
            indices = np.arange(count, dtype=np.int64)
            return samples.copy(), indices, effective_sample_size, False
        indices = np.sort(rng.choice(samples.shape[0], size=count, replace=False))
        return samples[indices].copy(), indices, effective_sample_size, True

    probability = np.asarray(weights, dtype=float)
    if probability.ndim != 1 or probability.size != samples.shape[0]:
        raise RobertValidationError("posterior weights must match sample rows")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise RobertValidationError("posterior weights must be finite and non-negative")
    total = float(np.sum(probability))
    if total <= 0.0:
        raise RobertValidationError("posterior weights must have a positive sum")
    probability = probability / total
    effective_sample_size = float(1.0 / np.sum(np.square(probability)))
    if np.allclose(probability, 1.0 / probability.size) and count == samples.shape[0]:
        indices = np.arange(count, dtype=np.int64)
        return samples.copy(), indices, effective_sample_size, False
    positions = (rng.random() + np.arange(count, dtype=float)) / count
    indices = np.searchsorted(np.cumsum(probability), positions, side="right")
    indices = np.minimum(indices, samples.shape[0] - 1).astype(np.int64)
    return samples[indices].copy(), indices, effective_sample_size, True


def _observation_metadata(
    problem: PointwiseRetrievalProblem,
) -> tuple[tuple[str, ...], tuple[str, ...], NDArray[np.float64], str]:
    if hasattr(problem, "observation"):
        observation = getattr(problem, "observation")
        valid = _valid_indices(observation)
        dataset = str(getattr(observation, "instrument", "observation") or "observation")
        ids = tuple(f"{dataset}[{index}]" for index in valid)
        datasets = tuple(dataset for _ in valid)
        wavelength = np.asarray(observation.wavelength, dtype=float)[valid]
        return ids, datasets, wavelength, str(observation.wavelength_unit)

    collection = getattr(problem, "observations", None)
    if collection is None:
        raise RobertValidationError("retrieval problem does not expose observation metadata")
    ids: list[str] = []
    datasets: list[str] = []
    wavelengths: list[float] = []
    units: list[str] = []
    for dataset in collection.datasets:
        observation = dataset.observation
        valid = _valid_indices(observation)
        ids.extend(f"{dataset.name}[{index}]" for index in valid)
        datasets.extend(dataset.name for _ in valid)
        wavelengths.extend(np.asarray(observation.wavelength, dtype=float)[valid])
        units.append(str(observation.wavelength_unit))
    if len(set(units)) == 1:
        wavelength = np.asarray(wavelengths, dtype=float)
        wavelength_unit = units[0]
    else:
        wavelength = np.arange(len(ids), dtype=float)
        wavelength_unit = "point_index"
    return tuple(ids), tuple(datasets), wavelength, wavelength_unit


def _valid_indices(observation: Any) -> NDArray[np.int64]:
    if observation.mask is None:
        return np.arange(observation.n_points, dtype=np.int64)
    return np.flatnonzero(np.asarray(observation.mask, dtype=bool)).astype(np.int64)


def _loo_attribute(result: Any, *names: str) -> Any:
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
        try:
            return result[name]
        except (KeyError, TypeError, IndexError):
            continue
    raise RobertDataError(f"ArviZ LOO result is missing expected fields: {', '.join(names)}")


def _readonly_vector(
    values: ArrayLike,
    name: str,
    *,
    finite: bool,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size < 1:
        raise RobertValidationError(f"{name} must be a non-empty one-dimensional array")
    if finite and not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


__all__ = [
    "LEAVE_ONE_OUT_ARRAYS_FILENAME",
    "LEAVE_ONE_OUT_SCHEMA_VERSION",
    "LEAVE_ONE_OUT_SUMMARY_FILENAME",
    "LeaveOneOutComparison",
    "LeaveOneOutResult",
    "compare_psis_leave_one_out",
    "plot_leave_one_out_result",
    "psis_leave_one_out",
    "run_psis_leave_one_out",
    "write_leave_one_out_result",
]
