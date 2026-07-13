"""Hybrid optimal-estimation and nested-sampling retrieval workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertDataError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

from .optimal_estimation import OptimalEstimationResult
from .priors import LogUniformPrior, RetrievalParameter, RetrievalParameterSet, UniformPrior
from .problem import RetrievalProblem
from .results import RetrievalResult
from .runner import run_retrieval
from .samplers import NestedSamplerResult

HYBRID_HANDOFF_FILENAME = "hybrid_handoff.json"


@dataclass(frozen=True)
class OENestedSamplingResult:
    """Result of an OE solve followed by nested sampling with refined priors."""

    optimal_estimation: RetrievalResult
    nested_sampling: RetrievalResult
    refined_problem: RetrievalProblem = field(repr=False)
    prior_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    output_dir: Path = Path(".")

    def __post_init__(self) -> None:
        object.__setattr__(self, "prior_bounds", immutable_mapping(self.prior_bounds))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True)
class NestedSamplingOEResult:
    """Result of nested sampling followed by an OE refinement."""

    nested_sampling: RetrievalResult
    optimal_estimation: RetrievalResult
    transferred_prior_state: Mapping[str, float] = field(default_factory=dict)
    output_dir: Path = Path(".")

    def __post_init__(self) -> None:
        object.__setattr__(self, "transferred_prior_state", immutable_mapping(self.transferred_prior_state))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def run_oe_then_nested_sampling(
    problem: RetrievalProblem,
    *,
    output_dir: str | Path,
    prior_sigma: float = 4.0,
    minimum_prior_fraction: float = 0.05,
    require_oe_convergence: bool = True,
    oe_kwargs: Mapping[str, object] | None = None,
    nested_kwargs: Mapping[str, object] | None = None,
    seed: int | None = None,
) -> OENestedSamplingResult:
    """Use OE location and covariance to define bounded nested-sampling priors.

    Each refined prior is centred on the OE state, spans ``prior_sigma`` OE
    standard deviations on either side, is clipped to the original scientific
    bounds, and retains at least ``minimum_prior_fraction`` of the original
    width. The original prior family (uniform or log-uniform) is preserved.
    """

    root = Path(output_dir).expanduser()
    oe_settings = _phase_kwargs(oe_kwargs, phase="optimal estimation")
    nested_settings = _phase_kwargs(nested_kwargs, phase="nested sampling")
    oe_result = run_retrieval(
        problem,
        method="optimal_estimation",
        output_dir=root / "optimal_estimation",
        **oe_settings,
    )
    if require_oe_convergence and not oe_result.converged:
        raise RobertConfigError(
            f"OE initialization did not converge: {oe_result.message}; "
            "nested sampling was not started"
        )
    inference = oe_result.inference_result
    if not isinstance(inference, OptimalEstimationResult):
        raise RobertConfigError("OE phase returned an unexpected inference result")
    refined_parameters = refine_priors_from_optimal_estimation(
        problem.parameters,
        inference,
        prior_sigma=prior_sigma,
        minimum_prior_fraction=minimum_prior_fraction,
    )
    refined_problem = replace(
        problem,
        parameters=refined_parameters,
        metadata={
            **dict(problem.metadata),
            "hybrid_workflow": "optimal_estimation_then_nested_sampling",
            "hybrid_source_problem": problem.name,
        },
    )
    nested_result = run_retrieval(
        refined_problem,
        method="ultranest",
        output_dir=root / "nested_sampling",
        seed=seed,
        **nested_settings,
    )
    bounds = dict(zip(refined_parameters.names, refined_parameters.bounds, strict=True))
    _write_handoff(
        root,
        {
            "workflow": "optimal_estimation_then_nested_sampling",
            "oe_converged": oe_result.converged,
            "oe_state": dict(oe_result.best_fit_parameters),
            "original_prior_bounds": _bounds_mapping(problem.parameters),
            "refined_prior_bounds": bounds,
            "prior_sigma": float(prior_sigma),
            "minimum_prior_fraction": float(minimum_prior_fraction),
        },
    )
    return OENestedSamplingResult(
        optimal_estimation=oe_result,
        nested_sampling=nested_result,
        refined_problem=refined_problem,
        prior_bounds=bounds,
        output_dir=root,
    )


def refine_priors_from_optimal_estimation(
    parameters: RetrievalParameterSet,
    result: OptimalEstimationResult,
    *,
    prior_sigma: float = 4.0,
    minimum_prior_fraction: float = 0.05,
) -> RetrievalParameterSet:
    """Return priors narrowed around an OE solution without exceeding original bounds."""

    sigma = float(prior_sigma)
    minimum = float(minimum_prior_fraction)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RobertValidationError("prior_sigma must be finite and positive")
    if not np.isfinite(minimum) or minimum <= 0.0 or minimum > 1.0:
        raise RobertValidationError("minimum_prior_fraction must be in (0, 1]")
    if result.parameter_names != parameters.names:
        raise RobertConfigError("OE result parameter order does not match the retrieval parameters")
    diagonal = np.diag(result.covariance)
    if diagonal.shape != (parameters.ndim,) or np.any(~np.isfinite(diagonal)) or np.any(diagonal < 0.0):
        raise RobertValidationError("OE covariance diagonal must be finite and non-negative")

    refined: list[RetrievalParameter] = []
    for index, parameter in enumerate(parameters.parameters):
        original_lower, original_upper = parameter.bounds
        original_width = original_upper - original_lower
        center = float(np.clip(result.state_vector[index], original_lower, original_upper))
        half_width = max(sigma * np.sqrt(diagonal[index]), 0.5 * minimum * original_width)
        lower = max(original_lower, center - half_width)
        upper = min(original_upper, center + half_width)
        required_width = minimum * original_width
        if upper - lower < required_width:
            lower = max(original_lower, min(center - 0.5 * required_width, original_upper - required_width))
            upper = lower + required_width
        prior_type = type(parameter.prior)
        if prior_type not in {UniformPrior, LogUniformPrior}:
            raise RobertConfigError(
                f"hybrid prior refinement does not support {prior_type.__name__}: {parameter.name}"
            )
        refined.append(replace(parameter, prior=prior_type(lower, upper)))
    return RetrievalParameterSet(tuple(refined))


def run_nested_sampling_then_oe(
    nested_problem: RetrievalProblem,
    *,
    output_dir: str | Path,
    oe_problem: RetrievalProblem | None = None,
    parameter_mapping: Mapping[str, str] | None = None,
    oe_state_overrides: Mapping[str, float] | None = None,
    require_nested_convergence: bool = True,
    covariance_floor_fraction: float = 1.0e-4,
    oe_prior_covariance: ArrayLike | None = None,
    nested_kwargs: Mapping[str, object] | None = None,
    oe_kwargs: Mapping[str, object] | None = None,
    seed: int | None = None,
) -> NestedSamplingOEResult:
    """Run nested sampling, then use its best fit and posterior covariance for OE."""

    root = Path(output_dir).expanduser()
    nested_result = run_retrieval(
        nested_problem,
        method="ultranest",
        output_dir=root / "nested_sampling",
        seed=seed,
        **_phase_kwargs(nested_kwargs, phase="nested sampling"),
    )
    oe_result, transferred = run_optimal_estimation_from_nested_result(
        nested_result,
        oe_problem=oe_problem or nested_problem,
        output_dir=root / "optimal_estimation",
        parameter_mapping=parameter_mapping,
        state_overrides=oe_state_overrides,
        require_convergence=require_nested_convergence,
        covariance_floor_fraction=covariance_floor_fraction,
        prior_covariance=oe_prior_covariance,
        oe_kwargs=oe_kwargs,
    )
    _write_handoff(
        root,
        {
            "workflow": "nested_sampling_then_optimal_estimation",
            "nested_converged": nested_result.converged,
            "nested_best_fit": dict(nested_result.best_fit_parameters),
            "oe_prior_state": transferred,
            "parameter_mapping": dict(parameter_mapping or {}),
        },
    )
    return NestedSamplingOEResult(
        nested_sampling=nested_result,
        optimal_estimation=oe_result,
        transferred_prior_state=transferred,
        output_dir=root,
    )


def run_optimal_estimation_from_nested_result(
    nested_result: RetrievalResult | NestedSamplerResult,
    *,
    oe_problem: RetrievalProblem,
    output_dir: str | Path,
    parameter_mapping: Mapping[str, str] | None = None,
    state_overrides: Mapping[str, float] | None = None,
    require_convergence: bool = True,
    covariance_floor_fraction: float = 1.0e-4,
    prior_covariance: ArrayLike | None = None,
    oe_kwargs: Mapping[str, object] | None = None,
) -> tuple[RetrievalResult, dict[str, float]]:
    """Refine an existing nested result with OE, including a higher-dimensional model.

    Parameters with the same name transfer automatically. ``parameter_mapping``
    maps OE parameter names to differently named nested parameters. New sounding
    or cloud parameters start at their prior midpoint unless supplied through
    ``state_overrides``.
    """

    nested = _nested_inference_result(nested_result)
    if require_convergence and not nested.converged:
        raise RobertConfigError(
            f"nested-sampling result has not converged: {nested.message}; OE was not started"
        )
    state, covariance, transferred = nested_posterior_oe_prior(
        nested,
        oe_problem,
        parameter_mapping=parameter_mapping,
        state_overrides=state_overrides,
        covariance_floor_fraction=covariance_floor_fraction,
    )
    settings = _phase_kwargs(oe_kwargs, phase="optimal estimation")
    overlap = {"initial_state", "prior_state", "prior_covariance"}.intersection(settings)
    if overlap:
        raise RobertConfigError(
            "oe_kwargs cannot replace transferred nested state: " + ", ".join(sorted(overlap))
        )
    if prior_covariance is not None:
        supplied_covariance = np.asarray(prior_covariance, dtype=float)
        if supplied_covariance.shape != covariance.shape:
            raise RobertValidationError(
                f"prior_covariance must have shape {covariance.shape}"
            )
        covariance = supplied_covariance
    result = run_retrieval(
        oe_problem,
        method="optimal_estimation",
        output_dir=output_dir,
        initial_state=state,
        prior_state=state,
        prior_covariance=covariance,
        **settings,
    )
    return result, transferred


def nested_posterior_oe_prior(
    nested: NestedSamplerResult,
    oe_problem: RetrievalProblem,
    *,
    parameter_mapping: Mapping[str, str] | None = None,
    state_overrides: Mapping[str, float] | None = None,
    covariance_floor_fraction: float = 1.0e-4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """Map a nested best fit and weighted posterior covariance into an OE state."""

    floor = float(covariance_floor_fraction)
    if not np.isfinite(floor) or floor <= 0.0:
        raise RobertValidationError("covariance_floor_fraction must be finite and positive")
    mapping = dict(parameter_mapping or {})
    overrides = {str(name): float(value) for name, value in (state_overrides or {}).items()}
    if not all(np.isfinite(value) for value in overrides.values()):
        raise RobertValidationError("OE state overrides must be finite")
    unknown = set(mapping).difference(oe_problem.parameter_names) | set(overrides).difference(oe_problem.parameter_names)
    if unknown:
        raise RobertConfigError("unknown OE state parameters: " + ", ".join(sorted(unknown)))
    source_index = {name: index for index, name in enumerate(nested.parameter_names)}
    target_sources: list[str | None] = []
    state = np.array(oe_problem.parameters.midpoint_vector(), copy=True)
    transferred: dict[str, float] = {}
    for index, target in enumerate(oe_problem.parameter_names):
        source = mapping.get(target, target if target in source_index else None)
        if source is not None and source not in source_index:
            raise RobertConfigError(f"nested result has no parameter '{source}' mapped to '{target}'")
        target_sources.append(source)
        if source is not None:
            state[index] = float(nested.best_fit_parameters[source])
        if target in overrides:
            state[index] = overrides[target]
        transferred[target] = float(state[index])

    bounds = np.asarray(oe_problem.parameters.bounds, dtype=float)
    state = np.clip(state, bounds[:, 0], bounds[:, 1])
    transferred = {
        name: float(value)
        for name, value in zip(oe_problem.parameter_names, state, strict=True)
    }
    covariance = np.diag(
        [parameter.approximate_standard_deviation**2 for parameter in oe_problem.parameters.parameters]
    )
    source_covariance = _weighted_covariance(nested)
    for left, left_source in enumerate(target_sources):
        if left_source is None:
            continue
        for right, right_source in enumerate(target_sources):
            if right_source is not None:
                covariance[left, right] = source_covariance[
                    source_index[left_source], source_index[right_source]
                ]
    scale_floor = np.array(
        [parameter.approximate_standard_deviation**2 * floor for parameter in oe_problem.parameters.parameters]
    )
    diagonal = np.maximum(np.diag(covariance), scale_floor)
    covariance[np.diag_indices_from(covariance)] = diagonal
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue <= 0.0:
        covariance += np.eye(oe_problem.ndim) * (abs(minimum_eigenvalue) + float(np.min(scale_floor)))
    state.setflags(write=False)
    covariance.setflags(write=False)
    return state, covariance, transferred


def load_nested_sampler_result(output_dir: str | Path) -> NestedSamplerResult:
    """Load a serialized ROBERT nested-sampling result for a later OE handoff."""

    root = Path(output_dir).expanduser()
    try:
        summary = json.loads((root / "result.json").read_text(encoding="utf-8"))
        with np.load(root / "result_arrays.npz", allow_pickle=False) as arrays:
            samples = np.array(arrays["samples"], copy=True)
            log_likelihood = np.array(arrays["log_likelihood"], copy=True)
            weights = np.array(arrays["weights"], copy=True) if "weights" in arrays.files else None
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RobertDataError(f"failed to load nested-sampling result from {root}") from exc
    if summary.get("method") not in {"ultranest", "nested_sampling"}:
        raise RobertDataError(f"result under {root} is not a nested-sampling result")
    return NestedSamplerResult(
        method=str(summary["method"]),
        parameter_names=tuple(str(name) for name in summary["parameter_names"]),
        samples=samples,
        log_likelihood=log_likelihood,
        weights=weights,
        log_evidence=summary.get("log_evidence"),
        log_evidence_error=summary.get("log_evidence_error"),
        best_fit_parameters=summary["best_fit_parameters"],
        metadata=summary.get("metadata", {}),
        converged=bool(summary.get("converged", False)),
        message=str(summary.get("message", "nested sampling did not report convergence")),
    )


def _weighted_covariance(result: NestedSamplerResult) -> NDArray[np.float64]:
    if result.samples.shape[0] < 2:
        raise RobertValidationError("at least two nested posterior samples are required for OE covariance")
    if result.weights is None:
        weights = np.full(result.samples.shape[0], 1.0 / result.samples.shape[0])
    else:
        total = float(np.sum(result.weights))
        if not np.isfinite(total) or total <= 0.0:
            raise RobertValidationError("nested posterior weights must have a positive finite sum")
        weights = result.weights / total
    mean = np.sum(result.samples * weights[:, np.newaxis], axis=0)
    delta = result.samples - mean
    correction = 1.0 - float(np.sum(np.square(weights)))
    denominator = correction if correction > np.finfo(float).eps else 1.0
    covariance = (delta * weights[:, np.newaxis]).T @ delta / denominator
    return 0.5 * (covariance + covariance.T)


def _nested_inference_result(result: RetrievalResult | NestedSamplerResult) -> NestedSamplerResult:
    inference = result.inference_result if isinstance(result, RetrievalResult) else result
    if not isinstance(inference, NestedSamplerResult):
        raise RobertConfigError("nested-to-OE handoff requires a nested-sampling result")
    return inference


def _phase_kwargs(settings: Mapping[str, object] | None, *, phase: str) -> dict[str, object]:
    values = dict(settings or {})
    reserved = {"method", "output_dir", "seed"}.intersection(values)
    if reserved:
        raise RobertConfigError(f"{phase} kwargs contain reserved settings: " + ", ".join(sorted(reserved)))
    return values


def _bounds_mapping(parameters: RetrievalParameterSet) -> dict[str, tuple[float, float]]:
    return dict(zip(parameters.names, parameters.bounds, strict=True))


def _write_handoff(output_dir: Path, payload: Mapping[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / HYBRID_HANDOFF_FILENAME
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    except OSError as exc:
        raise RobertDataError(f"failed to write hybrid handoff: {path}") from exc
    return path


__all__ = [
    "HYBRID_HANDOFF_FILENAME",
    "NestedSamplingOEResult",
    "OENestedSamplingResult",
    "load_nested_sampler_result",
    "nested_posterior_oe_prior",
    "refine_priors_from_optimal_estimation",
    "run_nested_sampling_then_oe",
    "run_oe_then_nested_sampling",
    "run_optimal_estimation_from_nested_result",
]
