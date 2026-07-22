"""Small optimal-estimation solver for retrieval smoke tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

from .multi_dataset import MultiDatasetRetrievalProblem
from .problem import RetrievalProblem

RetrievalProblemLike = RetrievalProblem | MultiDatasetRetrievalProblem


def log_pressure_correlated_covariance(
    pressure: ArrayLike,
    *,
    standard_deviation: float,
    correlation_length_dex: float,
) -> NDArray[np.float64]:
    """Build an exponential T-P prior covariance in log10 pressure.

    The covariance between levels ``i`` and ``j`` is
    ``sigma**2 * exp(-abs(log10(P_i/P_j)) / L)``, where ``L`` is the
    e-folding correlation length in pressure decades. This is the maintained
    form of the smoothing prior previously used by the HAT-P-32b OE benchmark.
    """

    pressure_values = np.asarray(pressure, dtype=float)
    sigma = float(standard_deviation)
    length = float(correlation_length_dex)
    if pressure_values.ndim != 1 or pressure_values.size < 2:
        raise RobertValidationError("pressure must contain at least two levels")
    if not np.all(np.isfinite(pressure_values)) or np.any(pressure_values <= 0.0):
        raise RobertValidationError("pressure must be finite and positive")
    if len(np.unique(pressure_values)) != pressure_values.size:
        raise RobertValidationError("pressure levels must be unique")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RobertValidationError("standard_deviation must be finite and positive")
    if not np.isfinite(length) or length <= 0.0:
        raise RobertValidationError(
            "correlation_length_dex must be finite and positive"
        )
    log_pressure = np.log10(pressure_values)
    covariance = sigma**2 * np.exp(
        -np.abs(log_pressure[:, np.newaxis] - log_pressure[np.newaxis, :])
        / length
    )
    covariance.setflags(write=False)
    return covariance


@dataclass(frozen=True)
class OptimalEstimationResult:
    """Result from a finite-difference optimal-estimation retrieval."""

    parameter_names: tuple[str, ...]
    state_vector: NDArray[np.float64]
    covariance: NDArray[np.float64]
    averaging_kernel: NDArray[np.float64]
    cost: float
    log_likelihood: float
    n_iterations: int
    converged: bool
    message: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state_vector", _readonly_array(self.state_vector, "state_vector", 1)
        )
        object.__setattr__(
            self, "covariance", _readonly_array(self.covariance, "covariance", 2)
        )
        object.__setattr__(
            self,
            "averaging_kernel",
            _readonly_array(self.averaging_kernel, "averaging_kernel", 2),
        )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def best_fit_parameters(self) -> dict[str, float]:
        """Return the retrieved state as a parameter mapping."""

        return {
            name: float(value)
            for name, value in zip(self.parameter_names, self.state_vector, strict=True)
        }


def run_optimal_estimation(
    problem: RetrievalProblemLike,
    *,
    initial_state: ArrayLike | None = None,
    prior_state: ArrayLike | None = None,
    prior_covariance: ArrayLike | None = None,
    max_iterations: int = 8,
    convergence_tolerance: float = 1.0e-4,
    finite_difference_fraction: float = 1.0e-4,
    damping: float = 0.0,
) -> OptimalEstimationResult:
    """Run a Gauss-Newton optimal-estimation solve.

    This is intentionally compact and NumPy-only. It is suitable for early
    retrieval plumbing tests and diagnostic development, not yet a replacement
    for the full mature NEMESIS-style OE machinery.

    The sampler priors are approximated by Gaussian distributions centred on
    their medians, with scales inferred from their central 68% intervals, unless
    ``prior_state`` and ``prior_covariance`` are supplied explicitly.
    """

    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise RobertValidationError("max_iterations must be at least one")
    if not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise RobertValidationError("convergence_tolerance must be finite and positive")
    if not np.isfinite(finite_difference_fraction) or finite_difference_fraction <= 0.0:
        raise RobertValidationError(
            "finite_difference_fraction must be finite and positive"
        )
    if not np.isfinite(damping) or damping < 0.0:
        raise RobertValidationError("damping must be finite and non-negative")
    uncertainty_parameters = _state_dependent_uncertainty_parameters(problem)
    retrieved_uncertainty_parameters = sorted(
        uncertainty_parameters.intersection(problem.parameter_names)
    )
    if retrieved_uncertainty_parameters:
        raise RobertConfigError(
            "optimal estimation does not yet support retrieved jitter or uncertainty "
            "scale parameters: "
            + ", ".join(retrieved_uncertainty_parameters)
            + "; supply fixed uncertainties or use nested sampling"
        )

    n_parameters = problem.ndim
    x_a = (
        np.array(prior_state, dtype=float, copy=True)
        if prior_state is not None
        else problem.parameters.midpoint_vector()
    )
    x = (
        np.array(initial_state, dtype=float, copy=True)
        if initial_state is not None
        else np.array(x_a, copy=True)
    )
    if x.shape != (n_parameters,) or x_a.shape != (n_parameters,):
        raise RobertValidationError(
            "initial_state and prior_state must match the retrieval dimension"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(x_a)):
        raise RobertValidationError("initial_state and prior_state must be finite")

    bounds = np.array(problem.parameters.bounds, dtype=float)
    x = _clip_to_bounds(x, bounds)
    x_a = _clip_to_bounds(x_a, bounds)
    s_a = _prior_covariance(problem, prior_covariance)
    s_a_inv = np.linalg.pinv(s_a)
    _, y, uncertainty = problem.gaussian_inputs_from_vector(x)
    s_e_inv = np.diag(1.0 / np.square(uncertainty))

    converged = False
    message = "maximum iterations reached"
    previous_cost = _cost(problem, x, x_a, s_a_inv)
    n_iterations = 0
    for iteration in range(1, max_iterations + 1):
        n_iterations = iteration
        model, current_data, current_uncertainty = problem.gaussian_inputs_from_vector(
            x
        )
        if not np.array_equal(current_data, y) or not np.array_equal(
            current_uncertainty, uncertainty
        ):
            raise RobertConfigError(
                "optimal-estimation data and fixed uncertainties changed with state"
            )
        jacobian = _finite_difference_jacobian(
            problem,
            x,
            bounds,
            fraction=finite_difference_fraction,
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gain_left = jacobian.T @ s_e_inv @ jacobian + s_a_inv
            if damping > 0.0:
                gain_left = gain_left + float(damping) * np.eye(n_parameters)
            rhs = jacobian.T @ s_e_inv @ (y - model + jacobian @ (x - x_a))
        if not np.all(np.isfinite(gain_left)) or not np.all(np.isfinite(rhs)):
            message = "non-finite optimal-estimation linear system"
            break
        proposed = x_a + np.linalg.pinv(gain_left) @ rhs
        proposed = _clip_to_bounds(proposed, bounds)
        step = _backtracking_step(
            problem,
            current=x,
            proposed=proposed,
            prior_state=x_a,
            prior_precision=s_a_inv,
            current_cost=previous_cost,
        )
        if step is None:
            message = "no valid improving optimal-estimation step"
            break
        x_next, cost = step
        step_norm = float(np.linalg.norm(x_next - x) / max(1.0, np.linalg.norm(x)))
        cost_change = (
            abs(previous_cost - cost) / max(1.0, abs(previous_cost))
            if np.isfinite(previous_cost)
            else np.inf
        )
        x = x_next
        previous_cost = cost
        if step_norm < convergence_tolerance or cost_change < convergence_tolerance:
            converged = True
            message = "converged"
            break

    final_model, final_data, final_uncertainty = problem.gaussian_inputs_from_vector(x)
    if not np.array_equal(final_data, y) or not np.array_equal(
        final_uncertainty, uncertainty
    ):
        raise RobertConfigError(
            "optimal-estimation data and fixed uncertainties changed with state"
        )
    final_jacobian = _finite_difference_jacobian(
        problem, x, bounds, fraction=finite_difference_fraction
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        posterior_precision = final_jacobian.T @ s_e_inv @ final_jacobian + s_a_inv
    if np.all(np.isfinite(posterior_precision)):
        covariance = np.linalg.pinv(posterior_precision)
        covariance = 0.5 * (covariance + covariance.T)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            averaging_kernel = covariance @ final_jacobian.T @ s_e_inv @ final_jacobian
    else:
        covariance = np.eye(n_parameters, dtype=float) * 1.0e300
        averaging_kernel = np.zeros((n_parameters, n_parameters), dtype=float)
    residual = y - final_model
    prior_delta = x - x_a
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        final_cost = float(
            residual.T @ s_e_inv @ residual + prior_delta.T @ s_a_inv @ prior_delta
        )
    if not np.isfinite(final_cost):
        final_cost = float("inf")
    return OptimalEstimationResult(
        parameter_names=problem.parameter_names,
        state_vector=x,
        covariance=covariance,
        averaging_kernel=averaging_kernel,
        cost=final_cost,
        log_likelihood=problem.log_likelihood_from_vector(x),
        n_iterations=n_iterations,
        converged=converged,
        message=message,
        metadata={"method": "optimal_estimation", "problem": problem.name},
    )


def _finite_difference_jacobian(
    problem: RetrievalProblemLike,
    state: NDArray[np.float64],
    bounds: NDArray[np.float64],
    *,
    fraction: float,
) -> NDArray[np.float64]:
    base, data, uncertainty = problem.gaussian_inputs_from_vector(state)
    jacobian = np.zeros((base.size, state.size), dtype=float)
    for index in range(state.size):
        width = bounds[index, 1] - bounds[index, 0]
        step = max(abs(state[index]) * fraction, width * fraction, 1.0e-8)
        high = np.array(state, copy=True)
        low = np.array(state, copy=True)
        high[index] = min(bounds[index, 1], state[index] + step)
        low[index] = max(bounds[index, 0], state[index] - step)
        actual_step = high[index] - low[index]
        if actual_step <= 0.0:
            continue
        high_model, high_data, high_uncertainty = problem.gaussian_inputs_from_vector(
            high
        )
        low_model, low_data, low_uncertainty = problem.gaussian_inputs_from_vector(low)
        if not np.array_equal(high_data, data) or not np.array_equal(low_data, data):
            raise RobertConfigError(
                "optimal-estimation data changed during finite differencing"
            )
        if not np.array_equal(high_uncertainty, uncertainty) or not np.array_equal(
            low_uncertainty, uncertainty
        ):
            raise RobertConfigError(
                "optimal estimation does not support state-dependent uncertainty"
            )
        jacobian[:, index] = (high_model - low_model) / actual_step
    return jacobian


def _prior_covariance(
    problem: RetrievalProblemLike,
    prior_covariance: ArrayLike | None,
) -> NDArray[np.float64]:
    if prior_covariance is not None:
        covariance = np.array(prior_covariance, dtype=float, copy=True)
    else:
        scales = np.array(
            [
                parameter.approximate_standard_deviation
                for parameter in problem.parameters.parameters
            ],
            dtype=float,
        )
        covariance = np.diag(np.square(scales))
    if covariance.shape != (problem.ndim, problem.ndim):
        raise RobertValidationError("prior_covariance must have shape (ndim, ndim)")
    if not np.all(np.isfinite(covariance)):
        raise RobertValidationError("prior_covariance must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-15):
        raise RobertValidationError("prior_covariance must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise RobertValidationError("prior_covariance must be positive definite")
    return covariance


def _cost(
    problem: RetrievalProblemLike,
    state: NDArray[np.float64],
    prior_state: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
) -> float:
    model, data, uncertainty = problem.gaussian_inputs_from_vector(state)
    residual = data - model
    measurement = np.sum(np.square(residual / uncertainty))
    prior_delta = state - prior_state
    prior = prior_delta.T @ prior_precision @ prior_delta
    return float(measurement + prior)


def _backtracking_step(
    problem: RetrievalProblemLike,
    *,
    current: NDArray[np.float64],
    proposed: NDArray[np.float64],
    prior_state: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    current_cost: float,
    max_backtracks: int = 16,
) -> tuple[NDArray[np.float64], float] | None:
    """Find a finite, non-increasing step through the physical model domain."""

    direction = proposed - current
    for backtrack in range(max_backtracks + 1):
        factor = 0.5**backtrack
        candidate = current + factor * direction
        try:
            cost = _cost(problem, candidate, prior_state, prior_precision)
        except (RobertError, ValueError, FloatingPointError, OverflowError):
            continue
        if np.isfinite(cost) and cost <= current_cost:
            return candidate, float(cost)
    return None


def _state_dependent_uncertainty_parameters(
    problem: RetrievalProblemLike,
) -> set[str]:
    if isinstance(problem, MultiDatasetRetrievalProblem):
        return {
            parameter
            for dataset in problem.observations.datasets
            for parameter in (
                dataset.jitter_parameter,
                dataset.uncertainty_scale_parameter,
            )
            if parameter is not None
        }
    return {
        parameter
        for parameter in (
            problem.likelihood.jitter_parameter,
            problem.likelihood.uncertainty_scale_parameter,
        )
        if parameter is not None
    }


def _clip_to_bounds(
    values: NDArray[np.float64], bounds: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.clip(values, bounds[:, 0], bounds[:, 1])


def _readonly_array(values: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
