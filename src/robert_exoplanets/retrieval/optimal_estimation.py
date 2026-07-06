"""Small optimal-estimation solver for retrieval smoke tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError

from .problem import RetrievalProblem


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
        object.__setattr__(self, "state_vector", _readonly_array(self.state_vector, "state_vector", 1))
        object.__setattr__(self, "covariance", _readonly_array(self.covariance, "covariance", 2))
        object.__setattr__(self, "averaging_kernel", _readonly_array(self.averaging_kernel, "averaging_kernel", 2))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def best_fit_parameters(self) -> dict[str, float]:
        """Return the retrieved state as a parameter mapping."""

        return {name: float(value) for name, value in zip(self.parameter_names, self.state_vector, strict=True)}


def run_optimal_estimation(
    problem: RetrievalProblem,
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
    """

    n_parameters = problem.ndim
    x_a = (
        np.array(prior_state, dtype=float, copy=True)
        if prior_state is not None
        else problem.parameters.midpoint_vector()
    )
    x = np.array(initial_state, dtype=float, copy=True) if initial_state is not None else np.array(x_a, copy=True)
    if x.shape != (n_parameters,) or x_a.shape != (n_parameters,):
        raise RobertValidationError("initial_state and prior_state must match the retrieval dimension")

    bounds = np.array(problem.parameters.bounds, dtype=float)
    x = _clip_to_bounds(x, bounds)
    x_a = _clip_to_bounds(x_a, bounds)
    s_a = _prior_covariance(problem, prior_covariance)
    s_a_inv = np.linalg.pinv(s_a)
    y = np.array(problem.observation.flux, dtype=float, copy=True)
    s_e_inv = np.diag(1.0 / np.square(problem.observation.uncertainty))

    converged = False
    message = "maximum iterations reached"
    previous_cost = float("inf")
    n_iterations = 0
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        n_iterations = iteration
        model = problem.model_values_from_vector(x)
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
        x_next = x_a + np.linalg.pinv(gain_left) @ rhs
        x_next = _clip_to_bounds(x_next, bounds)
        cost = _cost(problem, x_next, x_a, s_a_inv)
        step_norm = float(np.linalg.norm(x_next - x) / max(1.0, np.linalg.norm(x)))
        cost_change = abs(previous_cost - cost) / max(1.0, abs(previous_cost)) if np.isfinite(previous_cost) else np.inf
        x = x_next
        previous_cost = cost
        if step_norm < convergence_tolerance or cost_change < convergence_tolerance:
            converged = True
            message = "converged"
            break

    final_model = problem.model_values_from_vector(x)
    final_jacobian = _finite_difference_jacobian(problem, x, bounds, fraction=finite_difference_fraction)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        posterior_precision = final_jacobian.T @ s_e_inv @ final_jacobian + s_a_inv
    if np.all(np.isfinite(posterior_precision)):
        covariance = np.linalg.pinv(posterior_precision)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            averaging_kernel = covariance @ final_jacobian.T @ s_e_inv @ final_jacobian
    else:
        covariance = np.eye(n_parameters, dtype=float) * 1.0e300
        averaging_kernel = np.zeros((n_parameters, n_parameters), dtype=float)
    residual = y - final_model
    prior_delta = x - x_a
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        final_cost = float(residual.T @ s_e_inv @ residual + prior_delta.T @ s_a_inv @ prior_delta)
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
    problem: RetrievalProblem,
    state: NDArray[np.float64],
    bounds: NDArray[np.float64],
    *,
    fraction: float,
) -> NDArray[np.float64]:
    base = problem.model_values_from_vector(state)
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
        jacobian[:, index] = (problem.model_values_from_vector(high) - problem.model_values_from_vector(low)) / actual_step
    return jacobian


def _prior_covariance(problem: RetrievalProblem, prior_covariance: ArrayLike | None) -> NDArray[np.float64]:
    if prior_covariance is not None:
        covariance = np.array(prior_covariance, dtype=float, copy=True)
    else:
        widths = np.array([upper - lower for lower, upper in problem.parameters.bounds], dtype=float)
        covariance = np.diag(np.square(widths / 2.0))
    if covariance.shape != (problem.ndim, problem.ndim):
        raise RobertValidationError("prior_covariance must have shape (ndim, ndim)")
    if not np.all(np.isfinite(covariance)):
        raise RobertValidationError("prior_covariance must be finite")
    return covariance


def _cost(
    problem: RetrievalProblem,
    state: NDArray[np.float64],
    prior_state: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
) -> float:
    residual = problem.observation.flux - problem.model_values_from_vector(state)
    measurement = np.sum(np.square(residual / problem.observation.uncertainty))
    prior_delta = state - prior_state
    prior = prior_delta.T @ prior_precision @ prior_delta
    return float(measurement + prior)


def _clip_to_bounds(values: NDArray[np.float64], bounds: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(values, bounds[:, 0], bounds[:, 1])


def _readonly_array(values: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
