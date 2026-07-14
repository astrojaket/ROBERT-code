"""Finite-difference optimal estimation and atmospheric-sounding diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

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
    jacobian: NDArray[np.float64] | None = field(default=None, repr=False)
    gain_matrix: NDArray[np.float64] | None = field(default=None, repr=False)
    measurement_error_covariance: NDArray[np.float64] | None = field(default=None, repr=False)
    smoothing_error_covariance: NDArray[np.float64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_vector", _readonly_array(self.state_vector, "state_vector", 1))
        object.__setattr__(self, "covariance", _readonly_array(self.covariance, "covariance", 2))
        object.__setattr__(
            self,
            "averaging_kernel",
            _readonly_array(self.averaging_kernel, "averaging_kernel", 2),
        )
        for name in (
            "jacobian",
            "gain_matrix",
            "measurement_error_covariance",
            "smoothing_error_covariance",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly_array(value, name, 2))
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def best_fit_parameters(self) -> dict[str, float]:
        """Return the retrieved state as a parameter mapping."""

        return {name: float(value) for name, value in zip(self.parameter_names, self.state_vector, strict=True)}

    @property
    def degrees_of_freedom_for_signal(self) -> float:
        """Total independently resolved information, ``trace(A)``."""

        return float(np.trace(self.averaging_kernel))


def run_optimal_estimation(
    problem: RetrievalProblem,
    *,
    initial_state: ArrayLike | None = None,
    prior_state: ArrayLike | None = None,
    prior_covariance: ArrayLike | None = None,
    max_iterations: int = 8,
    convergence_tolerance: float = 1.0e-4,
    finite_difference_fraction: float = 1.0e-4,
    finite_difference_scheme: str = "central",
    damping: float = 0.0,
    marquardt_lambda: float = 1.0,
    marquardt_decrease: float = 0.3,
    marquardt_increase: float = 10.0,
    max_marquardt_trials: int = 12,
    forward_model_error: ArrayLike | float | None = None,
    measurement_covariance: ArrayLike | None = None,
) -> OptimalEstimationResult:
    """Run a Gauss-Newton optimal-estimation solve.

    This NumPy implementation supports full prior and measurement covariance,
    adaptive Marquardt damping, forward-model error, and the gain/averaging-
    kernel error diagnostics used for atmospheric sounding. It remains a
    compact reference path rather than a replacement for every mature NEMESIS
    forward-model and retrieval feature.

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
        raise RobertValidationError("finite_difference_fraction must be finite and positive")
    if finite_difference_scheme not in {"central", "forward"}:
        raise RobertValidationError("finite_difference_scheme must be 'central' or 'forward'")
    if not np.isfinite(damping) or damping < 0.0:
        raise RobertValidationError("damping must be finite and non-negative")
    marquardt_lambda = float(marquardt_lambda)
    marquardt_decrease = float(marquardt_decrease)
    marquardt_increase = float(marquardt_increase)
    max_marquardt_trials = int(max_marquardt_trials)
    if not np.isfinite(marquardt_lambda) or marquardt_lambda < 0.0:
        raise RobertValidationError("marquardt_lambda must be finite and non-negative")
    if not np.isfinite(marquardt_decrease) or not 0.0 < marquardt_decrease < 1.0:
        raise RobertValidationError("marquardt_decrease must be finite and in (0, 1)")
    if not np.isfinite(marquardt_increase) or marquardt_increase <= 1.0:
        raise RobertValidationError("marquardt_increase must be finite and greater than one")
    if max_marquardt_trials < 1:
        raise RobertValidationError("max_marquardt_trials must be at least one")
    jitter_parameter = problem.likelihood.jitter_parameter
    if jitter_parameter is not None and jitter_parameter in problem.parameter_names:
        raise RobertConfigError(
            "optimal estimation does not yet support a retrieved jitter parameter; "
            "supply fixed uncertainties or use nested sampling"
        )

    n_parameters = problem.ndim
    x_a = (
        np.array(prior_state, dtype=float, copy=True)
        if prior_state is not None
        else problem.parameters.midpoint_vector()
    )
    x = np.array(initial_state, dtype=float, copy=True) if initial_state is not None else np.array(x_a, copy=True)
    if x.shape != (n_parameters,) or x_a.shape != (n_parameters,):
        raise RobertValidationError("initial_state and prior_state must match the retrieval dimension")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(x_a)):
        raise RobertValidationError("initial_state and prior_state must be finite")

    bounds = np.array(problem.parameters.bounds, dtype=float)
    x = _clip_to_bounds(x, bounds)
    x_a = _clip_to_bounds(x_a, bounds)
    s_a = _prior_covariance(problem, prior_covariance)
    s_a_inv = np.linalg.pinv(s_a)
    _, y, uncertainty = problem.gaussian_inputs_from_vector(x)
    s_e = _measurement_covariance(
        uncertainty,
        measurement_covariance=measurement_covariance,
        forward_model_error=forward_model_error,
    )
    s_e_inv = np.linalg.pinv(s_e)

    converged = False
    message = "maximum iterations reached"
    previous_cost = _cost(problem, x, x_a, s_a_inv, measurement_precision=s_e_inv)
    n_iterations = 0
    current_lambda = marquardt_lambda
    for iteration in range(1, max_iterations + 1):
        n_iterations = iteration
        model, current_data, current_uncertainty = problem.gaussian_inputs_from_vector(x)
        if not np.array_equal(current_data, y) or not np.array_equal(current_uncertainty, uncertainty):
            raise RobertConfigError("optimal-estimation data and fixed uncertainties changed with state")
        jacobian = _finite_difference_jacobian(
            problem,
            x,
            bounds,
            fraction=finite_difference_fraction,
            scheme=finite_difference_scheme,
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
        step = _marquardt_step(
            problem,
            current=x,
            proposed=proposed,
            prior_state=x_a,
            prior_precision=s_a_inv,
            measurement_precision=s_e_inv,
            current_cost=previous_cost,
            marquardt_lambda=current_lambda,
            decrease=marquardt_decrease,
            increase=marquardt_increase,
            max_trials=max_marquardt_trials,
        )
        if step is None:
            message = "no valid improving optimal-estimation step"
            break
        x_next, cost, current_lambda = step
        step_norm = float(np.linalg.norm(x_next - x) / max(1.0, np.linalg.norm(x)))
        cost_change = abs(previous_cost - cost) / max(1.0, abs(previous_cost)) if np.isfinite(previous_cost) else np.inf
        x = x_next
        previous_cost = cost
        if step_norm < convergence_tolerance or cost_change < convergence_tolerance:
            converged = True
            message = "converged"
            break

    final_model, final_data, final_uncertainty = problem.gaussian_inputs_from_vector(x)
    if not np.array_equal(final_data, y) or not np.array_equal(final_uncertainty, uncertainty):
        raise RobertConfigError("optimal-estimation data and fixed uncertainties changed with state")
    final_jacobian = _finite_difference_jacobian(
        problem,
        x,
        bounds,
        fraction=finite_difference_fraction,
        scheme=finite_difference_scheme,
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        posterior_precision = final_jacobian.T @ s_e_inv @ final_jacobian + s_a_inv
    if np.all(np.isfinite(posterior_precision)):
        covariance = np.linalg.pinv(posterior_precision)
        covariance = 0.5 * (covariance + covariance.T)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gain_matrix = covariance @ final_jacobian.T @ s_e_inv
            averaging_kernel = gain_matrix @ final_jacobian
            measurement_error_covariance = gain_matrix @ s_e @ gain_matrix.T
            smoothing_operator = averaging_kernel - np.eye(n_parameters)
            smoothing_error_covariance = smoothing_operator @ s_a @ smoothing_operator.T
    else:
        covariance = np.eye(n_parameters, dtype=float) * 1.0e300
        averaging_kernel = np.zeros((n_parameters, n_parameters), dtype=float)
        gain_matrix = np.zeros((n_parameters, y.size), dtype=float)
        measurement_error_covariance = np.zeros((n_parameters, n_parameters), dtype=float)
        smoothing_error_covariance = covariance
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
        metadata={
            "method": "optimal_estimation",
            "problem": problem.name,
            "oe_scheme": "nemesis_style_marquardt",
            "final_marquardt_lambda": f"{current_lambda:.17g}",
            "forward_model_error_included": str(forward_model_error is not None).lower(),
            "finite_difference_scheme": finite_difference_scheme,
        },
        jacobian=final_jacobian,
        gain_matrix=gain_matrix,
        measurement_error_covariance=measurement_error_covariance,
        smoothing_error_covariance=smoothing_error_covariance,
    )


def _finite_difference_jacobian(
    problem: RetrievalProblem,
    state: NDArray[np.float64],
    bounds: NDArray[np.float64],
    *,
    fraction: float,
    scheme: str,
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
        if scheme == "forward" and high[index] > state[index]:
            high_model, high_data, high_uncertainty = problem.gaussian_inputs_from_vector(high)
            if not np.array_equal(high_data, data):
                raise RobertConfigError("optimal-estimation data changed during finite differencing")
            if not np.array_equal(high_uncertainty, uncertainty):
                raise RobertConfigError("optimal estimation does not support state-dependent uncertainty")
            jacobian[:, index] = (high_model - base) / (high[index] - state[index])
        elif scheme == "forward":
            low_model, low_data, low_uncertainty = problem.gaussian_inputs_from_vector(low)
            if not np.array_equal(low_data, data):
                raise RobertConfigError("optimal-estimation data changed during finite differencing")
            if not np.array_equal(low_uncertainty, uncertainty):
                raise RobertConfigError("optimal estimation does not support state-dependent uncertainty")
            jacobian[:, index] = (base - low_model) / (state[index] - low[index])
        else:
            high_model, high_data, high_uncertainty = problem.gaussian_inputs_from_vector(high)
            low_model, low_data, low_uncertainty = problem.gaussian_inputs_from_vector(low)
            if not np.array_equal(high_data, data) or not np.array_equal(low_data, data):
                raise RobertConfigError("optimal-estimation data changed during finite differencing")
            if not np.array_equal(high_uncertainty, uncertainty) or not np.array_equal(low_uncertainty, uncertainty):
                raise RobertConfigError("optimal estimation does not support state-dependent uncertainty")
            jacobian[:, index] = (high_model - low_model) / actual_step
    return jacobian


def _prior_covariance(problem: RetrievalProblem, prior_covariance: ArrayLike | None) -> NDArray[np.float64]:
    if prior_covariance is not None:
        covariance = np.array(prior_covariance, dtype=float, copy=True)
    else:
        scales = np.array(
            [parameter.approximate_standard_deviation for parameter in problem.parameters.parameters],
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
    problem: RetrievalProblem,
    state: NDArray[np.float64],
    prior_state: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    *,
    measurement_precision: NDArray[np.float64] | None = None,
) -> float:
    model, data, uncertainty = problem.gaussian_inputs_from_vector(state)
    residual = data - model
    measurement = (
        np.sum(np.square(residual / uncertainty))
        if measurement_precision is None
        else residual.T @ measurement_precision @ residual
    )
    prior_delta = state - prior_state
    prior = prior_delta.T @ prior_precision @ prior_delta
    return float(measurement + prior)


def _marquardt_step(
    problem: RetrievalProblem,
    *,
    current: NDArray[np.float64],
    proposed: NDArray[np.float64],
    prior_state: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    measurement_precision: NDArray[np.float64],
    current_cost: float,
    marquardt_lambda: float,
    decrease: float,
    increase: float,
    max_trials: int,
) -> tuple[NDArray[np.float64], float, float] | None:
    """Apply the adaptive Marquardt step used by NEMESIS-style OE."""

    direction = proposed - current
    trial_lambda = marquardt_lambda
    for _ in range(max_trials):
        factor = 1.0 / (1.0 + trial_lambda)
        candidate = current + factor * direction
        try:
            cost = _cost(
                problem,
                candidate,
                prior_state,
                prior_precision,
                measurement_precision=measurement_precision,
            )
        except (RobertError, ValueError, FloatingPointError, OverflowError):
            trial_lambda = max(np.finfo(float).eps, trial_lambda * increase)
            continue
        if np.isfinite(cost) and cost <= current_cost:
            next_lambda = trial_lambda * decrease
            return candidate, float(cost), float(next_lambda)
        trial_lambda = max(np.finfo(float).eps, trial_lambda * increase)
    return None


def _measurement_covariance(
    uncertainty: NDArray[np.float64],
    *,
    measurement_covariance: ArrayLike | None,
    forward_model_error: ArrayLike | float | None,
) -> NDArray[np.float64]:
    """Build fixed measurement covariance with optional model-error variance."""

    if measurement_covariance is None:
        covariance = np.diag(np.square(uncertainty))
    else:
        covariance = np.array(measurement_covariance, dtype=float, copy=True)
        if covariance.shape != (uncertainty.size, uncertainty.size):
            raise RobertValidationError("measurement_covariance must have shape (ndata, ndata)")
        if not np.all(np.isfinite(covariance)):
            raise RobertValidationError("measurement_covariance must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-15):
            raise RobertValidationError("measurement_covariance must be symmetric")
    if forward_model_error is not None:
        error = np.asarray(forward_model_error, dtype=float)
        if error.ndim == 0:
            error = np.full(uncertainty.size, float(error), dtype=float)
        if error.shape != uncertainty.shape or not np.all(np.isfinite(error)) or np.any(error < 0.0):
            raise RobertValidationError("forward_model_error must be finite, non-negative, and match the data")
        covariance[np.diag_indices_from(covariance)] += np.square(error)
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise RobertValidationError("measurement covariance must be positive definite")
    return covariance


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
