"""Coupled hemispheric-mean thermal two-stream radiative transfer.

The implementation follows the two-stream moment equations and source-function
reconstruction described by Toon et al. (1989). Optical depth increases from
the top of the atmosphere downward. The Planck function varies linearly in
optical depth within each layer, following the standard source-function method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

from robert_exoplanets.core import RobertValidationError


@dataclass(frozen=True)
class ThermalTwoStreamResult:
    """Flux moments and reconstructed outgoing intensities."""

    upward_flux_levels: NDArray[np.float64]
    downward_flux_levels: NDArray[np.float64]
    point_radiance: NDArray[np.float64]
    point_layer_contribution_radiance: NDArray[np.float64]
    point_bottom_contribution_radiance: NDArray[np.float64]


def solve_thermal_two_stream(
    extinction_tau: ArrayLike,
    single_scattering_albedo: ArrayLike,
    asymmetry_factor: ArrayLike,
    level_planck_radiance: ArrayLike,
    emission_angle_cosines: ArrayLike,
    *,
    bottom_planck_radiance: ArrayLike,
    source_quadrature_order: int = 4,
) -> ThermalTwoStreamResult:
    """Solve coupled thermal two-stream fluxes and reconstruct outgoing rays.

    Parameters have shapes ``(layer, spectral, g)`` except the level Planck
    radiance ``(level, spectral)``, emission cosines ``(angle,)``, and bottom
    Planck radiance ``(spectral,)``. Returned fluxes retain the g ordinate;
    returned radiances also retain it for caller-side correlated-k weighting.
    """

    tau = _finite_array(extinction_tau, "extinction_tau", ndim=3)
    omega = _finite_array(single_scattering_albedo, "single_scattering_albedo", shape=tau.shape)
    asymmetry = _finite_array(asymmetry_factor, "asymmetry_factor", shape=tau.shape)
    planck = _finite_array(
        level_planck_radiance,
        "level_planck_radiance",
        shape=(tau.shape[0] + 1, tau.shape[1]),
    )
    bottom_planck = _finite_array(
        bottom_planck_radiance,
        "bottom_planck_radiance",
        shape=(tau.shape[1],),
    )
    mu = _finite_array(emission_angle_cosines, "emission_angle_cosines", ndim=1)
    if np.any(tau < 0.0):
        raise RobertValidationError("extinction_tau must be non-negative")
    if np.any((omega < 0.0) | (omega > 1.0)):
        raise RobertValidationError("single_scattering_albedo must be in [0, 1]")
    if np.any((asymmetry < -1.0) | (asymmetry > 1.0)):
        raise RobertValidationError("asymmetry_factor must be in [-1, 1]")
    if np.any(planck < 0.0) or np.any(bottom_planck < 0.0):
        raise RobertValidationError("Planck radiances must be non-negative")
    if np.any((mu <= 0.0) | (mu > 1.0)):
        raise RobertValidationError("emission_angle_cosines must be in (0, 1]")
    if source_quadrature_order < 2:
        raise RobertValidationError("source_quadrature_order must be at least 2")

    nlayer, nspectral, ng = tau.shape
    upward = np.empty((nlayer + 1, nspectral, ng), dtype=float)
    downward = np.empty_like(upward)
    point_layer = np.zeros((mu.size, nlayer, nspectral, ng), dtype=float)
    point_bottom = np.zeros((mu.size, nspectral, ng), dtype=float)

    nodes, weights = np.polynomial.legendre.leggauss(source_quadrature_order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights

    flux_levels = _solve_flux_columns(tau, omega, asymmetry, planck, bottom_planck)
    upward[:] = flux_levels[..., 0]
    downward[:] = flux_levels[..., 1]

    point_layer, point_bottom = _reconstruct_outgoing_sources(
        tau,
        omega,
        asymmetry,
        planck,
        bottom_planck,
        mu,
        upward,
        downward,
        unit_nodes,
        unit_weights,
    )

    point_radiance = np.sum(point_layer, axis=1) + point_bottom
    for array in (upward, downward, point_radiance, point_layer, point_bottom):
        array.setflags(write=False)
    return ThermalTwoStreamResult(
        upward_flux_levels=upward,
        downward_flux_levels=downward,
        point_radiance=point_radiance,
        point_layer_contribution_radiance=point_layer,
        point_bottom_contribution_radiance=point_bottom,
    )


def _solve_flux_columns(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    planck_levels: NDArray[np.float64],
    bottom_planck: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve every spectral/g column with a batched block Thomas algorithm."""

    nlayer, nspectral, ng = tau.shape
    nlevel = nlayer + 1
    shape = (nlevel, nspectral, ng, 2, 2)
    lower = np.zeros(shape, dtype=float)
    diagonal = np.zeros(shape, dtype=float)
    upper = np.zeros(shape, dtype=float)
    rhs = np.zeros((nlevel, nspectral, ng, 2), dtype=float)

    gamma1 = 2.0 - omega * (1.0 + asymmetry)
    gamma2 = omega * (1.0 - asymmetry)
    eigenvalue = np.sqrt(np.maximum(gamma1 * gamma1 - gamma2 * gamma2, 0.0))
    argument = eigenvalue * tau
    inverse_scale = np.where(
        np.abs(argument) < 20.0,
        1.0 / np.cosh(argument),
        2.0 * np.exp(-np.abs(argument)) / (1.0 + np.exp(-2.0 * np.abs(argument))),
    )
    factor = np.divide(
        np.tanh(argument),
        eigenvalue,
        out=tau.copy(),
        where=eigenvalue >= 1.0e-12,
    )
    transfer = np.empty((nlayer, nspectral, ng, 2, 2), dtype=float)
    transfer[..., 0, 0] = 1.0 + factor * gamma1
    transfer[..., 0, 1] = -factor * gamma2
    transfer[..., 1, 0] = factor * gamma2
    transfer[..., 1, 1] = 1.0 - factor * gamma1

    slope = np.divide(
        planck_levels[1:, :, None] - planck_levels[:-1, :, None],
        tau,
        out=np.zeros_like(tau),
        where=tau > 0.0,
    )
    correction = np.divide(
        np.pi * slope,
        gamma1 + gamma2,
        out=np.zeros_like(tau),
        where=(1.0 - omega) >= 1.0e-14,
    )
    thermal_scale = ((1.0 - omega) >= 1.0e-14).astype(float)
    particular_top = np.empty((nlayer, nspectral, ng, 2), dtype=float)
    particular_bottom = np.empty_like(particular_top)
    particular_top[..., 0] = thermal_scale * np.pi * planck_levels[:-1, :, None] + correction
    particular_top[..., 1] = thermal_scale * np.pi * planck_levels[:-1, :, None] - correction
    particular_bottom[..., 0] = thermal_scale * np.pi * planck_levels[1:, :, None] + correction
    particular_bottom[..., 1] = thermal_scale * np.pi * planck_levels[1:, :, None] - correction
    layer_rhs = inverse_scale[..., None] * particular_bottom - np.einsum(
        "...ij,...j->...i",
        transfer,
        particular_top,
    )

    diagonal[0, ..., 0, 1] = 1.0
    diagonal[0, ..., 1, :] = -transfer[0, ..., 0, :]
    upper[0, ..., 1, 0] = inverse_scale[0]
    rhs[0, ..., 1] = layer_rhs[0, ..., 0]
    for level_index in range(1, nlayer):
        lower[level_index, ..., 0, :] = -transfer[level_index - 1, ..., 1, :]
        diagonal[level_index, ..., 0, 1] = inverse_scale[level_index - 1]
        rhs[level_index, ..., 0] = layer_rhs[level_index - 1, ..., 1]
        diagonal[level_index, ..., 1, :] = -transfer[level_index, ..., 0, :]
        upper[level_index, ..., 1, 0] = inverse_scale[level_index]
        rhs[level_index, ..., 1] = layer_rhs[level_index, ..., 0]
    lower[-1, ..., 0, :] = -transfer[-1, ..., 1, :]
    diagonal[-1, ..., 0, 1] = inverse_scale[-1]
    rhs[-1, ..., 0] = layer_rhs[-1, ..., 1]
    diagonal[-1, ..., 1, 0] = 1.0
    rhs[-1, ..., 1] = np.pi * bottom_planck[:, None]

    modified_upper = np.zeros_like(upper)
    modified_rhs = np.zeros_like(rhs)
    inverse = _inverse_2x2(diagonal[0])
    modified_upper[0] = np.einsum("...ij,...jk->...ik", inverse, upper[0])
    modified_rhs[0] = np.einsum("...ij,...j->...i", inverse, rhs[0])
    for level_index in range(1, nlevel):
        reduced_diagonal = diagonal[level_index] - np.einsum(
            "...ij,...jk->...ik",
            lower[level_index],
            modified_upper[level_index - 1],
        )
        reduced_rhs = rhs[level_index] - np.einsum(
            "...ij,...j->...i",
            lower[level_index],
            modified_rhs[level_index - 1],
        )
        inverse = _inverse_2x2(reduced_diagonal)
        modified_upper[level_index] = np.einsum(
            "...ij,...jk->...ik",
            inverse,
            upper[level_index],
        )
        modified_rhs[level_index] = np.einsum("...ij,...j->...i", inverse, reduced_rhs)

    solution = np.empty_like(rhs)
    solution[-1] = modified_rhs[-1]
    for level_index in range(nlevel - 2, -1, -1):
        solution[level_index] = modified_rhs[level_index] - np.einsum(
            "...ij,...j->...i",
            modified_upper[level_index],
            solution[level_index + 1],
        )
    if not np.all(np.isfinite(solution)):
        raise RobertValidationError("thermal two-stream flux solve produced non-finite values")
    return solution


def _inverse_2x2(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    determinant = matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    if np.any(determinant == 0.0) or not np.all(np.isfinite(determinant)):
        raise RobertValidationError("thermal two-stream boundary system is singular")
    inverse = np.empty_like(matrix)
    inverse[..., 0, 0] = matrix[..., 1, 1] / determinant
    inverse[..., 0, 1] = -matrix[..., 0, 1] / determinant
    inverse[..., 1, 0] = -matrix[..., 1, 0] / determinant
    inverse[..., 1, 1] = matrix[..., 0, 0] / determinant
    return inverse


def _reconstruct_outgoing_sources(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    planck_levels: NDArray[np.float64],
    bottom_planck: NDArray[np.float64],
    mu: NDArray[np.float64],
    upward: NDArray[np.float64],
    downward: NDArray[np.float64],
    unit_nodes: NDArray[np.float64],
    unit_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nlayer, nspectral, ng = tau.shape
    point_layer = np.zeros((mu.size, nlayer, nspectral, ng), dtype=float)
    point_bottom = np.zeros((mu.size, nspectral, ng), dtype=float)
    pi_planck_levels = np.pi * planck_levels

    for angle_index, mu_value in enumerate(mu):
        bottom_transmission = np.ones((nspectral, ng), dtype=float)
        for layer_index in range(nlayer - 1, -1, -1):
            layer_tau = tau[layer_index]
            layer_omega = omega[layer_index]
            layer_asymmetry = asymmetry[layer_index]
            transmission = np.exp(-layer_tau / mu_value)
            gamma1 = 2.0 - layer_omega * (1.0 + layer_asymmetry)
            gamma2 = layer_omega * (1.0 - layer_asymmetry)
            eigenvalue = np.sqrt(np.maximum(gamma1 * gamma1 - gamma2 * gamma2, 0.0))
            planck_top = planck_levels[layer_index, :, None]
            planck_bottom = planck_levels[layer_index + 1, :, None]
            slope = np.divide(
                planck_bottom - planck_top,
                layer_tau,
                out=np.zeros_like(layer_tau),
                where=layer_tau > 0.0,
            )
            correction = np.divide(
                np.pi * slope,
                gamma1 + gamma2,
                out=np.zeros_like(layer_tau),
                where=(1.0 - layer_omega) >= 1.0e-14,
            )
            thermal_scale = ((1.0 - layer_omega) >= 1.0e-14).astype(float)
            particular_top_up = thermal_scale * pi_planck_levels[layer_index, :, None] + correction
            particular_top_down = thermal_scale * pi_planck_levels[layer_index, :, None] - correction
            delta_up = upward[layer_index] - particular_top_up
            delta_down = downward[layer_index] - particular_top_down
            contribution = np.zeros((nspectral, ng), dtype=float)

            for node, weight in zip(unit_nodes, unit_weights, strict=True):
                optical_depth = layer_tau * float(node)
                argument = eigenvalue * optical_depth
                cosine = np.cosh(argument)
                sine_factor = np.divide(
                    np.sinh(argument),
                    eigenvalue,
                    out=optical_depth.copy(),
                    where=eigenvalue >= 1.0e-12,
                )
                flux_up = (
                    thermal_scale
                    * np.pi
                    * (planck_top + (planck_bottom - planck_top) * float(node))
                    + correction
                    + (cosine + sine_factor * gamma1) * delta_up
                    - sine_factor * gamma2 * delta_down
                )
                flux_down = (
                    thermal_scale
                    * np.pi
                    * (planck_top + (planck_bottom - planck_top) * float(node))
                    - correction
                    + sine_factor * gamma2 * delta_up
                    + (cosine - sine_factor * gamma1) * delta_down
                )
                mean_intensity = (flux_up + flux_down) / (2.0 * np.pi)
                first_moment = (flux_up - flux_down) / (4.0 * np.pi)
                planck = planck_top + (planck_bottom - planck_top) * float(node)
                source = (1.0 - layer_omega) * planck + layer_omega * (
                    mean_intensity + 3.0 * layer_asymmetry * mu_value * first_moment
                )
                contribution += float(weight) * source * np.exp(-optical_depth / mu_value)

            contribution *= np.divide(
                layer_tau,
                mu_value,
                out=np.zeros_like(layer_tau),
                where=layer_tau > 0.0,
            )
            point_layer[angle_index, layer_index + 1 :] *= transmission[None, :, :]
            point_layer[angle_index, layer_index] = contribution
            bottom_transmission *= transmission
        point_bottom[angle_index] = bottom_planck[:, None] * bottom_transmission
    return point_layer, point_bottom


def _solve_flux_column(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    planck_levels: NDArray[np.float64],
    bottom_planck: float,
) -> NDArray[np.float64]:
    nlayer = tau.size
    nvariable = 2 * (nlayer + 1)
    lower_bandwidth = upper_bandwidth = 2
    banded = np.zeros((lower_bandwidth + upper_bandwidth + 1, nvariable), dtype=float)
    right_hand_side = np.zeros(nvariable, dtype=float)

    _set_banded(banded, 0, 1, 1.0, upper_bandwidth)
    for layer_index in range(nlayer):
        matrix, inverse_scale = _scaled_hemispheric_transfer(
            float(tau[layer_index]),
            float(omega[layer_index]),
            float(asymmetry[layer_index]),
        )
        particular_top, particular_bottom = _linear_planck_particular_flux(
            float(tau[layer_index]),
            float(omega[layer_index]),
            float(asymmetry[layer_index]),
            float(planck_levels[layer_index]),
            float(planck_levels[layer_index + 1]),
        )
        for component in range(2):
            row = 1 + 2 * layer_index + component
            for top_component in range(2):
                _set_banded(
                    banded,
                    row,
                    2 * layer_index + top_component,
                    -matrix[component, top_component],
                    upper_bandwidth,
                )
            _set_banded(
                banded,
                row,
                2 * (layer_index + 1) + component,
                inverse_scale,
                upper_bandwidth,
            )
            right_hand_side[row] = (
                inverse_scale * particular_bottom[component]
                - matrix[component] @ particular_top
            )

    bottom_row = nvariable - 1
    _set_banded(banded, bottom_row, 2 * nlayer, 1.0, upper_bandwidth)
    right_hand_side[bottom_row] = np.pi * bottom_planck
    solution = solve_banded(
        (lower_bandwidth, upper_bandwidth),
        banded,
        right_hand_side,
        check_finite=False,
    )
    levels = solution.reshape(nlayer + 1, 2)
    if not np.all(np.isfinite(levels)):
        raise RobertValidationError("thermal two-stream flux solve produced non-finite values")
    return levels


def _scaled_hemispheric_transfer(
    tau: float,
    omega: float,
    asymmetry: float,
) -> tuple[NDArray[np.float64], float]:
    """Return transfer/cosh(lambda*tau) and its inverse scale.

    Scaling the layer equations before assembling the banded boundary-value
    problem avoids exponentially growing transfer coefficients without
    changing the equations or capping optical depth.
    """

    gamma1 = 2.0 - omega * (1.0 + asymmetry)
    gamma2 = omega * (1.0 - asymmetry)
    eigenvalue = np.sqrt(max(gamma1 * gamma1 - gamma2 * gamma2, 0.0))
    identity = np.eye(2, dtype=float)
    coefficient = np.array([[gamma1, -gamma2], [gamma2, -gamma1]], dtype=float)
    if eigenvalue < 1.0e-12:
        return identity + tau * coefficient, 1.0
    argument = eigenvalue * tau
    inverse_scale = _sech(argument)
    return identity + (np.tanh(argument) / eigenvalue) * coefficient, inverse_scale


def _layer_source_integral(
    flux_top: NDArray[np.float64],
    layer_tau: float,
    omega: float,
    asymmetry: float,
    planck_top: float,
    planck_bottom: float,
    mu: float,
    unit_nodes: NDArray[np.float64],
    unit_weights: NDArray[np.float64],
) -> float:
    if layer_tau == 0.0:
        return 0.0
    total = 0.0
    particular_top, _ = _linear_planck_particular_flux(
        layer_tau,
        omega,
        asymmetry,
        planck_top,
        planck_bottom,
    )
    for node, weight in zip(unit_nodes, unit_weights, strict=True):
        optical_depth = layer_tau * float(node)
        partial_matrix = _hemispheric_transfer_matrix(
            optical_depth,
            omega,
            asymmetry,
        )
        planck = planck_top + (planck_bottom - planck_top) * float(node)
        particular, _ = _linear_planck_particular_flux(
            layer_tau,
            omega,
            asymmetry,
            planck_top,
            planck_bottom,
            optical_depth=optical_depth,
        )
        flux = particular + partial_matrix @ (flux_top - particular_top)
        mean_intensity = (flux[0] + flux[1]) / (2.0 * np.pi)
        first_moment = (flux[0] - flux[1]) / (4.0 * np.pi)
        source = (1.0 - omega) * planck + omega * (
            mean_intensity + 3.0 * asymmetry * mu * first_moment
        )
        total += float(weight) * source * np.exp(-optical_depth / mu)
    return layer_tau * total / mu


def _hemispheric_transfer_matrix(tau: float, omega: float, asymmetry: float) -> NDArray[np.float64]:
    scaled, inverse_scale = _scaled_hemispheric_transfer(tau, omega, asymmetry)
    if inverse_scale == 0.0:
        raise RobertValidationError(
            "angular source reconstruction requires vertical refinement for this layer optical depth"
        )
    return scaled / inverse_scale


def _linear_planck_particular_flux(
    layer_tau: float,
    omega: float,
    asymmetry: float,
    planck_top: float,
    planck_bottom: float,
    *,
    optical_depth: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the affine thermal-source particular flux solution."""

    if layer_tau == 0.0:
        value = np.full(2, np.pi * planck_top)
        return value, value
    depth = 0.0 if optical_depth is None else float(optical_depth)
    slope = (planck_bottom - planck_top) / layer_tau
    if 1.0 - omega < 1.0e-14:
        correction = np.zeros(2, dtype=float)
        thermal_scale = 0.0
    else:
        gamma1 = 2.0 - omega * (1.0 + asymmetry)
        gamma2 = omega * (1.0 - asymmetry)
        coefficient = np.array([[gamma1, -gamma2], [gamma2, -gamma1]], dtype=float)
        correction = np.linalg.solve(coefficient, np.full(2, np.pi * slope))
        thermal_scale = 1.0

    def particular(at_depth: float) -> NDArray[np.float64]:
        planck = planck_top + slope * at_depth
        return thermal_scale * np.full(2, np.pi * planck) + correction

    return particular(depth), particular(layer_tau)


def _sech(value: float) -> float:
    absolute = abs(value)
    if absolute < 20.0:
        return float(1.0 / np.cosh(value))
    exponential = np.exp(-absolute)
    return float(2.0 * exponential / (1.0 + exponential * exponential))


def _set_banded(
    banded: NDArray[np.float64],
    row: int,
    column: int,
    value: float,
    upper_bandwidth: int,
) -> None:
    banded[upper_bandwidth + row - column, column] = value


def _finite_array(
    values: ArrayLike,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if shape is not None and array.shape != shape:
        raise RobertValidationError(f"{name} must have shape {shape}")
    if ndim is not None and array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    return array
