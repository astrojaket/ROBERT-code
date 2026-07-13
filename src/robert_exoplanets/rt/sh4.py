"""Four-term spherical-harmonics thermal radiative transfer.

This module implements the P3/SH4 multiple-scattering formulation of
Rooney, Batalha & Marley (2024).  Optical depth increases downward.  The
thermal source is linear in optical depth within each layer, the upper
boundary has no incident thermal radiation, and the lower boundary is a
black surface.  Emergent rays are obtained with the source-function method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

from robert_exoplanets.core import RobertValidationError

try:  # pragma: no cover - availability depends on the local environment.
    from numba import njit, prange
except Exception:  # pragma: no cover - dependency availability is environment-specific.
    njit = None
    prange = range

_NUMBA_AVAILABLE = njit is not None


_HALF_RANGE_MOMENTS = np.pi * np.array(
    [
        [1.0, -2.0, 5.0 / 4.0, 0.0],
        [-1.0 / 4.0, 0.0, 5.0 / 4.0, -2.0],
        [1.0, 2.0, 5.0 / 4.0, 0.0],
        [-1.0 / 4.0, 0.0, 5.0 / 4.0, 2.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ThermalSH4Result:
    """SH4 moments and source-function reconstructed outgoing intensities."""

    moment_levels: NDArray[np.float64]
    point_radiance: NDArray[np.float64]
    point_layer_contribution_radiance: NDArray[np.float64]
    point_bottom_contribution_radiance: NDArray[np.float64]
    phase_function_moments: NDArray[np.float64]
    scaled_extinction_tau: NDArray[np.float64]
    scaled_single_scattering_albedo: NDArray[np.float64]
    delta_m_applied: bool


@dataclass(frozen=True)
class ThermalSH4SpectrumResult:
    """Disk-integrated SH4 radiance without contribution diagnostics."""

    radiance: NDArray[np.float64]
    backend: str

    def __post_init__(self) -> None:
        radiance = _finite_array(self.radiance, "radiance", ndim=1)
        if np.any(radiance < 0.0):
            raise RobertValidationError("SH4 spectrum radiance must be non-negative")
        if self.backend not in {"numpy", "numba"}:
            raise RobertValidationError(
                "SH4 spectrum backend must be 'numpy' or 'numba'"
            )
        radiance.setflags(write=False)
        object.__setattr__(self, "radiance", radiance)


def sh4_spectrum_backend_name(value: str) -> str:
    """Normalize and validate a spectrum-only SH4 reconstruction backend."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        return "numba" if _NUMBA_AVAILABLE else "numpy"
    if normalized == "numpy":
        return "numpy"
    if normalized == "numba":
        if not _NUMBA_AVAILABLE:
            raise RobertValidationError(
                "SH4 spectrum backend 'numba' requires the optional numba package"
            )
        return "numba"
    raise RobertValidationError(
        "SH4 spectrum backend must be 'auto', 'numpy', or 'numba'"
    )


def henyey_greenstein_moments(
    asymmetry_factor: ArrayLike,
    *,
    order: int = 4,
) -> NDArray[np.float64]:
    """Return Legendre coefficients ``chi_l=(2l+1) g**l`` through ``order``."""

    asymmetry = _finite_array(asymmetry_factor, "asymmetry_factor")
    if order < 1:
        raise RobertValidationError("order must be positive")
    if np.any((asymmetry <= -1.0) | (asymmetry >= 1.0)):
        raise RobertValidationError(
            "asymmetry_factor must be strictly between -1 and 1"
        )
    degree = np.arange(order, dtype=float).reshape((order,) + (1,) * asymmetry.ndim)
    moments = (2.0 * degree + 1.0) * asymmetry[None, ...] ** degree
    return np.asarray(moments, dtype=float)


def solve_thermal_sh4(
    extinction_tau: ArrayLike,
    single_scattering_albedo: ArrayLike,
    asymmetry_factor: ArrayLike,
    level_planck_radiance: ArrayLike,
    emission_angle_cosines: ArrayLike,
    *,
    bottom_planck_radiance: ArrayLike,
    phase_function_moments: ArrayLike | None = None,
    delta_m_forward_fraction: ArrayLike | None = None,
    delta_m: bool = True,
    source_quadrature_order: int = 6,
) -> ThermalSH4Result:
    """Solve thermal multiple scattering with a four-term P3 expansion.

    The optical arrays have shape ``(layer, spectral, g)``.  Phase-function
    moments, when supplied, have shape ``(4, layer, spectral, g)`` and use the
    convention ``P(mu,mu') = sum chi_l P_l(mu) P_l(mu')``.  When omitted, a
    Henyey--Greenstein phase function is constructed explicitly from ``g``.
    """

    (
        tau,
        planck,
        bottom_planck,
        mu,
        scaled_tau,
        scaled_omega,
        scaled_moments,
    ) = _prepare_thermal_sh4_inputs(
        extinction_tau,
        single_scattering_albedo,
        asymmetry_factor,
        level_planck_radiance,
        emission_angle_cosines,
        bottom_planck_radiance=bottom_planck_radiance,
        phase_function_moments=phase_function_moments,
        delta_m_forward_fraction=delta_m_forward_fraction,
        delta_m=delta_m,
        source_quadrature_order=source_quadrature_order,
    )
    coefficients, eigen_top, eigen_bottom, particular_top, particular_bottom = (
        _layer_solution(
            scaled_tau,
            scaled_omega,
            scaled_moments,
            planck,
            bottom_planck,
        )
    )
    moment_levels = _moment_levels(
        coefficients,
        eigen_top,
        eigen_bottom,
        particular_top,
        particular_bottom,
    )
    moment_levels = moment_levels.reshape(
        tau.shape[0] + 1, tau.shape[1], tau.shape[2], 4
    )
    point_layer, point_bottom = _reconstruct_sources(
        coefficients,
        scaled_tau,
        scaled_omega,
        scaled_moments,
        planck,
        bottom_planck,
        mu,
        source_quadrature_order,
    )
    point_radiance = np.sum(point_layer, axis=1) + point_bottom
    if np.any(point_radiance < -1.0e-10 * np.maximum(np.max(point_radiance), 1.0)):
        raise RobertValidationError(
            "SH4 reconstructed a materially negative emergent intensity"
        )
    point_radiance = np.maximum(point_radiance, 0.0)
    for array in (
        moment_levels,
        point_radiance,
        point_layer,
        point_bottom,
        scaled_moments,
        scaled_tau,
        scaled_omega,
    ):
        array.setflags(write=False)
    return ThermalSH4Result(
        moment_levels=moment_levels,
        point_radiance=point_radiance,
        point_layer_contribution_radiance=point_layer,
        point_bottom_contribution_radiance=point_bottom,
        phase_function_moments=scaled_moments,
        scaled_extinction_tau=scaled_tau,
        scaled_single_scattering_albedo=scaled_omega,
        delta_m_applied=delta_m,
    )


def solve_thermal_sh4_spectrum(
    extinction_tau: ArrayLike,
    single_scattering_albedo: ArrayLike,
    asymmetry_factor: ArrayLike,
    level_planck_radiance: ArrayLike,
    emission_angle_cosines: ArrayLike,
    emission_angle_weights: ArrayLike,
    g_weights: ArrayLike,
    *,
    bottom_planck_radiance: ArrayLike,
    phase_function_moments: ArrayLike | None = None,
    delta_m_forward_fraction: ArrayLike | None = None,
    delta_m: bool = True,
    source_quadrature_order: int = 6,
    backend: str = "auto",
) -> ThermalSH4SpectrumResult:
    """Return only disk- and g-integrated SH4 radiance.

    The boundary-value system is identical to :func:`solve_thermal_sh4` and is
    solved by the SciPy reference path. The optional Numba backend compiles
    only source reconstruction and directly accumulates the final spectrum,
    avoiding moment-level and contribution-function output arrays.
    """

    (
        _tau,
        planck,
        bottom_planck,
        mu,
        scaled_tau,
        scaled_omega,
        scaled_moments,
    ) = _prepare_thermal_sh4_inputs(
        extinction_tau,
        single_scattering_albedo,
        asymmetry_factor,
        level_planck_radiance,
        emission_angle_cosines,
        bottom_planck_radiance=bottom_planck_radiance,
        phase_function_moments=phase_function_moments,
        delta_m_forward_fraction=delta_m_forward_fraction,
        delta_m=delta_m,
        source_quadrature_order=source_quadrature_order,
    )
    angle_weights = _normalized_weights(
        emission_angle_weights,
        mu.size,
        "emission_angle_weights",
    )
    quadrature_weights = _normalized_weights(
        g_weights,
        scaled_tau.shape[2],
        "g_weights",
    )
    coefficients, _, _, _, _ = _layer_solution(
        scaled_tau,
        scaled_omega,
        scaled_moments,
        planck,
        bottom_planck,
    )
    selected_backend = sh4_spectrum_backend_name(backend)
    nodes, node_weights = np.polynomial.legendre.leggauss(source_quadrature_order)
    nodes = 0.5 * (nodes + 1.0)
    node_weights = 0.5 * node_weights
    if selected_backend == "numba":
        radiance = _numba_reconstruct_spectrum(
            coefficients,
            scaled_tau,
            scaled_omega,
            scaled_moments,
            planck,
            bottom_planck,
            mu,
            angle_weights,
            quadrature_weights,
            nodes,
            node_weights,
        )
    else:
        point_layer, point_bottom = _reconstruct_sources(
            coefficients,
            scaled_tau,
            scaled_omega,
            scaled_moments,
            planck,
            bottom_planck,
            mu,
            source_quadrature_order,
        )
        point_radiance = np.sum(point_layer, axis=1) + point_bottom
        radiance = np.einsum(
            "a,asg,g->s",
            angle_weights,
            point_radiance,
            quadrature_weights,
        )
    if not np.all(np.isfinite(radiance)):
        raise RobertValidationError(
            "SH4 spectrum reconstruction produced non-finite values"
        )
    negative_limit = -1.0e-10 * max(float(np.max(radiance)), 1.0)
    if np.any(radiance < negative_limit):
        raise RobertValidationError(
            "SH4 spectrum reconstructed a materially negative emergent intensity"
        )
    return ThermalSH4SpectrumResult(
        radiance=np.maximum(radiance, 0.0),
        backend=selected_backend,
    )


def _prepare_thermal_sh4_inputs(
    extinction_tau: ArrayLike,
    single_scattering_albedo: ArrayLike,
    asymmetry_factor: ArrayLike,
    level_planck_radiance: ArrayLike,
    emission_angle_cosines: ArrayLike,
    *,
    bottom_planck_radiance: ArrayLike,
    phase_function_moments: ArrayLike | None,
    delta_m_forward_fraction: ArrayLike | None,
    delta_m: bool,
    source_quadrature_order: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate common SH4 inputs and apply the shared delta-M transform."""

    tau = _finite_array(extinction_tau, "extinction_tau", ndim=3)
    omega = _finite_array(
        single_scattering_albedo, "single_scattering_albedo", shape=tau.shape
    )
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
    if np.any((asymmetry <= -1.0) | (asymmetry >= 1.0)):
        raise RobertValidationError(
            "asymmetry_factor must be strictly between -1 and 1"
        )
    if np.any(planck < 0.0) or np.any(bottom_planck < 0.0):
        raise RobertValidationError("Planck radiances must be non-negative")
    if np.any((mu <= 0.0) | (mu > 1.0)):
        raise RobertValidationError("emission_angle_cosines must be in (0, 1]")
    if source_quadrature_order < 2:
        raise RobertValidationError("source_quadrature_order must be at least 2")

    if phase_function_moments is None:
        moments = henyey_greenstein_moments(asymmetry)
        moments_from_hg = True
        supplied_forward_fraction = None
    else:
        moments = _finite_array(
            phase_function_moments,
            "phase_function_moments",
            shape=(4,) + tau.shape,
        )
        moments_from_hg = False
        if not np.allclose(moments[0], 1.0, rtol=1.0e-12, atol=1.0e-12):
            raise RobertValidationError("phase_function_moments[0] must equal one")
        supplied_forward_fraction = None
        if delta_m_forward_fraction is not None:
            supplied_forward_fraction = _finite_array(
                delta_m_forward_fraction,
                "delta_m_forward_fraction",
                shape=tau.shape,
            )
            if np.any(
                (supplied_forward_fraction < 0.0) | (supplied_forward_fraction >= 1.0)
            ):
                raise RobertValidationError(
                    "delta_m_forward_fraction must lie in [0, 1)"
                )

    scaled_tau, scaled_omega, scaled_moments = _delta_m_scale(
        tau,
        omega,
        asymmetry,
        moments,
        enabled=delta_m,
        moments_from_hg=moments_from_hg,
        supplied_forward_fraction=supplied_forward_fraction,
    )
    return (
        tau,
        planck,
        bottom_planck,
        mu,
        scaled_tau,
        scaled_omega,
        scaled_moments,
    )


def _delta_m_scale(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    moments: NDArray[np.float64],
    *,
    enabled: bool,
    moments_from_hg: bool,
    supplied_forward_fraction: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not enabled:
        return tau.copy(), omega.copy(), moments.copy()
    if not moments_from_hg and supplied_forward_fraction is None:
        raise RobertValidationError(
            "delta_m with supplied phase moments requires an explicit omitted forward-peak moment"
        )
    forward_fraction = (
        asymmetry**4 if supplied_forward_fraction is None else supplied_forward_fraction
    )
    denominator = 1.0 - omega * forward_fraction
    phase_denominator = 1.0 - forward_fraction
    scaled_tau = denominator * tau
    scaled_omega = np.divide(
        omega * phase_denominator,
        denominator,
        out=np.zeros_like(omega),
        where=denominator > 0.0,
    )
    degree = np.arange(4, dtype=float).reshape((4,) + (1,) * tau.ndim)
    scaled_moments = np.divide(
        moments - (2.0 * degree + 1.0) * forward_fraction[None, ...],
        phase_denominator[None, ...],
    )
    return scaled_tau, scaled_omega, scaled_moments


def _layer_solution(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    moments: NDArray[np.float64],
    planck: NDArray[np.float64],
    bottom_planck: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    nlayer, nspectral, ng = tau.shape
    ncolumn = nspectral * ng
    tau_flat = tau.reshape(nlayer, ncolumn)
    omega_flat = omega.reshape(nlayer, ncolumn)
    moments_flat = moments.reshape(4, nlayer, ncolumn)
    a = (2.0 * np.arange(4) + 1.0)[:, None, None] - omega_flat[None, ...] * moments_flat

    beta = a[0] * a[1] + a[2] * a[3] / 9.0 + 4.0 * a[0] * a[3] / 9.0
    gamma = a[0] * a[1] * a[2] * a[3] / 9.0
    discriminant = np.maximum(beta * beta - 4.0 * gamma, 0.0)
    lambda1 = np.sqrt(0.5 * (beta + np.sqrt(discriminant)))
    lambda2 = np.sqrt(0.5 * (beta - np.sqrt(discriminant)))
    conservative = a[0] == 0.0

    eigen_top = np.empty((nlayer, ncolumn, 4, 4), dtype=float)
    eigen_bottom = np.empty_like(eigen_top)
    for pair, eigenvalue in enumerate((lambda1, lambda2)):
        safe_eigenvalue = np.where(conservative & (pair == 1), 1.0, eigenvalue)
        r_value = -a[0] / safe_eigenvalue
        q_value = 0.5 * (a[0] * a[1] / safe_eigenvalue**2 - 1.0)
        s_value = -1.5 / a[3] * (a[0] * a[1] / safe_eigenvalue - safe_eigenvalue)
        positive = np.stack((np.ones_like(r_value), r_value, q_value, s_value), axis=-1)
        negative = positive.copy()
        negative[..., 1] *= -1.0
        negative[..., 3] *= -1.0
        attenuation = np.exp(-np.minimum(eigenvalue * tau_flat, 700.0))
        first = 2 * pair
        eigen_top[..., first] = positive
        eigen_bottom[..., first] = positive * attenuation[..., None]
        eigen_top[..., first + 1] = negative * attenuation[..., None]
        eigen_bottom[..., first + 1] = negative

    if np.any(conservative):
        conservative_lambda = np.sqrt(a[2] * a[3]) / 3.0
        attenuation = np.exp(-np.minimum(conservative_lambda * tau_flat, 700.0))
        constant_mode = np.stack(
            (
                np.ones_like(a[0]),
                np.zeros_like(a[0]),
                np.zeros_like(a[0]),
                np.zeros_like(a[0]),
            ),
            axis=-1,
        )
        polynomial_top = np.stack(
            (
                np.zeros_like(a[0]),
                np.ones_like(a[0]),
                np.zeros_like(a[0]),
                np.zeros_like(a[0]),
            ),
            axis=-1,
        )
        polynomial_bottom = polynomial_top.copy()
        polynomial_bottom[..., 0] = a[1] * tau_flat
        decaying = np.stack(
            (
                -2.0 * np.ones_like(a[0]),
                np.zeros_like(a[0]),
                np.ones_like(a[0]),
                -3.0 * conservative_lambda / a[3],
            ),
            axis=-1,
        )
        growing = decaying.copy()
        growing[..., 3] *= -1.0
        for values, replacement in (
            (eigen_top[..., 0], constant_mode),
            (eigen_bottom[..., 0], constant_mode),
            (eigen_top[..., 1], polynomial_top),
            (eigen_bottom[..., 1], polynomial_bottom),
            (eigen_top[..., 2], decaying),
            (eigen_bottom[..., 2], decaying * attenuation[..., None]),
            (eigen_top[..., 3], growing * attenuation[..., None]),
            (eigen_bottom[..., 3], growing),
        ):
            values[conservative] = replacement[conservative]

    slope = np.divide(
        planck[1:, :, None] - planck[:-1, :, None],
        tau,
        out=np.zeros_like(tau),
        where=tau > 0.0,
    ).reshape(nlayer, ncolumn)
    particular_top = np.zeros((nlayer, ncolumn, 4), dtype=float)
    particular_bottom = np.zeros_like(particular_top)
    thermal_factor = np.divide(
        1.0 - omega_flat,
        a[0],
        out=np.zeros_like(omega_flat),
        where=a[0] > 0.0,
    )
    top_planck = np.broadcast_to(planck[:-1, :, None], tau.shape).reshape(
        nlayer, ncolumn
    )
    bottom_level_planck = np.broadcast_to(planck[1:, :, None], tau.shape).reshape(
        nlayer, ncolumn
    )
    particular_top[..., 0] = thermal_factor * top_planck
    particular_bottom[..., 0] = thermal_factor * bottom_level_planck
    particular_top[..., 1] = thermal_factor * slope / a[1]
    particular_bottom[..., 1] = particular_top[..., 1]

    flux_top = np.einsum("ij,lcjk->lcik", _HALF_RANGE_MOMENTS, eigen_top)
    flux_bottom = np.einsum("ij,lcjk->lcik", _HALF_RANGE_MOMENTS, eigen_bottom)
    particular_flux_top = np.einsum("ij,lcj->lci", _HALF_RANGE_MOMENTS, particular_top)
    particular_flux_bottom = np.einsum(
        "ij,lcj->lci", _HALF_RANGE_MOMENTS, particular_bottom
    )

    size = 4 * nlayer
    band = np.zeros((11, size, ncolumn), dtype=float)
    rhs = np.zeros((size, ncolumn), dtype=float)
    _put_banded(band, 0, 0, flux_top[0, :, :2, :])
    rhs[:2] = -particular_flux_top[0, :, :2].T
    for layer in range(nlayer - 1):
        row = 2 + 4 * layer
        _put_banded(band, row, 4 * layer, flux_bottom[layer])
        _put_banded(band, row, 4 * (layer + 1), -flux_top[layer + 1])
        rhs[row : row + 4] = (
            particular_flux_top[layer + 1] - particular_flux_bottom[layer]
        ).T
    bottom_row = size - 2
    _put_banded(band, bottom_row, size - 4, flux_bottom[-1, :, 2:, :])
    target = np.stack((np.pi * bottom_planck, -0.25 * np.pi * bottom_planck), axis=0)
    target = np.repeat(target[:, :, None], ng, axis=2).reshape(2, ncolumn)
    rhs[bottom_row:] = target - particular_flux_bottom[-1, :, 2:].T

    coefficients = np.empty((nlayer, ncolumn, 4), dtype=float)
    for column in range(ncolumn):
        try:
            coefficients[:, column, :] = solve_banded(
                (5, 5),
                band[:, :, column],
                rhs[:, column],
                check_finite=False,
            ).reshape(nlayer, 4)
        except np.linalg.LinAlgError as error:
            raise RobertValidationError(
                "SH4 multilayer boundary system is singular"
            ) from error
    if not np.all(np.isfinite(coefficients)):
        raise RobertValidationError(
            "SH4 multilayer solve produced non-finite coefficients"
        )
    return coefficients, eigen_top, eigen_bottom, particular_top, particular_bottom


def _put_banded(
    band: NDArray[np.float64],
    row_start: int,
    column_start: int,
    block: NDArray[np.float64],
) -> None:
    """Place a ``(column,row,column)`` block into SciPy banded storage."""

    ncolumn, nrow_block, ncol_block = block.shape
    if ncolumn != band.shape[2]:
        raise RobertValidationError("internal SH4 banded block column mismatch")
    for local_row in range(nrow_block):
        row = row_start + local_row
        for local_column in range(ncol_block):
            column = column_start + local_column
            band[5 + row - column, column, :] = block[:, local_row, local_column]


def _moment_levels(
    coefficients: NDArray[np.float64],
    eigen_top: NDArray[np.float64],
    eigen_bottom: NDArray[np.float64],
    particular_top: NDArray[np.float64],
    particular_bottom: NDArray[np.float64],
) -> NDArray[np.float64]:
    nlayer, ncolumn, _ = coefficients.shape
    levels = np.empty((nlayer + 1, ncolumn, 4), dtype=float)
    levels[0] = (
        np.einsum("cij,cj->ci", eigen_top[0], coefficients[0]) + particular_top[0]
    )
    for layer in range(nlayer):
        levels[layer + 1] = (
            np.einsum("cij,cj->ci", eigen_bottom[layer], coefficients[layer])
            + particular_bottom[layer]
        )
    return levels


def _reconstruct_sources(
    coefficients: NDArray[np.float64],
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    moments: NDArray[np.float64],
    planck: NDArray[np.float64],
    bottom_planck: NDArray[np.float64],
    mu: NDArray[np.float64],
    quadrature_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nlayer, nspectral, ng = tau.shape
    ncolumn = nspectral * ng
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    point_layer = np.zeros((mu.size, nlayer, nspectral, ng), dtype=float)

    tau_flat = tau.reshape(nlayer, ncolumn)
    omega_flat = omega.reshape(nlayer, ncolumn)
    moments_flat = moments.reshape(4, nlayer, ncolumn)
    a = (2.0 * np.arange(4) + 1.0)[:, None, None] - omega_flat[None, ...] * moments_flat
    beta = a[0] * a[1] + a[2] * a[3] / 9.0 + 4.0 * a[0] * a[3] / 9.0
    gamma = a[0] * a[1] * a[2] * a[3] / 9.0
    root = np.sqrt(np.maximum(beta * beta - 4.0 * gamma, 0.0))
    lambdas = (np.sqrt(0.5 * (beta + root)), np.sqrt(0.5 * (beta - root)))
    conservative = a[0] == 0.0
    thermal_factor = np.divide(
        1.0 - omega_flat,
        a[0],
        out=np.zeros_like(omega_flat),
        where=a[0] > 0.0,
    )
    slope = np.divide(
        planck[1:, :, None] - planck[:-1, :, None],
        tau,
        out=np.zeros_like(tau),
        where=tau > 0.0,
    ).reshape(nlayer, ncolumn)
    top_planck = np.broadcast_to(planck[:-1, :, None], tau.shape).reshape(
        nlayer, ncolumn
    )

    cumulative_before = np.cumsum(tau, axis=0) - tau
    for node, weight in zip(nodes, weights, strict=True):
        eigen = np.empty((nlayer, ncolumn, 4, 4), dtype=float)
        for pair, eigenvalue in enumerate(lambdas):
            safe_eigenvalue = np.where(conservative & (pair == 1), 1.0, eigenvalue)
            r_value = -a[0] / safe_eigenvalue
            q_value = 0.5 * (a[0] * a[1] / safe_eigenvalue**2 - 1.0)
            s_value = -1.5 / a[3] * (a[0] * a[1] / safe_eigenvalue - safe_eigenvalue)
            positive = np.stack(
                (np.ones_like(r_value), r_value, q_value, s_value), axis=-1
            )
            negative = positive.copy()
            negative[..., 1] *= -1.0
            negative[..., 3] *= -1.0
            eigen[..., 2 * pair] = (
                positive
                * np.exp(-np.minimum(eigenvalue * tau_flat * node, 700.0))[..., None]
            )
            eigen[..., 2 * pair + 1] = (
                negative
                * np.exp(-np.minimum(eigenvalue * tau_flat * (1.0 - node), 700.0))[
                    ..., None
                ]
            )
        if np.any(conservative):
            conservative_lambda = np.sqrt(a[2] * a[3]) / 3.0
            constant_mode = np.stack(
                (
                    np.ones_like(a[0]),
                    np.zeros_like(a[0]),
                    np.zeros_like(a[0]),
                    np.zeros_like(a[0]),
                ),
                axis=-1,
            )
            polynomial = np.stack(
                (
                    a[1] * tau_flat * node,
                    np.ones_like(a[0]),
                    np.zeros_like(a[0]),
                    np.zeros_like(a[0]),
                ),
                axis=-1,
            )
            decaying = (
                np.stack(
                    (
                        -2.0 * np.ones_like(a[0]),
                        np.zeros_like(a[0]),
                        np.ones_like(a[0]),
                        -3.0 * conservative_lambda / a[3],
                    ),
                    axis=-1,
                )
                * np.exp(-np.minimum(conservative_lambda * tau_flat * node, 700.0))[
                    ..., None
                ]
            )
            growing = (
                np.stack(
                    (
                        -2.0 * np.ones_like(a[0]),
                        np.zeros_like(a[0]),
                        np.ones_like(a[0]),
                        3.0 * conservative_lambda / a[3],
                    ),
                    axis=-1,
                )
                * np.exp(
                    -np.minimum(conservative_lambda * tau_flat * (1.0 - node), 700.0)
                )[..., None]
            )
            for values, replacement in (
                (eigen[..., 0], constant_mode),
                (eigen[..., 1], polynomial),
                (eigen[..., 2], decaying),
                (eigen[..., 3], growing),
            ):
                values[conservative] = replacement[conservative]
        local_planck = top_planck + node * slope * tau_flat
        particular = np.zeros((nlayer, ncolumn, 4), dtype=float)
        particular[..., 0] = thermal_factor * local_planck
        particular[..., 1] = thermal_factor * slope / a[1]
        local_moments = np.einsum("lcij,lcj->lci", eigen, coefficients) + particular
        local_moments = local_moments.reshape(nlayer, nspectral, ng, 4)
        for angle, mu_value in enumerate(mu):
            legendre = np.array(
                [
                    1.0,
                    mu_value,
                    0.5 * (3.0 * mu_value**2 - 1.0),
                    0.5 * (5.0 * mu_value**3 - 3.0 * mu_value),
                ]
            )
            scattering_source = omega * (
                local_moments[..., 0] * moments[0] * legendre[0]
                + local_moments[..., 1] * moments[1] * legendre[1]
                + local_moments[..., 2] * moments[2] * legendre[2]
                + local_moments[..., 3] * moments[3] * legendre[3]
            )
            source = scattering_source + (1.0 - omega) * local_planck.reshape(
                nlayer, nspectral, ng
            )
            attenuation = np.exp(-(cumulative_before + node * tau) / mu_value)
            point_layer[angle] += weight * source * attenuation * tau / mu_value
    total_tau = np.sum(tau, axis=0)
    point_bottom = bottom_planck[None, :, None] * np.exp(
        -total_tau[None, ...] / mu[:, None, None]
    )
    if not np.all(np.isfinite(point_layer)) or not np.all(np.isfinite(point_bottom)):
        raise RobertValidationError(
            "SH4 source reconstruction produced non-finite values"
        )
    return point_layer, point_bottom


def _normalized_weights(
    value: ArrayLike,
    size: int,
    name: str,
) -> NDArray[np.float64]:
    weights = _finite_array(value, name, shape=(size,))
    if np.any(weights < 0.0):
        raise RobertValidationError(f"{name} must be non-negative")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RobertValidationError(f"{name} must have a finite positive sum")
    return np.asarray(weights / total, dtype=float)


def _numba_reconstruct_spectrum(
    coefficients: NDArray[np.float64],
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    moments: NDArray[np.float64],
    planck: NDArray[np.float64],
    bottom_planck: NDArray[np.float64],
    mu: NDArray[np.float64],
    angle_weights: NDArray[np.float64],
    g_weights: NDArray[np.float64],
    nodes: NDArray[np.float64],
    node_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    if not _NUMBA_AVAILABLE:
        raise RobertValidationError(
            "SH4 spectrum backend 'numba' requires the optional numba package"
        )
    return _numba_reconstruct_spectrum_kernel(
        coefficients,
        tau,
        omega,
        moments,
        planck,
        bottom_planck,
        mu,
        angle_weights,
        g_weights,
        nodes,
        node_weights,
    )


if _NUMBA_AVAILABLE:

    @njit(parallel=True)
    def _numba_reconstruct_spectrum_kernel(
        coefficients,
        tau,
        omega,
        moments,
        planck,
        bottom_planck,
        mu,
        angle_weights,
        g_weights,
        nodes,
        node_weights,
    ):
        nlayer, nspectral, ng = tau.shape
        radiance = np.zeros(nspectral, dtype=np.float64)
        for spectral_index in prange(nspectral):
            spectral_radiance = 0.0
            for g_index in range(ng):
                column = spectral_index * ng + g_index
                local_moment0 = np.empty((nodes.size, nlayer), dtype=np.float64)
                local_moment1 = np.empty((nodes.size, nlayer), dtype=np.float64)
                local_moment2 = np.empty((nodes.size, nlayer), dtype=np.float64)
                local_moment3 = np.empty((nodes.size, nlayer), dtype=np.float64)
                local_plancks = np.empty((nodes.size, nlayer), dtype=np.float64)
                total_tau = 0.0
                for layer_index in range(nlayer):
                    total_tau += tau[layer_index, spectral_index, g_index]

                for node_index in range(nodes.size):
                    node = nodes[node_index]
                    for layer_index in range(nlayer):
                        layer_tau = tau[layer_index, spectral_index, g_index]
                        layer_omega = omega[layer_index, spectral_index, g_index]
                        a0 = (
                            1.0
                            - layer_omega
                            * moments[0, layer_index, spectral_index, g_index]
                        )
                        a1 = (
                            3.0
                            - layer_omega
                            * moments[1, layer_index, spectral_index, g_index]
                        )
                        a2 = (
                            5.0
                            - layer_omega
                            * moments[2, layer_index, spectral_index, g_index]
                        )
                        a3 = (
                            7.0
                            - layer_omega
                            * moments[3, layer_index, spectral_index, g_index]
                        )
                        slope = 0.0
                        if layer_tau > 0.0:
                            slope = (
                                planck[layer_index + 1, spectral_index]
                                - planck[layer_index, spectral_index]
                            ) / layer_tau
                        local_planck = (
                            planck[layer_index, spectral_index]
                            + node * slope * layer_tau
                        )
                        thermal_factor = 0.0
                        if a0 > 0.0:
                            thermal_factor = (1.0 - layer_omega) / a0

                        c0 = coefficients[layer_index, column, 0]
                        c1 = coefficients[layer_index, column, 1]
                        c2 = coefficients[layer_index, column, 2]
                        c3 = coefficients[layer_index, column, 3]
                        moment0 = thermal_factor * local_planck
                        moment1 = thermal_factor * slope / a1
                        moment2 = 0.0
                        moment3 = 0.0
                        if a0 == 0.0:
                            conservative_lambda = np.sqrt(a2 * a3) / 3.0
                            decay = np.exp(
                                -min(conservative_lambda * layer_tau * node, 700.0)
                            )
                            growth = np.exp(
                                -min(
                                    conservative_lambda * layer_tau * (1.0 - node),
                                    700.0,
                                )
                            )
                            moment0 += (
                                c0
                                + a1 * layer_tau * node * c1
                                - 2.0 * decay * c2
                                - 2.0 * growth * c3
                            )
                            moment1 += c1
                            moment2 += decay * c2 + growth * c3
                            moment3 += (
                                -3.0 * conservative_lambda / a3 * decay * c2
                                + 3.0 * conservative_lambda / a3 * growth * c3
                            )
                        else:
                            beta = a0 * a1 + a2 * a3 / 9.0 + 4.0 * a0 * a3 / 9.0
                            gamma = a0 * a1 * a2 * a3 / 9.0
                            root = np.sqrt(max(beta * beta - 4.0 * gamma, 0.0))
                            lambda1 = np.sqrt(0.5 * (beta + root))
                            lambda2 = np.sqrt(0.5 * (beta - root))
                            for pair in range(2):
                                eigenvalue = lambda1 if pair == 0 else lambda2
                                r_value = -a0 / eigenvalue
                                q_value = 0.5 * (
                                    a0 * a1 / (eigenvalue * eigenvalue) - 1.0
                                )
                                s_value = (
                                    -1.5 / a3 * (a0 * a1 / eigenvalue - eigenvalue)
                                )
                                positive_attenuation = np.exp(
                                    -min(eigenvalue * layer_tau * node, 700.0)
                                )
                                negative_attenuation = np.exp(
                                    -min(
                                        eigenvalue * layer_tau * (1.0 - node),
                                        700.0,
                                    )
                                )
                                positive_coefficient = c0 if pair == 0 else c2
                                negative_coefficient = c1 if pair == 0 else c3
                                positive = positive_attenuation * positive_coefficient
                                negative = negative_attenuation * negative_coefficient
                                moment0 += positive + negative
                                moment1 += r_value * (positive - negative)
                                moment2 += q_value * (positive + negative)
                                moment3 += s_value * (positive - negative)

                        local_moment0[node_index, layer_index] = moment0
                        local_moment1[node_index, layer_index] = moment1
                        local_moment2[node_index, layer_index] = moment2
                        local_moment3[node_index, layer_index] = moment3
                        local_plancks[node_index, layer_index] = local_planck

                for angle_index in range(mu.size):
                    mu_value = mu[angle_index]
                    legendre1 = mu_value
                    legendre2 = 0.5 * (3.0 * mu_value * mu_value - 1.0)
                    legendre3 = 0.5 * (
                        5.0 * mu_value * mu_value * mu_value - 3.0 * mu_value
                    )
                    point_radiance = bottom_planck[spectral_index] * np.exp(
                        -total_tau / mu_value
                    )
                    cumulative_tau = 0.0
                    for layer_index in range(nlayer):
                        layer_tau = tau[layer_index, spectral_index, g_index]
                        layer_omega = omega[layer_index, spectral_index, g_index]
                        for node_index in range(nodes.size):
                            node = nodes[node_index]
                            scattering_source = layer_omega * (
                                local_moment0[node_index, layer_index]
                                * moments[0, layer_index, spectral_index, g_index]
                                + local_moment1[node_index, layer_index]
                                * moments[1, layer_index, spectral_index, g_index]
                                * legendre1
                                + local_moment2[node_index, layer_index]
                                * moments[2, layer_index, spectral_index, g_index]
                                * legendre2
                                + local_moment3[node_index, layer_index]
                                * moments[3, layer_index, spectral_index, g_index]
                                * legendre3
                            )
                            source = (
                                scattering_source
                                + (1.0 - layer_omega)
                                * local_plancks[node_index, layer_index]
                            )
                            attenuation = np.exp(
                                -(cumulative_tau + node * layer_tau) / mu_value
                            )
                            point_radiance += (
                                node_weights[node_index]
                                * source
                                * attenuation
                                * layer_tau
                                / mu_value
                            )
                        cumulative_tau += layer_tau
                    spectral_radiance += (
                        g_weights[g_index] * angle_weights[angle_index] * point_radiance
                    )
            radiance[spectral_index] = spectral_radiance
        return radiance

else:

    def _numba_reconstruct_spectrum_kernel(*args):
        raise RobertValidationError(
            "SH4 spectrum backend 'numba' requires the optional numba package"
        )


def _finite_array(
    value: ArrayLike,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if shape is not None and array.shape != shape:
        raise RobertValidationError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)
