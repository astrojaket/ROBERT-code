"""Thermal source-function integration kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError

try:  # pragma: no cover - availability depends on the local environment.
    from numba import njit, prange
except Exception:  # pragma: no cover - dependency availability is environment-specific.
    njit = None
    prange = range

_NUMBA_AVAILABLE = njit is not None


@dataclass(frozen=True)
class ThermalEmissionIntegrationResult:
    """Point-wise thermal-emission integration diagnostics."""

    point_layer_contribution_radiance: ArrayLike
    point_bottom_contribution_radiance: ArrayLike
    backend: str

    def __post_init__(self) -> None:
        layer = np.array(self.point_layer_contribution_radiance, dtype=float, copy=True)
        bottom = np.array(self.point_bottom_contribution_radiance, dtype=float, copy=True)
        if layer.ndim != 3:
            raise RobertValidationError("point_layer_contribution_radiance must be three-dimensional")
        if bottom.shape != (layer.shape[0], layer.shape[2]):
            raise RobertValidationError("point_bottom_contribution_radiance has incorrect shape")
        if not np.all(np.isfinite(layer)) or np.any(layer < 0.0):
            raise RobertValidationError("point layer thermal contributions must be non-negative")
        if not np.all(np.isfinite(bottom)) or np.any(bottom < 0.0):
            raise RobertValidationError("point bottom thermal contributions must be non-negative")
        if not self.backend:
            raise RobertValidationError("thermal integration backend must not be empty")

        layer.setflags(write=False)
        bottom.setflags(write=False)
        object.__setattr__(self, "point_layer_contribution_radiance", layer)
        object.__setattr__(self, "point_bottom_contribution_radiance", bottom)


@dataclass(frozen=True)
class ThermalEmissionSpectrumIntegrationResult:
    """Disk-integrated thermal radiance without contribution diagnostics."""

    radiance: ArrayLike
    backend: str

    def __post_init__(self) -> None:
        radiance = _readonly_1d(self.radiance, "radiance")
        if np.any(radiance < 0.0):
            raise RobertValidationError("thermal radiance must be non-negative")
        if not self.backend:
            raise RobertValidationError("thermal integration backend must not be empty")
        object.__setattr__(self, "radiance", radiance)


def thermal_integration_backend_name(value: str) -> str:
    """Normalize and validate a requested thermal-integration backend."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        return "numba" if _NUMBA_AVAILABLE else "numpy"
    if normalized == "numpy":
        return "numpy"
    if normalized == "numba":
        if not _NUMBA_AVAILABLE:
            raise RobertValidationError(
                "thermal_integration_backend='numba' requires the optional numba package"
            )
        return "numba"
    raise RobertValidationError("thermal_integration_backend must be 'auto', 'numpy', or 'numba'")


def integrate_thermal_emission(
    tau_ordered: ArrayLike,
    source_ordered: ArrayLike,
    g_weights: ArrayLike,
    emission_path_factors: ArrayLike,
    *,
    level_source_ordered: ArrayLike | None = None,
    bottom_source: ArrayLike | None = None,
    bottom_visible: ArrayLike | None = None,
    backend: str = "auto",
) -> ThermalEmissionIntegrationResult:
    """Integrate thermal layer and bottom-boundary emission for each disc point.

    Inputs must already be ordered from top to bottom in the layer axis.
    """

    tau = _readonly_3d(tau_ordered, "tau_ordered")
    source = _readonly_array(source_ordered, "source_ordered", tau.shape[:2])
    level_source = None
    if level_source_ordered is not None:
        level_source = _readonly_array(
            level_source_ordered,
            "level_source_ordered",
            (tau.shape[0] + 1, tau.shape[1]),
        )
        if np.any(level_source < 0.0):
            raise RobertValidationError("level_source_ordered must be non-negative")
    weights = _readonly_1d(g_weights, "g_weights")
    if weights.shape != (tau.shape[2],):
        raise RobertValidationError("g_weights must match the tau g axis")
    if np.any(weights < 0.0):
        raise RobertValidationError("g_weights must be non-negative")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise RobertValidationError("g_weights must have a finite positive sum")
    normalized_weights = np.array(weights / weight_sum, dtype=float, copy=True)

    raw_path_factors = np.array(emission_path_factors, dtype=float, copy=True)
    if raw_path_factors.ndim != 2:
        raise RobertValidationError("emission_path_factors must be two-dimensional")
    path_factors = _readonly_array(
        raw_path_factors,
        "emission_path_factors",
        (raw_path_factors.shape[0], tau.shape[0]),
    )
    if np.any(path_factors < 0.0):
        raise RobertValidationError("emission_path_factors must be non-negative")

    if bottom_source is None:
        bottom = np.zeros(tau.shape[1], dtype=float)
        visible = np.zeros(path_factors.shape[0], dtype=np.bool_)
    else:
        bottom = _readonly_1d(bottom_source, "bottom_source")
        if bottom.shape != (tau.shape[1],):
            raise RobertValidationError("bottom_source must match the spectral axis")
        if np.any(bottom < 0.0):
            raise RobertValidationError("bottom_source must be non-negative")
        if bottom_visible is None:
            visible = np.ones(path_factors.shape[0], dtype=np.bool_)
        else:
            visible = np.array(bottom_visible, dtype=np.bool_, copy=True)
            if visible.shape != (path_factors.shape[0],):
                raise RobertValidationError("bottom_visible must match the emission-point axis")

    selected_backend = thermal_integration_backend_name(backend)
    if selected_backend == "numba":
        point_layer, point_bottom = _numba_integrate_thermal_emission(
            tau,
            source,
            level_source,
            normalized_weights,
            path_factors,
            bottom,
            visible,
        )
    else:
        point_layer, point_bottom = _numpy_integrate_thermal_emission(
            tau,
            source,
            level_source,
            normalized_weights,
            path_factors,
            bottom,
            visible,
        )

    return ThermalEmissionIntegrationResult(
        point_layer_contribution_radiance=point_layer,
        point_bottom_contribution_radiance=point_bottom,
        backend=selected_backend,
    )


def integrate_thermal_emission_spectrum(
    tau_ordered: ArrayLike,
    source_ordered: ArrayLike,
    g_weights: ArrayLike,
    emission_path_factors: ArrayLike,
    emission_point_weights: ArrayLike,
    *,
    level_source_ordered: ArrayLike | None = None,
    bottom_source: ArrayLike | None = None,
    bottom_visible: ArrayLike | None = None,
    backend: str = "auto",
) -> ThermalEmissionSpectrumIntegrationResult:
    """Integrate only disk-averaged radiance, omitting contribution arrays."""

    tau = _readonly_3d(tau_ordered, "tau_ordered")
    source = _readonly_array(source_ordered, "source_ordered", tau.shape[:2])
    level_source = None
    if level_source_ordered is not None:
        level_source = _readonly_array(
            level_source_ordered,
            "level_source_ordered",
            (tau.shape[0] + 1, tau.shape[1]),
        )
        if np.any(level_source < 0.0):
            raise RobertValidationError("level_source_ordered must be non-negative")
    weights = _normalized_nonnegative_weights(
        g_weights, tau.shape[2], "g_weights"
    )
    raw_paths = np.array(emission_path_factors, dtype=float, copy=True)
    if raw_paths.ndim != 2:
        raise RobertValidationError("emission_path_factors must be two-dimensional")
    paths = _readonly_array(
        raw_paths,
        "emission_path_factors",
        (raw_paths.shape[0], tau.shape[0]),
    )
    if np.any(paths < 0.0):
        raise RobertValidationError("emission_path_factors must be non-negative")
    point_weights = _normalized_nonnegative_weights(
        emission_point_weights, paths.shape[0], "emission_point_weights"
    )
    if bottom_source is None:
        bottom = np.zeros(tau.shape[1], dtype=float)
        visible = np.zeros(paths.shape[0], dtype=np.bool_)
    else:
        bottom = _readonly_1d(bottom_source, "bottom_source")
        if bottom.shape != (tau.shape[1],) or np.any(bottom < 0.0):
            raise RobertValidationError(
                "bottom_source must match the spectral axis and be non-negative"
            )
        visible = (
            np.ones(paths.shape[0], dtype=np.bool_)
            if bottom_visible is None
            else np.array(bottom_visible, dtype=np.bool_, copy=True)
        )
        if visible.shape != (paths.shape[0],):
            raise RobertValidationError(
                "bottom_visible must match the emission-point axis"
            )
    selected_backend = thermal_integration_backend_name(backend)
    use_linear_source = level_source is not None
    levels = (
        np.zeros((tau.shape[0] + 1, tau.shape[1]), dtype=float)
        if level_source is None
        else level_source
    )
    if selected_backend == "numba":
        radiance = _numba_integrate_thermal_spectrum_kernel(
            tau,
            source,
            levels,
            use_linear_source,
            weights,
            paths,
            point_weights,
            bottom,
            visible,
        )
    else:
        radiance = _numpy_integrate_thermal_spectrum(
            tau,
            source,
            levels,
            use_linear_source,
            weights,
            paths,
            point_weights,
            bottom,
            visible,
        )
    return ThermalEmissionSpectrumIntegrationResult(
        radiance=radiance,
        backend=selected_backend,
    )


def _numpy_integrate_thermal_emission(
    tau: NDArray[np.float64],
    source: NDArray[np.float64],
    level_source: NDArray[np.float64] | None,
    weights: NDArray[np.float64],
    path_factors: NDArray[np.float64],
    bottom_source: NDArray[np.float64],
    bottom_visible: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_points, n_layers = path_factors.shape
    n_spectral = tau.shape[1]
    point_layer = np.zeros((n_points, n_layers, n_spectral), dtype=float)
    point_bottom = np.zeros((n_points, n_spectral), dtype=float)
    for point_index in range(n_points):
        slant_tau = tau * path_factors[point_index, :, None, None]
        cumulative_before = _exclusive_cumulative(slant_tau)
        transmission_before = np.exp(-cumulative_before)
        escape = -np.expm1(-slant_tau)
        if level_source is None:
            emitted = source[:, :, None] * escape
        else:
            linear_weight = _linear_source_bottom_weight(slant_tau, escape)
            emitted = (
                level_source[:-1, :, None] * escape
                + (level_source[1:, :, None] - level_source[:-1, :, None])
                * linear_weight
            )
        layer_radiance_by_g = transmission_before * emitted
        point_layer[point_index] = np.sum(layer_radiance_by_g * weights[None, None, :], axis=-1)
        if bottom_visible[point_index]:
            total_transmission = np.exp(-np.sum(slant_tau, axis=0))
            point_bottom[point_index] = (
                np.sum(total_transmission * weights[None, :], axis=-1) * bottom_source
            )
    return point_layer, point_bottom


def _numpy_integrate_thermal_spectrum(
    tau: NDArray[np.float64],
    source: NDArray[np.float64],
    level_source: NDArray[np.float64],
    use_linear_source: bool,
    weights: NDArray[np.float64],
    path_factors: NDArray[np.float64],
    point_weights: NDArray[np.float64],
    bottom_source: NDArray[np.float64],
    bottom_visible: NDArray[np.bool_],
) -> NDArray[np.float64]:
    radiance = np.zeros(tau.shape[1], dtype=float)
    for point_index in range(path_factors.shape[0]):
        slant_tau = tau * path_factors[point_index, :, None, None]
        cumulative_before = _exclusive_cumulative(slant_tau)
        transmission_before = np.exp(-cumulative_before)
        escape = -np.expm1(-slant_tau)
        if use_linear_source:
            linear_weight = _linear_source_bottom_weight(slant_tau, escape)
            emitted = (
                level_source[:-1, :, None] * escape
                + (level_source[1:, :, None] - level_source[:-1, :, None])
                * linear_weight
            )
        else:
            emitted = source[:, :, None] * escape
        point_radiance = np.sum(
            transmission_before * emitted * weights[None, None, :],
            axis=(0, 2),
        )
        if bottom_visible[point_index]:
            point_radiance += (
                np.sum(
                    np.exp(-np.sum(slant_tau, axis=0)) * weights[None, :],
                    axis=-1,
                )
                * bottom_source
            )
        radiance += point_weights[point_index] * point_radiance
    return radiance


def _exclusive_cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    output = np.zeros_like(values)
    if values.shape[0] > 1:
        output[1:] = np.cumsum(values[:-1], axis=0)
    return output


def _linear_source_bottom_weight(
    slant_tau: NDArray[np.float64],
    escape: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the exact bottom-source weight for a linear layer source."""

    small = np.abs(slant_tau) < 1.0e-5
    weight = np.empty_like(slant_tau)
    value = slant_tau[small]
    weight[small] = value / 2.0 - value**2 / 3.0 + value**3 / 8.0 - value**4 / 30.0
    weight[~small] = (
        escape[~small] - slant_tau[~small] * np.exp(-slant_tau[~small])
    ) / slant_tau[~small]
    return weight


def _numba_integrate_thermal_emission(
    tau: NDArray[np.float64],
    source: NDArray[np.float64],
    level_source: NDArray[np.float64] | None,
    weights: NDArray[np.float64],
    path_factors: NDArray[np.float64],
    bottom_source: NDArray[np.float64],
    bottom_visible: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not _NUMBA_AVAILABLE:
        raise RobertValidationError(
            "thermal_integration_backend='numba' requires the optional numba package"
        )
    use_linear_source = level_source is not None
    numba_level_source = (
        np.zeros((tau.shape[0] + 1, tau.shape[1]), dtype=float)
        if level_source is None
        else level_source
    )
    return _numba_integrate_thermal_emission_kernel(
        tau,
        source,
        numba_level_source,
        use_linear_source,
        weights,
        path_factors,
        bottom_source,
        bottom_visible,
    )


if _NUMBA_AVAILABLE:

    @njit(parallel=True)
    def _numba_integrate_thermal_spectrum_kernel(
        tau,
        source,
        level_source,
        use_linear_source,
        weights,
        path_factors,
        point_weights,
        bottom_source,
        bottom_visible,
    ):
        n_points, n_layers = path_factors.shape
        n_spectral = tau.shape[1]
        n_g = tau.shape[2]
        radiance = np.zeros(n_spectral, dtype=np.float64)
        for spectral_index in prange(n_spectral):
            disk_value = 0.0
            for point_index in range(n_points):
                point_value = 0.0
                for g_index in range(n_g):
                    cumulative_tau = 0.0
                    g_weight = weights[g_index]
                    for layer_index in range(n_layers):
                        slant_tau = (
                            tau[layer_index, spectral_index, g_index]
                            * path_factors[point_index, layer_index]
                        )
                        transmission_before = np.exp(-cumulative_tau)
                        escape = -np.expm1(-slant_tau)
                        if use_linear_source:
                            if np.abs(slant_tau) < 1.0e-5:
                                linear_weight = (
                                    slant_tau / 2.0
                                    - slant_tau**2 / 3.0
                                    + slant_tau**3 / 8.0
                                    - slant_tau**4 / 30.0
                                )
                            else:
                                linear_weight = (
                                    escape - slant_tau * np.exp(-slant_tau)
                                ) / slant_tau
                            emitted = (
                                level_source[layer_index, spectral_index] * escape
                                + (
                                    level_source[layer_index + 1, spectral_index]
                                    - level_source[layer_index, spectral_index]
                                )
                                * linear_weight
                            )
                        else:
                            emitted = source[layer_index, spectral_index] * escape
                        point_value += transmission_before * emitted * g_weight
                        cumulative_tau += slant_tau
                    if bottom_visible[point_index]:
                        point_value += (
                            np.exp(-cumulative_tau)
                            * g_weight
                            * bottom_source[spectral_index]
                        )
                disk_value += point_weights[point_index] * point_value
            radiance[spectral_index] = disk_value
        return radiance

    @njit(parallel=True)
    def _numba_integrate_thermal_emission_kernel(
        tau,
        source,
        level_source,
        use_linear_source,
        weights,
        path_factors,
        bottom_source,
        bottom_visible,
    ):
        n_points, n_layers = path_factors.shape
        n_spectral = tau.shape[1]
        n_g = tau.shape[2]
        point_layer = np.zeros((n_points, n_layers, n_spectral), dtype=np.float64)
        point_bottom = np.zeros((n_points, n_spectral), dtype=np.float64)

        for point_index in prange(n_points):
            for spectral_index in range(n_spectral):
                for g_index in range(n_g):
                    cumulative_tau = 0.0
                    weight = weights[g_index]
                    for layer_index in range(n_layers):
                        slant_tau = tau[layer_index, spectral_index, g_index] * path_factors[
                            point_index,
                            layer_index,
                        ]
                        transmission_before = np.exp(-cumulative_tau)
                        escape = -np.expm1(-slant_tau)
                        if use_linear_source:
                            if np.abs(slant_tau) < 1.0e-5:
                                linear_weight = (
                                    slant_tau / 2.0
                                    - slant_tau**2 / 3.0
                                    + slant_tau**3 / 8.0
                                    - slant_tau**4 / 30.0
                                )
                            else:
                                linear_weight = (
                                    escape - slant_tau * np.exp(-slant_tau)
                                ) / slant_tau
                            emitted = (
                                level_source[layer_index, spectral_index] * escape
                                + (
                                    level_source[layer_index + 1, spectral_index]
                                    - level_source[layer_index, spectral_index]
                                )
                                * linear_weight
                            )
                        else:
                            emitted = source[layer_index, spectral_index] * escape
                        point_layer[point_index, layer_index, spectral_index] += (
                            transmission_before * emitted * weight
                        )
                        cumulative_tau += slant_tau
                    if bottom_visible[point_index]:
                        point_bottom[point_index, spectral_index] += (
                            np.exp(-cumulative_tau) * weight * bottom_source[spectral_index]
                        )
        return point_layer, point_bottom

else:

    def _numba_integrate_thermal_spectrum_kernel(*args):
        raise RobertValidationError(
            "thermal_integration_backend='numba' requires the optional numba package"
        )

    def _numba_integrate_thermal_emission_kernel(
        tau,
        source,
        level_source,
        use_linear_source,
        weights,
        path_factors,
        bottom_source,
        bottom_visible,
    ):
        raise RobertValidationError(
            "thermal_integration_backend='numba' requires the optional numba package"
        )


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise RobertValidationError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _normalized_nonnegative_weights(
    values: ArrayLike,
    size: int,
    name: str,
) -> NDArray[np.float64]:
    weights = _readonly_1d(values, name)
    if weights.shape != (size,) or np.any(weights < 0.0):
        raise RobertValidationError(
            f"{name} must match its integration axis and be non-negative"
        )
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RobertValidationError(f"{name} must have a finite positive sum")
    normalized = np.array(weights / total, dtype=float, copy=True)
    normalized.setflags(write=False)
    return normalized


def _readonly_3d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 3:
        raise RobertValidationError(f"{name} must be three-dimensional")
    if 0 in array.shape:
        raise RobertValidationError(f"{name} axes must be non-empty")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise RobertValidationError(f"{name} must be finite and non-negative")
    array.setflags(write=False)
    return array


def _readonly_array(
    values: ArrayLike,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != shape:
        raise RobertValidationError(f"{name} has incorrect shape")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
