"""Random-overlap correlated-k gas-combination helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError

try:  # pragma: no cover - exercised only when the optional perf extra is installed.
    from numba import njit, prange
except Exception:  # pragma: no cover - dependency availability is environment-specific.
    njit = None
    prange = range

_NUMBA_AVAILABLE = njit is not None


def random_overlap_species_tau(
    species_tau: ArrayLike,
    g_weights: ArrayLike,
    *,
    cutoff: float = 1.0e-12,
    backend: str = "auto",
) -> NDArray[np.float64]:
    """Combine species optical-depth distributions with random overlap.

    Parameters
    ----------
    species_tau
        Optical depth with shape ``(species, layer, wavelength, g)``.
    g_weights
        Correlated-k quadrature weights for the last axis.
    cutoff
        Species with optical depths below this threshold are skipped when more
        than one species is present. This avoids spending time on effectively
        transparent gases in multi-species random-overlap calculations.
    backend
        ``"auto"`` uses the optional Numba kernel when available and otherwise
        uses the NumPy reference implementation. ``"numpy"`` and ``"numba"``
        force a specific backend.
    """

    tau = _readonly_species_tau(species_tau)
    weights = _normalized_g_weights(g_weights)
    if tau.shape[-1] != weights.size:
        raise RobertValidationError("g_weights must match the species_tau g axis")
    if cutoff < 0.0 or not np.isfinite(cutoff):
        raise RobertValidationError(
            "random-overlap cutoff must be finite and non-negative"
        )

    n_species, n_layers, n_spectral, n_g = tau.shape
    if n_species == 1:
        output = np.array(tau[0], dtype=float, copy=True)
        output.setflags(write=False)
        return output

    selected_backend = _random_overlap_backend(backend)
    if selected_backend == "numba":
        combined = _numba_random_overlap_species_tau(tau, weights, float(cutoff))
        combined.setflags(write=False)
        return combined

    random_weights = (weights[:, None] * weights[None, :]).reshape(-1)
    target_edges = np.concatenate(([0.0], np.cumsum(weights)))
    combined = np.zeros((n_layers, n_spectral, n_g), dtype=float)
    for layer_index in range(n_layers):
        for spectral_index in range(n_spectral):
            combined[layer_index, spectral_index] = (
                _random_overlap_tau_vectors_unchecked(
                    tau[:, layer_index, spectral_index, :],
                    weights,
                    random_weights,
                    target_edges,
                    cutoff=cutoff,
                )
            )

    combined.setflags(write=False)
    return combined


def fused_random_overlap_kcoeff(
    kcoeff: ArrayLike,
    species_column_density_molecules_m2: ArrayLike,
    g_weights: ArrayLike,
    *,
    unit_scale_m2: float,
    cutoff: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Scale molecular k-coefficients and combine them without a species-tau cube.

    This retrieval-oriented kernel is numerically equivalent to constructing
    ``species_tau = kcoeff * column_density * unit_scale_m2`` and passing that
    array to :func:`random_overlap_species_tau`. The explicit species-resolved
    route remains the diagnostic and NumPy reference implementation.
    """

    coefficients = np.asarray(kcoeff, dtype=float)
    columns = np.asarray(species_column_density_molecules_m2, dtype=float)
    weights = _normalized_g_weights(g_weights)
    if coefficients.ndim != 4:
        raise RobertValidationError(
            "kcoeff must have shape species x layer x wavelength x g"
        )
    if columns.shape != coefficients.shape[:2]:
        raise RobertValidationError(
            "species column density must have shape species x layer"
        )
    if coefficients.shape[-1] != weights.size:
        raise RobertValidationError("g_weights must match the kcoeff g axis")
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
        raise RobertValidationError("kcoeff must be finite and non-negative")
    if not np.all(np.isfinite(columns)) or np.any(columns < 0.0):
        raise RobertValidationError(
            "species column density must be finite and non-negative"
        )
    scale = float(unit_scale_m2)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RobertValidationError("unit_scale_m2 must be finite and positive")
    if cutoff < 0.0 or not np.isfinite(cutoff):
        raise RobertValidationError(
            "random-overlap cutoff must be finite and non-negative"
        )
    if _NUMBA_AVAILABLE:
        combined = _numba_fused_random_overlap_kcoeff_kernel(
            coefficients,
            columns,
            weights,
            scale,
            float(cutoff),
        )
    else:  # Preserve the base-install reference behavior without the perf extra.
        species_tau = coefficients * columns[:, :, None, None] * scale
        combined = random_overlap_species_tau(
            species_tau,
            weights,
            cutoff=cutoff,
            backend="numpy",
        )
    combined.setflags(write=False)
    return combined


def fused_random_overlap_backend_name() -> str:
    """Return the active diagnostics-free random-overlap assembly backend."""

    return "fused_numba_random_overlap" if _NUMBA_AVAILABLE else "numpy_reference_fallback"


def random_overlap_tau_vectors(
    tau_by_species_g: ArrayLike,
    g_weights: ArrayLike,
    *,
    cutoff: float = 1.0e-12,
    backend: str = "auto",
) -> NDArray[np.float64]:
    """Combine one layer/wavelength set of species tau vectors."""

    tau = _readonly_tau_vectors(tau_by_species_g)
    weights = _normalized_g_weights(g_weights)
    if tau.shape[-1] != weights.size:
        raise RobertValidationError("g_weights must match the tau g axis")
    if cutoff < 0.0 or not np.isfinite(cutoff):
        raise RobertValidationError(
            "random-overlap cutoff must be finite and non-negative"
        )

    if tau.shape[0] == 1:
        output = np.array(tau[0], dtype=float, copy=True)
        output.setflags(write=False)
        return output

    selected_backend = _random_overlap_backend(backend)
    if selected_backend == "numba":
        tau_4d = tau[:, None, None, :]
        combined_4d = _numba_random_overlap_species_tau(tau_4d, weights, float(cutoff))
        combined = np.array(combined_4d[0, 0], dtype=float, copy=True)
        combined.setflags(write=False)
        return combined

    random_weights = (weights[:, None] * weights[None, :]).reshape(-1)
    target_edges = np.concatenate(([0.0], np.cumsum(weights)))
    combined = _random_overlap_tau_vectors_unchecked(
        tau,
        weights,
        random_weights,
        target_edges,
        cutoff=cutoff,
    )
    combined.setflags(write=False)
    return combined


def _random_overlap_tau_vectors_unchecked(
    tau: NDArray[np.float64],
    weights: NDArray[np.float64],
    random_weights: NDArray[np.float64],
    target_edges: NDArray[np.float64],
    *,
    cutoff: float,
) -> NDArray[np.float64]:
    if tau.shape[0] == 1:
        return np.array(tau[0], dtype=float, copy=True)

    active = [row for row in tau if np.max(row) >= cutoff]
    if not active:
        output = np.zeros(weights.size, dtype=float)
        return output

    combined = np.array(active[0], dtype=float, copy=True)
    for next_tau in active[1:]:
        combined = _combine_two_distributions(
            combined,
            next_tau,
            weights,
            random_weights,
            target_edges,
        )
    return combined


def rank_rebin_distribution(
    values: ArrayLike,
    weights: ArrayLike,
    target_weights: ArrayLike,
) -> NDArray[np.float64]:
    """Sort a weighted distribution and average it into target g bins."""

    value_array = _readonly_1d(values, "values")
    source_weights = _normalized_g_weights(weights)
    target = _normalized_g_weights(target_weights)
    if value_array.shape != source_weights.shape:
        raise RobertValidationError("values and weights must have the same shape")

    target_edges = np.concatenate(([0.0], np.cumsum(target)))
    rebinned = _rank_rebin_distribution_unchecked(
        value_array, source_weights, target, target_edges
    )
    rebinned.setflags(write=False)
    return rebinned


def _combine_two_distributions(
    left_tau: NDArray[np.float64],
    right_tau: NDArray[np.float64],
    g_weights: NDArray[np.float64],
    random_weights: NDArray[np.float64],
    target_edges: NDArray[np.float64],
) -> NDArray[np.float64]:
    random_values = (left_tau[:, None] + right_tau[None, :]).reshape(-1)
    return _rank_rebin_distribution_unchecked(
        random_values,
        random_weights,
        g_weights,
        target_edges,
    )


def _rank_rebin_distribution_unchecked(
    values: NDArray[np.float64],
    source_weights: NDArray[np.float64],
    target_weights: NDArray[np.float64],
    target_edges: NDArray[np.float64],
) -> NDArray[np.float64]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = source_weights[order]
    source_edges = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    source_edges[-1] = 1.0
    cumulative_integral = np.concatenate(
        ([0.0], np.cumsum(sorted_values * sorted_weights))
    )
    integral_at_target_edges = np.interp(
        target_edges, source_edges, cumulative_integral
    )
    rebinned = np.diff(integral_at_target_edges) / target_weights
    return np.array(rebinned, dtype=float, copy=True)


def _readonly_species_tau(values: ArrayLike) -> NDArray[np.float64]:
    tau = np.array(values, dtype=float, copy=True)
    if tau.ndim != 4:
        raise RobertValidationError(
            "species_tau must have shape species x layer x wavelength x g"
        )
    if tau.shape[0] < 1 or tau.shape[1] < 1 or tau.shape[2] < 1 or tau.shape[3] < 1:
        raise RobertValidationError("species_tau axes must be non-empty")
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError("species_tau must be finite and non-negative")
    tau.setflags(write=False)
    return tau


def _readonly_tau_vectors(values: ArrayLike) -> NDArray[np.float64]:
    tau = np.array(values, dtype=float, copy=True)
    if tau.ndim != 2:
        raise RobertValidationError("tau_by_species_g must have shape species x g")
    if tau.shape[0] < 1 or tau.shape[1] < 1:
        raise RobertValidationError("tau_by_species_g axes must be non-empty")
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError("tau_by_species_g must be finite and non-negative")
    tau.setflags(write=False)
    return tau


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size < 1:
        raise RobertValidationError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _normalized_g_weights(values: ArrayLike) -> NDArray[np.float64]:
    weights = _readonly_1d(values, "g_weights")
    if np.any(weights < 0.0):
        raise RobertValidationError("g_weights must be non-negative")
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise RobertValidationError("g_weights must have a finite positive sum")
    normalized = np.array(weights / total, dtype=float, copy=True)
    normalized.setflags(write=False)
    return normalized


def _random_overlap_backend(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "auto":
        return "numba" if _NUMBA_AVAILABLE else "numpy"
    if normalized == "numpy":
        return "numpy"
    if normalized == "numba":
        if not _NUMBA_AVAILABLE:
            raise RobertValidationError(
                "random-overlap backend 'numba' requires the optional numba package"
            )
        return "numba"
    raise RobertValidationError(
        "random-overlap backend must be 'auto', 'numpy', or 'numba'"
    )


def _numba_random_overlap_species_tau(
    tau: NDArray[np.float64],
    weights: NDArray[np.float64],
    cutoff: float,
) -> NDArray[np.float64]:
    if not _NUMBA_AVAILABLE:
        raise RobertValidationError(
            "random-overlap backend 'numba' requires the optional numba package"
        )
    return _numba_random_overlap_species_tau_kernel(tau, weights, cutoff)


if _NUMBA_AVAILABLE:

    @njit(parallel=True)
    def _numba_fused_random_overlap_kcoeff_kernel(
        kcoeff,
        species_columns,
        weights,
        unit_scale,
        cutoff,
    ):
        n_species, n_layers, n_spectral, n_g = kcoeff.shape
        output = np.zeros((n_layers, n_spectral, n_g), dtype=np.float64)
        random_weights = np.empty(n_g * n_g, dtype=np.float64)
        for i_g in range(n_g):
            for j_g in range(n_g):
                random_weights[i_g * n_g + j_g] = weights[i_g] * weights[j_g]

        n_points = n_layers * n_spectral
        if n_species == 1:
            for point_index in prange(n_points):
                layer_index = point_index // n_spectral
                spectral_index = point_index - layer_index * n_spectral
                species_scale = species_columns[0, layer_index] * unit_scale
                for g_index in range(n_g):
                    output[layer_index, spectral_index, g_index] = (
                        kcoeff[0, layer_index, spectral_index, g_index]
                        * species_scale
                    )
            return output
        for point_index in prange(n_points):
            layer_index = point_index // n_spectral
            spectral_index = point_index - layer_index * n_spectral
            first_active = -1
            for species_index in range(n_species):
                species_scale = (
                    species_columns[species_index, layer_index] * unit_scale
                )
                max_tau = 0.0
                for g_index in range(n_g):
                    value = (
                        kcoeff[
                            species_index,
                            layer_index,
                            spectral_index,
                            g_index,
                        ]
                        * species_scale
                    )
                    if value > max_tau:
                        max_tau = value
                if max_tau >= cutoff:
                    first_active = species_index
                    break

            if first_active < 0:
                continue

            combined = np.empty(n_g, dtype=np.float64)
            right_tau = np.empty(n_g, dtype=np.float64)
            next_combined = np.empty(n_g, dtype=np.float64)
            random_values = np.empty(n_g * n_g, dtype=np.float64)
            heap_values = np.empty(n_g, dtype=np.float64)
            heap_rows = np.empty(n_g, dtype=np.int64)
            heap_columns = np.empty(n_g, dtype=np.int64)
            first_scale = species_columns[first_active, layer_index] * unit_scale
            for g_index in range(n_g):
                combined[g_index] = (
                    kcoeff[
                        first_active,
                        layer_index,
                        spectral_index,
                        g_index,
                    ]
                    * first_scale
                )

            for species_index in range(first_active + 1, n_species):
                species_scale = (
                    species_columns[species_index, layer_index] * unit_scale
                )
                max_tau = 0.0
                for g_index in range(n_g):
                    value = (
                        kcoeff[
                            species_index,
                            layer_index,
                            spectral_index,
                            g_index,
                        ]
                        * species_scale
                    )
                    right_tau[g_index] = value
                    if value > max_tau:
                        max_tau = value
                if max_tau < cutoff:
                    continue

                right_is_sorted = True
                for g_index in range(1, n_g):
                    if right_tau[g_index] < right_tau[g_index - 1]:
                        right_is_sorted = False
                        break
                if right_is_sorted:
                    _numba_combine_sorted_distributions_into(
                        combined,
                        right_tau,
                        weights,
                        heap_values,
                        heap_rows,
                        heap_columns,
                        next_combined,
                    )
                else:
                    _numba_combine_two_distributions_into(
                        combined,
                        right_tau,
                        weights,
                        random_weights,
                        random_values,
                        next_combined,
                    )
                for g_index in range(n_g):
                    combined[g_index] = next_combined[g_index]

            for g_index in range(n_g):
                output[layer_index, spectral_index, g_index] = combined[g_index]
        return output

    @njit(parallel=True)
    def _numba_random_overlap_species_tau_kernel(tau, weights, cutoff):
        n_species, n_layers, n_spectral, n_g = tau.shape
        output = np.zeros((n_layers, n_spectral, n_g), dtype=np.float64)
        random_weights = np.empty(n_g * n_g, dtype=np.float64)
        for i_g in range(n_g):
            for j_g in range(n_g):
                random_weights[i_g * n_g + j_g] = weights[i_g] * weights[j_g]

        n_points = n_layers * n_spectral
        for point_index in prange(n_points):
            layer_index = point_index // n_spectral
            spectral_index = point_index - layer_index * n_spectral
            first_active = -1
            for species_index in range(n_species):
                max_tau = 0.0
                for g_index in range(n_g):
                    value = tau[species_index, layer_index, spectral_index, g_index]
                    if value > max_tau:
                        max_tau = value
                if max_tau >= cutoff:
                    first_active = species_index
                    break

            if first_active < 0:
                continue

            combined = np.empty(n_g, dtype=np.float64)
            next_combined = np.empty(n_g, dtype=np.float64)
            random_values = np.empty(n_g * n_g, dtype=np.float64)
            heap_values = np.empty(n_g, dtype=np.float64)
            heap_rows = np.empty(n_g, dtype=np.int64)
            heap_columns = np.empty(n_g, dtype=np.int64)
            for g_index in range(n_g):
                combined[g_index] = tau[
                    first_active, layer_index, spectral_index, g_index
                ]

            for species_index in range(first_active + 1, n_species):
                max_tau = 0.0
                for g_index in range(n_g):
                    value = tau[species_index, layer_index, spectral_index, g_index]
                    if value > max_tau:
                        max_tau = value
                if max_tau < cutoff:
                    continue

                right_tau = tau[species_index, layer_index, spectral_index]
                right_is_sorted = True
                for g_index in range(1, n_g):
                    if right_tau[g_index] < right_tau[g_index - 1]:
                        right_is_sorted = False
                        break
                if right_is_sorted:
                    _numba_combine_sorted_distributions_into(
                        combined,
                        right_tau,
                        weights,
                        heap_values,
                        heap_rows,
                        heap_columns,
                        next_combined,
                    )
                else:
                    _numba_combine_two_distributions_into(
                        combined,
                        right_tau,
                        weights,
                        random_weights,
                        random_values,
                        next_combined,
                    )
                for g_index in range(n_g):
                    combined[g_index] = next_combined[g_index]

            for g_index in range(n_g):
                output[layer_index, spectral_index, g_index] = combined[g_index]
        return output

    @njit
    def _numba_combine_sorted_distributions_into(
        left_tau,
        right_tau,
        weights,
        heap_values,
        heap_rows,
        heap_columns,
        rebinned,
    ):
        n_g = weights.size
        for row_index in range(n_g):
            heap_values[row_index] = left_tau[row_index] + right_tau[0]
            heap_rows[row_index] = row_index
            heap_columns[row_index] = 0
            rebinned[row_index] = 0.0

        heap_size = n_g
        for start_index in range(n_g // 2 - 1, -1, -1):
            _numba_sift_down(
                heap_values,
                heap_rows,
                heap_columns,
                heap_size,
                start_index,
            )

        target_index = 0
        target_left = 0.0
        target_right = weights[0]
        source_left = 0.0
        for _ in range(n_g * n_g):
            value = heap_values[0]
            row_index = heap_rows[0]
            column_index = heap_columns[0]
            source_right = source_left + weights[row_index] * weights[column_index]
            while target_index < n_g:
                overlap = min(source_right, target_right) - max(
                    source_left, target_left
                )
                if overlap > 0.0:
                    rebinned[target_index] += value * overlap
                if source_right >= target_right and target_index < n_g - 1:
                    target_left = target_right
                    target_index += 1
                    target_right = target_left + weights[target_index]
                    continue
                break
            source_left = source_right

            next_column = column_index + 1
            if next_column < n_g:
                heap_values[0] = left_tau[row_index] + right_tau[next_column]
                heap_rows[0] = row_index
                heap_columns[0] = next_column
            else:
                heap_size -= 1
                if heap_size == 0:
                    break
                heap_values[0] = heap_values[heap_size]
                heap_rows[0] = heap_rows[heap_size]
                heap_columns[0] = heap_columns[heap_size]
            _numba_sift_down(heap_values, heap_rows, heap_columns, heap_size, 0)

        for g_index in range(n_g):
            rebinned[g_index] /= weights[g_index]

    @njit(inline="always")
    def _numba_sift_down(heap_values, heap_rows, heap_columns, heap_size, start_index):
        parent = start_index
        while True:
            left_child = 2 * parent + 1
            if left_child >= heap_size:
                return
            right_child = left_child + 1
            smallest = left_child
            if (
                right_child < heap_size
                and heap_values[right_child] < heap_values[left_child]
            ):
                smallest = right_child
            if heap_values[parent] <= heap_values[smallest]:
                return
            heap_values[parent], heap_values[smallest] = (
                heap_values[smallest],
                heap_values[parent],
            )
            heap_rows[parent], heap_rows[smallest] = (
                heap_rows[smallest],
                heap_rows[parent],
            )
            heap_columns[parent], heap_columns[smallest] = (
                heap_columns[smallest],
                heap_columns[parent],
            )
            parent = smallest

    @njit
    def _numba_combine_two_distributions_into(
        left_tau,
        right_tau,
        weights,
        random_weights,
        random_values,
        rebinned,
    ):
        n_g = weights.size
        n_random = n_g * n_g
        for i_g in range(n_g):
            for j_g in range(n_g):
                random_values[i_g * n_g + j_g] = left_tau[i_g] + right_tau[j_g]

        order = np.argsort(random_values)
        for g_index in range(n_g):
            rebinned[g_index] = 0.0
        target_index = 0
        target_left = 0.0
        target_right = weights[0]
        source_left = 0.0
        for random_index in range(n_random):
            ordered_index = order[random_index]
            source_right = source_left + random_weights[ordered_index]
            while target_index < n_g:
                overlap = min(source_right, target_right) - max(
                    source_left, target_left
                )
                if overlap > 0.0:
                    rebinned[target_index] += random_values[ordered_index] * overlap
                if source_right >= target_right and target_index < n_g - 1:
                    target_left = target_right
                    target_index += 1
                    target_right = target_left + weights[target_index]
                    continue
                break
            source_left = source_right
        for g_index in range(n_g):
            rebinned[g_index] /= weights[g_index]

else:

    def _numba_fused_random_overlap_kcoeff_kernel(*args):
        raise RobertValidationError(
            "fused random-overlap k-coefficient assembly requires numba"
        )

    def _numba_random_overlap_species_tau_kernel(tau, weights, cutoff):
        raise RobertValidationError(
            "random-overlap backend 'numba' requires the optional numba package"
        )
