"""Random-overlap correlated-k gas-combination helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError


def random_overlap_species_tau(
    species_tau: ArrayLike,
    g_weights: ArrayLike,
    *,
    cutoff: float = 1.0e-12,
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
        than one species is present. This mirrors the small-opacity guard used
        in the NEMESIS/NemesisPy random-overlap path.
    """

    tau = _readonly_species_tau(species_tau)
    weights = _normalized_g_weights(g_weights)
    if tau.shape[-1] != weights.size:
        raise RobertValidationError("g_weights must match the species_tau g axis")
    if cutoff < 0.0 or not np.isfinite(cutoff):
        raise RobertValidationError("random-overlap cutoff must be finite and non-negative")

    n_species, n_layers, n_spectral, n_g = tau.shape
    if n_species == 1:
        output = np.array(tau[0], dtype=float, copy=True)
        output.setflags(write=False)
        return output

    random_weights = (weights[:, None] * weights[None, :]).reshape(-1)
    target_edges = np.concatenate(([0.0], np.cumsum(weights)))
    combined = np.zeros((n_layers, n_spectral, n_g), dtype=float)
    for layer_index in range(n_layers):
        for spectral_index in range(n_spectral):
            combined[layer_index, spectral_index] = _random_overlap_tau_vectors_unchecked(
                tau[:, layer_index, spectral_index, :],
                weights,
                random_weights,
                target_edges,
                cutoff=cutoff,
            )

    combined.setflags(write=False)
    return combined


def random_overlap_tau_vectors(
    tau_by_species_g: ArrayLike,
    g_weights: ArrayLike,
    *,
    cutoff: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Combine one layer/wavelength set of species tau vectors."""

    tau = _readonly_tau_vectors(tau_by_species_g)
    weights = _normalized_g_weights(g_weights)
    if tau.shape[-1] != weights.size:
        raise RobertValidationError("g_weights must match the tau g axis")
    if cutoff < 0.0 or not np.isfinite(cutoff):
        raise RobertValidationError("random-overlap cutoff must be finite and non-negative")

    if tau.shape[0] == 1:
        output = np.array(tau[0], dtype=float, copy=True)
        output.setflags(write=False)
        return output

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
    rebinned = _rank_rebin_distribution_unchecked(value_array, source_weights, target, target_edges)
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
    cumulative_integral = np.concatenate(([0.0], np.cumsum(sorted_values * sorted_weights)))
    integral_at_target_edges = np.interp(target_edges, source_edges, cumulative_integral)
    rebinned = np.diff(integral_at_target_edges) / target_weights
    return np.array(rebinned, dtype=float, copy=True)


def _readonly_species_tau(values: ArrayLike) -> NDArray[np.float64]:
    tau = np.array(values, dtype=float, copy=True)
    if tau.ndim != 4:
        raise RobertValidationError("species_tau must have shape species x layer x wavelength x g")
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
