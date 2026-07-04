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

    combined = np.zeros((n_layers, n_spectral, n_g), dtype=float)
    for layer_index in range(n_layers):
        for spectral_index in range(n_spectral):
            combined[layer_index, spectral_index] = random_overlap_tau_vectors(
                tau[:, layer_index, spectral_index, :],
                weights,
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

    active = [row for row in tau if np.max(row) >= cutoff]
    if not active:
        output = np.zeros(weights.size, dtype=float)
        output.setflags(write=False)
        return output

    combined = np.array(active[0], dtype=float, copy=True)
    for next_tau in active[1:]:
        combined = _combine_two_distributions(combined, next_tau, weights)

    combined.setflags(write=False)
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

    order = np.argsort(value_array)
    sorted_values = value_array[order]
    sorted_weights = source_weights[order]
    left_edges = np.concatenate(([0.0], np.cumsum(sorted_weights)[:-1]))
    right_edges = np.cumsum(sorted_weights)
    target_left = np.concatenate(([0.0], np.cumsum(target)[:-1]))
    target_right = np.cumsum(target)

    rebinned = np.zeros(target.size, dtype=float)
    for index, (left, right, width) in enumerate(zip(target_left, target_right, target)):
        overlap = np.minimum(right_edges, right) - np.maximum(left_edges, left)
        overlap = np.maximum(overlap, 0.0)
        rebinned[index] = np.sum(sorted_values * overlap) / width

    rebinned.setflags(write=False)
    return rebinned


def _combine_two_distributions(
    left_tau: NDArray[np.float64],
    right_tau: NDArray[np.float64],
    g_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    random_values = (left_tau[:, None] + right_tau[None, :]).reshape(-1)
    random_weights = (g_weights[:, None] * g_weights[None, :]).reshape(-1)
    return rank_rebin_distribution(random_values, random_weights, g_weights)


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
