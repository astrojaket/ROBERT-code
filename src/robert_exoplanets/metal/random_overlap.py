"""Float32 JAX conservative-RORR kernel for device-resident Metal graphs."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from robert_exoplanets.core import RobertValidationError

from .resources import MetalResourcePolicy


def metal_random_overlap_species_tau(
    species_tau: Any,
    g_weights: Any,
    *,
    cutoff: float = 1.0e-12,
    runtime: Any | None = None,
):
    """Conservatively mix device arrays without copying the result to host.

    Inputs have shape ``species x layer x wavelength x g``.  Each pairwise
    overlap is sorted then integrated at the original target-bin boundaries,
    exactly matching ROBERT's target-bin definition (within float32 error).
    The caller owns device placement and may pass a JAX device array directly.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - installation dependent.
        raise RobertValidationError("Metal RORR requires JAX") from exc
    if runtime is not None:
        if runtime.jax is not jax:
            raise RobertValidationError("runtime and imported JAX module disagree")
        tau = runtime.put(species_tau)
        weights = runtime.put(g_weights)
    else:
        tau = jnp.asarray(species_tau, dtype=jnp.float32)
        weights = jnp.asarray(g_weights, dtype=jnp.float32)
    if tau.ndim != 4 or any(size < 1 for size in tau.shape):
        raise RobertValidationError(
            "species_tau must have shape species x layer x wavelength x g"
        )
    if weights.shape != (tau.shape[-1],):
        raise RobertValidationError("g_weights must match the species_tau g axis")
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise RobertValidationError(
            "random-overlap cutoff must be finite and non-negative"
        )
    # Concrete host inputs are validated before transfer. Tracers/device arrays
    # are expected to come from a prepared, already validated outer graph.
    if isinstance(species_tau, (np.ndarray, list, tuple)):
        host_tau = np.asarray(species_tau)
        if not np.all(np.isfinite(host_tau)) or np.any(host_tau < 0.0):
            raise RobertValidationError("species_tau must be finite and non-negative")
    if isinstance(g_weights, (np.ndarray, list, tuple)):
        host_weights = np.asarray(g_weights)
        if (
            not np.all(np.isfinite(host_weights))
            or np.any(host_weights < 0.0)
            or np.sum(host_weights) <= 0.0
        ):
            raise RobertValidationError(
                "g_weights must be finite, non-negative, and have positive sum"
            )
    policy = MetalResourcePolicy.laptop_safe()
    policy.require_safe(
        "RORR graph",
        policy.estimate_rorr_bytes(
            species=tau.shape[0],
            layers=tau.shape[1],
            spectral=tau.shape[2],
            g_ordinates=tau.shape[3],
        ),
    )
    # Host-side shape validation is intentional; numerical work remains JIT/device-side.
    return _kernel()(tau, weights / jnp.sum(weights), jnp.float32(cutoff))


@lru_cache(maxsize=1)
def _kernel():
    import jax
    import jax.numpy as jnp

    lax = jax.lax

    def conservative_rebin(values, source_weights, target_weights):
        order = jnp.argsort(values)
        sorted_values = values[order]
        sorted_weights = source_weights[order]
        source_edges = jnp.concatenate(
            (jnp.zeros(1, values.dtype), jnp.cumsum(sorted_weights))
        )
        source_edges = source_edges.at[-1].set(1.0)
        integral = jnp.concatenate(
            (jnp.zeros(1, values.dtype), jnp.cumsum(sorted_values * sorted_weights))
        )
        target_edges = jnp.concatenate(
            (jnp.zeros(1, values.dtype), jnp.cumsum(target_weights))
        )
        target_edges = target_edges.at[-1].set(1.0)
        return (
            jnp.diff(jnp.interp(target_edges, source_edges, integral)) / target_weights
        )

    def one_point(species, weights, pair_weights, threshold):
        n_g = species.shape[-1]

        def add_species(carry, next_tau):
            combined, started = carry
            active = jnp.max(next_tau) >= threshold

            def combine(_):
                values = (combined[:, None] + next_tau[None, :]).reshape(-1)
                return conservative_rebin(values, pair_weights, weights), jnp.asarray(
                    True
                )

            def first(_):
                return next_tau, jnp.asarray(True)

            def active_update(_):
                return lax.cond(started, combine, first, operand=None)

            return lax.cond(active, active_update, lambda _: carry, operand=None), None

        (combined, started), _ = lax.scan(
            add_species, (jnp.zeros(n_g, species.dtype), jnp.asarray(False)), species
        )
        return jnp.where(started, combined, jnp.zeros_like(combined))

    @jax.jit
    def run(tau, weights, threshold):
        if tau.shape[0] == 1:
            return tau[0]
        pair_weights = (weights[:, None] * weights[None, :]).reshape(-1)
        points = jnp.transpose(tau, (1, 2, 0, 3)).reshape(
            (-1, tau.shape[0], tau.shape[-1])
        )
        mixed = jax.vmap(one_point, in_axes=(0, None, None, None))(
            points, weights, pair_weights, threshold
        )
        return mixed.reshape(tau.shape[1], tau.shape[2], tau.shape[3])

    return run


__all__ = ["metal_random_overlap_species_tau"]
