"""Optional JAX/XLA backend for conservative random-overlap recompression.

This module is independent of the production NumPy and Numba implementations.
It implements the same pairwise random-overlap construction and conservative
target-g-bin integration, using float64 JAX arrays. Importing this module does
not require JAX; JAX is loaded only when the backend is called.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertValidationError

_COMPILED_KERNEL: Any | None = None


def jax_random_overlap_species_tau(
    species_tau: ArrayLike,
    g_weights: ArrayLike,
    *,
    cutoff: float = 1.0e-12,
    platform: str = "cpu",
) -> NDArray[np.float64]:
    """Combine species with conservative RORR on the active JAX device.

    The returned array is copied to host NumPy memory. This explicit research
    API is not selected by ``backend="auto"`` anywhere in ROBERT.
    """

    tau, weights, threshold = _validated_inputs(species_tau, g_weights, cutoff)
    jax, jnp, kernel = _jax_runtime()
    device = _selected_device(jax, platform)
    result = kernel(
        jax.device_put(jnp.asarray(tau, dtype=jnp.float64), device),
        jax.device_put(jnp.asarray(weights, dtype=jnp.float64), device),
        threshold,
    )
    result.block_until_ready()
    output = np.asarray(jax.device_get(result), dtype=float)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0):
        raise RobertValidationError(
            "JAX random-overlap output must be finite and non-negative"
        )
    output.setflags(write=False)
    return output


def jax_random_overlap_device_info(platform: str | None = None) -> tuple[str, ...]:
    """Return the devices visible to the optional JAX runtime."""

    jax, _, _ = _jax_runtime()
    devices = jax.devices() if platform is None else jax.devices(_platform_name(platform))
    return tuple(f"{device.platform}:{device.device_kind}" for device in devices)


def _platform_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise RobertValidationError("JAX platform must be explicitly 'cpu' or 'gpu'")
    return normalized


def _selected_device(jax, platform: str):
    requested = _platform_name(platform)
    try:
        devices = jax.devices(requested)
    except RuntimeError as exc:
        raise RobertConfigError(
            f"requested JAX platform '{requested}' is unavailable; "
            f"visible devices: {jax.devices()}"
        ) from exc
    if not devices:
        raise RobertConfigError(
            f"requested JAX platform '{requested}' has no visible devices"
        )
    return devices[0]


def _validated_inputs(
    species_tau: ArrayLike,
    g_weights: ArrayLike,
    cutoff: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    tau = np.ascontiguousarray(np.asarray(species_tau, dtype=float))
    weights = np.ascontiguousarray(np.asarray(g_weights, dtype=float))
    if tau.ndim != 4 or any(size < 1 for size in tau.shape):
        raise RobertValidationError(
            "species_tau must have shape species x layer x wavelength x g"
        )
    if weights.shape != (tau.shape[-1],):
        raise RobertValidationError("g_weights must match the species_tau g axis")
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError("species_tau must be finite and non-negative")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RobertValidationError(
            "the JAX conservative-RORR backend requires finite positive g weights"
        )
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RobertValidationError("g_weights must have a finite positive sum")
    threshold = float(cutoff)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise RobertValidationError(
            "random-overlap cutoff must be finite and non-negative"
        )
    return tau, np.ascontiguousarray(weights / total), threshold


def _jax_runtime():
    global _COMPILED_KERNEL
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise RobertConfigError(
            "the JAX random-overlap backend requires robert-exoplanets[jax]"
        ) from exc
    if not jax.config.x64_enabled:
        raise RobertConfigError(
            "the science-grade JAX backend requires float64; set JAX_ENABLE_X64=1 "
            "before starting Python"
        )
    if _COMPILED_KERNEL is None:
        _COMPILED_KERNEL = _build_compiled_kernel(jax, jnp)
    return jax, jnp, _COMPILED_KERNEL


def _build_compiled_kernel(jax, jnp):
    lax = jax.lax

    def conservative_rebin(values, source_weights, target_weights):
        order = jnp.argsort(values)
        sorted_values = values[order]
        sorted_weights = source_weights[order]
        source_edges = jnp.concatenate(
            (jnp.zeros(1, dtype=values.dtype), jnp.cumsum(sorted_weights))
        )
        source_edges = source_edges.at[-1].set(1.0)
        cumulative_integral = jnp.concatenate(
            (
                jnp.zeros(1, dtype=values.dtype),
                jnp.cumsum(sorted_values * sorted_weights),
            )
        )
        target_edges = jnp.concatenate(
            (jnp.zeros(1, dtype=values.dtype), jnp.cumsum(target_weights))
        )
        target_edges = target_edges.at[-1].set(1.0)
        integral_at_edges = jnp.interp(
            target_edges,
            source_edges,
            cumulative_integral,
        )
        return jnp.diff(integral_at_edges) / target_weights

    def combine_two(left, right, weights, pair_weights):
        pair_values = (left[:, None] + right[None, :]).reshape(-1)
        return conservative_rebin(pair_values, pair_weights, weights)

    def mix_point(tau_by_species, weights, pair_weights, cutoff):
        n_g = tau_by_species.shape[-1]
        initial = (
            jnp.zeros(n_g, dtype=tau_by_species.dtype),
            jnp.asarray(False),
        )

        def scan_species(carry, next_tau):
            combined, started = carry
            active = jnp.max(next_tau) >= cutoff

            def use_species(_):
                next_combined = lax.cond(
                    started,
                    lambda __: combine_two(
                        combined, next_tau, weights, pair_weights
                    ),
                    lambda __: next_tau,
                    operand=None,
                )
                return next_combined, jnp.asarray(True)

            updated = lax.cond(
                active,
                use_species,
                lambda _: (combined, started),
                operand=None,
            )
            return updated, None

        (combined, started), _ = lax.scan(
            scan_species,
            initial,
            tau_by_species,
        )
        return jnp.where(started, combined, jnp.zeros_like(combined))

    def kernel(tau, weights, cutoff):
        n_species, n_layers, n_spectral, n_g = tau.shape
        if n_species == 1:
            return tau[0]
        pair_weights = (weights[:, None] * weights[None, :]).reshape(-1)
        points = jnp.transpose(tau, (1, 2, 0, 3)).reshape(
            n_layers * n_spectral, tau.shape[0], n_g
        )
        mixed = jax.vmap(mix_point, in_axes=(0, None, None, None))(
            points,
            weights,
            pair_weights,
            cutoff,
        )
        return mixed.reshape(n_layers, n_spectral, n_g)

    return jax.jit(kernel)


__all__ = [
    "jax_random_overlap_device_info",
    "jax_random_overlap_species_tau",
]
