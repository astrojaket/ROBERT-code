"""Unified-memory safety contracts for the optional Metal backend."""

from __future__ import annotations

import pytest

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.metal import (
    MetalResourcePolicy,
    MetalRuntime,
    compile_metal_batched_likelihood,
)


def test_resource_policy_rejects_unsafe_production_batches() -> None:
    policy = MetalResourcePolicy(
        total_memory_bytes=24 * 1024**3,
        maximum_fraction=0.08,
        absolute_maximum_bytes=1024**3,
    )
    single = policy.estimate_rorr_bytes(
        species=3, layers=80, spectral=128, g_ordinates=8, batch=1
    )
    batch_32 = policy.estimate_rorr_bytes(
        species=3, layers=80, spectral=128, g_ordinates=8, batch=32
    )
    policy.require_safe("single", single)
    with pytest.raises(RobertValidationError, match="batch 8"):
        policy.require_safe_batch("batch 8", 8)
    with pytest.raises(RobertValidationError, match="batch 32"):
        policy.require_safe_batch("batch 32", 32)
    with pytest.raises(RobertValidationError, match="refusing unsafe"):
        policy.require_safe("batch 32", batch_32)


def test_resource_estimates_are_monotonic_and_validate_shapes() -> None:
    policy = MetalResourcePolicy.laptop_safe()
    one = policy.estimate_sh4_bytes(layers=20, spectral=4, g_ordinates=4, batch=1)
    eight = policy.estimate_sh4_bytes(layers=20, spectral=4, g_ordinates=4, batch=8)
    assert eight == 8 * one
    sampled_small = policy.estimate_opacity_sampling_bytes(
        species=2,
        pressure_grid=4,
        temperature_grid=3,
        layers=20,
        spectral=100,
        disc_points=4,
    )
    sampled_large = policy.estimate_opacity_sampling_bytes(
        species=2,
        pressure_grid=4,
        temperature_grid=3,
        layers=20,
        spectral=1000,
        disc_points=4,
    )
    assert sampled_large == 10 * sampled_small
    with pytest.raises(RobertValidationError, match="positive"):
        policy.estimate_rorr_bytes(species=3, layers=0, spectral=10, g_ordinates=8)
    with pytest.raises(RobertValidationError, match="positive"):
        policy.estimate_opacity_sampling_bytes(
            species=2,
            pressure_grid=4,
            temperature_grid=3,
            layers=0,
            spectral=100,
            disc_points=4,
        )


def test_unsafe_batch_is_refused_before_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    runtime = MetalRuntime(jax=jax, jnp=jnp, device=object())
    called = False

    def forbidden_jit(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("jit must not be reached")

    monkeypatch.setattr(jax, "jit", forbidden_jit)
    with pytest.raises(RobertValidationError, match="batch 8"):
        compile_metal_batched_likelihood(
            lambda vector: vector[0], runtime, batch_size=8
        )
    assert not called
