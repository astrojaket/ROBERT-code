"""Opt-in JAX parity test for the isolated SH4 implementation."""

from __future__ import annotations

import os

import numpy as np
import pytest

from robert_exoplanets.metal.sh4 import metal_solve_thermal_sh4_spectrum
from robert_exoplanets.rt.sh4 import solve_thermal_sh4_spectrum


@pytest.fixture(autouse=True)
def _enable_jax_x64_for_isolated_parity() -> object:
    """Make strict CPU parity independent of test collection order."""

    jax = pytest.importorskip("jax")
    was_enabled = bool(jax.config.x64_enabled)
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax
    finally:
        jax.config.update("jax_enable_x64", was_enabled)


@pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_JAX_CPU_TESTS") != "1",
    reason="set ROBERT_RUN_JAX_CPU_TESTS=1 for optional JAX CPU SH4 parity",
)
def test_metal_sh4_matches_cpu_reference_on_mixed_scattering_case() -> None:
    tau = np.array([[[0.2, 0.3], [0.4, 0.5]], [[0.8, 0.6], [0.7, 0.9]]])
    omega = np.array([[[0.7, 0.8], [0.5, 0.6]], [[0.4, 0.9], [0.3, 0.75]]])
    asymmetry = np.array([[[0.3, 0.6], [-0.1, 0.2]], [[0.0, 0.7], [0.4, -0.2]]])
    planck = np.array([[1.0, 1.5], [2.0, 2.5], [4.0, 4.5]])
    arguments = (
        tau,
        omega,
        asymmetry,
        planck,
        np.array([0.25, 0.8]),
        np.array([0.4, 0.6]),
        np.array([0.3, 0.7]),
    )
    expected = solve_thermal_sh4_spectrum(
        *arguments,
        bottom_planck_radiance=np.array([5.0, 6.0]),
        delta_m=False,
        backend="numpy",
    )
    actual = metal_solve_thermal_sh4_spectrum(
        *arguments, bottom_planck_radiance=np.array([5.0, 6.0]), delta_m=False
    )
    actual.block_until_ready()
    np.testing.assert_allclose(
        np.asarray(actual), expected.radiance, rtol=2e-10, atol=2e-12
    )


@pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_JAX_CPU_TESTS") != "1",
    reason="set ROBERT_RUN_JAX_CPU_TESTS=1 for optional JAX CPU SH4 parity",
)
@pytest.mark.parametrize("case", ("absorption", "conservative", "moments"))
def test_metal_sh4_matches_cpu_across_physical_limits(case: str) -> None:
    tau = np.array([[[0.7], [1.1]], [[0.4], [0.9]]])
    omega = np.zeros_like(tau) if case == "absorption" else np.full_like(tau, 0.995)
    if case == "conservative":
        omega[:] = 1.0
    asymmetry = np.full_like(tau, 0.45)
    planck = np.array([[1.0, 1.3], [2.0, 2.7], [4.0, 4.8]])
    common = {
        "bottom_planck_radiance": np.array([5.0, 5.4]),
        "delta_m": case != "moments",
    }
    if case == "moments":
        moments = np.stack(
            (np.ones_like(tau), 3 * asymmetry, 5 * asymmetry**2, 7 * asymmetry**3)
        )
        common["phase_function_moments"] = moments
        common["delta_m_forward_fraction"] = np.full_like(tau, 0.08)
        common["delta_m"] = True
    arguments = (
        tau,
        omega,
        asymmetry,
        planck,
        np.array([0.2, 0.8]),
        np.array([0.3, 0.7]),
        np.array([1.0]),
    )
    expected = solve_thermal_sh4_spectrum(*arguments, backend="numpy", **common)
    actual = metal_solve_thermal_sh4_spectrum(*arguments, **common)
    actual.block_until_ready()
    np.testing.assert_allclose(
        np.asarray(actual), expected.radiance, rtol=3e-10, atol=3e-12
    )


@pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_JAX_CPU_TESTS") != "1",
    reason="set ROBERT_RUN_JAX_CPU_TESTS=1 for optional JAX CPU SH4 execution",
)
def test_metal_sh4_eighty_layers_uses_compact_banded_system() -> None:
    """The 80-layer graph executes without a dense 320 by 320 allocation."""
    import jax
    import jax.numpy as jnp

    nlayer = 80
    tau = np.full((nlayer, 1, 1), 0.02)
    omega = np.full_like(tau, 0.6)
    asymmetry = np.full_like(tau, 0.3)
    planck = np.linspace(1.0, 3.0, nlayer + 1)[:, None]
    arguments = (
        tau,
        omega,
        asymmetry,
        planck,
        np.array([0.5]),
        np.array([1.0]),
        np.array([1.0]),
    )
    actual = metal_solve_thermal_sh4_spectrum(
        *arguments, bottom_planck_radiance=np.array([3.2])
    )
    actual.block_until_ready()
    assert np.all(np.isfinite(np.asarray(actual)))
    # The JAXPR must show the compact expanded LU storage shape (16, 320, 1),
    # and no 320 x 320 array shape.
    graph = str(
        jax.make_jaxpr(
            lambda t, o, a, p: metal_solve_thermal_sh4_spectrum(
                t,
                o,
                a,
                p,
                np.array([0.5]),
                np.array([1.0]),
                np.array([1.0]),
                bottom_planck_radiance=np.array([3.2]),
            )
        )(
            jnp.asarray(tau),
            jnp.asarray(omega),
            jnp.asarray(asymmetry),
            jnp.asarray(planck),
        )
    )
    assert "f64[16,320,1]" in graph
    assert "f64[320,320" not in graph
