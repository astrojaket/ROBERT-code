"""Parity tests for the independent real-pair Metal Mie implementation."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import RefractiveIndexSpectrum, SpectralGrid
from robert_exoplanets.metal.mie import (
    cloud_extinction_tau,
    cloud_extinction_tau_device,
    lognormal_mie_optics as metal_optics,
    lognormal_mie_optics_device,
    mie_efficiencies_and_moments,
    mie_efficiencies_and_moments_device,
)
from robert_exoplanets.rt.mie import (
    lognormal_mie_optics,
    mie_efficiencies,
    mie_phase_function_moments,
)


@pytest.fixture(autouse=True)
def _cpu_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Algorithmic parity is portable; Metal execution needs a visible Apple GPU."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    pytest.importorskip("jax")


@pytest.mark.parametrize(
    "x,index",
    [(1e-4, 1.5 + 0.01j), (1.0, 1.5 + 0j), (5.0, 1.6 + 0.1j), (20.0, 1.3 + 0.02j)],
)
def test_real_pair_mie_matches_cpu(x: float, index: complex) -> None:
    qext, qsca, g, moments = mie_efficiencies_and_moments(
        [x], [index.real], [index.imag], maximum_order=40, platform="cpu"
    )
    np.testing.assert_allclose(
        [qext[0], qsca[0], g[0]], mie_efficiencies(x, index), rtol=3e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        moments[0], mie_phase_function_moments(x, index), rtol=5e-5, atol=3e-6
    )


def test_downward_capacity_matches_cpu_for_high_index_direct_nk_case() -> None:
    """The continued fraction must cover ``abs(m*x)``, not just n_stop."""
    x = 50.0
    index = 4.0 + 3.0j
    qext, qsca, g, moments = mie_efficiencies_and_moments(
        [x], [index.real], [index.imag], maximum_order=80, platform="cpu"
    )
    expected = mie_efficiencies(x, index)
    np.testing.assert_allclose([qext[0], qsca[0], g[0]], expected, rtol=8e-5, atol=3e-6)
    np.testing.assert_allclose(
        moments[0], mie_phase_function_moments(x, index), rtol=1e-4, atol=2e-5
    )


def test_resonance_sweep_matches_cpu_reference() -> None:
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    qext, qsca, g, moments = mie_efficiencies_and_moments(
        x, 1.33, 0.0, maximum_order=50, platform="cpu"
    )
    reference = np.array([mie_efficiencies(value, 1.33 + 0.0j) for value in x])
    reference_moments = np.array(
        [mie_phase_function_moments(value, 1.33 + 0.0j) for value in x]
    )
    np.testing.assert_allclose(
        np.column_stack((qext, qsca)), reference[:, :2], rtol=6e-5, atol=3e-6
    )
    # At the smallest, nearly isotropic point g is ~1e-3; float32 angular
    # quadrature has an absolute (rather than useful relative) error there.
    np.testing.assert_allclose(g, reference[:, 2], rtol=6e-5, atol=8e-5)
    # Tiny odd/high Legendre coefficients in the Rayleigh-adjacent part of the
    # sweep are cancellation limited in float32; their absolute error remains
    # below 3.4e-4 while the physically material moments meet the relative gate.
    np.testing.assert_allclose(moments, reference_moments, rtol=1e-4, atol=4e-4)


def test_lognormal_optics_and_cloud_tau_match_cpu() -> None:
    wave = np.array([1.0, 2.0, 4.0])
    grid = SpectralGrid.from_array(wave, unit="micron", role="opacity")
    index = RefractiveIndexSpectrum(
        wavelength_micron=[0.8, 5.0],
        real_index=[1.6, 1.6],
        imaginary_index=[0.02, 0.02],
    )
    cpu = lognormal_mie_optics(
        index,
        grid,
        effective_radius_micron=0.3,
        geometric_stddev=1.6,
        particle_density_kg_m3=3200.0,
        quadrature_points=12,
    )
    metal = metal_optics(
        wave,
        np.full(3, 1.6),
        np.full(3, 0.02),
        effective_radius_micron=0.3,
        geometric_stddev=1.6,
        particle_density_kg_m3=3200.0,
        quadrature_points=12,
        maximum_order=80,
        platform="cpu",
    )
    np.testing.assert_allclose(
        metal["mass_extinction_m2_kg"], cpu.mass_extinction_m2_kg, rtol=8e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        metal["mass_scattering_m2_kg"], cpu.mass_scattering_m2_kg, rtol=8e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        metal["phase_function_moments"],
        cpu.phase_function_moments,
        rtol=1.5e-4,
        atol=1e-5,
    )
    tau = cloud_extinction_tau(
        [100.0, 200.0], 10.0, [1e-4, 2e-4], metal["mass_extinction_m2_kg"]
    )
    expected = (
        np.array([100e-4 / 10.0, 200 * 2e-4 / 10.0])[:, None]
        * cpu.mass_extinction_m2_kg
    )
    np.testing.assert_allclose(tau, expected, rtol=8e-5, atol=2e-5)


def test_device_lognormal_optics_can_be_part_of_an_outer_jit() -> None:
    import jax
    import jax.numpy as jnp

    nodes, weights = np.polynomial.legendre.leggauss(8)
    angular_mu, angular_weight = np.polynomial.legendre.leggauss(87)

    @jax.jit
    def outer(radius):
        result = lognormal_mie_optics_device(
            jnp.array([1.0, 2.0, 4.0]),
            jnp.array([1.6, 1.6, 1.6]),
            jnp.array([0.02, 0.02, 0.02]),
            radius,
            jnp.float32(1.4),
            jnp.float32(3200.0),
            jnp.asarray(nodes, dtype=jnp.float32),
            jnp.asarray(weights, dtype=jnp.float32),
            jnp.asarray(angular_mu, dtype=jnp.float32),
            jnp.asarray(angular_weight, dtype=jnp.float32),
            maximum_order=80,
            maximum_downward_order=95,
        )
        tau = cloud_extinction_tau_device(
            jnp.array([100.0, 200.0]),
            jnp.float32(10.0),
            jnp.array([1.0e-4, 2.0e-4]),
            result["mass_extinction_m2_kg"],
        )
        return result["mass_extinction_m2_kg"], result["phase_function_moments"], tau

    extinction, moments, tau = outer(jnp.float32(0.3))
    assert isinstance(extinction, jax.Array)
    assert isinstance(moments, jax.Array)
    assert isinstance(tau, jax.Array)
    assert extinction.shape == (3,)
    assert moments.shape == (5, 3)
    assert tau.shape == (2, 3)


def test_device_mie_marks_insufficient_static_capacity_invalid() -> None:
    import jax.numpy as jnp

    mu, weight = np.polynomial.legendre.leggauss(24)
    result = mie_efficiencies_and_moments_device(
        jnp.array([50.0]),
        jnp.array([1.5]),
        jnp.array([0.1]),
        jnp.asarray(mu, dtype=jnp.float32),
        jnp.asarray(weight, dtype=jnp.float32),
        maximum_order=10,
        maximum_downward_order=80,
    )
    assert all(np.isnan(np.asarray(value)).all() for value in result)


def test_device_cloud_parameters_mark_nonphysical_states_invalid() -> None:
    import jax.numpy as jnp

    mu, weight = np.polynomial.legendre.leggauss(24)
    optics = lognormal_mie_optics_device(
        jnp.array([1.0]),
        jnp.array([1.5]),
        jnp.array([0.01]),
        jnp.float32(-0.3),
        jnp.float32(0.9),
        jnp.float32(3200.0),
        jnp.array([0.0]),
        jnp.array([1.0]),
        jnp.asarray(mu, dtype=jnp.float32),
        jnp.asarray(weight, dtype=jnp.float32),
        maximum_order=30,
        maximum_downward_order=45,
    )
    assert np.isnan(np.asarray(optics["mass_extinction_m2_kg"])).all()
    invalid_tau = cloud_extinction_tau_device(
        jnp.array([100.0]),
        jnp.float32(-1.0),
        jnp.array([1.2]),
        jnp.array([2.0]),
    )
    assert np.isnan(np.asarray(invalid_tau)).all()
