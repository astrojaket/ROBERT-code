"""Physics and public-API tests for the P3/SH4 thermal solver."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets.rt.sh4 import (
    henyey_greenstein_moments,
    sh4_spectrum_backend_name,
    solve_thermal_sh4,
    solve_thermal_sh4_spectrum,
)
from robert_exoplanets.core import RobertValidationError


def test_henyey_greenstein_phase_moments_follow_legendre_definition() -> None:
    asymmetry = np.array([[[0.6]]])

    moments = henyey_greenstein_moments(asymmetry)

    expected = np.array([1.0, 3.0 * 0.6, 5.0 * 0.6**2, 7.0 * 0.6**3])
    np.testing.assert_allclose(moments[:, 0, 0, 0], expected)


def test_absorbing_layer_matches_rutten_formal_solution() -> None:
    tau = 0.7
    mu = 0.43
    planck_top = 2.0
    planck_bottom = 5.0
    bottom_boundary = 7.0

    result = solve_thermal_sh4(
        np.array([[[tau]]]),
        np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1)),
        np.array([[planck_top], [planck_bottom]]),
        np.array([mu]),
        bottom_planck_radiance=np.array([bottom_boundary]),
        delta_m=False,
        source_quadrature_order=16,
    )

    slope = (planck_bottom - planck_top) / tau
    transmission = np.exp(-tau / mu)
    atmospheric = planck_top * (1.0 - transmission) + slope * (
        mu - (tau + mu) * transmission
    )
    expected = bottom_boundary * transmission + atmospheric
    np.testing.assert_allclose(result.point_radiance[0, 0, 0], expected, rtol=2.0e-13)


def test_delta_m_preserves_absorption_optical_depth_and_phase_normalization() -> None:
    tau = np.array([[[0.8]]])
    omega = np.array([[[0.93]]])
    result = solve_thermal_sh4(
        tau,
        omega,
        np.array([[[0.8]]]),
        np.array([[2.0], [3.0]]),
        np.array([0.4]),
        bottom_planck_radiance=np.array([4.0]),
        delta_m=True,
    )

    np.testing.assert_allclose(
        result.scaled_extinction_tau * (1.0 - result.scaled_single_scattering_albedo),
        tau * (1.0 - omega),
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(result.phase_function_moments[0], 1.0)
    assert result.delta_m_applied is True


def test_delta_m_accepts_explicit_forward_fraction_for_supplied_moments() -> None:
    tau = np.array([[[1.2]]])
    omega = np.array([[[0.8]]])
    moments = np.array([1.0, 1.5, 1.8, 1.7])[:, None, None, None]
    forward_fraction = np.array([[[0.2]]])

    result = solve_thermal_sh4(
        tau,
        omega,
        np.array([[[0.5]]]),
        np.array([[2.0], [3.0]]),
        np.array([0.5]),
        bottom_planck_radiance=np.array([4.0]),
        phase_function_moments=moments,
        delta_m_forward_fraction=forward_fraction,
        delta_m=True,
    )

    np.testing.assert_allclose(result.scaled_extinction_tau, tau * (1.0 - omega * 0.2))
    np.testing.assert_allclose(result.phase_function_moments[0], 1.0)


def test_sh4_contributions_reconstruct_emergent_intensity() -> None:
    result = solve_thermal_sh4(
        np.array([[[0.2]], [[0.8]]]),
        np.array([[[0.7]], [[0.4]]]),
        np.array([[[0.3]], [[-0.1]]]),
        np.array([[1.0], [2.0], [4.0]]),
        np.array([0.25, 0.8]),
        bottom_planck_radiance=np.array([5.0]),
        delta_m=False,
    )

    reconstructed = (
        np.sum(result.point_layer_contribution_radiance, axis=1)
        + result.point_bottom_contribution_radiance
    )
    np.testing.assert_allclose(result.point_radiance, reconstructed, rtol=2.0e-13)
    np.testing.assert_allclose(
        _HALF_RANGE_TOP @ result.moment_levels[0, 0, 0],
        0.0,
        atol=2.0e-13,
    )


def test_exactly_conservative_layer_uses_zero_thermal_emissivity_limit() -> None:
    common = {
        "extinction_tau": np.array([[[2.0]]]),
        "single_scattering_albedo": np.ones((1, 1, 1)),
        "asymmetry_factor": np.array([[[0.6]]]),
        "emission_angle_cosines": np.array([0.2, 0.8]),
        "bottom_planck_radiance": np.array([5.0]),
        "delta_m": False,
    }
    cool = solve_thermal_sh4(
        level_planck_radiance=np.array([[1.0], [2.0]]),
        **common,
    )
    hot = solve_thermal_sh4(
        level_planck_radiance=np.array([[100.0], [200.0]]),
        **common,
    )

    np.testing.assert_allclose(cool.point_radiance, hot.point_radiance, rtol=2.0e-13)
    assert np.all(np.isfinite(cool.point_radiance))
    assert np.all(cool.point_radiance > 0.0)


def test_spectrum_only_numpy_matches_weighted_diagnostic_result() -> None:
    tau, omega, asymmetry, planck, mu, angle_weights, g_weights = _spectrum_case()
    diagnostic = solve_thermal_sh4(
        tau,
        omega,
        asymmetry,
        planck,
        mu,
        bottom_planck_radiance=np.array([5.0, 6.0]),
        delta_m=False,
    )
    expected = np.einsum(
        "a,asg,g->s",
        angle_weights,
        np.sum(diagnostic.point_layer_contribution_radiance, axis=1)
        + diagnostic.point_bottom_contribution_radiance,
        g_weights,
    )

    spectrum = solve_thermal_sh4_spectrum(
        tau,
        omega,
        asymmetry,
        planck,
        mu,
        angle_weights,
        g_weights,
        bottom_planck_radiance=np.array([5.0, 6.0]),
        delta_m=False,
        backend="numpy",
    )

    np.testing.assert_allclose(spectrum.radiance, expected, rtol=2.0e-13)
    assert spectrum.backend == "numpy"
    assert not hasattr(spectrum, "point_layer_contribution_radiance")


def test_spectrum_only_numba_matches_numpy_for_mixed_scattering() -> None:
    pytest.importorskip("numba")
    tau, omega, asymmetry, planck, mu, angle_weights, g_weights = _spectrum_case()
    supplied_moments = henyey_greenstein_moments(asymmetry)
    supplied_moments[2] *= 0.85
    supplied_moments[3] *= 0.7
    common = {
        "bottom_planck_radiance": np.array([5.0, 6.0]),
        "phase_function_moments": supplied_moments,
        "delta_m_forward_fraction": np.full_like(tau, 0.08),
        "delta_m": True,
    }

    reference = solve_thermal_sh4_spectrum(
        tau,
        omega,
        asymmetry,
        planck,
        mu,
        angle_weights,
        g_weights,
        backend="numpy",
        **common,
    )
    accelerated = solve_thermal_sh4_spectrum(
        tau,
        omega,
        asymmetry,
        planck,
        mu,
        angle_weights,
        g_weights,
        backend="numba",
        **common,
    )

    np.testing.assert_allclose(
        accelerated.radiance,
        reference.radiance,
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    assert accelerated.backend == "numba"


def test_spectrum_only_numba_preserves_exactly_conservative_limit() -> None:
    pytest.importorskip("numba")
    tau = np.array([[[0.2, 0.8], [1.5, 2.0]]])
    omega = np.ones_like(tau)
    asymmetry = np.full_like(tau, 0.6)
    planck = np.array([[1.0, 2.0], [100.0, 200.0]])
    common = {
        "bottom_planck_radiance": np.array([5.0, 6.0]),
        "delta_m": False,
    }
    positional = (
        tau,
        omega,
        asymmetry,
        planck,
        np.array([0.2, 0.8]),
        np.array([0.4, 0.6]),
        np.array([0.3, 0.7]),
    )

    reference = solve_thermal_sh4_spectrum(*positional, backend="numpy", **common)
    accelerated = solve_thermal_sh4_spectrum(*positional, backend="numba", **common)

    np.testing.assert_allclose(
        accelerated.radiance,
        reference.radiance,
        rtol=2.0e-11,
        atol=2.0e-13,
    )


def test_spectrum_backend_validation_is_explicit() -> None:
    assert sh4_spectrum_backend_name("numpy") == "numpy"
    with pytest.raises(RobertValidationError, match="SH4 spectrum backend"):
        sh4_spectrum_backend_name("fast-ish")


def _spectrum_case():
    tau = np.array(
        [
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.8, 0.6], [0.7, 0.9]],
        ]
    )
    omega = np.array(
        [
            [[0.7, 0.8], [0.5, 0.6]],
            [[0.4, 0.9], [0.3, 0.75]],
        ]
    )
    asymmetry = np.array(
        [
            [[0.3, 0.6], [-0.1, 0.2]],
            [[0.0, 0.7], [0.4, -0.2]],
        ]
    )
    planck = np.array([[1.0, 1.5], [2.0, 2.5], [4.0, 4.5]])
    return (
        tau,
        omega,
        asymmetry,
        planck,
        np.array([0.25, 0.8]),
        np.array([0.4, 0.6]),
        np.array([0.3, 0.7]),
    )


_HALF_RANGE_TOP = np.pi * np.array(
    [[1.0, -2.0, 5.0 / 4.0, 0.0], [-1.0 / 4.0, 0.0, 5.0 / 4.0, -2.0]]
)
