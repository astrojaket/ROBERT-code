"""Physics tests for the coupled thermal two-stream reference solver."""

from __future__ import annotations

import numpy as np

from robert_exoplanets.rt.toon import (
    _solve_flux_column,
    _solve_flux_columns,
    solve_thermal_two_stream,
)


def test_absorbing_layer_matches_rutten_formal_solution() -> None:
    tau = 0.7
    mu = 0.43
    planck_top = 2.0
    planck_bottom = 5.0
    bottom_boundary = 7.0
    result = solve_thermal_two_stream(
        np.array([[[tau]]]),
        np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1)),
        np.array([[planck_top], [planck_bottom]]),
        np.array([mu]),
        bottom_planck_radiance=np.array([bottom_boundary]),
        source_quadrature_order=16,
    )

    slope = (planck_bottom - planck_top) / tau
    transmission = np.exp(-tau / mu)
    atmospheric = (
        planck_top * (1.0 - transmission)
        + slope * (mu - (tau + mu) * transmission)
    )
    expected = bottom_boundary * transmission + atmospheric
    np.testing.assert_allclose(result.point_radiance[0, 0, 0], expected, rtol=2.0e-13)


def test_contributions_reconstruct_emergent_intensity() -> None:
    result = solve_thermal_two_stream(
        np.array([[[0.2]], [[0.8]]]),
        np.array([[[0.7]], [[0.4]]]),
        np.array([[[0.3]], [[-0.1]]]),
        np.array([[1.0], [2.0], [4.0]]),
        np.array([0.25, 0.8]),
        bottom_planck_radiance=np.array([5.0]),
    )

    reconstructed = (
        np.sum(result.point_layer_contribution_radiance, axis=1)
        + result.point_bottom_contribution_radiance
    )
    np.testing.assert_allclose(result.point_radiance, reconstructed, rtol=2.0e-13)
    np.testing.assert_allclose(result.downward_flux_levels[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.upward_flux_levels[-1], np.pi * 5.0)


def test_rutten_isothermal_scattering_flux_suppression() -> None:
    planck = 3.0
    emissivities = (1.0, 0.5, 0.1, 0.01)
    fluxes = []
    for emissivity in emissivities:
        result = solve_thermal_two_stream(
            np.array([[[100.0]]]),
            np.array([[[1.0 - emissivity]]]),
            np.zeros((1, 1, 1)),
            np.array([[planck], [planck]]),
            np.array([0.5]),
            bottom_planck_radiance=np.array([planck]),
        )
        fluxes.append(float(result.upward_flux_levels[0, 0, 0]))

    normalized = np.asarray(fluxes) / fluxes[0]
    expected = np.array(
        [2.0 * np.sqrt(value) / (1.0 + np.sqrt(value)) for value in emissivities]
    )
    np.testing.assert_allclose(normalized, expected, rtol=2.0e-12)


def test_batched_flux_solver_matches_scalar_reference() -> None:
    rng = np.random.default_rng(481)
    tau = rng.uniform(0.01, 1.5, size=(4, 3, 2))
    omega = rng.uniform(0.0, 0.95, size=tau.shape)
    asymmetry = rng.uniform(-0.2, 0.8, size=tau.shape)
    planck_levels = rng.uniform(1.0, 5.0, size=(5, 3))
    bottom_planck = rng.uniform(2.0, 6.0, size=3)

    batched = _solve_flux_columns(tau, omega, asymmetry, planck_levels, bottom_planck)

    for spectral_index in range(3):
        for g_index in range(2):
            reference = _solve_flux_column(
                tau[:, spectral_index, g_index],
                omega[:, spectral_index, g_index],
                asymmetry[:, spectral_index, g_index],
                planck_levels[:, spectral_index],
                float(bottom_planck[spectral_index]),
            )
            np.testing.assert_allclose(
                batched[:, spectral_index, g_index],
                reference,
                rtol=2.0e-12,
                atol=1.0e-12,
            )
