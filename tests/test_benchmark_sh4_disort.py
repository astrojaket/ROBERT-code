"""Tests for the optional SH4 versus PythonicDISORT benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets.rt.sh4 import solve_thermal_sh4
from scripts.benchmark_sh4_disort import (
    ThermalScatteringCase,
    default_cases,
    robert_phase_moments,
    thermal_profile,
    unweighted_legendre_coefficients,
)


def test_phase_moments_use_unweighted_legendre_coefficients() -> None:
    case = next(case for case in default_cases() if case.phase_model == "henyey_greenstein")
    beta = unweighted_legendre_coefficients(case, 5)
    chi = robert_phase_moments(case)

    np.testing.assert_allclose(beta, 0.5 ** np.arange(5))
    np.testing.assert_allclose(chi, (2 * np.arange(4) + 1) * beta[:4])


def test_rayleigh_moments_match_the_declared_phase_function() -> None:
    case = next(case for case in default_cases() if case.phase_model == "rayleigh")

    np.testing.assert_array_equal(
        unweighted_legendre_coefficients(case, 5),
        np.array([1.0, 0.0, 0.1, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        robert_phase_moments(case),
        np.array([1.0, 0.0, 0.5, 0.0]),
    )


def test_isothermal_pure_absorption_has_blackbody_top_flux() -> None:
    case = ThermalScatteringCase(
        name="isothermal_absorption",
        layer_optical_depths=(0.2, 0.5, 1.3),
        single_scattering_albedo=0.0,
        phase_model="henyey_greenstein",
        phase_parameters=(0.0,),
        planck_top=1.0,
        planck_bottom=1.0,
    )
    tau_edges, planck_levels, _ = thermal_profile(case)
    mu = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))
    weights = np.full(2, 0.5)
    tau = np.asarray(case.layer_optical_depths)[:, None, None]
    omega = np.zeros_like(tau)
    asymmetry = np.zeros_like(tau)

    result = solve_thermal_sh4(
        tau,
        omega,
        asymmetry,
        planck_levels[:, None],
        mu,
        bottom_planck_radiance=np.array([case.planck_bottom]),
        delta_m=False,
        boundary_backend="scipy",
    )
    top_flux = 2.0 * np.pi * np.dot(weights * mu, result.point_radiance[:, 0, 0])

    assert tau_edges[-1] == pytest.approx(2.0)
    assert top_flux == pytest.approx(np.pi * case.planck_bottom, rel=1.0e-7)


def test_synthetic_cases_are_non_isothermal_and_positive() -> None:
    for case in default_cases():
        _, levels, slope = thermal_profile(case)

        assert levels[0] == pytest.approx(case.planck_top)
        assert levels[-1] == pytest.approx(case.planck_bottom)
        assert slope > 0.0
        assert np.all(levels > 0.0)


def test_strong_forward_case_is_not_softened() -> None:
    case = next(
        case
        for case in default_cases()
        if case.name == "strong_forward_hg_g090_omega095"
    )

    assert case.single_scattering_albedo == 0.95
    assert case.phase_parameters == (0.90,)


def test_pythonicdisort_stream_comparison_is_finite_when_installed() -> None:
    pytest.importorskip("PythonicDISORT")
    from scripts.benchmark_sh4_disort import (
        _pythonicdisort_modules,
        benchmark_case,
    )

    pydisort, subroutines, version = _pythonicdisort_modules()
    assert version == "1.8"
    report = benchmark_case(
        next(case for case in default_cases() if case.name == "henyey_greenstein_g050"),
        pydisort=pydisort,
        quadrature=subroutines.Gauss_Legendre_quad,
        warmups=0,
        samples=1,
    )

    flux_16 = report["pythonicdisort"]["16"]["upward_flux_top"]
    flux_32 = report["pythonicdisort"]["32"]["upward_flux_top"]
    flux_64 = report["pythonicdisort"]["64"]["upward_flux_top"]
    assert np.all(np.isfinite([flux_16, flux_32, flux_64]))
    assert flux_16 > 0.0 and flux_32 > 0.0 and flux_64 > 0.0
    assert report["stream_convergence"]["relative_change_32_to_64"] < 1.0e-4
