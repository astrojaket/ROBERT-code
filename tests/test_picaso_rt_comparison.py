"""Tests for the controlled external PICASO RT comparison harness."""

from __future__ import annotations

import numpy as np

from examples.compare_grey_cloud_rt_picaso import (
    GreyRTCase,
    _comparison_report,
    _disk_quadrature,
)


def test_disk_quadrature_normalizes_constant_intensity() -> None:
    mu, weights = _disk_quadrature(6)

    assert np.all((mu > 0.0) & (mu < 1.0))
    np.testing.assert_allclose(np.sum(weights), 1.0)
    np.testing.assert_allclose(np.dot(weights, np.ones_like(mu)), 1.0)


def test_comparison_report_fails_physics_outside_tolerance() -> None:
    case = GreyRTCase(
        name="known discrepancy",
        temperature_level_k=np.array([800.0, 1000.0]),
        extinction_tau=np.array([[1.0]]),
        single_scattering_albedo=0.9,
        asymmetry_factor=0.0,
        relative_tolerance=0.05,
    )
    picaso_point = np.array([[10.0], [20.0]])
    robert_point = 0.5 * picaso_point
    weights = np.array([0.4, 0.6])
    picaso_disk = np.tensordot(weights, picaso_point, axes=(0, 0))
    robert_disk = np.tensordot(weights, robert_point, axes=(0, 0))

    report = _comparison_report(
        case,
        robert_point,
        picaso_point,
        robert_disk,
        picaso_disk,
    )

    assert report["passed"] is False
    assert report["max_abs_relative_difference"] == 0.5
