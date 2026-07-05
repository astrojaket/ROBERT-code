"""Tests for radiative-transfer disc geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    DiscGeometry,
    DiscPoint,
    disk_average_quadrature,
    gauss_legendre_disk_geometry,
    geometry_from_emission_angles,
    lobatto_phase_geometry,
    normal_emission_geometry,
)
from robert_exoplanets.core import RobertValidationError


def test_normal_emission_geometry_has_unit_mu_and_weight() -> None:
    geometry = normal_emission_geometry()

    assert geometry.name == "normal_emission"
    assert geometry.n_points == 1
    np.testing.assert_allclose(geometry.emission_angle_cosines, [1.0])
    np.testing.assert_allclose(geometry.emission_angle_weights, [1.0])
    np.testing.assert_allclose(geometry.emission_angles_deg, [0.0])


def test_geometry_from_emission_angles_normalizes_weights() -> None:
    geometry = geometry_from_emission_angles(
        [0.25, 0.75],
        [2.0, 6.0],
        name="test_geometry",
    )

    assert geometry.name == "test_geometry"
    np.testing.assert_allclose(geometry.emission_angle_cosines, [0.25, 0.75])
    np.testing.assert_allclose(geometry.emission_angle_weights, [0.25, 0.75])


def test_gauss_legendre_disk_geometry_matches_legacy_quadrature() -> None:
    mu, weights = disk_average_quadrature(5)
    geometry = gauss_legendre_disk_geometry(5)

    np.testing.assert_allclose(geometry.emission_angle_cosines, mu)
    np.testing.assert_allclose(geometry.emission_angle_weights, weights)
    np.testing.assert_allclose(np.sum(geometry.emission_angle_weights), 1.0)


def test_lobatto_phase_geometry_provides_scattering_angles() -> None:
    geometry = lobatto_phase_geometry(phase_angle_deg=180.0, n_mu=4)

    assert geometry.name == "lobatto_phase_disc"
    assert geometry.quadrature == "lobatto_projected_disc"
    assert geometry.phase_angle_deg == 180.0
    assert geometry.n_points >= 4
    np.testing.assert_allclose(np.sum(geometry.emission_angle_weights), 1.0)
    assert np.all(geometry.emission_angle_cosines > 0.0)
    assert np.all(geometry.emission_angle_cosines <= 1.0)
    assert np.all(np.isfinite(geometry.latitudes_deg))
    assert np.all(np.isfinite(geometry.longitudes_deg))
    assert np.all(np.isfinite(geometry.stellar_zenith_deg))
    assert np.all(np.isfinite(geometry.stellar_azimuth_deg))
    assert np.all(geometry.stellar_mu >= -1.0)
    assert np.all(geometry.stellar_mu <= 1.0)


def test_disc_geometry_rejects_zero_total_weight() -> None:
    with pytest.raises(RobertValidationError, match="positive sum"):
        DiscGeometry(points=(DiscPoint(emission_mu=1.0, weight=0.0),))


def test_lobatto_phase_geometry_rejects_unsupported_nmu() -> None:
    with pytest.raises(RobertValidationError, match="2 <= n_mu <= 5"):
        lobatto_phase_geometry(phase_angle_deg=0.0, n_mu=6)
