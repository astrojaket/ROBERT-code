"""Tests for v0.2 core domain objects."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import Planet, PressureGrid, SpectralGrid, Spectrum, Star
from robert_exoplanets.core import RobertValidationError


def test_pressure_grid_logspace_builds_layer_centers() -> None:
    grid = PressureGrid.logspace(1.0e-6, 1.0e2, n_layers=4)

    assert grid.n_layers == 4
    assert grid.unit == "bar"
    assert grid.orientation == "increasing"
    np.testing.assert_allclose(grid.centers, np.sqrt(grid.edges[:-1] * grid.edges[1:]))


def test_pressure_grid_from_log_centers_includes_requested_endpoints() -> None:
    grid = PressureGrid.from_log_centers(100.0, 1.0e-6, n_layers=100, unit="bar")

    assert grid.n_layers == 100
    assert grid.orientation == "decreasing"
    assert grid.centers[0] == pytest.approx(100.0)
    assert grid.centers[-1] == pytest.approx(1.0e-6)
    np.testing.assert_allclose(np.diff(np.log10(grid.centers)), -8.0 / 99.0)


def test_pressure_grid_rejects_non_positive_pressures() -> None:
    with pytest.raises(RobertValidationError, match="positive"):
        PressureGrid(edges=[1.0, 0.0], centers=[0.5])


def test_pressure_grid_requires_centres_inside_matching_layers() -> None:
    with pytest.raises(RobertValidationError, match="strictly inside"):
        PressureGrid(edges=[1.0, 2.0, 3.0], centers=[1.5, 3.0])

    with pytest.raises(RobertValidationError, match="same orientation"):
        PressureGrid(edges=[1.0, 2.0, 3.0], centers=[2.5, 1.5])


def test_spectral_grid_requires_monotonic_values() -> None:
    with pytest.raises(RobertValidationError, match="strictly monotonic"):
        SpectralGrid.from_array([1.0, 2.0, 1.5])


def test_spectrum_values_match_grid_shape() -> None:
    grid = SpectralGrid.from_array([5.0, 6.0, 7.0], role="observed")
    spectrum = Spectrum(
        spectral_grid=grid,
        values=[1.0e-4, 1.1e-4, 1.2e-4],
        unit="eclipse_depth",
        observable="eclipse_depth",
    )

    assert spectrum.spectral_grid is grid
    assert spectrum.values.flags.writeable is False


def test_planet_requires_gravity_or_mass_and_radius() -> None:
    planet = Planet(name="WASP-43b", radius_m=7.0e7, gravity_m_s2=45.0)

    assert planet.name == "WASP-43b"
    assert planet.has_direct_gravity is True

    with pytest.raises(RobertValidationError, match="requires gravity"):
        Planet(name="No-gravity")

    with pytest.raises(RobertValidationError, match="finite and positive"):
        Planet(name="Invalid", radius_m=np.nan, gravity_m_s2=10.0)


def test_star_accepts_optional_spectrum() -> None:
    star = Star(name="WASP-43", radius_m=4.6e8, effective_temperature_k=4500.0)

    assert star.name == "WASP-43"
    assert star.spectrum is None
