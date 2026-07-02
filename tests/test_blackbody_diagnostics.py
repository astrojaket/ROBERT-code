"""Tests for blackbody diagnostic reference calculations."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    blackbody_eclipse_depth,
    blackbody_eclipse_depth_spectrum,
    planck_radiance_wavelength,
)
from robert_exoplanets.core import RobertValidationError


def test_planck_radiance_is_positive_and_hotter_is_brighter() -> None:
    wavelength = np.array([3.0, 5.0, 10.0])

    cool = planck_radiance_wavelength(wavelength, 1000.0)
    hot = planck_radiance_wavelength(wavelength, 1200.0)

    assert cool.shape == wavelength.shape
    assert cool.flags.writeable is False
    assert np.all(cool > 0.0)
    assert np.all(hot > cool)


def test_blackbody_eclipse_depth_matches_area_ratio_for_equal_temperatures() -> None:
    wavelength = np.array([1.0, 5.0, 10.0])

    depth = blackbody_eclipse_depth(
        wavelength,
        planet_temperature_k=1500.0,
        star_temperature_k=1500.0,
        planet_radius_m=2.0,
        star_radius_m=10.0,
    )

    np.testing.assert_allclose(depth, np.full(3, 0.04))
    assert depth.flags.writeable is False


def test_blackbody_eclipse_depth_spectrum_uses_eclipse_depth_units() -> None:
    spectrum = blackbody_eclipse_depth_spectrum(
        [4.0, 5.0, 6.0],
        planet_temperature_k=1800.0,
        star_temperature_k=6200.0,
        planet_radius_m=1.2e8,
        star_radius_m=8.0e8,
    )

    assert spectrum.unit == "eclipse_depth"
    assert spectrum.observable == "eclipse_depth"
    assert spectrum.spectral_grid.role == "reference"
    assert spectrum.metadata["reference"] == "blackbody"


def test_blackbody_helpers_reject_non_positive_inputs() -> None:
    with pytest.raises(RobertValidationError, match="wavelength"):
        planck_radiance_wavelength([0.0, 1.0], 1000.0)

    with pytest.raises(RobertValidationError, match="planet_radius"):
        blackbody_eclipse_depth([5.0], 1000.0, 5000.0, -1.0, 1.0)
