"""Tests for observation-grid response mapping."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import LinearObservationResponse, Observation, SpectralGrid, Spectrum
from robert_exoplanets.core import RobertCoverageError


def test_linear_observation_response_interpolates_to_observation_grid() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.5, 2.5],
        flux=[15.0, 25.0],
        uncertainty=[1.0, 1.0],
    )
    native = Spectrum(
        spectral_grid=SpectralGrid.from_array([1.0, 2.0, 3.0]),
        values=np.array([10.0, 20.0, 30.0]),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    response = LinearObservationResponse().prepare(observation)

    observed = response.observe(native)

    np.testing.assert_allclose(observed.values, np.array([15.0, 25.0]))
    assert observed.spectral_grid.role == "observed"


def test_linear_observation_response_rejects_uncovered_wavelengths() -> None:
    observation = Observation.from_arrays(
        wavelength=[0.5, 2.0],
        flux=[1.0, 1.0],
        uncertainty=[0.1, 0.1],
    )
    native = Spectrum(
        spectral_grid=SpectralGrid.from_array([1.0, 2.0]),
        values=np.array([1.0, 1.0]),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    response = LinearObservationResponse().prepare(observation)

    with pytest.raises(RobertCoverageError, match="outside native spectrum"):
        response.observe(native)
