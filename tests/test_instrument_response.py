"""Tests for observation-grid response mapping."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    LinearObservationResponse,
    Observation,
    SpectralGrid,
    Spectrum,
    StratifiedSamplingObservationResponse,
    TopHatObservationResponse,
)
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


def test_top_hat_response_integrates_piecewise_linear_spectrum() -> None:
    native = Spectrum(
        spectral_grid=SpectralGrid.from_array([1.0, 2.0, 3.0], unit="micron"),
        values=np.array([1.0, 4.0, 9.0]),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    observation = Observation.from_arrays(
        [1.5, 2.5],
        [0.0, 0.0],
        [1.0, 1.0],
        wavelength_bin_edges=[1.0, 2.0, 3.0],
    )

    observed = TopHatObservationResponse().prepare(observation).observe(native)

    np.testing.assert_allclose(observed.values, [2.5, 6.5])
    assert observed.metadata["bin_integration"] == "piecewise_linear_top_hat"


def test_stratified_sampling_selects_fixed_points_and_averages_bins() -> None:
    observation = Observation.from_arrays(
        [1.5, 2.5],
        [0.0, 0.0],
        [1.0, 1.0],
        wavelength_bin_edges=[1.0, 2.0, 3.0],
    )
    source_grid = SpectralGrid.from_array(np.linspace(1.0, 3.0, 41))
    prepared = StratifiedSamplingObservationResponse(samples_per_bin=2).prepare(
        observation, source_grid
    )
    sampled = Spectrum(
        spectral_grid=prepared.spectral_grid,
        values=prepared.spectral_grid.values**2,
        unit="eclipse_depth",
        observable="eclipse_depth",
    )

    observed = prepared.observe(sampled)

    np.testing.assert_allclose(prepared.spectral_grid.values, [1.25, 1.75, 2.25, 2.75])
    np.testing.assert_allclose(observed.values, [2.3125, 6.3125])
    assert observed.metadata["samples_per_bin"] == "2"
