"""Tests for observation containers."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import Observation


def test_observation_from_arrays_converts_to_float_arrays() -> None:
    observation = Observation.from_arrays(
        wavelength=[1, 2, 3],
        flux=[0.1, 0.2, 0.3],
        uncertainty=[0.01, 0.01, 0.02],
    )

    assert observation.wavelength.dtype == np.float64
    assert observation.flux.shape == (3,)
    assert observation.uncertainty.shape == (3,)


def test_observation_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        Observation.from_arrays(
            wavelength=[1, 2, 3],
            flux=[0.1, 0.2],
            uncertainty=[0.01, 0.01, 0.02],
        )


def test_observation_rejects_non_positive_uncertainty() -> None:
    with pytest.raises(ValueError, match="positive"):
        Observation.from_arrays(
            wavelength=[1, 2],
            flux=[0.1, 0.2],
            uncertainty=[0.01, 0.0],
        )


def test_observation_preserves_explicit_spectral_bins() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        wavelength_bin_edges=[0.5, 1.5, 2.5],
        flux=[0.1, 0.2],
        uncertainty=[0.01, 0.01],
    )

    np.testing.assert_allclose(observation.spectral_grid.bin_edges, [0.5, 1.5, 2.5])
    assert observation.wavelength_bin_edges.flags.writeable is False


def test_observation_rejects_centres_outside_spectral_bins() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        Observation.from_arrays(
            wavelength=[1.0, 2.0],
            wavelength_bin_edges=[1.0, 1.5, 2.5],
            flux=[0.1, 0.2],
            uncertainty=[0.01, 0.01],
        )
