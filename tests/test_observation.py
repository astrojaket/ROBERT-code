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
