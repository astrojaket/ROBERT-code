"""Tests for likelihood components."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import GaussianLikelihood, Observation, SpectralGrid, Spectrum


def _spectrum(values: list[float]) -> Spectrum:
    return Spectrum(
        spectral_grid=SpectralGrid.from_array([1.0, 2.0, 3.0]),
        values=np.array(values, dtype=float),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def test_gaussian_likelihood_matches_weighted_residual_sum() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[1.0, 2.0, 3.0],
        uncertainty=[1.0, 1.0, 1.0],
    )

    loglike = GaussianLikelihood().loglike(_spectrum([1.0, 1.0, 5.0]), observation)

    assert loglike == -2.5


def test_gaussian_likelihood_honors_mask_and_jitter() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[1.0, 2.0, 3.0],
        uncertainty=[1.0, 1.0, 1.0],
        mask=[True, False, True],
    )

    loglike = GaussianLikelihood().loglike(
        _spectrum([0.0, 99.0, 2.0]),
        observation,
        {"jitter": 1.0},
    )

    assert loglike == pytest.approx(-0.5)


def test_gaussian_likelihood_applies_offset_to_prediction() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[2.0, 3.0, 4.0],
        uncertainty=[1.0, 1.0, 1.0],
    )

    loglike = GaussianLikelihood().loglike(
        _spectrum([1.0, 2.0, 3.0]),
        observation,
        {"offset": 1.0},
    )

    assert loglike == 0.0
