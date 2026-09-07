"""Tests for data-set-specific likelihood composition."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    CorrelatedGaussianLikelihood,
    DenseCovariance,
    GaussianLikelihood,
    Observation,
    ObservationCollection,
    ObservationDataset,
    PositiveSemidefiniteCovariance,
    Spectrum,
)
from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.likelihoods.high_resolution import (
    PolynomialContinuumLikelihood,
)
from robert_exoplanets.likelihoods.mixed import MixedMultiDatasetLikelihood


def _observation(wavelength, flux, uncertainty, instrument):
    return Observation.from_arrays(
        wavelength,
        flux,
        uncertainty,
        instrument=instrument,
    )


class _StructuralCorrelatedLikelihood:
    """Small structural double for the exact covariance statistic contract."""

    name = "structural-correlated"
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])

    def effective_inputs(self, prediction, observation, parameters=None):
        del parameters
        return (
            np.asarray(prediction.values, dtype=float),
            np.asarray(observation.flux, dtype=float),
            np.sqrt(np.diag(self.covariance)),
        )

    def chi_square(self, prediction, observation, parameters=None):
        del parameters
        residual = np.asarray(observation.flux - prediction.values, dtype=float)
        return float(residual @ np.linalg.solve(self.covariance, residual))

    def effective_residual_rank(self, observation):
        return observation.n_points

    def loglike(self, prediction, observation, parameters=None):
        return -0.5 * self.chi_square(prediction, observation, parameters)

    def pointwise_loglike(self, prediction, observation, parameters=None):
        return np.asarray([self.loglike(prediction, observation, parameters)])


class _DiagonalOnlyCorrelatedLikelihood(_StructuralCorrelatedLikelihood):
    chi_square = None


def test_mixed_likelihood_sums_direct_and_projected_terms() -> None:
    low_observation = _observation(
        [1.0, 2.0],
        [1.0, 2.0],
        [0.1, 0.1],
        "low",
    )
    high_observation = _observation(
        [2.30, 2.31, 2.32, 2.33],
        [3.0, 3.2, 2.9, 3.1],
        [0.1, 0.1, 0.1, 0.1],
        "high",
    )
    observations = ObservationCollection(
        (
            ObservationDataset("low", low_observation),
            ObservationDataset("high", high_observation),
        )
    )
    spectra = {
        "low": Spectrum.from_arrays(
            low_observation.wavelength,
            [1.1, 1.9],
            "eclipse_depth",
            "eclipse_depth",
        ),
        "high": Spectrum.from_arrays(
            high_observation.wavelength,
            [3.1, 3.1, 3.0, 3.0],
            "eclipse_depth",
            "eclipse_depth",
        ),
    }
    low_likelihood = GaussianLikelihood(
        offset_parameter=None,
        jitter_parameter=None,
    )
    high_likelihood = PolynomialContinuumLikelihood(degree=1).prepare(
        high_observation
    )
    mixed = MixedMultiDatasetLikelihood(
        {"low": low_likelihood, "high": high_likelihood}
    )

    expected = low_likelihood.loglike(spectra["low"], low_observation)
    expected += high_likelihood.loglike(spectra["high"], high_observation)

    assert mixed.loglike(spectra, observations) == pytest.approx(expected)
    effective = mixed.effective_inputs_by_dataset(spectra, observations)
    assert effective["low"][0].shape == (2,)
    assert effective["high"][0].shape == (4,)
    assert mixed.pointwise_loglike(spectra, observations).shape == (6,)


def test_mixed_likelihood_requires_exact_dataset_names() -> None:
    observation = _observation([1.0], [1.0], [0.1], "one")
    observations = ObservationCollection((ObservationDataset("one", observation),))
    spectrum = Spectrum.from_arrays(
        [1.0],
        [1.0],
        "eclipse_depth",
        "eclipse_depth",
    )
    mixed = MixedMultiDatasetLikelihood(
        {
            "other": GaussianLikelihood(
                offset_parameter=None,
                jitter_parameter=None,
            )
        }
    )

    with pytest.raises(RobertValidationError, match="likelihood dataset names"):
        mixed.loglike({"one": spectrum}, observations)


def test_mixed_likelihood_preserves_structural_correlated_chi_square() -> None:
    observation = _observation(
        [1.0, 2.0],
        [1.0, 2.0],
        [np.sqrt(2.0), 1.0],
        "correlated",
    )
    spectrum = Spectrum.from_arrays(
        [1.0, 2.0],
        [0.0, 0.0],
        "eclipse_depth",
        "eclipse_depth",
    )
    observations = ObservationCollection(
        (ObservationDataset("correlated", observation),)
    )
    likelihood = MixedMultiDatasetLikelihood(
        {"correlated": _StructuralCorrelatedLikelihood()}
    )

    residual = observation.flux - spectrum.values
    expected = residual @ np.linalg.solve(
        _StructuralCorrelatedLikelihood.covariance,
        residual,
    )
    assert likelihood.chi_square_by_dataset(
        {"correlated": spectrum},
        observations,
    )["correlated"] == pytest.approx(expected)
    assert likelihood.effective_residual_rank_by_dataset(
        {"correlated": spectrum},
        observations,
    ) == {"correlated": 2}


def test_mixed_likelihood_does_not_use_covariance_diagonal_as_exact_statistic() -> None:
    observation = _observation(
        [1.0, 2.0],
        [1.0, 2.0],
        [np.sqrt(2.0), 1.0],
        "correlated",
    )
    spectrum = Spectrum.from_arrays(
        [1.0, 2.0],
        [0.0, 0.0],
        "eclipse_depth",
        "eclipse_depth",
    )
    observations = ObservationCollection(
        (ObservationDataset("correlated", observation),)
    )
    likelihood = MixedMultiDatasetLikelihood(
        {"correlated": _DiagonalOnlyCorrelatedLikelihood()}
    )

    with pytest.raises(RobertValidationError, match="diagonal effective inputs"):
        likelihood.chi_square_by_dataset({"correlated": spectrum}, observations)


def test_mixed_likelihood_reports_retained_dense_and_psd_covariance_rank() -> None:
    dense_observation = _observation(
        [1.0, 2.0],
        [1.0, 2.0],
        [np.sqrt(2.0), 1.0],
        "dense",
    )
    psd_observation = _observation(
        [3.0, 4.0],
        [1.0, 2.0],
        [1.0, 2.0],
        "psd",
    )
    observations = ObservationCollection(
        (
            ObservationDataset("dense", dense_observation),
            ObservationDataset("psd", psd_observation),
        )
    )
    spectra = {
        "dense": Spectrum.from_arrays(
            [1.0, 2.0],
            [0.0, 0.0],
            "eclipse_depth",
            "eclipse_depth",
        ),
        "psd": Spectrum.from_arrays(
            [3.0, 4.0],
            [0.0, 0.0],
            "eclipse_depth",
            "eclipse_depth",
        ),
    }
    likelihood = MixedMultiDatasetLikelihood(
        {
            "dense": CorrelatedGaussianLikelihood(
                DenseCovariance([[2.0, 0.5], [0.5, 1.0]])
            ),
            "psd": CorrelatedGaussianLikelihood(
                PositiveSemidefiniteCovariance([[1.0, 2.0], [2.0, 4.0]])
            ),
        }
    )

    assert likelihood.effective_residual_rank_by_dataset(spectra, observations) == {
        "dense": 2,
        "psd": 1,
    }
    chi_square = likelihood.chi_square_by_dataset(spectra, observations)
    assert chi_square["dense"] == pytest.approx(
        np.array([1.0, 2.0])
        @ np.linalg.solve(np.array([[2.0, 0.5], [0.5, 1.0]]), [1.0, 2.0])
    )
    assert chi_square["psd"] == pytest.approx(1.0)
