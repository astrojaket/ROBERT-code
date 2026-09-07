"""Tests for combined low- and high-resolution injection contracts."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets.core import RobertValidationError, Spectrum
from robert_exoplanets.instruments import Observation, ObservationCollection, ObservationDataset
from robert_exoplanets.likelihoods import (
    CrossCorrelationLikelihood,
    MixedMultiDatasetLikelihood,
    MultiDatasetGaussianLikelihood,
    PolynomialContinuumLikelihood,
)
from robert_exoplanets.validation.multi_dataset import (
    evaluate_multi_dataset_injection_recovery,
    inject_spectrum_collection,
)


def _spectra() -> dict[str, Spectrum]:
    return {
        "low_resolution": Spectrum.from_arrays(
            [2.0, 2.1, 2.2],
            [1.0, 1.1, 1.2],
            "eclipse_depth",
            "eclipse_depth",
        ),
        "high_resolution": Spectrum.from_arrays(
            [2.30, 2.31, 2.32],
            [0.9, 1.0, 1.1],
            "eclipse_depth",
            "eclipse_depth",
        ),
    }


def test_collection_injection_is_repeatable_but_uses_independent_noise() -> None:
    spectra = _spectra()
    uncertainty = {name: 0.1 for name in spectra}

    first = inject_spectrum_collection(spectra, uncertainty, seed=17)
    second = inject_spectrum_collection(spectra, uncertainty, seed=17)
    first_by_name = {dataset.name: dataset for dataset in first.datasets}
    second_by_name = {dataset.name: dataset for dataset in second.datasets}

    for name in first.names:
        np.testing.assert_array_equal(
            first_by_name[name].observation.flux,
            second_by_name[name].observation.flux,
        )
    low_noise = first_by_name["low_resolution"].observation.flux - spectra[
        "low_resolution"
    ].values
    high_noise = first_by_name["high_resolution"].observation.flux - spectra[
        "high_resolution"
    ].values
    assert not np.array_equal(low_noise, high_noise)
    assert (
        first_by_name["low_resolution"].observation.metadata[
            "collection_injection_seed"
        ]
        == "17"
    )


def test_combined_recovery_uses_all_effective_likelihood_points() -> None:
    spectra = _spectra()
    observations = inject_spectrum_collection(
        spectra,
        {name: 0.1 for name in spectra},
        seed=22,
        noise_scale=0.0,
    )

    report = evaluate_multi_dataset_injection_recovery(
        case_name="combined-exact",
        truth={"temperature": 1200.0},
        estimates={"temperature": 1200.0},
        absolute_tolerances={"temperature": 10.0},
        observations=observations,
        best_fit_spectra=spectra,
        likelihood=MultiDatasetGaussianLikelihood(),
        seed=22,
        reduced_chi_square_bounds=(0.0, 1.0),
    )

    assert report.passed
    assert report.n_active_points == 6
    assert report.metadata["dataset_names"] == (
        "low_resolution,high_resolution"
    )


def test_collection_injection_requires_exact_uncertainty_names() -> None:
    with pytest.raises(RobertValidationError, match="uncertainty names"):
        inject_spectrum_collection(
            _spectra(),
            {"low_resolution": 0.1},
            seed=1,
        )


class _ExactCorrelatedLikelihood:
    """Structural exact-statistic likelihood for multi-dataset validation."""

    name = "exact-correlated"
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


def test_multi_dataset_recovery_uses_exact_correlated_chi_square() -> None:
    observation = Observation.from_arrays(
        [1.0, 2.0],
        [1.0, 2.0],
        [np.sqrt(2.0), 1.0],
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
        {"correlated": _ExactCorrelatedLikelihood()}
    )

    report = evaluate_multi_dataset_injection_recovery(
        case_name="exact-correlated",
        truth={"parameter": 0.0},
        estimates={"parameter": 0.0},
        absolute_tolerances={"parameter": 1.0},
        observations=observations,
        best_fit_spectra={"correlated": spectrum},
        likelihood=likelihood,
        seed=1,
        reduced_chi_square_bounds=(0.0, 10.0),
    )

    residual = observation.flux - spectrum.values
    expected = residual @ np.linalg.solve(
        _ExactCorrelatedLikelihood.covariance,
        residual,
    )
    assert report.chi_square == pytest.approx(expected)
    assert report.n_active_points == 2


def test_multi_dataset_recovery_subtracts_polynomial_projection_rank() -> None:
    observation = Observation.from_arrays(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    spectrum = Spectrum.from_arrays(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        "eclipse_depth",
        "eclipse_depth",
    )
    observations = ObservationCollection((ObservationDataset("polynomial", observation),))
    prepared = PolynomialContinuumLikelihood(degree=1).prepare(observation)
    likelihood = MixedMultiDatasetLikelihood({"polynomial": prepared})

    report = evaluate_multi_dataset_injection_recovery(
        case_name="polynomial-rank",
        truth={"parameter": 0.0},
        estimates={"parameter": 0.0},
        absolute_tolerances={"parameter": 1.0},
        observations=observations,
        best_fit_spectra={"polynomial": spectrum},
        likelihood=likelihood,
        seed=2,
        reduced_chi_square_bounds=(0.0, 100.0),
    )

    expected = prepared.chi_square(spectrum)
    assert report.chi_square == pytest.approx(expected)
    assert report.n_active_points == 2  # four points minus polynomial rank two
    assert report.reduced_chi_square == pytest.approx(expected)


def test_multi_dataset_recovery_subtracts_profiled_ccf_amplitude_rank() -> None:
    observation = Observation.from_arrays(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 0.0],
        [1.0, 1.0, 1.0],
    )
    spectrum = Spectrum.from_arrays(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        "eclipse_depth",
        "eclipse_depth",
    )
    observations = ObservationCollection((ObservationDataset("ccf", observation),))
    prepared = CrossCorrelationLikelihood(
        detrend_degree=None,
        profile_amplitude=True,
    ).prepare(observation)
    likelihood = MixedMultiDatasetLikelihood({"ccf": prepared})

    report = evaluate_multi_dataset_injection_recovery(
        case_name="ccf-rank",
        truth={"parameter": 0.0},
        estimates={"parameter": 0.0},
        absolute_tolerances={"parameter": 1.0},
        observations=observations,
        best_fit_spectra={"ccf": spectrum},
        likelihood=likelihood,
        seed=3,
        reduced_chi_square_bounds=(0.0, 100.0),
    )

    expected = prepared.chi_square(spectrum)
    assert report.chi_square == pytest.approx(expected)
    assert report.n_active_points == 2  # three points minus fitted amplitude
    assert report.reduced_chi_square == pytest.approx(expected)
