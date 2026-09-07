"""Tests for full-covariance likelihoods and filtered noise propagation."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    CorrelatedGaussianLikelihood,
    DenseCovariance,
    LinearDataModelFilter,
    Observation,
    PositiveSemidefiniteCovariance,
    SpectralGrid,
    Spectrum,
    propagate_covariance_through_filter,
)
from robert_exoplanets.core import RobertValidationError


def _observation(
    wavelength: list[float],
    flux: list[float],
    *,
    uncertainty: list[float] | None = None,
    mask: list[bool] | None = None,
) -> Observation:
    return Observation.from_arrays(
        wavelength=wavelength,
        flux=flux,
        uncertainty=np.ones(len(wavelength)) if uncertainty is None else uncertainty,
        mask=mask,
    )


def _spectrum(wavelength: list[float], values: list[float]) -> Spectrum:
    return Spectrum(
        spectral_grid=SpectralGrid.from_array(wavelength),
        values=values,
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def test_correlated_gaussian_likelihood_matches_hand_calculated_normalized_value() -> None:
    observation = _observation([1.0, 2.0], [1.2, 1.8])
    prediction = _spectrum([1.0, 2.0], [1.0, 2.0])
    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    likelihood = CorrelatedGaussianLikelihood(
        covariance,
        include_normalization=True,
    )

    residual = observation.flux - prediction.values
    expected = -0.5 * (
        residual @ np.linalg.solve(covariance, residual)
        + np.linalg.slogdet(covariance)[1]
        + 2.0 * np.log(2.0 * np.pi)
    )

    assert likelihood.loglike(prediction, observation) == pytest.approx(expected)
    assert likelihood.pointwise_loglike(prediction, observation).sum() == pytest.approx(
        expected
    )


def test_mask_selects_the_matching_covariance_rows_and_columns() -> None:
    observation = _observation(
        [1.0, 2.0, 3.0],
        [1.2, 4.0, 2.8],
        mask=[True, False, True],
    )
    prediction = _spectrum([1.0, 2.0, 3.0], [1.0, 0.0, 3.0])
    covariance = DenseCovariance(
        [
            [2.0, 0.2, 0.1],
            [0.2, 1.5, 0.3],
            [0.1, 0.3, 1.0],
        ]
    )
    assert covariance.effective_rank == covariance.size
    likelihood = CorrelatedGaussianLikelihood(covariance)

    selected = np.array([True, False, True])
    residual = observation.flux[selected] - prediction.values[selected]
    selected_covariance = np.asarray(covariance.matrix)[np.ix_(selected, selected)]
    expected = -0.5 * (
        residual @ np.linalg.solve(selected_covariance, residual)
    )

    assert likelihood.loglike(prediction, observation) == pytest.approx(expected)
    np.testing.assert_allclose(
        likelihood.effective_covariance(observation).matrix,
        selected_covariance,
    )


def test_filter_propagates_full_covariance_and_likelihood_matches_hand_value() -> None:
    wavelength = [1.0, 2.0, 3.0]
    data = [1.2, 1.7, 2.4]
    model = [1.0, 2.0, 2.0]
    observation = _observation(wavelength, data)
    model_spectrum = _spectrum(wavelength, model)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 1.0, 0.0],
            [0.0, 0.2, 1.0],
        ]
    )
    covariance = np.array(
        [
            [1.4, 0.2, 0.1],
            [0.2, 1.1, 0.15],
            [0.1, 0.15, 0.9],
        ]
    )
    operator = LinearDataModelFilter(matrix, name="fixed-high-pass")
    prepared = operator.prepare(observation)
    filtered_data, filtered_model, output_mask = prepared.apply_data_model(data, model)
    propagated = prepared.propagate_covariance(covariance)
    propagated_from_function = propagate_covariance_through_filter(covariance, prepared)
    expected_covariance = matrix @ covariance @ matrix.T

    np.testing.assert_allclose(propagated, expected_covariance)
    np.testing.assert_allclose(propagated_from_function, expected_covariance)
    assert output_mask.tolist() == [True, True, True]

    filtered_observation = _observation(wavelength, filtered_data.tolist())
    filtered_model_spectrum = operator.apply_spectrum(model_spectrum)
    likelihood = CorrelatedGaussianLikelihood(
        propagated,
        include_normalization=True,
    )
    residual = filtered_data - filtered_model
    expected = -0.5 * (
        residual @ np.linalg.solve(expected_covariance, residual)
        + np.linalg.slogdet(expected_covariance)[1]
        + 3.0 * np.log(2.0 * np.pi)
    )

    assert likelihood.loglike(filtered_model_spectrum, filtered_observation) == pytest.approx(
        expected
    )


def test_filter_covariance_uses_combined_input_masks() -> None:
    operator = LinearDataModelFilter(
        np.eye(3),
        input_mask=[True, True, False],
    )
    observation = _observation(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        mask=[True, False, True],
    )
    prepared = operator.prepare(observation)
    covariance = np.diag([1.0, 2.0, 3.0])
    expected = np.diag([1.0, 0.0, 0.0])

    np.testing.assert_allclose(prepared.propagate_covariance(covariance), expected)
    with pytest.raises(RobertValidationError, match="prepared linear filter"):
        propagate_covariance_through_filter(covariance, prepared, [True, True, True])


def test_covariance_validation_rejects_non_symmetric_or_non_positive_definite_values() -> None:
    with pytest.raises(RobertValidationError, match="symmetric"):
        DenseCovariance([[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(RobertValidationError, match="positive definite"):
        DenseCovariance([[1.0, 2.0], [2.0, 1.0]])


def test_rank_one_filter_uses_psd_support_rank_and_pseudo_normalization() -> None:
    filter_operator = LinearDataModelFilter(
        [[1.0, 0.0], [2.0, 0.0]],
        name="rank-one-pca-filter",
    )
    propagated = filter_operator.propagate_covariance(np.eye(2))
    covariance = PositiveSemidefiniteCovariance(propagated)

    assert covariance.effective_rank == 1
    assert covariance.pseudo_log_determinant() == pytest.approx(np.log(5.0))
    assert covariance.log_determinant() == pytest.approx(np.log(5.0))
    assert covariance.supports([1.0, 2.0])
    np.testing.assert_allclose(covariance.solve([1.0, 2.0]), [0.2, 0.4])

    observation = _observation([1.0, 2.0], [1.0, 2.0])
    prediction = _spectrum([1.0, 2.0], [0.0, 0.0])
    likelihood = CorrelatedGaussianLikelihood(
        covariance,
        include_normalization=True,
    )
    expected = -0.5 * (1.0 + np.log(5.0) + np.log(2.0 * np.pi))

    assert likelihood.effective_residual_rank(observation) == 1
    assert likelihood.chi_square(prediction, observation) == pytest.approx(1.0)
    assert likelihood.loglike(prediction, observation) == pytest.approx(expected)


def test_psd_likelihood_rejects_residual_outside_degenerate_support() -> None:
    covariance = PositiveSemidefiniteCovariance([[1.0, 2.0], [2.0, 4.0]])
    observation = _observation([1.0, 2.0], [0.0, 1.0])
    prediction = _spectrum([1.0, 2.0], [0.0, 0.0])
    likelihood = CorrelatedGaussianLikelihood(covariance)

    assert not covariance.supports([0.0, 1.0])
    assert np.isinf(likelihood.chi_square(prediction, observation))
    assert likelihood.loglike(prediction, observation) == float("-inf")


def test_correlated_likelihood_rejects_zero_rank_covariance() -> None:
    zero_rank = PositiveSemidefiniteCovariance(np.zeros((2, 2)))
    assert zero_rank.effective_rank == 0
    with pytest.raises(RobertValidationError, match="effective_rank must be positive"):
        CorrelatedGaussianLikelihood(zero_rank)

    masked_rank = PositiveSemidefiniteCovariance(
        [[1.0, 0.0], [0.0, 0.0]],
    )
    likelihood = CorrelatedGaussianLikelihood(masked_rank)
    observation = _observation(
        [1.0, 2.0],
        [0.0, 0.0],
        mask=[False, True],
    )
    with pytest.raises(RobertValidationError, match="effective_rank must be positive"):
        likelihood.effective_residual_rank(observation)
