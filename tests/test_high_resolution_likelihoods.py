"""Tests for high-resolution likelihoods and reusable linear projections."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets.core import RobertValidationError, SpectralGrid, Spectrum
from robert_exoplanets.instruments import Observation
from robert_exoplanets.likelihoods.high_resolution import (
    CalibratedDirectFluxLikelihood,
    CrossCorrelationLikelihood,
    PolynomialContinuumLikelihood,
    PreparedPolynomialContinuumProjection,
    prepare_polynomial_continuum_projection,
)


def _observation(
    wavelength: list[float],
    flux: list[float],
    uncertainty: list[float],
    *,
    mask: list[bool] | None = None,
) -> Observation:
    return Observation.from_arrays(
        wavelength=wavelength,
        flux=flux,
        uncertainty=uncertainty,
        mask=mask,
    )


def _spectrum(wavelength: list[float], values: list[float]) -> Spectrum:
    return Spectrum(
        spectral_grid=SpectralGrid.from_array(wavelength),
        values=np.asarray(values, dtype=float),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def test_calibrated_direct_flux_matches_weighted_hand_calculation() -> None:
    observation = _observation(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 8.0],
        [1.0, 2.0, 4.0],
        mask=[True, False, True],
    )
    model = _spectrum([1.0, 2.0, 3.0], [1.0, 3.0, 7.0])

    likelihood = CalibratedDirectFluxLikelihood(
        calibration_scale=2.0,
        calibration_offset=-1.0,
    )
    # Calibrated model is [1, 5, 13], so the retained residuals are
    # [1, -5] with sigma [1, 4].
    expected = -0.5 * (1.0**2 + (-5.0 / 4.0) ** 2)

    assert likelihood.loglike(model, observation) == pytest.approx(expected)
    np.testing.assert_allclose(
        likelihood.pointwise_loglike(model, observation),
        [-0.5, -0.5 * (5.0 / 4.0) ** 2],
    )


def test_calibrated_direct_flux_supports_runtime_calibration_and_normalization() -> None:
    observation = _observation([1.0, 2.0], [2.0, 4.0], [2.0, 4.0])
    model = _spectrum([1.0, 2.0], [1.0, 2.0])
    likelihood = CalibratedDirectFluxLikelihood(
        calibration_scale_parameter="alpha",
        calibration_offset_parameter="beta",
        include_normalization=True,
    )

    value = likelihood.loglike(model, observation, {"alpha": 2.0, "beta": 0.0})
    expected = -0.5 * np.sum(np.log(2.0 * np.pi * np.square([2.0, 4.0])))
    assert value == pytest.approx(expected)


def test_calibrated_direct_flux_keeps_spectrum_observed_spectrum_contract() -> None:
    class ForwardOutput:
        def __init__(self, spectrum: Spectrum) -> None:
            self.observed_spectrum = spectrum

    observation = _observation([1.0, 2.0], [1.0, 2.0], [1.0, 1.0])
    model = _spectrum([1.0, 2.0], [1.0, 2.0])

    assert CalibratedDirectFluxLikelihood().loglike(
        ForwardOutput(model),
        observation,
    ) == pytest.approx(0.0)


def test_calibrated_direct_flux_rejects_contract_mismatch_and_bad_runtime_scale() -> None:
    observation = _observation([1.0, 2.0], [1.0, 2.0], [1.0, 1.0])
    model = _spectrum([1.0, 2.1], [1.0, 2.0])

    with pytest.raises(RobertValidationError, match="observation wavelength grid"):
        CalibratedDirectFluxLikelihood().loglike(model, observation)

    valid_model = _spectrum([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(RobertValidationError, match="calibration scale"):
        CalibratedDirectFluxLikelihood(
            calibration_scale_parameter="alpha",
        ).loglike(valid_model, observation, {"alpha": 0.0})


def test_weighted_projection_is_linear_and_removes_shared_polynomial() -> None:
    observation = _observation(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    projection = prepare_polynomial_continuum_projection(observation, degree=1)
    data = np.array([2.0, 2.5, 3.0, 3.5])
    model = np.array([1.0, 1.5, 2.0, 2.5])
    shared_continuum = np.array([7.0, 8.0, 9.0, 10.0])

    np.testing.assert_allclose(projection.apply(data), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        projection.apply(data + shared_continuum),
        projection.apply(data),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        projection.residual(data, model),
        projection.apply(data) - projection.apply(model),
    )
    assert isinstance(projection, PreparedPolynomialContinuumProjection)
    assert projection.effective_residual_rank(observation) == 2
    assert projection.design_matrix.flags.writeable is False
    assert projection.orthonormal_basis.flags.writeable is False


def test_profiled_polynomial_continuum_matches_hand_calculation() -> None:
    observation = _observation(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 4.0, 8.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    model = _spectrum([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0])
    likelihood = PolynomialContinuumLikelihood(degree=0).prepare(observation)

    mean = np.mean(observation.flux)
    expected_chi2 = np.sum(np.square(observation.flux - mean))
    assert likelihood.loglike(model) == pytest.approx(-0.5 * expected_chi2)
    assert np.sum(likelihood.pointwise_loglike(model)) == pytest.approx(
        likelihood.loglike(model)
    )
    assert likelihood.effective_residual_rank(observation) == 3


def test_profiled_continuum_is_invariant_to_a_shared_polynomial() -> None:
    wavelength = [1.0, 2.0, 3.0, 4.0, 5.0]
    observation = _observation(
        wavelength,
        [1.0, 2.0, 3.5, 5.0, 7.0],
        [1.0, 2.0, 1.0, 2.0, 1.0],
        mask=[True, True, False, True, True],
    )
    model = _spectrum(wavelength, [0.5, 1.5, 3.0, 4.0, 6.5])
    prepared = PolynomialContinuumLikelihood(degree=1).prepare(observation)
    shared = np.array([4.0, 7.0, 10.0, 13.0, 16.0])
    shifted_observation = _observation(
        wavelength,
        (observation.flux + shared).tolist(),
        observation.uncertainty.tolist(),
        mask=observation.mask.tolist() if observation.mask is not None else None,
    )
    shifted_model = _spectrum(wavelength, (model.values + shared).tolist())
    shifted_prepared = PolynomialContinuumLikelihood(degree=1).prepare(
        shifted_observation
    )

    assert prepared.loglike(model) == pytest.approx(
        shifted_prepared.loglike(shifted_model),
        abs=1.0e-12,
    )


def test_projection_rejects_insufficient_rank_and_invalid_application() -> None:
    observation = _observation(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        mask=[True, False, False],
    )
    with pytest.raises(RobertValidationError, match="fewer valid points"):
        prepare_polynomial_continuum_projection(observation, degree=1)

    observation = _observation([1.0, 2.0], [0.0, 0.0], [1.0, 1.0])
    projection = prepare_polynomial_continuum_projection(observation, degree=0)
    with pytest.raises(RobertValidationError, match="match the observation grid"):
        projection.apply([1.0])
    with pytest.raises(RobertValidationError, match="finite"):
        projection.apply([1.0, np.nan])


def test_cross_correlation_profiles_amplitude_and_is_scale_invariant() -> None:
    observation = _observation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
    model = _spectrum([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    prepared = CrossCorrelationLikelihood(detrend_degree=None).prepare(observation)

    assert prepared.correlation(model) == pytest.approx(1.0)
    assert prepared.profiled_amplitude(model) == pytest.approx(0.5)
    assert prepared.loglike(model) == pytest.approx(0.0)
    assert prepared.loglike(_spectrum([1.0, 2.0, 3.0], [20.0, 40.0, 60.0])) == pytest.approx(
        prepared.loglike(model)
    )


def test_cross_correlation_matches_profiled_hand_calculation_and_mask() -> None:
    observation = _observation(
        [1.0, 2.0, 3.0],
        [1.0, 100.0, 0.0],
        [1.0, 1.0, 1.0],
        mask=[True, False, True],
    )
    model = _spectrum([1.0, 2.0, 3.0], [1.0, 50.0, 1.0])
    prepared = CrossCorrelationLikelihood(detrend_degree=None).prepare(observation)

    # The mask leaves data=[1, 0] and model=[1, 1].  The fitted amplitude
    # is 1/2, giving chi2=1/2 and loglike=-1/4.
    assert prepared.correlation(model) == pytest.approx(1.0 / np.sqrt(2.0))
    assert prepared.profiled_amplitude(model) == pytest.approx(0.5)
    assert prepared.loglike(model) == pytest.approx(-0.25)
    assert prepared.effective_residual_rank(observation) == 1
    effective_model, effective_data, effective_uncertainty = (
        prepared.effective_inputs(model)
    )
    effective_loglike = -0.5 * np.sum(
        np.square((effective_data - effective_model) / effective_uncertainty)
    )
    assert effective_loglike == pytest.approx(prepared.loglike(model))


def test_cross_correlation_continuum_projection_is_shared_by_data_and_model() -> None:
    wavelength = [1.0, 2.0, 3.0, 4.0]
    observation = _observation(wavelength, [2.0, 3.0, 5.0, 8.0], [1.0] * 4)
    model = _spectrum(wavelength, [1.0, 2.0, 4.0, 7.0])
    likelihood = CrossCorrelationLikelihood(detrend_degree=1)
    prepared = likelihood.prepare(observation)
    shifted_observation = _observation(
        wavelength,
        (observation.flux + np.array([10.0, 20.0, 30.0, 40.0])).tolist(),
        [1.0] * 4,
    )
    shifted_model = _spectrum(
        wavelength,
        (model.values + np.array([10.0, 20.0, 30.0, 40.0])).tolist(),
    )

    shifted_prepared = CrossCorrelationLikelihood(detrend_degree=1).prepare(
        shifted_observation
    )
    assert prepared.loglike(model) == pytest.approx(shifted_prepared.loglike(shifted_model))


def test_cross_correlation_rejects_zero_norm_templates_and_accepts_normalization() -> None:
    observation = _observation([1.0, 2.0], [1.0, 2.0], [2.0, 2.0])
    zero_model = _spectrum([1.0, 2.0], [0.0, 0.0])
    prepared = CrossCorrelationLikelihood(detrend_degree=None).prepare(observation)
    assert prepared.loglike(zero_model) == float("-inf")
    assert np.isnan(prepared.correlation(zero_model))

    model = _spectrum([1.0, 2.0], [1.0, 2.0])
    normalized = CrossCorrelationLikelihood(
        detrend_degree=None,
        include_normalization=True,
    ).prepare(observation)
    expected = -0.5 * np.sum(np.log(2.0 * np.pi * np.square([2.0, 2.0])))
    assert normalized.loglike(model) == pytest.approx(expected)
