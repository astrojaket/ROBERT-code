"""Tests for covariance-backed observations and Gaussian likelihoods."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets import (
    GaussianLikelihood,
    MultiDatasetGaussianLikelihood,
    Observation,
    ObservationCollection,
    ObservationDataset,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    SpectralGrid,
    Spectrum,
    UniformPrior,
    run_optimal_estimation,
)
from robert_exoplanets.core import RobertConfigError, RobertValidationError
from robert_exoplanets.postprocessing import calculate_fit_statistics
from robert_exoplanets.retrieval.data import load_observation_npz, save_observation_npz


def _observation(
    covariance: np.ndarray | None = None,
    *,
    mask: list[bool] | None = None,
) -> Observation:
    matrix = (
        np.array([[0.04, 0.01], [0.01, 0.09]])
        if covariance is None
        else np.asarray(covariance, dtype=float)
    )
    return Observation.from_arrays(
        wavelength=[1.0, 2.0],
        flux=[1.2, 1.8],
        uncertainty=[0.2, 0.3],
        mask=mask,
        wavelength_bin_edges=[[0.9, 1.1], [1.9, 2.1]],
        covariance=matrix,
    )


def _spectrum(values: list[float]) -> Spectrum:
    return Spectrum(
        spectral_grid=SpectralGrid.from_array([1.0, 2.0]),
        values=np.asarray(values, dtype=float),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def _normal_loglike(residual: np.ndarray, covariance: np.ndarray) -> float:
    return float(
        -0.5
        * (
            residual @ np.linalg.solve(covariance, residual)
            + np.linalg.slogdet(covariance)[1]
            + residual.size * np.log(2.0 * np.pi)
        )
    )


def test_gaussian_likelihood_uses_full_covariance_and_normalization() -> None:
    observation = _observation()
    prediction = _spectrum([1.0, 2.0])
    residual = observation.flux - prediction.values
    expected = _normal_loglike(residual, observation.covariance)

    likelihood = GaussianLikelihood(include_normalization=True)
    assert likelihood.loglike(prediction, observation) == pytest.approx(expected)
    assert likelihood.chi_square(prediction, observation) == pytest.approx(
        residual @ np.linalg.solve(observation.covariance, residual)
    )
    # Correlated terms have one aggregate diagnostic value, not false
    # independent per-wavelength contributions.
    np.testing.assert_allclose(
        likelihood.pointwise_loglike(prediction, observation), [expected]
    )


def test_diagonal_covariance_matches_independent_likelihood() -> None:
    diagonal = np.diag([0.04, 0.09])
    observation = _observation(diagonal)
    prediction = _spectrum([1.0, 2.0])
    parameters = {"offset": 0.03, "scale": 1.4, "jitter": 0.02}
    independent = GaussianLikelihood(
        include_normalization=True,
        uncertainty_scale_parameter="scale",
    )
    correlated = independent.loglike(prediction, observation, parameters)

    variance = (
        diagonal.diagonal() * parameters["scale"] ** 2 + parameters["jitter"] ** 2
    )
    residual = observation.flux - prediction.values - parameters["offset"]
    expected = -0.5 * np.sum(residual**2 / variance + np.log(2.0 * np.pi * variance))
    assert correlated == pytest.approx(expected)

    independent_observation = Observation.from_arrays(
        wavelength=observation.wavelength,
        flux=observation.flux,
        uncertainty=observation.uncertainty,
        mask=observation.mask,
        wavelength_bin_edges=observation.wavelength_bin_edges,
    )
    assert independent.loglike(
        prediction, independent_observation, parameters
    ) == pytest.approx(correlated)


def test_mask_subsets_covariance_rows_and_columns() -> None:
    covariance = np.array(
        [[0.04, 0.01, 0.005], [0.01, 0.09, 0.02], [0.005, 0.02, 0.16]]
    )
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[1.2, 8.0, 2.8],
        uncertainty=np.sqrt(np.diag(covariance)),
        mask=[True, False, True],
        covariance=covariance,
    )
    prediction = Spectrum(
        spectral_grid=SpectralGrid.from_array([1.0, 2.0, 3.0]),
        values=[1.0, -99.0, 3.0],
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    selected = np.array([True, False, True])
    selected_covariance = covariance[np.ix_(selected, selected)]
    residual = observation.flux[selected] - prediction.values[selected]

    assert GaussianLikelihood(include_normalization=True).loglike(
        prediction, observation
    ) == pytest.approx(_normal_loglike(residual, selected_covariance))


def test_observation_npz_roundtrip_stores_optional_covariance(tmp_path) -> None:
    observation = _observation(mask=[True, False])
    path = save_observation_npz(observation, tmp_path / "with-covariance.npz")
    restored = load_observation_npz(path)

    np.testing.assert_array_equal(restored.covariance, observation.covariance)
    np.testing.assert_array_equal(
        restored.wavelength_bin_edges, observation.wavelength_bin_edges
    )
    assert restored.metadata["covariance_key"] == "covariance"

    legacy_path = tmp_path / "legacy.npz"
    np.savez(
        legacy_path,
        wavelength=[1.0, 2.0],
        data=[1.2, 1.8],
        err=[0.2, 0.3],
    )
    legacy = load_observation_npz(legacy_path)
    assert legacy.covariance is None
    assert "covariance_key" not in legacy.metadata


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.array([[0.04, 0.011], [0.01, 0.09]]), "symmetric"),
        (np.array([[0.04, 0.08], [0.08, 0.09]]), "positive definite"),
        (np.array([[0.04, 0.01], [0.01, 0.1]]), "diagonal"),
        (np.array([[0.04, np.nan], [np.nan, 0.09]]), "finite"),
        (np.eye(3), "shape"),
    ],
)
def test_observation_rejects_invalid_covariance(
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(RobertValidationError, match=message):
        _observation(covariance)


def test_covariance_observation_rejects_independent_oe_inputs() -> None:
    observation = _observation()
    model = _spectrum([1.0, 2.0])
    parameters = RetrievalParameterSet(
        (RetrievalParameter("offset", UniformPrior(-1.0, 1.0)),)
    )
    problem = RetrievalProblem(
        name="correlated-oe-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda _: model,
        likelihood=GaussianLikelihood(),
    )

    # Marginal uncertainties remain available to plot diagnostics.
    _, _, marginal_uncertainty = problem.likelihood.effective_inputs(model, observation)
    np.testing.assert_allclose(
        marginal_uncertainty, np.sqrt(np.diag(observation.covariance))
    )
    with pytest.raises(RobertConfigError, match="covariance-backed observations"):
        run_optimal_estimation(problem)


def test_postprocessing_reports_full_covariance_chi_square() -> None:
    observation = _observation()
    prediction = _spectrum([1.0, 2.0])
    collection = ObservationCollection((ObservationDataset("dataset", observation),))
    problem = SimpleNamespace(
        likelihood=MultiDatasetGaussianLikelihood(),
        observations=collection,
    )
    diagnostics = calculate_fit_statistics(
        problem,
        {"dataset": prediction},
        {},
        fitted_parameter_count=1,
    )
    residual = observation.flux - prediction.values
    expected = residual @ np.linalg.solve(observation.covariance, residual)

    assert problem.likelihood.loglike(
        {"dataset": prediction}, collection, {}
    ) == pytest.approx(-0.5 * expected)
    assert diagnostics["chi_squared"] == pytest.approx(expected)
    assert diagnostics["chi_squared_basis"] == "full_covariance"
    assert diagnostics["standardized_residual_summary"] == (
        "marginal_uncertainty; it does not include covariance"
    )
    assert diagnostics["per_dataset"]["dataset"]["chi_squared_basis"] == (
        "full_covariance"
    )
