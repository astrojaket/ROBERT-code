"""Tests for NEMESIS-style pressure-resolved optimal estimation."""

from __future__ import annotations

import numpy as np

from robert_exoplanets import (
    LayerByLayerStateVector,
    Observation,
    PressureGrid,
    RetrievalProblem,
    Spectrum,
    SplineFreeChemistry,
    VerticalProfileParameterization,
    pressure_correlated_covariance,
    run_optimal_estimation,
)


def test_pressure_correlated_covariance_uses_log_pressure_scale_heights() -> None:
    pressure = np.exp(np.arange(4.0))
    covariance = pressure_correlated_covariance(
        pressure,
        [2.0, 2.0, 2.0, 2.0],
        correlation_length=2.0,
    )

    np.testing.assert_allclose(np.diag(covariance), 4.0)
    np.testing.assert_allclose(covariance[0, 1], 4.0 * np.exp(-0.5))
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_layer_state_vector_decodes_temperature_vmr_and_aerosol_profiles() -> None:
    pressure = np.geomspace(1.0e-5, 1.0, 5)
    temperature = VerticalProfileParameterization.temperature(
        pressure=pressure,
        prior_temperature=np.linspace(900.0, 1400.0, 5),
        prior_sigma_K=150.0,
        correlation_length=1.5,
    )
    water = VerticalProfileParameterization.positive_profile(
        name="H2O",
        pressure=pressure,
        prior_profile=np.geomspace(1.0e-5, 1.0e-3, 5),
        prior_fractional_uncertainty=1.0,
        correlation_length=2.0,
        kind="vmr",
    )
    aerosol = VerticalProfileParameterization.positive_profile(
        name="cloud_0",
        pressure=pressure,
        prior_profile=np.geomspace(1.0e-12, 1.0e-8, 5),
        prior_fractional_uncertainty=2.0,
        correlation_length=1.0,
        kind="aerosol",
    )
    state = LayerByLayerStateVector((temperature, water, aerosol))
    mapping = state.retrieval_parameters.vector_to_mapping(state.prior_state)
    profiles = state.physical_profiles(mapping)

    np.testing.assert_allclose(profiles["temperature"], temperature.prior_state)
    np.testing.assert_allclose(profiles["H2O"], np.exp(water.prior_state))
    np.testing.assert_allclose(profiles["cloud_0"], np.exp(aerosol.prior_state))
    assert state.prior_covariance.shape == (15, 15)
    assert np.count_nonzero(state.prior_covariance[:5, 5:]) == 0


def test_spline_free_chemistry_accepts_layer_by_layer_log_vmr_state() -> None:
    grid = PressureGrid.from_log_centers(1.0, 1.0e-4, n_layers=5, unit="bar")
    h2o = VerticalProfileParameterization.positive_profile(
        name="H2O",
        pressure=grid.centers,
        prior_profile=np.geomspace(1.0e-3, 1.0e-5, 5),
        prior_fractional_uncertainty=1.0,
        correlation_length=1.0,
        kind="vmr",
    )
    chemistry = SplineFreeChemistry(
        active_species=("H2O",),
        knot_pressure=grid.centers,
        parameter_names={"H2O": h2o.parameter_names},
        parameter_mode="ln",
    )
    parameters = dict(zip(h2o.parameter_names, h2o.prior_state, strict=True))
    composition = chemistry.evaluate(parameters, grid, np.full(grid.n_layers, 1000.0))

    np.testing.assert_allclose(composition["H2O"], np.exp(h2o.prior_state))
    total = composition["H2O"] + composition["H2"] + composition["He"]
    np.testing.assert_allclose(total, 1.0)


def test_layer_by_layer_oe_returns_nemesis_diagnostics() -> None:
    pressure = np.geomspace(1.0e-5, 10.0, 8)
    prior_temperature = np.linspace(900.0, 1500.0, pressure.size)
    profile = VerticalProfileParameterization.temperature(
        pressure=pressure,
        prior_temperature=prior_temperature,
        prior_sigma_K=250.0,
        correlation_length=1.5,
        bound_sigma=4.0,
    )
    state = LayerByLayerStateVector((profile,))

    wavelength = np.linspace(2.8, 5.2, 12)
    centers = np.linspace(0.0, pressure.size - 1.0, wavelength.size)
    layers = np.arange(pressure.size, dtype=float)
    weighting = np.exp(-0.5 * np.square((centers[:, None] - layers[None, :]) / 1.1))
    weighting /= np.sum(weighting, axis=1, keepdims=True)
    sensitivity = weighting * 8.0e-6
    truth = prior_temperature + 180.0 * np.sin(np.linspace(0.0, np.pi, pressure.size))
    measured = sensitivity @ truth
    observation = Observation.from_arrays(
        wavelength=wavelength,
        flux=measured,
        uncertainty=np.full(wavelength.size, 30.0e-6),
    )

    def forward(parameters):
        temperature = profile.physical_profile(parameters)
        return Spectrum.from_arrays(
            wavelength,
            sensitivity @ temperature,
            unit="eclipse_depth",
            observable="eclipse_depth",
        )

    problem = RetrievalProblem(
        name="synthetic-layer-temperature-sounding",
        observation=observation,
        parameters=state.retrieval_parameters,
        forward_model=forward,
    )
    result = run_optimal_estimation(
        problem,
        initial_state=state.prior_state,
        prior_state=state.prior_state,
        prior_covariance=state.prior_covariance,
        max_iterations=10,
        finite_difference_scheme="forward",
    )

    assert result.converged
    assert result.jacobian is not None
    assert result.gain_matrix is not None
    assert result.measurement_error_covariance is not None
    assert result.smoothing_error_covariance is not None
    assert 0.0 < result.degrees_of_freedom_for_signal < pressure.size
    np.testing.assert_allclose(
        result.covariance,
        result.measurement_error_covariance + result.smoothing_error_covariance,
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    assert np.sqrt(np.mean(np.square(result.state_vector - truth))) < np.sqrt(
        np.mean(np.square(prior_temperature - truth))
    )


def test_forward_model_error_reduces_vertical_information() -> None:
    pressure = np.geomspace(1.0e-4, 1.0, 3)
    profile = VerticalProfileParameterization.temperature(
        pressure=pressure,
        prior_temperature=[900.0, 1000.0, 1100.0],
        prior_sigma_K=200.0,
        correlation_length=1.0,
    )
    state = LayerByLayerStateVector((profile,))
    wavelength = np.array([1.0, 2.0, 3.0])
    observation = Observation.from_arrays(
        wavelength=wavelength,
        flux=np.array([0.0095, 0.0105, 0.0115]),
        uncertainty=np.full(3, 1.0e-4),
    )

    def forward(parameters):
        return Spectrum.from_arrays(
            wavelength,
            profile.physical_profile(parameters) * 1.0e-5,
            unit="eclipse_depth",
            observable="eclipse_depth",
        )

    problem = RetrievalProblem("model-error-sounding", observation, state.retrieval_parameters, forward)
    settings = dict(
        initial_state=state.prior_state,
        prior_state=state.prior_state,
        prior_covariance=state.prior_covariance,
        max_iterations=6,
    )
    nominal = run_optimal_estimation(problem, **settings)
    conservative = run_optimal_estimation(problem, forward_model_error=5.0e-4, **settings)

    assert conservative.degrees_of_freedom_for_signal < nominal.degrees_of_freedom_for_signal
