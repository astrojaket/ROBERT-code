"""Tests for the v0.3 minimal forward-model pipeline."""

from __future__ import annotations

import numpy as np

from robert_exoplanets import (
    AtmosphereBuilder,
    ConstantChemistry,
    ForwardModel,
    GaussianLikelihood,
    IsothermalTemperatureProfile,
    LinearObservationResponse,
    Observation,
    Planet,
    PressureGrid,
    RobertConfig,
    SpectralGrid,
)


def test_minimal_forward_model_runs_from_typed_config() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[1.0, 1.5, 2.0],
        uncertainty=[0.1, 0.1, 0.1],
    )
    config = RobertConfig(
        run_name="forward-smoke",
        planet=Planet(name="WASP-43b", radius_m=7.0e7, gravity_m_s2=45.0),
        observations=(observation,),
    )
    pressure_grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    atmosphere_builder = AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=IsothermalTemperatureProfile(temperature=1000.0),
        chemistry_model=ConstantChemistry({"H2O": 1.0e-3}),
    )
    native_grid = SpectralGrid.from_array([1.0, 2.0, 3.0], role="native")
    response = LinearObservationResponse().prepare(config.observations[0])
    forward_model = ForwardModel(
        atmosphere_builder=atmosphere_builder,
        native_spectral_grid=native_grid,
        instrument_response=response,
    )

    prediction = forward_model.predict({"emission_baseline": 1.5, "emission_slope": 0.5})

    np.testing.assert_allclose(prediction.native_spectrum.values, np.array([1.0, 1.5, 2.0]))
    np.testing.assert_allclose(prediction.observed_spectrum.values, observation.flux)
    assert prediction.atmosphere.species == ("H2O",)
    assert prediction.opacity.extinction.shape == (1, 2, 3)
    assert forward_model.prepared_opacity.cache_key
    assert GaussianLikelihood().loglike(prediction, observation) == 0.0


def test_forward_model_prepares_deterministic_opacity_cache_key() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        flux=[1.0, 1.0],
        uncertainty=[0.1, 0.1],
    )
    pressure_grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=2)
    atmosphere_builder = AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=IsothermalTemperatureProfile(temperature=1000.0),
        chemistry_model=ConstantChemistry({"H2O": 1.0e-3}),
    )
    native_grid = SpectralGrid.from_array([1.0, 2.0], role="native")
    response = LinearObservationResponse().prepare(observation)

    first = ForwardModel(atmosphere_builder, native_grid, response)
    second = ForwardModel(atmosphere_builder, native_grid, response)

    assert first.prepared_opacity.cache_key == second.prepared_opacity.cache_key
