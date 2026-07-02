"""Run the v0.3 minimal forward-model pipeline."""

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


def build_config() -> RobertConfig:
    """Build a tiny typed config with one observation."""

    observation = Observation.from_arrays(
        wavelength=np.linspace(5.0, 12.0, 8),
        flux=np.array([980, 995, 1004, 1010, 1008, 1002, 994, 985], dtype=float) * 1e-6,
        uncertainty=np.full(8, 35e-6),
        instrument="JWST/MIRI LRS",
    )
    planet = Planet(name="WASP-Stub-b", radius_m=7.0e7, gravity_m_s2=45.0)
    return RobertConfig(run_name="minimal-forward-model", planet=planet, observations=(observation,))


def main() -> None:
    config = build_config()
    observation = config.observations[0]
    pressure_grid = PressureGrid.logspace(1.0e-5, 10.0, n_layers=4)
    native_grid = SpectralGrid.from_array(np.linspace(4.5, 12.5, 16), role="native")
    atmosphere_builder = AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=IsothermalTemperatureProfile(temperature=1200.0),
        chemistry_model=ConstantChemistry({"H2O": 1.0e-3, "CO": 1.0e-4}),
    )
    response = LinearObservationResponse().prepare(observation)
    forward_model = ForwardModel(
        atmosphere_builder=atmosphere_builder,
        native_spectral_grid=native_grid,
        instrument_response=response,
    )

    prediction = forward_model.predict({"emission_baseline": 1.0e-3})
    log_likelihood = GaussianLikelihood().loglike(prediction, observation)

    print(f"Run: {config.run_name}")
    print(f"Native spectral points: {prediction.native_spectrum.spectral_grid.size}")
    print(f"Observed spectral points: {prediction.observed_spectrum.spectral_grid.size}")
    print(f"Atmosphere layers: {prediction.atmosphere.n_layers}")
    print(f"Species: {', '.join(prediction.atmosphere.species)}")
    print(f"Emission backend: {prediction.diagnostics['emission_backend']}")
    print(f"Gaussian log-likelihood: {log_likelihood:.3f}")


if __name__ == "__main__":
    main()
