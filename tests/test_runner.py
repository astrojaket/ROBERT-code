"""Tests for retrieval runner stubs."""

from __future__ import annotations

import numpy as np

from robert_exoplanets import EmissionModel, Observation, RetrievalConfig, run_stub_retrieval


def test_emission_model_returns_linear_placeholder_spectrum() -> None:
    model = EmissionModel()
    wavelength = np.array([1.0, 2.0, 3.0])

    flux = model.evaluate(wavelength, {"baseline": 10.0, "slope": 2.0})

    np.testing.assert_allclose(flux, np.array([8.0, 10.0, 12.0]))


def test_run_stub_retrieval_returns_deterministic_result() -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[9.0, 10.0, 11.0],
        uncertainty=[1.0, 1.0, 1.0],
    )
    config = RetrievalConfig(target_name="WASP-43b", instrument="JWST/MIRI LRS")

    result = run_stub_retrieval(observation, config)

    assert result.config is config
    assert result.model_name == "stub-gray-emission"
    assert result.best_fit_parameters == {"baseline": 10.0, "slope": 0.0}
    assert result.converged is False
    assert "Stub retrieval completed" in result.message
    np.testing.assert_allclose(result.model_flux, np.array([10.0, 10.0, 10.0]))
    assert result.log_likelihood == -1.0
