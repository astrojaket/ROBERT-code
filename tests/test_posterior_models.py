"""Focused tests for shared posterior model diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, RobertValidationError, Spectrum
from robert_exoplanets.diagnostics import cloud_profiles as cloud_profiles_module
from robert_exoplanets.diagnostics.posterior_models import (
    POSTERIOR_QUANTILE_PROBABILITIES,
    PosteriorProductUnavailableError,
    collect_posterior_models,
)
from robert_exoplanets.forward.inhomogeneous import TwoRegionEmissionModel
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
)


@dataclass
class _Builder:
    pressure_grid: PressureGrid
    calls: int = 0

    def build(self, parameters: dict[str, float]) -> AtmosphereState:
        self.calls += 1
        x = float(parameters["x"])
        return AtmosphereState(
            pressure_grid=self.pressure_grid,
            temperature=np.array([500.0 + x, 700.0 + x]),
            composition={
                "H2O": np.array([0.01 + x / 1000.0, 0.02 + x / 1000.0]),
                "CO": np.array([0.1, 0.2]),
            },
            mean_molecular_weight=2.3,
        )


class _Model:
    def __init__(
        self, builder: _Builder, wavelength: list[float], shift: float = 0.0
    ) -> None:
        self.atmosphere_builder = builder
        self.cloud_model: object | None = None
        self.wavelength = wavelength
        self.shift = shift
        self.calls = 0

    def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
        self.calls += 1
        x = float(parameters["x"])
        values = np.asarray(
            [x + self.shift + index for index in range(len(self.wavelength))]
        )
        return Spectrum.from_arrays(
            self.wavelength,
            values,
            unit="eclipse_depth",
            observable="eclipse_depth",
        )


class _SingleProblem:
    def __init__(self, model: _Model, mask: list[bool] | None = None) -> None:
        self.forward_model = model
        self.observation = SimpleNamespace(
            mask=None if mask is None else np.asarray(mask)
        )
        self.likelihood = SimpleNamespace(offset_parameter="offset")
        self.prediction_calls = 0

    @staticmethod
    def parameter_mapping(vector: np.ndarray) -> dict[str, float]:
        return {"x": float(vector[0]), "offset": float(vector[1])}

    def model_spectrum(self, parameters: dict[str, float]) -> Spectrum:
        self.prediction_calls += 1
        return self.forward_model(parameters)


def _grid() -> PressureGrid:
    return PressureGrid(
        edges=np.asarray([0.01, 0.1, 1.0]),
        centers=np.asarray([0.03, 0.3]),
    )


def test_single_products_use_same_selected_draws_and_offset() -> None:
    builder = _Builder(_grid())
    model = _Model(builder, [1.0, 2.0])
    problem = _SingleProblem(model, mask=[True, False])
    draws = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [1.0, 0.5],
            [2.0, 1.0],
            [10.0, 0.0],
            [20.0, 1.0],
            [21.0, 2.0],
            [40.0, 3.0],
        ]
    )

    products = collect_posterior_models(problem, draws)

    expected_spectrum = np.asarray(
        [
            np.quantile(draws[:, 0] + draws[:, 1], probability)
            for probability in POSTERIOR_QUANTILE_PROBABILITIES
        ]
    )
    np.testing.assert_allclose(
        products.spectra["primary"].values[:, 0], expected_spectrum
    )
    np.testing.assert_allclose(
        products.temperature_profiles["primary"].values[:, 0],
        [
            np.quantile(500.0 + draws[:, 0], probability)
            for probability in POSTERIOR_QUANTILE_PROBABILITIES
        ],
    )
    assert products.spectra["primary"].wavelength.tolist() == [1.0, 2.0]
    assert products.spectra["primary"].mask is not None
    assert products.spectra["primary"].mask.tolist() == [True, False]
    assert products.spectra["primary"].unit == "eclipse_depth"
    assert products.temperature_profiles["primary"].coordinate_unit == "bar"
    assert products.vmr_profiles["primary"]["H2O"].value_unit == "volume_mixing_ratio"
    assert products.draw_vectors.flags.writeable is False
    assert problem.prediction_calls == len(draws)
    assert model.calls == len(draws)
    assert builder.calls == len(draws)


def test_multi_dataset_preserves_each_grid_and_shared_builder() -> None:
    builder = _Builder(_grid())
    left = _Model(builder, [1.0, 2.0])
    right = _Model(builder, [3.0, 4.0, 5.0], shift=10.0)

    class _Forward:
        atmosphere_builder = builder
        models = {"left": left, "right": right}

        def __call__(self, parameters: dict[str, float]) -> dict[str, Spectrum]:
            return {"left": left(parameters), "right": right(parameters)}

    observation = Observation.from_arrays(
        [1.0, 2.0], [0.0, 0.0], [1.0, 1.0], mask=[True, False]
    )
    right_observation = Observation.from_arrays(
        [3.0, 4.0, 5.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    )
    observations = ObservationCollection(
        (
            ObservationDataset("left", observation, offset_parameter="left_offset"),
            ObservationDataset(
                "right", right_observation, offset_parameter="right_offset"
            ),
        )
    )

    class _Problem:
        forward_model = _Forward()
        likelihood = SimpleNamespace()
        calls = 0
        observations: ObservationCollection

        @staticmethod
        def parameter_mapping(vector: np.ndarray) -> dict[str, float]:
            return {
                "x": float(vector[0]),
                "left_offset": float(vector[1]),
                "right_offset": float(vector[2]),
            }

        def model_spectra(self, parameters: dict[str, float]) -> dict[str, Spectrum]:
            self.calls += 1
            return self.forward_model(parameters)

    _Problem.observations = observations
    problem = _Problem()
    draws = np.asarray([[0.0, 1.0, 2.0], [10.0, 3.0, 4.0], [20.0, 5.0, 6.0]])
    products = collect_posterior_models(problem, draws)

    assert set(products.spectra) == {"left", "right"}
    assert products.spectra["left"].wavelength.tolist() == [1.0, 2.0]
    assert products.spectra["right"].wavelength.tolist() == [3.0, 4.0, 5.0]
    np.testing.assert_allclose(
        products.spectra["left"].values[2],
        np.quantile(draws[:, 0] + draws[:, 1], 0.5) + np.arange(2),
    )
    np.testing.assert_allclose(
        products.spectra["right"].values[2],
        np.quantile(draws[:, 0] + draws[:, 2], 0.5) + 10.0 + np.arange(3),
    )
    assert products.spectra["left"].mask is not None
    assert products.spectra["left"].mask.tolist() == [True, False]
    assert problem.calls == len(draws)
    assert left.calls == right.calls == len(draws)
    assert builder.calls == len(draws)


def test_two_region_profiles_keep_named_composition() -> None:
    hot_builder = _Builder(_grid())
    cold_builder = _Builder(_grid())
    hot = _Model(hot_builder, [1.0, 2.0], shift=10.0)
    cold = _Model(cold_builder, [1.0, 2.0], shift=1.0)
    forward = TwoRegionEmissionModel(hot, cold, hot_fraction_parameter="fraction")

    class _Problem:
        forward_model = forward
        observation = SimpleNamespace(mask=None)
        likelihood = SimpleNamespace(offset_parameter=None)

        @staticmethod
        def parameter_mapping(vector: np.ndarray) -> dict[str, float]:
            return {"x": float(vector[0]), "fraction": float(vector[1])}

        def model_spectrum(self, parameters: dict[str, float]) -> Spectrum:
            return self.forward_model(parameters)

    products = collect_posterior_models(
        _Problem(),
        np.asarray([[0.0, 0.25], [10.0, 0.25], [20.0, 0.75]]),
    )

    assert set(products.temperature_profiles) == {"hot", "cold"}
    assert set(products.vmr_profiles["hot"]) == {"H2O", "CO"}
    assert set(products.vmr_profiles["cold"]) == {"H2O", "CO"}
    assert hot_builder.calls == cold_builder.calls == 3
    assert hot.calls == cold.calls == 3


def test_cloud_profiles_are_collected_or_reported_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _Builder(_grid())
    model = _Model(builder, [1.0, 2.0])
    model.cloud_model = object()
    problem = _SingleProblem(model)

    def supported(
        _model: object, state: AtmosphereState, parameters: dict[str, float]
    ) -> dict[str, dict[str, object]]:
        value = float(parameters["x"])
        return {
            "tau": {"values": np.full(state.n_layers, value), "unit": "optical_depth"}
        }

    monkeypatch.setattr(cloud_profiles_module, "cloud_profiles", supported)
    products = collect_posterior_models(problem, np.asarray([[1.0, 0.0], [3.0, 0.0]]))
    assert set(products.cloud_profiles["primary"]) == {"tau"}
    assert products.cloud_profiles["primary"]["tau"].value_unit == "optical_depth"

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise cloud_profiles_module.CloudProfileUnavailableError(
            "custom profile unavailable"
        )

    monkeypatch.setattr(cloud_profiles_module, "cloud_profiles", unavailable)
    unavailable_products = collect_posterior_models(
        problem,
        np.asarray([[1.0, 0.0], [3.0, 0.0]]),
    )
    assert unavailable_products.cloud_profiles == {}
    assert unavailable_products.unsupported_cloud_profiles["primary"][
        "cloud_profiles"
    ] == ("custom profile unavailable")


def test_inconsistent_spectrum_grid_is_rejected() -> None:
    builder = _Builder(_grid())

    class _ChangingModel(_Model):
        def __call__(self, parameters: Mapping[str, float]) -> Spectrum:
            self.calls += 1
            wavelength = [1.0, 2.0] if parameters["x"] < 1.0 else [1.0, 2.1]
            return Spectrum.from_arrays(
                wavelength, [1.0, 2.0], "eclipse_depth", "eclipse_depth"
            )

    problem = _SingleProblem(_ChangingModel(builder, [1.0, 2.0]))
    with pytest.raises(RobertValidationError, match="changed grid"):
        collect_posterior_models(problem, np.asarray([[0.0, 0.0], [2.0, 0.0]]))


def test_unsupported_prediction_has_typed_error() -> None:
    class _Problem:
        @staticmethod
        def parameter_mapping(vector: np.ndarray) -> dict[str, float]:
            return {"x": float(vector[0])}

        def predict(self, parameters: dict[str, float]) -> dict[str, object]:
            return {"hrs": object()}

    with pytest.raises(PosteriorProductUnavailableError, match="Spectrum"):
        collect_posterior_models(_Problem(), np.asarray([[1.0]]))
