"""Regression tests for configured retrieval-problem construction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets import (
    CorrelatedKTable,
    MultiDatasetTwoRegionEmissionModel,
    Observation,
    ObservationCollection,
    ObservationDataset,
)
from robert_exoplanets.io import configured_tasks
from robert_exoplanets.io.task_config import TaskConfig, load_task_config


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_evaluation_can_validate_an_explicit_oe_state() -> None:
    state = np.asarray([1.0, 2.0])

    class StubProblem:
        ndim = 2
        invalid_loglike = -1.0e100

        def prior_transform(self, cube):
            raise AssertionError("the midpoint must not be used")

        def log_likelihood_from_vector(self, vector):
            np.testing.assert_array_equal(vector, state)
            return -3.0

    smoke = configured_tasks.smoke_evaluation(StubProblem(), state)

    assert smoke["log_likelihood"] == -3.0


def test_smoke_evaluation_exposes_the_underlying_model_error() -> None:
    class StubProblem:
        ndim = 1
        invalid_loglike = -1.0e100

        def prior_transform(self, cube):
            return cube

        def log_likelihood_from_vector(self, vector):
            return self.invalid_loglike

        def gaussian_inputs_from_vector(self, vector):
            raise ValueError("temperature is outside opacity coverage")

    with pytest.raises(ValueError, match="outside opacity coverage"):
        configured_tasks.smoke_evaluation(StubProblem())


def _table(species: str) -> CorrelatedKTable:
    return CorrelatedKTable(
        species=species,
        pressure_bar=np.array([1.0e-3, 1.0]),
        temperature_K=np.array([500.0, 1500.0]),
        wavenumber_cm_inverse=np.array([5000.0, 4000.0]),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        kcoeff=np.full((2, 2, 2, 1), 1.0e-24),
    )


def test_fastchem_problem_has_no_phantom_opacity_species(monkeypatch) -> None:
    config = load_task_config(
        ROOT / "configurations" / "wasp69b_cloud_free_R1000.yaml"
    )
    observation = Observation.from_arrays(
        wavelength=[2.0, 2.5],
        flux=[1.0e-3, 1.1e-3],
        uncertainty=[1.0e-4, 1.0e-4],
        instrument="NIRCam",
    )
    observations = ObservationCollection(
        (ObservationDataset("f322w2", observation),)
    )
    captured_factories = {}

    monkeypatch.setattr(
        configured_tasks,
        "_load_cached_table",
        lambda _config, _dataset, species: _table(species),
    )
    monkeypatch.setattr(configured_tasks, "load_nemesispy_cia_table", lambda: None)

    class StubForwardModel:
        models = {"f322w2": SimpleNamespace(opacity_identifiers={})}

        def __call__(self, parameters):
            return {}

    def capture_factories(factories, *, spectral_grids):
        captured_factories.update(factories)
        assert tuple(spectral_grids) == ("f322w2",)
        return StubForwardModel()

    monkeypatch.setattr(
        configured_tasks,
        "build_multi_dataset_emission_model",
        capture_factories,
    )

    problem = configured_tasks.build_problem(config, observations)

    assert problem.name == config.run.name
    assert captured_factories["f322w2"].opacity_free_species == ()


def test_configured_two_region_problem_builds_independent_columns(monkeypatch) -> None:
    base = load_task_config(
        ROOT / "configurations" / "wasp69b_cloud_free_R1000.yaml"
    )
    raw = deepcopy(base.model_dump(mode="python"))
    species = tuple(raw["opacity"]["species"])
    raw["disk_emission"] = {
        "model": "two_region",
        "hot_fraction_parameter": "hot_area_fraction",
        "cold_region": {
            "atmosphere": {
                "temperature": {"model": "isothermal", "temperature_k": 900.0},
                "chemistry": {
                    "model": "free",
                    "species": species,
                    "fixed_mixing_ratios": {name: 1.0e-8 for name in species},
                },
            },
            "clouds": {"model": "deck_haze"},
        },
    }
    raw["parameters"] = (
        *raw["parameters"],
        {
            "name": "hot_area_fraction",
            "prior": {"type": "uniform", "lower": 0.0, "upper": 1.0},
        },
        *(
            {
                "name": name,
                "prior": {"type": "uniform", "lower": -12.0, "upper": 4.0},
            }
            for name in (
                "log_cloud_top_pressure_bar",
                "log_cloud_optical_depth",
                "log_haze_mass_extinction",
                "haze_slope",
            )
        ),
    )
    config = TaskConfig.model_validate(raw)
    observation = Observation.from_arrays(
        wavelength=[2.0, 2.5],
        flux=[1.0e-3, 1.1e-3],
        uncertainty=[1.0e-4, 1.0e-4],
        instrument="NIRCam",
    )
    observations = ObservationCollection(
        (ObservationDataset("f322w2", observation),)
    )
    captured = []

    monkeypatch.setattr(
        configured_tasks,
        "_load_cached_table",
        lambda _config, _dataset, item: _table(item),
    )
    monkeypatch.setattr(configured_tasks, "load_nemesispy_cia_table", lambda: None)

    class StubRegionalModel:
        models = {"f322w2": SimpleNamespace(opacity_identifiers={})}

        def __call__(self, parameters):
            return {}

    def capture(factories, *, spectral_grids):
        captured.append(factories["f322w2"])
        return StubRegionalModel()

    monkeypatch.setattr(configured_tasks, "build_multi_dataset_emission_model", capture)

    problem = configured_tasks.build_problem(config, observations)

    assert isinstance(problem.forward_model, MultiDatasetTwoRegionEmissionModel)
    assert len(captured) == 2
    assert captured[0].chemistry_model.__class__.__name__ == "FastChemEquilibriumChemistry"
    assert captured[0].cloud_model is None
    assert captured[1].chemistry_model.__class__.__name__ == "FreeChemistry"
    assert captured[1].cloud_model.__class__.__name__ == "ParameterizedDeckHazeCloudModel"
