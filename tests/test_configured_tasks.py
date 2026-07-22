"""Regression tests for configured retrieval-problem construction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robert_exoplanets import (
    CorrelatedKTable,
    Observation,
    ObservationCollection,
    ObservationDataset,
)
from robert_exoplanets.io import configured_tasks
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]


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
