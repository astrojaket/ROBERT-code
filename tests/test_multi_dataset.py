"""Tests for named multi-instrument spectra and nuisance parameters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    NativeSpectrumMultiDatasetForwardModel,
    MultiDatasetGaussianLikelihood,
    MultiDatasetRetrievalProblem,
    LinearObservationResponse,
    Observation,
    ObservationCollection,
    ObservationDataset,
    SpectralGrid,
    Spectrum,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
    run_retrieval,
)
from robert_exoplanets.io import configured_tasks
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]


def _observation(wavelength, flux, instrument):
    return Observation.from_arrays(
        wavelength,
        flux,
        np.full(len(wavelength), 0.1),
        instrument=instrument,
    )


def test_multi_dataset_forward_model_retains_dataset_identity() -> None:
    collection = ObservationCollection(
        (
            ObservationDataset(
                "nircam", _observation([2.0, 3.0], [1.0, 2.0], "NIRCam")
            ),
            ObservationDataset("miri", _observation([5.0, 8.0], [3.0, 4.0], "MIRI")),
        )
    )
    native = Spectrum(
        spectral_grid=SpectralGrid.from_array([2.0, 3.0, 5.0, 8.0], unit="micron"),
        values=np.array([1.0, 2.0, 3.0, 4.0]),
        unit="eclipse_depth",
        observable="eclipse_depth",
    )
    model = NativeSpectrumMultiDatasetForwardModel(
        lambda parameters: native,
        collection,
        response=LinearObservationResponse(),
    )

    prediction = model({})

    assert tuple(prediction.spectra) == ("nircam", "miri")
    np.testing.assert_allclose(prediction.spectra["miri"].values, [3.0, 4.0])


def test_multi_dataset_likelihood_applies_only_named_dataset_offset() -> None:
    collection = ObservationCollection(
        (
            ObservationDataset("nircam", _observation([2.0], [1.0], "NIRCam")),
            ObservationDataset(
                "miri",
                _observation([5.0], [3.0], "MIRI"),
                offset_parameter="miri_offset",
            ),
        )
    )
    predictions = {
        "nircam": Spectrum.from_arrays([2.0], [1.0], "eclipse_depth", "eclipse_depth"),
        "miri": Spectrum.from_arrays([5.0], [2.5], "eclipse_depth", "eclipse_depth"),
    }

    loglike = MultiDatasetGaussianLikelihood().loglike(
        predictions,
        collection,
        {"miri_offset": 0.5},
    )

    assert loglike == 0.0


def test_multi_dataset_likelihood_inflates_only_named_dataset_uncertainty() -> None:
    collection = ObservationCollection(
        (
            ObservationDataset("nircam", _observation([2.0], [1.0], "NIRCam")),
            ObservationDataset(
                "miri",
                _observation([5.0], [3.0], "MIRI"),
                uncertainty_scale_parameter="miri_error_scale",
            ),
        )
    )
    predictions = {
        "nircam": Spectrum.from_arrays([2.0], [1.0], "eclipse_depth", "eclipse_depth"),
        "miri": Spectrum.from_arrays([5.0], [2.8], "eclipse_depth", "eclipse_depth"),
    }

    likelihood = MultiDatasetGaussianLikelihood()
    default = likelihood.loglike(predictions, collection, {})
    inflated = likelihood.loglike(
        predictions,
        collection,
        {"miri_error_scale": 2.0},
    )

    assert default == pytest.approx(-2.0)
    assert inflated == pytest.approx(-0.5)


def test_collection_reports_total_points() -> None:
    collection = ObservationCollection(
        (
            ObservationDataset("a", _observation([2.0, 3.0], [1.0, 2.0], "a")),
            ObservationDataset("b", _observation([5.0], [3.0], "b")),
        )
    )

    assert collection.n_datasets == 2
    assert collection.n_points == 3


def test_multi_dataset_retrieval_problem_includes_dataset_offset() -> None:
    collection = ObservationCollection(
        (
            ObservationDataset("nircam", _observation([2.0], [1.0], "NIRCam")),
            ObservationDataset(
                "miri",
                _observation([5.0], [3.0], "MIRI"),
                offset_parameter="miri_offset",
            ),
        )
    )
    predictions = {
        "nircam": Spectrum.from_arrays([2.0], [1.0], "eclipse_depth", "eclipse_depth"),
        "miri": Spectrum.from_arrays([5.0], [2.5], "eclipse_depth", "eclipse_depth"),
    }
    problem = MultiDatasetRetrievalProblem(
        name="test",
        observations=collection,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("miri_offset", UniformPrior(-1.0, 1.0)),)
        ),
        forward_model=lambda parameters: predictions,
    )

    assert problem.log_likelihood_from_vector([0.5]) == 0.0
    assert set(problem.model_spectra([0.5])) == {"nircam", "miri"}


def test_multi_dataset_optimal_estimation_writes_dataset_likelihood_manifest(
    tmp_path,
) -> None:
    collection = ObservationCollection(
        (
            ObservationDataset("nircam", _observation([2.0], [1.0], "NIRCam")),
            ObservationDataset(
                "miri",
                _observation([5.0], [3.0], "MIRI"),
                offset_parameter="miri_offset",
                uncertainty_scale=1.2,
            ),
        )
    )
    predictions = {
        "nircam": Spectrum.from_arrays([2.0], [1.0], "eclipse_depth", "eclipse_depth"),
        "miri": Spectrum.from_arrays([5.0], [2.5], "eclipse_depth", "eclipse_depth"),
    }
    problem = MultiDatasetRetrievalProblem(
        name="multi-dataset-oe",
        observations=collection,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("miri_offset", UniformPrior(-1.0, 1.0)),)
        ),
        forward_model=lambda parameters: predictions,
    )

    result = run_retrieval(
        problem,
        method="optimal_estimation",
        output_dir=tmp_path,
        max_iterations=6,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert result.best_fit_parameters["miri_offset"] == pytest.approx(0.5, abs=0.02)
    assert manifest["likelihood"]["type"] == "MultiDatasetGaussianLikelihood"
    assert manifest["likelihood"]["datasets"] == [
        {
            "name": "nircam",
            "offset_parameter": None,
            "jitter_parameter": None,
            "uncertainty_scale_parameter": None,
            "uncertainty_scale": 1.0,
        },
        {
            "name": "miri",
            "offset_parameter": "miri_offset",
            "jitter_parameter": None,
            "uncertainty_scale_parameter": None,
            "uncertainty_scale": 1.2,
        },
    ]
    assert (tmp_path / "result.json").is_file()


def test_smoke_preflight_checks_both_hybrid_manifests(monkeypatch) -> None:
    config = load_task_config(
        ROOT
        / "configurations"
        / "wasp69b_cloud_free_native_pg14_R1000_optimal_estimation_to_multinest.yaml"
    )
    calls = []

    def capture_manifest(problem, *, method, settings, random_seed):
        calls.append((method, settings, random_seed))

    monkeypatch.setattr(configured_tasks, "build_run_manifest", capture_manifest)

    configured_tasks._preflight_retrieval_manifests(config, object())

    assert [call[0] for call in calls] == ["optimal_estimation", "multinest"]
    assert calls[0][2] is None
    assert calls[1][2] == config.sampler.seed
