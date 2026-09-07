"""Tests for portable best-fit prediction products."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robert_exoplanets import Observation, Spectrum
from robert_exoplanets.instruments import HighResolutionEmissionTemplate
from robert_exoplanets.retrieval import (
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)
from robert_exoplanets.retrieval.manifest import RunManifest
from robert_exoplanets.retrieval.optimal_estimation import OptimalEstimationResult
from robert_exoplanets.retrieval.predictions import (
    BestFitPredictionArtifact,
    BestFitPredictionDataset,
    capture_best_fit_prediction,
    load_best_fit_prediction,
    write_best_fit_prediction,
)
from robert_exoplanets.retrieval.results import RetrievalResult, write_retrieval_result


def _observation() -> Observation:
    return Observation.from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[1.0, 2.0, 3.0],
        uncertainty=[0.1, 0.2, 0.3],
        mask=[True, False, True],
        wavelength_bin_edges=[0.5, 1.5, 2.5, 3.5],
        instrument="synthetic",
    )


def test_best_fit_prediction_roundtrip_preserves_science_arrays(tmp_path: Path) -> None:
    observation = _observation()
    problem = SimpleNamespace(
        name="portable-test",
        observation=observation,
        parameter_names=("level",),
        opacity_identifiers={"H2O": "sha256:test"},
        metadata={"source": "unit-test"},
        forward_model=SimpleNamespace(),
        predict=lambda parameters: Spectrum.from_arrays(
            observation.wavelength,
            [1.1, 1.9, 3.2],
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
    )

    artifact = capture_best_fit_prediction(problem, {"level": 1.0})
    assert artifact.status == "available"
    assert artifact.available_dataset_names == ("primary",)
    assert artifact.opacity_identifiers["H2O"] == "sha256:test"
    assert artifact.datasets["primary"].spectrum.values.flags.writeable is False

    write_best_fit_prediction(artifact, tmp_path)
    restored = load_best_fit_prediction(tmp_path)
    restored_dataset = restored.datasets["primary"]
    assert restored.status == "available"
    np.testing.assert_array_equal(restored_dataset.spectrum.values, [1.1, 1.9, 3.2])
    np.testing.assert_array_equal(restored_dataset.observation.mask, [True, False, True])
    np.testing.assert_array_equal(
        restored_dataset.observation.wavelength_bin_edges,
        [0.5, 1.5, 2.5, 3.5],
    )
    assert restored_dataset.observation.instrument == "synthetic"
    assert restored_dataset.observation.wavelength_unit == "micron"
    assert restored_dataset.observation.flux_unit == "eclipse_depth"
    assert restored_dataset.observation.observable == "eclipse_depth"
    assert restored_dataset.spectrum.unit == "eclipse_depth"
    assert restored_dataset.spectrum.observable == "eclipse_depth"
    assert restored.provenance["parameter_source"] == "best_fit_parameters"


def test_heterogeneous_capture_keeps_hrs_unavailable_without_flattening() -> None:
    observation = _observation()

    class Likelihood:
        def loglike(self, prediction, *args):
            return 0.0

    likelihood = HeterogeneousLikelihood(
        (
            HeterogeneousLikelihoodComponent(
                name="low-resolution",
                prediction_key="lrs",
                likelihood=Likelihood(),
                observation=observation,
            ),
            HeterogeneousLikelihoodComponent(
                name="high-resolution",
                prediction_key="hrs",
                likelihood=Likelihood(),
                observation=None,
            ),
        )
    )
    problem = HeterogeneousRetrievalProblem(
        name="heterogeneous-portable-test",
        parameters=RetrievalParameterSet(
            (RetrievalParameter("level", UniformPrior(0.0, 2.0)),)
        ),
        forward_model=lambda parameters: {},
        likelihood=likelihood,
    )
    lrs = Spectrum.from_arrays(
        observation.wavelength,
        observation.flux,
        unit=observation.flux_unit,
        observable=observation.observable,
    )
    hrs = HighResolutionEmissionTemplate(
        wavelength=np.linspace(1.0, 3.0, 8),
        planet_flux=np.ones(8),
        stellar_flux=np.ones(8),
    )

    artifact = capture_best_fit_prediction(
        problem,
        {"level": 1.0},
        prediction={"lrs": lrs, "hrs": hrs},
    )
    assert artifact.status == "partial"
    assert artifact.available_dataset_names == ("lrs",)
    assert artifact.unavailable_dataset_names == ("hrs",)
    assert "one-dimensional" in (artifact.datasets["hrs"].reason or "")


def test_status_only_artifact_roundtrips_without_spectrum_arrays(tmp_path: Path) -> None:
    artifact = BestFitPredictionArtifact(
        datasets={
            "hrs": BestFitPredictionDataset(
                name="hrs",
                status="unavailable",
                reason="prepared HRS cube has no portable one-dimensional view",
            ),
        },
        problem_name="status-only",
        status="unavailable",
    )
    write_best_fit_prediction(artifact, tmp_path)
    restored = load_best_fit_prediction(tmp_path)
    assert restored.status == "unavailable"
    assert restored.datasets["hrs"].spectrum is None
    assert restored.datasets["hrs"].reason.startswith("prepared HRS cube")


def test_result_writer_references_prediction_artifact(tmp_path: Path) -> None:
    observation = _observation()
    spectrum = Spectrum.from_arrays(
        observation.wavelength,
        observation.flux,
        unit=observation.flux_unit,
        observable=observation.observable,
    )
    artifact = BestFitPredictionArtifact(
        datasets={
            "primary": BestFitPredictionDataset(
                name="primary",
                spectrum=spectrum,
                observation=observation,
            )
        },
        problem_name="writer-test",
        parameter_names=("level",),
        best_fit_parameters={"level": 1.0},
    )
    inference = OptimalEstimationResult(
        parameter_names=("level",),
        state_vector=np.array([1.0]),
        covariance=np.array([[0.01]]),
        averaging_kernel=np.eye(1),
        cost=0.0,
        log_likelihood=0.0,
        n_iterations=1,
        converged=True,
        message="converged",
    )
    manifest = RunManifest(
        problem_name="writer-test",
        method="optimal_estimation",
        created_at_utc="2026-01-01T00:00:00+00:00",
        config_hash="sha256:test",
        parameter_names=("level",),
        parameter_priors=(),
        likelihood={},
        problem_metadata={},
        opacity_identifiers={},
        settings={},
        random_seed=None,
    )
    result = RetrievalResult(
        method="optimal_estimation",
        parameter_names=("level",),
        best_fit_parameters={"level": 1.0},
        best_fit_log_likelihood=0.0,
        converged=True,
        message="converged",
        manifest=manifest,
        output_dir=tmp_path,
        inference_result=inference,
    )

    write_retrieval_result(result, prediction_artifact=artifact)
    summary = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert summary["best_fit_prediction"]["status"] == "available"
    assert summary["best_fit_prediction"]["json"] == "best_fit_prediction.json"
    restored = load_best_fit_prediction(tmp_path)
    np.testing.assert_array_equal(restored.spectra["primary"].values, observation.flux)
