"""Tests for named multi-instrument spectra and nuisance parameters."""

from __future__ import annotations

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
)


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
