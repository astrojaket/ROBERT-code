"""Tests for sampler-independent retrieval infrastructure."""

from __future__ import annotations

import json
import numpy as np
import pytest

from robert_exoplanets import (
    LogUniformPrior,
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    load_emission_observation_npz,
    load_observation_npz,
    run_optimal_estimation,
    run_retrieval,
    save_observation_npz,
)
from robert_exoplanets.core import RobertDataError
from robert_exoplanets.retrieval.manifest import build_run_manifest
from robert_exoplanets.retrieval.results import build_retrieval_result
from robert_exoplanets.retrieval.samplers.ultranest import (
    _result_from_ultranest,
    _validate_mpi_world_size,
)


def test_self_describing_npz_preserves_transit_depth_semantics(tmp_path) -> None:
    observation = Observation.from_arrays(
        wavelength=[1.0, 1.5],
        flux=[0.012, 0.0121],
        uncertainty=[1.0e-5, 1.0e-5],
        flux_unit="transit_depth",
        observable="transit_depth",
        instrument="synthetic transit",
    )
    path = save_observation_npz(observation, tmp_path / "transit.npz")

    loaded = load_observation_npz(path)

    assert loaded.flux_unit == "transit_depth"
    assert loaded.observable == "transit_depth"
    assert loaded.instrument == "synthetic transit"
    assert loaded.metadata["source_format"] == "npz_spectral_observation"


def test_parameter_set_transforms_unit_cube_and_log_prior() -> None:
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("offset", UniformPrior(-1.0, 1.0)),
            RetrievalParameter("scale", LogUniformPrior(1.0e-3, 1.0e-1)),
        )
    )

    vector = parameters.transform([0.75, 0.5])

    np.testing.assert_allclose(vector, [0.5, 1.0e-2])
    assert parameters.vector_to_mapping(vector) == {"offset": 0.5, "scale": pytest.approx(1.0e-2)}
    assert np.isfinite(parameters.log_prior_from_vector(vector))
    assert parameters.parameters[1].midpoint == pytest.approx(1.0e-2)


def test_load_emission_observation_npz_reads_hat_p_32b_style_keys(tmp_path) -> None:
    path = tmp_path / "obs.npz"
    np.savez(path, wavelength=[3.0, 4.0], data=[1.0e-3, 2.0e-3], err=[1.0e-4, 2.0e-4])

    observation = load_emission_observation_npz(path, instrument="G395H")

    np.testing.assert_allclose(observation.wavelength, [3.0, 4.0])
    np.testing.assert_allclose(observation.flux, [1.0e-3, 2.0e-3])
    assert observation.instrument == "G395H"
    assert observation.metadata["source_format"] == "npz_emission_observation"


def test_load_emission_observation_npz_requires_keys(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, wavelength=[3.0], data=[1.0])

    with pytest.raises(RobertDataError, match="uncertainty"):
        load_emission_observation_npz(path)


def test_retrieval_problem_loglike_and_oe_recover_linear_model(tmp_path) -> None:
    path = tmp_path / "obs.npz"
    wavelength = np.array([1.0, 2.0, 3.0, 4.0])
    flux = 2.0 + 0.5 * (wavelength - np.mean(wavelength))
    np.savez(path, wavelength=wavelength, data=flux, err=np.full_like(flux, 0.05))
    observation = load_emission_observation_npz(path, flux_unit="eclipse_depth")
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("baseline", UniformPrior(0.0, 4.0)),
            RetrievalParameter("slope", UniformPrior(-2.0, 2.0)),
        )
    )
    problem = RetrievalProblem(
        name="linear-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda p: Spectrum.from_arrays(
            observation.wavelength,
            p["baseline"] + p["slope"] * (observation.wavelength - np.mean(observation.wavelength)),
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
    )

    assert problem.log_likelihood_from_vector([2.0, 0.5]) == pytest.approx(0.0)
    result = run_optimal_estimation(problem, max_iterations=6)

    assert result.converged
    assert result.best_fit_parameters["baseline"] == pytest.approx(2.0, abs=1.0e-5)
    assert result.best_fit_parameters["slope"] == pytest.approx(0.5, abs=1.0e-4)
    np.testing.assert_allclose(result.covariance, result.covariance.T, rtol=0.0, atol=0.0)


def test_optimal_estimation_honors_observation_mask(tmp_path) -> None:
    observation = load_emission_observation_npz_from_arrays()
    observation = type(observation).from_arrays(
        wavelength=[1.0, 2.0, 3.0],
        flux=[1.25, 999.0, 1.25],
        uncertainty=[0.05, 0.05, 0.05],
        mask=[True, False, True],
    )
    parameters = RetrievalParameterSet((RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),))
    problem = RetrievalProblem(
        name="masked-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda p: Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, p["baseline"]),
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
    )

    result = run_optimal_estimation(problem, max_iterations=6)

    assert result.best_fit_parameters["baseline"] == pytest.approx(1.25, abs=5.0e-4)


def test_run_retrieval_writes_manifest_and_unified_result(tmp_path) -> None:
    observation = load_emission_observation_npz_from_arrays()
    parameters = RetrievalParameterSet((RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),))
    problem = RetrievalProblem(
        name="serialized-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda p: Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, p["baseline"]),
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
        opacity_identifiers={"H2O": "sha256:test"},
    )

    result = run_retrieval(
        problem,
        method="optimal_estimation",
        output_dir=tmp_path,
        max_iterations=4,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    with np.load(tmp_path / "result_arrays.npz") as arrays:
        assert arrays["state_vector"].shape == (1,)
        assert arrays["covariance"].shape == (1, 1)
    assert result.method == "optimal_estimation"
    assert manifest["schema_version"] == "1.0"
    assert manifest["opacity_identifiers"] == {"H2O": "sha256:test"}
    assert summary["metadata"]["config_hash"] == manifest["config_hash"]


def test_run_retrieval_dispatch_rejects_missing_ultranest_output_dir() -> None:
    observation = load_emission_observation_npz_from_arrays()
    parameters = RetrievalParameterSet((RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),))
    problem = RetrievalProblem(
        name="dispatch-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda p: Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, p["baseline"]),
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
    )

    with pytest.raises(ValueError, match="output_dir"):
        run_retrieval(problem, method="ultranest")


def test_ultranest_result_adapter_extracts_best_fit() -> None:
    observation = load_emission_observation_npz_from_arrays()
    parameters = RetrievalParameterSet((RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),))
    problem = RetrievalProblem(
        name="adapter-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda p: Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, p["baseline"]),
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        ),
    )
    raw_result = {
        "weighted_samples": {
            "points": np.array([[0.8], [1.0], [1.2]]),
            "weights": np.array([0.2, 0.6, 0.2]),
            "logl": np.array([-2.0, 0.0, -1.0]),
        },
        "logz": 10.0,
        "logzerr": 0.1,
        "ncall": 500,
        "maximum_likelihood": {"point": np.array([1.0]), "logl": 0.0},
    }

    result = _result_from_ultranest(problem, raw_result, log_dir=tmp_path_like(), mpi_nprocs=3)

    assert result.method == "ultranest"
    assert result.log_evidence == pytest.approx(10.0)
    assert result.best_fit_parameters == {"baseline": 1.0}
    assert result.metadata["mpi_nprocs"] == "3"
    assert result.metadata["ncall"] == "500"

    manifest = build_run_manifest(
        problem,
        method="ultranest",
        settings={"min_num_live_points": 40},
        random_seed=42,
    )
    stable_result = build_retrieval_result(result, manifest=manifest, output_dir=tmp_path_like())

    assert stable_result.metadata["mpi_nprocs"] == "3"
    assert stable_result.metadata["ncall"] == "500"


def test_ultranest_rejects_requested_mpi_size_that_is_not_present() -> None:
    from robert_exoplanets.core import RobertConfigError

    with pytest.raises(RobertConfigError, match="MPI communicator size mismatch"):
        _validate_mpi_world_size(2)


def test_ultranest_accepts_the_actual_single_process_communicator() -> None:
    _validate_mpi_world_size(1)


def load_emission_observation_npz_from_arrays():
    from robert_exoplanets.instruments import Observation

    return Observation.from_arrays(wavelength=[1.0, 2.0], flux=[1.0, 1.0], uncertainty=[0.1, 0.1])


def tmp_path_like():
    from pathlib import Path

    return Path("/tmp/ultranest-test")
