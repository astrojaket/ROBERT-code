"""Tests for sampler-independent retrieval infrastructure."""

from __future__ import annotations

import json
import numpy as np
import pytest

from robert_exoplanets import (
    CenteredLogRatioPrior,
    CorrelatedGaussianLikelihood,
    GaussianLikelihood,
    LogUniformPrior,
    MixedMultiDatasetLikelihood,
    MultiDatasetRetrievalProblem,
    Observation,
    ObservationCollection,
    ObservationDataset,
    PolynomialContinuumLikelihood,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    centered_log_ratio_prior_transform,
    load_observation_npz,
    run_optimal_estimation,
    run_retrieval,
    save_observation_npz,
)
from robert_exoplanets.core import RobertConfigError, RobertDataError



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


def test_centered_log_ratio_prior_closes_composition_without_privileged_gas() -> None:
    free_log_vmr = centered_log_ratio_prior_transform(
        [0.25, 0.50, 0.75], lower_log10_vmr=-12.0
    )
    free_vmr = 10.0**free_log_vmr
    derived_vmr = 1.0 - float(np.sum(free_vmr))

    assert np.all(free_log_vmr >= -12.0)
    assert np.all(free_log_vmr <= 0.0)
    assert derived_vmr >= 1.0e-12
    assert derived_vmr + float(np.sum(free_vmr)) == pytest.approx(1.0)


def test_centered_log_ratio_prior_rejects_cube_outside_simplex() -> None:
    transformed = centered_log_ratio_prior_transform(
        [1.0, 1.0, 1.0], lower_log10_vmr=-12.0
    )

    np.testing.assert_allclose(transformed, np.full(3, -50.0))


@pytest.mark.parametrize(
    ("cube", "poseidon_free_log10_vmr"),
    [
        ([0.25], [-6.000000434294265]),
        (
            [0.25, 0.50, 0.75],
            [-8.82109965230034, -4.410559887560278, -2.0122820214138306e-05],
        ),
    ],
)
def test_centered_log_ratio_matches_poseidon_reference_vectors(
    cube, poseidon_free_log10_vmr
) -> None:
    """Values frozen from POSEIDON CLR_Prior at commit d1632214c9f3."""

    np.testing.assert_allclose(
        centered_log_ratio_prior_transform(cube),
        poseidon_free_log10_vmr,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_parameter_set_applies_named_centered_log_ratio_group_jointly() -> None:
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("temperature", UniformPrior(200.0, 800.0)),
            RetrievalParameter(
                "log_SO2", CenteredLogRatioPrior(-12.0, 0.0, "atmosphere")
            ),
            RetrievalParameter(
                "log_CO2", CenteredLogRatioPrior(-12.0, 0.0, "atmosphere")
            ),
        )
    )

    vector = parameters.transform([0.5, 0.25, 0.75])
    mapping = parameters.vector_to_mapping(vector)

    assert mapping["temperature"] == pytest.approx(500.0)
    assert 10.0 ** mapping["log_SO2"] + 10.0 ** mapping["log_CO2"] < 1.0
    np.testing.assert_allclose(
        parameters.midpoint_vector(), parameters.transform([0.5, 0.5, 0.5])
    )


def test_parameter_set_rejects_invalid_centered_log_ratio_sentinel() -> None:
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("log_A", CenteredLogRatioPrior()),
            RetrievalParameter("log_B", CenteredLogRatioPrior()),
        )
    )

    with pytest.raises(ValueError, match="composition simplex"):
        parameters.vector_to_mapping([-50.0, -50.0])


def test_load_observation_npz_reads_minimal_array_keys(tmp_path) -> None:
    path = tmp_path / "obs.npz"
    np.savez(path, wavelength=[3.0, 4.0], data=[1.0e-3, 2.0e-3], err=[1.0e-4, 2.0e-4])

    observation = load_observation_npz(path, instrument="G395H")

    np.testing.assert_allclose(observation.wavelength, [3.0, 4.0])
    np.testing.assert_allclose(observation.flux, [1.0e-3, 2.0e-3])
    assert observation.instrument == "G395H"
    assert observation.metadata["source_format"] == "npz_spectral_observation"


def test_load_observation_npz_requires_keys(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, wavelength=[3.0], data=[1.0])

    with pytest.raises(RobertDataError, match="uncertainty"):
        load_observation_npz(path)


def test_retrieval_problem_loglike_and_oe_recover_linear_model(tmp_path) -> None:
    path = tmp_path / "obs.npz"
    wavelength = np.array([1.0, 2.0, 3.0, 4.0])
    flux = 2.0 + 0.5 * (wavelength - np.mean(wavelength))
    np.savez(path, wavelength=wavelength, data=flux, err=np.full_like(flux, 0.05))
    observation = load_observation_npz(path, flux_unit="eclipse_depth")
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


def test_optimal_estimation_rejects_correlated_single_dataset_likelihood() -> None:
    observation = _oe_observation([1.0, 2.0])
    problem = _single_oe_problem(
        observation,
        CorrelatedGaussianLikelihood([[1.0, 0.2], [0.2, 1.0]]),
    )

    with pytest.raises(RobertConfigError, match="supported independent Gaussian"):
        run_optimal_estimation(problem)


def test_optimal_estimation_rejects_correlated_child_in_mixed_multi_dataset_problem() -> None:
    problem = _mixed_oe_problem(
        MixedMultiDatasetLikelihood(
            {
                "independent": GaussianLikelihood(
                    offset_parameter=None,
                    jitter_parameter=None,
                ),
                "second": CorrelatedGaussianLikelihood(
                    [[1.0, 0.2], [0.2, 1.0]]
                ),
            }
        )
    )

    with pytest.raises(RobertConfigError, match="supported independent Gaussian"):
        run_optimal_estimation(problem)


def test_optimal_estimation_rejects_profiled_likelihood() -> None:
    observation = _oe_observation([1.0, 2.0, 3.0])
    problem = _single_oe_problem(
        observation,
        PolynomialContinuumLikelihood(degree=0).prepare(observation),
    )

    with pytest.raises(RobertConfigError, match="supported independent Gaussian"):
        run_optimal_estimation(problem)


def test_optimal_estimation_accepts_mixed_all_gaussian_likelihood() -> None:
    problem = _mixed_oe_problem(
        MixedMultiDatasetLikelihood(
            {
                "independent": GaussianLikelihood(
                    offset_parameter=None,
                    jitter_parameter=None,
                ),
                "second": GaussianLikelihood(
                    offset_parameter=None,
                    jitter_parameter=None,
                ),
            }
        ),
        flux=1.25,
    )

    result = run_optimal_estimation(problem, max_iterations=6)

    assert result.converged
    assert result.best_fit_parameters["baseline"] == pytest.approx(1.25, abs=1.0e-3)


def _oe_observation(wavelength: list[float], *, flux: float = 1.0) -> Observation:
    return Observation.from_arrays(
        wavelength=wavelength,
        flux=np.full(len(wavelength), flux),
        uncertainty=np.full(len(wavelength), 0.1),
    )


def _single_oe_problem(observation: Observation, likelihood: object) -> RetrievalProblem:
    parameters = RetrievalParameterSet(
        (RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),)
    )
    return RetrievalProblem(
        name="oe-likelihood-contract-test",
        observation=observation,
        parameters=parameters,
        forward_model=lambda p: _constant_spectrum(observation, p["baseline"]),
        likelihood=likelihood,
    )


def _mixed_oe_problem(
    likelihood: MixedMultiDatasetLikelihood,
    *,
    flux: float = 1.0,
) -> MultiDatasetRetrievalProblem:
    independent_observation = _oe_observation([1.0, 2.0], flux=flux)
    second_observation = _oe_observation([3.0, 4.0], flux=flux)
    observations = ObservationCollection(
        (
            ObservationDataset("independent", independent_observation),
            ObservationDataset("second", second_observation),
        )
    )
    parameters = RetrievalParameterSet(
        (RetrievalParameter("baseline", UniformPrior(0.0, 2.0)),)
    )

    def forward(parameter_values: dict[str, float]) -> dict[str, Spectrum]:
        baseline = parameter_values["baseline"]
        return {
            "independent": _constant_spectrum(independent_observation, baseline),
            "second": _constant_spectrum(second_observation, baseline),
        }

    return MultiDatasetRetrievalProblem(
        name="mixed-oe-likelihood-contract-test",
        observations=observations,
        parameters=parameters,
        forward_model=forward,
        likelihood=likelihood,
    )


def _constant_spectrum(observation: Observation, value: float) -> Spectrum:
    return Spectrum.from_arrays(
        observation.wavelength,
        np.full(observation.n_points, value),
        unit=observation.flux_unit,
        observable=observation.observable,
        wavelength_unit=observation.wavelength_unit,
    )


def test_optimal_estimation_honors_observation_mask(tmp_path) -> None:
    observation = load_observation_npz_from_arrays()
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
    observation = load_observation_npz_from_arrays()
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


def test_run_retrieval_dispatch_rejects_missing_multinest_output_dir() -> None:
    observation = load_observation_npz_from_arrays()
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
        run_retrieval(problem, method="multinest")


def load_observation_npz_from_arrays():
    from robert_exoplanets.instruments import Observation

    return Observation.from_arrays(wavelength=[1.0, 2.0], flux=[1.0, 1.0], uncertainty=[0.1, 0.1])
