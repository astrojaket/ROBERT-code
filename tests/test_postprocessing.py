"""Tests for sampler-independent retrieval and forward post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robert_exoplanets import (
    IsothermalTemperatureProfile,
    Observation,
    PressureGrid,
    Spectrum,
)
from robert_exoplanets.instruments import ObservationCollection, ObservationDataset
from robert_exoplanets.postprocessing import (
    calculate_fit_statistics,
    discover_retrieval_result_directories,
    posterior_summary,
    postprocess_forward_output,
    postprocess_retrieval_output,
    weighted_quantile,
)
from robert_exoplanets.retrieval import (
    MultiDatasetRetrievalProblem,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)
def _problem() -> MultiDatasetRetrievalProblem:
    observation = Observation.from_arrays(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [0.1, 0.1, 0.2],
        instrument="synthetic",
    )
    observations = ObservationCollection(
        (ObservationDataset("synthetic", observation),)
    )

    class Forward:
        atmosphere_builder = SimpleNamespace(
            temperature_profile=IsothermalTemperatureProfile(
                parameter_name="level"
            ),
            pressure_grid=PressureGrid.logspace(1.0e-5, 10.0, 5, unit="bar"),
        )

        def __call__(self, parameters):
            level = parameters["level"]
            return {
                "synthetic": Spectrum.from_arrays(
                    observation.wavelength,
                    level * observation.wavelength,
                    unit=observation.flux_unit,
                    observable=observation.observable,
                )
            }

    return MultiDatasetRetrievalProblem(
        name="synthetic-postprocessing",
        observations=observations,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("level", UniformPrior(0.5, 1.5)),)
        ),
        forward_model=Forward(),
    )


def test_fit_statistics_and_weighted_quantiles() -> None:
    problem = _problem()
    parameters = {"level": 1.0}
    statistics = calculate_fit_statistics(
        problem,
        problem.model_spectra(parameters),
        parameters,
        fitted_parameter_count=1,
    )

    assert statistics["chi_squared"] == 0.0
    assert statistics["reduced_chi_squared"] == 0.0
    assert statistics["degrees_of_freedom"] == 2
    assert statistics["per_dataset"]["synthetic"]["number_points"] == 3
    quantile = weighted_quantile(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.25, 0.5, 0.25]),
        (0.5,),
    )
    assert quantile[0] == 0.5


def test_retrieval_postprocessing_writes_statistics_and_plots(tmp_path: Path) -> None:
    problem = _problem()
    result_dir = tmp_path / "outputs" / "ultranest"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "parameter_names": ["level"],
                "best_fit_parameters": {"level": 1.0},
                "best_fit_log_likelihood": 0.0,
                "method": "ultranest",
                "converged": True,
                "message": "finished",
                "log_evidence": -1.0,
                "log_evidence_error": 0.1,
                "metadata": {"inference_elapsed_seconds": "12.5"},
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        result_dir / "result_arrays.npz",
        samples=np.array([[0.9], [1.0], [1.1]]),
        weights=np.array([0.2, 0.6, 0.2]),
        log_likelihood=np.array([-1.0, 0.0, -1.0]),
    )

    plot_dir = tmp_path / "plots" / "ultranest"
    diagnostics = postprocess_retrieval_output(
        problem,
        result_dir,
        plot_dir=plot_dir,
        native_spectrum_model=lambda parameters: Spectrum.from_arrays(
            np.linspace(1.0, 3.0, 9),
            parameters["level"] * np.linspace(1.0, 3.0, 9),
            unit="relative_flux",
            observable="relative_flux",
        ),
    )

    assert diagnostics["reduced_chi_squared"] == 0.0
    assert diagnostics["posterior"]["effective_sample_size"] > 1.0
    assert (plot_dir / "fit_statistics.json").is_file()
    assert (plot_dir / "posterior_summary.json").is_file()
    assert (plot_dir / "fit_spectrum_residuals.png").is_file()
    assert (plot_dir / "posterior_marginals.png").is_file()
    assert (plot_dir / "parameter_correlation.png").is_file()
    assert (plot_dir / "posterior_corner.png").is_file()
    assert (plot_dir / "temperature_profiles.png").is_file()
    predictive_path = plot_dir / "posterior_predictive_quantiles.npz"
    assert predictive_path.is_file()
    with np.load(predictive_path) as predictive:
        assert "spectrum_synthetic_q16" in predictive
        assert "spectrum_synthetic_q50" in predictive
        assert "spectrum_synthetic_q84" in predictive
        assert "native_q50" in predictive
        assert "temperature_primary_q50_K" in predictive
    assert diagnostics["posterior_predictive_draws"] == 3
    assert diagnostics["native_opacity_spectrum"] is True
    assert discover_retrieval_result_directories(tmp_path / "outputs") == (result_dir,)


def test_forward_postprocessing_writes_statistics_and_plot(tmp_path: Path) -> None:
    problem = _problem()
    forward_path = tmp_path / "forward_model.npz"
    np.savez_compressed(
        forward_path,
        synthetic_wavelength_micron=np.array([1.0, 2.0, 3.0]),
        synthetic_model=np.array([1.0, 2.0, 3.0]),
        parameter_level=1.0,
    )

    plot_dir = tmp_path / "plots" / "forward"
    diagnostics = postprocess_forward_output(
        problem,
        forward_path,
        plot_dir=plot_dir,
    )

    assert diagnostics["chi_squared"] == 0.0
    assert (plot_dir / "fit_statistics.json").is_file()
    assert (plot_dir / "forward_parameters.json").is_file()
    assert (plot_dir / "forward_spectrum_residuals.png").is_file()


def test_optimal_estimation_posterior_summary() -> None:
    summary = posterior_summary(
        ("a", "b"),
        {
            "state_vector": np.array([1.0, 2.0]),
            "covariance": np.diag([0.04, 0.25]),
        },
    )

    assert summary["kind"] == "optimal_estimation_gaussian"
    assert summary["quantiles_16_50_84"]["a"] == [0.8, 1.0, 1.2]
