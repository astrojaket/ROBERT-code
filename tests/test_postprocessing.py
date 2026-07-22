"""Tests for sampler-independent retrieval and forward post-processing."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
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
from robert_exoplanets.io.task_config import load_task_config
from scripts.create_run_directory import create_run_directory


ROOT = Path(__file__).resolve().parents[1]


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


def test_benchmark_comparison_script_writes_partial_matrix(tmp_path: Path) -> None:
    project = tmp_path / "my_project"
    for config_name in (
        "wasp69b_cloud_free_native_pg14_R1000.yaml",
        "wasp69b_cloud_free_native_pg14_R1000_multinest.yaml",
    ):
        run_dir = create_run_directory(
            project_dir=project,
            source_config=ROOT / "configurations" / config_name,
        )
        config = load_task_config(run_dir / "configuration.yaml")
        result_dir = config.outputs.directory / config.sampler.engine
        result_dir.mkdir(parents=True)
        names = tuple(parameter.name for parameter in config.parameters)
        state = np.array(
            [0.5 * (parameter.prior.lower + parameter.prior.upper) for parameter in config.parameters]
        )
        samples = np.vstack((state * 0.99, state, state * 1.01))
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "parameter_names": names,
                    "best_fit_parameters": dict(zip(names, state, strict=True)),
                    "best_fit_log_likelihood": -10.0,
                    "method": config.sampler.engine,
                    "converged": True,
                    "message": "finished",
                    "log_evidence": -12.0,
                    "log_evidence_error": 0.2,
                    "metadata": {
                        "inference_elapsed_seconds": "3600.0",
                        "ncall": "1000",
                    },
                }
            ),
            encoding="utf-8",
        )
        np.savez(
            result_dir / "result_arrays.npz",
            samples=samples,
            weights=np.array([0.2, 0.6, 0.2]),
            log_likelihood=np.array([-11.0, -10.0, -11.0]),
        )
        plot_dir = config.outputs.directory / "plots" / result_dir.name
        plot_dir.mkdir(parents=True)
        (plot_dir / "fit_statistics.json").write_text(
            json.dumps(
                {
                    "chi_squared": 20.0,
                    "reduced_chi_squared": 1.1,
                    "aic": 40.0,
                    "aicc": 41.0,
                    "bic": 50.0,
                }
            ),
            encoding="utf-8",
        )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "postprocess_wasp69b_sampler_benchmark.py"),
            "--project-dir",
            str(project),
            "--allow-incomplete",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = project / "benchmark_comparison"
    assert (output / "benchmark_summary.json").is_file()
    assert (output / "benchmark_summary.csv").is_file()
    assert (output / "benchmark_runtime.png").is_file()
    assert (output / "benchmark_fit_statistics.png").is_file()
    assert (output / "benchmark_evidence.png").is_file()
    assert (output / "benchmark_parameters_clear.png").is_file()
