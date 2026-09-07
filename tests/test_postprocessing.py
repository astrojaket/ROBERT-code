"""Tests for sampler-independent retrieval and forward post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robert_exoplanets.postprocessing as postprocessing_module

from robert_exoplanets.diagnostics.benchmark_style import (
    ROBERT_COLOR,
    ROBERT_MATPLOTLIB_STYLE,
)

from robert_exoplanets import (
    IsothermalTemperatureProfile,
    Observation,
    PressureGrid,
    Spectrum,
)
from robert_exoplanets.instruments import ObservationCollection, ObservationDataset
from robert_exoplanets.likelihoods import MultiDatasetGaussianLikelihood
from robert_exoplanets.postprocessing import (
    calculate_fit_statistics,
    discover_retrieval_result_directories,
    posterior_summary,
    postprocess_forward_output,
    postprocess_retrieval_output,
    postprocess_saved_best_fit_output,
    weighted_quantile,
)
from robert_exoplanets.retrieval import (
    MultiDatasetRetrievalProblem,
    RetrievalParameter,
    RetrievalParameterSet,
    UniformPrior,
)
from robert_exoplanets.retrieval.predictions import (
    capture_best_fit_prediction,
    write_best_fit_prediction,
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
        class Builder:
            temperature_profile = IsothermalTemperatureProfile(
                parameter_name="level"
            )
            pressure_grid = PressureGrid.logspace(1.0e-5, 10.0, 5, unit="bar")

            def build(self, parameters):
                level = float(parameters["level"])
                return SimpleNamespace(
                    pressure_grid=self.pressure_grid,
                    temperature=np.full(self.pressure_grid.n_layers, level),
                    composition={
                        "H2": np.full(self.pressure_grid.n_layers, 0.85),
                        "He": np.full(self.pressure_grid.n_layers, 0.15),
                    },
                    temperature_unit="K",
                    composition_convention="volume_mixing_ratio",
                )

        atmosphere_builder = Builder()

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
    dominant = weighted_quantile(
        np.array([0.0, 1.0, 10.0]),
        np.array([0.01, 0.01, 0.98]),
        (0.5,),
    )
    assert dominant[0] == pytest.approx(5.408163265306122)


def test_retrieval_postprocessing_writes_statistics_and_plots(tmp_path: Path) -> None:
    problem = _problem()
    result_dir = tmp_path / "outputs" / "multinest"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "parameter_names": ["level"],
                "best_fit_parameters": {"level": 1.0},
                "best_fit_log_likelihood": 0.0,
                "method": "multinest",
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

    plot_dir = tmp_path / "plots" / "multinest"
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
        assert predictive["posterior_draw_vectors"].shape == (100, 1)
        assert predictive["quantile_probabilities"].shape == (5,)
        assert predictive["quantile_labels"].tolist() == [
            "lower_2sigma",
            "lower_1sigma",
            "median",
            "upper_1sigma",
            "upper_2sigma",
        ]
        assert predictive["spectrum_synthetic_quantiles"].shape == (5, 3)
        assert predictive["spectrum_synthetic_lower_2sigma"].shape == (3,)
        assert predictive["native_spectrum_quantiles"].shape == (5, 9)
        assert predictive["temperature_primary_quantiles"].shape == (5, 5)
        assert predictive["vmr_primary_H2_quantiles"].shape == (5, 5)
        assert not any("q16" in name or "q84" in name for name in predictive.files)
    assert diagnostics["posterior_predictive_draws"] == 100
    assert diagnostics["native_opacity_spectrum"] is True
    manifest = json.loads((plot_dir / "plot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["style"] == "default"
    assert manifest["resolved_style"] == "robert-science"
    assert discover_retrieval_result_directories(tmp_path / "outputs") == (result_dir,)


def test_saved_best_fit_postprocessing_does_not_need_forward_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    result_dir = tmp_path / "outputs" / "multinest"
    result_dir.mkdir(parents=True)
    artifact = capture_best_fit_prediction(
        problem,
        {"level": 1.0},
        prediction=problem.model_spectra({"level": 1.0}),
    )
    write_best_fit_prediction(artifact, result_dir)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "parameter_names": ["level"],
                "best_fit_parameters": {"level": 1.0},
                "best_fit_log_likelihood": 0.0,
                "method": "multinest",
                "converged": True,
                "message": "finished",
                "best_fit_prediction": artifact.reference_mapping(),
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
    monkeypatch.setattr(
        postprocessing_module,
        "_plot_parameters",
        lambda *args, **kwargs: None,
    )

    diagnostics = postprocess_saved_best_fit_output(
        result_dir,
        plot_dir=tmp_path / "plots" / "saved",
    )

    assert diagnostics["statistic_source"] == "saved_best_fit_prediction"
    assert diagnostics["chi_squared"] == 0.0
    assert diagnostics["posterior_predictive"]["status"] == "not_available"
    assert (tmp_path / "plots" / "saved" / "fit_spectrum_residuals.png").is_file()


def test_saved_fit_with_problem_keeps_posterior_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    result_dir = tmp_path / "outputs" / "multinest"
    result_dir.mkdir(parents=True)
    artifact = capture_best_fit_prediction(
        problem,
        {"level": 1.0},
        prediction=problem.model_spectra({"level": 1.0}),
    )
    write_best_fit_prediction(artifact, result_dir)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "parameter_names": ["level"],
                "best_fit_parameters": {"level": 1.0},
                "best_fit_log_likelihood": 0.0,
                "method": "multinest",
                "converged": True,
                "message": "finished",
                "best_fit_prediction": artifact.reference_mapping(),
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        result_dir / "result_arrays.npz",
        samples=np.array([[0.9], [1.0], [1.1]]),
        weights=np.array([0.2, 0.6, 0.2]),
    )
    captured: dict[str, object] = {}

    def capture_plot(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(postprocessing_module, "_plot_fit", capture_plot)
    monkeypatch.setattr(
        postprocessing_module,
        "_plot_parameters",
        lambda *args, **kwargs: None,
    )

    diagnostics = postprocess_retrieval_output(
        problem,
        result_dir,
        plot_dir=tmp_path / "plots" / "saved-with-problem",
    )

    assert diagnostics["posterior_predictive"]["status"] == "complete"
    bands = captured["posterior_quantiles"]
    assert isinstance(bands, dict)
    assert bands["synthetic"].shape == (5, 3)
    assert captured["native_quantiles"] is None


def test_posterior_draw_selection_is_deterministic_and_exact() -> None:
    arrays = {
        "samples": np.array([[0.2], [0.8]]),
        "weights": np.array([0.9, 0.1]),
    }
    bounds = np.array([[0.0, 1.0]])
    first = postprocessing_module._posterior_parameter_draws(
        ("x",), arrays, maximum=100, seed=11, bounds=bounds
    )
    second = postprocessing_module._posterior_parameter_draws(
        ("x",), arrays, maximum=100, seed=11, bounds=bounds
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (100, 1)
    assert set(first[:, 0]) <= {0.2, 0.8}
    assert np.count_nonzero(first[:, 0] == 0.2) > np.count_nonzero(first[:, 0] == 0.8)


def test_optimal_estimation_draws_reject_prior_outside_candidates() -> None:
    draws = postprocessing_module._posterior_parameter_draws(
        ("x",),
        {"state_vector": np.array([0.5]), "covariance": np.array([[0.25]])},
        maximum=100,
        seed=3,
        bounds=np.array([[0.0, 1.0]]),
    )
    assert draws.shape == (100, 1)
    assert np.all((draws >= 0.0) & (draws <= 1.0))


def test_saved_fit_replays_gaussian_nuisance_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = Observation.from_arrays(
        [1.0, 2.0, 3.0],
        [1.0, 1.5, 2.0],
        [0.1, 0.2, 0.3],
        mask=[True, False, True],
    )
    observations = ObservationCollection(
        (
            ObservationDataset(
                "calibrated",
                observation,
                offset_parameter="offset",
                jitter_parameter="jitter",
                uncertainty_scale_parameter="scale",
                uncertainty_scale=1.5,
            ),
        )
    )
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("level", UniformPrior(0.0, 2.0)),
            RetrievalParameter("offset", UniformPrior(-1.0, 1.0)),
            RetrievalParameter("jitter", UniformPrior(0.0, 1.0)),
            RetrievalParameter("scale", UniformPrior(0.5, 2.0)),
        )
    )
    likelihood = MultiDatasetGaussianLikelihood(include_normalization=False)
    problem = MultiDatasetRetrievalProblem(
        name="nuisance-replay",
        observations=observations,
        parameters=parameters,
        forward_model=lambda values: {
            "calibrated": Spectrum.from_arrays(
                observation.wavelength,
                [0.8, 1.7, 2.4],
                unit=observation.flux_unit,
                observable=observation.observable,
            )
        },
        likelihood=likelihood,
    )
    best = {"level": 1.0, "offset": 0.1, "jitter": 0.05, "scale": 1.2}
    prediction = problem.model_spectra(best)
    expected = calculate_fit_statistics(
        problem,
        prediction,
        best,
        fitted_parameter_count=len(best),
    )
    artifact = capture_best_fit_prediction(problem, best, prediction=prediction)
    result_dir = tmp_path / "outputs" / "multinest"
    result_dir.mkdir(parents=True)
    write_best_fit_prediction(artifact, result_dir)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "parameter_names": list(problem.parameter_names),
                "best_fit_parameters": best,
                "best_fit_log_likelihood": expected["log_likelihood_recomputed"],
                "method": "multinest",
                "converged": True,
                "message": "finished",
                "best_fit_prediction": artifact.reference_mapping(),
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        result_dir / "result_arrays.npz",
        samples=np.array([list(best.values()), list(best.values())]),
        weights=np.array([0.5, 0.5]),
        log_likelihood=np.array([-1.0, -1.0]),
    )
    monkeypatch.setattr(
        postprocessing_module,
        "_plot_parameters",
        lambda *args, **kwargs: None,
    )

    diagnostics = postprocess_saved_best_fit_output(
        result_dir,
        plot_dir=tmp_path / "plots" / "nuisance",
    )

    assert diagnostics["statistic_source"] == "saved_best_fit_prediction"
    assert diagnostics["chi_squared"] == pytest.approx(expected["chi_squared"])
    assert diagnostics["log_likelihood_recomputed"] == pytest.approx(
        expected["log_likelihood_recomputed"]
    )


def test_default_plot_style_is_robert_science_style() -> None:
    assert ROBERT_COLOR == "mediumpurple"
    assert ROBERT_MATPLOTLIB_STYLE["axes.spines.top"] is False
    assert ROBERT_MATPLOTLIB_STYLE["axes.spines.right"] is False
    assert ROBERT_MATPLOTLIB_STYLE["legend.frameon"] is False


def test_posterior_fit_plot_draws_median_and_bands_in_robert_colour(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    problem = _problem()
    parameters = {"level": 1.0}
    spectra = problem.model_spectra(parameters)
    statistics = calculate_fit_statistics(
        problem,
        spectra,
        parameters,
        fitted_parameter_count=1,
    )
    base = np.array([1.0, 2.0, 3.0])
    quantiles = np.vstack(
        (0.7 * base, 0.9 * base, base, 1.1 * base, 1.3 * base)
    )
    captured: dict[str, object] = {}
    original_subplots = plt.subplots

    def recording_subplots(*args: object, **kwargs: object):
        figure, axes = original_subplots(*args, **kwargs)
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(postprocessing_module, "_pyplot", lambda: plt)
    monkeypatch.setattr(plt, "subplots", recording_subplots)
    postprocessing_module._plot_fit(
        problem,
        spectra,
        parameters,
        statistics,
        tmp_path / "fit.png",
        dataset_colors=None,
        style="default",
        dpi=72,
        posterior_quantiles={"synthetic": quantiles},
    )

    fit_axis, _ = captured["axes"]
    median_lines = [
        line
        for line in fit_axis.lines
        if line.get_color() == ROBERT_COLOR
        and np.array_equal(line.get_ydata(), base * 1.0e6)
    ]
    assert median_lines
    alphas = sorted(
        float(collection.get_alpha())
        for collection in fit_axis.collections
        if float(collection.get_alpha()) in {0.12, 0.28}
    )
    assert alphas == pytest.approx([0.12, 0.28])


def test_profile_panels_keep_units_and_high_pressure_at_bottom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    pressure = np.array([1.0e-5, 1.0e-2, 1.0])
    quantiles = np.vstack(
        (
            np.full(3, 1.0e-4),
            np.full(3, 1.1e-4),
            np.full(3, 1.2e-4),
            np.full(3, 1.3e-4),
            np.full(3, 1.4e-4),
        )
    )
    profiles = {
        "hot": {
            "H2": {
                "pressure": pressure,
                "pressure_unit": "bar",
                "quantiles": quantiles,
                "value_unit": "volume_mixing_ratio",
            }
        },
        "cold": {
            "H2": {
                "pressure": pressure[::-1],
                "pressure_unit": "mbar",
                "quantiles": quantiles,
                "value_unit": "volume_mixing_ratio",
            }
        },
    }
    captured: dict[str, object] = {}
    original_subplots = plt.subplots

    def recording_subplots(*args: object, **kwargs: object):
        figure, axes = original_subplots(*args, **kwargs)
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(postprocessing_module, "_pyplot", lambda: plt)
    monkeypatch.setattr(plt, "subplots", recording_subplots)
    postprocessing_module._plot_atmospheric_profiles(
        profiles,
        tmp_path / "vmr.png",
        title="VMR",
        x_label="Volume mixing ratio",
        style="default",
        dpi=72,
    )

    axes = np.asarray(captured["axes"], dtype=object).ravel()
    assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[0].get_ylim()[0] > axes[0].get_ylim()[1]
    assert axes[1].get_ylim()[0] > axes[1].get_ylim()[1]
    assert "bar" in axes[0].get_ylabel()
    assert "mbar" in axes[1].get_ylabel()
    assert axes[0].get_xlabel() == "Volume mixing ratio"
    assert axes[0].get_xscale() == "linear"


def test_corner_plot_uses_weights_and_wasp_style(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import corner
    import matplotlib.pyplot as plt

    captured: dict[str, object] = {}

    def fake_corner(samples: np.ndarray, **kwargs: object):
        captured["samples"] = samples
        captured.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(corner, "corner", fake_corner)
    weights = np.array([0.2, 0.3, 0.5])
    postprocessing_module._plot_corner(
        np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        weights,
        ("a", "b"),
        {"a": "A", "b": "B"},
        tmp_path / "corner.png",
        style="default",
        dpi=72,
    )

    np.testing.assert_allclose(captured["weights"], weights)
    assert captured["quantiles"] == [0.16, 0.5, 0.84]
    assert captured["color"] == "mediumpurple"
    assert captured["plot_datapoints"] is False
    assert captured["fill_contours"] is True


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
