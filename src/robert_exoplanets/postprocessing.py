"""Sampler-independent fit statistics and plotting for configured ROBERT runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from robert_exoplanets.core import RobertDataError, RobertValidationError, Spectrum
from robert_exoplanets.retrieval import MultiDatasetRetrievalProblem


DEFAULT_DATASET_COLORS = (
    "#20639b",
    "#ef5675",
    "#2ca25f",
    "#ffa600",
    "#7a5195",
    "#00a6a6",
)
RETRIEVAL_RESULT_DIRECTORIES = (
    "ultranest",
    "multinest",
    "optimal_estimation",
    "nested_sampling",
)


def discover_retrieval_result_directories(output_dir: str | Path) -> tuple[Path, ...]:
    """Return completed retrieval phase directories in a stable order."""

    root = Path(output_dir).expanduser()
    return tuple(
        directory
        for name in RETRIEVAL_RESULT_DIRECTORIES
        if (directory := root / name).joinpath("result.json").is_file()
    )


def postprocess_retrieval_output(
    problem: MultiDatasetRetrievalProblem,
    result_dir: str | Path,
    *,
    plot_dir: str | Path,
    parameter_labels: Mapping[str, str] | None = None,
    dataset_colors: Mapping[str, str] | None = None,
    style: str = "default",
    image_format: str = "png",
    dpi: int = 180,
    max_posterior_samples: int = 20_000,
    posterior_predictive_samples: int = 200,
    posterior_predictive_seed: int = 0,
    corner_max_parameters: int = 20,
    native_spectrum_model: object | None = None,
    leave_one_out: bool = False,
    loo_max_posterior_draws: int = 2_000,
    loo_seed: int = 0,
    loo_pareto_k_threshold: float | None = None,
) -> dict[str, Any]:
    """Calculate diagnostics and plot one serialized retrieval phase."""

    _validate_plot_options(style, image_format, dpi)
    if max_posterior_samples < 1:
        raise RobertValidationError("max_posterior_samples must be positive")
    if posterior_predictive_samples < 1:
        raise RobertValidationError("posterior_predictive_samples must be positive")
    result_path = Path(result_dir).expanduser()
    summary = _read_json(result_path / "result.json")
    arrays = _read_npz(result_path / "result_arrays.npz")
    names = tuple(str(name) for name in summary.get("parameter_names", ()))
    if not names:
        raise RobertDataError(f"retrieval result has no parameter names: {result_path}")
    if names != problem.parameter_names:
        raise RobertDataError(
            "retrieval result parameter order does not match the configured problem"
        )
    best = _float_mapping(summary.get("best_fit_parameters"), "best_fit_parameters")
    spectra = problem.model_spectra(best)
    diagnostics = calculate_fit_statistics(
        problem,
        spectra,
        best,
        fitted_parameter_count=len(names),
    )
    diagnostics.update(
        {
            "result_directory": str(result_path.resolve()),
            "method": summary.get("method"),
            "converged": bool(summary.get("converged")),
            "message": summary.get("message"),
            "best_fit_parameters": best,
            "best_fit_log_likelihood_reported": summary.get(
                "best_fit_log_likelihood"
            ),
            "log_evidence": summary.get("log_evidence"),
            "log_evidence_error": summary.get("log_evidence_error"),
            "sampler_metadata": summary.get("metadata", {}),
            "inference_elapsed_seconds": _metadata_float(
                summary, "inference_elapsed_seconds"
            ),
        }
    )
    posterior = posterior_summary(names, arrays)
    diagnostics["posterior"] = posterior
    posterior_draws = _posterior_parameter_draws(
        names,
        arrays,
        maximum=posterior_predictive_samples,
        seed=posterior_predictive_seed,
    )
    bounds = np.asarray(problem.parameters.bounds, dtype=float)
    posterior_draws = np.clip(
        posterior_draws,
        bounds[:, 0],
        bounds[:, 1],
    )
    spectral_quantiles = _posterior_spectral_quantiles(problem, posterior_draws)
    native_quantiles = _posterior_native_spectral_quantiles(
        problem,
        posterior_draws,
        native_spectrum_model,
    )
    temperature_quantiles = _posterior_temperature_quantiles(
        problem,
        posterior_draws,
    )
    diagnostics["posterior_predictive_draws"] = int(posterior_draws.shape[0])
    diagnostics["native_opacity_spectrum"] = native_quantiles is not None
    diagnostics["temperature_profile_regions"] = tuple(temperature_quantiles)

    destination = Path(plot_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    diagnostics["leave_one_out"] = {"enabled": bool(leave_one_out)}
    if leave_one_out:
        samples = arrays.get("samples")
        if samples is None:
            diagnostics["leave_one_out"].update(
                {
                    "status": "not_available",
                    "reason": "PSIS-LOO requires nested-sampling posterior draws",
                }
            )
        else:
            from robert_exoplanets.diagnostics import (
                plot_leave_one_out_result,
                run_psis_leave_one_out,
                write_leave_one_out_result,
            )

            loo = run_psis_leave_one_out(
                problem,
                samples,
                weights=arrays.get("weights"),
                max_posterior_draws=loo_max_posterior_draws,
                seed=loo_seed,
                pareto_k_threshold=loo_pareto_k_threshold,
            )
            write_leave_one_out_result(loo, destination)
            plot_leave_one_out_result(
                loo,
                destination / f"leave_one_out.{image_format}",
                style=style,
                dpi=dpi,
            )
            loo_mapping = loo.to_mapping()
            loo_mapping.pop("points")
            diagnostics["leave_one_out"].update(
                {"status": "complete", **loo_mapping}
            )
    _write_json(destination / "fit_statistics.json", diagnostics)
    _write_json(destination / "posterior_summary.json", posterior)
    _plot_fit(
        problem,
        spectra,
        best,
        diagnostics,
        destination / f"fit_spectrum_residuals.{image_format}",
        dataset_colors=dataset_colors,
        style=style,
        dpi=dpi,
        posterior_quantiles=spectral_quantiles,
        native_quantiles=native_quantiles,
    )
    if temperature_quantiles:
        _plot_temperature_profiles(
            temperature_quantiles,
            destination / f"temperature_profiles.{image_format}",
            style=style,
            dpi=dpi,
        )
    _plot_parameters(
        names,
        arrays,
        destination,
        parameter_labels=parameter_labels,
        style=style,
        image_format=image_format,
        dpi=dpi,
        max_samples=max_posterior_samples,
        seed=posterior_predictive_seed,
        corner_max_parameters=corner_max_parameters,
    )
    _write_plot_manifest(
        destination,
        source=result_path,
        kind="retrieval",
        style=style,
        image_format=image_format,
        dpi=dpi,
    )
    return diagnostics


def postprocess_forward_output(
    problem: MultiDatasetRetrievalProblem,
    forward_path: str | Path,
    *,
    plot_dir: str | Path,
    dataset_colors: Mapping[str, str] | None = None,
    style: str = "default",
    image_format: str = "png",
    dpi: int = 180,
) -> dict[str, Any]:
    """Calculate diagnostics and plot one serialized configured forward model."""

    _validate_plot_options(style, image_format, dpi)
    source = Path(forward_path).expanduser()
    arrays = _read_npz(source)
    parameters = {
        key.removeprefix("parameter_"): float(np.asarray(value))
        for key, value in arrays.items()
        if key.startswith("parameter_")
    }
    spectra = _forward_spectra(problem, arrays)
    diagnostics = calculate_fit_statistics(
        problem,
        spectra,
        parameters,
        fitted_parameter_count=len(parameters),
    )
    diagnostics.update(
        {
            "forward_file": str(source.resolve()),
            "parameters": parameters,
            "note": (
                "Degrees of freedom and information criteria assume the configured "
                "parameters were fitted; for a purely predictive forward model they "
                "are descriptive only."
            ),
        }
    )
    destination = Path(plot_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "fit_statistics.json", diagnostics)
    _write_json(destination / "forward_parameters.json", parameters)
    _plot_fit(
        problem,
        spectra,
        parameters,
        diagnostics,
        destination / f"forward_spectrum_residuals.{image_format}",
        dataset_colors=dataset_colors,
        style=style,
        dpi=dpi,
    )
    _write_plot_manifest(
        destination,
        source=source,
        kind="forward",
        style=style,
        image_format=image_format,
        dpi=dpi,
    )
    return diagnostics


def calculate_fit_statistics(
    problem: MultiDatasetRetrievalProblem,
    spectra: Mapping[str, Spectrum],
    parameters: Mapping[str, float],
    *,
    fitted_parameter_count: int,
) -> dict[str, Any]:
    """Return total and per-dataset Gaussian fit diagnostics."""

    effective = problem.likelihood.effective_inputs_by_dataset(
        spectra, problem.observations, parameters
    )
    per_dataset: dict[str, dict[str, float | int]] = {}
    chi_squared = 0.0
    squared_residual_sum = 0.0
    standardized: list[np.ndarray] = []
    number_points = 0
    for name, (model, data, uncertainty) in effective.items():
        residual = data - model
        standardized_residual = residual / uncertainty
        dataset_chi_squared = float(np.sum(np.square(standardized_residual)))
        count = int(data.size)
        per_dataset[name] = {
            "number_points": count,
            "chi_squared": dataset_chi_squared,
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mean_standardized_residual": float(np.mean(standardized_residual)),
            "rms_standardized_residual": float(
                np.sqrt(np.mean(np.square(standardized_residual)))
            ),
            "maximum_absolute_standardized_residual": float(
                np.max(np.abs(standardized_residual))
            ),
        }
        number_points += count
        chi_squared += dataset_chi_squared
        squared_residual_sum += float(np.sum(np.square(residual)))
        standardized.append(standardized_residual)

    parameter_count = int(fitted_parameter_count)
    degrees_of_freedom = number_points - parameter_count
    log_likelihood = float(
        problem.likelihood.loglike(spectra, problem.observations, parameters)
    )
    aic = 2.0 * parameter_count - 2.0 * log_likelihood
    bic = parameter_count * math.log(number_points) - 2.0 * log_likelihood
    aicc = (
        aic
        + 2.0
        * parameter_count
        * (parameter_count + 1)
        / (number_points - parameter_count - 1)
        if number_points > parameter_count + 1
        else None
    )
    all_standardized = np.concatenate(standardized)
    survival_probability = None
    if degrees_of_freedom > 0:
        from scipy.special import gammaincc

        survival_probability = float(
            gammaincc(0.5 * degrees_of_freedom, 0.5 * chi_squared)
        )
    return {
        "number_points": number_points,
        "number_fitted_parameters": parameter_count,
        "degrees_of_freedom": degrees_of_freedom,
        "chi_squared": chi_squared,
        "reduced_chi_squared": (
            chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else None
        ),
        "chi_squared_survival_probability": survival_probability,
        "rmse": float(np.sqrt(squared_residual_sum / number_points)),
        "mean_standardized_residual": float(np.mean(all_standardized)),
        "rms_standardized_residual": float(
            np.sqrt(np.mean(np.square(all_standardized)))
        ),
        "maximum_absolute_standardized_residual": float(
            np.max(np.abs(all_standardized))
        ),
        "log_likelihood_recomputed": log_likelihood,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "per_dataset": per_dataset,
    }


def posterior_summary(
    parameter_names: Sequence[str],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Summarize nested posterior samples or an OE Gaussian approximation."""

    names = tuple(parameter_names)
    if "samples" in arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(names):
            raise RobertDataError("nested posterior samples do not match parameter names")
        weights = _normalized_weights(arrays.get("weights"), samples.shape[0])
        quantiles = {
            name: weighted_quantile(samples[:, index], weights, (0.16, 0.5, 0.84))
            .astype(float)
            .tolist()
            for index, name in enumerate(names)
        }
        return {
            "kind": "weighted_samples",
            "number_samples": int(samples.shape[0]),
            "effective_sample_size": float(1.0 / np.sum(np.square(weights))),
            "quantiles_16_50_84": quantiles,
        }
    if "state_vector" in arrays and "covariance" in arrays:
        state = np.asarray(arrays["state_vector"], dtype=float)
        covariance = np.asarray(arrays["covariance"], dtype=float)
        if state.shape != (len(names),) or covariance.shape != (len(names), len(names)):
            raise RobertDataError("OE state or covariance does not match parameter names")
        sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        return {
            "kind": "optimal_estimation_gaussian",
            "quantiles_16_50_84": {
                name: [
                    float(state[index] - sigma[index]),
                    float(state[index]),
                    float(state[index] + sigma[index]),
                ]
                for index, name in enumerate(names)
            },
        }
    raise RobertDataError("result arrays contain neither nested samples nor OE state")


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: Sequence[float],
) -> np.ndarray:
    """Return deterministic weighted quantiles for one-dimensional samples."""

    data = np.asarray(values, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    normalized = _normalized_weights(weights, data.size)
    if data.ndim != 1 or not np.all(np.isfinite(data)):
        raise RobertValidationError("weighted quantile values must be finite and 1D")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise RobertValidationError("weighted quantile probabilities must be in [0, 1]")
    order = np.argsort(data)
    sorted_data = data[order]
    cumulative = np.cumsum(normalized[order])
    cumulative[-1] = 1.0
    return np.interp(probability, cumulative, sorted_data)


def _posterior_parameter_draws(
    names: Sequence[str],
    arrays: Mapping[str, np.ndarray],
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if "samples" in arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(names):
            raise RobertDataError("nested posterior samples do not match parameter names")
        weights = _normalized_weights(arrays.get("weights"), samples.shape[0])
        count = min(maximum, max(1, samples.shape[0]))
        indices = rng.choice(samples.shape[0], size=count, replace=True, p=weights)
        return np.asarray(samples[indices], dtype=float)
    if "state_vector" in arrays and "covariance" in arrays:
        state = np.asarray(arrays["state_vector"], dtype=float)
        covariance = np.asarray(arrays["covariance"], dtype=float)
        return np.asarray(
            rng.multivariate_normal(
                state,
                covariance,
                size=maximum,
                check_valid="raise",
            ),
            dtype=float,
        )
    raise RobertDataError("result arrays contain neither nested samples nor OE state")


def _posterior_spectral_quantiles(
    problem: MultiDatasetRetrievalProblem,
    draws: np.ndarray,
) -> dict[str, np.ndarray]:
    predictions: dict[str, list[np.ndarray]] = {
        name: [] for name in problem.observations.names
    }
    for draw in draws:
        spectra = problem.model_spectra(draw)
        for name in predictions:
            predictions[name].append(np.asarray(spectra[name].values, dtype=float))
    return {
        name: np.quantile(np.stack(values), (0.16, 0.5, 0.84), axis=0)
        for name, values in predictions.items()
    }


def _posterior_native_spectral_quantiles(
    problem: MultiDatasetRetrievalProblem,
    draws: np.ndarray,
    model: object | None,
) -> dict[str, np.ndarray | str] | None:
    if model is None:
        return None
    spectra = [model(problem.parameter_mapping(draw)) for draw in draws]
    if not spectra or any(not isinstance(spectrum, Spectrum) for spectrum in spectra):
        raise RobertValidationError("native spectrum model must return Spectrum")
    reference = spectra[0]
    if any(
        spectrum.unit != reference.unit
        or spectrum.observable != reference.observable
        or not np.array_equal(
            spectrum.spectral_grid.values,
            reference.spectral_grid.values,
        )
        for spectrum in spectra[1:]
    ):
        raise RobertValidationError(
            "native posterior spectra must share one grid, unit, and observable"
        )
    return {
        "wavelength": np.asarray(reference.spectral_grid.values, dtype=float),
        "quantiles": np.quantile(
            np.stack([spectrum.values for spectrum in spectra]),
            (0.16, 0.5, 0.84),
            axis=0,
        ),
        "unit": reference.unit,
    }


def _posterior_temperature_quantiles(
    problem: MultiDatasetRetrievalProblem,
    draws: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    builders = _named_atmosphere_builders(problem.forward_model)
    output = {}
    for name, builder in builders.items():
        profiles = []
        for draw in draws:
            parameters = problem.parameter_mapping(draw)
            profiles.append(
                np.asarray(
                    builder.temperature_profile.evaluate(
                        parameters,
                        builder.pressure_grid,
                    ),
                    dtype=float,
                )
            )
        output[name] = {
            "pressure": np.asarray(builder.pressure_grid.centers, dtype=float),
            "quantiles": np.quantile(
                np.stack(profiles),
                (0.16, 0.5, 0.84),
                axis=0,
            ),
        }
    return output


def _named_atmosphere_builders(model: object) -> dict[str, object]:
    if hasattr(model, "hot_model") and hasattr(model, "cold_model"):
        output = {}
        for prefix, regional in (
            ("hot", model.hot_model),
            ("cold", model.cold_model),
        ):
            nested = _named_atmosphere_builders(regional)
            for name, builder in nested.items():
                output[prefix if name == "primary" else f"{prefix}_{name}"] = builder
        return output
    if hasattr(model, "emission_model"):
        return _named_atmosphere_builders(model.emission_model)
    builder = getattr(model, "atmosphere_builder", None)
    if builder is not None:
        return {"primary": builder}
    models = getattr(model, "models", None)
    if isinstance(models, Mapping) and models:
        return _named_atmosphere_builders(next(iter(models.values())))
    return {}


def _plot_temperature_profiles(
    profiles: Mapping[str, Mapping[str, np.ndarray]],
    output: Path,
    *,
    style: str,
    dpi: int,
) -> None:
    plt = _pyplot()
    with plt.style.context(style):
        figure, axis = plt.subplots(figsize=(6.5, 7.0))
        for index, (name, values) in enumerate(profiles.items()):
            pressure = values["pressure"]
            quantiles = values["quantiles"]
            color = DEFAULT_DATASET_COLORS[index % len(DEFAULT_DATASET_COLORS)]
            axis.fill_betweenx(
                pressure,
                quantiles[0],
                quantiles[2],
                color=color,
                alpha=0.22,
            )
            axis.plot(quantiles[1], pressure, color=color, label=f"{name} median")
        axis.set_yscale("log")
        axis.invert_yaxis()
        axis.set_xlabel("Temperature (K)")
        axis.set_ylabel("Pressure (bar)")
        axis.set_title("Posterior temperature-pressure profile (68% interval)")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_fit(
    problem: MultiDatasetRetrievalProblem,
    spectra: Mapping[str, Spectrum],
    parameters: Mapping[str, float],
    statistics: Mapping[str, Any],
    output: Path,
    *,
    dataset_colors: Mapping[str, str] | None,
    style: str,
    dpi: int,
    posterior_quantiles: Mapping[str, np.ndarray] | None = None,
    native_quantiles: Mapping[str, np.ndarray | str] | None = None,
) -> None:
    plt = _pyplot()
    colors = _dataset_color_mapping(problem, dataset_colors)
    effective = problem.likelihood.effective_inputs_by_dataset(
        spectra, problem.observations, parameters
    )
    with plt.style.context(style):
        figure, (fit_axis, residual_axis) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": (3, 1)},
        )
        flux_label = "Model and observation"
        if native_quantiles is not None:
            native_wavelength = np.asarray(native_quantiles["wavelength"], dtype=float)
            native_values = np.asarray(native_quantiles["quantiles"], dtype=float)
            native_scale, flux_label = _plot_scale(str(native_quantiles["unit"]))
            fit_axis.fill_between(
                native_wavelength,
                native_values[0] * native_scale,
                native_values[2] * native_scale,
                color="0.35",
                alpha=0.18,
                linewidth=0.0,
                label="Native-grid 68% envelope",
            )
            fit_axis.plot(
                native_wavelength,
                native_values[1] * native_scale,
                color="0.15",
                linewidth=1.15,
                label="Native-opacity-grid median",
            )
        for dataset in problem.observations.datasets:
            name = dataset.name
            observation = dataset.observation
            valid = (
                np.ones(observation.n_points, dtype=bool)
                if observation.mask is None
                else observation.mask
            )
            model, data, uncertainty = effective[name]
            scale, flux_label = _plot_scale(observation.flux_unit)
            wavelength = observation.wavelength[valid]
            color = colors[name]
            fit_axis.errorbar(
                wavelength,
                data * scale,
                yerr=uncertainty * scale,
                fmt=".",
                color=color,
                alpha=0.75,
                label=observation.instrument or name,
            )
            quantiles = (
                None
                if posterior_quantiles is None
                else posterior_quantiles.get(name)
            )
            if quantiles is not None and quantiles.shape[1] == valid.size:
                quantiles = quantiles[:, valid]
            plotted_model = model
            if quantiles is not None:
                plotted_model = quantiles[1]
                fit_axis.fill_between(
                    wavelength,
                    quantiles[0] * scale,
                    quantiles[2] * scale,
                    color=color,
                    alpha=0.2,
                    linewidth=0.0,
                    label=(
                        "68% posterior envelope"
                        if dataset is problem.observations.datasets[0]
                        else None
                    ),
                )
            fit_axis.plot(
                wavelength,
                plotted_model * scale,
                linestyle="none",
                marker="s",
                markersize=4.0,
                markerfacecolor="none",
                markeredgecolor=color,
                label=(
                    "Observation-grid model"
                    if dataset is problem.observations.datasets[0]
                    else None
                ),
            )
            residual_axis.plot(
                wavelength,
                (data - model) / uncertainty,
                ".",
                color=color,
            )
        reduced = statistics.get("reduced_chi_squared")
        reduced_text = "undefined" if reduced is None else f"{float(reduced):.3f}"
        fit_axis.set_ylabel(flux_label)
        fit_axis.set_title(
            f"{problem.name}: best fit (reduced $\\chi^2$ = {reduced_text})"
        )
        fit_axis.legend(fontsize=8, ncol=max(1, min(3, len(colors))))
        residual_axis.axhline(0.0, color="0.2", linewidth=0.8)
        residual_axis.axhline(1.0, color="0.6", linestyle="--", linewidth=0.7)
        residual_axis.axhline(-1.0, color="0.6", linestyle="--", linewidth=0.7)
        residual_axis.set_ylabel(r"Residual / $\sigma$")
        residual_axis.set_xlabel("Wavelength (micron)")
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_parameters(
    names: Sequence[str],
    arrays: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    parameter_labels: Mapping[str, str] | None,
    style: str,
    image_format: str,
    dpi: int,
    max_samples: int,
    seed: int,
    corner_max_parameters: int,
) -> None:
    plt = _pyplot()
    labels = {name: name for name in names}
    labels.update(parameter_labels or {})
    if "samples" in arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        weights = _normalized_weights(arrays.get("weights"), samples.shape[0])
        selected = _plot_sample_indices(samples.shape[0], max_samples)
        plot_samples = samples[selected]
        plot_weights = weights[selected]
        plot_weights /= plot_weights.sum()
        covariance = _weighted_covariance(samples, weights)
        with plt.style.context(style):
            rows = math.ceil(len(names) / 3)
            figure, axes = plt.subplots(rows, 3, figsize=(12, 3.2 * rows))
            flat_axes = np.atleast_1d(axes).ravel()
            for index, name in enumerate(names):
                flat_axes[index].hist(
                    plot_samples[:, index],
                    bins=40,
                    weights=plot_weights,
                    color="#20639b",
                    alpha=0.75,
                    density=True,
                )
                quantile = weighted_quantile(
                    samples[:, index], weights, (0.16, 0.5, 0.84)
                )
                for value, linestyle in zip(quantile, ("--", "-", "--"), strict=True):
                    flat_axes[index].axvline(value, color="#ef5675", linestyle=linestyle)
                flat_axes[index].set_xlabel(labels[name])
                flat_axes[index].set_yticks([])
            for axis in flat_axes[len(names) :]:
                axis.set_visible(False)
            figure.suptitle("Posterior marginal distributions")
            figure.tight_layout()
            figure.savefig(
                output_dir / f"posterior_marginals.{image_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)
        _plot_correlation(
            covariance,
            names,
            labels,
            output_dir / f"parameter_correlation.{image_format}",
            style=style,
            dpi=dpi,
        )
        if len(names) <= corner_max_parameters:
            _plot_corner(
                _posterior_parameter_draws(
                    names,
                    arrays,
                    maximum=max_samples,
                    seed=seed,
                ),
                names,
                labels,
                output_dir / f"posterior_corner.{image_format}",
                style=style,
                dpi=dpi,
            )
        return
    if "state_vector" in arrays and "covariance" in arrays:
        state = np.asarray(arrays["state_vector"], dtype=float)
        covariance = np.asarray(arrays["covariance"], dtype=float)
        sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        with plt.style.context(style):
            figure, axis = plt.subplots(figsize=(10, max(4, 0.5 * len(names))))
            positions = np.arange(len(names))
            axis.errorbar(state, positions, xerr=sigma, fmt="o", color="#20639b")
            axis.set_yticks(positions, [labels[name] for name in names])
            axis.invert_yaxis()
            axis.set_xlabel(r"Optimal-estimation state (1$\sigma$ uncertainty)")
            axis.set_title("Optimal-estimation parameter constraints")
            figure.tight_layout()
            figure.savefig(
                output_dir / f"optimal_estimation_parameters.{image_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)
        _plot_correlation(
            covariance,
            names,
            labels,
            output_dir / f"parameter_correlation.{image_format}",
            style=style,
            dpi=dpi,
        )


def _plot_corner(
    samples: np.ndarray,
    names: Sequence[str],
    labels: Mapping[str, str],
    output: Path,
    *,
    style: str,
    dpi: int,
) -> None:
    plt = _pyplot()
    count = len(names)
    with plt.style.context(style):
        figure, axes = plt.subplots(
            count,
            count,
            figsize=(max(3.2, 2.15 * count), max(3.2, 2.15 * count)),
            squeeze=False,
        )
        for row in range(count):
            for column in range(count):
                axis = axes[row, column]
                if row < column:
                    axis.set_visible(False)
                    continue
                if row == column:
                    axis.hist(
                        samples[:, column],
                        bins=35,
                        color="#20639b",
                        alpha=0.8,
                        density=True,
                    )
                    axis.set_yticks([])
                else:
                    axis.plot(
                        samples[:, column],
                        samples[:, row],
                        ".",
                        color="#20639b",
                        alpha=min(0.35, max(0.03, 150.0 / samples.shape[0])),
                        markersize=1.4,
                        rasterized=True,
                    )
                if row == count - 1:
                    axis.set_xlabel(labels[names[column]], fontsize=8)
                else:
                    axis.set_xticklabels([])
                if column == 0 and row > 0:
                    axis.set_ylabel(labels[names[row]], fontsize=8)
                elif column > 0:
                    axis.set_yticklabels([])
                axis.tick_params(labelsize=7)
        figure.suptitle("Posterior corner plot", y=1.0)
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_correlation(
    covariance: np.ndarray,
    names: Sequence[str],
    labels: Mapping[str, str],
    output: Path,
    *,
    style: str,
    dpi: int,
) -> None:
    plt = _pyplot()
    diagonal = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(diagonal, diagonal)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=float),
        where=denominator > 0.0,
    )
    with plt.style.context(style):
        size = max(6.0, 0.65 * len(names))
        figure, axis = plt.subplots(figsize=(size, size))
        image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(len(names)), [labels[name] for name in names], rotation=90)
        axis.set_yticks(range(len(names)), [labels[name] for name in names])
        figure.colorbar(image, ax=axis, label="Correlation")
        axis.set_title("Parameter correlation")
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _forward_spectra(
    problem: MultiDatasetRetrievalProblem,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Spectrum]:
    spectra = {}
    for dataset in problem.observations.datasets:
        wavelength_key = f"{dataset.name}_wavelength_micron"
        model_key = f"{dataset.name}_model"
        if wavelength_key not in arrays or model_key not in arrays:
            raise RobertDataError(
                f"forward model is missing arrays for dataset {dataset.name!r}"
            )
        observation = dataset.observation
        spectra[dataset.name] = Spectrum.from_arrays(
            arrays[wavelength_key],
            arrays[model_key],
            unit=observation.flux_unit,
            observable=observation.observable,
            wavelength_unit=observation.wavelength_unit,
        )
    return spectra


def _dataset_color_mapping(
    problem: MultiDatasetRetrievalProblem,
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    selected = dict(overrides or {})
    return {
        name: selected.get(name, DEFAULT_DATASET_COLORS[index % len(DEFAULT_DATASET_COLORS)])
        for index, name in enumerate(problem.observations.names)
    }


def _plot_scale(unit: str) -> tuple[float, str]:
    if unit in {"eclipse_depth", "transit_depth", "relative_flux"}:
        return 1.0e6, f"{unit.replace('_', ' ').title()} (ppm)"
    return 1.0, unit


def _normalized_weights(values: object, count: int) -> np.ndarray:
    if count < 1:
        raise RobertValidationError("at least one posterior sample is required")
    weights = (
        np.ones(count, dtype=float)
        if values is None
        else np.asarray(values, dtype=float)
    )
    if weights.shape != (count,) or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise RobertValidationError("posterior weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise RobertValidationError("posterior weights must have positive sum")
    return weights / total


def _weighted_covariance(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.sum(samples * weights[:, None], axis=0)
    centered = samples - mean
    denominator = 1.0 - float(np.sum(np.square(weights)))
    if denominator <= 0.0:
        return np.zeros((samples.shape[1], samples.shape[1]), dtype=float)
    return (centered * weights[:, None]).T @ centered / denominator


def _plot_sample_indices(count: int, maximum: int) -> np.ndarray:
    if maximum < 1:
        raise RobertValidationError("max_posterior_samples must be positive")
    if count <= maximum:
        return np.arange(count)
    return np.linspace(0, count - 1, maximum, dtype=int)


def _validate_plot_options(style: str, image_format: str, dpi: int) -> None:
    if not style:
        raise RobertValidationError("plot style must not be empty")
    if image_format not in {"png", "pdf", "svg"}:
        raise RobertValidationError("image format must be png, pdf, or svg")
    if isinstance(dpi, bool) or int(dpi) < 1:
        raise RobertValidationError("plot dpi must be positive")


def _metadata_float(summary: Mapping[str, Any], key: str) -> float | None:
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get(key) in (None, ""):
        return None
    return float(metadata[key])


def _float_mapping(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise RobertDataError(f"retrieval result {name} must be a mapping")
    return {str(key): float(item) for key, item in value.items()}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobertDataError(f"failed to read JSON post-processing input: {path}") from exc
    if not isinstance(value, dict):
        raise RobertDataError(f"post-processing JSON input must be a mapping: {path}")
    return value


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except (OSError, ValueError) as exc:
        raise RobertDataError(f"failed to read numerical post-processing input: {path}") from exc


def _write_json(path: Path, value: object) -> None:
    try:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RobertDataError(f"failed to write post-processing output: {path}") from exc


def _write_plot_manifest(
    output_dir: Path,
    *,
    source: Path,
    kind: str,
    style: str,
    image_format: str,
    dpi: int,
) -> None:
    _write_json(
        output_dir / "plot_manifest.json",
        {
            "kind": kind,
            "source": str(source.resolve()),
            "style": style,
            "image_format": image_format,
            "dpi": int(dpi),
        },
    )


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RobertDataError(
            "plotting requires matplotlib; install the ROBERT dev or complete dependencies"
        ) from exc
    return plt


__all__ = [
    "calculate_fit_statistics",
    "discover_retrieval_result_directories",
    "posterior_summary",
    "postprocess_forward_output",
    "postprocess_retrieval_output",
    "weighted_quantile",
]
