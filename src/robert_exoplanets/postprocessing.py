"""Sampler-independent fit statistics and plotting for configured ROBERT runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from robert_exoplanets.core import (
    RobertDataError,
    RobertValidationError,
    SpectralGrid,
    Spectrum,
)
from robert_exoplanets.diagnostics.benchmark_style import (
    PURPLE_PALETTE,
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
    ROBERT_MATPLOTLIB_STYLE,
)
from robert_exoplanets.diagnostics.posterior_models import (
    POSTERIOR_QUANTILE_LABELS,
    POSTERIOR_QUANTILE_PROBABILITIES,
    PosteriorModelCollection,
    PosteriorProductUnavailableError,
    collect_posterior_models,
)
from robert_exoplanets.instruments import ObservationCollection, ObservationDataset
from robert_exoplanets.opacity.metadata import spectral_grid_values_in_unit
from robert_exoplanets.retrieval import MultiDatasetRetrievalProblem
from robert_exoplanets.retrieval.predictions import (
    BestFitPredictionArtifact,
    load_best_fit_prediction,
)


DEFAULT_DATASET_COLORS = PURPLE_PALETTE
RETRIEVAL_RESULT_DIRECTORIES = (
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
    problem: MultiDatasetRetrievalProblem | object | None,
    result_dir: str | Path,
    *,
    plot_dir: str | Path,
    parameter_labels: Mapping[str, str] | None = None,
    dataset_colors: Mapping[str, str] | None = None,
    style: str = "default",
    image_format: str = "png",
    dpi: int = 180,
    max_posterior_samples: int = 20_000,
    posterior_predictive_samples: int = 100,
    posterior_predictive_seed: int = 0,
    corner_max_parameters: int = 20,
    native_spectrum_model: object | None = None,
    leave_one_out: bool = False,
    loo_max_posterior_draws: int = 2_000,
    loo_seed: int = 0,
    loo_pareto_k_threshold: float | None = None,
) -> dict[str, Any]:
    """Calculate diagnostics and plot one serialized retrieval phase.

    A saved best-fit prediction is used when advertised by the current result
    summary, so a plotting call can run without the original forward model.
    Posterior products use one shared weighted-resampled draw set for spectra,
    temperature, composition, and supported cloud profiles.
    """

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
    if problem is not None and names != problem.parameter_names:
        raise RobertDataError(
            "retrieval result parameter order does not match the configured problem"
        )
    best = _float_mapping(summary.get("best_fit_parameters"), "best_fit_parameters")
    saved_artifact = _load_saved_best_fit_prediction(result_path, summary)
    if (
        saved_artifact is not None
        and saved_artifact.parameter_names
        and tuple(saved_artifact.parameter_names) != names
    ):
        raise RobertDataError(
            "saved best-fit prediction parameter order does not match the result"
        )
    spectra: Mapping[str, Spectrum] = {}
    saved_fit = saved_artifact is not None
    if saved_artifact is not None and saved_artifact.status in {"available", "partial"}:
        spectra = saved_artifact.spectra
        if (
            problem is not None
            and saved_artifact.status == "available"
            and _supports_configured_fit_statistics(problem)
        ):
            _validate_saved_dataset_names(problem, saved_artifact)
            fit_problem = problem
        else:
            fit_problem = _saved_problem(saved_artifact)
        if fit_problem is not None and _supports_fit_statistics(fit_problem):
            diagnostics = calculate_fit_statistics(
                fit_problem,
                spectra,
                best,
                fitted_parameter_count=len(names),
            )
            diagnostics["statistic_source"] = "saved_best_fit_prediction"
        else:
            diagnostics = _unavailable_fit_statistics(
                len(names),
                saved_artifact.reason
                or "saved likelihood does not expose a portable fit-statistics contract",
            )
    elif problem is not None and not saved_fit:
        model_output = problem.model_spectra(best)
        spectra = (
            model_output
            if isinstance(model_output, Mapping)
            else {"primary": model_output}
        )
        if _supports_configured_fit_statistics(problem):
            diagnostics = calculate_fit_statistics(
                problem,
                spectra,
                best,
                fitted_parameter_count=len(names),
            )
            diagnostics["statistic_source"] = "configured_forward_model"
        else:
            diagnostics = calculate_fit_statistics(
                _saved_single_problem(problem, spectra["primary"]),
                spectra,
                best,
                fitted_parameter_count=len(names),
            )
            diagnostics["statistic_source"] = "configured_forward_model"
    else:
        diagnostics = _unavailable_fit_statistics(
            len(names),
            saved_artifact.reason
            if saved_artifact is not None
            else "best-fit prediction artifact was not requested or is missing",
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
            "best_fit_prediction": (
                {"status": "not_available", "reason": "not present"}
                if saved_artifact is None
                else saved_artifact.reference_mapping()
            ),
        }
    )
    posterior = posterior_summary(names, arrays)
    diagnostics["posterior"] = posterior
    posterior_draws = np.empty((0, len(names)), dtype=float)
    posterior_models: PosteriorModelCollection | None = None
    spectral_quantiles: dict[str, np.ndarray] | None = None
    native_quantiles: dict[str, np.ndarray | str] | None = None
    temperature_quantiles: dict[str, dict[str, np.ndarray | str]] = {}
    vmr_quantiles: dict[str, dict[str, dict[str, np.ndarray | str]]] = {}
    cloud_quantiles: dict[str, dict[str, dict[str, np.ndarray | str]]] = {}
    if problem is not None:
        bounds = _problem_parameter_bounds(problem, len(names))
        posterior_draws = _posterior_parameter_draws(
            names,
            arrays,
            maximum=posterior_predictive_samples,
            seed=posterior_predictive_seed,
            bounds=bounds,
        )
        try:
            posterior_models = collect_posterior_models(problem, posterior_draws)
        except PosteriorProductUnavailableError as error:
            diagnostics["posterior_predictive"] = {
                "status": "not_available",
                "reason": str(error),
            }
        else:
            spectral_quantiles = {
                name: np.asarray(summary.values, dtype=float)
                for name, summary in posterior_models.spectra.items()
            }
            temperature_quantiles = _profile_quantile_mapping(
                posterior_models.temperature_profiles
            )
            vmr_quantiles = _nested_profile_quantile_mapping(
                posterior_models.vmr_profiles
            )
            cloud_quantiles = _nested_profile_quantile_mapping(
                posterior_models.cloud_profiles
            )
            native_quantiles = _posterior_native_spectral_quantiles(
                problem,
                posterior_draws,
                native_spectrum_model,
            )
            draw_source = (
                "weighted_resampled_nested_posterior"
                if "samples" in arrays
                else "bounded_gaussian_approximation"
            )
            diagnostics["posterior_predictive"] = {
                "status": "complete",
                "reason": None,
                "draw_source": draw_source,
                "quantile_labels": list(POSTERIOR_QUANTILE_LABELS),
                "quantile_probabilities": list(POSTERIOR_QUANTILE_PROBABILITIES),
                "optimal_estimation_approximation": "bounded Gaussian rejection"
                if "samples" not in arrays
                else None,
            }
    else:
        diagnostics["posterior_predictive"] = {
            "status": "not_available",
            "reason": (
                "posterior predictive products require a configured forward model"
            ),
        }
    diagnostics["posterior_predictive_draws"] = int(posterior_draws.shape[0])
    diagnostics["native_opacity_spectrum"] = native_quantiles is not None
    diagnostics["temperature_profile_regions"] = tuple(temperature_quantiles)
    diagnostics["vmr_profile_regions"] = tuple(vmr_quantiles)
    diagnostics["cloud_profile_regions"] = tuple(cloud_quantiles)
    if posterior_models is not None and posterior_models.unsupported_cloud_profiles:
        diagnostics["unsupported_cloud_profiles"] = {
            str(region): {str(name): str(reason) for name, reason in reasons.items()}
            for region, reasons in posterior_models.unsupported_cloud_profiles.items()
        }

    destination = Path(plot_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    posterior_products = destination / "posterior_predictive_quantiles.npz"
    if problem is not None and spectral_quantiles is not None:
        _write_posterior_predictive_quantiles(
            posterior_products,
            native_quantiles=native_quantiles,
            temperature_quantiles=temperature_quantiles,
            vmr_quantiles=vmr_quantiles,
            cloud_quantiles=cloud_quantiles,
            posterior_models=posterior_models,
            draw_vectors=(
                posterior_models.draw_vectors
                if posterior_models is not None
                else posterior_draws
            ),
            parameter_names=names,
        )
        diagnostics["posterior_predictive_file"] = str(posterior_products.resolve())
    diagnostics["leave_one_out"] = {"enabled": bool(leave_one_out)}
    if leave_one_out:
        samples = arrays.get("samples")
        if samples is None or problem is None:
            diagnostics["leave_one_out"].update(
                {
                    "status": "not_available",
                    "reason": (
                        "PSIS-LOO requires nested-sampling posterior draws and a "
                        "configured retrieval problem"
                    ),
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
    fit_plot = destination / f"fit_spectrum_residuals.{image_format}"
    if problem is not None and spectra and not saved_fit:
        if _supports_configured_fit_statistics(problem):
            _plot_fit(
                problem,
                spectra,
                best,
                diagnostics,
                fit_plot,
                dataset_colors=dataset_colors,
                style=style,
                dpi=dpi,
                posterior_quantiles=spectral_quantiles,
                native_quantiles=native_quantiles,
            )
        else:
            _plot_fit(
                _saved_single_problem(problem, spectra["primary"]),
                spectra,
                best,
                diagnostics,
                fit_plot,
                dataset_colors=dataset_colors,
                style=style,
                dpi=dpi,
            )
    elif saved_artifact is not None and spectra:
        plot_problem = (
            problem
            if (
                problem is not None
                and posterior_models is not None
                and _supports_configured_fit_statistics(problem)
            )
            else _saved_problem(saved_artifact)
        )
        _plot_fit(
            plot_problem,
            spectra,
            best,
            diagnostics,
            fit_plot,
            dataset_colors=dataset_colors,
            style=style,
            dpi=dpi,
            posterior_quantiles=(
                spectral_quantiles
                if plot_problem is problem
                else None
            ),
            native_quantiles=(
                native_quantiles
                if plot_problem is problem
                else None
            ),
        )
    if temperature_quantiles:
        _plot_temperature_profiles(
            temperature_quantiles,
            destination / f"temperature_profiles.{image_format}",
            style=style,
            dpi=dpi,
        )
    if vmr_quantiles:
        _plot_atmospheric_profiles(
            vmr_quantiles,
            destination / f"vmr_profiles.{image_format}",
            title="Posterior volume-mixing-ratio profiles",
            x_label="Volume mixing ratio",
            style=style,
            dpi=dpi,
        )
    if cloud_quantiles:
        _plot_atmospheric_profiles(
            cloud_quantiles,
            destination / f"cloud_profiles.{image_format}",
            title="Posterior cloud profiles",
            x_label="Cloud profile value",
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


def postprocess_saved_best_fit_output(
    result_dir: str | Path,
    *,
    plot_dir: str | Path,
    parameter_labels: Mapping[str, str] | None = None,
    dataset_colors: Mapping[str, str] | None = None,
    style: str = "default",
    image_format: str = "png",
    dpi: int = 180,
    max_posterior_samples: int = 20_000,
    corner_max_parameters: int = 20,
) -> dict[str, Any]:
    """Post-process saved best-fit arrays without a model or external data."""

    return postprocess_retrieval_output(
        None,
        result_dir,
        plot_dir=plot_dir,
        parameter_labels=parameter_labels,
        dataset_colors=dataset_colors,
        style=style,
        image_format=image_format,
        dpi=dpi,
        max_posterior_samples=max_posterior_samples,
        corner_max_parameters=corner_max_parameters,
    )


def _load_saved_best_fit_prediction(
    result_path: Path,
    summary: Mapping[str, Any],
) -> BestFitPredictionArtifact | None:
    """Load the current result's advertised prediction artifact."""

    reference = summary.get("best_fit_prediction")
    if isinstance(reference, Mapping):
        status = str(reference.get("status", ""))
        if status in {"not_requested", "not_available"}:
            return None
        relative = reference.get("json")
        if not isinstance(relative, str) or not relative:
            raise RobertDataError(
                "result best_fit_prediction reference must contain a JSON path"
            )
        return load_best_fit_prediction(result_path / relative)
    return None


def _supports_configured_fit_statistics(problem: object) -> bool:
    """Return whether the configured multi-dataset Gaussian path is available."""

    return isinstance(problem, MultiDatasetRetrievalProblem) and callable(
        getattr(getattr(problem, "likelihood", None), "effective_inputs_by_dataset", None)
    )


def _supports_fit_statistics(problem: object) -> bool:
    """Return whether a configured or saved objective can replay statistics."""

    likelihood = getattr(problem, "likelihood", None)
    advertised = getattr(likelihood, "supports_fit_statistics", None)
    if advertised is not None:
        return bool(advertised)
    return callable(getattr(likelihood, "effective_inputs_by_dataset", None)) and callable(
        getattr(likelihood, "loglike", None)
    )


def _unavailable_fit_statistics(parameter_count: int, reason: str) -> dict[str, Any]:
    """Return an explicit status when a saved objective cannot be replayed."""

    return {
        "number_points": 0,
        "number_fitted_parameters": int(parameter_count),
        "degrees_of_freedom": None,
        "chi_squared": None,
        "reduced_chi_squared": None,
        "chi_squared_survival_probability": None,
        "rmse": None,
        "mean_standardized_residual": None,
        "rms_standardized_residual": None,
        "maximum_absolute_standardized_residual": None,
        "log_likelihood_recomputed": None,
        "aic": None,
        "aicc": None,
        "bic": None,
        "per_dataset": {},
        "statistic_source": "unavailable",
        "reason": reason,
    }


def _validate_saved_dataset_names(
    problem: object,
    artifact: BestFitPredictionArtifact,
) -> None:
    """Check that a complete saved artifact matches a multi-dataset problem."""

    collection = getattr(problem, "observations", None)
    datasets = getattr(collection, "datasets", None)
    if datasets is None:
        return
    expected = tuple(str(dataset.name) for dataset in datasets)
    if tuple(artifact.datasets) != expected:
        raise RobertDataError(
            "saved best-fit prediction dataset names do not match the configured problem"
        )


class _SavedLikelihood:
    """Adapter for exact saved Gaussian settings or raw plotting arrays."""

    def __init__(
        self,
        specification: Mapping[str, Any] | None = None,
        actual: object | None = None,
    ) -> None:
        from robert_exoplanets.likelihoods import (
            GaussianLikelihood,
            MultiDatasetGaussianLikelihood,
        )

        if actual is not None:
            specification = {
                "kind": "gaussian"
                if isinstance(actual, GaussianLikelihood)
                else "multi_dataset_gaussian"
                if isinstance(actual, MultiDatasetGaussianLikelihood)
                else "unsupported"
            }
        specification = {} if specification is None else dict(specification)
        kind = str(specification.get("kind", "unsupported"))
        self._kind = kind
        self._likelihood = None
        self.supports_fit_statistics = kind in {
            "gaussian",
            "multi_dataset_gaussian",
        }
        if actual is not None and self.supports_fit_statistics:
            self._likelihood = actual
            return
        if not self.supports_fit_statistics:
            return
        common = {
            "include_normalization": bool(
                specification.get("include_normalization", False)
            ),
            "invalid_model_loglike": float(
                specification.get("invalid_model_loglike", float("-inf"))
            ),
            "coordinate_rtol": float(specification.get("coordinate_rtol", 1.0e-12)),
            "coordinate_atol": float(specification.get("coordinate_atol", 0.0)),
        }
        if kind == "gaussian":
            self._likelihood = GaussianLikelihood(
                **common,
                offset_parameter=specification.get("offset_parameter", "offset"),
                jitter_parameter=specification.get("jitter_parameter", "jitter"),
                uncertainty_scale_parameter=specification.get(
                    "uncertainty_scale_parameter"
                ),
                uncertainty_scale=float(specification.get("uncertainty_scale", 1.0)),
            )
        else:
            self._likelihood = MultiDatasetGaussianLikelihood(**common)

    def effective_inputs_by_dataset(
        self,
        spectra: Mapping[str, Spectrum],
        observations: ObservationCollection,
        parameters: Mapping[str, float],
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if self.supports_fit_statistics:
            if self._likelihood is None:
                raise RobertDataError("saved Gaussian likelihood is not available")
            if self._kind == "gaussian":
                if len(observations.datasets) != 1:
                    raise RobertDataError(
                        "saved single Gaussian likelihood requires one observation dataset"
                    )
                dataset = observations.datasets[0]
                if dataset.name not in spectra:
                    return {}
                return {
                    dataset.name: self._likelihood.effective_inputs(
                        spectra[dataset.name], dataset.observation, parameters
                    )
                }
            return self._likelihood.effective_inputs_by_dataset(
                spectra, observations, parameters
            )
        for dataset in observations.datasets:
            if dataset.name not in spectra:
                continue
            observation = dataset.observation
            valid = (
                np.ones(observation.n_points, dtype=bool)
                if observation.mask is None
                else np.asarray(observation.mask, dtype=bool)
            )
            output[dataset.name] = (
                np.asarray(spectra[dataset.name].values, dtype=float)[valid],
                np.asarray(observation.flux, dtype=float)[valid],
                np.asarray(observation.uncertainty, dtype=float)[valid],
            )
        return output

    def loglike(
        self,
        spectra: Mapping[str, Spectrum],
        observations: ObservationCollection,
        parameters: Mapping[str, float],
    ) -> float:
        if not self.supports_fit_statistics or self._likelihood is None:
            raise RobertDataError(
                "fit statistics are unavailable for the saved likelihood type"
            )
        if self._kind == "gaussian":
            if len(observations.datasets) != 1:
                raise RobertDataError(
                    "saved single Gaussian likelihood requires one observation dataset"
                )
            dataset = observations.datasets[0]
            return float(
                self._likelihood.loglike(
                    spectra[dataset.name], dataset.observation, parameters
                )
            )
        return float(self._likelihood.loglike(spectra, observations, parameters))


class _SavedProblem:
    """Adapter that lets existing fit statistics and plotting consume saved data."""

    def __init__(
        self,
        artifact: BestFitPredictionArtifact,
        pairs: Mapping[str, tuple[Spectrum, object]] | None = None,
        actual_likelihood: object | None = None,
    ) -> None:
        if pairs is None:
            pairs = {
                name: (dataset.spectrum, dataset.observation)
                for name, dataset in artifact.datasets.items()
                if dataset.status == "available" and dataset.spectrum is not None and dataset.observation is not None
            }
        likelihood_spec = artifact.provenance.get("likelihood", {})
        dataset_specs = (
            likelihood_spec.get("datasets", {})
            if isinstance(likelihood_spec, Mapping)
            else {}
        )
        observation_datasets = []
        for name, (_, observation) in pairs.items():
            settings = dataset_specs.get(name, {}) if isinstance(dataset_specs, Mapping) else {}
            settings = settings if isinstance(settings, Mapping) else {}
            observation_datasets.append(
                ObservationDataset(
                    name,
                    observation,
                    offset_parameter=settings.get("offset_parameter"),
                    jitter_parameter=settings.get("jitter_parameter"),
                    uncertainty_scale_parameter=settings.get(
                        "uncertainty_scale_parameter"
                    ),
                    uncertainty_scale=float(settings.get("uncertainty_scale", 1.0)),
                )
            )
        self.name = artifact.problem_name
        self.observations = ObservationCollection(
            tuple(observation_datasets)
        )
        self.likelihood = _SavedLikelihood(likelihood_spec, actual_likelihood)


def _saved_problem(artifact: BestFitPredictionArtifact) -> _SavedProblem:
    return _SavedProblem(artifact)


def _saved_single_problem(problem: object, spectrum: Spectrum) -> _SavedProblem:
    from robert_exoplanets.instruments import Observation

    observation = getattr(problem, "observation", None)
    if not isinstance(observation, Observation):
        raise RobertDataError("single retrieval problem does not expose an Observation")
    return _SavedProblem(
        BestFitPredictionArtifact(
            datasets={},
            problem_name=str(getattr(problem, "name", "retrieval")),
            status="unavailable",
        ),
        {"primary": (spectrum, observation)},
        actual_likelihood=getattr(problem, "likelihood", None),
    )


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


def _problem_parameter_bounds(problem: object, count: int) -> np.ndarray:
    """Return finite lower and upper bounds for posterior draw validation."""

    parameters = getattr(problem, "parameters", None)
    raw_bounds = getattr(parameters, "bounds", None)
    if raw_bounds is None:
        raise RobertDataError("configured problem does not expose parameter bounds")
    bounds = np.asarray(raw_bounds, dtype=float)
    if bounds.shape != (count, 2) or not np.all(np.isfinite(bounds)):
        raise RobertDataError("configured problem parameter bounds are invalid")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise RobertDataError("configured problem parameter bounds are reversed")
    return bounds


def _posterior_parameter_draws(
    names: Sequence[str],
    arrays: Mapping[str, np.ndarray],
    *,
    maximum: int,
    seed: int,
    bounds: np.ndarray,
) -> np.ndarray:
    """Select exactly ``maximum`` valid physical vectors before any RT call."""

    if maximum < 1:
        raise RobertValidationError("posterior predictive draw count must be positive")
    rng = np.random.default_rng(seed)
    if "samples" in arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(names):
            raise RobertDataError("nested posterior samples do not match parameter names")
        if not np.all(np.isfinite(samples)):
            raise RobertDataError("nested posterior samples must be finite")
        _validate_draw_bounds(samples, bounds, "nested posterior samples")
        weights = _normalized_weights(arrays.get("weights"), samples.shape[0])
        indices = rng.choice(samples.shape[0], size=maximum, replace=True, p=weights)
        return np.asarray(samples[indices], dtype=float)
    if "state_vector" in arrays and "covariance" in arrays:
        state = np.asarray(arrays["state_vector"], dtype=float)
        covariance = np.asarray(arrays["covariance"], dtype=float)
        if state.shape != (len(names),) or covariance.shape != (len(names), len(names)):
            raise RobertDataError("OE state or covariance does not match parameter names")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
            raise RobertDataError("OE state and covariance must be finite")
        accepted: list[np.ndarray] = []
        accepted_count = 0
        attempts = 0
        maximum_attempts = max(10_000, 1_000 * maximum)
        while accepted_count < maximum:
            remaining = maximum - accepted_count
            batch_size = max(remaining, min(remaining * 4, 2_048))
            try:
                candidates = np.asarray(
                    rng.multivariate_normal(
                        state,
                        covariance,
                        size=batch_size,
                        check_valid="raise",
                    ),
                    dtype=float,
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                raise RobertDataError("OE covariance cannot generate posterior draws") from error
            attempts += batch_size
            valid = np.all(
                (candidates >= bounds[:, 0]) & (candidates <= bounds[:, 1]),
                axis=1,
            )
            if np.any(valid):
                accepted_batch = candidates[valid]
                accepted.append(accepted_batch)
                accepted_count += accepted_batch.shape[0]
            if attempts >= maximum_attempts and accepted_count < maximum:
                raise RobertValidationError(
                    "bounded Gaussian posterior draw rejection did not produce enough "
                    "in-prior candidates"
                )
        return np.concatenate(accepted, axis=0)[:maximum]
    raise RobertDataError("result arrays contain neither nested samples nor OE state")


def _validate_draw_bounds(draws: np.ndarray, bounds: np.ndarray, label: str) -> None:
    if bounds.shape != (draws.shape[1], 2):
        raise RobertDataError(f"{label} do not match configured parameter bounds")
    valid = np.all(
        (draws >= bounds[:, 0]) & (draws <= bounds[:, 1]),
        axis=1,
    )
    if not np.all(valid):
        raise RobertValidationError(f"{label} contain values outside prior bounds")


def _profile_quantile_mapping(
    profiles: Mapping[str, object],
) -> dict[str, dict[str, np.ndarray | str]]:
    output: dict[str, dict[str, np.ndarray | str]] = {}
    for name, profile in profiles.items():
        output[str(name)] = {
            "pressure": np.asarray(profile.coordinate, dtype=float),
            "pressure_unit": str(profile.coordinate_unit),
            "quantiles": np.asarray(profile.values, dtype=float),
            "value_unit": str(profile.value_unit),
        }
    return output


def _nested_profile_quantile_mapping(
    profiles: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, dict[str, np.ndarray | str]]]:
    return {
        str(region): _profile_quantile_mapping(region_profiles)
        for region, region_profiles in profiles.items()
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
            POSTERIOR_QUANTILE_PROBABILITIES,
            axis=0,
        ),
        "wavelength_unit": reference.spectral_grid.unit,
        "observable": reference.observable,
        "bin_edges": reference.spectral_grid.bin_edges,
        "unit": reference.unit,
    }


def _write_posterior_predictive_quantiles(
    output: Path,
    *,
    native_quantiles: Mapping[str, np.ndarray | str] | None,
    temperature_quantiles: Mapping[str, Mapping[str, np.ndarray | str]],
    vmr_quantiles: Mapping[str, Mapping[str, Mapping[str, np.ndarray | str]]],
    cloud_quantiles: Mapping[str, Mapping[str, Mapping[str, np.ndarray | str]]],
    posterior_models: PosteriorModelCollection | None,
    draw_vectors: np.ndarray,
    parameter_names: Sequence[str],
) -> None:
    """Persist the numerical products behind posterior interval plots."""

    arrays: dict[str, object] = {
        "posterior_draw_vectors": np.asarray(draw_vectors, dtype=float),
        "parameter_names": np.asarray(tuple(parameter_names), dtype=str),
        "quantile_probabilities": np.asarray(
            POSTERIOR_QUANTILE_PROBABILITIES,
            dtype=float,
        ),
        "quantile_labels": np.asarray(POSTERIOR_QUANTILE_LABELS, dtype=str),
    }
    if posterior_models is not None:
        for name, summary in posterior_models.spectra.items():
            prefix = f"spectrum_{name}"
            arrays[f"{prefix}_wavelength"] = np.asarray(summary.wavelength, dtype=float)
            arrays[f"{prefix}_wavelength_unit"] = np.asarray(
                summary.wavelength_unit,
                dtype=str,
            )
            arrays[f"{prefix}_unit"] = np.asarray(summary.unit, dtype=str)
            arrays[f"{prefix}_observable"] = np.asarray(summary.observable, dtype=str)
            _add_quantile_fields(arrays, prefix, summary.values)
            if summary.bin_edges is not None:
                arrays[f"{prefix}_bin_edges"] = np.asarray(summary.bin_edges, dtype=float)
            if summary.mask is not None:
                arrays[f"{prefix}_mask"] = np.asarray(summary.mask, dtype=bool)
    if native_quantiles is not None:
        quantiles = np.asarray(native_quantiles["quantiles"], dtype=float)
        prefix = "native_spectrum"
        arrays[f"{prefix}_wavelength"] = np.asarray(
            native_quantiles["wavelength"], dtype=float
        )
        arrays[f"{prefix}_wavelength_unit"] = np.asarray(
            native_quantiles.get("wavelength_unit", ""),
            dtype=str,
        )
        arrays[f"{prefix}_unit"] = np.asarray(native_quantiles["unit"], dtype=str)
        arrays[f"{prefix}_observable"] = np.asarray(
            native_quantiles.get("observable", ""),
            dtype=str,
        )
        _add_quantile_fields(arrays, prefix, quantiles)
        edges = native_quantiles.get("bin_edges")
        if edges is not None:
            arrays[f"{prefix}_bin_edges"] = np.asarray(edges, dtype=float)
    for name, values in temperature_quantiles.items():
        _add_profile_fields(arrays, f"temperature_{name}", values)
    for region, profiles in vmr_quantiles.items():
        for name, values in profiles.items():
            _add_profile_fields(arrays, f"vmr_{region}_{name}", values)
    for region, profiles in cloud_quantiles.items():
        for name, values in profiles.items():
            _add_profile_fields(arrays, f"cloud_{region}_{name}", values)
    _atomic_savez(output, arrays)


def _add_quantile_fields(
    arrays: dict[str, object],
    prefix: str,
    values: object,
) -> None:
    quantiles = np.asarray(values, dtype=float)
    if (
        quantiles.ndim != 2
        or quantiles.shape[0] != len(POSTERIOR_QUANTILE_LABELS)
        or quantiles.shape[1] < 1
        or not np.all(np.isfinite(quantiles))
    ):
        raise RobertValidationError(
            f"{prefix} must contain five finite posterior quantile rows"
        )
    arrays[f"{prefix}_quantiles"] = quantiles
    for label, row in zip(POSTERIOR_QUANTILE_LABELS, quantiles, strict=True):
        arrays[f"{prefix}_{label}"] = np.asarray(row, dtype=float)


def _add_profile_fields(
    arrays: dict[str, object],
    prefix: str,
    values: Mapping[str, np.ndarray | str],
) -> None:
    arrays[f"{prefix}_pressure"] = np.asarray(values["pressure"], dtype=float)
    arrays[f"{prefix}_pressure_unit"] = np.asarray(
        values["pressure_unit"],
        dtype=str,
    )
    arrays[f"{prefix}_unit"] = np.asarray(values["value_unit"], dtype=str)
    _add_quantile_fields(arrays, prefix, values["quantiles"])


def _atomic_savez(output: Path, arrays: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
    except OSError as error:
        raise RobertDataError(f"failed to write posterior predictive products: {output}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _plot_temperature_profiles(
    profiles: Mapping[str, Mapping[str, np.ndarray | str]],
    output: Path,
    *,
    style: str,
    dpi: int,
) -> None:
    """Plot one independent pressure panel for each atmospheric region."""

    entries = [(str(name), values) for name, values in profiles.items()]
    if not entries:
        return
    plt = _pyplot()
    columns = min(3, len(entries))
    rows = math.ceil(len(entries) / columns)
    with _plot_style_context(plt, style):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.5 * columns, 6.0 * rows),
            squeeze=False,
            sharey=False,
        )
        flat_axes = axes.ravel()
        for index, (name, values) in enumerate(entries):
            _plot_profile_quantiles(
                flat_axes[index],
                values,
                x_label="Temperature",
                title=name,
                legend=index == 0,
            )
        for axis in flat_axes[len(entries) :]:
            axis.set_visible(False)
        figure.suptitle("Posterior temperature-pressure profiles")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_atmospheric_profiles(
    profiles: Mapping[str, Mapping[str, Mapping[str, np.ndarray | str]]],
    output: Path,
    *,
    title: str,
    x_label: str,
    style: str,
    dpi: int,
) -> None:
    """Plot VMR or cloud profiles while retaining each region and species."""

    entries = [
        (f"{region}: {name}", values)
        for region, region_profiles in profiles.items()
        for name, values in region_profiles.items()
    ]
    if not entries:
        return
    plt = _pyplot()
    columns = min(3, len(entries))
    rows = math.ceil(len(entries) / columns)
    with _plot_style_context(plt, style):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.5 * columns, 5.5 * rows),
            squeeze=False,
            sharey=False,
        )
        flat_axes = axes.ravel()
        for axis, (name, values) in zip(flat_axes[: len(entries)], entries, strict=True):
            _plot_profile_quantiles(
                axis,
                values,
                x_label=x_label,
                title=name,
                legend=False,
                logarithmic_x=True,
            )
        for axis in flat_axes[len(entries) :]:
            axis.set_visible(False)
        flat_axes[0].legend(fontsize=8)
        figure.suptitle(title)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_profile_quantiles(
    axis: object,
    values: Mapping[str, np.ndarray | str],
    *,
    x_label: str,
    title: str,
    legend: bool,
    logarithmic_x: bool = False,
) -> None:
    """Draw five posterior quantiles on one pressure axis."""

    pressure = np.asarray(values["pressure"], dtype=float)
    quantiles = np.asarray(values["quantiles"], dtype=float)
    if (
        pressure.ndim != 1
        or pressure.size == 0
        or not np.all(np.isfinite(pressure))
        or np.any(pressure <= 0.0)
        or quantiles.ndim != 2
        or quantiles.shape != (len(POSTERIOR_QUANTILE_LABELS), pressure.size)
        or not np.all(np.isfinite(quantiles))
        or not (
            np.all(np.diff(pressure) > 0.0)
            or np.all(np.diff(pressure) < 0.0)
            or pressure.size == 1
        )
    ):
        raise RobertValidationError("posterior profile plotting values are invalid")
    pressure_unit = str(values.get("pressure_unit", ""))
    value_unit = str(values.get("value_unit", ""))
    axis.fill_betweenx(
        pressure,
        quantiles[0],
        quantiles[4],
        color=ROBERT_COLOR,
        alpha=0.12,
        linewidth=0.0,
        label="95.45% interval",
    )
    axis.fill_betweenx(
        pressure,
        quantiles[1],
        quantiles[3],
        color=ROBERT_COLOR,
        alpha=0.28,
        linewidth=0.0,
        label="68.27% interval",
    )
    axis.plot(
        quantiles[2],
        pressure,
        color=ROBERT_COLOR,
        linewidth=2.0,
        label="Median",
    )
    if logarithmic_x and np.all(quantiles > 0.0):
        dynamic_range = float(np.max(quantiles) / np.min(quantiles))
        if dynamic_range > 10.0:
            axis.set_xscale("log")
        else:
            from matplotlib.ticker import MaxNLocator

            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.set_yscale("log")
    # Use explicit limits so high pressure is always at the bottom, regardless
    # of whether the saved coordinate is increasing or decreasing.
    axis.set_ylim(float(np.max(pressure)), float(np.min(pressure)))
    axis.set_title(title)
    display_unit = "" if value_unit in {"1", "dimensionless", "volume_mixing_ratio"} else value_unit
    axis.set_xlabel(f"{x_label} ({display_unit})" if display_unit else x_label)
    axis.set_ylabel(f"Pressure ({pressure_unit})" if pressure_unit else "Pressure")
    axis.grid(axis="y", alpha=0.16)
    if legend:
        axis.legend(fontsize=8)


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
    with _plot_style_context(plt, style):
        figure, (fit_axis, residual_axis) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": (3, 1)},
        )
        flux_label = "Model and observation"
        if native_quantiles is not None:
            native_wavelength = _wavelength_in_microns(
                native_quantiles["wavelength"],
                str(native_quantiles.get("wavelength_unit", "")),
            )
            native_values = np.asarray(native_quantiles["quantiles"], dtype=float)
            native_scale, flux_label = _plot_scale(str(native_quantiles["unit"]))
            fit_axis.fill_between(
                native_wavelength,
                native_values[0] * native_scale,
                native_values[4] * native_scale,
                color=ROBERT_COLOR,
                alpha=0.12,
                linewidth=0.0,
                label="Native-grid 95.45% posterior predictive interval",
            )
            fit_axis.fill_between(
                native_wavelength,
                native_values[1] * native_scale,
                native_values[3] * native_scale,
                color=ROBERT_COLOR,
                alpha=0.28,
                linewidth=0.0,
                label="Native-grid 68.27% posterior predictive interval",
            )
            fit_axis.plot(
                native_wavelength,
                native_values[2] * native_scale,
                color=ROBERT_COLOR,
                linewidth=1.8,
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
            wavelength = _wavelength_in_microns(
                observation.wavelength,
                observation.wavelength_unit,
            )[valid]
            color = colors[name]
            fit_axis.errorbar(
                wavelength,
                data * scale,
                yerr=uncertainty * scale,
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=3.2,
                linewidth=0.8,
                capsize=1.5,
                alpha=0.9,
                label=observation.instrument or name,
                zorder=5,
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
                plotted_model = quantiles[2]
                fit_axis.fill_between(
                    wavelength,
                    quantiles[0] * scale,
                    quantiles[4] * scale,
                    color=ROBERT_COLOR,
                    alpha=0.12,
                    linewidth=0.0,
                    label=(
                        "95.45% posterior predictive interval"
                        if dataset is problem.observations.datasets[0]
                        else None
                    ),
                )
                fit_axis.fill_between(
                    wavelength,
                    quantiles[1] * scale,
                    quantiles[3] * scale,
                    color=ROBERT_COLOR,
                    alpha=0.28,
                    linewidth=0.0,
                    label=(
                        "68.27% posterior predictive interval"
                        if dataset is problem.observations.datasets[0]
                        else None
                    ),
                )
            fit_axis.plot(
                wavelength,
                plotted_model * scale,
                color=ROBERT_COLOR if quantiles is not None else color,
                linewidth=2.2,
                label=(
                    (
                        "Observation-grid posterior median"
                        if quantiles is not None
                        else "Observation-grid best-fit model"
                    )
                    if dataset is problem.observations.datasets[0]
                    else None
                ),
            )
            residual_axis.plot(
                wavelength,
                (data - model) / uncertainty,
                "o",
                color=RESIDUAL_COLOR,
                markersize=3.0,
                alpha=0.85,
            )
        reduced = statistics.get("reduced_chi_squared")
        reduced_text = "undefined" if reduced is None else f"{float(reduced):.3f}"
        fit_axis.set_ylabel(flux_label)
        fit_axis.set_title(
            f"{problem.name}: retrieval fit (reduced $\\chi^2$ = {reduced_text})"
        )
        fit_axis.grid(axis="y", alpha=0.16)
        fit_axis.legend(fontsize=9, ncol=max(1, min(3, len(colors))))
        residual_axis.axhline(0.0, color="0.2", linewidth=0.8)
        residual_axis.axhline(1.0, color="0.6", linestyle="--", linewidth=0.7)
        residual_axis.axhline(-1.0, color="0.6", linestyle="--", linewidth=0.7)
        residual_axis.set_ylabel(r"(Data - best fit) / $\sigma$")
        residual_axis.set_xlabel(r"Wavelength [$\mu$m]")
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
        with _plot_style_context(plt, style):
            rows = math.ceil(len(names) / 3)
            figure, axes = plt.subplots(rows, 3, figsize=(12, 3.2 * rows))
            flat_axes = np.atleast_1d(axes).ravel()
            for index, name in enumerate(names):
                flat_axes[index].hist(
                    plot_samples[:, index],
                    bins=40,
                    weights=plot_weights,
                    histtype="stepfilled",
                    color=ROBERT_COLOR,
                    edgecolor=ROBERT_COLOR,
                    linewidth=1.4,
                    alpha=0.28,
                    density=True,
                )
                quantile = weighted_quantile(
                    samples[:, index], weights, (0.16, 0.5, 0.84)
                )
                for value, linestyle in zip(quantile, ("--", "-", "--"), strict=True):
                    flat_axes[index].axvline(
                        value,
                        color=ROBERT_COLOR,
                        linewidth=1.5,
                        linestyle=linestyle,
                    )
                flat_axes[index].set_xlabel(labels[name])
                flat_axes[index].set_yticks([])
                flat_axes[index].set_title(
                    _quantile_title(labels[name], quantile),
                    fontsize=10,
                )
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
                plot_samples,
                plot_weights,
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
        with _plot_style_context(plt, style):
            figure, axis = plt.subplots(figsize=(10, max(4, 0.5 * len(names))))
            positions = np.arange(len(names))
            axis.errorbar(
                state,
                positions,
                xerr=sigma,
                fmt="o",
                color=ROBERT_COLOR,
                capsize=2.0,
            )
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
    weights: np.ndarray,
    names: Sequence[str],
    labels: Mapping[str, str],
    output: Path,
    *,
    style: str,
    dpi: int,
) -> None:
    try:
        import corner
    except ImportError as exc:
        raise RobertDataError(
            "corner plots require corner; install the ROBERT diagnostics dependencies"
        ) from exc

    plt = _pyplot()
    with _plot_style_context(plt, style):
        figure = corner.corner(
            samples,
            weights=weights,
            labels=[labels[name] for name in names],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
            color=ROBERT_COLOR,
            plot_datapoints=False,
            fill_contours=True,
            smooth=1.0,
            smooth1d=1.0,
        )
        figure.suptitle("Posterior distribution", y=1.0)
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
    with _plot_style_context(plt, style):
        from matplotlib.colors import LinearSegmentedColormap

        size = max(6.0, 0.65 * len(names))
        figure, axis = plt.subplots(figsize=(size, size))
        color_map = LinearSegmentedColormap.from_list(
            "robert_correlation",
            (REFERENCE_COLOR, "white", ROBERT_COLOR),
        )
        image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap=color_map)
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


def _wavelength_in_microns(values: object, unit: str) -> np.ndarray:
    """Convert a saved spectral coordinate to the common plot unit."""

    grid = SpectralGrid.from_array(np.asarray(values, dtype=float), unit=unit)
    return np.asarray(spectral_grid_values_in_unit(grid, "micron"), dtype=float)


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


def _quantile_title(label: str, quantiles: Sequence[float]) -> str:
    q16, q50, q84 = (float(value) for value in quantiles)
    return f"{label} = {q50:.2f} +{q84 - q50:.2f} / -{q50 - q16:.2f}"


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
            "resolved_style": (
                "robert-science" if style in {"default", "robert"} else style
            ),
            "image_format": image_format,
            "dpi": int(dpi),
        },
    )


def _plot_style_context(plt: object, style: str):
    styles: list[object] = [ROBERT_MATPLOTLIB_STYLE]
    if style not in {"default", "robert"}:
        styles.append(style)
    return plt.style.context(styles)


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
    "load_best_fit_prediction",
    "posterior_summary",
    "postprocess_forward_output",
    "postprocess_retrieval_output",
    "postprocess_saved_best_fit_output",
    "weighted_quantile",
]
