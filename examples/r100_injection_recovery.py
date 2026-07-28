#!/usr/bin/env python3
"""Generate and evaluate ROBERT's local R=100 injection-recovery checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from robert_exoplanets import SpectralGrid, Spectrum, weighted_quantile
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
    infer_wavelength_bin_edges,
)
from robert_exoplanets.io.configured_tasks import (
    build_problem,
    load_observations,
    prepare_opacity,
    run_forward_task,
)
from robert_exoplanets.io.task_config import (
    initialize_task_directories,
    load_task_config,
)
from robert_exoplanets.retrieval import save_observation_npz
from robert_exoplanets.validation import inject_spectrum


UNCERTAINTY_PPM = 60.0
WAVELENGTH_MIN_MICRON = 1.10
WAVELENGTH_MAX_MICRON = 1.70
N_WAVELENGTH = 18
CONFIDENCE_LEVEL = 0.95
REDUCED_CHI_SQUARE_BOUNDS = (0.35, 1.90)


def generate_fixture(config_path: Path) -> dict[str, object]:
    """Generate a forward truth and use it to create a noisy observation."""

    config = load_task_config(config_path)
    initialize_task_directories(config)
    dataset_name = config.observations.datasets[0]
    wavelength = np.geomspace(
        WAVELENGTH_MIN_MICRON,
        WAVELENGTH_MAX_MICRON,
        N_WAVELENGTH,
    )
    edges = infer_wavelength_bin_edges(wavelength)
    observable, flux_unit = _observable(config.radiative_transfer.model)
    uncertainty = np.full(wavelength.shape, UNCERTAINTY_PPM * 1.0e-6)
    placeholder = Observation(
        wavelength=wavelength,
        wavelength_bin_edges=edges,
        flux=np.full(wavelength.shape, 0.01),
        uncertainty=uncertainty,
        wavelength_unit="micron",
        flux_unit=flux_unit,
        observable=observable,
        instrument="Synthetic WFC3-like spectrograph",
        metadata={"validation_only": "true"},
    )
    save_observation_npz(placeholder, config.observations.path, overwrite=True)
    observations = ObservationCollection(
        datasets=(
            ObservationDataset(
                name=dataset_name,
                observation=placeholder,
            ),
        ),
        name=f"{config.run.name} synthetic grid",
    )
    prepare_opacity(config, observations)
    forward_path = run_forward_task(config, config_path)

    with np.load(forward_path, allow_pickle=False) as archive:
        model_wavelength = np.asarray(
            archive[f"{dataset_name}_wavelength_micron"],
            dtype=float,
        )
        model_values = np.asarray(archive[f"{dataset_name}_model"], dtype=float)
        truth = {
            name.removeprefix("parameter_"): float(archive[name])
            for name in archive.files
            if name.startswith("parameter_")
        }
    spectrum = Spectrum(
        spectral_grid=SpectralGrid(
            values=model_wavelength,
            bin_edges=edges,
            unit="micron",
            role="observed",
        ),
        values=model_values,
        unit=flux_unit,
        observable=observable,
    )
    if config.sampler.seed is None:
        raise ValueError("R=100 validation configurations require a sampler seed")
    injected = inject_spectrum(
        spectrum,
        uncertainty,
        seed=config.sampler.seed,
        instrument="Synthetic WFC3-like spectrograph",
        metadata={
            "case": config.run.name,
            "opacity": "ROBERT bundled ExoMolOP H2O R=100",
            "uncertainty_ppm": f"{UNCERTAINTY_PPM:g}",
        },
    )
    observation_path = save_observation_npz(
        injected,
        config.observations.path,
        overwrite=True,
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "case": config.run.name,
        "radiative_transfer": config.radiative_transfer.model,
        "seed": config.sampler.seed,
        "uncertainty_ppm": UNCERTAINTY_PPM,
        "noise_scale": 1.0,
        "wavelength_min_micron": WAVELENGTH_MIN_MICRON,
        "wavelength_max_micron": WAVELENGTH_MAX_MICRON,
        "n_wavelength": N_WAVELENGTH,
        "parameters": truth,
        "forward_model": forward_path.name,
        "forward_model_sha256": _sha256(forward_path),
        "synthetic_observation": observation_path.name,
        "synthetic_observation_sha256": _sha256(observation_path),
        "opacity": "ROBERT bundled ExoMolOP H2O R=100",
    }
    truth_path = config.outputs.directory / "injection_truth.json"
    truth_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def evaluate_recovery(config_path: Path) -> dict[str, object]:
    """Compare a completed MultiNest posterior with the injected truth."""

    config = load_task_config(config_path)
    if config.sampler.engine != "multinest":
        raise ValueError("R=100 validation retrievals must use MultiNest")
    result_directory = config.outputs.directory / "multinest"
    result = json.loads((result_directory / "result.json").read_text(encoding="utf-8"))
    truth_payload = json.loads(
        (config.outputs.directory / "injection_truth.json").read_text(encoding="utf-8")
    )
    truth = {
        str(name): float(value) for name, value in truth_payload["parameters"].items()
    }
    parameter_names = tuple(str(name) for name in result["parameter_names"])
    with np.load(result_directory / "result_arrays.npz", allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        weights = (
            np.asarray(archive["weights"], dtype=float)
            if "weights" in archive.files
            else np.ones(samples.shape[0], dtype=float)
        )
    weights = weights / np.sum(weights)
    lower_probability = (1.0 - CONFIDENCE_LEVEL) / 2.0
    probabilities = (
        lower_probability,
        0.16,
        0.50,
        0.84,
        1.0 - lower_probability,
    )
    summaries: dict[str, dict[str, float | bool]] = {}
    posterior_median: dict[str, float] = {}
    for index, name in enumerate(parameter_names):
        quantiles = weighted_quantile(samples[:, index], weights, probabilities)
        mean = float(np.sum(samples[:, index] * weights))
        variance = float(np.sum(np.square(samples[:, index] - mean) * weights))
        standard_deviation = float(np.sqrt(variance))
        injected = truth[name]
        posterior_median[name] = float(quantiles[2])
        summaries[name] = {
            "truth": injected,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "q02_5": float(quantiles[0]),
            "q16": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q84": float(quantiles[3]),
            "q97_5": float(quantiles[4]),
            "truth_in_95_percent_interval": bool(
                quantiles[0] <= injected <= quantiles[4]
            ),
            "median_standardized_error": (
                abs(float(quantiles[2]) - injected) / standard_deviation
            ),
        }

    observations = load_observations(config)
    problem = build_problem(config, observations)
    dataset_name = config.observations.datasets[0]
    model = problem.model_spectra(posterior_median)[dataset_name]
    observation = observations.datasets[0].observation
    residual = (observation.flux - model.values) / observation.uncertainty
    degrees_of_freedom = observation.n_points - problem.ndim
    chi_square = float(np.sum(np.square(residual)))
    reduced_chi_square = chi_square / degrees_of_freedom
    fit_passed = bool(
        REDUCED_CHI_SQUARE_BOUNDS[0]
        <= reduced_chi_square
        <= REDUCED_CHI_SQUARE_BOUNDS[1]
    )
    parameters_passed = all(
        bool(summary["truth_in_95_percent_interval"]) for summary in summaries.values()
    )
    converged = bool(result["converged"])
    report: dict[str, object] = {
        "schema_version": "1.0",
        "case": config.run.name,
        "passed": converged and fit_passed and parameters_passed,
        "inference_converged": converged,
        "parameters_passed": parameters_passed,
        "fit_passed": fit_passed,
        "confidence_level": CONFIDENCE_LEVEL,
        "parameter_recoveries": summaries,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "reduced_chi_square_bounds": list(REDUCED_CHI_SQUARE_BOUNDS),
        "n_wavelength": observation.n_points,
        "n_parameters": problem.ndim,
        "effective_sample_size": float(1.0 / np.sum(np.square(weights))),
        "result": "multinest/result.json",
        "truth": "injection_truth.json",
    }
    report_path = config.outputs.directory / "injection_recovery_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _observable(model: str) -> tuple[str, str]:
    if model == "emission":
        return "eclipse_depth", "eclipse_depth"
    if model == "transmission":
        return "transit_depth", "transit_depth"
    raise ValueError(f"unsupported radiative-transfer model: {model}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--generate",
        action="store_true",
        help="write the forward truth and synthetic observation",
    )
    actions.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate an existing MultiNest result against the truth",
    )
    args = parser.parse_args(argv)
    if args.generate:
        payload = generate_fixture(args.config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    report = evaluate_recovery(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
