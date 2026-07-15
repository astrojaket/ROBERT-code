#!/usr/bin/env python3
"""Create the deterministic YAML-driven synthetic transmission fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
)
from robert_exoplanets.io.task_config import (
    initialize_task_directories,
    load_task_config,
)
from robert_exoplanets.retrieval import save_observation_npz
from robert_exoplanets.validation import (
    evaluate_injection_recovery,
    inject_spectrum,
    write_injection_recovery_report,
)

SEED = 20260715
UNCERTAINTY_PPM = 12.0


def create_fixture(config_path: Path) -> tuple[Path, Path]:
    """Write the observation and prepared opacity selected by the YAML file."""

    config = load_task_config(config_path)
    initialize_task_directories(config)
    wavelength = np.geomspace(0.85, 5.0, 32)
    edges = infer_wavelength_bin_edges(wavelength)
    uncertainty = np.full(wavelength.shape, UNCERTAINTY_PPM * 1.0e-6)
    placeholder = Observation(
        wavelength=wavelength,
        wavelength_bin_edges=edges,
        flux=np.full(wavelength.shape, 0.018),
        uncertainty=uncertainty,
        wavelength_unit="micron",
        flux_unit="transit_depth",
        observable="transit_depth",
        instrument="Synthetic spectrograph",
        metadata={"validation_only": "true"},
    )
    save_observation_npz(placeholder, config.observations.path, overwrite=True)

    observations = ObservationCollection(
        datasets=(
            ObservationDataset(
                name=config.observations.datasets[0],
                observation=placeholder,
            ),
        ),
        name="synthetic transmission injection grid",
    )
    prepare_opacity(config, observations)
    problem = build_problem(config, observations)
    truth = {
        parameter.name: float(item.value)
        for item, parameter in zip(
            config.parameters,
            problem.parameters.parameters,
            strict=True,
        )
    }
    spectrum = problem.model_spectra(truth)[config.observations.datasets[0]]
    injected = inject_spectrum(
        spectrum,
        uncertainty,
        seed=SEED,
        instrument="Synthetic spectrograph",
        metadata={
            "case": config.run.name,
            "opacity": "ExoMolOP H2O POKAZATEL R=15000",
        },
    )
    observation_path = save_observation_npz(
        injected,
        config.observations.path,
        overwrite=True,
    )
    truth_path = config.outputs.directory / "injection_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "case": config.run.name,
                "seed": SEED,
                "uncertainty_ppm": UNCERTAINTY_PPM,
                "parameters": truth,
                "opacity": "ExoMolOP H2O POKAZATEL R=15000",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return observation_path, truth_path


def evaluate_recovery(config_path: Path) -> Path:
    """Evaluate the completed configured MultiNest run against its injection."""

    config = load_task_config(config_path)
    observations = load_observations(config)
    problem = build_problem(config, observations)
    result_dir = config.outputs.directory / "multinest"
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    truth_payload = json.loads(
        (config.outputs.directory / "injection_truth.json").read_text(
            encoding="utf-8"
        )
    )
    truth = {
        str(name): float(value)
        for name, value in truth_payload["parameters"].items()
    }
    estimates = {
        str(name): float(value)
        for name, value in result["best_fit_parameters"].items()
    }
    with np.load(result_dir / "result_arrays.npz", allow_pickle=False) as arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        weights = np.asarray(arrays["weights"], dtype=float)
    weights /= weights.sum()
    posterior_mean = np.sum(samples * weights[:, None], axis=0)
    centered = samples - posterior_mean
    covariance = (centered * weights[:, None]).T @ centered
    best_fit = problem.model_spectra(estimates)[config.observations.datasets[0]]
    report = evaluate_injection_recovery(
        case_name=config.run.name,
        truth=truth,
        estimates=estimates,
        absolute_tolerances={"log_H2O": 0.15, "radius_scale": 0.001},
        observation=observations.datasets[0].observation,
        best_fit_spectrum=best_fit,
        seed=int(truth_payload["seed"]),
        parameter_order=problem.parameter_names,
        posterior_covariance=covariance,
        inference_converged=bool(result["converged"]),
        reduced_chi_square_bounds=(0.5, 1.5),
        metadata={
            "opacity": str(truth_payload["opacity"]),
            "sampler": "MultiNest",
            "mpi_processes": "2",
            "live_points": "40",
        },
    )
    return write_injection_recovery_report(
        report,
        config.outputs.directory / "injection_recovery_report.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evaluate-result",
        action="store_true",
        help="evaluate an existing MultiNest result instead of regenerating data",
    )
    args = parser.parse_args()
    if args.evaluate_result:
        print(f"Recovery: {evaluate_recovery(args.config)}")
        return
    observation, truth = create_fixture(args.config)
    print(f"Observation: {observation}")
    print(f"Truth: {truth}")


if __name__ == "__main__":
    main()
