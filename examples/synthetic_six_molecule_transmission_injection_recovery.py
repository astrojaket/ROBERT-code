#!/usr/bin/env python3
"""Create and evaluate the six-molecule transmission injection fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

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

UNCERTAINTY_PPM = 8.0
N_WAVELENGTH = 96
OPACITY_DESCRIPTION = (
    "ExoMolOP R=15000 H2O/POKAZATEL, CO/Li2015, CO2/UCL-4000, "
    "CH4/YT34to10, NH3/CoYuTe, HCN/Harris"
)
ABSOLUTE_TOLERANCES = {
    "log_H2O": 0.30,
    "log_CO": 0.35,
    "log_CO2": 0.30,
    "log_CH4": 0.35,
    "log_NH3": 0.35,
    "log_HCN": 0.40,
    "radius_scale": 0.001,
    "log_cloud_top_pressure_bar": 0.35,
    "log_cloud_optical_depth": 0.50,
    "log_haze_mass_extinction": 0.75,
    "haze_slope": 1.50,
}


def create_fixture(config_path: Path) -> tuple[Path, Path]:
    """Write the observation and six prepared opacity tables selected by YAML."""

    config = load_task_config(config_path)
    initialize_task_directories(config)
    wavelength = np.geomspace(0.6, 12.0, N_WAVELENGTH)
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
        instrument="Synthetic 0.6-12 micron spectrograph",
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
        name="synthetic six-molecule transmission injection grid",
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
        seed=config.sampler.seed,
        instrument="Synthetic 0.6-12 micron spectrograph",
        metadata={
            "case": config.run.name,
            "opacity": OPACITY_DESCRIPTION,
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
                "seed": config.sampler.seed,
                "uncertainty_ppm": UNCERTAINTY_PPM,
                "parameters": truth,
                "opacity": OPACITY_DESCRIPTION,
                "wavelength_micron": [float(wavelength[0]), float(wavelength[-1])],
                "n_wavelength": int(wavelength.size),
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
        absolute_tolerances={
            name: ABSOLUTE_TOLERANCES[name] for name in problem.parameter_names
        },
        observation=observations.datasets[0].observation,
        best_fit_spectrum=best_fit,
        seed=int(truth_payload["seed"]),
        parameter_order=problem.parameter_names,
        posterior_covariance=covariance,
        inference_converged=bool(result["converged"]),
        reduced_chi_square_bounds=(0.7, 1.3),
        metadata={
            "opacity": str(truth_payload["opacity"]),
            "sampler": "MultiNest",
            "mpi_processes": str(config.runtime.mpi_processes),
            "live_points": str(config.sampler.live_points),
            "wavelength_bins": str(truth_payload["n_wavelength"]),
        },
    )
    return write_injection_recovery_report(
        report,
        config.outputs.directory / "injection_recovery_report.json",
    )


def export_validation_snapshot(config_path: Path, destination: Path) -> None:
    """Copy the compact recovery record and plots into a tracked snapshot."""

    config = load_task_config(config_path)
    output = config.outputs.directory
    plot_dir = output / "plots" / "multinest"
    result_dir = output / "multinest"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "injection_truth.json": output / "injection_truth.json",
        "injection_recovery_report.json": output / "injection_recovery_report.json",
        "multinest_result.json": result_dir / "result.json",
        "sampler_status.json": result_dir / "sampler_status.json",
        "posterior_summary.json": plot_dir / "posterior_summary.json",
        "fit_spectrum_residuals.png": plot_dir / "fit_spectrum_residuals.png",
        "posterior_marginals.png": plot_dir / "posterior_marginals.png",
        "parameter_correlation.png": plot_dir / "parameter_correlation.png",
    }
    for name, source in sources.items():
        shutil.copyfile(source, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evaluate-result",
        action="store_true",
        help="evaluate an existing MultiNest result instead of regenerating data",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        help="copy compact recovery outputs to this validation directory",
    )
    args = parser.parse_args()
    if args.evaluate_result:
        print(f"Recovery: {evaluate_recovery(args.config)}")
        if args.validation_dir is not None:
            export_validation_snapshot(args.config, args.validation_dir)
            print(f"Validation snapshot: {args.validation_dir}")
        return
    observation, truth = create_fixture(args.config)
    print(f"Observation: {observation}")
    print(f"Truth: {truth}")


if __name__ == "__main__":
    main()
