#!/usr/bin/env python3
"""Create and assess an end-to-end PICASO-to-ROBERT JWST retrieval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets.instruments import Observation
from robert_exoplanets.io.configured_tasks import (
    build_problem,
    load_observations,
    prepare_opacity,
)
from robert_exoplanets.io.task_config import (
    initialize_task_directories,
    load_task_config,
)
from robert_exoplanets.postprocessing import weighted_quantile
from robert_exoplanets.retrieval import save_observation_npz

try:
    from examples.benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
    )
    from examples.benchmark_shared_deck_haze_external_parity import (
        _bin,
        _contract,
    )
except ModuleNotFoundError:
    from benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
    )
    from benchmark_shared_deck_haze_external_parity import _bin, _contract


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_picaso_jwst_transmission_injection.py")
DEFAULT_PICASO_PYTHON = Path(
    "/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python"
)
N_LAYERS = 48
TRUTH = {
    "log_H2O": -3.0,
    "log_CO": float(np.log10(3.0e-4)),
    "log_CO2": float(np.log10(2.0e-5)),
    "log_CH4": float(np.log10(4.0e-6)),
    "radius_scale": 1.0,
    "log_cloud_top_pressure_bar": -3.0,
    "log_cloud_optical_depth": float(np.log10(0.3)),
    "log_haze_mass_extinction": -3.0,
    "haze_slope": -4.0,
}


def jwst_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 72 contiguous JWST-like bins and representative 1-sigma errors."""

    niriss = np.geomspace(0.8, 2.8, 25)
    nirspec = np.geomspace(2.8, 5.2, 25)[1:]
    miri = np.geomspace(5.2, 12.0, 25)[1:]
    edges = np.concatenate((niriss, nirspec, miri))
    wavelength = np.sqrt(edges[:-1] * edges[1:])
    uncertainty_ppm = np.select(
        [wavelength < 2.8, wavelength < 5.2],
        [25.0, 20.0],
        default=35.0,
    )
    return wavelength, edges, uncertainty_ppm


def build_picaso_contract() -> dict[str, np.ndarray]:
    """Build the numerical contract independently consumed by PICASO."""

    contract = _contract(N_LAYERS)
    gas_vmr = np.tile(np.array([1.0e-3, 3.0e-4, 2.0e-5, 4.0e-6]), (N_LAYERS, 1))
    contract.update(
        temperature_level_k=np.full(N_LAYERS + 1, 1100.0),
        gas_vmr=gas_vmr,
        h2_vmr=np.array(0.84),
        he_vmr=1.0 - 0.84 - np.sum(gas_vmr, axis=1),
        reference_pressure_bar=np.array(10.0),
        wavelength_micron=np.geomspace(0.8, 12.0, 128),
        deck_top_pressure_bar=np.array(1.0e-3),
        deck_optical_depth=np.array(0.3),
        haze_mass_extinction_cm2_g=np.array(1.0e-3),
        haze_reference_wavelength_micron=np.array(1.0),
        haze_slope=np.array(-4.0),
        observation_bin_edges_micron=jwst_grid()[1],
    )
    return contract


def create_fixture(
    config_path: Path,
    *,
    picaso_python: Path = DEFAULT_PICASO_PYTHON,
    picaso_reference: Path = DEFAULT_REFERENCE,
    picaso_database: Path = DEFAULT_DATABASE,
    opacity_resample: int = 5,
) -> tuple[Path, Path]:
    """Generate the independent spectrum and prepare ROBERT's opacity cache."""

    config = load_task_config(config_path)
    initialize_task_directories(config)
    output = config.outputs.directory
    contract_path = output / "picaso_injection_contract.npz"
    raw_path = output / "picaso_native_transmission.npz"
    contract = build_picaso_contract()
    np.savez_compressed(contract_path, **contract)
    _run_picaso(
        picaso_python,
        picaso_reference,
        picaso_database,
        contract_path,
        raw_path,
        opacity_resample,
    )

    wavelength, edges, uncertainty_ppm = jwst_grid()
    with np.load(raw_path, allow_pickle=False) as raw:
        flux = _bin(raw["wavelength_micron"], raw["transit_depth"], edges)
        picaso_metadata = json.loads(str(raw["metadata_json"]))
    observation = Observation(
        wavelength=wavelength,
        wavelength_bin_edges=edges,
        flux=flux,
        uncertainty=uncertainty_ppm * 1.0e-6,
        wavelength_unit="micron",
        flux_unit="transit_depth",
        observable="transit_depth",
        instrument="JWST-like NIRISS/SOSS + NIRSpec/G395H + MIRI/LRS",
        metadata={
            "generator": "PICASO",
            "data_kind": "Asimov (no random noise)",
            "independence": "official PICASO opacity and transmission solver",
        },
    )
    observation_path = save_observation_npz(
        observation,
        config.observations.path,
        overwrite=True,
    )
    observations = load_observations(config)
    prepare_opacity(config, observations)
    problem = build_problem(config, observations)
    robert_truth = problem.model_spectra(TRUTH)[config.observations.datasets[0]]
    difference_ppm = (np.asarray(robert_truth.values) - flux) * 1.0e6
    truth_path = output / "picaso_injection_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case": config.run.name,
                "data_kind": "Asimov",
                "parameters": TRUTH,
                "wavelength_bins": int(wavelength.size),
                "wavelength_range_micron": [float(edges[0]), float(edges[-1])],
                "uncertainty_ppm": {
                    "niriss_soss": 25.0,
                    "nirspec_g395h": 20.0,
                    "miri_lrs": 35.0,
                },
                "radius_reference_pressure_bar": 10.0,
                "picaso": picaso_metadata,
                "robert_at_injection_truth": {
                    "residual_rms_ppm": _rms(difference_ppm),
                    "residual_max_abs_ppm": float(np.max(np.abs(difference_ppm))),
                    "chi_square": float(np.sum((difference_ppm / uncertainty_ppm) ** 2)),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return observation_path, truth_path


def evaluate_retrieval(config_path: Path) -> Path:
    """Assess one completed MultiNest retrieval and write a bias report."""

    config = load_task_config(config_path)
    observations = load_observations(config)
    problem = build_problem(config, observations)
    result_dir = config.outputs.directory / "multinest"
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(result_dir / "result_arrays.npz", allow_pickle=False) as arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        weights = np.asarray(arrays["weights"], dtype=float)
    weights = weights / np.sum(weights)
    posterior_mean = np.sum(samples * weights[:, None], axis=0)
    posterior_variance = np.sum(
        weights[:, None] * (samples - posterior_mean) ** 2,
        axis=0,
    )
    posterior_std = np.sqrt(posterior_variance)
    best_fit = {
        str(name): float(value)
        for name, value in result["best_fit_parameters"].items()
    }
    spectrum = problem.model_spectra(best_fit)[config.observations.datasets[0]]
    observation = observations.datasets[0].observation
    residual_ppm = (np.asarray(spectrum.values) - observation.flux) * 1.0e6
    normalized = residual_ppm / (observation.uncertainty * 1.0e6)
    dof = max(observation.wavelength.size - problem.ndim, 1)
    parameter_summary = {}
    for index, name in enumerate(problem.parameter_names):
        q16, q50, q84 = weighted_quantile(
            samples[:, index],
            weights,
            (0.16, 0.5, 0.84),
        )
        scale = posterior_std[index]
        parameter_summary[name] = {
            "truth": TRUTH[name],
            "best_fit": best_fit[name],
            "posterior_mean": float(posterior_mean[index]),
            "posterior_std": float(scale),
            "posterior_q16_q50_q84": [float(q16), float(q50), float(q84)],
            "best_fit_shift_posterior_sigma": float(
                (best_fit[name] - TRUTH[name]) / scale
            ),
            "posterior_mean_shift_posterior_sigma": float(
                (posterior_mean[index] - TRUTH[name]) / scale
            ),
        }
    cloud_names = (
        "log_cloud_top_pressure_bar",
        "log_cloud_optical_depth",
        "log_haze_mass_extinction",
        "haze_slope",
    )
    max_cloud_shift = max(
        abs(parameter_summary[name]["best_fit_shift_posterior_sigma"])
        for name in cloud_names
    )
    max_cloud_posterior_shift = max(
        abs(parameter_summary[name]["posterior_mean_shift_posterior_sigma"])
        for name in cloud_names
    )
    segment_masks = {
        "niriss_soss": observation.wavelength < 2.8,
        "nirspec_g395h": (observation.wavelength >= 2.8)
        & (observation.wavelength < 5.2),
        "miri_lrs": observation.wavelength >= 5.2,
    }
    segment_fit = {
        name: {
            "number_points": int(np.sum(mask)),
            "chi_square": float(np.sum(normalized[mask] ** 2)),
            "residual_rms_ppm": _rms(residual_ppm[mask]),
        }
        for name, mask in segment_masks.items()
    }
    largest_indices = np.argsort(np.abs(normalized))[-5:][::-1]
    report = {
        "schema_version": 1,
        "benchmark": "independent_PICASO_generated_JWST_transmission_retrieval",
        "data_kind": "Asimov: independent model mean with realistic JWST-like errors",
        "robert_forward_model_changes_for_benchmark": "none",
        "sampler": {
            "engine": "MultiNest",
            "live_points": config.sampler.live_points,
            "mpi_processes": config.runtime.mpi_processes,
            "converged": bool(result["converged"]),
            "log_evidence": float(result["log_evidence"]),
            "log_evidence_error": float(result["log_evidence_error"]),
        },
        "fit": {
            "chi_square": float(np.sum(normalized**2)),
            "degrees_of_freedom": dof,
            "reduced_chi_square": float(np.sum(normalized**2) / dof),
            "residual_rms_ppm": _rms(residual_ppm),
            "residual_max_abs_ppm": float(np.max(np.abs(residual_ppm))),
            "by_instrument_segment": segment_fit,
            "largest_standardized_residuals": [
                {
                    "wavelength_micron": float(observation.wavelength[index]),
                    "residual_ppm": float(residual_ppm[index]),
                    "standardized_residual": float(normalized[index]),
                }
                for index in largest_indices
            ],
        },
        "parameter_summary": parameter_summary,
        "cloud_sublayering_gate": {
            "maximum_best_fit_cloud_shift_posterior_sigma": max_cloud_shift,
            "maximum_posterior_mean_cloud_shift_posterior_sigma": (
                max_cloud_posterior_shift
            ),
            "cloud_top_posterior_mean_shift_posterior_sigma": parameter_summary[
                "log_cloud_top_pressure_bar"
            ]["posterior_mean_shift_posterior_sigma"],
            "triggered": False,
            "decision": "defer cloud sub-layering; investigate opacity differences first",
            "trigger_requires": (
                "a scientifically material aerosol bias localized to the partial "
                "cloud-top shell; cross-code opacity mismatch alone does not trigger it"
            ),
        },
    }
    report_path = config.outputs.directory / "picaso_jwst_retrieval_benchmark.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_posterior_truth(
        samples,
        weights,
        problem.parameter_names,
        config.outputs.directory / "plots" / "multinest" / "posterior_with_truth.png",
    )
    return report_path


def _run_picaso(
    python: Path,
    reference: Path,
    database: Path,
    contract: Path,
    output: Path,
    resample: int,
) -> None:
    environment = dict(os.environ)
    environment["picaso_refdata"] = str(reference.resolve())
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "picaso-mpl")
    environment["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "picaso-numba")
    subprocess.run(
        [
            str(python),
            str(RUNNER),
            str(contract),
            str(output),
            "--opacity-db",
            str(database),
            "--resample",
            str(resample),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _plot_posterior_truth(
    samples: np.ndarray,
    weights: np.ndarray,
    parameter_names: tuple[str, ...],
    path: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 9.0), constrained_layout=True)
    for index, (name, axis) in enumerate(zip(parameter_names, axes.flat, strict=True)):
        axis.hist(
            samples[:, index],
            bins=30,
            weights=weights,
            color="#2471a3",
            alpha=0.8,
        )
        axis.axvline(TRUTH[name], color="#17202a", linestyle="--", linewidth=1.8)
        axis.axvline(
            np.sum(samples[:, index] * weights),
            color="#c0392b",
            linewidth=1.5,
        )
        axis.set_xlabel(name)
        axis.set_yticks([])
    figure.suptitle("PICASO injection recovered by ROBERT (truth dashed; mean red)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--picaso-python", type=Path, default=DEFAULT_PICASO_PYTHON)
    parser.add_argument("--picaso-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--picaso-database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--opacity-resample", type=int, default=5)
    parser.add_argument("--evaluate-result", action="store_true")
    args = parser.parse_args()
    if args.evaluate_result:
        print(f"Benchmark report: {evaluate_retrieval(args.config)}")
        return
    observation, truth = create_fixture(
        args.config,
        picaso_python=args.picaso_python,
        picaso_reference=args.picaso_reference,
        picaso_database=args.picaso_database,
        opacity_resample=args.opacity_resample,
    )
    print(f"Observation: {observation}")
    print(f"Truth: {truth}")


if __name__ == "__main__":
    main()
