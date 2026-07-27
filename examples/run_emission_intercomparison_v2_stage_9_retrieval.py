#!/usr/bin/env python3
"""Run one frozen Stage-9 MultiNest retrieval under a 12-rank addqueue job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_intercomparison_v2_stage_9_native import (  # noqa: E402
    MOLECULAR_SPECIES,
    atmospheric_state,
    build_native_forward,
    load_common_contract,
)
from robert_exoplanets import Observation  # noqa: E402
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    MULTINEST_SETTINGS,
    build_run_matrix,
    parameter_definitions,
)
from robert_exoplanets.retrieval import (  # noqa: E402
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    UniformPrior,
    run_retrieval,
)

POSTERIOR_ENVELOPE_QUANTILES = (0.025, 0.16, 0.50, 0.84, 0.975)


def _mpi() -> tuple[int, Any | None]:
    try:
        from mpi4py import MPI
    except ImportError:
        return 0, None
    communicator = MPI.COMM_WORLD
    return int(communicator.Get_rank()), communicator


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, sorted_values))


def _weighted_column_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise RuntimeError("posterior values and weights have incompatible shapes")
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("posterior values are empty or non-finite")
    return tuple(
        np.asarray(
            [
                _weighted_quantile(values[:, column], weights, quantile)
                for column in range(values.shape[1])
            ],
            dtype=float,
        )
        for quantile in POSTERIOR_ENVELOPE_QUANTILES
    )


def _weighted_spectral_quantiles(
    spectra: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = _weighted_column_quantiles(spectra, weights)
    return quantiles[1], quantiles[2], quantiles[3]


def _posterior_chemistry(
    parameter_names: tuple[str, ...],
    samples: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    active = np.column_stack(
        [
            np.power(
                10.0,
                samples[:, parameter_names.index(f"log10_vmr_{species}")],
            )
            for species in MOLECULAR_SPECIES
        ]
    )
    remainder = 1.0 - np.sum(active, axis=1)
    if np.any(remainder <= 0.0):
        raise RuntimeError("saved posterior contains an invalid active-gas VMR sum")
    chemistry = np.column_stack(
        (remainder * 0.8547, remainder * 0.1453, active)
    )
    return ("H2", "He", *MOLECULAR_SPECIES), chemistry


def _distributed_posterior_spectra(
    parameter_names: tuple[str, ...],
    samples: np.ndarray,
    forward: Any,
    communicator: Any,
) -> np.ndarray | None:
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    indices = np.arange(rank, samples.shape[0], size, dtype=np.int64)
    if indices.size:
        local = np.asarray(
            [
                forward.eclipse_depth(
                    dict(zip(parameter_names, samples[index], strict=True))
                )
                for index in indices
            ],
            dtype=float,
        )
    else:
        local = np.empty((0, np.asarray(forward.r100_centers).size), dtype=float)
    gathered = communicator.gather((indices, local), root=0)
    if rank != 0:
        return None
    if not gathered:
        raise RuntimeError("MPI posterior-spectrum gather returned no payloads")
    wavelength_count = next(
        (values.shape[1] for _, values in gathered if values.ndim == 2 and values.size),
        0,
    )
    if wavelength_count == 0:
        raise RuntimeError("posterior-spectrum gather returned no spectra")
    spectra = np.empty((samples.shape[0], wavelength_count), dtype=float)
    for gathered_indices, values in gathered:
        if values.shape != (gathered_indices.size, wavelength_count):
            raise RuntimeError("MPI posterior-spectrum payload has an invalid shape")
        spectra[gathered_indices] = values
    return spectra


def _distributed_posterior_temperatures(
    parameter_names: tuple[str, ...],
    samples: np.ndarray,
    common: dict[str, Any],
    scenario: str,
    communicator: Any,
) -> np.ndarray | None:
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    indices = np.arange(rank, samples.shape[0], size, dtype=np.int64)
    pressure_count = int(
        next(
            item["n_cells"]
            for item in common["pressure_grids"]
            if item["n_cells"] == 80
        )
    )
    if indices.size:
        local = np.asarray(
            [
                atmospheric_state(
                    common,
                    scenario,
                    dict(zip(parameter_names, samples[index], strict=True)),
                ).temperature_cells_k
                for index in indices
            ],
            dtype=float,
        )
    else:
        local = np.empty((0, pressure_count), dtype=float)
    gathered = communicator.gather((indices, local), root=0)
    if rank != 0:
        return None
    if not gathered:
        raise RuntimeError("MPI posterior-TP gather returned no payloads")
    profiles = np.empty((samples.shape[0], pressure_count), dtype=float)
    for gathered_indices, values in gathered:
        if values.shape != (gathered_indices.size, pressure_count):
            raise RuntimeError("MPI posterior-TP payload has an invalid shape")
        profiles[gathered_indices] = values
    if not np.all(np.isfinite(profiles)):
        raise RuntimeError("MPI posterior-TP gather returned non-finite profiles")
    return profiles


def _load_saved_posterior_samples(
    run_directory: str | Path,
    parameter_count: int,
) -> np.ndarray:
    """Load the rank-0 serialized posterior after the retrieval MPI barrier."""

    arrays_path = Path(run_directory) / "result_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        if "samples" not in archive.files:
            raise RuntimeError(f"saved retrieval arrays contain no samples: {arrays_path}")
        samples = np.asarray(archive["samples"], dtype=float)
    if samples.ndim != 2 or samples.shape[1] != parameter_count:
        raise RuntimeError(
            "saved posterior samples have an incompatible parameter dimension"
        )
    if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise RuntimeError("saved posterior samples are empty or non-finite")
    return samples


def _load_observation(run: dict[str, Any]) -> Observation:
    injection_path = Path(run["injection_product"])
    with np.load(injection_path, allow_pickle=False) as archive:
        wavelength = np.asarray(archive["wavelength_micron"], dtype=float)
        mean = np.asarray(archive["eclipse_depth"], dtype=float)
    sigma = np.full(mean.size, float(run["noise_ppm"]) * 1.0e-6)
    if run["noise_vector"] is None:
        data = mean
    else:
        with np.load(Path(run["noise_vector"]), allow_pickle=False) as archive:
            noise = np.asarray(archive["standard_normal"], dtype=float)
        if noise.shape != mean.shape:
            raise RuntimeError("Stage-9 noise and injection grids do not match")
        data = mean + sigma * noise
    return Observation.from_arrays(
        wavelength,
        data,
        sigma,
        wavelength_unit="micron",
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
        instrument="stage9-r100",
    )


def _validate_run(run: dict[str, Any]) -> None:
    expected = {item.run_id: item for item in build_run_matrix()}
    if run.get("run_id") not in expected:
        raise RuntimeError("run configuration is not in the frozen Stage-9 matrix")
    item = expected[run["run_id"]]
    for name in (
        "scenario",
        "injector",
        "retriever",
        "noise_ppm",
        "noise_id",
        "control",
        "shard_id",
        "sampler_seed",
    ):
        if run.get(name) != getattr(item, name):
            raise RuntimeError(f"run configuration changed frozen field: {name}")
    if run.get("mpi_ranks") != 12 or run.get("threads_per_rank") != 1:
        raise RuntimeError(
            "Stage-9 retrievals require exactly 12 MPI ranks and one thread each"
        )
    expected_sampler = {"engine": "multinest", **MULTINEST_SETTINGS}
    legacy_quiet_sampler = {**expected_sampler, "verbose": False}
    if run.get("sampler") not in (expected_sampler, legacy_quiet_sampler):
        raise RuntimeError("run configuration changed the frozen MultiNest settings")


def _compact_success_products(
    run: dict[str, Any],
    result: Any,
    forward: Any,
    posterior_spectra: np.ndarray,
    posterior_temperatures: np.ndarray,
    common: dict[str, Any],
) -> None:
    output = Path(run["run_directory"])
    arrays_path = output / "result_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        loglike = np.asarray(archive["log_likelihood"], dtype=float)
        weights = (
            np.asarray(archive["weights"], dtype=float)
            if "weights" in archive.files
            else np.full(samples.shape[0], 1.0 / samples.shape[0])
        )
    weights = weights / np.sum(weights)
    if posterior_spectra.shape[0] != samples.shape[0]:
        raise RuntimeError(
            "posterior spectral sample count differs from saved posterior"
        )
    (
        spectrum_q025,
        spectrum_q16,
        spectrum_q50,
        spectrum_q84,
        spectrum_q975,
    ) = _weighted_column_quantiles(
        posterior_spectra,
        weights,
    )
    if posterior_temperatures.shape[0] != samples.shape[0]:
        raise RuntimeError(
            "posterior TP sample count differs from saved posterior"
        )
    tp_q025, tp_q16, tp_q50, tp_q84, tp_q975 = _weighted_column_quantiles(
        posterior_temperatures,
        weights,
    )
    names = tuple(result.parameter_names)
    chemistry_names, posterior_chemistry = _posterior_chemistry(names, samples)
    (
        chemistry_q025,
        chemistry_q16,
        chemistry_q50,
        chemistry_q84,
        chemistry_q975,
    ) = _weighted_column_quantiles(posterior_chemistry, weights)
    definitions = {item.name: item for item in parameter_definitions(run["scenario"])}
    median = {
        name: _weighted_quantile(samples[:, index], weights, 0.5)
        for index, name in enumerate(names)
    }
    intervals = {
        name: {
            "q025": _weighted_quantile(samples[:, index], weights, 0.025),
            "q16": _weighted_quantile(samples[:, index], weights, 0.16),
            "q50": median[name],
            "q84": _weighted_quantile(samples[:, index], weights, 0.84),
            "q975": _weighted_quantile(samples[:, index], weights, 0.975),
            "posterior_standard_deviation": float(
                np.sqrt(
                    np.sum(
                        weights
                        * (samples[:, index] - np.sum(weights * samples[:, index])) ** 2
                    )
                )
            ),
        }
        for index, name in enumerate(names)
    }
    for name, values in intervals.items():
        values["median_bias"] = values["q50"] - definitions[name].truth
        scale = values["posterior_standard_deviation"]
        values["median_bias_posterior_sigma"] = (
            values["median_bias"] / scale if scale > 0.0 else None
        )
        values["truth_covered_68"] = (
            values["q16"] <= definitions[name].truth <= values["q84"]
        )
        values["truth_covered_95"] = (
            values["q025"] <= definitions[name].truth <= values["q975"]
        )
    best = dict(result.best_fit_parameters)
    best_spectrum = forward.eclipse_depth(best)
    median_spectrum = spectrum_q50
    best_state = atmospheric_state(
        common,
        run["scenario"],
        best,
    )
    if best_state.gas_names != chemistry_names:
        raise RuntimeError("diagnostic chemistry species order is inconsistent")
    pressure = np.asarray(
        next(
            item["centers_bar"]
            for item in common["pressure_grids"]
            if item["n_cells"] == 80
        ),
        dtype=float,
    )
    observation = _load_observation(run)
    best_residual = best_spectrum - observation.flux
    median_residual = median_spectrum - observation.flux
    degrees_of_freedom = max(observation.n_points - len(names), 1)
    np.savez_compressed(
        output / "diagnostic_spectra.npz",
        wavelength_micron=observation.wavelength,
        observed_eclipse_depth=observation.flux,
        observational_uncertainty_eclipse_depth=observation.uncertainty,
        best_fit_eclipse_depth=best_spectrum,
        posterior_spectrum_q025_eclipse_depth=spectrum_q025,
        posterior_spectrum_q16_eclipse_depth=spectrum_q16,
        posterior_spectrum_q50_eclipse_depth=spectrum_q50,
        posterior_spectrum_q84_eclipse_depth=spectrum_q84,
        posterior_spectrum_q975_eclipse_depth=spectrum_q975,
        posterior_spectral_sample_count=np.asarray(samples.shape[0], dtype=np.int64),
    )
    np.savez_compressed(
        output / "diagnostic_tp.npz",
        pressure_bar=pressure,
        best_fit_temperature_k=best_state.temperature_cells_k,
        posterior_temperature_q025_k=tp_q025,
        posterior_temperature_q16_k=tp_q16,
        posterior_temperature_q50_k=tp_q50,
        posterior_temperature_q84_k=tp_q84,
        posterior_temperature_q975_k=tp_q975,
        posterior_temperature_sample_count=np.asarray(
            samples.shape[0], dtype=np.int64
        ),
    )
    np.savez_compressed(
        output / "diagnostic_chemistry.npz",
        species=np.asarray(chemistry_names),
        best_fit_vmr=best_state.gas_vmr,
        posterior_vmr_q025=chemistry_q025,
        posterior_vmr_q16=chemistry_q16,
        posterior_vmr_q50=chemistry_q50,
        posterior_vmr_q84=chemistry_q84,
        posterior_vmr_q975=chemistry_q975,
        posterior_chemistry_sample_count=np.asarray(
            samples.shape[0], dtype=np.int64
        ),
        vertically_constant=np.asarray(True),
    )
    (output / "posterior_envelope_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run["run_id"],
                "sample_count": int(samples.shape[0]),
                "sample_selection": "all_saved_weighted_posterior_samples",
                "quantiles": list(POSTERIOR_ENVELOPE_QUANTILES),
                "spectral_forward_framework": run["retriever"],
                "temperature_parameterization": "ParmentierGuillot2014",
                "chemistry_representation": "volume_mixing_ratio",
                "chemistry_species": list(chemistry_names),
                "products": [
                    "diagnostic_spectra.npz",
                    "diagnostic_tp.npz",
                    "diagnostic_chemistry.npz",
                ],
                "mpi_ranks": 12,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "posterior_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run["run_id"],
                "credible_intervals": intervals,
                "best_fit_parameters": best,
                "log_evidence": result.log_evidence,
                "log_evidence_error": result.log_evidence_error,
                "converged": bool(result.converged),
                "posterior_spectral_envelope": {
                    "sample_count": int(samples.shape[0]),
                    "sample_selection": "all_saved_weighted_posterior_samples",
                    "quantiles": list(POSTERIOR_ENVELOPE_QUANTILES),
                    "native_forward_framework": run["retriever"],
                },
                "posterior_temperature_envelope": {
                    "sample_count": int(samples.shape[0]),
                    "sample_selection": "all_saved_weighted_posterior_samples",
                    "quantiles": list(POSTERIOR_ENVELOPE_QUANTILES),
                    "temperature_parameterization": "ParmentierGuillot2014",
                },
                "posterior_chemistry_envelope": {
                    "sample_count": int(samples.shape[0]),
                    "sample_selection": "all_saved_weighted_posterior_samples",
                    "quantiles": list(POSTERIOR_ENVELOPE_QUANTILES),
                    "species": list(chemistry_names),
                    "representation": "volume_mixing_ratio",
                    "vertically_constant": True,
                },
                "fit_metrics": {
                    "n_data": observation.n_points,
                    "n_parameters": len(names),
                    "degrees_of_freedom": degrees_of_freedom,
                    "best_fit_chi_square": float(
                        np.sum((best_residual / observation.uncertainty) ** 2)
                    ),
                    "best_fit_reduced_chi_square": float(
                        np.sum((best_residual / observation.uncertainty) ** 2)
                        / degrees_of_freedom
                    ),
                    "best_fit_residual_rms_ppm": float(
                        np.sqrt(np.mean(best_residual**2)) * 1.0e6
                    ),
                    "posterior_median_residual_rms_ppm": float(
                        np.sqrt(np.mean(median_residual**2)) * 1.0e6
                    ),
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        arrays_path,
        samples=samples,
        log_likelihood=loglike,
        weights=weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", type=Path)
    parser.add_argument(
        "--pilot-output",
        type=Path,
        help="separate non-production output directory for an approved bounded resume pilot",
    )
    parser.add_argument("--pilot-max-iter", type=int, default=200)
    parser.add_argument("--pilot-live-points", type=int, default=50)
    args = parser.parse_args()
    if os.environ.get("STAGE9_CLUSTER") != "glamdring":
        raise RuntimeError("Stage-9 retrieval execution is restricted to Glamdring")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for the frozen MPI layout")

    run = json.loads(args.run_config.expanduser().resolve().read_text(encoding="utf-8"))
    _validate_run(run)
    pilot_output = None
    if args.pilot_output is not None:
        if args.pilot_max_iter <= 0 or args.pilot_live_points <= 0:
            parser.error("pilot iteration and live-point limits must be positive")
        pilot_output = args.pilot_output.expanduser().resolve()
        production_root = Path(run["project_root"]) / "runs"
        if pilot_output == production_root or production_root in pilot_output.parents:
            parser.error("pilot output must be outside the production runs tree")
    rank, communicator = _mpi()
    if communicator is None or communicator.Get_size() != 12:
        raise RuntimeError(
            "all 12 MPI ranks must join before a Stage-9 retrieval starts"
        )
    common = load_common_contract(run["common_contract"])
    observation = _load_observation(run)
    forward = build_native_forward(run["retriever"], common, run["scenario"])
    parameters = RetrievalParameterSet(
        tuple(
            RetrievalParameter(
                item.name,
                UniformPrior(item.lower, item.upper),
                label=item.label,
                unit=item.unit,
            )
            for item in parameter_definitions(run["scenario"])
        )
    )
    problem = RetrievalProblem(
        name=(f"pilot__{run['run_id']}" if pilot_output is not None else run["run_id"]),
        observation=observation,
        parameters=parameters,
        forward_model=forward.spectrum,
        invalid_loglike=-1.0e100,
        metadata={
            "stage": "9",
            "track": "track_b_native_retrieval",
            "injector": run["injector"],
            "retriever": run["retriever"],
            "scenario": run["scenario"],
            "noise_ppm": str(run["noise_ppm"]),
            "noise_id": run["noise_id"],
            "normalization_parameters_retrieved": "none",
            "native_binning_method": forward.native_binning_method,
        },
        opacity_identifiers={
            "retriever_native_opacity": run["retriever"],
            "shared_tensor": "none",
        },
    )
    settings = dict(MULTINEST_SETTINGS)
    settings.pop("engine", None)
    if pilot_output is not None:
        settings["max_iter"] = args.pilot_max_iter
        settings["n_live_points"] = args.pilot_live_points
    result = run_retrieval(
        problem,
        method="multinest",
        output_dir=(pilot_output if pilot_output is not None else run["run_directory"]),
        seed=int(run["sampler_seed"]),
        **settings,
    )
    communicator.Barrier()
    posterior_spectra = None
    posterior_temperatures = None
    if pilot_output is None:
        samples = _load_saved_posterior_samples(
            run["run_directory"],
            len(result.parameter_names),
        )
        posterior_spectra = _distributed_posterior_spectra(
            tuple(result.parameter_names),
            samples,
            forward,
            communicator,
        )
        posterior_temperatures = _distributed_posterior_temperatures(
            tuple(result.parameter_names),
            samples,
            common,
            run["scenario"],
            communicator,
        )
    communicator.Barrier()
    if rank == 0 and pilot_output is None:
        if posterior_spectra is None or posterior_temperatures is None:
            raise RuntimeError("primary rank did not receive posterior diagnostics")
        _compact_success_products(
            run,
            result,
            forward,
            posterior_spectra,
            posterior_temperatures,
            common,
        )
    elif rank == 0:
        (pilot_output / "PILOT_ONLY").write_text(
            "This bounded product is an infrastructure/resume pilot and is not a Stage-9 science retrieval.\n",
            encoding="utf-8",
        )
    communicator.Barrier()


if __name__ == "__main__":
    main()
