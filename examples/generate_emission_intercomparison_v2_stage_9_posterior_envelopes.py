#!/usr/bin/env python3
"""Backfill exact weighted posterior spectral and TP quantiles for one run.

This is a science post-processing workload for Glamdring. It evaluates the
retriever-native forward model for every saved posterior sample, distributes
the spectral and TP evaluations over the frozen 12 MPI ranks, and stores only
the weighted q16/q50/q84 envelopes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_intercomparison_v2_stage_9_native import (  # noqa: E402
    atmospheric_state,
    build_native_forward,
    load_common_contract,
)
from run_emission_intercomparison_v2_stage_9_retrieval import (  # noqa: E402
    POSTERIOR_ENVELOPE_QUANTILES,
    _distributed_posterior_spectra,
    _distributed_posterior_temperatures,
    _load_observation,
    _mpi,
    _posterior_chemistry,
    _validate_run,
    _weighted_column_quantiles,
)


QUANTILE_KEYS = (
    "posterior_spectrum_q025_eclipse_depth",
    "posterior_spectrum_q16_eclipse_depth",
    "posterior_spectrum_q50_eclipse_depth",
    "posterior_spectrum_q84_eclipse_depth",
    "posterior_spectrum_q975_eclipse_depth",
)
TP_QUANTILE_KEYS = (
    "posterior_temperature_q025_k",
    "posterior_temperature_q16_k",
    "posterior_temperature_q50_k",
    "posterior_temperature_q84_k",
    "posterior_temperature_q975_k",
)
CHEMISTRY_QUANTILE_KEYS = (
    "posterior_vmr_q025",
    "posterior_vmr_q16",
    "posterior_vmr_q50",
    "posterior_vmr_q84",
    "posterior_vmr_q975",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", type=Path)
    args = parser.parse_args()
    if os.environ.get("STAGE9_CLUSTER") != "glamdring":
        raise RuntimeError("Stage-9 envelope backfill is restricted to Glamdring")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for the frozen MPI layout")

    run = json.loads(args.run_config.expanduser().resolve().read_text(encoding="utf-8"))
    _validate_run(run)
    rank, communicator = _mpi()
    if communicator is None or communicator.Get_size() != 12:
        raise RuntimeError("posterior-envelope backfill requires exactly 12 MPI ranks")

    run_directory = Path(run["run_directory"])
    arrays_path = run_directory / "result_arrays.npz"
    result_path = run_directory / "result.json"
    spectra_path = run_directory / "diagnostic_spectra.npz"
    tp_path = run_directory / "diagnostic_tp.npz"
    chemistry_path = run_directory / "diagnostic_chemistry.npz"
    missing = [
        str(path) for path in (arrays_path, result_path) if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "posterior-envelope backfill requires a completed retrieval; missing: "
            + ", ".join(missing)
        )
    spectral_complete = False
    if spectra_path.is_file():
        with np.load(spectra_path, allow_pickle=False) as archive:
            spectral_complete = all(
                name in archive.files
                for name in (
                    *QUANTILE_KEYS,
                    "observed_eclipse_depth",
                    "observational_uncertainty_eclipse_depth",
                )
            )
    tp_complete = False
    if tp_path.is_file():
        with np.load(tp_path, allow_pickle=False) as archive:
            tp_complete = all(name in archive.files for name in TP_QUANTILE_KEYS)
    chemistry_complete = False
    if chemistry_path.is_file():
        with np.load(chemistry_path, allow_pickle=False) as archive:
            chemistry_complete = all(
                name in archive.files for name in CHEMISTRY_QUANTILE_KEYS
            )
    complete = bool(
        communicator.bcast(
            spectral_complete and tp_complete and chemistry_complete,
            root=0,
        )
    )
    if complete:
        if rank == 0:
            print(f"{run_directory} already contains posterior envelopes")
        return

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_best = {
        str(name): float(value)
        for name, value in result["best_fit_parameters"].items()
    }
    parameter_names = tuple(str(name) for name in result["parameter_names"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        weights = (
            np.asarray(archive["weights"], dtype=float)
            if "weights" in archive.files
            else np.ones(samples.shape[0], dtype=float)
        )
    if samples.ndim != 2 or samples.shape[1] != len(parameter_names):
        raise RuntimeError("saved posterior dimensions do not match parameter names")
    if weights.shape != (samples.shape[0],) or np.sum(weights) <= 0.0:
        raise RuntimeError("saved posterior weights are invalid")
    weights = weights / np.sum(weights)

    common = load_common_contract(run["common_contract"])
    posterior_spectra = None
    if not spectral_complete:
        forward = build_native_forward(run["retriever"], common, run["scenario"])
        posterior_spectra = _distributed_posterior_spectra(
            parameter_names,
            samples,
            forward,
            communicator,
        )
    posterior_temperatures = None
    if not tp_complete:
        posterior_temperatures = _distributed_posterior_temperatures(
            parameter_names,
            samples,
            common,
            run["scenario"],
            communicator,
        )
    communicator.Barrier()
    if rank != 0:
        return

    if not spectral_complete:
        if posterior_spectra is None:
            raise RuntimeError("primary rank did not receive posterior spectra")
        q025, q16, q50, q84, q975 = _weighted_column_quantiles(
            posterior_spectra,
            weights,
        )
        payload = {}
        if spectra_path.is_file():
            with np.load(spectra_path, allow_pickle=False) as archive:
                payload = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                    if name
                    not in {
                        "injection_eclipse_depth",
                        "input_eclipse_depth",
                    }
                }
        observation = _load_observation(run)
        wavelength = np.asarray(observation.wavelength, dtype=float)
        if q16.shape != wavelength.shape:
            raise RuntimeError(
                "posterior spectral quantiles do not match saved wavelength"
            )
        payload.update(
            {
                "wavelength_micron": wavelength,
                "observed_eclipse_depth": np.asarray(
                    observation.flux, dtype=float
                ),
                "observational_uncertainty_eclipse_depth": np.asarray(
                    observation.uncertainty, dtype=float
                ),
                "best_fit_eclipse_depth": forward.eclipse_depth(result_best),
                QUANTILE_KEYS[0]: q025,
                QUANTILE_KEYS[1]: q16,
                QUANTILE_KEYS[2]: q50,
                QUANTILE_KEYS[3]: q84,
                QUANTILE_KEYS[4]: q975,
                "posterior_spectral_sample_count": np.asarray(
                    samples.shape[0], dtype=np.int64
                ),
            }
        )
        temporary = spectra_path.with_name(f".{spectra_path.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(spectra_path)

    if not tp_complete:
        if posterior_temperatures is None:
            raise RuntimeError("primary rank did not receive posterior TP profiles")
        tp_q025, tp_q16, tp_q50, tp_q84, tp_q975 = _weighted_column_quantiles(
            posterior_temperatures,
            weights,
        )
        pressure = np.asarray(
            next(
                item["centers_bar"]
                for item in common["pressure_grids"]
                if item["n_cells"] == 80
            ),
            dtype=float,
        )
        best_tp = atmospheric_state(
            common,
            run["scenario"],
            result_best,
        ).temperature_cells_k
        temporary = tp_path.with_name(f".{tp_path.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                pressure_bar=pressure,
                best_fit_temperature_k=best_tp,
                posterior_temperature_q025_k=tp_q025,
                posterior_temperature_q16_k=tp_q16,
                posterior_temperature_q50_k=tp_q50,
                posterior_temperature_q84_k=tp_q84,
                posterior_temperature_q975_k=tp_q975,
                posterior_temperature_sample_count=np.asarray(
                    samples.shape[0], dtype=np.int64
                ),
            )
        temporary.replace(tp_path)

    if not chemistry_complete:
        chemistry_names, posterior_chemistry = _posterior_chemistry(
            parameter_names,
            samples,
        )
        (
            chemistry_q025,
            chemistry_q16,
            chemistry_q50,
            chemistry_q84,
            chemistry_q975,
        ) = _weighted_column_quantiles(posterior_chemistry, weights)
        best_state = atmospheric_state(common, run["scenario"], result_best)
        if best_state.gas_names != chemistry_names:
            raise RuntimeError("diagnostic chemistry species order is inconsistent")
        temporary = chemistry_path.with_name(f".{chemistry_path.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
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
        temporary.replace(chemistry_path)

    (run_directory / "posterior_envelope_metadata.json").write_text(
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
                "chemistry_species": [
                    "H2",
                    "He",
                    "H2O",
                    "CO",
                    "CO2",
                    "CH4",
                ],
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
    print(spectra_path)
    print(tp_path)
    print(chemistry_path)


if __name__ == "__main__":
    main()
