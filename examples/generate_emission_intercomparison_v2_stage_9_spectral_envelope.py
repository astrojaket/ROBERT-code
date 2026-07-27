#!/usr/bin/env python3
"""Backfill exact weighted posterior spectral quantiles for one Stage-9 run.

This is a science post-processing workload for Glamdring. It evaluates the
retriever-native forward model for every saved posterior sample, distributes
those evaluations over the frozen 12 MPI ranks, and stores only the wavelength-
wise weighted q16/q50/q84 spectra.
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
    build_native_forward,
    load_common_contract,
)
from run_emission_intercomparison_v2_stage_9_retrieval import (  # noqa: E402
    _distributed_posterior_spectra,
    _mpi,
    _validate_run,
    _weighted_spectral_quantiles,
)


QUANTILE_KEYS = (
    "posterior_spectrum_q16_eclipse_depth",
    "posterior_spectrum_q50_eclipse_depth",
    "posterior_spectrum_q84_eclipse_depth",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", type=Path)
    args = parser.parse_args()
    if os.environ.get("STAGE9_CLUSTER") != "glamdring":
        raise RuntimeError("Stage-9 spectral backfill is restricted to Glamdring")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for the frozen MPI layout")

    run = json.loads(args.run_config.expanduser().resolve().read_text(encoding="utf-8"))
    _validate_run(run)
    rank, communicator = _mpi()
    if communicator is None or communicator.Get_size() != 12:
        raise RuntimeError("spectral-envelope backfill requires exactly 12 MPI ranks")

    run_directory = Path(run["run_directory"])
    arrays_path = run_directory / "result_arrays.npz"
    result_path = run_directory / "result.json"
    spectra_path = run_directory / "diagnostic_spectra.npz"
    missing = [
        str(path)
        for path in (arrays_path, result_path, spectra_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "spectral-envelope backfill requires a completed retrieval; missing: "
            + ", ".join(missing)
        )
    with np.load(spectra_path, allow_pickle=False) as archive:
        already_complete = all(name in archive.files for name in QUANTILE_KEYS)
    already_complete = bool(communicator.bcast(already_complete, root=0))
    if already_complete:
        if rank == 0:
            print(f"{spectra_path} already contains posterior spectral quantiles")
        return

    result = json.loads(result_path.read_text(encoding="utf-8"))
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
    forward = build_native_forward(run["retriever"], common, run["scenario"])
    posterior_spectra = _distributed_posterior_spectra(
        parameter_names,
        samples,
        forward,
        communicator,
    )
    communicator.Barrier()
    if rank != 0:
        return
    if posterior_spectra is None:
        raise RuntimeError("primary rank did not receive posterior spectra")
    q16, q50, q84 = _weighted_spectral_quantiles(posterior_spectra, weights)

    with np.load(spectra_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    wavelength = np.asarray(payload["wavelength_micron"], dtype=float)
    if q16.shape != wavelength.shape:
        raise RuntimeError("posterior spectral quantiles do not match saved wavelength")
    payload.update(
        {
            QUANTILE_KEYS[0]: q16,
            QUANTILE_KEYS[1]: q50,
            QUANTILE_KEYS[2]: q84,
            "posterior_spectral_sample_count": np.asarray(
                samples.shape[0], dtype=np.int64
            ),
        }
    )
    temporary = spectra_path.with_name(f".{spectra_path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(spectra_path)
    (run_directory / "spectral_envelope_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run["run_id"],
                "sample_count": int(samples.shape[0]),
                "sample_selection": "all_saved_weighted_posterior_samples",
                "quantiles": [0.16, 0.50, 0.84],
                "forward_framework": run["retriever"],
                "mpi_ranks": 12,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(spectra_path)


if __name__ == "__main__":
    main()
