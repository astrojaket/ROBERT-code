"""Run one isolated Version-2 Stage-5 PICASO or stable-pRT worker."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE_3_WORKER_PATH = HERE / "run_emission_intercomparison_v2_stage_3_external.py"
SPEC = importlib.util.spec_from_file_location("emission_v2_stage_3_worker", STAGE_3_WORKER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {STAGE_3_WORKER_PATH}")
stage_3_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_3_worker)

PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _validate_contract(contract: dict[str, np.ndarray]) -> None:
    required = {
        "case_id", "profile_name", "profile_index", "perturbation_center_index",
        "perturbation_sign", "perturbation_amplitude_k", "gas_name", "gas_mass_u",
        "gas_vmr", "mean_molecular_weight_u", "molecular_species_name",
        "molecular_species_active", "pressure_edges_bar", "pressure_centers_bar",
        "temperature_edges_k", "temperature_cells_k", "gravity_m_s2", "emission_mu",
        "legendre_weights", "disk_weights", "include_h2_h2_cia", "include_h2_he_cia",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"Stage-5 contract is missing fields: {', '.join(missing)}")
    count = contract["case_id"].size
    n_cells = contract["pressure_centers_bar"].size
    if n_cells not in {40, 80, 160}:
        raise ValueError("Stage-5 pressure grid must have 40, 80, or 160 cells")
    if contract["pressure_edges_bar"].shape != (n_cells + 1,):
        raise ValueError("pressure edges must have one more entry than pressure cells")
    if contract["temperature_edges_k"].shape != (count, n_cells + 1):
        raise ValueError("temperature_edges_k has an unexpected shape")
    if contract["temperature_cells_k"].shape != (count, n_cells):
        raise ValueError("temperature_cells_k has an unexpected shape")
    if not set(contract["profile_name"].tolist()) <= set(PROFILES):
        raise ValueError("Stage 5 accepts only the three frozen Version-2 profiles")
    gas_names = [str(value) for value in contract["gas_name"]]
    if set(gas_names) != {"H2", "He", *MOLECULAR_SPECIES}:
        raise ValueError("Stage 5 requires exactly H2, He, and four absorbers")
    if contract["gas_vmr"].shape != (count, 6):
        raise ValueError("gas_vmr must have shape case by six gases")
    if not np.all(contract["gas_vmr"] == contract["gas_vmr"][[0]]):
        raise ValueError("Stage-5 composition must remain identical in every case")
    if not np.allclose(np.sum(contract["gas_vmr"], axis=1), 1.0, rtol=0.0, atol=5e-16):
        raise ValueError("each Stage-5 composition must sum to one")
    if tuple(contract["molecular_species_name"].tolist()) != MOLECULAR_SPECIES:
        raise ValueError("Stage 5 requires the exact four molecular absorbers")
    if not np.all(contract["molecular_species_active"]):
        raise ValueError("all four molecular absorbers must remain active")
    if not np.all(contract["include_h2_h2_cia"]) or not np.all(
        contract["include_h2_he_cia"]
    ):
        raise ValueError("Stage 5 fixes both H2-H2 and H2-He CIA on")


def _expected_python(mode: str) -> Path:
    return PICASO_PYTHON if mode == "picaso_ck" else PRT_PYTHON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("picaso_ck", "petitradtrans_native", "petitradtrans_shared")
    )
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--picaso-ck-directory", type=Path)
    parser.add_argument("--prt-input-data", type=Path)
    args = parser.parse_args()
    expected = _expected_python(args.mode)
    if os.path.realpath(sys.executable) != os.path.realpath(expected):
        raise RuntimeError(f"{args.mode} must run with {expected}")
    contract = _load(args.contract)
    _validate_contract(contract)
    started = perf_counter()
    if args.mode == "picaso_ck":
        if args.picaso_ck_directory is None:
            parser.error("picaso_ck requires --picaso-ck-directory")
        output = stage_3_worker._picaso(contract, ck_directory=args.picaso_ck_directory)
        package = "picaso"
        representation = "native_correlated_k_resort_rebin_absorbing_formal"
        limitations = [
            "Native total taugas is retained; separate molecular/CIA components are not exposed.",
            "Exact-omega0=0 native thermal probes are pathological capability evidence, separate from the absorbing-formal scientific spectrum.",
            "Pressure-resolved arrays are absorbing-formal diagnostics, not native SH contribution functions.",
        ]
    elif args.mode == "petitradtrans_native":
        if args.prt_input_data is None:
            parser.error("petitradtrans_native requires --prt-input-data")
        output = stage_3_worker._petitradtrans_native(contract, args.prt_input_data)
        package = "petitRADTRANS"
        representation = "native_correlated_k_plus_both_cia_pairs"
        limitations = [
            "Stable pRT does not expose a native layer optical-depth tensor through the supported high-level flux interface."
        ]
    else:
        output = stage_3_worker._petitradtrans_shared(contract)
        package = "petitRADTRANS"
        representation = "track_a_identical_frozen_mean_tau_source_response"
        limitations = []
    metadata: dict[str, Any] = {
        "mode": args.mode,
        "representation": representation,
        "python": os.path.realpath(sys.executable),
        "expected_python": str(expected),
        "interpreter_matches_expected": True,
        "package": package,
        "version": importlib.metadata.version(package),
        "platform": platform.platform(),
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": stage_3_worker._peak_rss_bytes(),
        "molecular_species_always_enabled": list(MOLECULAR_SPECIES),
        "fixed_cia_pairs": ["H2-H2", "H2-He"],
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cloud_enabled": False,
        "limitations": limitations,
        "known_warnings": (
            [
                "Optional Vega spectrum absent; Version 2 uses an explicit blackbody star.",
                "Exact-zero cloud/Rayleigh arrays can emit harmless invalid-divide warnings; zeros are retained.",
            ]
            if package == "picaso"
            else []
        ),
    }
    if package == "picaso":
        metadata["picaso_environment"] = stage_3_worker._validate_picaso_environment()
        metadata["absolute_line_vmr_restored_after_resort_rebin"] = True
        names = [str(value) for value in contract["gas_name"]]
        metadata["absolute_line_vmr_sum"] = float(
            sum(contract["gas_vmr"][0, names.index(name)] for name in MOLECULAR_SPECIES)
        )
    output["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()
