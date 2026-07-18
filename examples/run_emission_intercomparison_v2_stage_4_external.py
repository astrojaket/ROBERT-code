"""Run one isolated Version-2 Stage-4 PICASO or stable-pRT worker."""

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
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import-system guard
    raise RuntimeError(f"cannot load {STAGE_3_WORKER_PATH}")
stage_3_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_3_worker)

PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")
FIXED_CIA_PAIRS = ("H2-H2", "H2-He")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _validate_contract(contract: dict[str, np.ndarray]) -> None:
    required = {
        "case_id",
        "profile_name",
        "gas_name",
        "gas_mass_u",
        "gas_vmr",
        "mean_molecular_weight_u",
        "molecular_species_name",
        "molecular_species_active",
        "pressure_edges_bar",
        "pressure_centers_bar",
        "temperature_edges_k",
        "temperature_cells_k",
        "gravity_m_s2",
        "emission_mu",
        "legendre_weights",
        "disk_weights",
        "include_h2_h2_cia",
        "include_h2_he_cia",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"Stage-4 contract is missing fields: {', '.join(missing)}")
    case_count = contract["case_id"].size
    n_cells = contract["pressure_centers_bar"].size
    profile_names = contract["profile_name"].tolist()
    expected_order = [name for name in PROFILES if name in profile_names]
    if profile_names != expected_order or len(set(profile_names)) != case_count:
        raise ValueError("Stage 4 requires a canonical ordered subset of frozen profiles")
    if n_cells not in {40, 80, 160}:
        raise ValueError("Stage-4 pressure grid must have 40, 80, or 160 cells")
    if contract["pressure_edges_bar"].shape != (n_cells + 1,):
        raise ValueError("pressure edges must have one more entry than pressure cells")
    if contract["temperature_edges_k"].shape != (case_count, n_cells + 1):
        raise ValueError("temperature_edges_k has an unexpected shape")
    if contract["temperature_cells_k"].shape != (case_count, n_cells):
        raise ValueError("temperature_cells_k has an unexpected shape")
    gas_names = [str(value) for value in contract["gas_name"]]
    if set(gas_names) != {"H2", "He", *MOLECULAR_SPECIES}:
        raise ValueError("Stage 4 requires exactly H2, He, and four absorbers")
    if contract["gas_vmr"].shape != (case_count, len(gas_names)):
        raise ValueError("gas_vmr must have shape case by six gases")
    if not np.all(contract["gas_vmr"] > 0.0):
        raise ValueError("all six frozen gases must be present in every case")
    if not np.all(contract["gas_vmr"] == contract["gas_vmr"][[0]]):
        raise ValueError("Stage-4 cases must use one identical frozen composition")
    if not np.allclose(np.sum(contract["gas_vmr"], axis=1), 1.0, rtol=0.0, atol=5e-16):
        raise ValueError("each frozen Stage-4 composition must sum to one")
    if tuple(contract["molecular_species_name"].tolist()) != MOLECULAR_SPECIES:
        raise ValueError("Stage 4 requires the exact four molecular absorbers")
    if not np.all(contract["molecular_species_active"]):
        raise ValueError("all four molecular absorbers must remain active")
    if not np.all(contract["include_h2_h2_cia"]):
        raise ValueError("Stage 4 fixes H2-H2 CIA on")
    if not np.all(contract["include_h2_he_cia"]):
        raise ValueError("Stage 4 fixes H2-He CIA on")


def _expected_python(mode: str) -> Path:
    return PICASO_PYTHON if mode == "picaso_ck" else PRT_PYTHON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("picaso_ck", "petitradtrans_native", "petitradtrans_shared"),
    )
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--picaso-ck-directory", type=Path)
    parser.add_argument("--prt-input-data", type=Path)
    args = parser.parse_args()
    expected_python = _expected_python(args.mode)
    if os.path.realpath(sys.executable) != os.path.realpath(expected_python):
        raise RuntimeError(f"{args.mode} must run with {expected_python}")
    contract = _load(args.contract)
    _validate_contract(contract)
    started = perf_counter()
    if args.mode == "picaso_ck":
        if args.picaso_ck_directory is None:
            parser.error("picaso_ck requires --picaso-ck-directory")
        output = stage_3_worker._picaso(contract, ck_directory=args.picaso_ck_directory)
        package = "picaso"
        representation = "primary_correlated_k_resort_rebin"
        limitations = [
            "Native taugas is retained but separate molecular/CIA component tensors are not exposed by the supported high-level path.",
            "Exact-omega0=0 native thermal probes are capability evidence; the independently labelled absorbing-formal spectrum is the scientific comparison product.",
            "Pressure-resolved arrays are absorbing-formal diagnostics applied to native taugas, not native SH contribution functions.",
        ]
    elif args.mode == "petitradtrans_native":
        if args.prt_input_data is None:
            parser.error("petitradtrans_native requires --prt-input-data")
        output = stage_3_worker._petitradtrans_native(contract, args.prt_input_data)
        package = "petitRADTRANS"
        representation = "native_correlated_k_with_both_cia_pairs"
        limitations = [
            "Stable pRT does not expose a native layer optical-depth tensor through the supported high-level flux interface; spectra and native emission contributions are retained."
        ]
    else:
        output = stage_3_worker._petitradtrans_shared(contract)
        package = "petitRADTRANS"
        representation = "track_a_identical_mean_tau"
        limitations = []
    metadata: dict[str, Any] = {
        "mode": args.mode,
        "representation": representation,
        "python": os.path.realpath(sys.executable),
        "expected_python": str(expected_python),
        "interpreter_matches_expected": True,
        "package": package,
        "version": importlib.metadata.version(package),
        "platform": platform.platform(),
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": stage_3_worker._peak_rss_bytes(),
        "molecular_species_always_enabled": list(MOLECULAR_SPECIES),
        "fixed_cia_pairs": list(FIXED_CIA_PAIRS),
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cloud_enabled": False,
        "known_warnings": (
            [
                "Optional Vega spectrum absent; Version 2 uses an explicit blackbody star.",
                "Exact-zero cloud/Rayleigh arrays can emit harmless invalid-divide warnings; retained arrays verify both remain zero.",
            ]
            if package == "picaso"
            else []
        ),
        "limitations": limitations,
    }
    if package == "picaso":
        metadata["picaso_environment"] = stage_3_worker._validate_picaso_environment()
        metadata["absolute_line_vmr_restored_after_resort_rebin"] = True
        gas_names = [str(value) for value in contract["gas_name"]]
        line_indices = [gas_names.index(name) for name in MOLECULAR_SPECIES]
        metadata["absolute_line_vmr_sum"] = float(
            np.sum(contract["gas_vmr"][0, line_indices])
        )
    output["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()
