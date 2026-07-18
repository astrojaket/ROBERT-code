"""Run one isolated Version-2 Stage-6 PICASO or stable-pRT worker."""

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
GAS_NAMES = ("H2", "He", *MOLECULAR_SPECIES)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _validate_contract(contract: dict[str, np.ndarray]) -> None:
    required = {
        "case_id", "profile_name", "profile_index", "target_species_name",
        "target_species_index", "perturbation_center_index", "perturbation_sign",
        "perturbation_amplitude_dex", "gas_name", "gas_mass_u", "gas_vmr_edges",
        "gas_vmr_cells", "mean_molecular_weight_edges", "mean_molecular_weight_cells",
        "pressure_edges_bar", "pressure_centers_bar", "temperature_edges_k",
        "temperature_cells_k", "gravity_m_s2", "emission_mu", "legendre_weights",
        "disk_weights", "include_h2_h2_cia", "include_h2_he_cia",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"Stage-6 contract is missing fields: {', '.join(missing)}")
    count = contract["case_id"].size
    n_cells = contract["pressure_centers_bar"].size
    if n_cells not in {40, 80, 160}:
        raise ValueError("Stage-6 pressure grid must have 40, 80, or 160 cells")
    if tuple(contract["gas_name"].tolist()) != GAS_NAMES:
        raise ValueError("Stage 6 requires exact H2, He, H2O, CO, CO2, CH4 order")
    if contract["gas_vmr_edges"].shape != (count, n_cells + 1, 6):
        raise ValueError("gas_vmr_edges has an unexpected shape")
    if contract["gas_vmr_cells"].shape != (count, n_cells, 6):
        raise ValueError("gas_vmr_cells has an unexpected shape")
    for name in ("gas_vmr_edges", "gas_vmr_cells"):
        values = contract[name]
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("every Stage-6 gas VMR must be finite and positive")
        if not np.allclose(np.sum(values, axis=-1), 1.0, rtol=0.0, atol=5e-16):
            raise ValueError("every Stage-6 six-gas composition must sum to one")
    if np.unique(contract["target_species_name"]).size != 1:
        raise ValueError("each Stage-6 shard must contain one target species")
    if not set(contract["profile_name"].tolist()) <= set(PROFILES):
        raise ValueError("Stage 6 accepts only frozen Version-2 profiles")
    if not np.all(contract["include_h2_h2_cia"]) or not np.all(
        contract["include_h2_he_cia"]
    ):
        raise ValueError("Stage 6 fixes both H2-H2 and H2-He CIA on")
    if not np.allclose(
        contract["mean_molecular_weight_cells"],
        np.sum(contract["gas_vmr_cells"] * contract["gas_mass_u"], axis=-1),
        rtol=2e-15,
        atol=2e-15,
    ):
        raise ValueError("Stage-6 cell mean molecular weight is inconsistent")


def _picaso(contract: dict[str, np.ndarray], ck_directory: Path) -> dict[str, np.ndarray]:
    stage_3_worker._validate_picaso_environment()
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi

    opacity = jdi.opannection(
        method="resortrebin",
        ck_db=str(ck_directory),
        preload_gases=list(MOLECULAR_SPECIES),
        wave_range=[0.3, 12.0],
        verbose=False,
    )
    stage_3_worker._restore_resort_rebin_absolute_vmr(opacity)
    output_flux: list[np.ndarray] = []
    output_probe: list[np.ndarray] = []
    output_tau: list[np.ndarray] = []
    output_vertical: list[np.ndarray] = []
    timings: list[float] = []
    correction_sums: list[np.ndarray] = []
    maximum_rayleigh_tau = 0.0
    maximum_cloud_tau = 0.0
    wavelength = np.empty(0)
    base_get_opacities = opacity.get_opacities
    for case_index in range(contract["case_id"].size):
        observed_pairs: set[str] = set()
        opacity.get_opacities = base_get_opacities
        stage_3_worker._filter_picaso_continuum(
            opacity, ("H2-H2", "H2-He"), observed_pairs
        )
        profile: dict[str, Any] = {
            "pressure": contract["pressure_edges_bar"],
            "temperature": contract["temperature_edges_k"][case_index],
        }
        for gas_index, gas_name in enumerate(GAS_NAMES):
            profile[gas_name] = contract["gas_vmr_edges"][case_index, :, gas_index]
        correction_sums.append(
            np.sum(contract["gas_vmr_edges"][case_index, :, 2:], axis=-1)
        )
        case = jdi.inputs(calculation="browndwarf")
        case.gravity(gravity=float(contract["gravity_m_s2"]), gravity_unit=u.m / u.s**2)
        case.atmosphere(df=pd.DataFrame(profile), verbose=False)
        case.approx(
            rt_method="SH", stream=4, delta_eddington=False, raman="none",
            w_single_rayleigh="off", w_multi_rayleigh="off", psingle_rayleigh="off",
        )
        started = perf_counter()
        result = case.spectrum(opacity, calculation="thermal", full_output=True)
        timings.append(perf_counter() - started)
        if observed_pairs != {"H2-H2", "H2-He"}:
            raise RuntimeError("PICASO did not retain both frozen CIA pairs")
        native_wavelength = 1.0e4 / np.asarray(result["wavenumber"], dtype=float)
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        output_probe.append(np.asarray(result["thermal"], dtype=float)[order] * 0.1)
        full = result["full_output"]
        tau = np.asarray(full["taugas"], dtype=float)[:, order]
        rayleigh = np.asarray(full["tauray"], dtype=float)
        cloud = np.asarray(full["taucld"], dtype=float)
        maximum_rayleigh_tau = max(maximum_rayleigh_tau, float(np.max(np.abs(rayleigh))))
        maximum_cloud_tau = max(maximum_cloud_tau, float(np.max(np.abs(cloud))))
        output_tau.append(tau.astype(np.float32))
        layers = stage_3_worker._formal_contribution(
            wavelength,
            contract["temperature_edges_k"][case_index],
            tau,
            np.asarray(opacity.gauss_wts, dtype=float),
            contract["emission_mu"],
            contract["disk_weights"],
            normalize=False,
        )
        output_flux.append(np.pi * np.sum(layers, axis=0))
        output_vertical.append(stage_3_worker._normalize(layers).astype(np.float32))
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(output_flux),
        "native_framework_probe_flux_w_m2_m": np.asarray(output_probe),
        "layer_tau": np.asarray(output_tau),
        "normalized_vertical_diagnostic": np.asarray(output_vertical),
        "runtime_s": np.asarray(timings),
        "g_weights": np.asarray(opacity.gauss_wts, dtype=float),
        "state_dependent_absolute_line_vmr_sum_edges": np.asarray(correction_sums),
        "maximum_abs_rayleigh_tau": np.array(maximum_rayleigh_tau),
        "maximum_abs_cloud_tau": np.array(maximum_cloud_tau),
    }


def _petitradtrans_native(
    contract: dict[str, np.ndarray], input_data: Path
) -> dict[str, np.ndarray]:
    from petitRADTRANS.radtrans import Radtrans

    atmosphere = Radtrans(
        pressures=contract["pressure_centers_bar"],
        wavelength_boundaries=np.array([0.3, 12.1]),
        line_species=list(stage_3_worker.PRT_LINE_SPECIES.values()),
        gas_continuum_contributors=list(stage_3_worker.PRT_CIA_SPECIES.values()),
        scattering_in_emission=False,
        emission_angle_grid=np.vstack((contract["emission_mu"], contract["legendre_weights"])),
        path_input_data=str(input_data),
    )
    flux: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    timings: list[float] = []
    wavelength = np.empty(0)
    masses = contract["gas_mass_u"]
    for case_index in range(contract["case_id"].size):
        vmr = contract["gas_vmr_cells"][case_index]
        mmw = contract["mean_molecular_weight_cells"][case_index]
        fractions = vmr * masses[None, :] / mmw[:, None]
        mass_fractions = {
            line: fractions[:, GAS_NAMES.index(molecule)]
            for molecule, line in stage_3_worker.PRT_LINE_SPECIES.items()
        }
        mass_fractions["H2"] = fractions[:, 0]
        mass_fractions["He"] = fractions[:, 1]
        started = perf_counter()
        result = atmosphere.calculate_flux(
            temperatures=contract["temperature_cells_k"][case_index],
            mass_fractions=mass_fractions,
            mean_molar_masses=mmw,
            reference_gravity=float(contract["gravity_m_s2"]) * 100.0,
            frequencies_to_wavelengths=True,
            return_contribution=True,
        )
        timings.append(perf_counter() - started)
        native_wavelength = np.asarray(result[0], dtype=float) * 1.0e4
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        flux.append(np.asarray(result[1], dtype=float)[order] * 0.1)
        contribution = np.asarray(result[2]["emission_contribution"], dtype=float)
        if contribution.shape[0] != contract["pressure_centers_bar"].size:
            contribution = contribution.T
        vertical.append(stage_3_worker._normalize(contribution[:, order]).astype(np.float32))
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "normalized_vertical_diagnostic": np.asarray(vertical),
        "runtime_s": np.asarray(timings),
    }


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
        output = _picaso(contract, args.picaso_ck_directory)
        package = "picaso"
        limitations = [
            "Native total taugas is retained; separate molecular/CIA components are not exposed.",
            "Exact-omega0=0 native thermal probes are pathological capability evidence, separate from the absorbing-formal spectrum.",
            "Pressure diagnostics are absorbing-formal diagnostics, not native SH contribution functions.",
        ]
    elif args.mode == "petitradtrans_native":
        if args.prt_input_data is None:
            parser.error("petitradtrans_native requires --prt-input-data")
        output = _petitradtrans_native(contract, args.prt_input_data)
        package = "petitRADTRANS"
        limitations = [
            "Stable pRT does not expose a native layer optical-depth tensor through the supported high-level flux interface."
        ]
    else:
        output = stage_3_worker._petitradtrans_shared(contract)
        package = "petitRADTRANS"
        limitations = []
    metadata: dict[str, Any] = {
        "mode": args.mode,
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
        "composition_and_mmw_state_dependent": True,
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cloud_enabled": False,
        "limitations": limitations,
        "known_warnings": (
            [
                "Optional Vega spectrum absent; Version 2 uses an explicit blackbody star.",
                "Exact-zero cloud/Rayleigh arrays can emit harmless invalid-divide warnings; zeros are retained.",
            ]
            if package == "picaso" else []
        ),
    }
    if package == "picaso":
        metadata["picaso_environment"] = stage_3_worker._validate_picaso_environment()
        metadata["absolute_line_vmr_restored_after_resort_rebin"] = True
        metadata["absolute_line_vmr_correction_state_dependent"] = True
        metadata["absolute_line_vmr_correction_algorithm"] = (
            "multiply normalized resort-rebin molecular opacity by actual layer summed H2O+CO+CO2+CH4 VMR"
        )
    output["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()
