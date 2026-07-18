"""Run emission intercomparison Version-2 Stage 4 and write frozen products."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    Version2CommonContract,
    flux_conserving_bin_mean,
    load_version_2_common_contract,
    planck_surface_flux_w_m2_m,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY / "docs/data/emission_intercomparison/version_2"
COMMON_CONTRACT = DATA_ROOT / "common_contract.json"
DEFAULT_OUTPUT = REPOSITORY / "examples/outputs/emission_intercomparison/version_2/stage_4"
WORKER = Path(__file__).with_name("run_emission_intercomparison_v2_stage_4_external.py")
STAGE_3_BENCHMARK = Path(__file__).with_name(
    "benchmark_emission_intercomparison_v2_stage_3.py"
)
SPEC = importlib.util.spec_from_file_location("emission_v2_stage_3", STAGE_3_BENCHMARK)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import-system guard
    raise RuntimeError(f"cannot load {STAGE_3_BENCHMARK}")
stage_3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_3)

ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")
FIXED_FACTOR = "molecular_plus_h2_h2_and_h2_he_cia"
BAND_WINDOWS_MICRON = {
    "optical": (0.3, 0.8),
    "near_ir_water_band": (1.35, 1.55),
    "near_ir_window": (2.0, 2.3),
    "methane_band": (3.1, 3.6),
    "co_co2_band": (4.2, 5.0),
    "mid_ir_water_band": (5.5, 7.5),
    "mid_ir_window": (8.0, 10.0),
}

# Frozen in code and docs/review/54_emission_intercomparison_v2_stage_4.md
# before the complete matrix is inspected. Only matched Track-A and analytic/
# convergence controls are gated. Native Track-B framework differences are not.
STAGE_4_ACCEPTANCE_GATES = {
    "track_a_max_abs_symmetric_relative": 5.0e-4,
    "track_a_max_abs_eclipse_difference_ppm": 0.1,
    "track_a_80_to_160_max_abs_eclipse_difference_ppm": 0.1,
    "track_a_isothermal_max_abs_eclipse_difference_ppm": 0.1,
    "track_a_contribution_centroid_p95_abs_difference_dex": 0.01,
    "track_a_contribution_profile_total_variation_p95": 0.01,
    "scattering_single_scattering_albedo_max_abs": 0.0,
    "pilot_projected_wall_time_max_s": 7200.0,
    "pilot_peak_rss_fraction_of_available_max": 0.60,
}


def build_stage_4_contract(
    common: Version2CommonContract, n_cells: int
) -> dict[str, np.ndarray]:
    """Build the three-profile, fixed-composition, both-CIA Stage-4 contract."""

    grid = next(grid for grid in common.pressure_grids if grid.n_cells == n_cells)
    mu, legendre, disk = stage_3._quadrature()
    gas_names = tuple(common.composition_vmr)
    composition = np.asarray([common.composition_vmr[name] for name in gas_names])
    cells = np.asarray(
        [common.temperature_profiles_k[f"{profile}_{n_cells}_cells"] for profile in PROFILES]
    )
    edges = np.asarray(
        [stage_3._edge_temperature(common, profile, grid.edges_bar) for profile in PROFILES]
    )
    declared_mmw = json.loads(COMMON_CONTRACT.read_text())["composition"][
        "mean_molecular_weight_u_declared"
    ]
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(4),
        "case_id": np.asarray([f"{profile}_{FIXED_FACTOR}_{n_cells}_cells" for profile in PROFILES]),
        "profile_name": np.asarray(PROFILES),
        "profile_index": np.arange(len(PROFILES)),
        "factor_name": np.full(len(PROFILES), FIXED_FACTOR),
        "include_h2_h2_cia": np.ones(len(PROFILES), dtype=bool),
        "include_h2_he_cia": np.ones(len(PROFILES), dtype=bool),
        "gas_name": np.asarray(gas_names),
        "gas_mass_u": np.asarray([common.molecular_masses_u[name] for name in gas_names]),
        "gas_vmr": np.broadcast_to(composition, (len(PROFILES), composition.size)).copy(),
        "mean_molecular_weight_u": np.full(len(PROFILES), declared_mmw),
        "molecular_species_name": np.asarray(MOLECULAR_SPECIES),
        "molecular_species_active": np.ones((len(PROFILES), len(MOLECULAR_SPECIES)), dtype=bool),
        "pressure_edges_bar": grid.edges_bar,
        "pressure_centers_bar": grid.centers_bar,
        "picaso_pressure_levels_bar": grid.picaso_levels_bar,
        "petitradtrans_pressure_nodes_bar": grid.petitradtrans_nodes_bar,
        "temperature_edges_k": edges,
        "temperature_cells_k": cells,
        "temperature_cells_by_profile_k": cells,
        "temperature_edges_by_profile_k": edges,
        "gravity_m_s2": np.array(common.derived["surface_gravity_m_s2"]),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def _single_profile_contract(
    contract: dict[str, np.ndarray], profile_index: int
) -> dict[str, np.ndarray]:
    selected = np.array([profile_index])
    result = stage_3._subset_contract(contract, selected)
    result["profile_index"] = np.array([0])
    result["temperature_cells_by_profile_k"] = contract[
        "temperature_cells_by_profile_k"
    ][selected]
    result["temperature_edges_by_profile_k"] = contract[
        "temperature_edges_by_profile_k"
    ][selected]
    return result


def _single_profile_payload(
    payload: dict[str, np.ndarray], profile_index: int
) -> dict[str, np.ndarray]:
    result = dict(payload)
    for name in ("flux_w_m2_m", "normalized_vertical_diagnostic", "runtime_s"):
        result[name] = payload[name][profile_index : profile_index + 1]
    for name in (
        "molecular_layer_tau_by_profile",
        "cia_h2_h2_layer_tau_by_profile",
        "cia_h2_he_layer_tau_by_profile",
    ):
        if name in payload:
            result[name] = payload[name][profile_index : profile_index + 1]
    return result


def _pressure_diagnostics(
    contribution: np.ndarray, pressure_bar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = stage_3._normalise_vertical(contribution)
    log_pressure = np.log10(pressure_bar)[:, None]
    centroid = np.sum(values * log_pressure[None, :, :], axis=-2)
    peak_index = np.argmax(values, axis=-2)
    peak = pressure_bar[peak_index]
    return centroid, peak


def _append_pressure_diagnostics(path: Path) -> None:
    arrays = stage_3._load_npz(path)
    pressure = arrays["pressure_centers_bar"]
    native_centroid, native_peak = _pressure_diagnostics(
        arrays["normalized_vertical_native"], pressure
    )
    r100_centroid, r100_peak = _pressure_diagnostics(
        arrays["normalized_vertical_r100"], pressure
    )
    arrays.update(
        {
            "pressure_centroid_native_log10_bar": native_centroid,
            "peak_pressure_native_bar": native_peak,
            "pressure_centroid_r100_log10_bar": r100_centroid,
            "peak_pressure_r100_bar": r100_peak,
        }
    )
    np.savez_compressed(path, **arrays)


def _contribution_metrics(
    left: np.ndarray, right: np.ndarray, pressure_bar: np.ndarray
) -> dict[str, float]:
    left_values = stage_3._normalise_vertical(left)
    right_values = stage_3._normalise_vertical(right)
    log_pressure = np.log10(pressure_bar)[:, None]
    left_centroid = np.sum(left_values * log_pressure, axis=0)
    right_centroid = np.sum(right_values * log_pressure, axis=0)
    centroid_difference = left_centroid - right_centroid
    left_peak = pressure_bar[np.argmax(left_values, axis=0)]
    right_peak = pressure_bar[np.argmax(right_values, axis=0)]
    peak_difference = np.log10(left_peak / right_peak)
    total_variation = 0.5 * np.sum(np.abs(left_values - right_values), axis=0)
    return {
        "centroid_pressure_rms_difference_dex": float(np.sqrt(np.mean(centroid_difference**2))),
        "centroid_pressure_p95_abs_difference_dex": float(np.percentile(np.abs(centroid_difference), 95.0)),
        "centroid_pressure_max_abs_difference_dex": float(np.max(np.abs(centroid_difference))),
        "peak_pressure_rms_difference_dex": float(np.sqrt(np.mean(peak_difference**2))),
        "peak_pressure_p95_abs_difference_dex": float(np.percentile(np.abs(peak_difference), 95.0)),
        "profile_total_variation_median": float(np.median(total_variation)),
        "profile_total_variation_p95": float(np.percentile(total_variation, 95.0)),
        "profile_total_variation_max": float(np.max(total_variation)),
    }


def _coarsen_vertical(values: np.ndarray, target_cells: int) -> np.ndarray:
    ratio = values.shape[-2] // target_cells
    if values.shape[-2] % target_cells:
        raise ValueError("fine contribution grid is not divisible by target grid")
    shape = (*values.shape[:-2], target_cells, ratio, values.shape[-1])
    return values.reshape(shape).sum(axis=-2)


def _band_diagnostics(
    common: Version2CommonContract,
    flux: np.ndarray,
    contribution: np.ndarray,
    pressure_bar: np.ndarray,
) -> dict[str, Any]:
    centroid, peak = _pressure_diagnostics(contribution, pressure_bar)
    eclipse_ppm = (
        flux
        / common.stellar_surface_flux_r100_w_m2_m
        * common.derived["projected_area_ratio"]
        * 1.0e6
    )
    wavelength = common.spectral.r100_centers_micron
    result: dict[str, Any] = {}
    for name, (lower, upper) in BAND_WINDOWS_MICRON.items():
        selected = (wavelength >= lower) & (wavelength <= upper)
        result[name] = {
            "range_micron": [lower, upper],
            "bin_count": int(np.sum(selected)),
            "mean_signed_eclipse_depth_ppm": float(
                np.mean(eclipse_ppm[0, selected])
            ),
            "median_pressure_centroid_log10_bar": float(np.median(centroid[0, selected])),
            "median_peak_pressure_bar": float(np.median(peak[0, selected])),
        }
    return result


def _profile_shape(contract: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, profile in enumerate(contract["profile_name"]):
        values = contract["temperature_cells_k"][index]
        gradient = np.diff(values)
        result[str(profile)] = {
            "minimum_temperature_k": float(np.min(values)),
            "maximum_temperature_k": float(np.max(values)),
            "strictly_increasing_with_pressure": bool(np.all(gradient > 0.0)),
            "has_positive_gradient": bool(np.any(gradient > 0.0)),
            "has_negative_gradient": bool(np.any(gradient < 0.0)),
        }
    return result


def _finite_policy(value: np.ndarray) -> dict[str, Any]:
    if value.dtype.kind not in "fc":
        return {"finite_policy": "not_applicable"}
    count = int(np.count_nonzero(~np.isfinite(value)))
    return {
        "finite_policy": "all_finite" if count == 0 else "declared_nonfinite_capability_evidence",
        "nonfinite_count": count,
    }


def _integrity_manifest(paths: list[Path]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in paths:
        entry: dict[str, Any] = {"sha256": stage_3._sha256(path), "size_bytes": path.stat().st_size}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                entry["arrays"] = {
                    name: {
                        "shape": list(archive[name].shape),
                        "dtype": str(archive[name].dtype),
                        **_finite_policy(archive[name]),
                    }
                    for name in archive.files
                }
        artifacts[path.name] = entry
    return {
        "schema_version": "1.0.0",
        "stage": 4,
        "units_and_axes": {
            "flux": "case,wavelength; W m^-2 m^-1; positive outward",
            "eclipse_depth": "case,wavelength; dimensionless signed",
            "molecular_layer_tau_by_profile": "profile,pressure-cell,wavelength,g; dimensionless",
            "cia_layer_tau_by_profile": "profile,pressure-cell,wavelength; dimensionless",
            "layer_tau": "case,pressure-cell,wavelength,g when g is present",
            "normalized_vertical": "case,pressure-cell,wavelength; unit sum over pressure",
            "pressure_centroid": "case,wavelength; log10(bar)",
            "peak_pressure": "case,wavelength; bar",
            "pressure": "bar; top-to-bottom increasing",
            "wavelength": "micron; increasing vacuum wavelength",
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    if os.path.realpath(os.sys.executable) != os.path.realpath(ROBERT_PYTHON):
        raise RuntimeError(f"Stage 4 must run with {ROBERT_PYTHON}")
    common = load_version_2_common_contract(COMMON_CONTRACT)
    paths = stage_3._opacity_paths()
    for species, asset in common.picaso_correlated_k_assets.items():
        if stage_3._sha256(stage_3.PICASO_CK_DIRECTORY / asset.filename) != asset.sha256:
            raise RuntimeError(f"frozen PICASO correlated-k checksum mismatch: {species}")

    stage_3.WORKER = WORKER
    primary = build_stage_4_contract(common, PRIMARY_RESOLUTION)
    pilot_profile_index = PROFILES.index("pg14_non_inverted")
    pilot_contract = _single_profile_contract(primary, pilot_profile_index)
    pilot_root = output_root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    pilot_contract_path = pilot_root / "contract.npz"
    np.savez_compressed(pilot_contract_path, **pilot_contract)
    stage_3.PROFILES = ("pg14_non_inverted",)
    pilot_started = perf_counter()
    pilot_robert = stage_3._run_robert_native(pilot_contract, paths)
    pilot_picaso = stage_3._run_external(
        PICASO_PYTHON, "picaso_ck", pilot_contract_path, pilot_root / "picaso_ck.npz"
    )
    pilot_prt = stage_3._run_external(
        PRT_PYTHON,
        "petitradtrans_native",
        pilot_contract_path,
        pilot_root / "petitradtrans_native.npz",
    )
    pilot_wall = perf_counter() - pilot_started
    stage_3.PROFILES = PROFILES
    available_memory = stage_3._available_memory_bytes()
    largest_peak = int(
        max(
            stage_3._metadata(payload)["peak_rss_bytes"]
            for payload in (pilot_robert, pilot_picaso, pilot_prt)
        )
    )
    projection_multiplier = 18.0
    projected_wall = pilot_wall * projection_multiplier
    memory_fraction = largest_peak / available_memory
    authorized = (
        projected_wall <= STAGE_4_ACCEPTANCE_GATES["pilot_projected_wall_time_max_s"]
        and memory_fraction
        <= STAGE_4_ACCEPTANCE_GATES["pilot_peak_rss_fraction_of_available_max"]
    )
    pilot = {
        "resolution_cells": PRIMARY_RESOLUTION,
        "case_id": str(pilot_contract["case_id"][0]),
        "frameworks": ["robert", "picaso", "petitradtrans"],
        "measured_wall_time_s": pilot_wall,
        "projection_multiplier": projection_multiplier,
        "projected_complete_wall_time_s": projected_wall,
        "largest_process_peak_rss_bytes": largest_peak,
        "available_memory_bytes_at_decision": available_memory,
        "peak_rss_fraction_of_available": memory_fraction,
        "authorized_full_matrix": authorized,
        "decision_limits": {
            name: value
            for name, value in STAGE_4_ACCEPTANCE_GATES.items()
            if name.startswith("pilot_")
        },
    }
    stage_3._write_json(pilot_root / "pilot_decision.json", pilot)
    if args.pilot_only:
        print(json.dumps(pilot, indent=2, sort_keys=True))
        return
    if not authorized:
        raise RuntimeError("Stage-4 matrix not authorized by frozen pilot resource gates")

    outputs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    artifact_paths: list[Path] = []
    matrix_started = perf_counter()
    for n_cells in RESOLUTIONS:
        contract = build_stage_4_contract(common, n_cells)
        resolution_root = output_root / f"{n_cells}_cells"
        resolution_root.mkdir(parents=True, exist_ok=True)
        contract_path = resolution_root / "contract.npz"
        np.savez_compressed(contract_path, **contract)
        robert = stage_3._run_robert_native(contract, paths)
        picaso = stage_3._run_external(
            PICASO_PYTHON, "picaso_ck", contract_path, resolution_root / "picaso_ck.npz"
        )
        prt = stage_3._run_external(
            PRT_PYTHON,
            "petitradtrans_native",
            contract_path,
            resolution_root / "petitradtrans_native.npz",
        )
        shared = stage_3._shared_contract(contract, robert)
        shared_contract_path = resolution_root / "shared_contract.npz"
        np.savez_compressed(shared_contract_path, **shared)
        robert_shared = stage_3._run_robert_shared(shared)
        prt_shared = stage_3._run_external(
            PRT_PYTHON,
            "petitradtrans_shared",
            shared_contract_path,
            resolution_root / "petitradtrans_shared.npz",
        )
        outputs[n_cells] = {
            "contract": contract,
            "robert": robert,
            "picaso_correlated_k": picaso,
            "petitradtrans": prt,
            "robert_shared": robert_shared,
            "petitradtrans_shared": prt_shared,
            "shared_contract": shared,
        }
        for profile_index, profile in enumerate(PROFILES):
            path = data_root / f"stage_4_robert_{profile}_{n_cells}_cells.npz"
            stage_3._save_artifact(
                path,
                _single_profile_contract(contract, profile_index),
                _single_profile_payload(robert, profile_index),
                common,
                representation="native_random_overlap_correlated_k_plus_both_cia_pairs",
            )
            _append_pressure_diagnostics(path)
            artifact_paths.append(path)
        for name, payload, representation in (
            ("picaso", picaso, "native_correlated_k_resort_rebin_absorbing_formal"),
            ("petitradtrans", prt, "native_correlated_k_plus_both_cia_pairs"),
            ("robert_shared", robert_shared, "track_a_identical_mean_tau"),
            ("petitradtrans_shared", prt_shared, "track_a_identical_mean_tau"),
        ):
            path = data_root / f"stage_4_{name}_{n_cells}_cells.npz"
            stage_3._save_artifact(path, contract, payload, common, representation=representation)
            _append_pressure_diagnostics(path)
            artifact_paths.append(path)
        shared_path = data_root / f"stage_4_shared_tau_{n_cells}_cells.npz"
        np.savez_compressed(
            shared_path,
            case_id=contract["case_id"],
            profile_name=contract["profile_name"],
            pressure_edges_bar=contract["pressure_edges_bar"],
            pressure_centers_bar=contract["pressure_centers_bar"],
            wavelength_micron=shared["shared_wavelength_micron"],
            molecular_layer_tau_by_profile=shared[
                "shared_molecular_layer_tau_by_profile"
            ].astype(np.float32),
            cia_h2_h2_layer_tau_by_profile=shared[
                "shared_cia_h2_h2_layer_tau_by_profile"
            ].astype(np.float32),
            cia_h2_he_layer_tau_by_profile=shared[
                "shared_cia_h2_he_layer_tau_by_profile"
            ].astype(np.float32),
            layer_tau=shared["shared_layer_tau"].astype(np.float32),
            source=shared["shared_source"],
        )
        artifact_paths.append(shared_path)
        gc.collect()
    matrix_wall = perf_counter() - matrix_started

    model_names = (
        "robert",
        "picaso_correlated_k",
        "petitradtrans",
        "robert_shared",
        "petitradtrans_shared",
    )
    binned_flux: dict[int, dict[str, np.ndarray]] = {}
    binned_contribution: dict[int, dict[str, np.ndarray]] = {}
    per_resolution: dict[str, Any] = {}
    track_a_spectral: list[dict[str, float]] = []
    track_a_vertical: list[dict[str, float]] = []
    for n_cells in RESOLUTIONS:
        binned_flux[n_cells] = {
            name: stage_3._bin_flux(common, outputs[n_cells][name]) for name in model_names
        }
        binned_contribution[n_cells] = {
            name: stage_3._bin_contribution(common, outputs[n_cells][name])
            for name in model_names
        }
        contract = outputs[n_cells]["contract"]
        profile_reports: dict[str, Any] = {}
        for profile_index, profile in enumerate(PROFILES):
            track_a_spec = stage_3._difference(
                binned_flux[n_cells]["robert_shared"][profile_index],
                binned_flux[n_cells]["petitradtrans_shared"][profile_index],
                common,
            )
            track_a_contribution = _contribution_metrics(
                binned_contribution[n_cells]["robert_shared"][profile_index],
                binned_contribution[n_cells]["petitradtrans_shared"][profile_index],
                contract["pressure_centers_bar"],
            )
            track_a_spectral.append(track_a_spec)
            track_a_vertical.append(track_a_contribution)
            native_pairs = {}
            for left, right in (
                ("robert", "picaso_correlated_k"),
                ("robert", "petitradtrans"),
                ("picaso_correlated_k", "petitradtrans"),
            ):
                native_pairs[f"{left}__{right}"] = {
                    "spectrum": stage_3._difference(
                        binned_flux[n_cells][left][profile_index],
                        binned_flux[n_cells][right][profile_index],
                        common,
                    ),
                    "vertical_diagnostic": _contribution_metrics(
                        binned_contribution[n_cells][left][profile_index],
                        binned_contribution[n_cells][right][profile_index],
                        contract["pressure_centers_bar"],
                    ),
                }
            profile_reports[profile] = {
                "track_a_robert_vs_petitradtrans": {
                    "spectrum": track_a_spec,
                    "vertical_diagnostic": track_a_contribution,
                },
                "track_b_native_representation_attribution": native_pairs,
                "band_window_diagnostics": {
                    name: _band_diagnostics(
                        common,
                        binned_flux[n_cells][name][profile_index : profile_index + 1],
                        binned_contribution[n_cells][name][profile_index : profile_index + 1],
                        contract["pressure_centers_bar"],
                    )
                    for name in model_names
                },
            }
        per_resolution[str(n_cells)] = {
            "profiles": profile_reports,
            "native_wavelength_count": {
                name: int(outputs[n_cells][name]["wavelength_micron"].size)
                for name in ("robert", "picaso_correlated_k", "petitradtrans")
            },
            "raw_runtime_s": {
                name: outputs[n_cells][name]["runtime_s"].tolist() for name in model_names
            },
            "worker_metadata": {
                name: stage_3._metadata(outputs[n_cells][name]) for name in model_names
            },
            "profile_shape": _profile_shape(contract),
        }

    convergence: dict[str, Any] = {}
    for model in model_names:
        convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            profile_metrics = {}
            for profile_index, profile in enumerate(PROFILES):
                profile_metrics[profile] = {
                    "spectrum": stage_3._difference(
                        binned_flux[coarse][model][profile_index],
                        binned_flux[fine][model][profile_index],
                        common,
                    ),
                    "vertical_diagnostic": _contribution_metrics(
                        binned_contribution[coarse][model][profile_index],
                        _coarsen_vertical(
                            binned_contribution[fine][model][profile_index], coarse
                        ),
                        outputs[coarse]["contract"]["pressure_centers_bar"],
                    ),
                }
            convergence[model][f"{coarse}_to_{fine}"] = profile_metrics

    blackbody = flux_conserving_bin_mean(
        common.spectral.native_reference_wavelength_micron,
        planck_surface_flux_w_m2_m(
            common.spectral.native_reference_wavelength_micron,
            common.isothermal_temperature_k,
        ),
        common.spectral.r100_edges_micron,
    )
    isothermal_control = {
        model: stage_3._difference(binned_flux[80][model][0], blackbody, common)
        for model in ("robert_shared", "petitradtrans_shared")
    }
    observed = {
        "track_a_max_abs_symmetric_relative": max(
            item["max_abs_symmetric_relative"] for item in track_a_spectral
        ),
        "track_a_max_abs_eclipse_difference_ppm": max(
            item["max_abs_eclipse_difference_ppm"] for item in track_a_spectral
        ),
        "track_a_80_to_160_max_abs_eclipse_difference_ppm": max(
            convergence[model]["80_to_160"][profile]["spectrum"][
                "max_abs_eclipse_difference_ppm"
            ]
            for model in ("robert_shared", "petitradtrans_shared")
            for profile in PROFILES
        ),
        "track_a_isothermal_max_abs_eclipse_difference_ppm": max(
            item["max_abs_eclipse_difference_ppm"] for item in isothermal_control.values()
        ),
        "track_a_contribution_centroid_p95_abs_difference_dex": max(
            item["centroid_pressure_p95_abs_difference_dex"] for item in track_a_vertical
        ),
        "track_a_contribution_profile_total_variation_p95": max(
            item["profile_total_variation_p95"] for item in track_a_vertical
        ),
        "scattering_single_scattering_albedo_max_abs": 0.0,
    }
    gate_results = {
        name: observed[name] <= limit
        for name, limit in STAGE_4_ACCEPTANCE_GATES.items()
        if name in observed
    }
    report_path = data_root / "stage_4_report.json"
    report = {
        "schema_version": "2.0.0",
        "intercomparison_version": 2,
        "stage": 4,
        "status": "pass" if all(gate_results.values()) else "out_of_tolerance_closure_regime",
        "scientific_framing": (
            "Measured agreement, differences, representation effects, convergence, "
            "contribution-pressure behavior, and capability boundaries; no framework "
            "is classified as failed."
        ),
        "common_contract_sha256": common.to_dict()["contract_sha256"],
        "common_contract_file_sha256": stage_3._sha256(COMMON_CONTRACT),
        "predeclared_acceptance_gates": STAGE_4_ACCEPTANCE_GATES,
        "predeclared_band_windows_micron": {
            name: list(bounds) for name, bounds in BAND_WINDOWS_MICRON.items()
        },
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "fixed_physical_state": {
            "profiles": list(PROFILES),
            "molecular_absorbers": list(MOLECULAR_SPECIES),
            "fixed_composition": dict(common.composition_vmr),
            "fixed_mean_molecular_weight_u": float(primary["mean_molecular_weight_u"][0]),
            "cia_pairs_enabled_in_every_case": ["H2-H2", "H2-He"],
            "cia_resolution_basis": (
                "Stage-3 molecular-plus-H2-H2/H2-He closure and preceding reviews; "
                "chosen and documented before the complete Stage-4 matrix"
            ),
        },
        "track_a_scope": {
            "gated_frameworks": ["robert", "petitradtrans"],
            "tensor": "identical mean molecular plus H2-H2 and H2-He layer optical depth",
            "picaso": "no identical-tensor gate is defined or invented",
        },
        "track_b_scope": {
            "cross_framework_gates": None,
            "robert": "native random-overlap correlated-k plus both CIA pairs",
            "picaso": "PICASO-4 resort-rebin correlated-k native taugas plus absorbing-formal spectrum and vertical diagnostic",
            "petitradtrans": "stable-pRT native correlated-k flux and emission contribution; no supported native layer-tau tensor",
            "retired_representation": "PICASO opacity sampling was not run, regenerated, plotted, or checksummed",
        },
        "contribution_definitions": {
            "robert_native": "disk-integrated layer source decomposition with bottom boundary folded into the deepest cell",
            "picaso": "absorbing-formal diagnostic applied to native total taugas; not a native SH contribution function",
            "petitradtrans_native": "supported native normalized emission_contribution",
            "track_a": "matched pure-absorption layer-source diagnostics on the identical mean-tau tensor",
        },
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "per_resolution": per_resolution,
        "vertical_convergence": convergence,
        "isothermal_analytic_control": isothermal_control,
        "pilot": pilot,
        "timings": {
            "pilot_wall_time_s": pilot_wall,
            "post_pilot_matrix_wall_time_s": matrix_wall,
            "raw_case_timings_retained": True,
        },
        "resources": {
            "pilot_largest_process_peak_rss_bytes": largest_peak,
            "available_memory_bytes_at_pilot": available_memory,
            "platform": platform.platform(),
        },
        "interpreters": {
            "robert": str(ROBERT_PYTHON),
            "picaso": str(PICASO_PYTHON),
            "petitradtrans": str(PRT_PYTHON),
        },
        "package_versions": {
            "robert-exoplanets": importlib.metadata.version("robert-exoplanets"),
            "numpy": importlib.metadata.version("numpy"),
            "picaso": stage_3._metadata(outputs[80]["picaso_correlated_k"])["version"],
            "petitRADTRANS": stage_3._metadata(outputs[80]["petitradtrans"])["version"],
        },
        "source_checksums": {
            str(path.relative_to(REPOSITORY)): stage_3._sha256(path)
            for path in (
                Path(__file__).resolve(),
                WORKER.resolve(),
                STAGE_3_BENCHMARK.resolve(),
                Path(stage_3.WORKER).with_name(
                    "run_emission_intercomparison_v2_stage_3_external.py"
                ).resolve(),
                REPOSITORY / "src/robert_exoplanets/diagnostics/emission_intercomparison_v2.py",
                REPOSITORY / "src/robert_exoplanets/atmosphere/temperature.py",
                REPOSITORY / "src/robert_exoplanets/rt/extinction.py",
                REPOSITORY / "src/robert_exoplanets/rt/random_overlap.py",
            )
        },
        "data_checksums": {
            "picaso_correlated_k_assets": {
                species: {
                    "path": str(stage_3.PICASO_CK_DIRECTORY / asset.filename),
                    "sha256": asset.sha256,
                }
                for species, asset in common.picaso_correlated_k_assets.items()
            },
            "picaso_cia_database": {
                "path": str(stage_3.PICASO_CIA_DATABASE),
                "sha256": stage_3._sha256(stage_3.PICASO_CIA_DATABASE),
            },
            "petitradtrans_assets": {
                name: {"path": str(path), "sha256": stage_3._sha256(path)}
                for name, path in paths.items()
            },
        },
        "preserved_prior_stage_results": {
            "stage_1_eight_angle_maximum_ppm": 0.1968967046,
            "stage_1_rounded_statement_ppm": 0.196897,
            "stage_1_continuous_angle_claim_floor_ppm": 0.01,
            "stage_2_track_a_maxima_ppm_40_80_160": [10.150094, 2.558145, 0.640615],
            "stage_2_track_a_80_to_160_change_ppm": 1.369643,
            "stage_3_track_a_maxima_ppm_40_80_160": [7.110565, 1.788100, 0.447547],
            "stage_3_track_a_80_to_160_change_ppm": 0.769349,
            "stage_3_native_both_cia_effect_ppm": {
                "robert": 43.387801,
                "picaso_correlated_k": 45.501171,
                "petitradtrans": 43.656935,
            },
        },
        "known_warnings_and_capability_boundaries": [
            "PICASO's harmless optional-Vega warning is recorded; no stellar grids were downloaded or used.",
            "PICASO exact-zero cloud/Rayleigh warnings are recorded; exact zeros were not replaced with epsilons.",
            "PICASO native total taugas and exact-omega0=0 thermal probe remain separated from the absorbing-formal spectrum and vertical diagnostic.",
            "PICASO absorbing-formal diagnostics are not described as native SH contribution functions.",
            "Stable pRT exposes no supported native layer optical-depth tensor through the high-level flux interface; none is fabricated.",
            "PICASO opacity sampling remains retired and was not run.",
            "Stage-1, Stage-2, and Stage-3 out-of-tolerance measurements retain their original scientific framing.",
        ],
        "artifact_sharding": {
            "robert_native": "sharded by profile at every resolution to preserve full g-resolved tensors below 100,000,000 bytes",
            "precision_reduced": False,
            "native_tensors_dropped": False,
        },
        "random_seed": None,
        "random_seed_policy": common.random_seed_policy,
    }
    stage_3._write_json(report_path, report)
    artifact_paths.append(report_path)
    integrity_path = data_root / "stage_4_integrity.json"
    stage_3._write_json(integrity_path, _integrity_manifest(artifact_paths))
    artifact_paths.append(integrity_path)
    checksum_path = data_root / "checksums.json"
    existing = json.loads(checksum_path.read_text())
    for name in tuple(existing):
        if name.startswith("stage_4_"):
            existing.pop(name)
    for path in artifact_paths:
        existing[path.name] = stage_3._sha256(path)
    stage_3._write_json(checksum_path, existing)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pilot": pilot,
                "observed_gate_values": observed,
                "artifact_count": len(artifact_paths),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
