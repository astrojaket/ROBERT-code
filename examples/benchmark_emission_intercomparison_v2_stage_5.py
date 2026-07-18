"""Run Version-2 Stage 5 localized temperature-response comparisons."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
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
DEFAULT_OUTPUT = REPOSITORY / "examples/outputs/emission_intercomparison/version_2/stage_5"
WORKER = Path(__file__).with_name("run_emission_intercomparison_v2_stage_5_external.py")
STAGE_3_PATH = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_3.py")
STAGE_4_PATH = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_4.py")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_3 = _load_module("emission_v2_stage_3", STAGE_3_PATH)
stage_4 = _load_module("emission_v2_stage_4", STAGE_4_PATH)

ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
MODELS = ("robert", "picaso", "petitradtrans")
TRACK_A_MODELS = ("robert", "petitradtrans")
CENTERS_BAR = np.geomspace(1.0e-4, 10.0, 6)
PRIMARY_AMPLITUDE_K = 10.0
LINEARITY_AMPLITUDES_K = (5.0, 10.0, 20.0)
LOCALIZATION_SIGMA_DEX = 0.35
PILOT_PROJECTION_MULTIPLIER = 12.0
BAND_WINDOWS_MICRON = stage_4.BAND_WINDOWS_MICRON

# Frozen here and in docs/review/55_emission_intercomparison_v2_stage_5.md
# before the complete matrix is inspected. Native Track-B cross-framework
# comparisons have no acceptance gate.
STAGE_5_ACCEPTANCE_GATES = {
    "track_a_primary_p95_abs_jacobian_difference_over_pair_peak": 0.05,
    "track_a_primary_rms_eclipse_jacobian_difference_ppm_per_k": 0.02,
    "track_a_primary_centroid_rms_difference_dex": 0.15,
    "track_a_primary_response_total_variation_p95": 0.08,
    "track_a_80_to_160_p95_abs_jacobian_difference_over_pair_peak": 0.05,
    "track_a_80_to_160_rms_eclipse_jacobian_difference_ppm_per_k": 0.02,
    "track_a_80_to_160_centroid_rms_difference_dex": 0.15,
    "track_a_80_to_160_response_total_variation_p95": 0.08,
    "track_a_isothermal_baseline_max_abs_eclipse_difference_ppm": 0.1,
    "finite_difference_linearity_p95_relative": 0.02,
    "finite_difference_symmetry_p95_relative": 0.02,
    "exact_zero_normalization_max_abs": 0.0,
    "pilot_projected_wall_time_max_s": 7200.0,
    "pilot_peak_rss_fraction_of_available_max": 0.60,
}


def temperature_localization(
    pressure_bar: np.ndarray,
    center_bar: float,
    *,
    sigma_dex: float = LOCALIZATION_SIGMA_DEX,
) -> np.ndarray:
    """Evaluate the frozen unit-peak Gaussian localization in log pressure."""

    pressure = np.asarray(pressure_bar, dtype=float)
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("pressure_bar must contain finite positive values")
    if not np.isfinite(center_bar) or center_bar <= 0.0:
        raise ValueError("center_bar must be finite and positive")
    if not np.isfinite(sigma_dex) or sigma_dex <= 0.0:
        raise ValueError("sigma_dex must be finite and positive")
    return np.exp(-0.5 * (np.log10(pressure / center_bar) / sigma_dex) ** 2)


def normalize_absolute_response(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize absolute centre responses, retaining exact-zero columns."""

    response = np.abs(np.asarray(values, dtype=float))
    total = np.sum(response, axis=-2, keepdims=True)
    normalized = np.divide(response, total, out=np.zeros_like(response), where=total != 0.0)
    return normalized, np.squeeze(total == 0.0, axis=-2)


def build_stage_5_contract(
    common: Version2CommonContract,
    n_cells: int,
    *,
    profiles: tuple[str, ...] = PROFILES,
    amplitudes_k: tuple[float, ...] = (PRIMARY_AMPLITUDE_K,),
) -> dict[str, np.ndarray]:
    """Build baseline and symmetric localized-temperature cases."""

    baseline = stage_4.build_stage_4_contract(common, n_cells)
    selected_profiles = [PROFILES.index(profile) for profile in profiles]
    case_ids: list[str] = []
    profile_names: list[str] = []
    profile_indices: list[int] = []
    center_indices: list[int] = []
    signs: list[int] = []
    amplitudes: list[float] = []
    cells: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    for global_index in selected_profiles:
        profile = PROFILES[global_index]
        case_ids.append(f"{profile}_baseline_{n_cells}_cells")
        profile_names.append(profile)
        profile_indices.append(global_index)
        center_indices.append(-1)
        signs.append(0)
        amplitudes.append(0.0)
        cells.append(baseline["temperature_cells_k"][global_index])
        edges.append(baseline["temperature_edges_k"][global_index])
    for global_index in selected_profiles:
        profile = PROFILES[global_index]
        baseline_cells = baseline["temperature_cells_k"][global_index]
        baseline_edges = baseline["temperature_edges_k"][global_index]
        for amplitude in amplitudes_k:
            if not np.isfinite(amplitude) or amplitude <= 0.0:
                raise ValueError("amplitudes_k must contain finite positive values")
            for center_index, center in enumerate(CENTERS_BAR):
                cell_shape = temperature_localization(baseline["pressure_centers_bar"], center)
                edge_shape = temperature_localization(baseline["pressure_edges_bar"], center)
                for sign, label in ((-1, "minus"), (1, "plus")):
                    case_ids.append(
                        f"{profile}_p{center:.0e}_{label}_{amplitude:g}k_{n_cells}_cells"
                    )
                    profile_names.append(profile)
                    profile_indices.append(global_index)
                    center_indices.append(center_index)
                    signs.append(sign)
                    amplitudes.append(float(amplitude))
                    cells.append(baseline_cells + sign * amplitude * cell_shape)
                    edges.append(baseline_edges + sign * amplitude * edge_shape)
    count = len(case_ids)
    composition = baseline["gas_vmr"][0]
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(5),
        "case_id": np.asarray(case_ids),
        "profile_name": np.asarray(profile_names),
        "profile_index": np.asarray(profile_indices),
        "perturbation_center_index": np.asarray(center_indices),
        "perturbation_sign": np.asarray(signs),
        "perturbation_amplitude_k": np.asarray(amplitudes),
        "perturbation_centers_bar": CENTERS_BAR,
        "localization_sigma_dex": np.array(LOCALIZATION_SIGMA_DEX),
        "factor_name": np.full(count, stage_4.FIXED_FACTOR),
        "include_h2_h2_cia": np.ones(count, dtype=bool),
        "include_h2_he_cia": np.ones(count, dtype=bool),
        "gas_name": baseline["gas_name"],
        "gas_mass_u": baseline["gas_mass_u"],
        "gas_vmr": np.broadcast_to(composition, (count, composition.size)).copy(),
        "mean_molecular_weight_u": np.full(count, baseline["mean_molecular_weight_u"][0]),
        "molecular_species_name": baseline["molecular_species_name"],
        "molecular_species_active": np.ones((count, 4), dtype=bool),
        "pressure_edges_bar": baseline["pressure_edges_bar"],
        "pressure_centers_bar": baseline["pressure_centers_bar"],
        "picaso_pressure_levels_bar": baseline["picaso_pressure_levels_bar"],
        "petitradtrans_pressure_nodes_bar": baseline["petitradtrans_pressure_nodes_bar"],
        "temperature_edges_k": np.asarray(edges),
        "temperature_cells_k": np.asarray(cells),
        "gravity_m_s2": baseline["gravity_m_s2"],
        "emission_mu": baseline["emission_mu"],
        "legendre_weights": baseline["legendre_weights"],
        "disk_weights": baseline["disk_weights"],
    }


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024)


def _run_robert_native(
    contract: dict[str, np.ndarray],
    paths: dict[str, Path],
    *,
    tensor_root: Path | None = None,
) -> dict[str, np.ndarray]:
    """Recompute ROBERT native opacity for every Stage-5 temperature state."""

    from robert_exoplanets import (
        AtmosphereState,
        CiaTable,
        CorrelatedKOpacityProvider,
        CorrelatedKTable,
        EvaluatedCorrelatedKOpacity,
        PreparedCorrelatedKOpacity,
        PressureGrid,
        SpectralGrid,
        assemble_gas_optical_depth,
        cia_optical_depth,
        gauss_legendre_disk_geometry,
        solve_emission,
    )

    pressure_grid = PressureGrid(
        edges=contract["pressure_edges_bar"],
        centers=contract["pressure_centers_bar"],
        unit="bar",
        name="version-2-stage-5",
    )
    tables = {
        species: CorrelatedKTable.from_petitradtrans_hdf(paths[species], species=species)
        for species in stage_3.MOLECULAR_SPECIES
    }
    first = tables["H2O"]
    mask = (first.wavelength_micron >= 0.3) & (first.wavelength_micron <= 12.1)
    wavelength = np.sort(first.wavelength_micron[mask])
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="stage5-pRT-R1000"
    )
    providers = {
        species: CorrelatedKOpacityProvider(
            {species: table},
            name=f"stage5-{species}",
            interpolation="log_pressure_temperature_log_k",
        )
        for species, table in tables.items()
    }
    prepared_by_species = {
        species: provider.prepare(spectral_grid, pressure_grid, species=(species,))
        for species, provider in providers.items()
    }
    prepared = PreparedCorrelatedKOpacity(
        provider_name="pRT-HDF-four-species",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=stage_3.MOLECULAR_SPECIES,
        g_samples=first.g_samples,
        g_weights=first.g_weights,
        cache_key=f"stage5-{pressure_grid.n_layers}",
        metadata={"interpolation": "log_pressure_temperature_log_k"},
    )
    cia_tables = {
        pair: CiaTable.from_petitradtrans_hdf(paths[pair], collision_pair=pair)
        for pair in ("H2-H2", "H2-He")
    }
    names = [str(value) for value in contract["gas_name"]]
    count = contract["case_id"].size
    flux = np.empty((count, wavelength.size))
    contribution = np.empty((count, pressure_grid.n_layers, wavelength.size), dtype=np.float32)
    runtime = np.empty(count)
    if tensor_root is not None:
        tensor_root.mkdir(parents=True, exist_ok=True)
    for case_index in range(count):
        composition = dict(zip(names, contract["gas_vmr"][case_index], strict=True))
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=contract["temperature_cells_k"][case_index],
            temperature_edges=contract["temperature_edges_k"][case_index],
            composition={
                name: np.full(pressure_grid.n_layers, value)
                for name, value in composition.items()
            },
            mean_molecular_weight=np.full(
                pressure_grid.n_layers, contract["mean_molecular_weight_u"][case_index]
            ),
        )
        evaluated = np.empty(
            (4, pressure_grid.n_layers, wavelength.size, first.g_weights.size)
        )
        for species_index, species in enumerate(stage_3.MOLECULAR_SPECIES):
            evaluated[species_index] = (
                providers[species].evaluate(atmosphere, prepared_by_species[species]).kcoeff[0]
            )
        opacity = EvaluatedCorrelatedKOpacity(
            prepared=prepared,
            kcoeff=evaluated,
            unit="cm^2/molecule",
            metadata={"source": "petitRADTRANS HDF5 correlated-k tables"},
        )
        gas_tau = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=float(contract["gravity_m_s2"]),
            gas_combination="random_overlap",
        )
        cia = {
            pair: cia_optical_depth(
                gas_tau,
                table,
                coefficient_interpolation="log",
                temperature_extrapolation="clip",
                spectral_extrapolation="zero",
            )
            for pair, table in cia_tables.items()
        }
        started = perf_counter()
        result = solve_emission(
            gas_tau,
            geometry=gauss_legendre_disk_geometry(n_mu=8),
            bottom_boundary="blackbody",
            additional_optical_depths=list(cia.values()),
            multiple_scattering_backend="none",
        )
        runtime[case_index] = perf_counter() - started
        flux[case_index] = np.pi * np.asarray(result.radiance.values)
        vertical = np.asarray(result.layer_contribution_radiance, dtype=float).copy()
        vertical[-1] += np.asarray(result.bottom_contribution_radiance)
        contribution[case_index] = stage_3._normalise_vertical(vertical).astype(np.float32)
        if tensor_root is not None:
            np.savez_compressed(
                tensor_root / f"{case_index:03d}.npz",
                case_id=contract["case_id"][case_index],
                profile_name=contract["profile_name"][case_index],
                perturbation_center_index=contract["perturbation_center_index"][case_index],
                perturbation_sign=contract["perturbation_sign"][case_index],
                perturbation_amplitude_k=contract["perturbation_amplitude_k"][case_index],
                pressure_centers_bar=contract["pressure_centers_bar"],
                wavelength_micron=wavelength,
                g_weights=first.g_weights,
                molecular_layer_tau=gas_tau.total_tau.astype(np.float32),
                cia_h2_h2_layer_tau=cia["H2-H2"].tau.astype(np.float32),
                cia_h2_he_layer_tau=cia["H2-He"].tau.astype(np.float32),
            )
        del atmosphere, evaluated, opacity, gas_tau, cia, result, vertical
        gc.collect()
    metadata = {
        "model": "robert",
        "mode": "native_temperature_dependent_random_overlap_plus_both_cia",
        "python": os.path.realpath(sys.executable),
        "version": importlib.metadata.version("robert-exoplanets"),
        "peak_rss_bytes": _peak_rss_bytes(),
        "opacity_recomputed_for_every_temperature_state": True,
        "molecular_components": list(stage_3.MOLECULAR_SPECIES),
        "cia_components": ["H2-H2", "H2-He"],
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cloud_enabled": False,
    }
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "normalized_vertical_diagnostic": contribution,
        "runtime_s": runtime,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _shared_contract(
    contract: dict[str, np.ndarray], n_cells: int
) -> dict[str, np.ndarray]:
    with np.load(DATA_ROOT / f"stage_4_shared_tau_{n_cells}_cells.npz", allow_pickle=False) as a:
        tau = np.asarray(a["layer_tau"], dtype=float)
        wavelength = np.asarray(a["wavelength_micron"], dtype=float)
    return {
        **contract,
        "shared_wavelength_micron": wavelength,
        "shared_layer_tau": tau[contract["profile_index"]],
        "shared_source": np.array(
            "Stage-4 ROBERT shared mean molecular plus H2-H2/H2-He layer tau; frozen under Stage-5 source perturbations"
        ),
    }


def _extract_difference(
    flux: np.ndarray,
    contract: dict[str, np.ndarray],
    *,
    amplitude_k: float = PRIMARY_AMPLITUDE_K,
) -> tuple[np.ndarray, np.ndarray]:
    response = np.empty((len(PROFILES), CENTERS_BAR.size, flux.shape[-1]))
    for profile_index in range(len(PROFILES)):
        for center_index in range(CENTERS_BAR.size):
            common = (
                (contract["profile_index"] == profile_index)
                & (contract["perturbation_center_index"] == center_index)
                & (contract["perturbation_amplitude_k"] == amplitude_k)
            )
            minus = np.flatnonzero(common & (contract["perturbation_sign"] == -1))
            plus = np.flatnonzero(common & (contract["perturbation_sign"] == 1))
            if minus.size != 1 or plus.size != 1:
                raise ValueError("finite-difference cases are not one-to-one")
            response[profile_index, center_index] = 0.5 * (flux[plus[0]] - flux[minus[0]])
    return response, response / amplitude_k


def _pressure_diagnostics(response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized, _zero = normalize_absolute_response(response)
    log_pressure = np.log10(CENTERS_BAR)[:, None]
    centroid = np.sum(normalized * log_pressure[None, :, :], axis=1)
    peak = CENTERS_BAR[np.argmax(normalized, axis=1)]
    return centroid, peak


def _response_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_n, _ = normalize_absolute_response(left)
    right_n, _ = normalize_absolute_response(right)
    left_centroid, _ = _pressure_diagnostics(left)
    right_centroid, _ = _pressure_diagnostics(right)
    centroid = left_centroid - right_centroid
    total_variation = 0.5 * np.sum(np.abs(left_n - right_n), axis=1)
    return {
        "centroid_pressure_rms_difference_dex": float(np.sqrt(np.mean(centroid**2))),
        "centroid_pressure_p95_abs_difference_dex": float(np.percentile(np.abs(centroid), 95)),
        "profile_total_variation_median": float(np.median(total_variation)),
        "profile_total_variation_p95": float(np.percentile(total_variation, 95)),
    }


def _jacobian_metrics(
    left: np.ndarray,
    right: np.ndarray,
    common: Version2CommonContract,
) -> dict[str, float]:
    peak = np.maximum(np.max(np.abs(left), axis=-1), np.max(np.abs(right), axis=-1))[..., None]
    scaled = np.divide(np.abs(left - right), peak, out=np.zeros_like(left), where=peak != 0.0)
    eclipse = (
        (left - right)
        / common.stellar_surface_flux_r100_w_m2_m[None, None, :]
        * common.derived["projected_area_ratio"]
        * 1.0e6
    )
    return {
        "p95_abs_difference_over_pair_peak": float(np.percentile(scaled, 95)),
        "max_abs_difference_over_pair_peak": float(np.max(scaled)),
        "rms_eclipse_jacobian_difference_ppm_per_k": float(np.sqrt(np.mean(eclipse**2))),
        "max_abs_eclipse_jacobian_difference_ppm_per_k": float(np.max(np.abs(eclipse))),
    }


def _band_response_diagnostics(
    jacobian: np.ndarray, common: Version2CommonContract
) -> dict[str, Any]:
    """Summarize complete R=100 responses in predeclared spectral windows."""

    centroid, peak = _pressure_diagnostics(jacobian[None, :, :])
    eclipse = (
        jacobian
        / common.stellar_surface_flux_r100_w_m2_m[None, :]
        * common.derived["projected_area_ratio"]
        * 1.0e6
    )
    wavelength = common.spectral.r100_centers_micron
    output: dict[str, Any] = {}
    for name, (lower, upper) in BAND_WINDOWS_MICRON.items():
        selected = (wavelength >= lower) & (wavelength <= upper)
        output[name] = {
            "range_micron": [lower, upper],
            "bin_count": int(np.sum(selected)),
            "median_response_centroid_log10_bar": float(np.median(centroid[0, selected])),
            "median_response_peak_bar": float(np.median(peak[0, selected])),
            "rms_eclipse_jacobian_ppm_per_k": float(
                np.sqrt(np.mean(eclipse[:, selected] ** 2))
            ),
            "maximum_abs_eclipse_jacobian_ppm_per_k": float(
                np.max(np.abs(eclipse[:, selected]))
            ),
        }
    return output


def _artifact_arrays(
    contract: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    common: Version2CommonContract,
    *,
    representation: str,
) -> dict[str, np.ndarray]:
    r100_flux = stage_3._bin_flux(common, payload)
    native_response, native_jacobian = _extract_difference(payload["flux_w_m2_m"], contract)
    r100_response, r100_jacobian = _extract_difference(r100_flux, contract)
    native_normalized, native_zero = normalize_absolute_response(native_jacobian)
    r100_normalized, r100_zero = normalize_absolute_response(r100_jacobian)
    native_centroid, native_peak = _pressure_diagnostics(native_jacobian)
    r100_centroid, r100_peak = _pressure_diagnostics(r100_jacobian)
    native_stellar = planck_surface_flux_w_m2_m(
        payload["wavelength_micron"],
        common.measurements["stellar_effective_temperature"].si_value,
    )
    native_eclipse = (
        payload["flux_w_m2_m"] / native_stellar * common.derived["projected_area_ratio"]
    )
    r100_eclipse = (
        r100_flux
        / common.stellar_surface_flux_r100_w_m2_m
        * common.derived["projected_area_ratio"]
    )
    arrays = {
        name: contract[name]
        for name in (
            "case_id", "profile_name", "profile_index", "perturbation_center_index",
            "perturbation_sign", "perturbation_amplitude_k", "perturbation_centers_bar",
            "localization_sigma_dex", "gas_name", "gas_mass_u", "gas_vmr",
            "mean_molecular_weight_u", "molecular_species_name",
            "molecular_species_active", "include_h2_h2_cia", "include_h2_he_cia",
            "pressure_edges_bar", "pressure_centers_bar", "picaso_pressure_levels_bar",
            "petitradtrans_pressure_nodes_bar", "temperature_edges_k", "temperature_cells_k",
        )
    }
    arrays.update(
        {
            "native_wavelength_micron": payload["wavelength_micron"],
            "native_flux_w_m2_m": payload["flux_w_m2_m"],
            "native_eclipse_depth": native_eclipse,
            "r100_edges_micron": common.spectral.r100_edges_micron,
            "r100_centers_micron": common.spectral.r100_centers_micron,
            "r100_flux_w_m2_m": r100_flux,
            "r100_eclipse_depth": r100_eclipse,
            "signed_temperature_response_native_w_m2_m": native_response,
            "temperature_jacobian_native_w_m2_m_k": native_jacobian,
            "eclipse_jacobian_native_ppm_k": (
                native_jacobian / native_stellar[None, None, :]
                * common.derived["projected_area_ratio"] * 1.0e6
            ),
            "normalized_absolute_response_native": native_normalized,
            "zero_signal_mask_native": native_zero,
            "signed_temperature_response_r100_w_m2_m": r100_response,
            "temperature_jacobian_r100_w_m2_m_k": r100_jacobian,
            "eclipse_jacobian_r100_ppm_k": (
                r100_jacobian / common.stellar_surface_flux_r100_w_m2_m[None, None, :]
                * common.derived["projected_area_ratio"] * 1.0e6
            ),
            "normalized_absolute_response_r100": r100_normalized,
            "zero_signal_mask_r100": r100_zero,
            "response_centroid_native_log10_bar": native_centroid,
            "response_peak_native_bar": native_peak,
            "response_centroid_r100_log10_bar": r100_centroid,
            "response_peak_r100_bar": r100_peak,
            "localization_cells": np.stack(
                [temperature_localization(contract["pressure_centers_bar"], c) for c in CENTERS_BAR]
            ),
            "localization_edges": np.stack(
                [temperature_localization(contract["pressure_edges_bar"], c) for c in CENTERS_BAR]
            ),
            "runtime_s": payload["runtime_s"],
            "representation": np.array(representation),
            "metadata_json": payload["metadata_json"],
        }
    )
    return arrays


def _subset_cases(arrays: dict[str, np.ndarray], profile_index: int) -> dict[str, np.ndarray]:
    count = arrays["case_id"].size
    selected = arrays["profile_index"] == profile_index
    output: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if value.ndim > 0 and value.shape[0] == count:
            output[name] = value[selected]
        elif value.ndim > 0 and value.shape[0] == len(PROFILES) and name not in {
            "gas_name", "molecular_species_name"
        }:
            output[name] = value[profile_index : profile_index + 1]
        else:
            output[name] = value
    return output


def _project_stage_4_contribution(
    n_cells: int, model: str, profile_index: int
) -> np.ndarray:
    if model == "robert":
        path = DATA_ROOT / f"stage_4_robert_{PROFILES[profile_index]}_{n_cells}_cells.npz"
    else:
        path = DATA_ROOT / f"stage_4_{model}_{n_cells}_cells.npz"
    with np.load(path, allow_pickle=False) as archive:
        contribution = np.asarray(archive["normalized_vertical_r100"], dtype=float)
        if contribution.shape[0] > 1:
            contribution = contribution[profile_index : profile_index + 1]
        pressure = np.asarray(archive["pressure_centers_bar"], dtype=float)
    kernels = np.stack([temperature_localization(pressure, center) for center in CENTERS_BAR])
    projected, _ = normalize_absolute_response(kernels @ contribution[0])
    return projected


def _linearity_diagnostics(
    primary_contract: dict[str, np.ndarray],
    primary_payload: dict[str, np.ndarray],
    validation_contract: dict[str, np.ndarray],
    validation_payload: dict[str, np.ndarray],
    common: Version2CommonContract,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    primary_r100 = stage_3._bin_flux(common, primary_payload)
    validation_r100 = stage_3._bin_flux(common, validation_payload)
    global_profile = PROFILES.index("pg14_non_inverted")

    def derivative(amplitude: float) -> tuple[np.ndarray, np.ndarray]:
        source_contract = primary_contract if amplitude == 10.0 else validation_contract
        source_flux = primary_r100 if amplitude == 10.0 else validation_r100
        output = np.empty((CENTERS_BAR.size, source_flux.shape[-1]))
        even = np.empty_like(output)
        baseline_index = np.flatnonzero(
            (source_contract["profile_index"] == global_profile)
            & (source_contract["perturbation_sign"] == 0)
        )[0]
        baseline = source_flux[baseline_index]
        for center_index in range(CENTERS_BAR.size):
            selected = (
                (source_contract["profile_index"] == global_profile)
                & (source_contract["perturbation_center_index"] == center_index)
                & (source_contract["perturbation_amplitude_k"] == amplitude)
            )
            minus = np.flatnonzero(selected & (source_contract["perturbation_sign"] == -1))[0]
            plus = np.flatnonzero(selected & (source_contract["perturbation_sign"] == 1))[0]
            output[center_index] = (source_flux[plus] - source_flux[minus]) / (2 * amplitude)
            even[center_index] = source_flux[plus] + source_flux[minus] - 2 * baseline
        return output, even

    derivatives: dict[float, np.ndarray] = {}
    evens: dict[float, np.ndarray] = {}
    for amplitude in LINEARITY_AMPLITUDES_K:
        derivatives[amplitude], evens[amplitude] = derivative(amplitude)
    reference = derivatives[10.0]
    scale = np.max(np.abs(reference), axis=1)[:, None]
    linearity_values = np.concatenate(
        [
            np.divide(
                np.abs(derivatives[a] - reference),
                scale,
                out=np.zeros_like(reference),
                where=scale != 0.0,
            ).ravel()
            for a in (5.0, 20.0)
        ]
    )
    response_scale = 20.0 * scale
    symmetry_values = np.concatenate(
        [
            np.divide(
                np.abs(evens[a]),
                response_scale,
                out=np.zeros_like(reference),
                where=response_scale != 0.0,
            ).ravel()
            for a in LINEARITY_AMPLITUDES_K
        ]
    )
    return (
        {
            "linearity_p95_relative": float(np.percentile(linearity_values, 95)),
            "linearity_max_relative": float(np.max(linearity_values)),
            "symmetry_p95_relative": float(np.percentile(symmetry_values, 95)),
            "symmetry_max_relative": float(np.max(symmetry_values)),
        },
        {
            "amplitudes_k": np.asarray(LINEARITY_AMPLITUDES_K),
            "temperature_jacobian_r100_w_m2_m_k": np.stack(
                [derivatives[a] for a in LINEARITY_AMPLITUDES_K]
            ),
            "even_residual_r100_w_m2_m": np.stack([evens[a] for a in LINEARITY_AMPLITUDES_K]),
        },
    )


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
        "stage": 5,
        "units_and_axes": {
            "state_spectra": "case,wavelength; W m^-2 m^-1; signed positive outward",
            "temperature_response": "profile,pressure-centre,wavelength; W m^-2 m^-1",
            "temperature_jacobian": "profile,pressure-centre,wavelength; W m^-2 m^-1 K^-1",
            "eclipse_jacobian": "profile,pressure-centre,wavelength; ppm K^-1",
            "normalized_response": "profile,pressure-centre,wavelength; exact zero column or unit sum",
            "pressure": "bar; top-to-bottom increasing",
            "wavelength": "micron; increasing vacuum wavelength",
        },
        "artifacts": artifacts,
    }


def _run_pilot(
    common: Version2CommonContract, output_root: Path, paths: dict[str, Path]
) -> dict[str, Any]:
    pilot_root = output_root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    contract = build_stage_5_contract(
        common, 80, profiles=("pg14_non_inverted",), amplitudes_k=(10.0,)
    )
    contract_path = pilot_root / "contract.npz"
    np.savez_compressed(contract_path, **contract)
    stage_3.WORKER = WORKER
    started = perf_counter()
    robert = _run_robert_native(contract, paths)
    picaso = stage_3._run_external(
        PICASO_PYTHON, "picaso_ck", contract_path, pilot_root / "picaso.npz"
    )
    prt = stage_3._run_external(
        PRT_PYTHON,
        "petitradtrans_native",
        contract_path,
        pilot_root / "petitradtrans.npz",
    )
    measured = perf_counter() - started
    peak = max(
        int(json.loads(str(payload["metadata_json"]))["peak_rss_bytes"])
        for payload in (robert, picaso, prt)
    )
    available = stage_3._available_memory_bytes()
    projected = measured * PILOT_PROJECTION_MULTIPLIER
    fraction = peak / available
    authorized = (
        projected <= STAGE_5_ACCEPTANCE_GATES["pilot_projected_wall_time_max_s"]
        and fraction <= STAGE_5_ACCEPTANCE_GATES["pilot_peak_rss_fraction_of_available_max"]
    )
    result = {
        "resolution_cells": 80,
        "profile": "pg14_non_inverted",
        "case_count_per_framework": int(contract["case_id"].size),
        "frameworks": list(MODELS),
        "measured_wall_time_s": measured,
        "projection_multiplier": PILOT_PROJECTION_MULTIPLIER,
        "projected_complete_wall_time_s": projected,
        "largest_process_tree_member_peak_rss_bytes": peak,
        "available_memory_bytes_at_decision": available,
        "peak_rss_fraction_of_available": fraction,
        "authorized_full_matrix": authorized,
    }
    stage_3._write_json(pilot_root / "pilot_decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if os.path.realpath(sys.executable) != os.path.realpath(ROBERT_PYTHON):
        raise RuntimeError(f"Stage 5 must run with {ROBERT_PYTHON}")
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    common = load_version_2_common_contract(COMMON_CONTRACT)
    paths = stage_3._opacity_paths()
    for species, asset in common.picaso_correlated_k_assets.items():
        actual = stage_3._sha256(stage_3.PICASO_CK_DIRECTORY / asset.filename)
        if actual != asset.sha256:
            raise RuntimeError(f"frozen PICASO correlated-k checksum mismatch: {species}")
    pilot = _run_pilot(common, output_root, paths)
    if args.pilot_only:
        print(json.dumps(pilot, indent=2, sort_keys=True))
        return
    if not pilot["authorized_full_matrix"]:
        raise RuntimeError("Stage-5 matrix not authorized by frozen pilot resource gates")

    stage_3.WORKER = WORKER
    outputs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    artifacts: list[Path] = []
    matrix_started = perf_counter()
    for n_cells in RESOLUTIONS:
        resolution_root = output_root / f"{n_cells}_cells"
        resolution_root.mkdir(parents=True, exist_ok=True)
        contract = build_stage_5_contract(common, n_cells)
        contract_path = resolution_root / "contract.npz"
        np.savez_compressed(contract_path, **contract)
        tensor_root = resolution_root / "robert_tensors"
        robert = _run_robert_native(contract, paths, tensor_root=tensor_root)
        picaso = stage_3._run_external(
            PICASO_PYTHON, "picaso_ck", contract_path, resolution_root / "picaso.npz"
        )
        prt = stage_3._run_external(
            PRT_PYTHON,
            "petitradtrans_native",
            contract_path,
            resolution_root / "petitradtrans.npz",
        )
        shared_contract = _shared_contract(contract, n_cells)
        shared_path = resolution_root / "shared_contract.npz"
        np.savez_compressed(shared_path, **shared_contract)
        robert_shared = stage_3._run_robert_shared(shared_contract)
        prt_shared = stage_3._run_external(
            PRT_PYTHON,
            "petitradtrans_shared",
            shared_path,
            resolution_root / "petitradtrans_shared.npz",
        )
        outputs[n_cells] = {
            "contract": contract,
            "robert": robert,
            "picaso": picaso,
            "petitradtrans": prt,
            "robert_shared": robert_shared,
            "petitradtrans_shared": prt_shared,
        }
        representations = {
            "robert": "native_temperature_dependent_random_overlap_plus_both_cia",
            "picaso": "native_correlated_k_resort_rebin_absorbing_formal",
            "petitradtrans": "native_correlated_k_plus_both_cia",
            "robert_shared": "track_a_frozen_stage4_mean_tau_source_response",
            "petitradtrans_shared": "track_a_frozen_stage4_mean_tau_source_response",
        }
        for name in representations:
            arrays = _artifact_arrays(
                contract, outputs[n_cells][name], common, representation=representations[name]
            )
            if name in {"picaso", "petitradtrans", "robert_shared", "petitradtrans_shared"}:
                for profile_index, profile in enumerate(PROFILES):
                    selected_arrays = _subset_cases(arrays, profile_index)
                    if name == "picaso":
                        mask = contract["profile_index"] == profile_index
                        selected_arrays["native_total_taugas"] = picaso["layer_tau"][mask]
                        selected_arrays["native_framework_probe_flux_w_m2_m"] = picaso[
                            "native_framework_probe_flux_w_m2_m"
                        ][mask]
                        selected_arrays["maximum_abs_rayleigh_tau"] = picaso[
                            "maximum_abs_rayleigh_tau"
                        ]
                        selected_arrays["maximum_abs_cloud_tau"] = picaso[
                            "maximum_abs_cloud_tau"
                        ]
                    path = data_root / f"stage_5_{name}_{profile}_{n_cells}_cells.npz"
                    np.savez_compressed(path, **selected_arrays)
                    artifacts.append(path)
            else:
                path = data_root / f"stage_5_{name}_{n_cells}_cells.npz"
                np.savez_compressed(path, **arrays)
                artifacts.append(path)
        for raw_tensor in sorted(tensor_root.glob("*.npz")):
            with np.load(raw_tensor, allow_pickle=False) as archive:
                if int(archive["perturbation_sign"]) == 0:
                    continue
                case_id = str(archive["case_id"])
            target = data_root / f"stage_5_robert_opacity_{case_id}.npz"
            shutil.copyfile(raw_tensor, target)
            artifacts.append(target)
        gc.collect()
    matrix_wall = perf_counter() - matrix_started

    validation_contract = build_stage_5_contract(
        common,
        80,
        profiles=("pg14_non_inverted",),
        amplitudes_k=(5.0, 20.0),
    )
    validation_root = output_root / "linearity_80_cells"
    validation_root.mkdir(parents=True, exist_ok=True)
    validation_path = validation_root / "contract.npz"
    np.savez_compressed(validation_path, **validation_contract)
    validation_native = {
        "robert": _run_robert_native(validation_contract, paths),
        "picaso": stage_3._run_external(
            PICASO_PYTHON, "picaso_ck", validation_path, validation_root / "picaso.npz"
        ),
        "petitradtrans": stage_3._run_external(
            PRT_PYTHON,
            "petitradtrans_native",
            validation_path,
            validation_root / "petitradtrans.npz",
        ),
    }
    validation_shared_contract = _shared_contract(validation_contract, 80)
    validation_shared_path = validation_root / "shared_contract.npz"
    np.savez_compressed(validation_shared_path, **validation_shared_contract)
    validation_shared = {
        "robert": stage_3._run_robert_shared(validation_shared_contract),
        "petitradtrans": stage_3._run_external(
            PRT_PYTHON,
            "petitradtrans_shared",
            validation_shared_path,
            validation_root / "petitradtrans_shared.npz",
        ),
    }

    r100_jacobians: dict[int, dict[str, np.ndarray]] = {}
    r100_responses: dict[int, dict[str, np.ndarray]] = {}
    for n_cells in RESOLUTIONS:
        r100_jacobians[n_cells] = {}
        r100_responses[n_cells] = {}
        for name in ("robert", "picaso", "petitradtrans", "robert_shared", "petitradtrans_shared"):
            r100_flux = stage_3._bin_flux(common, outputs[n_cells][name])
            _signed, jacobian = _extract_difference(r100_flux, outputs[n_cells]["contract"])
            r100_jacobians[n_cells][name] = jacobian
            r100_responses[n_cells][name] = normalize_absolute_response(jacobian)[0]

    per_resolution: dict[str, Any] = {}
    track_a_primary_metrics: list[dict[str, float]] = []
    track_a_primary_response_metrics: list[dict[str, float]] = []
    contribution_relations: dict[str, Any] = {}
    for n_cells in RESOLUTIONS:
        profile_report: dict[str, Any] = {}
        for profile_index, profile in enumerate(PROFILES):
            track_a_j = _jacobian_metrics(
                r100_jacobians[n_cells]["robert_shared"][profile_index : profile_index + 1],
                r100_jacobians[n_cells]["petitradtrans_shared"][profile_index : profile_index + 1],
                common,
            )
            track_a_r = _response_metrics(
                r100_jacobians[n_cells]["robert_shared"][profile_index : profile_index + 1],
                r100_jacobians[n_cells]["petitradtrans_shared"][profile_index : profile_index + 1],
            )
            if n_cells == 80:
                track_a_primary_metrics.append(track_a_j)
                track_a_primary_response_metrics.append(track_a_r)
            native_pairs = {}
            for left, right in (("robert", "picaso"), ("robert", "petitradtrans"), ("picaso", "petitradtrans")):
                native_pairs[f"{left}__{right}"] = {
                    "jacobian": _jacobian_metrics(
                        r100_jacobians[n_cells][left][profile_index : profile_index + 1],
                        r100_jacobians[n_cells][right][profile_index : profile_index + 1],
                        common,
                    ),
                    "response": _response_metrics(
                        r100_jacobians[n_cells][left][profile_index : profile_index + 1],
                        r100_jacobians[n_cells][right][profile_index : profile_index + 1],
                    ),
                }
            relation = {}
            for model in MODELS:
                projected = _project_stage_4_contribution(n_cells, model, profile_index)
                relation[model] = _response_metrics(
                    r100_jacobians[n_cells][model][profile_index : profile_index + 1],
                    projected[None, :, :],
                )
            contribution_relations[f"{n_cells}_{profile}"] = relation
            profile_report[profile] = {
                "track_a_robert_vs_petitradtrans": {
                    "jacobian": track_a_j,
                    "response": track_a_r,
                },
                "track_b_native_attribution": native_pairs,
                "stage_4_contribution_relation": relation,
                "band_window_diagnostics": {
                    model: _band_response_diagnostics(
                        r100_jacobians[n_cells][model][profile_index], common
                    )
                    for model in MODELS
                },
            }
        per_resolution[str(n_cells)] = {"profiles": profile_report}

    convergence: dict[str, Any] = {}
    track_a_convergence_j: list[dict[str, float]] = []
    track_a_convergence_r: list[dict[str, float]] = []
    for model in ("robert", "picaso", "petitradtrans", "robert_shared", "petitradtrans_shared"):
        convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            metrics_j = _jacobian_metrics(
                r100_jacobians[coarse][model], r100_jacobians[fine][model], common
            )
            metrics_r = _response_metrics(
                r100_jacobians[coarse][model], r100_jacobians[fine][model]
            )
            convergence[model][f"{coarse}_to_{fine}"] = {
                "jacobian": metrics_j,
                "response": metrics_r,
            }
            if model in {"robert_shared", "petitradtrans_shared"} and coarse == 80:
                track_a_convergence_j.append(metrics_j)
                track_a_convergence_r.append(metrics_r)

    linearity: dict[str, Any] = {"track_a": {}, "track_b": {}}
    linearity_observed: list[dict[str, float]] = []
    for model in TRACK_A_MODELS:
        metrics, arrays = _linearity_diagnostics(
            outputs[80]["contract"],
            outputs[80][f"{model}_shared"],
            validation_shared_contract,
            validation_shared[model],
            common,
        )
        linearity["track_a"][model] = metrics
        linearity_observed.append(metrics)
        path = data_root / f"stage_5_linearity_track_a_{model}_80_cells.npz"
        np.savez_compressed(path, **arrays)
        artifacts.append(path)
    for model in MODELS:
        metrics, arrays = _linearity_diagnostics(
            outputs[80]["contract"],
            outputs[80][model],
            validation_contract,
            validation_native[model],
            common,
        )
        linearity["track_b"][model] = metrics
        path = data_root / f"stage_5_linearity_track_b_{model}_80_cells.npz"
        np.savez_compressed(path, **arrays)
        artifacts.append(path)

    baseline_indices = np.flatnonzero(
        (outputs[80]["contract"]["profile_index"] == 0)
        & (outputs[80]["contract"]["perturbation_sign"] == 0)
    )
    blackbody = flux_conserving_bin_mean(
        common.spectral.native_reference_wavelength_micron,
        planck_surface_flux_w_m2_m(
            common.spectral.native_reference_wavelength_micron,
            common.isothermal_temperature_k,
        ),
        common.spectral.r100_edges_micron,
    )
    isothermal_differences = []
    for model in ("robert_shared", "petitradtrans_shared"):
        flux = stage_3._bin_flux(common, outputs[80][model])[baseline_indices[0]]
        eclipse = (
            (flux - blackbody)
            / common.stellar_surface_flux_r100_w_m2_m
            * common.derived["projected_area_ratio"] * 1e6
        )
        isothermal_differences.append(float(np.max(np.abs(eclipse))))
    zero_values = []
    for model_values in r100_jacobians.values():
        for jacobian in model_values.values():
            normalized, zero = normalize_absolute_response(jacobian)
            if np.any(zero):
                zero_values.append(float(np.max(np.abs(normalized[..., zero]))))
    observed = {
        "track_a_primary_p95_abs_jacobian_difference_over_pair_peak": max(
            item["p95_abs_difference_over_pair_peak"] for item in track_a_primary_metrics
        ),
        "track_a_primary_rms_eclipse_jacobian_difference_ppm_per_k": max(
            item["rms_eclipse_jacobian_difference_ppm_per_k"] for item in track_a_primary_metrics
        ),
        "track_a_primary_centroid_rms_difference_dex": max(
            item["centroid_pressure_rms_difference_dex"] for item in track_a_primary_response_metrics
        ),
        "track_a_primary_response_total_variation_p95": max(
            item["profile_total_variation_p95"] for item in track_a_primary_response_metrics
        ),
        "track_a_80_to_160_p95_abs_jacobian_difference_over_pair_peak": max(
            item["p95_abs_difference_over_pair_peak"] for item in track_a_convergence_j
        ),
        "track_a_80_to_160_rms_eclipse_jacobian_difference_ppm_per_k": max(
            item["rms_eclipse_jacobian_difference_ppm_per_k"] for item in track_a_convergence_j
        ),
        "track_a_80_to_160_centroid_rms_difference_dex": max(
            item["centroid_pressure_rms_difference_dex"] for item in track_a_convergence_r
        ),
        "track_a_80_to_160_response_total_variation_p95": max(
            item["profile_total_variation_p95"] for item in track_a_convergence_r
        ),
        "track_a_isothermal_baseline_max_abs_eclipse_difference_ppm": max(isothermal_differences),
        "finite_difference_linearity_p95_relative": max(
            item["linearity_p95_relative"] for item in linearity_observed
        ),
        "finite_difference_symmetry_p95_relative": max(
            item["symmetry_p95_relative"] for item in linearity_observed
        ),
        "exact_zero_normalization_max_abs": max(zero_values, default=0.0),
    }
    gate_results = {
        name: observed[name] <= threshold
        for name, threshold in STAGE_5_ACCEPTANCE_GATES.items()
        if name in observed
    }
    report_path = data_root / "stage_5_report.json"
    report = {
        "schema_version": "2.0.0",
        "intercomparison_version": 2,
        "stage": 5,
        "status": "pass" if all(gate_results.values()) else "out_of_tolerance_characterized_regime",
        "scientific_framing": "Matched Track-A gates and native Track-B attribution; no framework is classified as failed.",
        "common_contract_sha256": common.to_dict()["contract_sha256"],
        "common_contract_file_sha256": stage_3._sha256(COMMON_CONTRACT),
        "predeclared_acceptance_gates": STAGE_5_ACCEPTANCE_GATES,
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "perturbation_contract": {
            "centers_bar": CENTERS_BAR.tolist(),
            "primary_amplitude_k": PRIMARY_AMPLITUDE_K,
            "linearity_amplitudes_k": list(LINEARITY_AMPLITUDES_K),
            "localization_sigma_dex": LOCALIZATION_SIGMA_DEX,
            "finite_difference": "symmetric centered difference",
            "edge_cell_mapping": "continuous kernel evaluated independently on frozen edges and geometric cell centres",
            "zero_signal_convention": "exact zero sum gives an exact-zero normalized response; no epsilon",
        },
        "track_a_scope": {
            "gated_frameworks": list(TRACK_A_MODELS),
            "opacity_convention": "completed Stage-4 shared mean molecular-plus-both-CIA optical depth frozen under localized source perturbations",
            "picaso": "no identical-tensor path or gate",
        },
        "track_b_scope": {
            "cross_framework_gates": None,
            "opacity_recomputed_for_every_temperature_state": True,
            "picaso": "PICASO 4.0 resort-rebin correlated-k with unchanged absolute line-VMR restoration",
            "petitradtrans_capability": "supported native flux and contribution; no fabricated native layer optical-depth tensor",
        },
        "pilot": pilot,
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "profiles": list(PROFILES),
        "per_resolution": per_resolution,
        "vertical_and_r100_spectral_convergence": convergence,
        "finite_difference_diagnostics": linearity,
        "stage_4_contribution_relations": contribution_relations,
        "timings": {"matrix_wall_time_s": matrix_wall},
        "interpreters": {
            "robert": str(ROBERT_PYTHON),
            "picaso": str(PICASO_PYTHON),
            "petitradtrans": str(PRT_PYTHON),
        },
        "package_versions": {
            model: json.loads(str(outputs[80][model]["metadata_json"]))["version"]
            for model in MODELS
        },
        "source_checksums": {
            path.name: stage_3._sha256(path)
            for path in (Path(__file__), WORKER)
        },
        "data_checksums": {
            "common_contract.json": stage_3._sha256(COMMON_CONTRACT),
            "stage_4_integrity.json": stage_3._sha256(DATA_ROOT / "stage_4_integrity.json"),
            **{
                asset.filename: asset.sha256
                for asset in common.picaso_correlated_k_assets.values()
            },
            **{path.name: stage_3._sha256(path) for path in paths.values()},
        },
        "artifact_sharding": {
            "robert_native_opacity": "one full-precision perturbed state per NPZ; baselines are the checksum-linked Stage-4 tensors",
            "picaso_native_taugas": "one profile per resolution",
            "petitradtrans_native_tau": "not exposed by the supported stable high-level interface",
            "precision_reduced": False,
            "native_tensors_dropped": False,
        },
        "known_warnings_and_capability_boundaries": {
            "picaso": [
                "optional Vega warning retained",
                "exact-zero cloud/Rayleigh warnings retained",
                "native exact-omega0=0 probe is separate capability evidence",
                "absorbing-formal vertical diagnostics are not native SH contributions",
            ],
            "petitradtrans": "stable native high-level flux interface exposes no stable layer optical-depth tensor",
        },
        "preserved_prior_stage_results": {
            "stage_1_eight_angle_maximum_ppm": 0.1968967046,
            "stage_2_track_a_maxima_ppm": [10.150094, 2.558145, 0.640615],
            "stage_3_track_a_maxima_ppm": [7.110565, 1.788100, 0.447547],
            "stage_4_status": "out_of_tolerance_closure_regime",
        },
        "random_seed": None,
        "random_seed_policy": common.random_seed_policy,
    }
    stage_3._write_json(report_path, report)
    artifacts.append(report_path)
    integrity_path = data_root / "stage_5_integrity.json"
    stage_3._write_json(integrity_path, _integrity_manifest(artifacts))
    artifacts.append(integrity_path)
    checksums_path = data_root / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    for name in tuple(checksums):
        if name.startswith("stage_5_"):
            checksums.pop(name)
    for path in artifacts:
        checksums[path.name] = stage_3._sha256(path)
    stage_3._write_json(checksums_path, checksums)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pilot": pilot,
                "observed_gate_values": observed,
                "artifact_count": len(artifacts),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
