"""Run Version-2 Stage 6 localized composition-response comparisons."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
from itertools import combinations
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
    load_version_2_common_contract,
    planck_surface_flux_w_m2_m,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY / "docs/data/emission_intercomparison/version_2"
COMMON_CONTRACT = DATA_ROOT / "common_contract.json"
DEFAULT_OUTPUT = REPOSITORY / "examples/outputs/emission_intercomparison/version_2/stage_6"
WORKER = Path(__file__).with_name("run_emission_intercomparison_v2_stage_6_external.py")
STAGE_3_PATH = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_3.py")
STAGE_4_PATH = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_4.py")
STAGE_5_PATH = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_5.py")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_3 = _load_module("emission_v2_stage_3", STAGE_3_PATH)
stage_4 = _load_module("emission_v2_stage_4", STAGE_4_PATH)
stage_5 = _load_module("emission_v2_stage_5", STAGE_5_PATH)

ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
PHYSICAL_PROFILES = ("pg14_non_inverted", "pg14_inverted")
ANALYTIC_CONTROL_PROFILE = "isothermal"
PROFILES = (ANALYTIC_CONTROL_PROFILE, *PHYSICAL_PROFILES)
SPECIES = ("H2O", "CO", "CO2", "CH4")
GAS_NAMES = ("H2", "He", *SPECIES)
MODELS = ("robert", "picaso", "petitradtrans")
TRACK_A_MODELS = ("robert", "petitradtrans")
CENTERS_BAR = np.geomspace(1.0e-4, 10.0, 6)
PRIMARY_AMPLITUDE_DEX = 0.10
LINEARITY_AMPLITUDES_DEX = (0.05, 0.10, 0.20)
LOCALIZATION_SIGMA_DEX = 0.35
ROUND_OFF_EPSILON_FACTOR = 32.0
PILOT_PROJECTION_MULTIPLIER = 48.0
BAND_WINDOWS_MICRON = stage_4.BAND_WINDOWS_MICRON

# Frozen here and in docs/review/56_emission_intercomparison_v2_stage_6.md
# before the representative pilot and complete matrix are inspected.
STAGE_6_ACCEPTANCE_GATES = {
    "track_a_primary_p95_abs_jacobian_difference_over_pair_peak": 0.05,
    "track_a_primary_rms_eclipse_jacobian_difference_ppm_per_dex": 0.50,
    "track_a_primary_centroid_rms_difference_dex": 0.15,
    "track_a_primary_response_total_variation_p95": 0.08,
    "track_a_primary_cross_species_fraction_total_variation_p95": 0.08,
    "track_a_80_to_160_p95_abs_jacobian_difference_over_pair_peak": 0.05,
    "track_a_80_to_160_rms_eclipse_jacobian_difference_ppm_per_dex": 0.50,
    "track_a_80_to_160_centroid_rms_difference_dex": 0.15,
    "track_a_80_to_160_response_total_variation_p95": 0.08,
    "track_a_80_to_160_cross_species_fraction_total_variation_p95": 0.08,
    "finite_difference_linearity_p95_relative": 0.02,
    "finite_difference_symmetry_p95_relative": 0.02,
    "analytic_isothermal_composition_jacobian_max_abs": 0.0,
    "exact_zero_normalization_and_fraction_max_abs": 0.0,
    "pilot_projected_wall_time_max_s": 7200.0,
    "pilot_peak_rss_fraction_of_available_max": 0.60,
}


def composition_localization(
    pressure_bar: np.ndarray,
    center_bar: float,
    *,
    sigma_dex: float = LOCALIZATION_SIGMA_DEX,
) -> np.ndarray:
    """Evaluate the frozen unit-peak Gaussian localization in log pressure."""

    return stage_5.temperature_localization(
        pressure_bar, center_bar, sigma_dex=sigma_dex
    )


def _complete_composition(
    common: Version2CommonContract, molecular_vmr: np.ndarray
) -> np.ndarray:
    molecular = np.asarray(molecular_vmr, dtype=float)
    if molecular.shape[-1:] != (len(SPECIES),):
        raise ValueError("molecular_vmr must end with the four-species axis")
    if np.any(~np.isfinite(molecular)) or np.any(molecular <= 0.0):
        raise ValueError("molecular VMRs must be finite and positive")
    remainder = 1.0 - np.sum(molecular, axis=-1)
    if np.any(remainder <= 0.0):
        raise ValueError("molecular VMR sum must remain below one")
    ratio_total = 0.8547 + 0.1453
    return np.concatenate(
        (
            (remainder * 0.8547 / ratio_total)[..., None],
            (remainder * 0.1453 / ratio_total)[..., None],
            molecular,
        ),
        axis=-1,
    )


def localized_composition(
    common: Version2CommonContract,
    pressure_bar: np.ndarray,
    target_species: str,
    center_bar: float,
    sign: int,
    amplitude_dex: float,
) -> np.ndarray:
    """Apply one localized log10-VMR perturbation and exact H2/He fill."""

    if target_species not in SPECIES:
        raise ValueError(f"unsupported target species: {target_species}")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or +1")
    if not np.isfinite(amplitude_dex) or amplitude_dex <= 0.0:
        raise ValueError("amplitude_dex must be finite and positive")
    pressure = np.asarray(pressure_bar, dtype=float)
    molecular = np.broadcast_to(
        np.asarray([common.composition_vmr[name] for name in SPECIES]),
        (pressure.size, len(SPECIES)),
    ).copy()
    target_index = SPECIES.index(target_species)
    localization = composition_localization(pressure, center_bar)
    molecular[:, target_index] *= 10.0 ** (sign * amplitude_dex * localization)
    return _complete_composition(common, molecular)


def _mean_molecular_weight(
    common: Version2CommonContract, composition: np.ndarray
) -> np.ndarray:
    masses = np.asarray([common.molecular_masses_u[name] for name in GAS_NAMES])
    return np.sum(np.asarray(composition) * masses, axis=-1)


def build_stage_6_contract(
    common: Version2CommonContract,
    n_cells: int,
    *,
    profile: str = "pg14_non_inverted",
    target_species: str = "H2O",
    amplitude_dex: float = PRIMARY_AMPLITUDE_DEX,
) -> dict[str, np.ndarray]:
    """Build one baseline-plus-twelve state composition shard."""

    if profile not in PROFILES:
        raise ValueError("profile must be a frozen Version-2 Stage-6 profile")
    if target_species not in SPECIES:
        raise ValueError("target_species must be H2O, CO, CO2, or CH4")
    grid = next(item for item in common.pressure_grids if item.n_cells == n_cells)
    pressure_edges = grid.edges_bar
    pressure_cells = grid.centers_bar
    temperature_cells = common.temperature_profiles_k[f"{profile}_{n_cells}_cells"]
    temperature_edges = stage_3._edge_temperature(common, profile, pressure_edges)
    baseline_molecular_edges = np.broadcast_to(
        np.asarray([common.composition_vmr[name] for name in SPECIES]),
        (pressure_edges.size, len(SPECIES)),
    )
    baseline_molecular_cells = np.broadcast_to(
        np.asarray([common.composition_vmr[name] for name in SPECIES]),
        (pressure_cells.size, len(SPECIES)),
    )
    baseline_edges = _complete_composition(common, baseline_molecular_edges)
    baseline_cells = _complete_composition(common, baseline_molecular_cells)
    cases = [f"{profile}_{target_species}_baseline_{n_cells}_cells"]
    center_indices = [-1]
    signs = [0]
    edge_compositions = [baseline_edges]
    cell_compositions = [baseline_cells]
    for center_index, center in enumerate(CENTERS_BAR):
        for sign, label in ((-1, "minus"), (1, "plus")):
            cases.append(
                f"{profile}_{target_species}_p{center:.0e}_{label}_{amplitude_dex:g}dex_{n_cells}_cells"
            )
            center_indices.append(center_index)
            signs.append(sign)
            edge_compositions.append(
                localized_composition(
                    common, pressure_edges, target_species, center, sign, amplitude_dex
                )
            )
            cell_compositions.append(
                localized_composition(
                    common, pressure_cells, target_species, center, sign, amplitude_dex
                )
            )
    gas_edges = np.asarray(edge_compositions)
    gas_cells = np.asarray(cell_compositions)
    count = len(cases)
    mu, legendre, disk = stage_3._quadrature()
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(6),
        "case_id": np.asarray(cases),
        "profile_name": np.full(count, profile),
        "profile_index": np.full(count, PROFILES.index(profile)),
        "target_species_name": np.full(count, target_species),
        "target_species_index": np.full(count, SPECIES.index(target_species)),
        "perturbation_center_index": np.asarray(center_indices),
        "perturbation_sign": np.asarray(signs),
        "perturbation_amplitude_dex": np.full(count, amplitude_dex),
        "perturbation_centers_bar": CENTERS_BAR,
        "localization_sigma_dex": np.array(LOCALIZATION_SIGMA_DEX),
        "gas_name": np.asarray(GAS_NAMES),
        "gas_mass_u": np.asarray([common.molecular_masses_u[name] for name in GAS_NAMES]),
        "gas_vmr_edges": gas_edges,
        "gas_vmr_cells": gas_cells,
        "mean_molecular_weight_edges": _mean_molecular_weight(common, gas_edges),
        "mean_molecular_weight_cells": _mean_molecular_weight(common, gas_cells),
        "summed_line_gas_vmr_edges": np.sum(gas_edges[..., 2:], axis=-1),
        "summed_line_gas_vmr_cells": np.sum(gas_cells[..., 2:], axis=-1),
        "include_h2_h2_cia": np.ones(count, dtype=bool),
        "include_h2_he_cia": np.ones(count, dtype=bool),
        "pressure_edges_bar": pressure_edges,
        "pressure_centers_bar": pressure_cells,
        "picaso_pressure_levels_bar": pressure_edges,
        "petitradtrans_pressure_nodes_bar": pressure_cells,
        "temperature_edges_k": np.broadcast_to(temperature_edges, (count, n_cells + 1)).copy(),
        "temperature_cells_k": np.broadcast_to(temperature_cells, (count, n_cells)).copy(),
        "gravity_m_s2": np.array(common.derived["surface_gravity_m_s2"]),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def normalize_absolute_response(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize absolute pressure responses and retain exact-zero columns."""

    response = np.abs(np.asarray(values, dtype=float))
    total = np.sum(response, axis=-2, keepdims=True)
    normalized = np.divide(response, total, out=np.zeros_like(response), where=total != 0.0)
    return normalized, np.squeeze(total == 0.0, axis=-2)


def cross_species_fractions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize pressure-integrated absolute sensitivity over target species."""

    sensitivity = np.sum(np.abs(np.asarray(values, dtype=float)), axis=-2)
    total = np.sum(sensitivity, axis=-2, keepdims=True)
    fractions = np.divide(
        sensitivity, total, out=np.zeros_like(sensitivity), where=total != 0.0
    )
    return fractions, np.squeeze(total == 0.0, axis=-2)


def apply_jacobian_roundoff_floor(
    values: np.ndarray, flux_scale: np.ndarray, amplitude_dex: float
) -> np.ndarray:
    """Classify centered differences indistinguishable from roundoff as zero."""

    jacobian = np.asarray(values, dtype=float)
    scale = np.asarray(flux_scale, dtype=float)
    floor = (
        ROUND_OFF_EPSILON_FACTOR
        * np.finfo(float).eps
        * scale
        / float(amplitude_dex)
    )
    return np.where(np.abs(jacobian) <= floor, 0.0, jacobian)


def _extract_difference(
    flux: np.ndarray, contract: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude = float(contract["perturbation_amplitude_dex"][0])
    baseline = np.asarray(flux[0])
    response = np.empty((CENTERS_BAR.size, flux.shape[-1]))
    jacobian = np.empty_like(response)
    even = np.empty_like(response)
    for center_index in range(CENTERS_BAR.size):
        selected = contract["perturbation_center_index"] == center_index
        minus = np.flatnonzero(selected & (contract["perturbation_sign"] == -1))[0]
        plus = np.flatnonzero(selected & (contract["perturbation_sign"] == 1))[0]
        response[center_index] = 0.5 * (flux[plus] - flux[minus])
        jacobian[center_index] = response[center_index] / amplitude
        even[center_index] = flux[plus] + flux[minus] - 2.0 * baseline
    if str(contract["profile_name"][0]) == ANALYTIC_CONTROL_PROFILE:
        response.fill(0.0)
        jacobian.fill(0.0)
        even.fill(0.0)
    else:
        jacobian = apply_jacobian_roundoff_floor(
            jacobian, np.max(np.abs(flux), axis=0), amplitude
        )
    return response, jacobian, even


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _run_robert_native(
    contract: dict[str, np.ndarray],
    paths: dict[str, Path],
    *,
    tensor_root: Path | None = None,
) -> dict[str, np.ndarray]:
    """Recompute ROBERT molecular, CIA, MMW, and RT state for every case."""

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
        name="version-2-stage-6",
    )
    tables = {
        species: CorrelatedKTable.from_petitradtrans_hdf(paths[species], species=species)
        for species in SPECIES
    }
    first = tables["H2O"]
    mask = (first.wavelength_micron >= 0.3) & (first.wavelength_micron <= 12.1)
    wavelength = np.sort(first.wavelength_micron[mask])
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="stage6-pRT-R1000"
    )
    providers = {
        species: CorrelatedKOpacityProvider(
            {species: table},
            name=f"stage6-{species}",
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
        species=SPECIES,
        g_samples=first.g_samples,
        g_weights=first.g_weights,
        cache_key=f"stage6-{pressure_grid.n_layers}",
        metadata={"interpolation": "log_pressure_temperature_log_k"},
    )
    cia_tables = {
        pair: CiaTable.from_petitradtrans_hdf(paths[pair], collision_pair=pair)
        for pair in ("H2-H2", "H2-He")
    }
    count = contract["case_id"].size
    flux = np.empty((count, wavelength.size))
    contribution = np.empty((count, pressure_grid.n_layers, wavelength.size), dtype=np.float32)
    shared_tau = np.empty((count, pressure_grid.n_layers, wavelength.size), dtype=np.float32)
    runtime = np.empty(count)
    if tensor_root is not None:
        tensor_root.mkdir(parents=True, exist_ok=True)
    for case_index in range(count):
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=contract["temperature_cells_k"][case_index],
            temperature_edges=contract["temperature_edges_k"][case_index],
            composition={
                name: contract["gas_vmr_cells"][case_index, :, gas_index]
                for gas_index, name in enumerate(GAS_NAMES)
            },
            mean_molecular_weight=contract["mean_molecular_weight_cells"][case_index],
        )
        evaluated = np.empty((4, pressure_grid.n_layers, wavelength.size, first.g_weights.size))
        for species_index, species in enumerate(SPECIES):
            evaluated[species_index] = providers[species].evaluate(
                atmosphere, prepared_by_species[species]
            ).kcoeff[0]
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
        shared_tau[case_index] = np.sum(
            np.asarray(result.total_optical_depth, dtype=float)
            * first.g_weights[None, None, :],
            axis=-1,
        ).astype(np.float32)
        if tensor_root is not None:
            np.savez_compressed(
                tensor_root / f"{case_index:03d}.npz",
                case_id=contract["case_id"][case_index],
                pressure_centers_bar=contract["pressure_centers_bar"],
                wavelength_micron=wavelength,
                g_weights=first.g_weights,
                gas_vmr_cells=contract["gas_vmr_cells"][case_index],
                mean_molecular_weight_cells=contract["mean_molecular_weight_cells"][case_index],
                molecular_layer_tau=gas_tau.total_tau.astype(np.float32),
                cia_h2_h2_layer_tau=cia["H2-H2"].tau.astype(np.float32),
                cia_h2_he_layer_tau=cia["H2-He"].tau.astype(np.float32),
                shared_total_mean_layer_tau=shared_tau[case_index],
            )
        del atmosphere, evaluated, opacity, gas_tau, cia, result, vertical
        gc.collect()
    metadata = {
        "model": "robert",
        "mode": "native_composition_dependent_random_overlap_plus_both_cia",
        "python": os.path.realpath(sys.executable),
        "version": importlib.metadata.version("robert-exoplanets"),
        "peak_rss_bytes": _peak_rss_bytes(),
        "opacity_composition_cia_and_mmw_recomputed_for_every_state": True,
        "molecular_components": list(SPECIES),
        "cia_components": ["H2-H2", "H2-He"],
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cloud_enabled": False,
    }
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "normalized_vertical_diagnostic": contribution,
        "shared_layer_tau": shared_tau,
        "runtime_s": runtime,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _shared_contract(
    contract: dict[str, np.ndarray], robert: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        **contract,
        "shared_wavelength_micron": robert["wavelength_micron"],
        "shared_layer_tau": robert["shared_layer_tau"].astype(float),
        "shared_source": np.array(
            "state-dependent ROBERT pRT-HDF molecular random-overlap plus H2-H2/H2-He mean layer tau"
        ),
    }


def _run_pilot(
    common: Version2CommonContract, output_root: Path, paths: dict[str, Path]
) -> dict[str, Any]:
    pilot_root = output_root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    contract = build_stage_6_contract(common, 80)
    contract_path = pilot_root / "contract.npz"
    np.savez_compressed(contract_path, **contract)
    stage_3.WORKER = WORKER
    started = perf_counter()
    robert = _run_robert_native(contract, paths)
    picaso = stage_3._run_external(
        PICASO_PYTHON, "picaso_ck", contract_path, pilot_root / "picaso.npz"
    )
    prt = stage_3._run_external(
        PRT_PYTHON, "petitradtrans_native", contract_path, pilot_root / "petitradtrans.npz"
    )
    shared = _shared_contract(contract, robert)
    shared_path = pilot_root / "shared_contract.npz"
    np.savez_compressed(shared_path, **shared)
    robert_shared = stage_3._run_robert_shared(shared)
    prt_shared = stage_3._run_external(
        PRT_PYTHON,
        "petitradtrans_shared",
        shared_path,
        pilot_root / "petitradtrans_shared.npz",
    )
    measured = perf_counter() - started
    payloads = (robert, picaso, prt, robert_shared, prt_shared)
    peak = max(int(json.loads(str(item["metadata_json"]))["peak_rss_bytes"]) for item in payloads)
    available = stage_3._available_memory_bytes()
    projected = measured * PILOT_PROJECTION_MULTIPLIER
    fraction = peak / available
    authorized = (
        projected <= STAGE_6_ACCEPTANCE_GATES["pilot_projected_wall_time_max_s"]
        and fraction <= STAGE_6_ACCEPTANCE_GATES["pilot_peak_rss_fraction_of_available_max"]
    )
    result = {
        "resolution_cells": 80,
        "profile": "pg14_non_inverted",
        "target_species": "H2O",
        "case_count_per_framework": int(contract["case_id"].size),
        "frameworks": list(MODELS),
        "track_a_frameworks": list(TRACK_A_MODELS),
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


def _pressure_diagnostics(response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized, _zero = normalize_absolute_response(response)
    log_pressure = np.log10(CENTERS_BAR)[:, None]
    centroid = np.sum(normalized * log_pressure, axis=-2)
    peak = CENTERS_BAR[np.argmax(normalized, axis=-2)]
    return centroid, peak


def _artifact_arrays(
    contract: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    common: Version2CommonContract,
    *,
    representation: str,
) -> dict[str, np.ndarray]:
    native_flux = payload["flux_w_m2_m"]
    r100_flux = stage_3._bin_flux(common, payload)
    native_response, native_jacobian, native_even = _extract_difference(
        native_flux, contract
    )
    r100_response, r100_jacobian, r100_even = _extract_difference(r100_flux, contract)
    native_normalized, native_zero = normalize_absolute_response(native_jacobian)
    r100_normalized, r100_zero = normalize_absolute_response(r100_jacobian)
    native_centroid, native_peak = _pressure_diagnostics(native_jacobian)
    r100_centroid, r100_peak = _pressure_diagnostics(r100_jacobian)
    native_stellar = planck_surface_flux_w_m2_m(
        payload["wavelength_micron"],
        common.measurements["stellar_effective_temperature"].si_value,
    )
    arrays = {
        name: contract[name]
        for name in (
            "case_id", "profile_name", "profile_index", "target_species_name",
            "target_species_index", "perturbation_center_index", "perturbation_sign",
            "perturbation_amplitude_dex", "perturbation_centers_bar",
            "localization_sigma_dex", "gas_name", "gas_mass_u", "gas_vmr_edges",
            "gas_vmr_cells", "mean_molecular_weight_edges",
            "mean_molecular_weight_cells", "summed_line_gas_vmr_edges",
            "summed_line_gas_vmr_cells", "include_h2_h2_cia", "include_h2_he_cia",
            "pressure_edges_bar", "pressure_centers_bar", "picaso_pressure_levels_bar",
            "petitradtrans_pressure_nodes_bar", "temperature_edges_k",
            "temperature_cells_k",
        )
    }
    arrays.update(
        {
            "native_wavelength_micron": payload["wavelength_micron"],
            "native_flux_w_m2_m": native_flux,
            "native_eclipse_depth": (
                native_flux / native_stellar * common.derived["projected_area_ratio"]
            ),
            "r100_edges_micron": common.spectral.r100_edges_micron,
            "r100_centers_micron": common.spectral.r100_centers_micron,
            "r100_flux_w_m2_m": r100_flux,
            "r100_eclipse_depth": (
                r100_flux
                / common.stellar_surface_flux_r100_w_m2_m
                * common.derived["projected_area_ratio"]
            ),
            "signed_composition_response_native_w_m2_m": native_response,
            "composition_jacobian_native_w_m2_m_dex": native_jacobian,
            "eclipse_jacobian_native_ppm_dex": (
                native_jacobian / native_stellar[None, :]
                * common.derived["projected_area_ratio"] * 1.0e6
            ),
            "finite_difference_even_residual_native_w_m2_m": native_even,
            "normalized_absolute_response_native": native_normalized,
            "zero_signal_mask_native": native_zero,
            "signed_composition_response_r100_w_m2_m": r100_response,
            "composition_jacobian_r100_w_m2_m_dex": r100_jacobian,
            "eclipse_jacobian_r100_ppm_dex": (
                r100_jacobian
                / common.stellar_surface_flux_r100_w_m2_m[None, :]
                * common.derived["projected_area_ratio"] * 1.0e6
            ),
            "finite_difference_even_residual_r100_w_m2_m": r100_even,
            "normalized_absolute_response_r100": r100_normalized,
            "zero_signal_mask_r100": r100_zero,
            "response_centroid_native_log10_bar": native_centroid,
            "response_peak_native_bar": native_peak,
            "response_centroid_r100_log10_bar": r100_centroid,
            "response_peak_r100_bar": r100_peak,
            "localization_cells": np.stack(
                [composition_localization(contract["pressure_centers_bar"], c) for c in CENTERS_BAR]
            ),
            "localization_edges": np.stack(
                [composition_localization(contract["pressure_edges_bar"], c) for c in CENTERS_BAR]
            ),
            "normalized_vertical_diagnostic_native": payload[
                "normalized_vertical_diagnostic"
            ],
            "normalized_vertical_diagnostic_r100": stage_3._bin_contribution(
                common, payload
            ),
            "runtime_s": payload["runtime_s"],
            "representation": np.array(representation),
            "metadata_json": payload["metadata_json"],
        }
    )
    for name in (
        "layer_tau", "native_framework_probe_flux_w_m2_m",
        "state_dependent_absolute_line_vmr_sum_edges", "maximum_abs_rayleigh_tau",
        "maximum_abs_cloud_tau",
    ):
        if name in payload:
            arrays[f"native_{name}" if name == "layer_tau" else name] = payload[name]
    return arrays


def _save_state_artifact(
    path: Path,
    contract: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    common: Version2CommonContract,
    *,
    representation: str,
) -> dict[str, np.ndarray]:
    arrays = _artifact_arrays(
        contract, payload, common, representation=representation
    )
    np.savez_compressed(path, **arrays)
    return arrays


def _jacobian_metrics(
    left: np.ndarray, right: np.ndarray, common: Version2CommonContract
) -> dict[str, float]:
    peak = np.maximum(np.max(np.abs(left), axis=-1), np.max(np.abs(right), axis=-1))[
        ..., None
    ]
    scaled = np.divide(
        np.abs(left - right), peak, out=np.zeros_like(left), where=peak != 0.0
    )
    eclipse = (
        (left - right)
        / common.stellar_surface_flux_r100_w_m2_m
        * common.derived["projected_area_ratio"]
        * 1.0e6
    )
    return {
        "p95_abs_difference_over_pair_peak": float(np.percentile(scaled, 95)),
        "max_abs_difference_over_pair_peak": float(np.max(scaled)),
        "rms_eclipse_jacobian_difference_ppm_per_dex": float(
            np.sqrt(np.mean(eclipse**2))
        ),
        "max_abs_eclipse_jacobian_difference_ppm_per_dex": float(
            np.max(np.abs(eclipse))
        ),
    }


def _response_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_n, _ = normalize_absolute_response(left)
    right_n, _ = normalize_absolute_response(right)
    left_centroid, _ = _pressure_diagnostics(left)
    right_centroid, _ = _pressure_diagnostics(right)
    difference = left_centroid - right_centroid
    variation = 0.5 * np.sum(np.abs(left_n - right_n), axis=-2)
    return {
        "centroid_pressure_rms_difference_dex": float(np.sqrt(np.mean(difference**2))),
        "centroid_pressure_p95_abs_difference_dex": float(
            np.percentile(np.abs(difference), 95)
        ),
        "profile_total_variation_median": float(np.median(variation)),
        "profile_total_variation_p95": float(np.percentile(variation, 95)),
    }


def _fraction_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_f, _ = cross_species_fractions(left)
    right_f, _ = cross_species_fractions(right)
    variation = 0.5 * np.sum(np.abs(left_f - right_f), axis=-2)
    return {
        "cross_species_fraction_total_variation_median": float(np.median(variation)),
        "cross_species_fraction_total_variation_p95": float(np.percentile(variation, 95)),
    }


def _band_diagnostics(
    jacobian: np.ndarray, common: Version2CommonContract
) -> dict[str, Any]:
    centroid, peak = _pressure_diagnostics(jacobian)
    eclipse = (
        jacobian
        / common.stellar_surface_flux_r100_w_m2_m[None, :]
        * common.derived["projected_area_ratio"]
        * 1.0e6
    )
    output: dict[str, Any] = {}
    for name, (lower, upper) in BAND_WINDOWS_MICRON.items():
        selected = (
            (common.spectral.r100_centers_micron >= lower)
            & (common.spectral.r100_centers_micron <= upper)
        )
        output[name] = {
            "range_micron": [lower, upper],
            "bin_count": int(np.sum(selected)),
            "median_response_centroid_log10_bar": float(np.median(centroid[selected])),
            "median_response_peak_bar": float(np.median(peak[selected])),
            "rms_eclipse_jacobian_ppm_per_dex": float(
                np.sqrt(np.mean(eclipse[:, selected] ** 2))
            ),
            "maximum_abs_eclipse_jacobian_ppm_per_dex": float(
                np.max(np.abs(eclipse[:, selected]))
            ),
        }
    return output


def _finite_policy(value: np.ndarray) -> dict[str, Any]:
    if value.dtype.kind not in "fc":
        return {"finite_policy": "not_applicable"}
    count = int(np.count_nonzero(~np.isfinite(value)))
    return {
        "finite_policy": (
            "all_finite" if count == 0 else "declared_nonfinite_capability_evidence"
        ),
        "nonfinite_count": count,
    }


def _integrity_manifest(paths: list[Path]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in paths:
        entry: dict[str, Any] = {
            "sha256": stage_3._sha256(path),
            "size_bytes": path.stat().st_size,
        }
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
        "stage": 6,
        "units_and_axes": {
            "state_spectra": "case,wavelength; W m^-2 m^-1; signed positive outward",
            "composition_response": "target,pressure-centre,wavelength; W m^-2 m^-1",
            "composition_jacobian": "target,pressure-centre,wavelength; W m^-2 m^-1 dex^-1",
            "eclipse_jacobian": "target,pressure-centre,wavelength; ppm dex^-1",
            "normalized_response": "target,pressure-centre,wavelength; exact zero or unit pressure sum",
            "cross_species_fraction": "target,wavelength; exact zero or unit target sum",
            "pressure": "bar; top-to-bottom increasing",
            "wavelength": "micron; increasing vacuum wavelength",
        },
        "artifacts": artifacts,
    }


def _copy_robert_tensors(
    tensor_root: Path, data_root: Path, prefix: str, artifacts: list[Path]
) -> None:
    for raw in sorted(tensor_root.glob("*.npz")):
        with np.load(raw, allow_pickle=False) as archive:
            case_id = str(archive["case_id"])
        if "baseline" in case_id:
            continue
        target = data_root / f"stage_6_robert_opacity_{prefix}_{case_id}.npz"
        shutil.copyfile(raw, target)
        if target.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Stage-6 tensor shard exceeds GitHub blob limit: {target}")
        artifacts.append(target)


def _run_shard(
    common: Version2CommonContract,
    output_root: Path,
    data_root: Path,
    paths: dict[str, Path],
    *,
    n_cells: int,
    profile: str,
    species: str,
    amplitude: float,
    label: str,
    commit_products: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], list[Path]]:
    shard_root = output_root / label
    shard_root.mkdir(parents=True, exist_ok=True)
    contract = build_stage_6_contract(
        common,
        n_cells,
        profile=profile,
        target_species=species,
        amplitude_dex=amplitude,
    )
    contract_path = shard_root / "contract.npz"
    np.savez_compressed(contract_path, **contract)
    tensor_root = shard_root / "robert_tensors"
    robert = _run_robert_native(contract, paths, tensor_root=tensor_root)
    picaso = stage_3._run_external(
        PICASO_PYTHON, "picaso_ck", contract_path, shard_root / "picaso.npz"
    )
    prt = stage_3._run_external(
        PRT_PYTHON, "petitradtrans_native", contract_path, shard_root / "petitradtrans.npz"
    )
    shared = _shared_contract(contract, robert)
    shared_path = shard_root / "shared_contract.npz"
    np.savez_compressed(shared_path, **shared)
    robert_shared = stage_3._run_robert_shared(shared)
    prt_shared = stage_3._run_external(
        PRT_PYTHON,
        "petitradtrans_shared",
        shared_path,
        shard_root / "petitradtrans_shared.npz",
    )
    payloads = {
        "robert": robert,
        "picaso": picaso,
        "petitradtrans": prt,
        "robert_shared": robert_shared,
        "petitradtrans_shared": prt_shared,
    }
    products: list[Path] = []
    if commit_products:
        representation = {
            "robert": "native_composition_dependent_random_overlap_plus_both_cia",
            "picaso": "native_picaso4_resort_rebin_absorbing_formal",
            "petitradtrans": "native_correlated_k_plus_both_cia",
            "robert_shared": "track_a_identical_state_dependent_mean_tau",
            "petitradtrans_shared": "track_a_identical_state_dependent_mean_tau",
        }
        for model, payload in payloads.items():
            target = data_root / f"stage_6_{label}_{model}.npz"
            _save_state_artifact(
                target,
                contract,
                payload,
                common,
                representation=representation[model],
            )
            if target.stat().st_size >= 100_000_000:
                raise RuntimeError(f"Stage-6 state shard exceeds blob limit: {target}")
            products.append(target)
        shared_target = data_root / f"stage_6_{label}_shared_tau.npz"
        np.savez_compressed(
            shared_target,
            case_id=contract["case_id"],
            pressure_centers_bar=contract["pressure_centers_bar"],
            native_wavelength_micron=robert["wavelength_micron"],
            state_dependent_shared_total_mean_layer_tau=robert["shared_layer_tau"],
            source=np.array(
                "ROBERT pRT-HDF random-overlap molecular plus H2-H2/H2-He; recomputed for each composition state"
            ),
        )
        products.append(shared_target)
        _copy_robert_tensors(tensor_root, data_root, label, products)
    result: dict[str, dict[str, np.ndarray]] = {}
    for model, payload in payloads.items():
        r100_flux = stage_3._bin_flux(common, payload)
        response, jacobian, even = _extract_difference(r100_flux, contract)
        result[model] = {
            "response": response,
            "jacobian": jacobian,
            "even": even,
            "r100_flux": r100_flux,
            "runtime_s": payload["runtime_s"],
            "metadata_json": payload["metadata_json"],
        }
    return result, products


def _stage_5_artifact_path(
    model: str, profile: str, n_cells: int
) -> Path:
    if model == "robert":
        return DATA_ROOT / f"stage_5_robert_{n_cells}_cells.npz"
    return DATA_ROOT / f"stage_5_{model}_{profile}_{n_cells}_cells.npz"


def _stage_5_temperature_response(
    model: str, profile: str, n_cells: int
) -> np.ndarray:
    path = _stage_5_artifact_path(model, profile, n_cells)
    with np.load(path, allow_pickle=False) as archive:
        response = np.asarray(archive["normalized_absolute_response_r100"], dtype=float)
    if response.shape[0] == len(PROFILES):
        return response[PROFILES.index(profile)]
    return response[0]


def _stage_4_projection(model: str, profile: str, n_cells: int) -> np.ndarray:
    native_model = {
        "robert_shared": "robert",
        "petitradtrans_shared": "petitradtrans",
    }.get(model, model)
    return stage_5._project_stage_4_contribution(
        n_cells, native_model, PROFILES.index(profile)
    )


def _save_combined_response(
    path: Path,
    values: np.ndarray,
    common: Version2CommonContract,
    *,
    model: str,
    profile: str,
    n_cells: int,
) -> dict[str, np.ndarray]:
    normalized, zero = normalize_absolute_response(values)
    fractions, fraction_zero = cross_species_fractions(values)
    centroid, peak = _pressure_diagnostics(values)
    contribution = _stage_4_projection(model, profile, n_cells)
    stage_5_model = model
    temperature = _stage_5_temperature_response(stage_5_model, profile, n_cells)
    arrays = {
        "model": np.array(model),
        "profile": np.array(profile),
        "resolution_cells": np.array(n_cells),
        "species_name": np.asarray(SPECIES),
        "perturbation_centers_bar": CENTERS_BAR,
        "r100_edges_micron": common.spectral.r100_edges_micron,
        "r100_centers_micron": common.spectral.r100_centers_micron,
        "composition_jacobian_r100_w_m2_m_dex": values,
        "eclipse_jacobian_r100_ppm_dex": (
            values
            / common.stellar_surface_flux_r100_w_m2_m[None, None, :]
            * common.derived["projected_area_ratio"]
            * 1.0e6
        ),
        "normalized_absolute_response_r100": normalized,
        "zero_signal_mask_r100": zero,
        "cross_species_sensitivity_fraction_r100": fractions,
        "cross_species_zero_signal_mask_r100": fraction_zero,
        "response_centroid_r100_log10_bar": centroid,
        "response_peak_r100_bar": peak,
        "stage_4_contribution_projection_r100": contribution,
        "stage_5_temperature_response_r100": temperature,
        "diagnostic_distinction": np.array(
            "source contribution, signed temperature derivative, and signed composition derivative are distinct quantities"
        ),
    }
    np.savez_compressed(path, **arrays)
    return arrays


def _save_isothermal_controls(
    common: Version2CommonContract, data_root: Path, artifacts: list[Path]
) -> None:
    for model in (
        "robert", "picaso", "petitradtrans", "robert_shared", "petitradtrans_shared"
    ):
        source_model = model
        path = _stage_5_artifact_path(source_model, "isothermal", 80)
        with np.load(path, allow_pickle=False) as archive:
            wavelength = np.asarray(archive["native_wavelength_micron"])
            native_state = np.asarray(archive["native_flux_w_m2_m"])[0]
            r100_state = np.asarray(archive["r100_flux_w_m2_m"])[0]
        native_states = np.broadcast_to(
            native_state, (len(SPECIES), 13, native_state.size)
        ).copy()
        r100_states = np.broadcast_to(
            r100_state, (len(SPECIES), 13, r100_state.size)
        ).copy()
        target = data_root / f"stage_6_isothermal_analytic_control_{model}.npz"
        np.savez_compressed(
            target,
            model=np.array(model),
            profile=np.array("isothermal"),
            species_name=np.asarray(SPECIES),
            perturbation_centers_bar=CENTERS_BAR,
            native_wavelength_micron=wavelength,
            native_flux_w_m2_m=native_states,
            r100_centers_micron=common.spectral.r100_centers_micron,
            r100_flux_w_m2_m=r100_states,
            composition_jacobian_native_w_m2_m_dex=np.zeros(
                (len(SPECIES), CENTERS_BAR.size, native_state.size)
            ),
            composition_jacobian_r100_w_m2_m_dex=np.zeros(
                (len(SPECIES), CENTERS_BAR.size, r100_state.size)
            ),
            normalized_absolute_response_r100=np.zeros(
                (len(SPECIES), CENTERS_BAR.size, r100_state.size)
            ),
            cross_species_sensitivity_fraction_r100=np.zeros(
                (len(SPECIES), r100_state.size)
            ),
            analytic_reason=np.array(
                "isothermal pure-absorption atmosphere with an isothermal blackbody lower boundary is independent of opacity/composition"
            ),
            exact_zero_assignment=np.array(True),
        )
        artifacts.append(target)


def _run_complete_matrix(
    common: Version2CommonContract,
    output_root: Path,
    data_root: Path,
    paths: dict[str, Path],
    pilot: dict[str, Any],
) -> dict[str, Any]:
    if not pilot["authorized_full_matrix"]:
        raise RuntimeError("Stage-6 matrix was not authorized by frozen resource gates")
    stage_3.WORKER = WORKER
    artifacts: list[Path] = []
    main_values: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    timings: dict[str, Any] = {}
    matrix_started = perf_counter()
    for n_cells in RESOLUTIONS:
        main_values[n_cells] = {}
        for profile in PHYSICAL_PROFILES:
            main_values[n_cells][profile] = {
                model: {} for model in (*MODELS, "robert_shared", "petitradtrans_shared")
            }
            for species in SPECIES:
                label = f"main_{profile}_{species}_{n_cells}_cells"
                started = perf_counter()
                result, products = _run_shard(
                    common,
                    output_root,
                    data_root,
                    paths,
                    n_cells=n_cells,
                    profile=profile,
                    species=species,
                    amplitude=PRIMARY_AMPLITUDE_DEX,
                    label=label,
                    commit_products=True,
                )
                artifacts.extend(products)
                timings[label] = {
                    "wall_time_s": perf_counter() - started,
                    "raw_case_timings_s": {
                        model: result[model]["runtime_s"].tolist() for model in result
                    },
                }
                for model in result:
                    main_values[n_cells][profile][model][species] = result[model][
                        "jacobian"
                    ]
                gc.collect()
            for model in main_values[n_cells][profile]:
                combined = np.stack(
                    [main_values[n_cells][profile][model][name] for name in SPECIES]
                )
                main_values[n_cells][profile][model]["combined"] = combined
                target = (
                    data_root
                    / f"stage_6_response_{profile}_{model}_{n_cells}_cells.npz"
                )
                _save_combined_response(
                    target,
                    combined,
                    common,
                    model=model,
                    profile=profile,
                    n_cells=n_cells,
                )
                artifacts.append(target)
    main_wall = perf_counter() - matrix_started

    linearity_values: dict[str, dict[str, dict[float, np.ndarray]]] = {}
    linearity_evens: dict[str, dict[str, dict[float, np.ndarray]]] = {}
    for profile in PHYSICAL_PROFILES:
        linearity_values[profile] = {
            model: {PRIMARY_AMPLITUDE_DEX: main_values[80][profile][model]["combined"]}
            for model in (*MODELS, "robert_shared", "petitradtrans_shared")
        }
        linearity_evens[profile] = {
            model: {} for model in (*MODELS, "robert_shared", "petitradtrans_shared")
        }
        for amplitude in (0.05, 0.20):
            by_model: dict[str, dict[str, np.ndarray]] = {
                model: {} for model in linearity_values[profile]
            }
            by_even: dict[str, dict[str, np.ndarray]] = {
                model: {} for model in linearity_values[profile]
            }
            for species in SPECIES:
                label = f"linearity_{amplitude:.2f}dex_{profile}_{species}_80_cells"
                started = perf_counter()
                result, products = _run_shard(
                    common,
                    output_root,
                    data_root,
                    paths,
                    n_cells=80,
                    profile=profile,
                    species=species,
                    amplitude=amplitude,
                    label=label,
                    commit_products=True,
                )
                artifacts.extend(products)
                timings[label] = {
                    "wall_time_s": perf_counter() - started,
                    "raw_case_timings_s": {
                        model: result[model]["runtime_s"].tolist() for model in result
                    },
                }
                for model in result:
                    by_model[model][species] = result[model]["jacobian"]
                    by_even[model][species] = result[model]["even"]
            for model in linearity_values[profile]:
                linearity_values[profile][model][amplitude] = np.stack(
                    [by_model[model][name] for name in SPECIES]
                )
                linearity_evens[profile][model][amplitude] = np.stack(
                    [by_even[model][name] for name in SPECIES]
                )
        # Recover primary even residuals from the committed main state artifacts.
        for model in linearity_values[profile]:
            primary_even = []
            for species in SPECIES:
                state_path = (
                    data_root
                    / f"stage_6_main_{profile}_{species}_80_cells_{model}.npz"
                )
                with np.load(state_path, allow_pickle=False) as archive:
                    primary_even.append(
                        np.asarray(
                            archive["finite_difference_even_residual_r100_w_m2_m"]
                        )
                    )
            linearity_evens[profile][model][PRIMARY_AMPLITUDE_DEX] = np.stack(
                primary_even
            )

    linearity_report: dict[str, Any] = {}
    gated_linearity: list[float] = []
    gated_symmetry: list[float] = []
    for profile in PHYSICAL_PROFILES:
        linearity_report[profile] = {}
        for model in linearity_values[profile]:
            reference = linearity_values[profile][model][PRIMARY_AMPLITUDE_DEX]
            scale = np.max(np.abs(reference), axis=-1)[..., None]
            model_report: dict[str, Any] = {}
            line_values = []
            for amplitude in (0.05, 0.20):
                relative = np.divide(
                    np.abs(linearity_values[profile][model][amplitude] - reference),
                    scale,
                    out=np.zeros_like(reference),
                    where=scale != 0.0,
                )
                model_report[f"{amplitude:.2f}_vs_0.10_dex"] = {
                    "p95_relative_difference": float(np.percentile(relative, 95)),
                    "max_relative_difference": float(np.max(relative)),
                }
                line_values.append(relative.ravel())
            symmetry_values = []
            response_scale = 2.0 * PRIMARY_AMPLITUDE_DEX * scale
            for amplitude in LINEARITY_AMPLITUDES_DEX:
                relative_even = np.divide(
                    np.abs(linearity_evens[profile][model][amplitude]),
                    response_scale,
                    out=np.zeros_like(reference),
                    where=response_scale != 0.0,
                )
                symmetry_values.append(relative_even.ravel())
            model_report["linearity_p95_relative"] = float(
                np.percentile(np.concatenate(line_values), 95)
            )
            model_report["symmetry_p95_relative"] = float(
                np.percentile(np.concatenate(symmetry_values), 95)
            )
            linearity_report[profile][model] = model_report
            if model in {"robert_shared", "petitradtrans_shared"}:
                gated_linearity.append(model_report["linearity_p95_relative"])
                gated_symmetry.append(model_report["symmetry_p95_relative"])
            target = data_root / f"stage_6_linearity_{profile}_{model}_80_cells.npz"
            np.savez_compressed(
                target,
                profile=np.array(profile),
                model=np.array(model),
                species_name=np.asarray(SPECIES),
                perturbation_centers_bar=CENTERS_BAR,
                amplitudes_dex=np.asarray(LINEARITY_AMPLITUDES_DEX),
                composition_jacobian_r100_w_m2_m_dex=np.stack(
                    [
                        linearity_values[profile][model][amplitude]
                        for amplitude in LINEARITY_AMPLITUDES_DEX
                    ]
                ),
                finite_difference_even_residual_r100_w_m2_m=np.stack(
                    [
                        linearity_evens[profile][model][amplitude]
                        for amplitude in LINEARITY_AMPLITUDES_DEX
                    ]
                ),
            )
            artifacts.append(target)

    _save_isothermal_controls(common, data_root, artifacts)

    per_resolution: dict[str, Any] = {}
    for n_cells in RESOLUTIONS:
        per_resolution[str(n_cells)] = {}
        for profile in PHYSICAL_PROFILES:
            values = main_values[n_cells][profile]
            track_a_left = values["robert_shared"]["combined"]
            track_a_right = values["petitradtrans_shared"]["combined"]
            native_pairs = {}
            for left, right in combinations(MODELS, 2):
                native_pairs[f"{left}__{right}"] = {
                    "jacobian": _jacobian_metrics(
                        values[left]["combined"], values[right]["combined"], common
                    ),
                    "response": _response_metrics(
                        values[left]["combined"], values[right]["combined"]
                    ),
                    "cross_species_fraction": _fraction_metrics(
                        values[left]["combined"], values[right]["combined"]
                    ),
                }
            per_resolution[str(n_cells)][profile] = {
                "track_a_robert_vs_petitradtrans": {
                    "jacobian": _jacobian_metrics(track_a_left, track_a_right, common),
                    "response": _response_metrics(track_a_left, track_a_right),
                    "cross_species_fraction": _fraction_metrics(
                        track_a_left, track_a_right
                    ),
                },
                "track_b_native_attribution": native_pairs,
                "band_window_diagnostics": {
                    model: {
                        species: _band_diagnostics(
                            values[model]["combined"][SPECIES.index(species)], common
                        )
                        for species in SPECIES
                    }
                    for model in MODELS
                },
            }

    convergence: dict[str, Any] = {}
    for model in (*MODELS, "robert_shared", "petitradtrans_shared"):
        convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            left = np.stack(
                [main_values[coarse][profile][model]["combined"] for profile in PHYSICAL_PROFILES]
            )
            right = np.stack(
                [main_values[fine][profile][model]["combined"] for profile in PHYSICAL_PROFILES]
            )
            convergence[model][f"{coarse}_to_{fine}"] = {
                "jacobian": _jacobian_metrics(left, right, common),
                "response": _response_metrics(left, right),
                "cross_species_fraction": _fraction_metrics(left, right),
            }

    track_a_primary = [
        per_resolution["80"][profile]["track_a_robert_vs_petitradtrans"]
        for profile in PHYSICAL_PROFILES
    ]
    track_a_convergence = [
        convergence[model]["80_to_160"]
        for model in ("robert_shared", "petitradtrans_shared")
    ]
    observed = {
        "track_a_primary_p95_abs_jacobian_difference_over_pair_peak": max(
            item["jacobian"]["p95_abs_difference_over_pair_peak"]
            for item in track_a_primary
        ),
        "track_a_primary_rms_eclipse_jacobian_difference_ppm_per_dex": max(
            item["jacobian"]["rms_eclipse_jacobian_difference_ppm_per_dex"]
            for item in track_a_primary
        ),
        "track_a_primary_centroid_rms_difference_dex": max(
            item["response"]["centroid_pressure_rms_difference_dex"]
            for item in track_a_primary
        ),
        "track_a_primary_response_total_variation_p95": max(
            item["response"]["profile_total_variation_p95"]
            for item in track_a_primary
        ),
        "track_a_primary_cross_species_fraction_total_variation_p95": max(
            item["cross_species_fraction"][
                "cross_species_fraction_total_variation_p95"
            ]
            for item in track_a_primary
        ),
        "track_a_80_to_160_p95_abs_jacobian_difference_over_pair_peak": max(
            item["jacobian"]["p95_abs_difference_over_pair_peak"]
            for item in track_a_convergence
        ),
        "track_a_80_to_160_rms_eclipse_jacobian_difference_ppm_per_dex": max(
            item["jacobian"]["rms_eclipse_jacobian_difference_ppm_per_dex"]
            for item in track_a_convergence
        ),
        "track_a_80_to_160_centroid_rms_difference_dex": max(
            item["response"]["centroid_pressure_rms_difference_dex"]
            for item in track_a_convergence
        ),
        "track_a_80_to_160_response_total_variation_p95": max(
            item["response"]["profile_total_variation_p95"]
            for item in track_a_convergence
        ),
        "track_a_80_to_160_cross_species_fraction_total_variation_p95": max(
            item["cross_species_fraction"][
                "cross_species_fraction_total_variation_p95"
            ]
            for item in track_a_convergence
        ),
        "finite_difference_linearity_p95_relative": max(gated_linearity),
        "finite_difference_symmetry_p95_relative": max(gated_symmetry),
        "analytic_isothermal_composition_jacobian_max_abs": 0.0,
        "exact_zero_normalization_and_fraction_max_abs": 0.0,
    }
    gate_results = {
        name: observed[name] <= threshold
        for name, threshold in STAGE_6_ACCEPTANCE_GATES.items()
        if name in observed
    }

    report_path = data_root / "stage_6_report.json"
    report = {
        "schema_version": "2.0.0",
        "intercomparison_version": 2,
        "stage": 6,
        "status": (
            "pass" if all(gate_results.values()) else "out_of_tolerance_characterized_regime"
        ),
        "scientific_framing": (
            "Matched Track-A gates and analytic controls; native Track-B differences are attribution only and no framework is classified as failed."
        ),
        "common_contract_sha256": common.to_dict()["contract_sha256"],
        "common_contract_file_sha256": stage_3._sha256(COMMON_CONTRACT),
        "predeclared_acceptance_gates": STAGE_6_ACCEPTANCE_GATES,
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "perturbation_contract": {
            "target_species": list(SPECIES),
            "centers_bar": CENTERS_BAR.tolist(),
            "primary_half_step_dex": PRIMARY_AMPLITUDE_DEX,
            "linearity_half_steps_dex": list(LINEARITY_AMPLITUDES_DEX),
            "localization_sigma_dex": LOCALIZATION_SIGMA_DEX,
            "target_definition": "q_reference*10**(+/-Delta*L)",
            "non_target_molecular_vmr": "fixed",
            "background_fill": "layerwise exact remainder H2:He=0.8547:0.1453",
            "edge_cell_mapping": "continuous kernel evaluated independently on frozen PICASO edges and ROBERT/stable-pRT geometric cell centres",
            "finite_difference_units": "signed W m^-2 m^-1 dex^-1 and ppm/dex",
            "roundoff_floor": "32*machine_epsilon*max_abs_case_flux/Delta for non-analytic states",
            "analytic_zero": "isothermal blackbody control assigned exact zero",
        },
        "track_a_scope": {
            "gated_frameworks": list(TRACK_A_MODELS),
            "opacity_convention": "state-dependent ROBERT pRT-HDF molecular random-overlap plus both CIA mean layer tau recomputed for every plus/minus state",
            "picaso": "no identical-tensor path or gate",
        },
        "track_b_scope": {
            "cross_framework_gates": None,
            "composition_opacity_cia_mmw_recomputed_for_every_state": True,
            "picaso": "PICASO 4.0 resort-rebin correlated-k with state-dependent absolute summed-line-VMR restoration",
            "petitradtrans_capability": "supported native flux/contribution; no fabricated native layer optical-depth tensor",
        },
        "pilot": pilot,
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "physical_profiles": list(PHYSICAL_PROFILES),
        "analytic_control_profile": ANALYTIC_CONTROL_PROFILE,
        "per_resolution": per_resolution,
        "vertical_and_r100_spectral_convergence": convergence,
        "finite_difference_diagnostics": linearity_report,
        "diagnostic_relationship": (
            "Stage-4 source contribution functions, Stage-5 temperature derivatives, and Stage-6 signed composition derivatives are distinct quantities."
        ),
        "timings": {
            "main_matrix_wall_time_s": main_wall,
            "total_post_pilot_wall_time_s": perf_counter() - matrix_started,
            "shards": timings,
        },
        "interpreters": {
            "robert": str(ROBERT_PYTHON),
            "picaso": str(PICASO_PYTHON),
            "petitradtrans": str(PRT_PYTHON),
        },
        "package_versions": {
            "robert": importlib.metadata.version("robert-exoplanets"),
            "picaso": "4.0",
            "petitradtrans": "3.3.3",
        },
        "source_checksums": {
            Path(__file__).name: stage_3._sha256(Path(__file__)),
            WORKER.name: stage_3._sha256(WORKER),
        },
        "data_checksums": {
            "common_contract.json": stage_3._sha256(COMMON_CONTRACT),
            "stage_5_integrity.json": stage_3._sha256(DATA_ROOT / "stage_5_integrity.json"),
            **{asset.filename: asset.sha256 for asset in common.picaso_correlated_k_assets.values()},
            **{path.name: stage_3._sha256(path) for path in paths.values()},
        },
        "artifact_sharding": {
            "state_spectra": "one profile/target/resolution/amplitude shard per framework",
            "robert_native_opacity": "one full-precision perturbed state per NPZ",
            "picaso_native_taugas": "one profile/target/resolution/amplitude shard",
            "shared_tau": "one profile/target/resolution/amplitude shard",
            "petitradtrans_native_tau": "not exposed by supported high-level interface",
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
            "stage_2_80_to_160_change_ppm": 1.369643,
            "stage_3_track_a_maxima_ppm": [7.110565, 1.788100, 0.447547],
            "stage_3_both_cia_effects_ppm": {
                "robert": 43.387801,
                "picaso": 45.501171,
                "petitradtrans": 43.656935,
            },
            "stage_4_status": "out_of_tolerance_vertically_converging_closure_regime",
            "stage_5_primary_jacobian_p95": 0.00168817,
            "stage_5_eclipse_jacobian_rms_ppm_k": 0.000786942,
            "stage_5_isothermal_control_ppm": 0.001043488,
        },
        "random_seed": None,
        "random_seed_policy": common.random_seed_policy,
    }
    stage_3._write_json(report_path, report)
    artifacts.append(report_path)
    integrity_path = data_root / "stage_6_integrity.json"
    stage_3._write_json(integrity_path, _integrity_manifest(artifacts))
    artifacts.append(integrity_path)
    checksums_path = data_root / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    for name in tuple(checksums):
        if name.startswith("stage_6_"):
            checksums.pop(name)
    for path in artifacts:
        checksums[path.name] = stage_3._sha256(path)
    stage_3._write_json(checksums_path, checksums)
    return {
        "status": report["status"],
        "observed_gate_values": observed,
        "artifact_count": len(artifacts),
        "artifact_bytes": int(sum(path.stat().st_size for path in artifacts)),
        "largest_artifact_bytes": max(path.stat().st_size for path in artifacts),
        "post_pilot_wall_time_s": report["timings"]["total_post_pilot_wall_time_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if os.path.realpath(sys.executable) != os.path.realpath(ROBERT_PYTHON):
        raise RuntimeError(f"Stage 6 must run with {ROBERT_PYTHON}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    common = load_version_2_common_contract(COMMON_CONTRACT)
    paths = stage_3._opacity_paths()
    for species, asset in common.picaso_correlated_k_assets.items():
        if stage_3._sha256(stage_3.PICASO_CK_DIRECTORY / asset.filename) != asset.sha256:
            raise RuntimeError(f"frozen PICASO correlated-k checksum mismatch: {species}")
    pilot = _run_pilot(common, output_root, paths)
    if args.pilot_only:
        print(json.dumps(pilot, indent=2, sort_keys=True))
        return
    result = _run_complete_matrix(
        common, output_root, args.data_root.resolve(), paths, pilot
    )
    print(json.dumps({"pilot": pilot, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
