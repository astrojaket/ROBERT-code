"""Run Version-2 Stage 7 absorbing-cloud placement/extinction comparison.

Generated workers and full-precision products remain below the ignored local
Stage-7 output tree.  Ordinary Git receives code, tests, compact contracts,
documentation, and small human-readable summaries only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from time import perf_counter
from typing import Any

import numpy as np
import psutil

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    Version2CommonContract,
    load_version_2_common_contract,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_7 import (
    CLOUD_BOTTOM_PRESSURE_BAR,
    CLOUD_OPTICAL_DEPTHS,
    CLOUD_TOP_PRESSURES_BAR,
    EXTINCTION_SLOPES,
    REFERENCE_WAVELENGTH_MICRON,
    build_cloud_extinction_matrix,
    cloud_definitions,
    pilot_resource_decision,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY / "docs/data/emission_intercomparison/version_2"
COMMON_CONTRACT = DATA_ROOT / "common_contract.json"
DEFAULT_OUTPUT = REPOSITORY / "examples/outputs/emission_intercomparison/version_2/stage_7"
DEFAULT_PRODUCT_ROOT = DEFAULT_OUTPUT / "products"
WORKER = Path(__file__).with_name("run_emission_intercomparison_v2_stage_7_external.py")
STAGE_3_PATH = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_3.py")
STAGE_4_PATH = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_4.py")
ARCHIVED_CLOUD_CONTRACT = (
    REPOSITORY / "data/validation/end_to_end_cloud_parity/shared_physical_contract.npz"
)
ARCHIVED_CLOUD_OUTPUT = (
    REPOSITORY / "data/validation/end_to_end_cloud_parity/picaso_virga_independent_output.npz"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_3 = _load_module("emission_v2_stage_3_for_stage_7", STAGE_3_PATH)
stage_4 = _load_module("emission_v2_stage_4_for_stage_7", STAGE_4_PATH)

ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
PHYSICAL_PROFILES = ("pg14_non_inverted", "pg14_inverted")
MODELS = ("robert", "picaso", "petitradtrans")
TRACK_A_MODELS = ("robert", "petitradtrans")
TRACKS = ("track_a_shared_tau", "track_b_native_cloud")
BAND_WINDOWS_MICRON = stage_4.BAND_WINDOWS_MICRON
PILOT_PROFILE = "pg14_non_inverted"
PILOT_CLOUD_LABEL = "deck_tau1_top10mbar_slope+0"


# Frozen before the pilot and complete matrix.  Track B remains ungated.
STAGE_7_ACCEPTANCE_GATES = {
    "track_a_primary_absolute_spectrum_p95_symmetric_relative": 0.01,
    "track_a_primary_cloud_effect_p95_difference_over_pair_peak": 0.02,
    "track_a_primary_cloud_effect_eclipse_rms_ppm": 0.50,
    "track_a_primary_contribution_centroid_rms_dex": 0.05,
    "track_a_primary_contribution_profile_tv_p95": 0.05,
    "track_a_primary_cloud_response_profile_tv_p95": 0.08,
    "track_a_80_to_160_absolute_spectrum_p95_symmetric_relative": 0.01,
    "track_a_80_to_160_cloud_effect_p95_difference_over_pair_peak": 0.02,
    "track_a_80_to_160_cloud_effect_eclipse_rms_ppm": 0.50,
    "track_a_80_to_160_contribution_centroid_rms_dex": 0.05,
    "track_a_80_to_160_contribution_profile_tv_p95": 0.05,
    "track_a_80_to_160_cloud_response_profile_tv_p95": 0.08,
    "analytic_isothermal_cloud_effect_max_abs_ppm": 1.0e-10,
    "single_scattering_albedo_max_abs": 0.0,
    "pilot_projected_wall_time_max_s": 7200.0,
    "pilot_peak_rss_fraction_of_available_max": 0.60,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _archived_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(ARCHIVED_CLOUD_CONTRACT, allow_pickle=False) as contract:
        pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    with np.load(ARCHIVED_CLOUD_OUTPUT, allow_pickle=False) as output:
        wavelength = np.asarray(output["wavelength_micron"], dtype=float)
        extinction = np.asarray(output["cloud_tau"], dtype=float)
    return pressure_edges, wavelength, extinction


def _cloud_input_wavelength(common: Version2CommonContract) -> np.ndarray:
    return np.unique(
        np.concatenate(
            (
                np.array([0.3, 12.0]),
                common.spectral.r100_centers_micron,
            )
        )
    )


def build_stage_7_contract(
    common: Version2CommonContract,
    n_cells: int,
    *,
    profiles: tuple[str, ...] = PROFILES,
    cloud_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the frozen gas, profile, cloud, and process-interface contract."""

    base = stage_4.build_stage_4_contract(common, n_cells)
    profile_indices = np.asarray([PROFILES.index(name) for name in profiles], dtype=int)
    definitions = cloud_definitions()
    selected_clouds = (
        np.arange(len(definitions), dtype=int)
        if cloud_indices is None
        else np.asarray(cloud_indices, dtype=int)
    )
    if selected_clouds.ndim != 1 or selected_clouds.size == 0:
        raise ValueError("cloud_indices must select at least one definition")
    archived_pressure, archived_wavelength, archived_extinction = _archived_cloud()
    input_wavelength = _cloud_input_wavelength(common)
    cloud_input_tau = build_cloud_extinction_matrix(
        base["pressure_edges_bar"],
        input_wavelength,
        archived_pressure_edges_bar=archived_pressure,
        archived_wavelength_micron=archived_wavelength,
        archived_extinction_tau=archived_extinction,
    )
    case_profile_original = np.repeat(profile_indices, selected_clouds.size)
    case_profile_index = np.repeat(np.arange(len(profiles)), selected_clouds.size)
    case_cloud_index = np.tile(selected_clouds, len(profiles))
    case_ids = np.asarray(
        [
            f"{PROFILES[original]}__{definitions[cloud].label}__{n_cells}_cells"
            for original, cloud in zip(
                case_profile_original, case_cloud_index, strict=True
            )
        ]
    )
    composition = base["gas_vmr"][0]
    declared_mmw = float(base["mean_molecular_weight_u"][0])
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(7),
        "case_id": case_ids,
        "profile_name": np.asarray(profiles),
        "profile_index": case_profile_index,
        "case_cloud_index": case_cloud_index,
        "cloud_label": np.asarray([item.label for item in definitions]),
        "cloud_kind": np.asarray([item.kind for item in definitions]),
        "cloud_optical_depth_at_reference": np.asarray(
            [item.optical_depth_at_reference for item in definitions]
        ),
        "cloud_top_pressure_bar": np.asarray(
            [item.cloud_top_pressure_bar for item in definitions]
        ),
        "cloud_extinction_slope": np.asarray(
            [item.extinction_slope for item in definitions]
        ),
        "cloud_reference_wavelength_micron": np.array(
            REFERENCE_WAVELENGTH_MICRON
        ),
        "cloud_bottom_pressure_bar": np.array(CLOUD_BOTTOM_PRESSURE_BAR),
        "cloud_single_scattering_albedo": np.zeros(len(definitions)),
        "cloud_asymmetry_factor": np.zeros(len(definitions)),
        "cloud_input_wavelength_micron": input_wavelength,
        "cloud_input_extinction_tau": cloud_input_tau,
        "selected_cloud_index": selected_clouds,
        "archived_pressure_edges_bar": archived_pressure,
        "archived_wavelength_micron": archived_wavelength,
        "archived_extinction_tau": archived_extinction,
        "gas_name": base["gas_name"],
        "gas_mass_u": base["gas_mass_u"],
        "gas_vmr": np.broadcast_to(composition, (case_ids.size, composition.size)).copy(),
        "mean_molecular_weight_u": np.full(case_ids.size, declared_mmw),
        "molecular_species_name": base["molecular_species_name"],
        "molecular_species_active": np.ones(
            (case_ids.size, base["molecular_species_name"].size), dtype=bool
        ),
        "include_h2_h2_cia": np.ones(case_ids.size, dtype=bool),
        "include_h2_he_cia": np.ones(case_ids.size, dtype=bool),
        "pressure_edges_bar": base["pressure_edges_bar"],
        "pressure_centers_bar": base["pressure_centers_bar"],
        "picaso_pressure_levels_bar": base["picaso_pressure_levels_bar"],
        "petitradtrans_pressure_nodes_bar": base["petitradtrans_pressure_nodes_bar"],
        "temperature_edges_by_profile_k": base["temperature_edges_by_profile_k"]
        [profile_indices],
        "temperature_cells_by_profile_k": base["temperature_cells_by_profile_k"]
        [profile_indices],
        "temperature_edges_k": base["temperature_edges_by_profile_k"][case_profile_original],
        "temperature_cells_k": base["temperature_cells_by_profile_k"][case_profile_original],
        "gravity_m_s2": base["gravity_m_s2"],
        "emission_mu": base["emission_mu"],
        "legendre_weights": base["legendre_weights"],
        "disk_weights": base["disk_weights"],
    }


def _gas_contract(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Collapse a Stage-7 case matrix to one clear gas case per profile."""

    count = contract["profile_name"].size
    first_by_profile = np.asarray(
        [int(np.flatnonzero(contract["profile_index"] == index)[0]) for index in range(count)]
    )
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(4),
        "case_id": np.asarray(
            [f"{name}_molecular_plus_h2_h2_and_h2_he_cia_gas" for name in contract["profile_name"]]
        ),
        "profile_name": contract["profile_name"],
        "profile_index": np.arange(count),
        "factor_name": np.full(count, "molecular_plus_h2_h2_and_h2_he_cia"),
        "include_h2_h2_cia": np.ones(count, dtype=bool),
        "include_h2_he_cia": np.ones(count, dtype=bool),
        "gas_name": contract["gas_name"],
        "gas_mass_u": contract["gas_mass_u"],
        "gas_vmr": contract["gas_vmr"][first_by_profile],
        "mean_molecular_weight_u": contract["mean_molecular_weight_u"][first_by_profile],
        "molecular_species_name": contract["molecular_species_name"],
        "molecular_species_active": np.ones(
            (count, contract["molecular_species_name"].size), dtype=bool
        ),
        "pressure_edges_bar": contract["pressure_edges_bar"],
        "pressure_centers_bar": contract["pressure_centers_bar"],
        "picaso_pressure_levels_bar": contract["picaso_pressure_levels_bar"],
        "petitradtrans_pressure_nodes_bar": contract["petitradtrans_pressure_nodes_bar"],
        "temperature_edges_k": contract["temperature_edges_by_profile_k"],
        "temperature_cells_k": contract["temperature_cells_by_profile_k"],
        "temperature_edges_by_profile_k": contract["temperature_edges_by_profile_k"],
        "temperature_cells_by_profile_k": contract["temperature_cells_by_profile_k"],
        "gravity_m_s2": contract["gravity_m_s2"],
        "emission_mu": contract["emission_mu"],
        "legendre_weights": contract["legendre_weights"],
        "disk_weights": contract["disk_weights"],
    }


def _native_cloud_tau(
    contract: dict[str, np.ndarray], wavelength: np.ndarray
) -> np.ndarray:
    return build_cloud_extinction_matrix(
        contract["pressure_edges_bar"],
        wavelength,
        archived_pressure_edges_bar=contract["archived_pressure_edges_bar"],
        archived_wavelength_micron=contract["archived_wavelength_micron"],
        archived_extinction_tau=contract["archived_extinction_tau"],
    )


def _run_robert_native(
    contract: dict[str, np.ndarray], paths: dict[str, Path]
) -> dict[str, np.ndarray]:
    from robert_exoplanets import planck_radiance_wavelength
    from robert_exoplanets.rt import integrate_thermal_emission

    previous_profiles = stage_3.PROFILES
    stage_3.PROFILES = tuple(str(value) for value in contract["profile_name"])
    try:
        gas = stage_3._run_robert_native(_gas_contract(contract), paths)
    finally:
        stage_3.PROFILES = previous_profiles
    wavelength = gas["wavelength_micron"]
    cloud_tau = _native_cloud_tau(contract, wavelength)
    molecular = gas["molecular_layer_tau_by_profile"].astype(float)
    cia_h2_h2 = gas["cia_h2_h2_layer_tau_by_profile"].astype(float)
    cia_h2_he = gas["cia_h2_he_layer_tau_by_profile"].astype(float)
    g_weights = gas["g_weights"]
    flux = []
    vertical = []
    timings = []
    for case_index, cloud_index_value in enumerate(contract["case_cloud_index"]):
        profile_index = int(contract["profile_index"][case_index])
        total_tau = (
            molecular[profile_index]
            + cia_h2_h2[profile_index, :, :, None]
            + cia_h2_he[profile_index, :, :, None]
            + cloud_tau[int(cloud_index_value), :, :, None]
        )
        cells = contract["temperature_cells_k"][case_index]
        edges = contract["temperature_edges_k"][case_index]
        layer_source = np.stack(
            [planck_radiance_wavelength(wavelength, value) for value in cells]
        )
        level_source = np.stack(
            [planck_radiance_wavelength(wavelength, value) for value in edges]
        )
        started = perf_counter()
        result = integrate_thermal_emission(
            total_tau,
            layer_source,
            g_weights,
            np.broadcast_to(
                1.0 / contract["emission_mu"][:, None],
                (contract["emission_mu"].size, total_tau.shape[0]),
            ),
            level_source_ordered=level_source,
            bottom_source=level_source[-1],
            backend="numpy",
        )
        timings.append(perf_counter() - started)
        layers = np.tensordot(
            contract["disk_weights"], result.point_layer_contribution_radiance, axes=(0, 0)
        )
        bottom = np.tensordot(
            contract["disk_weights"], result.point_bottom_contribution_radiance, axes=(0, 0)
        )
        layers[-1] += bottom
        flux.append(np.pi * np.sum(layers, axis=0))
        vertical.append(stage_3._normalise_vertical(layers).astype(np.float32))
    metadata = {
        "model": "robert",
        "mode": "native_random_overlap_gas_plus_layer_absorbing_cloud",
        "python": os.path.realpath(sys.executable),
        "version": importlib.metadata.version("robert-exoplanets"),
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cloud_single_scattering_albedo": 0.0,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "normalized_vertical_diagnostic": np.asarray(vertical),
        "runtime_s": np.asarray(timings),
        "g_weights": g_weights,
        "molecular_layer_tau_by_profile": molecular.astype(np.float32),
        "cia_h2_h2_layer_tau_by_profile": cia_h2_h2.astype(np.float32),
        "cia_h2_he_layer_tau_by_profile": cia_h2_he.astype(np.float32),
        "native_cloud_extinction_tau_by_cloud": cloud_tau.astype(np.float32),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _shared_contract(
    contract: dict[str, np.ndarray], robert_native: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    molecular = np.sum(
        robert_native["molecular_layer_tau_by_profile"].astype(float)
        * robert_native["g_weights"][None, None, None, :],
        axis=-1,
    )
    cloud = robert_native["native_cloud_extinction_tau_by_cloud"].astype(float)
    total = np.empty(
        (
            contract["case_id"].size,
            contract["pressure_centers_bar"].size,
            robert_native["wavelength_micron"].size,
        )
    )
    for case_index, cloud_index_value in enumerate(contract["case_cloud_index"]):
        profile_index = int(contract["profile_index"][case_index])
        total[case_index] = (
            molecular[profile_index]
            + robert_native["cia_h2_h2_layer_tau_by_profile"][profile_index]
            + robert_native["cia_h2_he_layer_tau_by_profile"][profile_index]
            + cloud[int(cloud_index_value)]
        )
    return {
        **contract,
        "shared_wavelength_micron": robert_native["wavelength_micron"],
        "shared_gas_layer_tau_by_profile": (
            molecular
            + robert_native["cia_h2_h2_layer_tau_by_profile"].astype(float)
            + robert_native["cia_h2_he_layer_tau_by_profile"].astype(float)
        ),
        "shared_cloud_extinction_tau_by_cloud": cloud,
        "shared_layer_tau": total,
        "shared_source": np.array(
            "identical ROBERT-derived mean molecular plus H2-H2/H2-He CIA tau "
            "and identical absorbing-cloud layer tau"
        ),
    }


def _run_external(
    python: Path,
    mode: str,
    contract_path: Path,
    output_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    command = [str(python), str(WORKER), mode, str(contract_path), str(output_path)]
    if mode == "picaso_native":
        command.extend(["--picaso-ck-directory", str(stage_3.PICASO_CK_DIRECTORY)])
    elif mode == "petitradtrans_native":
        command.extend(["--prt-input-data", str(stage_3.PRT_INPUT_DATA)])
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_btl", "self")
    if mode == "picaso_native":
        cache_root = output_path.parent / "picaso-v4-cache"
        numba_cache = cache_root / "picaso-v4-numba-cache"
        mpl_cache = cache_root / "picaso-v4-matplotlib"
        numba_cache.mkdir(parents=True, exist_ok=True)
        mpl_cache.mkdir(parents=True, exist_ok=True)
        environment["picaso_refdata"] = str(stage_3.PICASO_REFERENCE)
        environment["NUMBA_CACHE_DIR"] = str(numba_cache)
        environment["MPLCONFIGDIR"] = str(mpl_cache)
    stdout_path = output_path.with_suffix(".stdout.log")
    stderr_path = output_path.with_suffix(".stderr.log")
    started = perf_counter()
    peak_tree = 0
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                members = [monitored, *monitored.children(recursive=True)]
                peak_tree = max(
                    peak_tree,
                    sum(item.memory_info().rss for item in members if item.is_running()),
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
            time.sleep(0.05)
        return_code = process.wait()
    wall = perf_counter() - started
    stderr_text = stderr_path.read_text()
    if return_code != 0:
        raise RuntimeError(
            f"Stage-7 {mode} worker failed ({return_code})\n{stderr_text}"
        )
    payload = _load_npz(output_path)
    metadata = json.loads(str(payload["metadata_json"]))
    execution = {
        "command": command,
        "wall_time_s": wall,
        "process_tree_peak_rss_bytes": max(
            peak_tree, int(metadata.get("peak_rss_bytes", 0))
        ),
        "warnings": [line for line in stderr_text.splitlines() if line.strip()],
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
    }
    return payload, execution


def _bin_vertical(
    common: Version2CommonContract, payload: dict[str, np.ndarray]
) -> np.ndarray:
    return stage_3._bin_contribution(common, payload).astype(np.float32)


def _augment_r100(
    common: Version2CommonContract,
    contract: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = dict(payload)
    flux = stage_3._bin_flux(common, payload)
    vertical = _bin_vertical(common, payload)
    eclipse = (
        flux
        / common.stellar_surface_flux_r100_w_m2_m
        * common.derived["projected_area_ratio"]
    )
    effect_flux = np.zeros_like(flux)
    effect_eclipse = np.zeros_like(eclipse)
    response = np.zeros_like(vertical)
    raw_isothermal_max = 0.0
    for profile_index, profile in enumerate(contract["profile_name"]):
        selected = np.flatnonzero(contract["profile_index"] == profile_index)
        clear = selected[contract["case_cloud_index"][selected] == 0]
        if clear.size != 1:
            raise ValueError("each Stage-7 profile requires one clear control")
        raw_flux = flux[selected] - flux[clear[0]]
        raw_eclipse = eclipse[selected] - eclipse[clear[0]]
        if str(profile) == "isothermal":
            raw_isothermal_max = max(raw_isothermal_max, float(np.max(np.abs(raw_eclipse))))
        else:
            effect_flux[selected] = raw_flux
            effect_eclipse[selected] = raw_eclipse
        difference = np.abs(vertical[selected] - vertical[clear[0]])
        response[selected] = stage_3._normalise_vertical(difference).astype(np.float32)
    result.update(
        {
            "r100_centers_micron": common.spectral.r100_centers_micron,
            "r100_edges_micron": common.spectral.r100_edges_micron,
            "r100_flux_w_m2_m": flux,
            "r100_eclipse_depth": eclipse,
            "r100_cloud_effect_flux_w_m2_m": effect_flux,
            "r100_cloud_effect_eclipse_ppm": effect_eclipse * 1.0e6,
            "r100_normalized_vertical_diagnostic": vertical,
            "r100_normalized_cloud_effect": response,
            "raw_isothermal_max_abs_eclipse_cancellation": np.array(raw_isothermal_max),
        }
    )
    return result


def _run_resolution(
    common: Version2CommonContract,
    n_cells: int,
    root: Path,
    paths: dict[str, Path],
    *,
    profiles: tuple[str, ...] = PROFILES,
    cloud_indices: np.ndarray | None = None,
    suffix: str = "",
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    run_root = root / f"{n_cells}_cells{suffix}"
    run_root.mkdir(parents=True, exist_ok=True)
    contract = build_stage_7_contract(
        common, n_cells, profiles=profiles, cloud_indices=cloud_indices
    )
    contract_path = run_root / "native_contract.npz"
    np.savez_compressed(contract_path, **contract)
    execution: dict[str, Any] = {track: {} for track in TRACKS}
    outputs: dict[str, dict[str, np.ndarray]] = {}

    started = perf_counter()
    robert_native = _run_robert_native(contract, paths)
    execution["track_b_native_cloud"]["robert"] = {
        "wall_time_s": perf_counter() - started,
        "process_tree_peak_rss_bytes": _peak_rss_bytes(),
        "warnings": [],
    }
    robert_native = _augment_r100(common, contract, robert_native)
    np.savez_compressed(run_root / "track_b_robert.npz", **robert_native)
    outputs["track_b_robert"] = robert_native

    for model, python, mode in (
        ("picaso", PICASO_PYTHON, "picaso_native"),
        ("petitradtrans", PRT_PYTHON, "petitradtrans_native"),
    ):
        output_path = run_root / f"track_b_{model}.npz"
        payload, details = _run_external(
            python, mode, contract_path, output_path
        )
        execution["track_b_native_cloud"][model] = details
        payload = _augment_r100(common, contract, payload)
        np.savez_compressed(output_path, **payload)
        outputs[f"track_b_{model}"] = payload
        gc.collect()

    shared = _shared_contract(contract, robert_native)
    shared_path = run_root / "shared_contract.npz"
    np.savez_compressed(shared_path, **shared)
    started = perf_counter()
    robert_shared = stage_3._run_robert_shared(shared)
    execution["track_a_shared_tau"]["robert"] = {
        "wall_time_s": perf_counter() - started,
        "process_tree_peak_rss_bytes": _peak_rss_bytes(),
        "warnings": [],
    }
    robert_shared = _augment_r100(common, contract, robert_shared)
    np.savez_compressed(run_root / "track_a_robert.npz", **robert_shared)
    outputs["track_a_robert"] = robert_shared
    prt_shared_path = run_root / "track_a_petitradtrans.npz"
    prt_shared, details = _run_external(
        PRT_PYTHON, "petitradtrans_shared", shared_path, prt_shared_path
    )
    execution["track_a_shared_tau"]["petitradtrans"] = details
    prt_shared = _augment_r100(common, contract, prt_shared)
    np.savez_compressed(prt_shared_path, **prt_shared)
    outputs["track_a_petitradtrans"] = prt_shared
    return outputs, execution, contract


def _pilot(
    common: Version2CommonContract, output_root: Path, paths: dict[str, Path]
) -> dict[str, Any]:
    definitions = cloud_definitions()
    moderate = next(index for index, item in enumerate(definitions) if item.label == PILOT_CLOUD_LABEL)
    selected = np.asarray([0, moderate])
    repetitions: dict[str, Any] = {}
    timings: dict[str, dict[str, float]] = {}
    largest_peak = 0
    pilot_started = perf_counter()
    for temperature_state, suffix in (("cold", "_pilot_cold"), ("warm", "_pilot_warm")):
        outputs, execution, contract = _run_resolution(
            common,
            PRIMARY_RESOLUTION,
            output_root / "pilot",
            paths,
            profiles=(PILOT_PROFILE,),
            cloud_indices=selected,
            suffix=suffix,
        )
        repetitions[temperature_state] = execution
        for track, models in execution.items():
            for model, details in models.items():
                label = f"{track}:{model}"
                timings.setdefault(label, {})[
                    f"{temperature_state}_wall_time_s"
                ] = float(details["wall_time_s"])
                largest_peak = max(
                    largest_peak, int(details["process_tree_peak_rss_bytes"])
                )
        if temperature_state == "warm":
            native_counts = {
                label: int(payload["wavelength_micron"].size)
                for label, payload in outputs.items()
            }
            tensor_sizes = {
                label: {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "bytes": int(value.nbytes),
                    }
                    for name, value in payload.items()
                    if name
                    in {
                        "molecular_layer_tau_by_profile",
                        "native_cloud_extinction_tau_by_cloud",
                        "native_gas_tau_by_profile",
                        "native_cloud_tau_by_cloud",
                        "normalized_vertical_diagnostic",
                    }
                }
                for label, payload in outputs.items()
            }
    available = int(psutil.virtual_memory().available)
    decision = pilot_resource_decision(
        timings,
        pilot_case_count=contract["case_id"].size,
        peak_process_tree_rss_bytes=largest_peak,
        available_memory_bytes=available,
    )
    pilot = {
        "schema_version": "2.0.0",
        "stage": 7,
        "kind": "mandatory_cold_and_warm_three_framework_primary_grid_pilot",
        "profile": PILOT_PROFILE,
        "clouds": ["clear", PILOT_CLOUD_LABEL],
        "resolution_cells": PRIMARY_RESOLUTION,
        "framework_track_timings": timings,
        "cold_and_warm_execution": repetitions,
        "native_wavelength_counts": native_counts,
        "retained_tensor_sizes": tensor_sizes,
        "decision": decision,
        "pilot_total_wall_time_s": perf_counter() - pilot_started,
    }
    _write_json(output_root / "pilot" / "stage_7_pilot_report.json", pilot)
    _write_json(
        output_root / "pilot" / "stage_7_pilot_integrity.json",
        _integrity_manifest(output_root / "pilot"),
    )
    return pilot


def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    denominator = np.abs(left) + np.abs(right)
    symmetric = np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros_like(left),
        where=denominator > 0.0,
    )
    return {
        "p95_abs_symmetric_relative": float(np.percentile(symmetric, 95.0)),
        "max_abs_symmetric_relative": float(np.max(symmetric)),
    }


def _effect_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_values = np.atleast_2d(left)
    right_values = np.atleast_2d(right)
    peak = np.maximum(
        np.max(np.abs(left_values), axis=-1), np.max(np.abs(right_values), axis=-1)
    )[:, None]
    scaled = np.divide(
        np.abs(left_values - right_values),
        peak,
        out=np.zeros_like(left_values),
        where=peak > 0.0,
    )
    return {
        "p95_abs_difference_over_pair_peak": float(np.percentile(scaled, 95.0)),
        "rms_eclipse_difference_ppm": float(
            np.sqrt(np.mean((left_values - right_values) ** 2))
        ),
    }


def _vertical_metrics(
    left: np.ndarray, right: np.ndarray, pressure_bar: np.ndarray
) -> dict[str, float]:
    left_values = stage_3._normalise_vertical(left)
    right_values = stage_3._normalise_vertical(right)
    log_pressure = np.log10(pressure_bar)[:, None]
    left_centroid = np.sum(left_values * log_pressure, axis=-2)
    right_centroid = np.sum(right_values * log_pressure, axis=-2)
    difference = left_centroid - right_centroid
    total_variation = 0.5 * np.sum(np.abs(left_values - right_values), axis=-2)
    return {
        "centroid_rms_difference_dex": float(np.sqrt(np.mean(difference**2))),
        "profile_total_variation_p95": float(np.percentile(total_variation, 95.0)),
    }


def _coarsen_vertical(values: np.ndarray, target_cells: int) -> np.ndarray:
    ratio = values.shape[-2] // target_cells
    shape = (*values.shape[:-2], target_cells, ratio, values.shape[-1])
    return values.reshape(shape).sum(axis=-2)


def _summarize_and_gate(
    by_resolution: dict[int, dict[str, dict[str, np.ndarray]]],
    contracts: dict[int, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, float], dict[str, bool]]:
    primary = by_resolution[80]
    robert = primary["track_a_robert"]
    prt = primary["track_a_petitradtrans"]
    spectrum_metrics = []
    effect_metrics = []
    contribution_metrics = []
    response_metrics = []
    for case_index in range(contracts[80]["case_id"].size):
        spectrum_metrics.append(
            _difference(robert["r100_flux_w_m2_m"][case_index], prt["r100_flux_w_m2_m"][case_index])
        )
        effect_metrics.append(
            _effect_difference(
                robert["r100_cloud_effect_eclipse_ppm"][case_index],
                prt["r100_cloud_effect_eclipse_ppm"][case_index],
            )
        )
        contribution_metrics.append(
            _vertical_metrics(
                robert["r100_normalized_vertical_diagnostic"][case_index],
                prt["r100_normalized_vertical_diagnostic"][case_index],
                contracts[80]["pressure_centers_bar"],
            )
        )
        response_metrics.append(
            _vertical_metrics(
                robert["r100_normalized_cloud_effect"][case_index],
                prt["r100_normalized_cloud_effect"][case_index],
                contracts[80]["pressure_centers_bar"],
            )
        )

    convergence_records = []
    for model in TRACK_A_MODELS:
        coarse = by_resolution[80][f"track_a_{model}"]
        fine = by_resolution[160][f"track_a_{model}"]
        for case_index in range(contracts[80]["case_id"].size):
            convergence_records.append(
                {
                    "spectrum": _difference(
                        coarse["r100_flux_w_m2_m"][case_index],
                        fine["r100_flux_w_m2_m"][case_index],
                    ),
                    "effect": _effect_difference(
                        coarse["r100_cloud_effect_eclipse_ppm"][case_index],
                        fine["r100_cloud_effect_eclipse_ppm"][case_index],
                    ),
                    "contribution": _vertical_metrics(
                        coarse["r100_normalized_vertical_diagnostic"][case_index],
                        _coarsen_vertical(
                            fine["r100_normalized_vertical_diagnostic"][case_index], 80
                        ),
                        contracts[80]["pressure_centers_bar"],
                    ),
                    "response": _vertical_metrics(
                        coarse["r100_normalized_cloud_effect"][case_index],
                        _coarsen_vertical(
                            fine["r100_normalized_cloud_effect"][case_index], 80
                        ),
                        contracts[80]["pressure_centers_bar"],
                    ),
                }
            )
    observed = {
        "track_a_primary_absolute_spectrum_p95_symmetric_relative": max(
            item["p95_abs_symmetric_relative"] for item in spectrum_metrics
        ),
        "track_a_primary_cloud_effect_p95_difference_over_pair_peak": max(
            item["p95_abs_difference_over_pair_peak"] for item in effect_metrics
        ),
        "track_a_primary_cloud_effect_eclipse_rms_ppm": max(
            item["rms_eclipse_difference_ppm"] for item in effect_metrics
        ),
        "track_a_primary_contribution_centroid_rms_dex": max(
            item["centroid_rms_difference_dex"] for item in contribution_metrics
        ),
        "track_a_primary_contribution_profile_tv_p95": max(
            item["profile_total_variation_p95"] for item in contribution_metrics
        ),
        "track_a_primary_cloud_response_profile_tv_p95": max(
            item["profile_total_variation_p95"] for item in response_metrics
        ),
        "track_a_80_to_160_absolute_spectrum_p95_symmetric_relative": max(
            item["spectrum"]["p95_abs_symmetric_relative"] for item in convergence_records
        ),
        "track_a_80_to_160_cloud_effect_p95_difference_over_pair_peak": max(
            item["effect"]["p95_abs_difference_over_pair_peak"] for item in convergence_records
        ),
        "track_a_80_to_160_cloud_effect_eclipse_rms_ppm": max(
            item["effect"]["rms_eclipse_difference_ppm"] for item in convergence_records
        ),
        "track_a_80_to_160_contribution_centroid_rms_dex": max(
            item["contribution"]["centroid_rms_difference_dex"] for item in convergence_records
        ),
        "track_a_80_to_160_contribution_profile_tv_p95": max(
            item["contribution"]["profile_total_variation_p95"] for item in convergence_records
        ),
        "track_a_80_to_160_cloud_response_profile_tv_p95": max(
            item["response"]["profile_total_variation_p95"] for item in convergence_records
        ),
        "analytic_isothermal_cloud_effect_max_abs_ppm": max(
            float(
                np.max(
                    np.abs(payload["r100_cloud_effect_eclipse_ppm"][contracts[80]["profile_index"] == 0])
                )
            )
            for payload in primary.values()
        ),
        "single_scattering_albedo_max_abs": float(
            np.max(np.abs(contracts[80]["cloud_single_scattering_albedo"]))
        ),
    }
    gate_results = {
        name: observed[name] <= limit
        for name, limit in STAGE_7_ACCEPTANCE_GATES.items()
        if name in observed
    }
    representative: dict[str, Any] = {}
    for label in (PILOT_CLOUD_LABEL, "archived_virga_mie_extinction"):
        cloud_index = int(np.flatnonzero(contracts[80]["cloud_label"] == label)[0])
        case_index = int(
            np.flatnonzero(
                (contracts[80]["profile_name"][contracts[80]["profile_index"]] == "pg14_non_inverted")
                & (contracts[80]["case_cloud_index"] == cloud_index)
            )[0]
        )
        representative[label] = {
            track_model: {
                "minimum_cloud_effect_ppm": float(
                    np.min(payload["r100_cloud_effect_eclipse_ppm"][case_index])
                ),
                "maximum_cloud_effect_ppm": float(
                    np.max(payload["r100_cloud_effect_eclipse_ppm"][case_index])
                ),
                "rms_cloud_effect_ppm": float(
                    np.sqrt(
                        np.mean(payload["r100_cloud_effect_eclipse_ppm"][case_index] ** 2)
                    )
                ),
            }
            for track_model, payload in primary.items()
        }
    return {"representative_cases": representative}, observed, gate_results


def _integrity_manifest(root: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {
            "stage_7_integrity.json",
            "stage_7_pilot_integrity.json",
        }:
            continue
        entry: dict[str, Any] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                entry["arrays"] = {
                    name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
                    for name in archive.files
                }
        artifacts[str(path.relative_to(root))] = entry
    return {
        "schema_version": "1.0.0",
        "stage": 7,
        "local_archival_release_candidate": True,
        "artifacts": artifacts,
    }


def _run_complete(
    common: Version2CommonContract,
    output_root: Path,
    product_root: Path,
    paths: dict[str, Path],
    pilot: dict[str, Any],
) -> dict[str, Any]:
    matrix_started = perf_counter()
    by_resolution: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    contracts: dict[int, dict[str, np.ndarray]] = {}
    execution: dict[str, Any] = {}
    for n_cells in RESOLUTIONS:
        outputs, resolution_execution, contract = _run_resolution(
            common, n_cells, product_root / "matrix", paths
        )
        by_resolution[n_cells] = {
            name: {
                key: value
                for key, value in payload.items()
                if key.startswith("r100_") or key == "wavelength_micron"
            }
            for name, payload in outputs.items()
        }
        contracts[n_cells] = contract
        execution[str(n_cells)] = resolution_execution
        del outputs
        gc.collect()
    compact, observed, gate_results = _summarize_and_gate(by_resolution, contracts)
    matrix_wall = perf_counter() - matrix_started
    complete_matrix_peak_rss = max(
        metadata["process_tree_peak_rss_bytes"]
        for resolution_execution in execution.values()
        for track_execution in resolution_execution.values()
        for metadata in track_execution.values()
    )
    report = {
        "schema_version": "2.0.0",
        "intercomparison_version": 2,
        "stage": 7,
        "status": "pass" if all(gate_results.values()) else "out_of_tolerance_characterized_regime",
        "scientific_framing": (
            "Absorbing-cloud placement and extinction are characterized without "
            "classifying a framework as failed or tuning definitions after inspection."
        ),
        "common_contract_sha256": common.to_dict()["contract_sha256"],
        "predeclared_acceptance_gates": STAGE_7_ACCEPTANCE_GATES,
        "predeclared_band_windows_micron": {
            name: list(bounds) for name, bounds in BAND_WINDOWS_MICRON.items()
        },
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "profiles": list(PROFILES),
        "physical_profiles": list(PHYSICAL_PROFILES),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "cloud_contract": {
            "optical_depth_at_reference": list(CLOUD_OPTICAL_DEPTHS),
            "cloud_top_pressure_bar": list(CLOUD_TOP_PRESSURES_BAR),
            "extinction_slope": list(EXTINCTION_SLOPES),
            "reference_wavelength_micron": REFERENCE_WAVELENGTH_MICRON,
            "cloud_bottom_pressure_bar": CLOUD_BOTTOM_PRESSURE_BAR,
            "placement": "continuous top; fractional boundary cell; uniform d_tau/d_log_pressure",
            "spectral_sign": "tau(lambda)=tau_ref*(lambda/5 micron)^slope",
            "archived_case": "versioned PICASO/Virga extinction only; omega0 overwritten by exact Stage-7 zero",
            "tabulated_mapping": "log-pressure-conservative; positive spectral log interpolation; constant wavelength endpoints; exact zeros retained",
        },
        "tracks": {
            "track_a_shared_tau": {
                "gated_frameworks": ["robert", "petitradtrans"],
                "picaso": "no identical exact-omega0 tensor interface or invented gate",
            },
            "track_b_native_cloud": {
                "cross_framework_gates": None,
                "robert": "native random-overlap gas plus layer cloud extinction",
                "picaso": "native cloud-table placement and resort-rebin gas; exact-zero native probe separated from absorbing-formal spectrum",
                "petitradtrans": "native additional absorption callback; no fabricated layer-tau tensor",
            },
        },
        "scattering_freeze": {
            "single_scattering_albedo": 0.0,
            "rayleigh": False,
            "cloud_scattering": False,
            "delta_m": False,
        },
        "pilot": pilot,
        "execution": execution,
        "timings": {
            "pilot_total_wall_time_s": pilot["pilot_total_wall_time_s"],
            "post_pilot_matrix_and_assembly_wall_time_s": matrix_wall,
            "complete_launcher_wall_time_s": pilot["pilot_total_wall_time_s"]
            + matrix_wall,
        },
        "resources": {
            "pilot_peak_process_tree_rss_bytes": pilot["decision"]["peak_process_tree_rss_bytes"],
            "complete_matrix_peak_process_tree_rss_bytes": complete_matrix_peak_rss,
            "available_memory_bytes_at_pilot": pilot["decision"]["available_memory_bytes_at_decision"],
            "platform": platform.platform(),
        },
        "interpreters": {
            "robert": str(ROBERT_PYTHON),
            "picaso": str(PICASO_PYTHON),
            "petitradtrans": str(PRT_PYTHON),
        },
        "retired_picaso_opacity_sampling": True,
        "generated_products_policy": (
            "all raw workers, full-precision tensors, report, and integrity index remain "
            "under the ignored local Stage-7 tree for later archival release"
        ),
        **compact,
    }
    _write_json(product_root / "stage_7_report.json", report)
    _write_json(product_root / "stage_7_summary.json", {
        "stage": 7,
        "status": report["status"],
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "pilot_decision": pilot["decision"],
        "timings": report["timings"],
        "resources": report["resources"],
        "representative_cases": compact["representative_cases"],
    })
    integrity = _integrity_manifest(product_root)
    _write_json(product_root / "stage_7_integrity.json", integrity)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--product-root", type=Path, default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument(
        "--override-resource-gate",
        action="store_true",
        help=(
            "run the unchanged complete matrix after an explicitly authorized "
            "resource-gate override; the failed frozen decision remains recorded"
        ),
    )
    args = parser.parse_args()
    if os.path.realpath(sys.executable) != os.path.realpath(ROBERT_PYTHON):
        raise RuntimeError(f"Stage 7 must run with {ROBERT_PYTHON}")
    output_root = args.output_root.resolve()
    product_root = args.product_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    product_root.mkdir(parents=True, exist_ok=True)
    common = load_version_2_common_contract(COMMON_CONTRACT)
    paths = stage_3._opacity_paths()
    for species, asset in common.picaso_correlated_k_assets.items():
        if _sha256(stage_3.PICASO_CK_DIRECTORY / asset.filename) != asset.sha256:
            raise RuntimeError(f"frozen PICASO correlated-k checksum mismatch: {species}")
    pilot = _pilot(common, output_root, paths)
    if args.override_resource_gate:
        pilot["execution_override"] = {
            "authorized": True,
            "scope": "wall_time_only_complete_unchanged_matrix",
            "frozen_decision_preserved": True,
        }
        _write_json(output_root / "pilot" / "stage_7_pilot_report.json", pilot)
    if args.pilot_only or (
        not pilot["decision"]["continue_full_matrix"]
        and not args.override_resource_gate
    ):
        print(json.dumps(pilot, indent=2, sort_keys=True))
        return
    report = _run_complete(common, output_root, product_root, paths, pilot)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pilot": pilot["decision"],
                "observed_gate_values": report["observed_gate_values"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
