"""Run Stage 7 absorbing-cloud placement/extinction intercomparison.

Track A supplies identical gas and absorbing-cloud optical-depth contracts to
the three pure-absorption paths.  Track B supplies the same high-level cloud
definitions through each framework's native opacity/cloud interface.  Cloud
scattering, Rayleigh scattering, and delta-M scaling are explicitly disabled.
"""

from __future__ import annotations

import argparse
import gc
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from time import perf_counter
from typing import Any

import numpy as np
import psutil

from benchmark_emission_intercomparison_stage_5 import _opacity_paths, _timing_summary
from benchmark_emission_intercomparison_stages_1_3 import (
    DEFAULT_OUTPUT,
    DEFAULT_PICASO_DATABASE,
    DEFAULT_PICASO_PYTHON,
    DEFAULT_PICASO_REFERENCE,
    DEFAULT_PRT_INPUT,
    DEFAULT_PRT_PYTHON,
    DEFAULT_REPORTS,
    _load,
    _robert_metadata,
    _save_contract,
)
from emission_intercomparison_common import (
    SPECIES,
    STAGE_4_PROFILE_NAMES,
    STAGE_5_PERTURBATION_CENTERS_BAR,
    STAGE_7_CLOUD_OPTICAL_DEPTHS,
    STAGE_7_CLOUD_TOP_PRESSURES_BAR,
    STAGE_7_EXTINCTION_SLOPES,
    STAGE_7_REFERENCE_WAVELENGTH_MICRON,
    bin_mean,
    contribution_metrics,
    difference_metrics,
    eclipse_depth,
    normalize_contribution,
    r100_edges,
    sha256,
    stage_7_contract,
    temperature_localization,
    write_checksums,
    write_json,
)
from run_emission_intercomparison_stage_7_external import _formal_contribution


REPOSITORY = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("run_emission_intercomparison_stage_7_external.py")
ARCHIVED_CLOUD_CONTRACT = (
    REPOSITORY / "data/validation/end_to_end_cloud_parity/shared_physical_contract.npz"
)
ARCHIVED_CLOUD_OUTPUT = (
    REPOSITORY
    / "data/validation/end_to_end_cloud_parity/picaso_virga_independent_output.npz"
)
STAGE_4_REPORT = DEFAULT_REPORTS / "stage_4_report.json"
STAGE_5_REPORT = DEFAULT_REPORTS / "stage_5_report.json"
STAGE_5_TENSORS = DEFAULT_REPORTS / "stage_5_response_profiles.npz"
STAGE_6_REPORT = DEFAULT_REPORTS / "stage_6_report.json"
STAGE_6_TENSORS = DEFAULT_REPORTS / "stage_6_response_tensors.npz"
REVIEW_PATH = REPOSITORY / "docs/review/50_emission_intercomparison_stage_7.md"

RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
MODELS = ("robert", "picaso", "petitradtrans")
TRACKS = ("track_a_shared_tau", "track_b_native_cloud")
PILOT_CLOUD_LABELS = (
    "clear",
    "deck_tau1_top10mbar_slope+0",
    "deck_tau10_top1mbar_slope-4",
    "deck_tau100_top100mbar_slope+2",
    "archived_virga_mie_extinction",
)
BAND_WINDOWS_MICRON = {
    "H2O_1p4_band": (1.35, 1.50),
    "CH4_3p3_band": (3.15, 3.50),
    "CO2_4p3_band": (4.10, 4.50),
    "CO_4p7_band": (4.55, 4.90),
    "H2O_6p3_band": (5.80, 6.80),
    "CH4_7p7_band": (7.20, 8.20),
    "window_1p25": (1.18, 1.30),
    "window_3p9": (3.75, 4.00),
    "window_5p2": (5.00, 5.50),
    "window_10": (9.00, 11.00),
}

# Frozen before the pilot/full matrix.  Track B has no cross-framework gates.
TRACK_A_GATES = {
    "primary_absolute_spectrum_p95_symmetric_relative": 0.01,
    "primary_cloud_effect_p95_difference_over_pair_peak": 0.02,
    "primary_cloud_effect_eclipse_rms_ppm": 0.50,
    "primary_contribution_centroid_rms_dex": 0.05,
    "primary_contribution_profile_tv_p95": 0.05,
    "primary_cloud_response_profile_tv_p95": 0.08,
    "isothermal_cloud_effect_max_abs_ppm": 1.0e-10,
    "omega0_max_abs": 0.0,
    "80_to_160_absolute_spectrum_p95_symmetric_relative": 0.01,
    "80_to_160_cloud_effect_p95_difference_over_pair_peak": 0.02,
    "80_to_160_cloud_effect_eclipse_rms_ppm": 0.50,
    "80_to_160_contribution_centroid_rms_dex": 0.05,
    "80_to_160_contribution_profile_tv_p95": 0.05,
    "80_to_160_cloud_response_profile_tv_p95": 0.08,
}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _archived_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(ARCHIVED_CLOUD_CONTRACT, allow_pickle=False) as contract:
        pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    with np.load(ARCHIVED_CLOUD_OUTPUT, allow_pickle=False) as output:
        wavelength = np.asarray(output["wavelength_micron"], dtype=float)
        extinction_tau = np.asarray(output["cloud_tau"], dtype=float)
    return pressure_edges, wavelength, extinction_tau


def _robert_wavelength(input_data: Path) -> np.ndarray:
    from robert_exoplanets import CorrelatedKTable

    table = CorrelatedKTable.from_petitradtrans_hdf(
        _opacity_paths(input_data)["H2O"], species="H2O"
    )
    selected = (table.wavelength_micron >= 0.5) & (table.wavelength_micron <= 12.0)
    return np.sort(table.wavelength_micron[selected])


def _build_contract(
    resolution: int,
    input_data: Path,
    *,
    cloud_labels: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    pressure, wavelength, extinction = _archived_cloud()
    full = stage_7_contract(
        resolution,
        _robert_wavelength(input_data),
        archived_pressure_edges_bar=pressure,
        archived_wavelength_micron=wavelength,
        archived_extinction_tau=extinction,
    )
    if cloud_labels is None:
        return full
    indices = np.asarray(
        [int(np.flatnonzero(full["cloud_label"] == label)[0]) for label in cloud_labels]
    )
    return stage_7_contract(
        resolution,
        full["wavelength_micron"],
        archived_pressure_edges_bar=pressure,
        archived_wavelength_micron=wavelength,
        archived_extinction_tau=extinction,
        cloud_indices=indices,
    )


def _native_robert(
    contract: dict[str, np.ndarray], input_data: Path
) -> dict[str, np.ndarray]:
    from robert_exoplanets import (
        AtmosphereState,
        CiaTable,
        CloudOpticalProperties,
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

    paths = _opacity_paths(input_data)
    pressure_edges = contract["pressure_edges_bar"]
    pressure_centers = contract["pressure_centers_bar"]
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_centers,
        unit="bar",
        name="emission_intercomparison_stage_7_cells",
    )
    tables = {
        species: CorrelatedKTable.from_petitradtrans_hdf(paths[species], species=species)
        for species in SPECIES
    }
    first = tables["H2O"]
    wavelength = contract["wavelength_micron"]
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="pRT-R1000"
    )
    providers = {
        species: CorrelatedKOpacityProvider(
            {species: tables[species]},
            name=f"emission-intercomparison-stage7-{species}",
            interpolation="log_pressure_temperature_log_k",
        )
        for species in SPECIES
    }
    prepared_by_species = {
        species: providers[species].prepare(
            spectral_grid, pressure_grid, species=(species,)
        )
        for species in SPECIES
    }
    prepared = PreparedCorrelatedKOpacity(
        provider_name="pRT-HDF-four-species",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=SPECIES,
        g_samples=first.g_samples,
        g_weights=first.g_weights,
        cache_key=f"stage7-{pressure_centers.size}",
        metadata={"interpolation": "log_pressure_temperature_log_k"},
    )
    cia_tables = (
        CiaTable.from_petitradtrans_hdf(paths["H2-H2"], collision_pair="H2-H2"),
        CiaTable.from_petitradtrans_hdf(paths["H2-He"], collision_pair="H2-He"),
    )
    names = ("H2", "He", *SPECIES)
    vmr = contract["gas_vmr"][0]
    composition = {
        name: np.full(pressure_centers.size, vmr[index])
        for index, name in enumerate(names)
    }
    mean_molecular_weight = sum(
        composition[name] * mass
        for name, mass in zip(
            names,
            (2.01588, 4.002602, 18.01528, 28.0101, 44.0095, 16.04246),
            strict=True,
        )
    )
    flux = np.empty((contract["case_id"].size, wavelength.size))
    contribution = np.empty(
        (contract["case_id"].size, pressure_centers.size, wavelength.size)
    )
    runtime = np.empty(contract["case_id"].size)
    shared_gas_tau = np.empty(
        (len(STAGE_4_PROFILE_NAMES), pressure_centers.size, wavelength.size)
    )
    case_lookup = {
        (int(profile), int(cloud)): index
        for index, (profile, cloud) in enumerate(
            zip(contract["profile_index"], contract["case_cloud_index"], strict=True)
        )
    }
    for profile_index in range(len(STAGE_4_PROFILE_NAMES)):
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=contract["temperature_cells_by_profile_k"][profile_index],
            temperature_edges=contract["temperature_edges_by_profile_k"][profile_index],
            composition=composition,
            mean_molecular_weight=mean_molecular_weight,
        )
        evaluated = np.empty(
            (
                len(SPECIES),
                pressure_centers.size,
                wavelength.size,
                first.g_weights.size,
            )
        )
        for species_index, species in enumerate(SPECIES):
            evaluated[species_index] = (
                providers[species]
                .evaluate(atmosphere, prepared_by_species[species])
                .kcoeff[0]
            )
        opacity = EvaluatedCorrelatedKOpacity(
            prepared=prepared,
            kcoeff=evaluated,
            unit="cm^2/molecule",
            metadata={"source": "petitRADTRANS HDF5 correlated-k tables"},
        )
        gas_tau = assemble_gas_optical_depth(
            atmosphere, opacity, gravity_m_s2=15.0, gas_combination="random_overlap"
        )
        cia = [
            cia_optical_depth(
                gas_tau,
                table,
                coefficient_interpolation="log",
                temperature_extrapolation="clip",
                spectral_extrapolation="zero",
            )
            for table in cia_tables
        ]
        selected_clouds = contract["selected_cloud_index"]
        for cloud_index in selected_clouds:
            case_index = case_lookup[(profile_index, int(cloud_index))]
            cloud = CloudOpticalProperties(
                name=str(contract["cloud_label"][cloud_index]),
                extinction_tau=contract["cloud_extinction_tau"][cloud_index],
                spectral_grid=spectral_grid,
                pressure_grid=pressure_grid,
                single_scattering_albedo=0.0,
                asymmetry_factor=0.0,
                metadata={
                    "stage": "7",
                    "scattering": "explicitly_off",
                    "delta_m": "false",
                },
            )
            started = perf_counter()
            result = solve_emission(
                gas_tau,
                geometry=gauss_legendre_disk_geometry(n_mu=8),
                bottom_boundary="blackbody",
                additional_optical_depths=[*cia, cloud],
                multiple_scattering_backend="none",
            )
            runtime[case_index] = perf_counter() - started
            flux[case_index] = np.pi * np.asarray(result.radiance.values)
            layer = np.array(result.layer_contribution_radiance, copy=True)
            layer[-1] += np.asarray(result.bottom_contribution_radiance)
            contribution[case_index] = normalize_contribution(layer)
            if int(cloud_index) == 0:
                shared_gas_tau[profile_index] = np.sum(
                    np.asarray(result.total_optical_depth)
                    * first.g_weights[None, None, :],
                    axis=-1,
                )
        del atmosphere, evaluated, opacity, gas_tau, cia
        gc.collect()
    metadata = {
        **_robert_metadata(),
        "track": "native",
        "omega0": 0.0,
        "scattering_in_emission": False,
        "rayleigh_scattering": "not_supplied",
        "delta_m": False,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return {
        "case_id": contract["case_id"],
        "wavelength_micron": wavelength,
        "pressure_bar": pressure_centers,
        "flux_w_m2_m": flux,
        "normalized_contribution": contribution,
        "cloud_extinction_tau": contract["cloud_extinction_tau"]
        [contract["case_cloud_index"]],
        "shared_gas_tau": shared_gas_tau,
        "runtime_s": runtime,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _shared_robert(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from robert_exoplanets import planck_radiance_wavelength
    from robert_exoplanets.rt import integrate_thermal_emission_spectrum

    wavelength = contract["wavelength_micron"]
    total_tau = (
        contract["shared_gas_tau"][contract["profile_index"]]
        + contract["cloud_extinction_tau"][contract["case_cloud_index"]]
    )
    flux = np.empty((contract["case_id"].size, wavelength.size))
    contribution = np.empty(
        (contract["case_id"].size, total_tau.shape[1], wavelength.size)
    )
    runtime = np.empty(contract["case_id"].size)
    for case_index, tau in enumerate(total_tau):
        level_temperature = contract["temperature_edges_k"][case_index]
        layer_temperature = 0.5 * (level_temperature[:-1] + level_temperature[1:])
        layer_source = np.stack(
            [planck_radiance_wavelength(wavelength, value) for value in layer_temperature]
        )
        level_source = np.stack(
            [planck_radiance_wavelength(wavelength, value) for value in level_temperature]
        )
        started = perf_counter()
        result = integrate_thermal_emission_spectrum(
            tau[:, :, None],
            layer_source,
            np.array([1.0]),
            np.broadcast_to(
                1.0 / contract["emission_mu"][:, None],
                (contract["emission_mu"].size, tau.shape[0]),
            ),
            contract["disk_weights"],
            level_source_ordered=level_source,
            bottom_source=level_source[-1],
            backend="numpy",
        )
        runtime[case_index] = perf_counter() - started
        flux[case_index] = np.pi * result.radiance
        contribution[case_index] = _formal_contribution(
            wavelength,
            level_temperature,
            tau,
            np.array([1.0]),
            contract["emission_mu"],
            contract["disk_weights"],
        )
    metadata = {
        **_robert_metadata(),
        "track": "shared",
        "omega0": 0.0,
        "scattering_in_emission": False,
        "rayleigh_scattering": "explicitly_absent_from_frozen_tau",
        "delta_m": False,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return {
        "case_id": contract["case_id"],
        "wavelength_micron": wavelength,
        "pressure_bar": contract["pressure_centers_bar"],
        "flux_w_m2_m": flux,
        "normalized_contribution": contribution,
        "cloud_extinction_tau": contract["cloud_extinction_tau"]
        [contract["case_cloud_index"]],
        "runtime_s": runtime,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _run_worker(
    python: Path,
    model: str,
    track: str,
    contract_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        str(python),
        str(WORKER),
        model,
        track,
        str(contract_path),
        str(output_path),
    ]
    if model == "picaso" and track == "native":
        command.extend(
            [
                "--picaso-reference",
                str(args.picaso_reference),
                "--picaso-database",
                str(args.picaso_database),
                "--picaso-resample",
                str(args.picaso_resample),
            ]
        )
    if model == "petitradtrans" and track == "native":
        command.extend(["--input-data", str(args.prt_input)])
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_btl", "self")
    if model == "picaso":
        environment["picaso_refdata"] = str(args.picaso_reference.resolve())
    if model == "petitradtrans":
        home = contract_path.parent / ".petitradtrans-stage7-home"
        home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(home)
    stdout_path = output_path.with_suffix(".stdout.log")
    stderr_path = output_path.with_suffix(".stderr.log")
    started = perf_counter()
    peak_rss = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env=environment, cwd=REPOSITORY
        )
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [monitored, *monitored.children(recursive=True)]
                peak_rss = max(
                    peak_rss,
                    sum(item.memory_info().rss for item in processes if item.is_running()),
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, PermissionError):
                pass
            time.sleep(0.10)
        return_code = process.wait()
    wall_time = perf_counter() - started
    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    if return_code != 0:
        raise RuntimeError(
            f"Stage-7 {model}/{track} worker failed ({return_code})\n"
            f"stdout:\n{stdout_text}\nstderr:\n{stderr_text}"
        )
    return {
        "command": command,
        "wall_time_s": wall_time,
        "peak_rss_bytes": peak_rss,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "warnings": [line for line in stderr_text.splitlines() if line.strip()],
    }


def _effect_metrics(
    left: np.ndarray, right: np.ndarray, wavelength_micron: np.ndarray
) -> dict[str, float]:
    left_values = np.atleast_2d(np.asarray(left, dtype=float))
    right_values = np.atleast_2d(np.asarray(right, dtype=float))
    if left_values.shape != right_values.shape:
        raise ValueError("cloud-effect arrays must match")
    peak = np.maximum(
        np.max(np.abs(left_values), axis=1), np.max(np.abs(right_values), axis=1)
    )[:, None]
    scaled = np.divide(
        np.abs(left_values - right_values),
        peak,
        out=np.zeros_like(left_values),
        where=peak > 0.0,
    )
    eclipse = eclipse_depth(left_values - right_values, wavelength_micron) * 1.0e6
    return {
        "median_abs_difference_over_pair_peak": float(np.median(scaled)),
        "p95_abs_difference_over_pair_peak": float(np.percentile(scaled, 95.0)),
        "max_abs_difference_over_pair_peak": float(np.max(scaled)),
        "rms_eclipse_difference_ppm": float(np.sqrt(np.mean(eclipse**2))),
        "max_abs_eclipse_difference_ppm": float(np.max(np.abs(eclipse))),
    }


def _bin_payload(
    payload: dict[str, np.ndarray],
    contract: dict[str, np.ndarray],
    edges: np.ndarray,
) -> dict[str, np.ndarray | dict[str, Any]]:
    wavelength = np.asarray(payload["wavelength_micron"], dtype=float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    case_count = contract["case_id"].size
    layer_count = contract["pressure_centers_bar"].size
    flux = np.empty((case_count, centers.size))
    contribution = np.empty((case_count, layer_count, centers.size))
    cloud_tau = np.empty_like(contribution)
    for case_index in range(case_count):
        stacked = np.vstack(
            (
                payload["flux_w_m2_m"][case_index][None, :],
                payload["normalized_contribution"][case_index],
                payload["cloud_extinction_tau"][case_index],
            )
        )
        binned = bin_mean(wavelength, stacked, edges)
        flux[case_index] = binned[0]
        contribution[case_index] = normalize_contribution(
            np.clip(binned[1 : 1 + layer_count], 0.0, None)
        )
        cloud_tau[case_index] = np.clip(binned[1 + layer_count :], 0.0, None)

    effect = np.empty_like(flux)
    response = np.empty_like(contribution)
    raw_isothermal_max = 0.0
    for profile_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
        selected = np.flatnonzero(contract["profile_index"] == profile_index)
        clear = np.flatnonzero(
            (contract["profile_index"] == profile_index)
            & (contract["case_cloud_index"] == 0)
        )
        if clear.size != 1:
            raise ValueError("every profile requires one clear control")
        raw = flux[selected] - flux[clear[0]]
        if profile_name == "isothermal":
            raw_isothermal_max = max(raw_isothermal_max, float(np.max(np.abs(raw))))
            effect[selected] = 0.0
        else:
            effect[selected] = raw
        for case_index in selected:
            response[case_index] = normalize_contribution(
                np.abs(contribution[case_index] - contribution[clear[0]])
            )
    return {
        "flux_r100": flux,
        "eclipse_depth_r100": eclipse_depth(flux, centers),
        "cloud_effect_flux_r100": effect,
        "cloud_effect_eclipse_ppm_r100": eclipse_depth(effect, centers) * 1.0e6,
        "normalized_contribution_r100": contribution,
        "normalized_cloud_response_r100": response,
        "cloud_extinction_tau_r100": cloud_tau,
        "runtime_s": np.asarray(payload["runtime_s"], dtype=float),
        "metadata": json.loads(str(payload["metadata_json"])),
        "raw_isothermal_max_abs_flux_cancellation": raw_isothermal_max,
    }


def _pressure_summary(values: np.ndarray, pressure_bar: np.ndarray) -> dict[str, float]:
    normalized = normalize_contribution(np.asarray(values, dtype=float))
    log_pressure = np.log10(pressure_bar)[:, None]
    centroid = 10.0 ** np.sum(normalized * log_pressure, axis=0)
    peak = pressure_bar[np.argmax(normalized, axis=0)]
    return {
        "centroid_pressure_median_bar": float(np.median(centroid)),
        "centroid_pressure_p05_bar": float(np.percentile(centroid, 5.0)),
        "centroid_pressure_p95_bar": float(np.percentile(centroid, 95.0)),
        "peak_pressure_median_bar": float(np.median(peak)),
        "peak_pressure_p05_bar": float(np.percentile(peak, 5.0)),
        "peak_pressure_p95_bar": float(np.percentile(peak, 95.0)),
    }


def _band_window_metrics(
    effect_flux: np.ndarray, wavelength: np.ndarray
) -> dict[str, dict[str, float]]:
    effect_ppm = eclipse_depth(effect_flux, wavelength) * 1.0e6
    output = {}
    for name, (lower, upper) in BAND_WINDOWS_MICRON.items():
        selected = (wavelength >= lower) & (wavelength <= upper)
        output[name] = {
            "mean_cloud_effect_ppm": float(np.mean(effect_ppm[selected])),
            "rms_cloud_effect_ppm": float(np.sqrt(np.mean(effect_ppm[selected] ** 2))),
            "minimum_cloud_effect_ppm": float(np.min(effect_ppm[selected])),
            "maximum_cloud_effect_ppm": float(np.max(effect_ppm[selected])),
        }
    return output


def _native_extrema(
    payload: dict[str, np.ndarray], contract: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    wavelength = payload["wavelength_micron"]
    flux = payload["flux_w_m2_m"]
    output = {}
    for case_index, case_id in enumerate(contract["case_id"]):
        profile_index = int(contract["profile_index"][case_index])
        clear = int(
            np.flatnonzero(
                (contract["profile_index"] == profile_index)
                & (contract["case_cloud_index"] == 0)
            )[0]
        )
        effect = eclipse_depth(flux[case_index] - flux[clear], wavelength) * 1.0e6
        if STAGE_4_PROFILE_NAMES[profile_index] == "isothermal":
            effect = np.zeros_like(effect)
        minimum = int(np.argmin(effect))
        maximum = int(np.argmax(effect))
        output[str(case_id)] = {
            "minimum_cloud_effect_ppm": float(effect[minimum]),
            "minimum_wavelength_micron": float(wavelength[minimum]),
            "maximum_cloud_effect_ppm": float(effect[maximum]),
            "maximum_wavelength_micron": float(wavelength[maximum]),
        }
    return output


def _write_robert_worker_output(
    path: Path, payload: dict[str, np.ndarray]
) -> None:
    np.savez_compressed(path, **payload)


def _reuse_resolution(
    args: argparse.Namespace, resolution: int
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
    list[Path],
]:
    """Reanalyze complete preserved Stage-7 worker artifacts."""

    stage_dir = args.output_root / "stage_7"
    label = f"main_L{resolution}"
    native_path = stage_dir / f"contract_track_b_{label}.npz"
    shared_path = stage_dir / f"contract_track_a_{label}.npz"
    contract = _load(native_path)
    edges = r100_edges()
    outputs: dict[str, dict[str, dict[str, Any]]] = {track: {} for track in TRACKS}
    execution: dict[str, Any] = {track: {} for track in TRACKS}
    extrema: dict[str, Any] = {track: {} for track in TRACKS}
    paths = [native_path, shared_path]
    for track, short in (
        ("track_b_native_cloud", "track_b"),
        ("track_a_shared_tau", "track_a"),
    ):
        for model in MODELS:
            output_path = stage_dir / f"{short}_{model}_{label}.npz"
            payload = _load(output_path)
            metadata = json.loads(str(payload["metadata_json"]))
            stderr_path = output_path.with_suffix(".stderr.log")
            stdout_path = output_path.with_suffix(".stdout.log")
            warnings = (
                [
                    line
                    for line in stderr_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if stderr_path.is_file()
                else []
            )
            execution[track][model] = {
                "wall_time_s": float(
                    metadata.get("wall_time_s", np.sum(payload["runtime_s"]))
                ),
                "peak_rss_bytes": int(metadata.get("peak_rss_bytes", 0)),
                "warnings": warnings,
                "reused_complete_worker_artifact": True,
            }
            if stderr_path.is_file():
                execution[track][model]["stderr_sha256"] = sha256(stderr_path)
                paths.append(stderr_path)
            if stdout_path.is_file():
                execution[track][model]["stdout_sha256"] = sha256(stdout_path)
                paths.append(stdout_path)
            extrema[track][model] = _native_extrema(payload, contract)
            outputs[track][model] = _bin_payload(payload, contract, edges)
            paths.append(output_path)
            del payload
            gc.collect()
    return outputs, execution, {"contract": contract, "native_extrema": extrema}, paths


def _run_resolution(
    args: argparse.Namespace,
    resolution: int,
    *,
    pilot: bool,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
    list[Path],
]:
    if bool(getattr(args, "reuse_existing", False)) and not pilot:
        return _reuse_resolution(args, resolution)
    stage_dir = args.output_root / "stage_7"
    stage_dir.mkdir(parents=True, exist_ok=True)
    contract = _build_contract(
        resolution,
        args.prt_input,
        cloud_labels=PILOT_CLOUD_LABELS if pilot else None,
    )
    label = f"{'pilot' if pilot else 'main'}_L{resolution}"
    native_path = stage_dir / f"contract_track_b_{label}.npz"
    _save_contract(native_path, {**contract, "track": np.array("track_b_native_cloud")})
    paths = [native_path]
    edges = r100_edges()
    outputs: dict[str, dict[str, dict[str, Any]]] = {track: {} for track in TRACKS}
    execution: dict[str, Any] = {track: {} for track in TRACKS}
    extrema: dict[str, Any] = {track: {} for track in TRACKS}

    for model, python in (
        ("picaso", args.picaso_python),
        ("petitradtrans", args.prt_python),
    ):
        output_path = stage_dir / f"track_b_{model}_{label}.npz"
        execution["track_b_native_cloud"][model] = _run_worker(
            python, model, "native", native_path, output_path, args
        )
        payload = _load(output_path)
        worker_metadata = json.loads(str(payload["metadata_json"]))
        execution["track_b_native_cloud"][model]["peak_rss_bytes"] = max(
            int(execution["track_b_native_cloud"][model]["peak_rss_bytes"]),
            int(worker_metadata.get("peak_rss_bytes", 0)),
        )
        extrema["track_b_native_cloud"][model] = _native_extrema(payload, contract)
        outputs["track_b_native_cloud"][model] = _bin_payload(payload, contract, edges)
        paths.extend(
            [
                output_path,
                output_path.with_suffix(".stdout.log"),
                output_path.with_suffix(".stderr.log"),
            ]
        )
        del payload
        gc.collect()

    started = perf_counter()
    robert_native = _native_robert(contract, args.prt_input)
    execution["track_b_native_cloud"]["robert"] = {
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": int(json.loads(str(robert_native["metadata_json"]))["peak_rss_bytes"]),
        "warnings": [],
    }
    extrema["track_b_native_cloud"]["robert"] = _native_extrema(
        robert_native, contract
    )
    outputs["track_b_native_cloud"]["robert"] = _bin_payload(
        robert_native, contract, edges
    )
    robert_native_path = stage_dir / f"track_b_robert_{label}.npz"
    _write_robert_worker_output(robert_native_path, robert_native)
    paths.append(robert_native_path)

    shared_contract = {
        **contract,
        "track": np.array("track_a_shared_tau"),
        "shared_gas_tau": robert_native["shared_gas_tau"],
    }
    shared_path = stage_dir / f"contract_track_a_{label}.npz"
    _save_contract(shared_path, shared_contract)
    paths.append(shared_path)
    del robert_native
    gc.collect()

    for model, python in (
        ("picaso", args.picaso_python),
        ("petitradtrans", args.prt_python),
    ):
        output_path = stage_dir / f"track_a_{model}_{label}.npz"
        execution["track_a_shared_tau"][model] = _run_worker(
            python, model, "shared", shared_path, output_path, args
        )
        payload = _load(output_path)
        worker_metadata = json.loads(str(payload["metadata_json"]))
        execution["track_a_shared_tau"][model]["peak_rss_bytes"] = max(
            int(execution["track_a_shared_tau"][model]["peak_rss_bytes"]),
            int(worker_metadata.get("peak_rss_bytes", 0)),
        )
        extrema["track_a_shared_tau"][model] = _native_extrema(payload, contract)
        outputs["track_a_shared_tau"][model] = _bin_payload(payload, contract, edges)
        paths.extend(
            [
                output_path,
                output_path.with_suffix(".stdout.log"),
                output_path.with_suffix(".stderr.log"),
            ]
        )
        del payload
        gc.collect()

    started = perf_counter()
    robert_shared = _shared_robert(shared_contract)
    execution["track_a_shared_tau"]["robert"] = {
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": int(json.loads(str(robert_shared["metadata_json"]))["peak_rss_bytes"]),
        "warnings": [],
    }
    extrema["track_a_shared_tau"]["robert"] = _native_extrema(
        robert_shared, contract
    )
    outputs["track_a_shared_tau"]["robert"] = _bin_payload(
        robert_shared, contract, edges
    )
    robert_shared_path = stage_dir / f"track_a_robert_{label}.npz"
    _write_robert_worker_output(robert_shared_path, robert_shared)
    paths.append(robert_shared_path)
    del robert_shared
    gc.collect()
    return outputs, execution, {"contract": contract, "native_extrema": extrema}, paths


def _pairwise_case_report(
    outputs: dict[str, dict[str, Any]],
    contract: dict[str, np.ndarray],
    wavelength: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for case_index, case_id in enumerate(contract["case_id"]):
        case = {
            "profile": str(
                contract["profile_name"][int(contract["profile_index"][case_index])]
            ),
            "cloud": str(contract["cloud_label"][contract["case_cloud_index"][case_index]]),
            "pairwise": {},
            "per_model": {},
        }
        for left, right in combinations(MODELS, 2):
            pair = f"{left}__{right}"
            pair_report = {
                "absolute_spectrum_r100": difference_metrics(
                    outputs[left]["flux_r100"][case_index],
                    outputs[right]["flux_r100"][case_index],
                    wavelength,
                ),
                "cloud_effect_r100": _effect_metrics(
                    outputs[left]["cloud_effect_flux_r100"][case_index],
                    outputs[right]["cloud_effect_flux_r100"][case_index],
                    wavelength,
                ),
                "contribution_r100": contribution_metrics(
                    outputs[left]["normalized_contribution_r100"][case_index],
                    outputs[right]["normalized_contribution_r100"][case_index],
                    contract["pressure_centers_bar"],
                ),
                "cloud_response_r100": contribution_metrics(
                    outputs[left]["normalized_cloud_response_r100"][case_index],
                    outputs[right]["normalized_cloud_response_r100"][case_index],
                    contract["pressure_centers_bar"],
                ),
            }
            if int(contract["case_cloud_index"][case_index]) != 0:
                pair_report["cloud_placement_r100"] = contribution_metrics(
                    outputs[left]["cloud_extinction_tau_r100"][case_index],
                    outputs[right]["cloud_extinction_tau_r100"][case_index],
                    contract["pressure_centers_bar"],
                )
            case["pairwise"][pair] = pair_report
        for model in MODELS:
            case["per_model"][model] = {
                "contribution_pressure": _pressure_summary(
                    outputs[model]["normalized_contribution_r100"][case_index],
                    contract["pressure_centers_bar"],
                ),
                "cloud_response_pressure": _pressure_summary(
                    outputs[model]["normalized_cloud_response_r100"][case_index],
                    contract["pressure_centers_bar"],
                ),
                "band_window_cloud_effect": _band_window_metrics(
                    outputs[model]["cloud_effect_flux_r100"][case_index], wavelength
                ),
            }
            if int(contract["case_cloud_index"][case_index]) != 0:
                case["per_model"][model]["cloud_extinction_pressure"] = (
                    _pressure_summary(
                        outputs[model]["cloud_extinction_tau_r100"][case_index],
                        contract["pressure_centers_bar"],
                    )
                )
        report[str(case_id)] = case
    return report


def _regrid_profile(
    values: np.ndarray, source_edges: np.ndarray, target_edges: np.ndarray
) -> np.ndarray:
    normalized = normalize_contribution(values)
    source_log = np.log(source_edges)
    target_log = np.log(target_edges)
    output = np.zeros((target_edges.size - 1, normalized.shape[1]))
    for source_index in range(source_edges.size - 1):
        width = source_log[source_index + 1] - source_log[source_index]
        low = np.maximum(target_log[:-1], source_log[source_index])
        high = np.minimum(target_log[1:], source_log[source_index + 1])
        fraction = np.clip(high - low, 0.0, None) / width
        output += fraction[:, None] * normalized[source_index][None, :]
    return normalize_contribution(output)


def _convergence_report(
    by_resolution: dict[int, dict[str, dict[str, dict[str, Any]]]],
    contracts: dict[int, dict[str, np.ndarray]],
    wavelength: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for coarse, fine in ((40, 80), (80, 160)):
        label = f"{coarse}_to_{fine}"
        target_edges = contracts[fine]["pressure_edges_bar"]
        target_pressure = contracts[fine]["pressure_centers_bar"]
        output[label] = {}
        for track in TRACKS:
            output[label][track] = {}
            for model in MODELS:
                cases = {}
                for case_index, case_id in enumerate(contracts[coarse]["case_id"]):
                    coarse_payload = by_resolution[coarse][track][model]
                    fine_payload = by_resolution[fine][track][model]
                    coarse_contribution = _regrid_profile(
                        coarse_payload["normalized_contribution_r100"][case_index],
                        contracts[coarse]["pressure_edges_bar"],
                        target_edges,
                    )
                    coarse_response = _regrid_profile(
                        coarse_payload["normalized_cloud_response_r100"][case_index],
                        contracts[coarse]["pressure_edges_bar"],
                        target_edges,
                    )
                    cases[str(case_id)] = {
                        "absolute_spectrum_r100": difference_metrics(
                            coarse_payload["flux_r100"][case_index],
                            fine_payload["flux_r100"][case_index],
                            wavelength,
                        ),
                        "cloud_effect_r100": _effect_metrics(
                            coarse_payload["cloud_effect_flux_r100"][case_index],
                            fine_payload["cloud_effect_flux_r100"][case_index],
                            wavelength,
                        ),
                        "contribution_r100": contribution_metrics(
                            coarse_contribution,
                            fine_payload["normalized_contribution_r100"][case_index],
                            target_pressure,
                        ),
                        "cloud_response_r100": contribution_metrics(
                            coarse_response,
                            fine_payload["normalized_cloud_response_r100"][case_index],
                            target_pressure,
                        ),
                    }
                output[label][track][model] = cases
    return output


def _project_to_stage56_centers(
    values: np.ndarray, pressure_bar: np.ndarray
) -> np.ndarray:
    localization = np.stack(
        [temperature_localization(pressure_bar, center) for center in STAGE_5_PERTURBATION_CENTERS_BAR]
    )
    return normalize_contribution(localization @ values)


def _prior_stage_relations(
    by_resolution: dict[int, dict[str, dict[str, dict[str, Any]]]],
    contracts: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    stage5 = _load(STAGE_5_TENSORS)
    stage6 = _load(STAGE_6_TENSORS)
    output: dict[str, Any] = {
        "interpretation": (
            "Stage-7 cloud contribution shifts are compared on the six Stage-5/6 "
            "localization centres; contribution, temperature response, composition "
            "response, and cloud response are related pressure diagnostics, not identities."
        )
    }
    for resolution in RESOLUTIONS:
        output[str(resolution)] = {}
        contract = contracts[resolution]
        selected_clouds = {
            "deck_tau10_top10mbar_slope+0",
            "archived_virga_mie_extinction",
        }
        for track in TRACKS:
            prior_track = (
                "track_a_shared_tau" if track == "track_a_shared_tau" else "track_b_native_opacity"
            )
            output[str(resolution)][track] = {}
            for model in MODELS:
                model_output = {}
                stage4_reference = stage6[
                    f"stage4_projected_contribution_L{resolution}_{model}"
                ]
                stage5_reference = stage5[
                    f"normalized_vertical_response_{prior_track}_L{resolution}_{model}"
                ]
                stage6_reference = np.mean(
                    stage6[
                        f"normalized_vertical_response_{prior_track}_L{resolution}_{model}"
                    ],
                    axis=1,
                )
                stage6_reference = np.stack(
                    [normalize_contribution(item) for item in stage6_reference]
                )
                for case_index, case_id in enumerate(contract["case_id"]):
                    cloud_label = str(
                        contract["cloud_label"][contract["case_cloud_index"][case_index]]
                    )
                    if cloud_label not in selected_clouds:
                        continue
                    profile_index = int(contract["profile_index"][case_index])
                    projected = _project_to_stage56_centers(
                        by_resolution[resolution][track][model][
                            "normalized_cloud_response_r100"
                        ][case_index],
                        contract["pressure_centers_bar"],
                    )
                    model_output[str(case_id)] = {
                        "versus_stage4_projected_contribution": contribution_metrics(
                            projected,
                            stage4_reference[profile_index],
                            np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
                        ),
                        "versus_stage5_temperature_response": contribution_metrics(
                            projected,
                            stage5_reference[profile_index],
                            np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
                        ),
                        "versus_stage6_mean_composition_response": contribution_metrics(
                            projected,
                            stage6_reference[profile_index],
                            np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
                        ),
                    }
                output[str(resolution)][track][model] = model_output
    return output


def _gate_results(
    by_resolution: dict[int, dict[str, dict[str, dict[str, Any]]]],
    contracts: dict[int, dict[str, np.ndarray]],
    convergence: dict[str, Any],
    wavelength: np.ndarray,
) -> tuple[dict[str, Any], bool]:
    primary = by_resolution[PRIMARY_RESOLUTION]["track_a_shared_tau"]
    primary_spectrum = []
    primary_effect = []
    primary_contribution = []
    primary_response = []
    for case_index in range(contracts[PRIMARY_RESOLUTION]["case_id"].size):
        for left, right in combinations(MODELS, 2):
            primary_spectrum.append(
                difference_metrics(
                    primary[left]["flux_r100"][case_index],
                    primary[right]["flux_r100"][case_index],
                    wavelength,
                )
            )
            primary_effect.append(
                _effect_metrics(
                    primary[left]["cloud_effect_flux_r100"][case_index],
                    primary[right]["cloud_effect_flux_r100"][case_index],
                    wavelength,
                )
            )
            primary_contribution.append(
                contribution_metrics(
                    primary[left]["normalized_contribution_r100"][case_index],
                    primary[right]["normalized_contribution_r100"][case_index],
                    contracts[PRIMARY_RESOLUTION]["pressure_centers_bar"],
                )
            )
            primary_response.append(
                contribution_metrics(
                    primary[left]["normalized_cloud_response_r100"][case_index],
                    primary[right]["normalized_cloud_response_r100"][case_index],
                    contracts[PRIMARY_RESOLUTION]["pressure_centers_bar"],
                )
            )
    convergence_cases = [
        case
        for model in convergence["80_to_160"]["track_a_shared_tau"].values()
        for case in model.values()
    ]
    isothermal_ppm = max(
        float(
            np.max(
                np.abs(
                    primary[model]["cloud_effect_eclipse_ppm_r100"][
                        contracts[PRIMARY_RESOLUTION]["profile_index"] == 0
                    ]
                )
            )
        )
        for model in MODELS
    )
    observed = {
        "primary_absolute_spectrum_p95_symmetric_relative": max(
            item["p95_abs_symmetric_relative"] for item in primary_spectrum
        ),
        "primary_cloud_effect_p95_difference_over_pair_peak": max(
            item["p95_abs_difference_over_pair_peak"] for item in primary_effect
        ),
        "primary_cloud_effect_eclipse_rms_ppm": max(
            item["rms_eclipse_difference_ppm"] for item in primary_effect
        ),
        "primary_contribution_centroid_rms_dex": max(
            item["centroid_pressure_rms_difference_dex"] for item in primary_contribution
        ),
        "primary_contribution_profile_tv_p95": max(
            item["profile_total_variation_p95"] for item in primary_contribution
        ),
        "primary_cloud_response_profile_tv_p95": max(
            item["profile_total_variation_p95"] for item in primary_response
        ),
        "isothermal_cloud_effect_max_abs_ppm": isothermal_ppm,
        "omega0_max_abs": float(
            np.max(np.abs(contracts[PRIMARY_RESOLUTION]["cloud_single_scattering_albedo"]))
        ),
        "80_to_160_absolute_spectrum_p95_symmetric_relative": max(
            item["absolute_spectrum_r100"]["p95_abs_symmetric_relative"]
            for item in convergence_cases
        ),
        "80_to_160_cloud_effect_p95_difference_over_pair_peak": max(
            item["cloud_effect_r100"]["p95_abs_difference_over_pair_peak"]
            for item in convergence_cases
        ),
        "80_to_160_cloud_effect_eclipse_rms_ppm": max(
            item["cloud_effect_r100"]["rms_eclipse_difference_ppm"]
            for item in convergence_cases
        ),
        "80_to_160_contribution_centroid_rms_dex": max(
            item["contribution_r100"]["centroid_pressure_rms_difference_dex"]
            for item in convergence_cases
        ),
        "80_to_160_contribution_profile_tv_p95": max(
            item["contribution_r100"]["profile_total_variation_p95"]
            for item in convergence_cases
        ),
        "80_to_160_cloud_response_profile_tv_p95": max(
            item["cloud_response_r100"]["profile_total_variation_p95"]
            for item in convergence_cases
        ),
    }
    results = {
        name: {
            "threshold": threshold,
            "observed": observed[name],
            "passed": bool(observed[name] <= threshold),
        }
        for name, threshold in TRACK_A_GATES.items()
    }
    return results, all(item["passed"] for item in results.values())


def _artifact(
    by_resolution: dict[int, dict[str, dict[str, dict[str, Any]]]],
    contracts: dict[int, dict[str, np.ndarray]],
    wavelength: np.ndarray,
) -> dict[str, np.ndarray]:
    def pack_probability(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        quantized = np.rint(clipped * 4095.0).astype(np.uint16).ravel()
        original_size = quantized.size
        if original_size % 2:
            quantized = np.append(quantized, np.uint16(0))
        left = quantized[0::2]
        right = quantized[1::2]
        packed = np.empty(left.size * 3, dtype=np.uint8)
        packed[0::3] = (left & 0xFF).astype(np.uint8)
        packed[1::3] = (((left >> 8) & 0x0F) | ((right & 0x0F) << 4)).astype(
            np.uint8
        )
        packed[2::3] = (right >> 4).astype(np.uint8)
        shape = np.asarray((*clipped.shape, original_size), dtype=np.int64)
        return packed, shape

    def store_profile(name: str, values: np.ndarray) -> None:
        packed, shape = pack_probability(values)
        artifact[f"{name}_uint12_packed"] = packed
        artifact[f"{name}_shape"] = shape

    artifact: dict[str, np.ndarray] = {
        "schema_version": np.array(1),
        "stage": np.array(7),
        "wavelength_r100_micron": wavelength,
        "profile_name": np.asarray(STAGE_4_PROFILE_NAMES),
        "cloud_label": contracts[PRIMARY_RESOLUTION]["cloud_label"],
        "cloud_kind": contracts[PRIMARY_RESOLUTION]["cloud_kind"],
        "cloud_optical_depth_at_reference": contracts[PRIMARY_RESOLUTION][
            "cloud_optical_depth_at_reference"
        ],
        "cloud_top_pressure_bar": contracts[PRIMARY_RESOLUTION]["cloud_top_pressure_bar"],
        "cloud_extinction_slope": contracts[PRIMARY_RESOLUTION]["cloud_extinction_slope"],
        "reference_wavelength_micron": np.array(STAGE_7_REFERENCE_WAVELENGTH_MICRON),
        "single_scattering_albedo": np.zeros(
            contracts[PRIMARY_RESOLUTION]["cloud_label"].size
        ),
        "normalized_profile_quantization_scale": np.array(4095.0),
        "normalized_profile_max_abs_quantization_error": np.array(
            0.5 / 4095.0
        ),
        "normalized_profile_storage": np.array(
            "packed uint12 pairs in little nibble order; unpack and divide by scale"
        ),
    }
    for resolution in RESOLUTIONS:
        artifact[f"pressure_edges_L{resolution}_bar"] = contracts[resolution][
            "pressure_edges_bar"
        ]
        artifact[f"pressure_centers_L{resolution}_bar"] = contracts[resolution][
            "pressure_centers_bar"
        ]
        artifact[f"case_id_L{resolution}"] = contracts[resolution]["case_id"]
        artifact[f"profile_index_L{resolution}"] = contracts[resolution]["profile_index"]
        artifact[f"cloud_index_L{resolution}"] = contracts[resolution]["case_cloud_index"]
        for track in TRACKS:
            for model in MODELS:
                prefix = f"{track}_L{resolution}_{model}"
                payload = by_resolution[resolution][track][model]
                artifact[f"flux_{prefix}_w_m2_m"] = np.asarray(
                    payload["flux_r100"], dtype=np.float32
                )
                artifact[f"eclipse_depth_{prefix}"] = np.asarray(
                    payload["eclipse_depth_r100"], dtype=np.float32
                )
                artifact[f"cloud_effect_flux_{prefix}_w_m2_m"] = np.asarray(
                    payload["cloud_effect_flux_r100"], dtype=np.float32
                )
                artifact[f"cloud_effect_eclipse_{prefix}_ppm"] = np.asarray(
                    payload["cloud_effect_eclipse_ppm_r100"], dtype=np.float32
                )
                if track == "track_b_native_cloud":
                    store_profile(
                        f"normalized_contribution_{prefix}",
                        payload["normalized_contribution_r100"],
                    )
                    store_profile(
                        f"normalized_cloud_response_{prefix}",
                        payload["normalized_cloud_response_r100"],
                    )
                    first_profile = contracts[resolution]["profile_index"] == 0
                    artifact[f"cloud_extinction_tau_{prefix}"] = np.asarray(
                        payload["cloud_extinction_tau_r100"][first_profile],
                        dtype=np.float32,
                    )
        shared = by_resolution[resolution]["track_a_shared_tau"]["robert"]
        shared_prefix = f"track_a_shared_tau_L{resolution}_identical_formal_profile"
        store_profile(
            f"normalized_contribution_{shared_prefix}",
            shared["normalized_contribution_r100"],
        )
        store_profile(
            f"normalized_cloud_response_{shared_prefix}",
            shared["normalized_cloud_response_r100"],
        )
        first_profile = contracts[resolution]["profile_index"] == 0
        artifact[f"cloud_extinction_tau_track_a_shared_tau_L{resolution}"] = (
            np.asarray(
                shared["cloud_extinction_tau_r100"][first_profile], dtype=np.float32
            )
        )
    return artifact


def _pilot_decision(
    execution: dict[str, Any], case_count: int
) -> dict[str, Any]:
    measured_wall = sum(
        float(details["wall_time_s"])
        for track in execution.values()
        for details in track.values()
    )
    full_case_count = len(STAGE_4_PROFILE_NAMES) * (
        1
        + len(STAGE_7_CLOUD_OPTICAL_DEPTHS)
        * len(STAGE_7_CLOUD_TOP_PRESSURES_BAR)
        * len(STAGE_7_EXTINCTION_SLOPES)
        + 1
    )
    resolution_work = sum(RESOLUTIONS) / PRIMARY_RESOLUTION
    projection = measured_wall * full_case_count / case_count * resolution_work
    peak = max(
        int(details.get("peak_rss_bytes", 0))
        for track in execution.values()
        for details in track.values()
    )
    try:
        available = int(psutil.virtual_memory().available)
    except (OSError, PermissionError):
        available = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    wall_safe = projection <= 2.0 * 3600.0
    memory_safe = peak <= 0.60 * available
    return {
        "pilot_case_count": case_count,
        "full_case_count_per_resolution": full_case_count,
        "measured_sum_worker_wall_time_s": measured_wall,
        "projection_method": (
            "pilot summed worker wall time multiplied by the full/pilot case ratio "
            "and by sum(40,80,160)/80; intentionally conservative about setup costs"
        ),
        "projected_complete_wall_time_s": projection,
        "wall_time_limit_s": 7200.0,
        "peak_measured_process_tree_rss_bytes": peak,
        "available_memory_at_decision_bytes": available,
        "memory_limit_fraction_of_available": 0.60,
        "wall_time_safe": wall_safe,
        "memory_safe": memory_safe,
        "continue_full_matrix": bool(wall_safe and memory_safe),
    }


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    stage_dir = args.output_root / "stage_7"
    stage_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = stage_dir / "stage_7_pilot_report.json"
    if args.reuse_existing:
        pilot_record = json.loads(pilot_path.read_text(encoding="utf-8"))
    else:
        pilot_started = perf_counter()
        pilot_outputs, pilot_execution, pilot_context, pilot_paths = _run_resolution(
            args, PRIMARY_RESOLUTION, pilot=True
        )
        pilot_decision = _pilot_decision(
            pilot_execution, pilot_context["contract"]["case_id"].size
        )
        pilot_record = {
            "schema_version": 1,
            "stage": 7,
            "kind": "representative_cross_framework_primary_resolution_pilot",
            "resolution": PRIMARY_RESOLUTION,
            "cloud_labels": list(PILOT_CLOUD_LABELS),
            "profiles": list(STAGE_4_PROFILE_NAMES),
            "execution": pilot_execution,
            "decision": pilot_decision,
            "wall_time_s": perf_counter() - pilot_started,
            "artifact_checksums": {
                str(path.relative_to(args.output_root)): sha256(path)
                for path in pilot_paths
            },
        }
        write_json(pilot_path, pilot_record)
        if args.pilot_only or not pilot_decision["continue_full_matrix"]:
            report = {
                "schema_version": 1,
                "stage": 7,
                "status": (
                    "pilot_only" if args.pilot_only else "stopped_after_unsafe_pilot"
                ),
                "pilot": pilot_record,
            }
            return report, None
        del pilot_outputs
        gc.collect()

    started = perf_counter()
    by_resolution: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    contracts: dict[int, dict[str, np.ndarray]] = {}
    per_resolution: dict[str, Any] = {}
    artifact_checksums: dict[str, str] = {}
    execution_all: dict[str, Any] = {}
    native_extrema: dict[str, Any] = {}
    wavelength = np.sqrt(r100_edges()[:-1] * r100_edges()[1:])
    for resolution in RESOLUTIONS:
        outputs, execution, context, paths = _run_resolution(
            args, resolution, pilot=False
        )
        by_resolution[resolution] = outputs
        contracts[resolution] = context["contract"]
        execution_all[str(resolution)] = execution
        native_extrema[str(resolution)] = context["native_extrema"]
        per_resolution[str(resolution)] = {
            "cases": {
                track: _pairwise_case_report(
                    outputs[track], context["contract"], wavelength
                )
                for track in TRACKS
            },
            "execution": {
                track: {
                    model: {
                        **execution[track][model],
                        "worker_metadata": outputs[track][model]["metadata"],
                        "timing_summary": _timing_summary(
                            outputs[track][model]["runtime_s"]
                        ),
                        "raw_case_timings_s": outputs[track][model][
                            "runtime_s"
                        ].tolist(),
                        "raw_isothermal_max_abs_flux_cancellation": outputs[track][
                            model
                        ]["raw_isothermal_max_abs_flux_cancellation"],
                    }
                    for model in MODELS
                }
                for track in TRACKS
            },
            "native_resolution_extrema": context["native_extrema"],
        }
        for path in paths:
            artifact_checksums[str(path.relative_to(args.output_root))] = sha256(path)
        gc.collect()

    convergence = _convergence_report(by_resolution, contracts, wavelength)
    relations = _prior_stage_relations(by_resolution, contracts)
    gates, passed = _gate_results(by_resolution, contracts, convergence, wavelength)
    response_artifact = _artifact(by_resolution, contracts, wavelength)

    input_paths = _opacity_paths(args.prt_input)
    input_paths.update(
        {
            "PICASO-database": args.picaso_database,
            "PICASO-reference-config": args.picaso_reference / "config.json",
            "PICASO-reference-version": args.picaso_reference / "version.md",
            "archived-cloud-contract": ARCHIVED_CLOUD_CONTRACT,
            "archived-cloud-output": ARCHIVED_CLOUD_OUTPUT,
            "Stage-4-report": STAGE_4_REPORT,
            "Stage-5-report": STAGE_5_REPORT,
            "Stage-5-response-artifact": STAGE_5_TENSORS,
            "Stage-6-report": STAGE_6_REPORT,
            "Stage-6-response-artifact": STAGE_6_TENSORS,
        }
    )
    source_paths = {
        "stage_7_launcher": Path(__file__),
        "stage_7_external_worker": WORKER,
        "shared_contracts_and_metrics": Path(__file__).with_name(
            "emission_intercomparison_common.py"
        ),
        "predeclared_review": REVIEW_PATH,
    }
    report = {
        "schema_version": 1,
        "stage": 7,
        "status": "passed" if passed else "failed_track_a_gates",
        "orchestrator": _robert_metadata(),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "profiles": list(STAGE_4_PROFILE_NAMES),
        "cloud_matrix": {
            "optical_depth_at_reference": list(STAGE_7_CLOUD_OPTICAL_DEPTHS),
            "cloud_top_pressure_bar": list(STAGE_7_CLOUD_TOP_PRESSURES_BAR),
            "extinction_slope": list(STAGE_7_EXTINCTION_SLOPES),
            "reference_wavelength_micron": STAGE_7_REFERENCE_WAVELENGTH_MICRON,
            "vertical_distribution": "uniform_d_tau_d_log_pressure_below_cloud_top",
            "boundary_convention": "fractional_log_pressure_layer_overlap",
            "archived_case": "end_to_end_cloud_parity PICASO/Virga cloud_tau",
            "single_scattering_albedo": 0.0,
        },
        "vertical_grid_contract": {
            "robert": "40/80/160 cells bounded by pressure_edges_bar",
            "picaso": "matching 41/81/161 pressure levels",
            "petitradtrans": "ROBERT geometric cell centres as pressure nodes",
        },
        "scattering_freeze": {
            "omega0": 0.0,
            "rayleigh": "explicitly off or absent in every framework",
            "cloud_scattering": "no scattering opacity/source supplied",
            "delta_m": False,
            "solver_order_comparison": "reserved for Stage 8",
        },
        "method_definitions": {
            "track_a": (
                "identical ROBERT source-HDF gas+CIA g-weighted layer tau and identical "
                "pressure-by-wavelength absorbing-cloud tau supplied to all three "
                "pure-absorption paths"
            ),
            "track_b": (
                "ROBERT CloudOpticalProperties uses layer extinction and fractional "
                "log-pressure placement; PICASO ingests dimensionless layer opd through "
                "its native cloud dataframe and interpolates spectrally; pRT ingests a "
                "continuous cm2/g additional absorption callback on pressure nodes"
            ),
            "tabulated_cloud": (
                "archived Virga/Mie extinction is conservatively remapped in log pressure, "
                "log-linearly interpolated in positive extinction, and held constant beyond "
                "the archived 1-12 micron wavelength endpoints"
            ),
            "cloud_effect": "signed cloudy-minus-clear spectrum within each thermal profile",
            "isothermal": (
                "isothermal blackbody-boundary cloud effects are analytically zero and are "
                "set exactly to zero before any normalization; raw cancellation maxima remain recorded"
            ),
            "cloud_response": (
                "absolute difference between normalized cloudy and clear layer contribution, "
                "renormalized over pressure independently at every wavelength"
            ),
            "contribution": (
                "ROBERT and pRT native contributions where available; PICASO and all Track-A "
                "paths use the independently implemented Stage-1 absorbing formal decomposition"
            ),
        },
        "acceptance_gates_predeclared_before_pilot": gates,
        "tracks": {
            "track_a_shared_tau": {
                "status": "passed" if passed else "failed",
                "acceptance_gates": gates,
            },
            "track_b_native_cloud": {
                "status": "characterized_no_cross_framework_gate",
                "interpretation": (
                    "native gas opacity, placement, interpolation, wavelength sampling, units, "
                    "and boundary differences are attribution results"
                ),
            },
        },
        "pilot": pilot_record,
        "per_resolution": per_resolution,
        "self_convergence": convergence,
        "prior_stage_relations": relations,
        "artifact_checksums": artifact_checksums,
        "benchmark_source_checksums": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "input_data_checksums": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "peak_orchestrator_rss_bytes": _peak_rss_bytes(),
    }
    analysis_elapsed = perf_counter() - started
    if args.reuse_existing:
        report["analysis_wall_time_s"] = analysis_elapsed
        report["wall_time_s"] = float(args.full_matrix_wall_time_s)
        report["reanalysis_note"] = (
            "complete preserved worker artifacts were reanalyzed after replacing the "
            "invalid exact-omega0 PICASO low-level shared call with the Stage-1-validated "
            "exact absorbing formal path; gates and all physical contracts were unchanged"
        )
    else:
        report["wall_time_s"] = analysis_elapsed
    return report, response_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--picaso-python", type=Path, default=DEFAULT_PICASO_PYTHON)
    parser.add_argument("--prt-python", type=Path, default=DEFAULT_PRT_PYTHON)
    parser.add_argument(
        "--picaso-reference", type=Path, default=DEFAULT_PICASO_REFERENCE
    )
    parser.add_argument("--picaso-database", type=Path, default=DEFAULT_PICASO_DATABASE)
    parser.add_argument("--prt-input", type=Path, default=DEFAULT_PRT_INPUT)
    parser.add_argument("--picaso-resample", type=int, default=50)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--full-matrix-wall-time-s", type=float, default=3666.374573249952)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.report_root = args.report_root.resolve()
    args.report_root.mkdir(parents=True, exist_ok=True)

    overall_started = perf_counter()
    report, artifact = _run(args)
    if artifact is not None:
        artifact_path = args.report_root / "stage_7_absorbing_cloud_arrays.npz"
        np.savez_compressed(artifact_path, **artifact)
        report["cloud_array_artifact"] = {
            "path": artifact_path.name,
            "sha256": sha256(artifact_path),
            "contents": (
                "complete R=100 absolute spectra, eclipse depths, signed cloud effects, "
                "normalized contributions and cloud responses, and cloud extinction tensors "
                "for both tracks, all frameworks, and 40/80/160 grids; normalized profiles "
                "use documented loss-bounded packed-uint12 storage"
            ),
        }
    launcher_elapsed = perf_counter() - overall_started
    if args.reuse_existing:
        report["reanalysis_launcher_wall_time_s"] = launcher_elapsed
        report["total_launcher_wall_time_s"] = float(
            args.full_matrix_wall_time_s + report["pilot"]["wall_time_s"]
        )
    else:
        report["total_launcher_wall_time_s"] = launcher_elapsed
    write_json(args.report_root / "stage_7_report.json", report)
    write_checksums(args.report_root)
    print(json.dumps({"stage": 7, "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
