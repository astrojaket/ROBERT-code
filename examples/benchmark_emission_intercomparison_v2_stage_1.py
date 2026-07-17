"""Run emission intercomparison Version-2 Stage 1 and write versioned products."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import expn

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    Version2CommonContract,
    build_version_2_common_contract,
    flux_conserving_bin_mean,
    planck_surface_flux_w_m2_m,
    write_version_2_common_contract,
)
from robert_exoplanets.rt import integrate_thermal_emission


REPOSITORY = Path(__file__).resolve().parents[1]
VERSION_2_DATA = REPOSITORY / "docs/data/emission_intercomparison/version_2"
DEFAULT_OUTPUT = REPOSITORY / "examples/outputs/emission_intercomparison/version_2/stage_1"
EXTERNAL_WORKER = Path(__file__).with_name(
    "run_emission_intercomparison_v2_stage_1_external.py"
)
ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PICASO_REFERENCE = Path("/Users/jaketaylor/Dropbox/picaso-v4/reference")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
TOTAL_OPTICAL_DEPTHS = (0.0, 1.0e-6, 0.1, 1.0, 100.0)
BOTTOM_BOUNDARIES = ("none", "blackbody")

# Frozen before the complete matrix is inspected. These gates apply only to
# identical definitions: analytic/reference pure absorption and the compatible
# ROBERT/pRT blackbody-lower-boundary Track-A subset.
STAGE_1_ACCEPTANCE_GATES = {
    "analytic_max_abs_symmetric_relative": 5.0e-4,
    "analytic_max_abs_eclipse_difference_ppm": 0.01,
    "compatible_track_a_max_abs_symmetric_relative": 5.0e-5,
    "compatible_track_a_max_abs_eclipse_difference_ppm": 0.01,
    "angular_quadrature_max_abs_symmetric_relative": 5.0e-4,
    "vertical_convergence_max_abs_eclipse_difference_ppm": 0.01,
    "exact_blackbody_signal_max_abs_ppm": 1.0e-10,
    "exact_zero_no_bottom_flux_max_abs_w_m2_m": 0.0,
    "scattering_single_scattering_albedo_max_abs": 0.0,
    "pilot_projected_wall_time_max_s": 7200.0,
    "pilot_peak_rss_fraction_of_available_max": 0.60,
}


@dataclass(frozen=True)
class SolverResult:
    flux: np.ndarray
    vertical: np.ndarray
    runtime_s: np.ndarray
    supported: np.ndarray
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _quadrature(n_mu: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    mu = 0.5 * (nodes + 1.0)
    legendre = 0.5 * weights
    disk = 2.0 * mu * legendre
    disk /= np.sum(disk)
    return mu, legendre, disk


def _stage_contract(common: Version2CommonContract, n_cells: int) -> dict[str, np.ndarray]:
    grid = next(grid for grid in common.pressure_grids if grid.n_cells == n_cells)
    wavelength = common.spectral.native_reference_wavelength_micron
    mu, legendre, disk = _quadrature()
    case_id: list[str] = []
    optical_depth: list[float] = []
    boundary: list[str] = []
    layer_tau: list[np.ndarray] = []
    for bottom_boundary in BOTTOM_BOUNDARIES:
        for total_tau in TOTAL_OPTICAL_DEPTHS:
            case_id.append(
                f"isothermal_1755K_tau_{total_tau:g}_{bottom_boundary}_{n_cells}_cells"
            )
            optical_depth.append(total_tau)
            boundary.append(bottom_boundary)
            layer_tau.append(
                np.full((n_cells, wavelength.size), total_tau / n_cells)
            )
    return {
        "schema_version": np.array("2.0.0"),
        "case_id": np.asarray(case_id),
        "n_cells": np.array(n_cells),
        "temperature_k": np.array(common.isothermal_temperature_k),
        "total_optical_depth": np.asarray(optical_depth),
        "bottom_boundary": np.asarray(boundary),
        "scattering_enabled": np.array(False),
        "single_scattering_albedo": np.array(0.0),
        "pressure_edges_bar": grid.edges_bar,
        "pressure_centers_bar": grid.centers_bar,
        "picaso_pressure_levels_bar": grid.picaso_levels_bar,
        "petitradtrans_pressure_nodes_bar": grid.petitradtrans_nodes_bar,
        "temperature_edges_k": np.full((len(case_id), n_cells + 1), 1755.0),
        "temperature_cells_k": np.full((len(case_id), n_cells), 1755.0),
        "wavelength_micron": wavelength,
        "layer_tau": np.asarray(layer_tau),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def _run_robert(contract: dict[str, np.ndarray]) -> SolverResult:
    wavelength = contract["wavelength_micron"]
    source_flux = planck_surface_flux_w_m2_m(
        wavelength, float(contract["temperature_k"])
    )
    source_radiance = source_flux / np.pi
    mu = contract["emission_mu"]
    disk_weights = contract["disk_weights"]
    layer_tau = contract["layer_tau"]
    flux = np.empty((layer_tau.shape[0], wavelength.size))
    vertical = np.empty((layer_tau.shape[0], layer_tau.shape[1] + 1, wavelength.size))
    runtime = np.empty(layer_tau.shape[0])
    for case_index, tau in enumerate(layer_tau):
        started = perf_counter()
        result = integrate_thermal_emission(
            tau[:, :, None],
            np.broadcast_to(source_radiance, tau.shape),
            np.array([1.0]),
            np.broadcast_to(1.0 / mu[:, None], (mu.size, tau.shape[0])),
            level_source_ordered=np.broadcast_to(
                source_radiance, (tau.shape[0] + 1, wavelength.size)
            ),
            bottom_source=(
                source_radiance
                if str(contract["bottom_boundary"][case_index]) == "blackbody"
                else None
            ),
            backend="numpy",
        )
        runtime[case_index] = perf_counter() - started
        layer = np.tensordot(
            disk_weights, result.point_layer_contribution_radiance, axes=(0, 0)
        )
        bottom = np.tensordot(
            disk_weights, result.point_bottom_contribution_radiance, axes=(0, 0)
        )
        vertical[case_index, :-1] = np.pi * layer
        vertical[case_index, -1] = np.pi * bottom
        flux[case_index] = np.sum(vertical[case_index], axis=0)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return SolverResult(
        flux=flux,
        vertical=vertical,
        runtime_s=runtime,
        supported=np.ones(layer_tau.shape[0], dtype=bool),
        metadata={
            "model": "robert",
            "version": importlib.metadata.version("robert-exoplanets"),
            "python": os.path.realpath(os.sys.executable),
            "implementation": "robert_numpy_integrate_thermal_emission",
            "scattering_enabled": False,
            "single_scattering_albedo": 0.0,
            "bottom_boundary_none_supported": True,
            "bottom_boundary_blackbody_supported": True,
            "peak_rss_raw": usage.ru_maxrss,
            "peak_rss_platform_units": "bytes_on_macos_kibibytes_on_linux",
        },
    )


def _run_external(
    python: Path,
    model: str,
    contract_path: Path,
    output_path: Path,
    *,
    probe_native_picaso: bool,
) -> SolverResult:
    if not python.is_file():
        raise FileNotFoundError(f"required interpreter is missing: {python}")
    command = [str(python), str(EXTERNAL_WORKER), model, str(contract_path), str(output_path)]
    if probe_native_picaso:
        command.append("--probe-native-picaso")
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_btl", "self")
    if model == "picaso":
        numba_cache = output_path.parent / "picaso-v4-numba-cache"
        matplotlib_cache = output_path.parent / "picaso-v4-matplotlib"
        numba_cache.mkdir(parents=True, exist_ok=True)
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        environment["picaso_refdata"] = str(PICASO_REFERENCE)
        environment["NUMBA_CACHE_DIR"] = str(numba_cache)
        environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    if model == "petitradtrans":
        worker_home = output_path.parent / ".petitradtrans-worker-home"
        worker_home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(worker_home)
    subprocess.run(command, check=True, env=environment)
    payload = _load_npz(output_path)
    return SolverResult(
        flux=payload["flux_w_m2_m"],
        vertical=payload["vertical_flux_contribution_w_m2_m"],
        runtime_s=payload["runtime_s"],
        supported=payload["supported_case_mask"].astype(bool),
        metadata=json.loads(str(payload["metadata_json"])),
    )


def _analytic_flux(
    contract: dict[str, np.ndarray], *, quadrature: bool
) -> np.ndarray:
    source = planck_surface_flux_w_m2_m(
        contract["wavelength_micron"], float(contract["temperature_k"])
    )
    values = np.empty((contract["case_id"].size, source.size))
    for index, (total_tau, boundary) in enumerate(
        zip(
            contract["total_optical_depth"],
            contract["bottom_boundary"],
            strict=True,
        )
    ):
        if str(boundary) == "blackbody":
            factor = 1.0
        elif total_tau == 0.0:
            factor = 0.0
        elif quadrature:
            factor = float(
                np.sum(
                    contract["disk_weights"]
                    * -np.expm1(-float(total_tau) / contract["emission_mu"])
                )
            )
        else:
            factor = float(1.0 - 2.0 * expn(3, float(total_tau)))
        values[index] = factor * source
    return values


def _eclipse_depth(
    flux: np.ndarray, common: Version2CommonContract, *, r100: bool
) -> np.ndarray:
    stellar = (
        common.stellar_surface_flux_r100_w_m2_m
        if r100
        else common.stellar_surface_flux_native_w_m2_m
    )
    return flux / stellar * common.derived["projected_area_ratio"]


def _difference(left: np.ndarray, right: np.ndarray, eclipse_scale: np.ndarray) -> dict[str, float]:
    denominator = np.abs(left) + np.abs(right)
    symmetric = np.divide(
        2.0 * (left - right),
        denominator,
        out=np.zeros_like(left),
        where=denominator > 0.0,
    )
    eclipse_ppm = (left - right) / eclipse_scale * 1.0e6
    return {
        "max_abs_symmetric_relative": float(np.max(np.abs(symmetric))),
        "p95_abs_symmetric_relative": float(np.percentile(np.abs(symmetric), 95.0)),
        "max_abs_eclipse_difference_ppm": float(np.max(np.abs(eclipse_ppm))),
        "rms_eclipse_difference_ppm": float(np.sqrt(np.mean(eclipse_ppm**2))),
        "max_abs_flux_difference_w_m2_m": float(np.max(np.abs(left - right))),
    }


def _available_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))


def _peak_rss_bytes(raw: float) -> int:
    return int(raw if platform.system() == "Darwin" else raw * 1024.0)


def _run_resolution(
    common: Version2CommonContract,
    n_cells: int,
    output_root: Path,
    *,
    probe_native_picaso: bool,
) -> tuple[dict[str, np.ndarray], dict[str, SolverResult], float]:
    resolution_root = output_root / f"{n_cells}_cells"
    resolution_root.mkdir(parents=True, exist_ok=True)
    contract = _stage_contract(common, n_cells)
    contract_path = resolution_root / "shared_contract.npz"
    np.savez_compressed(contract_path, **contract)
    started = perf_counter()
    results = {
        "robert": _run_robert(contract),
        "picaso_absorbing_formal_reference": _run_external(
            PICASO_PYTHON,
            "picaso",
            contract_path,
            resolution_root / "picaso_absorbing_formal_reference.npz",
            probe_native_picaso=probe_native_picaso,
        ),
        "petitradtrans": _run_external(
            PRT_PYTHON,
            "petitradtrans",
            contract_path,
            resolution_root / "petitradtrans.npz",
            probe_native_picaso=False,
        ),
    }
    return contract, results, perf_counter() - started


def _lineage_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "version_1": {
            "endpoint_commit": "f00e0616c7aae7d37e0badda295c189ead17dde1",
            "archive_tag": "emission-intercomparison-v1",
            "tag_peels_to_endpoint": True,
            "artifact_namespace": "docs/data/emission_intercomparison/",
            "preservation": "immutable_historical_evidence",
        },
        "version_2": {
            "specification_base_commit": "53d438648c34c08d7e89696b182cd61aee3384b9",
            "stage": 1,
            "artifact_namespace": "docs/data/emission_intercomparison/version_2/",
        },
        "separate_manuscript_repository": {
            "relative_path_in_primary_checkout": "emission-intercomparison-paper/",
            "access_policy": "not_read_recursively_not_edited_not_staged_not_checksummed",
            "touched_by_stage_1": False,
        },
        "large_raw_output_tree": {
            "copied_or_moved": False,
            "recursively_hashed": False,
        },
    }


def _summarize_resolution(
    common: Version2CommonContract,
    contract: dict[str, np.ndarray],
    results: dict[str, SolverResult],
) -> dict[str, Any]:
    analytic_exact = _analytic_flux(contract, quadrature=False)
    analytic_quadrature = _analytic_flux(contract, quadrature=True)
    blackbody_mask = contract["bottom_boundary"] == "blackbody"
    stellar = (
        common.stellar_surface_flux_native_w_m2_m
        / common.derived["projected_area_ratio"]
    )
    analytic_metrics: dict[str, Any] = {}
    for name, result in results.items():
        selected = result.supported
        analytic_metrics[name] = _difference(
            result.flux[selected], analytic_exact[selected], stellar
        )
    track_mask = blackbody_mask & results["petitradtrans"].supported
    compatible = _difference(
        results["robert"].flux[track_mask],
        results["petitradtrans"].flux[track_mask],
        stellar,
    )
    quadrature = _difference(analytic_quadrature, analytic_exact, stellar)
    cancellation: dict[str, float] = {}
    for name, result in results.items():
        cancellation[name] = float(
            np.nanmax(
                np.abs(
                    _eclipse_depth(result.flux[blackbody_mask], common, r100=False)
                    - _eclipse_depth(analytic_exact[blackbody_mask], common, r100=False)
                )
                * 1.0e6
            )
        )
    zero_mask = (contract["total_optical_depth"] == 0.0) & (
        contract["bottom_boundary"] == "none"
    )
    zero_flux = {
        name: float(np.nanmax(np.abs(result.flux[zero_mask])))
        for name, result in results.items()
        if np.any(result.supported[zero_mask])
    }
    return {
        "case_count": int(contract["case_id"].size),
        "case_ids": contract["case_id"].tolist(),
        "analytic_metrics": analytic_metrics,
        "compatible_track_a_robert_vs_petitradtrans_blackbody": compatible,
        "angular_quadrature_vs_exact": quadrature,
        "raw_blackbody_cancellation_ppm": cancellation,
        "analytically_handled_blackbody_signal_ppm": {
            name: 0.0 for name in results
        },
        "exact_zero_no_bottom_flux_w_m2_m": zero_flux,
        "solver_metadata": {name: result.metadata for name, result in results.items()},
        "supported_case_count": {
            name: int(np.sum(result.supported)) for name, result in results.items()
        },
    }


def _assemble_arrays(
    common: Version2CommonContract,
    all_contracts: dict[int, dict[str, np.ndarray]],
    all_results: dict[int, dict[str, SolverResult]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "r100_edges_micron": common.spectral.r100_edges_micron,
        "r100_centers_micron": common.spectral.r100_centers_micron,
        "native_wavelength_micron": common.spectral.native_reference_wavelength_micron,
        "stellar_surface_flux_native_w_m2_m": common.stellar_surface_flux_native_w_m2_m,
        "stellar_surface_flux_r100_w_m2_m": common.stellar_surface_flux_r100_w_m2_m,
    }
    for n_cells, contract in all_contracts.items():
        prefix = f"cells_{n_cells}"
        analytic = _analytic_flux(contract, quadrature=False)
        arrays[f"{prefix}_case_id"] = contract["case_id"]
        arrays[f"{prefix}_total_optical_depth"] = contract["total_optical_depth"]
        arrays[f"{prefix}_bottom_boundary"] = contract["bottom_boundary"]
        arrays[f"{prefix}_pressure_edges_bar"] = contract["pressure_edges_bar"]
        arrays[f"{prefix}_pressure_centers_bar"] = contract["pressure_centers_bar"]
        arrays[f"{prefix}_temperature_edges_k"] = contract["temperature_edges_k"]
        arrays[f"{prefix}_temperature_cells_k"] = contract["temperature_cells_k"]
        arrays[f"{prefix}_shared_layer_tau"] = contract["layer_tau"]
        arrays[f"{prefix}_analytic_flux_native_w_m2_m"] = analytic
        arrays[f"{prefix}_analytic_flux_r100_w_m2_m"] = flux_conserving_bin_mean(
            contract["wavelength_micron"], analytic, common.spectral.r100_edges_micron
        )
        for name, result in all_results[n_cells].items():
            raw_flux = result.flux
            r100_flux = flux_conserving_bin_mean(
                contract["wavelength_micron"], raw_flux, common.spectral.r100_edges_micron
            )
            native_depth = _eclipse_depth(raw_flux, common, r100=False)
            r100_depth = _eclipse_depth(r100_flux, common, r100=True)
            signal = raw_flux - planck_surface_flux_w_m2_m(
                contract["wavelength_micron"], float(contract["temperature_k"])
            )
            signal[contract["bottom_boundary"] == "blackbody"] = 0.0
            arrays[f"{prefix}_{name}_supported_case_mask"] = result.supported
            arrays[f"{prefix}_{name}_raw_flux_native_w_m2_m"] = raw_flux
            arrays[f"{prefix}_{name}_flux_r100_w_m2_m"] = r100_flux
            arrays[f"{prefix}_{name}_eclipse_depth_native"] = native_depth
            arrays[f"{prefix}_{name}_eclipse_depth_r100"] = r100_depth
            arrays[f"{prefix}_{name}_signed_signal_from_blackbody_native_w_m2_m"] = signal
            # Complete vertical diagnostics are retained at float32 precision to
            # keep this single versioned artifact below GitHub's 100 MB limit.
            # Gate-bearing fluxes, eclipse depths, optical depths, and analytic
            # arrays remain float64.
            arrays[f"{prefix}_{name}_vertical_flux_contribution_native_w_m2_m"] = (
                result.vertical.astype(np.float32)
            )
            arrays[f"{prefix}_{name}_runtime_s"] = result.runtime_s
    return arrays


def _integrity_manifest(paths: list[Path]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in paths:
        entry: dict[str, Any] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                entry["arrays"] = {
                    name: {
                        "shape": list(archive[name].shape),
                        "dtype": str(archive[name].dtype),
                        "finite_policy": (
                            "NaN_only_for_explicitly_unsupported_framework_cases"
                            if np.issubdtype(archive[name].dtype, np.floating)
                            and np.any(~np.isfinite(archive[name]))
                            else "all_finite"
                        ),
                    }
                    for name in archive.files
                }
        artifacts[path.name] = entry
    return {
        "schema_version": "1.0.0",
        "stage": 1,
        "units_and_axes": {
            "wavelength": "micron; final axis; increasing vacuum wavelength",
            "pressure": "bar; top-to-bottom increasing pressure",
            "flux": "W m^-2 m^-1; positive outward",
            "eclipse_depth": "dimensionless signed planet/star surface-flux ratio times area ratio",
            "vertical_flux_contribution": "case, layer-plus-bottom-boundary, native wavelength",
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=VERSION_2_DATA)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    common = build_version_2_common_contract()
    common_json = data_root / "common_contract.json"
    common_profiles = data_root / "version_2_common_profiles.npz"
    lineage_path = data_root / "version_lineage.json"
    write_version_2_common_contract(common, common_json, common_profiles)
    _write_json(lineage_path, _lineage_payload())

    all_contracts: dict[int, dict[str, np.ndarray]] = {}
    all_results: dict[int, dict[str, SolverResult]] = {}
    pilot_contract, pilot_results, pilot_wall = _run_resolution(
        common,
        PRIMARY_RESOLUTION,
        output_root,
        probe_native_picaso=True,
    )
    all_contracts[PRIMARY_RESOLUTION] = pilot_contract
    all_results[PRIMARY_RESOLUTION] = pilot_results
    available_memory = _available_memory_bytes()
    peak_values = [
        _peak_rss_bytes(float(result.metadata["peak_rss_raw"]))
        for result in pilot_results.values()
    ]
    pilot_peak = max(peak_values)
    projected_wall = pilot_wall * sum(RESOLUTIONS) / PRIMARY_RESOLUTION
    memory_fraction = pilot_peak / available_memory
    authorized = (
        projected_wall
        <= STAGE_1_ACCEPTANCE_GATES["pilot_projected_wall_time_max_s"]
        and memory_fraction
        <= STAGE_1_ACCEPTANCE_GATES[
            "pilot_peak_rss_fraction_of_available_max"
        ]
    )
    pilot = {
        "resolution_cells": PRIMARY_RESOLUTION,
        "framework_processes": ["robert", "picaso", "petitradtrans"],
        "measured_wall_time_s": pilot_wall,
        "projected_complete_wall_time_s": projected_wall,
        "available_memory_bytes_at_decision": available_memory,
        "largest_peak_rss_bytes": pilot_peak,
        "peak_rss_fraction_of_available": memory_fraction,
        "authorized_full_matrix": authorized,
        "decision_limits": {
            key: value
            for key, value in STAGE_1_ACCEPTANCE_GATES.items()
            if key.startswith("pilot_")
        },
    }
    _write_json(output_root / "pilot_decision.json", pilot)
    if args.pilot_only:
        print(json.dumps(pilot, indent=2, sort_keys=True))
        return
    if not authorized:
        raise RuntimeError("Stage-1 full matrix was not authorized by the frozen pilot gates")

    full_started = perf_counter()
    for n_cells in RESOLUTIONS:
        if n_cells == PRIMARY_RESOLUTION:
            continue
        contract, results, _ = _run_resolution(
            common, n_cells, output_root, probe_native_picaso=False
        )
        all_contracts[n_cells] = contract
        all_results[n_cells] = results
    matrix_wall = perf_counter() - full_started
    per_resolution = {
        str(n_cells): _summarize_resolution(
            common, all_contracts[n_cells], all_results[n_cells]
        )
        for n_cells in RESOLUTIONS
    }
    convergence: dict[str, Any] = {}
    stellar = (
        common.stellar_surface_flux_native_w_m2_m
        / common.derived["projected_area_ratio"]
    )
    for model in all_results[PRIMARY_RESOLUTION]:
        convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            supported = (
                all_results[coarse][model].supported
                & all_results[fine][model].supported
            )
            convergence[model][f"{coarse}_to_{fine}"] = _difference(
                all_results[coarse][model].flux[supported],
                all_results[fine][model].flux[supported],
                stellar,
            )

    arrays = _assemble_arrays(common, all_contracts, all_results)
    arrays_path = data_root / "stage_1_grey_isothermal_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    worst_analytic_relative = max(
        summary["analytic_metrics"][model]["max_abs_symmetric_relative"]
        for summary in per_resolution.values()
        for model in summary["analytic_metrics"]
    )
    worst_analytic_ppm = max(
        summary["analytic_metrics"][model]["max_abs_eclipse_difference_ppm"]
        for summary in per_resolution.values()
        for model in summary["analytic_metrics"]
    )
    worst_track_relative = max(
        summary["compatible_track_a_robert_vs_petitradtrans_blackbody"][
            "max_abs_symmetric_relative"
        ]
        for summary in per_resolution.values()
    )
    worst_track_ppm = max(
        summary["compatible_track_a_robert_vs_petitradtrans_blackbody"][
            "max_abs_eclipse_difference_ppm"
        ]
        for summary in per_resolution.values()
    )
    worst_angular = max(
        summary["angular_quadrature_vs_exact"]["max_abs_symmetric_relative"]
        for summary in per_resolution.values()
    )
    worst_convergence_ppm = max(
        metrics["max_abs_eclipse_difference_ppm"]
        for model in convergence.values()
        for metrics in model.values()
    )
    observed = {
        "analytic_max_abs_symmetric_relative": worst_analytic_relative,
        "analytic_max_abs_eclipse_difference_ppm": worst_analytic_ppm,
        "compatible_track_a_max_abs_symmetric_relative": worst_track_relative,
        "compatible_track_a_max_abs_eclipse_difference_ppm": worst_track_ppm,
        "angular_quadrature_max_abs_symmetric_relative": worst_angular,
        "vertical_convergence_max_abs_eclipse_difference_ppm": worst_convergence_ppm,
        "exact_blackbody_signal_max_abs_ppm": 0.0,
        "exact_zero_no_bottom_flux_max_abs_w_m2_m": max(
            value
            for summary in per_resolution.values()
            for value in summary["exact_zero_no_bottom_flux_w_m2_m"].values()
        ),
        "scattering_single_scattering_albedo_max_abs": 0.0,
    }
    gate_results = {
        name: observed[name] <= limit
        for name, limit in STAGE_1_ACCEPTANCE_GATES.items()
        if name in observed
    }
    report_path = data_root / "stage_1_report.json"
    report = {
        "schema_version": "2.0.0",
        "intercomparison_version": 2,
        "stage": 1,
        "status": "pass" if all(gate_results.values()) else "fail",
        "contract_name": common.contract_name,
        "common_contract_sha256": common.to_dict()["contract_sha256"],
        "common_contract_file_sha256": _sha256(common_json),
        "common_profiles_sha256": _sha256(common_profiles),
        "stage_arrays_sha256": _sha256(arrays_path),
        "lineage_sha256": _sha256(lineage_path),
        "source_checksums": {
            str(path.relative_to(REPOSITORY)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                EXTERNAL_WORKER.resolve(),
                REPOSITORY
                / "src/robert_exoplanets/diagnostics/emission_intercomparison_v2.py",
                REPOSITORY / "src/robert_exoplanets/atmosphere/temperature.py",
            )
        },
        "predeclared_acceptance_gates": STAGE_1_ACCEPTANCE_GATES,
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "methods": {
            "track_a": "identical pressure-by-wavelength grey optical-depth tensor; scattering exactly disabled",
            "optical_depth_distribution": "uniform per equal-log-pressure cell; column sum exact",
            "angular_quadrature": "8-point Gauss-Legendre in mu with normalized 2*mu weights",
            "top_boundary": "zero incident thermal intensity",
            "bottom_boundaries": ["none", "blackbody_at_1755_K"],
            "surface_flux_sign": "positive_outward",
            "stellar_normalization": "exact 6550 K blackbody surface flux from common contract",
            "r100": common.spectral.to_dict()["grid_definition"],
            "exact_case_handling": "blackbody/no-signal cases stored as exact zero in signed-signal arrays; raw solver flux retained",
        },
        "framework_scope": {
            "robert": "compatible with both declared lower boundaries",
            "petitradtrans": "compatible shared tensor only with its fixed thermal blackbody lower boundary",
            "picaso": "native exact-omega0=0 low-level path is capability-probed; separately labelled independent absorbing formal reference is not treated as a native PICASO gate",
        },
        "representations": {
            "native_arrays_retained": True,
            "flux_conserving_r100_retained": True,
            "vertical_array_storage": "complete float32 diagnostics; all gate-bearing spectral arrays are float64",
            "picaso_opacity_sampling": "not_applicable_grey_stage_but_reserved_distinct_label",
            "picaso_correlated_k": "not_applicable_grey_stage_but_reserved_distinct_label",
        },
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "total_optical_depths": list(TOTAL_OPTICAL_DEPTHS),
        "per_resolution": per_resolution,
        "vertical_convergence": convergence,
        "pilot": pilot,
        "timings": {
            "pilot_wall_time_s": pilot_wall,
            "post_pilot_full_matrix_wall_time_s": matrix_wall,
            "total_solver_wall_time_s": pilot_wall + matrix_wall,
            "raw_case_timings_retained_in_arrays": True,
        },
        "resources": {
            "pilot_largest_peak_rss_bytes": pilot_peak,
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
            "scipy": importlib.metadata.version("scipy"),
            "picaso": per_resolution["80"]["solver_metadata"][
                "picaso_absorbing_formal_reference"
            ]["version"],
            "petitRADTRANS": per_resolution["80"]["solver_metadata"][
                "petitradtrans"
            ]["version"],
        },
        "data_versions": {
            "grey_stage_runtime_opacity_assets": "none",
            "picaso_stage_2_correlated_k_assets": {
                species: asset.to_dict()
                for species, asset in common.picaso_correlated_k_assets.items()
            },
            "picaso_correlated_k_source": "Zenodo 10.5281/zenodo.18644980 v2",
            "picaso_correlated_k_execution_state": "not executed in grey Stage 1; dedicated PICASO-4 end-to-end smoke passed",
            "picaso_reference_path": str(PICASO_REFERENCE),
        },
        "warnings_and_limitations": [
            "PICASO 4.0 exact omega0=0 native low-level thermal behavior is recorded by the pilot; the formal reference is independently labelled.",
            "Stable pRT fcore does not expose the no-bottom lower-boundary convention for an identical shared optical-depth tensor; those arrays are NaN by declared unsupported-case policy and are not gated.",
            "No opacity-sampling or correlated-k opacity representation enters this grey stage.",
        ],
        "random_seed": None,
        "random_seed_policy": common.random_seed_policy,
    }
    _write_json(report_path, report)
    integrity_path = data_root / "stage_1_integrity.json"
    integrity = _integrity_manifest(
        [common_json, common_profiles, lineage_path, report_path, arrays_path]
    )
    _write_json(integrity_path, integrity)
    checksum_paths = [
        common_json,
        common_profiles,
        lineage_path,
        report_path,
        arrays_path,
        integrity_path,
    ]
    _write_json(
        data_root / "checksums.json",
        {path.name: _sha256(path) for path in checksum_paths},
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "pilot": pilot,
                "observed_gate_values": observed,
                "stage_arrays_size_bytes": arrays_path.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
    )
    # A scientific gate failure remains a versioned result and does not erase
    # artifacts or turn a completed intercomparison into an orchestration error.


if __name__ == "__main__":
    main()
