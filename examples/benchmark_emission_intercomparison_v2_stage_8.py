"""Run the controlled Version-2 Stage-8 isotropic grey-aerosol study.

The broad 8A--8E matrix is a future study. This launcher runs only the 36-case
common-denominator Track-B contract and keeps all numerical products local and
ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from time import perf_counter
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (  # noqa: E402
    flux_conserving_bin_mean,
    load_version_2_common_contract,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_7 import (  # noqa: E402
    cloud_definitions,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_8_pilots import (  # noqa: E402
    CONTROLLED_CLOUD_STATES,
    CONTROLLED_PROFILE_GRIDS,
    CONTROLLED_SOLVER_PATHS,
    CURRENT_STAGE_8_STATUS,
    TRACK,
    controlled_study_payload,
)


COMMON = ROOT / "docs/data/emission_intercomparison/version_2/common_contract.json"
CONTRACT_JSON = ROOT / "docs/data/emission_intercomparison/version_2/stage_8_controlled_study_contract.json"
OUTPUT = ROOT / "examples/outputs/emission_intercomparison/version_2/stage_8/controlled_study"
WORKER = Path(__file__).with_name("run_emission_intercomparison_v2_stage_8_pilot.py")
STAGE7 = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_7.py")
PLOTTER = Path(__file__).with_name("plot_emission_intercomparison_v2_stage_8.py")
ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PYTHONS = {"robert": ROBERT_PYTHON, "picaso": PICASO_PYTHON, "petitradtrans": PRT_PYTHON}
PICASO_REFERENCE = Path("/Users/jaketaylor/Dropbox/picaso-v4/reference")
ROBERT_PROJECTED_PEAK_BYTES = 6_680_389_376
ROBERT_STRICT_AVAILABLE_BYTES = 11_133_982_294
ROBERT_ABSOLUTE_MIN_AVAILABLE_BYTES = 7_500_000_000
USER_AUTHORIZED_MEMORY_GATE_OVERRIDE = True
STATE_PARAMETERS = {
    item.name: (item.cloud_tau_at_5_micron, item.single_scattering_albedo, item.asymmetry_factor)
    for item in CONTROLLED_CLOUD_STATES
}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_controlled_contract(profile: str, n_cells: int) -> dict[str, np.ndarray]:
    """Build one accepted-placement profile/grid contract for three native states."""

    common = load_version_2_common_contract(COMMON)
    stage7 = _load_module("stage7_for_controlled_stage8", STAGE7)
    definitions = cloud_definitions()
    moderate = next(
        index
        for index, item in enumerate(definitions)
        if item.label == "deck_tau1_top10mbar_slope+0"
    )
    contract = stage7.build_stage_7_contract(
        common,
        n_cells,
        profiles=(profile,),
        cloud_indices=np.array([moderate]),
    )
    contract.update(
        {
            "schema_version": np.array("2.0.0"),
            "stage": np.array(8),
            "track": np.array(TRACK),
            "study_status": np.array(CURRENT_STAGE_8_STATUS),
            "controlled_study": np.array(True),
        }
    )
    return contract


def _run_case(
    *,
    framework: str,
    path: str,
    profile: str,
    n_cells: int,
    state: str,
    contract_path: Path,
    root: Path,
) -> dict[str, Any]:
    tau, omega0, g = STATE_PARAMETERS[state]
    run_root = root / framework / f"{profile}_{n_cells}_cells" / state
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / "native_output.npz"
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    command = [
        str(PYTHONS[framework]),
        str(WORKER),
        framework,
        path,
        str(contract_path),
        str(output),
        "--omega0",
        repr(omega0),
        "--tau",
        repr(tau),
        "--g",
        repr(g),
        "--delta-m",
        "off" if framework in {"robert", "picaso"} else "none",
        "--study-status",
        CURRENT_STAGE_8_STATUS,
        "--cloud-state",
        state,
    ]
    environment = os.environ.copy()
    environment["OMPI_MCA_btl"] = "self"
    if framework == "picaso":
        cache = root / "cache" / "picaso-v4"
        numba_cache = cache / "numba"
        mpl_cache = cache / "matplotlib"
        numba_cache.mkdir(parents=True, exist_ok=True)
        mpl_cache.mkdir(parents=True, exist_ok=True)
        environment["picaso_refdata"] = str(PICASO_REFERENCE)
        environment["NUMBA_CACHE_DIR"] = str(numba_cache)
        environment["MPLCONFIGDIR"] = str(mpl_cache)
    available = int(psutil.virtual_memory().available)
    override = False
    if framework == "robert":
        if available < ROBERT_ABSOLUTE_MIN_AVAILABLE_BYTES:
            raise RuntimeError(
                "ROBERT controlled-study launch has insufficient absolute memory: "
                f"{available} < {ROBERT_ABSOLUTE_MIN_AVAILABLE_BYTES} bytes"
            )
        override = available < ROBERT_STRICT_AVAILABLE_BYTES
        if override and not USER_AUTHORIZED_MEMORY_GATE_OVERRIDE:  # pragma: no cover
            raise RuntimeError("ROBERT strict memory gate failed without authorization")
    started = perf_counter()
    peak_tree = 0
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                members = [monitored, *monitored.children(recursive=True)]
                peak_tree = max(
                    peak_tree,
                    sum(member.memory_info().rss for member in members if member.is_running()),
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
            time.sleep(0.02)
        return_code = process.wait()
    wall = perf_counter() - started
    stderr_text = stderr_path.read_text()
    if return_code != 0:
        raise RuntimeError(
            f"controlled Stage 8 {framework}/{profile}/{n_cells}/{state} failed "
            f"({return_code})\n{stderr_text}"
        )
    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    if not bool(metadata["finite_spectrum"]):
        raise RuntimeError(f"controlled Stage 8 produced non-finite {framework}/{profile}/{n_cells}/{state}")
    return {
        "framework": framework,
        "path": path,
        "profile": profile,
        "cells": n_cells,
        "state": state,
        "tau": tau,
        "omega0": omega0,
        "g": g,
        "wall_time_s": wall,
        "process_tree_peak_rss_bytes": max(peak_tree, int(metadata["peak_rss_bytes"])),
        "available_memory_bytes_before": available,
        "available_memory_bytes_after": int(psutil.virtual_memory().available),
        "strict_memory_gate_passed": available >= ROBERT_STRICT_AVAILABLE_BYTES if framework == "robert" else True,
        "user_authorized_memory_override_used": override,
        "projected_peak_rss_bytes": ROBERT_PROJECTED_PEAK_BYTES if framework == "robert" else None,
        "metadata": metadata,
        "output": str(output.relative_to(root)),
        "warnings": [line for line in stderr_text.splitlines() if line.strip()],
        "output_sha256": _sha256(output),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
    }


def _close_edges(
    wavelength: np.ndarray, values: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output_wavelength = np.asarray(wavelength)
    output_values = np.asarray(values)
    if output_wavelength[0] > edges[0]:
        output_wavelength = np.concatenate(([edges[0]], output_wavelength))
        output_values = np.concatenate((output_values[..., :1], output_values), axis=-1)
    if output_wavelength[-1] < edges[-1]:
        output_wavelength = np.concatenate((output_wavelength, [edges[-1]]))
        output_values = np.concatenate((output_values, output_values[..., -1:]), axis=-1)
    return output_wavelength, output_values


def _bin_output(path: Path, edges: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        wavelength = np.asarray(archive["wavelength_micron"], dtype=float)
        flux = np.asarray(archive["flux_w_m2_m"], dtype=float)
    wavelength, flux = _close_edges(wavelength, flux, edges)
    return flux_conserving_bin_mean(wavelength, flux, edges)


def _p95_pair_peak(left: np.ndarray, right: np.ndarray) -> float:
    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0e-300)
    return float(np.percentile(np.abs(left - right) / peak, 95.0))


def _symmetric_p95(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.abs(left) + np.abs(right)
    values = np.divide(2.0 * np.abs(left - right), denominator, out=np.zeros_like(left), where=denominator > 0.0)
    return float(np.percentile(values, 95.0))


def _assemble(root: Path, execution: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    common = load_version_2_common_contract(COMMON)
    frameworks = tuple(item.framework for item in CONTROLLED_SOLVER_PATHS)
    states = tuple(item.name for item in CONTROLLED_CLOUD_STATES)
    shards = tuple(CONTROLLED_PROFILE_GRIDS)
    flux = np.empty((len(frameworks), len(shards), len(states), common.spectral.r100_centers_micron.size))
    for framework_index, framework in enumerate(frameworks):
        for shard_index, (profile, cells) in enumerate(shards):
            for state_index, state in enumerate(states):
                row = next(
                    item
                    for item in execution
                    if item["framework"] == framework
                    and item["profile"] == profile
                    and item["cells"] == cells
                    and item["state"] == state
                )
                flux[framework_index, shard_index, state_index] = _bin_output(
                    root / row["output"], common.spectral.r100_edges_micron
                )
    eclipse = (
        flux
        / common.stellar_surface_flux_r100_w_m2_m[None, None, None, :]
        * float(common.derived["projected_area_ratio"])
    )
    clear_index = states.index("clear")
    absorbing_index = states.index("absorbing_cloud")
    scattering_index = states.index("scattering_cloud")
    cloud_effect_ppm = (eclipse - eclipse[:, :, clear_index : clear_index + 1]) * 1.0e6
    scattering_increment_ppm = (
        eclipse[:, :, scattering_index] - eclipse[:, :, absorbing_index]
    ) * 1.0e6
    primary_shard = shards.index(("pg14_non_inverted", 80))
    high_shard = shards.index(("pg14_non_inverted", 160))
    isothermal_shard = shards.index(("isothermal", 80))
    framework_metrics: dict[str, Any] = {}
    for framework_index, framework in enumerate(frameworks):
        increment = scattering_increment_ppm[framework_index, primary_shard]
        framework_metrics[framework] = {
            "primary_scattering_increment_rms_ppm": float(np.sqrt(np.mean(increment**2))),
            "primary_scattering_increment_max_abs_ppm": float(np.max(np.abs(increment))),
            "primary_scattering_increment_median_ppm": float(np.median(increment)),
            "isothermal_scattering_increment_max_abs_ppm": float(
                np.max(np.abs(scattering_increment_ppm[framework_index, isothermal_shard]))
            ),
            "80_to_160_scattering_increment_p95_difference_over_pair_peak": _p95_pair_peak(
                scattering_increment_ppm[framework_index, primary_shard],
                scattering_increment_ppm[framework_index, high_shard],
            ),
            "80_to_160_scattering_spectrum_p95_symmetric_relative": _symmetric_p95(
                flux[framework_index, primary_shard, scattering_index],
                flux[framework_index, high_shard, scattering_index],
            ),
        }
    pair_metrics: dict[str, Any] = {}
    for left_index, left in enumerate(frameworks):
        for right_index in range(left_index + 1, len(frameworks)):
            right = frameworks[right_index]
            left_values = scattering_increment_ppm[left_index, primary_shard]
            right_values = scattering_increment_ppm[right_index, primary_shard]
            pair_metrics[f"{left}_vs_{right}"] = {
                "scattering_increment_rms_difference_ppm": float(
                    np.sqrt(np.mean((left_values - right_values) ** 2))
                ),
                "scattering_increment_p95_difference_over_pair_peak": _p95_pair_peak(
                    left_values, right_values
                ),
            }
    arrays = {
        "framework": np.asarray(frameworks),
        "profile": np.asarray([item[0] for item in shards]),
        "cells": np.asarray([item[1] for item in shards]),
        "state": np.asarray(states),
        "r100_centers_micron": common.spectral.r100_centers_micron,
        "r100_edges_micron": common.spectral.r100_edges_micron,
        "r100_flux_w_m2_m": flux,
        "r100_eclipse_depth": eclipse,
        "r100_cloud_effect_eclipse_ppm": cloud_effect_ppm,
        "r100_scattering_increment_eclipse_ppm": scattering_increment_ppm,
    }
    metrics = {
        "frameworks": framework_metrics,
        "cross_framework_pairs_descriptive_not_gated": pair_metrics,
        "all_finite": bool(all(np.all(np.isfinite(value)) for value in arrays.values() if np.issubdtype(value.dtype, np.number))),
        "primary_profile": "pg14_non_inverted",
        "primary_cells": 80,
        "primary_metric": "scattering_cloud_minus_absorbing_cloud",
    }
    return arrays, metrics


def _integrity(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "integrity.json":
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"schema_version": "2.0.0", "stage": 8, "status": CURRENT_STAGE_8_STATUS, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise ValueError("controlled Stage-8 products must remain in the ignored output root")
    args.output.mkdir(parents=True, exist_ok=True)
    documented = json.loads(CONTRACT_JSON.read_text())
    if documented["status"] != CURRENT_STAGE_8_STATUS:
        raise RuntimeError("versioned controlled-study contract status is not current")
    _write_json(args.output / "controlled_contract.json", controlled_study_payload())
    execution_path = args.output / "execution.partial.json"
    execution: list[dict[str, Any]] = []
    if args.resume and execution_path.exists():
        execution = json.loads(execution_path.read_text())["execution"]
    complete = {
        (row["framework"], row["profile"], row["cells"], row["state"])
        for row in execution
    }
    launcher_started = perf_counter()
    for solver in CONTROLLED_SOLVER_PATHS:
        for profile, cells in CONTROLLED_PROFILE_GRIDS:
            contract = build_controlled_contract(profile, cells)
            contract_path = args.output / "contracts" / f"{profile}_{cells}_cells.npz"
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            if not contract_path.exists():
                np.savez_compressed(contract_path, **contract)
            for state in STATE_PARAMETERS:
                key = (solver.framework, profile, cells, state)
                if key in complete:
                    continue
                execution.append(
                    _run_case(
                        framework=solver.framework,
                        path=solver.path,
                        profile=profile,
                        n_cells=cells,
                        state=state,
                        contract_path=contract_path,
                        root=args.output,
                    )
                )
                _write_json(execution_path, {"execution": execution})
    arrays, metrics = _assemble(args.output, execution)
    arrays_path = args.output / "controlled_study_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "schema_version": "2.0.0",
        "stage": 8,
        "status": CURRENT_STAGE_8_STATUS,
        "track": TRACK,
        "contract": documented,
        "execution": execution,
        "metrics": metrics,
        "resources": {
            "launcher_wall_time_s": perf_counter() - launcher_started,
            "sum_case_wall_time_s": sum(float(row["wall_time_s"]) for row in execution),
            "peak_process_tree_rss_bytes": max(int(row["process_tree_peak_rss_bytes"]) for row in execution),
            "minimum_available_memory_bytes_before": min(int(row["available_memory_bytes_before"]) for row in execution),
            "robert_memory_override_authorized_by_user": USER_AUTHORIZED_MEMORY_GATE_OVERRIDE,
            "robert_memory_override_used": any(bool(row["user_authorized_memory_override_used"]) for row in execution),
        },
        "production_case_count": len(execution),
        "future_study_matrix_started": False,
    }
    report_path = args.output / "controlled_study_report.json"
    _write_json(report_path, report)
    plot_command = [str(ROBERT_PYTHON), str(PLOTTER), "--product-root", str(args.output)]
    completed = subprocess.run(plot_command, cwd=ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"controlled Stage-8 plotting failed\n{completed.stderr}")
    _write_json(args.output / "integrity.json", _integrity(args.output))
    print(json.dumps({"report": str(report_path), "cases": len(execution), "all_finite": metrics["all_finite"]}, sort_keys=True))


if __name__ == "__main__":
    main()
