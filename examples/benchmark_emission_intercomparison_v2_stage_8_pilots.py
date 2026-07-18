"""Run Version-2 Stage-8 Track-B resource-planning pilots only.

This launcher cannot run a production matrix.  It creates one frozen 80-cell
case, executes cold and warm isolated native-code workers, and projects the
previously frozen full-matrix counts without changing any scientific input.
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
    load_version_2_common_contract,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_7 import (  # noqa: E402
    cloud_definitions,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_8_pilots import (  # noqa: E402
    PILOT_PATHS,
    PILOT_PROFILE,
    PRIMARY_CELLS,
    FUTURE_STUDY_STATUS,
    TRACK,
    PilotPath,
    contract_payload,
    project_path_resources,
    supported_paths,
)


COMMON = ROOT / "docs/data/emission_intercomparison/version_2/common_contract.json"
OUTPUT = ROOT / "examples/outputs/emission_intercomparison/version_2/stage_8/pilots"
WORKER = Path(__file__).with_name("run_emission_intercomparison_v2_stage_8_pilot.py")
STAGE7 = Path(__file__).with_name("benchmark_emission_intercomparison_v2_stage_7.py")
ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PYTHONS = {"robert": ROBERT_PYTHON, "picaso": PICASO_PYTHON, "petitradtrans": PRT_PYTHON}
PICASO_REFERENCE = Path("/Users/jaketaylor/Dropbox/picaso-v4/reference")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_pilot_contract() -> dict[str, np.ndarray]:
    """Build exactly one accepted-placement 80-cell native case."""

    common = load_version_2_common_contract(COMMON)
    stage7 = _load_module("stage7_for_stage8_pilot", STAGE7)
    definitions = cloud_definitions()
    moderate = next(
        index
        for index, item in enumerate(definitions)
        if item.label == "deck_tau1_top10mbar_slope+0"
    )
    contract = stage7.build_stage_7_contract(
        common,
        PRIMARY_CELLS,
        profiles=(PILOT_PROFILE,),
        cloud_indices=np.array([moderate]),
    )
    contract["schema_version"] = np.array("2.0.0")
    contract["stage"] = np.array(8)
    contract["track"] = np.array(TRACK)
    contract["pilot_only"] = np.array(True)
    return contract


def _run(path: PilotPath, state: str, contract_path: Path, root: Path) -> dict[str, Any]:
    run_root = root / path.subsection / path.framework / path.path
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / f"{state}.npz"
    stdout_path = run_root / f"{state}.stdout.log"
    stderr_path = run_root / f"{state}.stderr.log"
    delta = "none" if path.delta_m is None else "on" if path.delta_m else "off"
    command = [
        str(PYTHONS[path.framework]), str(WORKER), path.framework, path.path,
        str(contract_path), str(output), "--omega0", repr(path.pilot_omega0),
        "--tau", repr(path.pilot_tau), "--g", repr(path.pilot_g), "--delta-m", delta,
    ]
    environment = os.environ.copy()
    environment["OMPI_MCA_btl"] = "self"
    if path.framework == "picaso":
        cache_root = root / "cache" / "picaso-v4"
        numba_cache = cache_root / "numba"
        mpl_cache = cache_root / "matplotlib"
        numba_cache.mkdir(parents=True, exist_ok=True)
        mpl_cache.mkdir(parents=True, exist_ok=True)
        environment["picaso_refdata"] = str(PICASO_REFERENCE)
        environment["NUMBA_CACHE_DIR"] = str(numba_cache)
        environment["MPLCONFIGDIR"] = str(mpl_cache)
    available_before = int(psutil.virtual_memory().available)
    started = perf_counter()
    peak_tree = 0
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                members = [monitored, *monitored.children(recursive=True)]
                peak_tree = max(peak_tree, sum(member.memory_info().rss for member in members if member.is_running()))
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
            time.sleep(0.02)
        return_code = process.wait()
    wall = perf_counter() - started
    stderr_text = stderr_path.read_text()
    if return_code != 0:
        raise RuntimeError(f"{path.subsection} {path.framework}/{path.path} {state} failed ({return_code})\n{stderr_text}")
    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    return {
        "state": state,
        "command": command,
        "wall_time_s": wall,
        "process_tree_peak_rss_bytes": max(peak_tree, int(metadata["peak_rss_bytes"])),
        "available_memory_bytes_before": available_before,
        "available_memory_bytes_after": int(psutil.virtual_memory().available),
        "metadata": metadata,
        "stderr_warnings": [line for line in stderr_text.splitlines() if line.strip()],
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "output_sha256": _sha256(output),
    }


def _integrity(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "integrity.json":
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"schema_version": "2.0.0", "stage": 8, "track": TRACK, "files": files}


def _native_sizes(row: dict[str, Any]) -> dict[str, int | None]:
    """Separate native spectral, quadrature, angle, and solver-order counts."""

    metadata = row["warm"]["metadata"]
    framework = row["framework"]
    path = row["path"]
    stream_count: int | None = None
    angle_count: int | None = None
    g_count: int | None = None
    if framework == "robert":
        angle_count = 8
        g_count = int(metadata["native_g_or_angle_or_stream_count"])
        stream_count = 2 if path.startswith("toon") else 4 if "sh4" in path else None
    elif framework == "picaso":
        g_count = int(metadata["native_g_or_angle_or_stream_count"])
        stream_count = 2 if path.startswith("toon") else 4
    else:
        angle_count = int(metadata["native_g_or_angle_or_stream_count"])
    return {
        "wavelength_count": int(metadata["native_wavelength_count"]),
        "bin_count": int(metadata["native_bin_count"]),
        "correlated_k_g_count": g_count,
        "emission_angle_count": angle_count,
        "stream_count": stream_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--subsection", choices=("8A", "8B", "8C", "8D", "8E"), action="append")
    parser.add_argument("--path", action="append", help="Optional exact framework:path selector")
    parser.add_argument("--resume", action="store_true", help="Reuse checkpointed completed path measurements")
    parser.add_argument("--force", action="store_true", help="Retime selected paths while resuming all others")
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise ValueError("Stage-8 pilot products must remain in the frozen ignored output root")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = contract_payload()
    _write_json(args.output / "frozen_pilot_contract.json", frozen)
    contract = build_pilot_contract()
    contract_path = args.output / "pilot_case_80_cells.npz"
    np.savez_compressed(contract_path, **contract)
    selected = supported_paths()
    if args.subsection:
        selected = tuple(item for item in selected if item.subsection in set(args.subsection))
    if args.path:
        names = set(args.path)
        selected = tuple(item for item in selected if f"{item.framework}:{item.path}" in names)
    if not selected:
        raise ValueError("no supported frozen paths selected")
    partial_path = args.output / "pilot_report.partial.json"
    results: list[dict[str, Any]] = []
    if args.resume and partial_path.exists():
        results = json.loads(partial_path.read_text())["results"]
    if args.force:
        selected_keys = {
            (item.subsection, item.framework, item.path) for item in selected
        }
        results = [
            row
            for row in results
            if (row["subsection"], row["framework"], row["path"])
            not in selected_keys
        ]
    completed_keys = {
        (row["subsection"], row["framework"], row["path"]) for row in results
    }
    for item in selected:
        if (item.subsection, item.framework, item.path) in completed_keys:
            continue
        cold = _run(item, "cold", contract_path, args.output)
        warm = _run(item, "warm", contract_path, args.output)
        warm_metadata = warm["metadata"]
        projection = project_path_resources(
            cold_wall_s=float(cold["wall_time_s"]), warm_wall_s=float(warm["wall_time_s"]),
            warm_setup_s=float(warm_metadata["setup_time_s"]), warm_case_s=float(warm_metadata["case_time_s"]),
            peak_rss_bytes=max(int(cold["process_tree_peak_rss_bytes"]), int(warm["process_tree_peak_rss_bytes"])),
            available_memory_bytes=min(int(cold["available_memory_bytes_before"]), int(warm["available_memory_bytes_before"])),
            retained_tensor_bytes=int(warm_metadata["retained_tensor_bytes"]), full_case_count=item.full_case_count,
        )
        projection["basis"] = (
            "completed_representative_case"
            if bool(cold["metadata"]["finite_spectrum"])
            and bool(warm["metadata"]["finite_spectrum"])
            else "time_to_native_failure_lower_bound"
        )
        results.append({
            "subsection": item.subsection, "framework": item.framework, "path": item.path,
            "frozen_path": item.__dict__, "cold": cold, "warm": warm, "projection": projection,
        })
        _write_json(partial_path, {"frozen_contract": frozen, "results": results})
    subsection_projection: dict[str, dict[str, float | int | bool]] = {}
    for row in results:
        row["native_sizes"] = _native_sizes(row)
    for subsection in ("8A", "8B", "8C", "8D", "8E"):
        rows = [row for row in results if row["subsection"] == subsection]
        if rows:
            subsection_projection[subsection] = {
                "serial_wall_time_s": sum(float(row["projection"]["projected_wall_time_s"]) for row in rows),
                "parallel_peak_rss_bytes": sum(int(row["projection"]["projected_peak_rss_bytes"]) for row in rows),
                "serial_peak_rss_bytes": max(int(row["projection"]["projected_peak_rss_bytes"]) for row in rows),
                "path_count": len(rows),
            }
    report = {
        "schema_version": "2.0.0", "stage": 8, "kind": "timing_and_feasibility_pilots_only",
        "status": FUTURE_STUDY_STATUS,
        "track": TRACK, "frozen_contract": frozen,
        "unsupported_paths": [item.__dict__ for item in PILOT_PATHS if not item.supported],
        "results": results, "subsection_projection": subsection_projection,
        "combined_projected_serial_wall_time_s": sum(float(value["serial_wall_time_s"]) for value in subsection_projection.values()),
        "combined_projected_all_path_parallel_peak_rss_bytes": sum(int(value["parallel_peak_rss_bytes"]) for value in subsection_projection.values()),
        "complete_supported_path_count": len(results) == len(supported_paths()),
        "production_matrix_started": False,
    }
    _write_json(args.output / "pilot_report.json", report)
    _write_json(args.output / "integrity.json", _integrity(args.output))
    print(json.dumps({"report": str(args.output / "pilot_report.json"), "supported_paths_timed": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
