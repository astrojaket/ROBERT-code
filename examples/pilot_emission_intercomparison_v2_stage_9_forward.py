#!/usr/bin/env python3
"""Measure an explicitly approved Stage-9 native forward pilot on Glamdring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_intercomparison_v2_stage_9_native import (  # noqa: E402
    build_native_forward,
    load_common_contract,
    truth_parameters,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    FRAMEWORKS,
    SCENARIOS,
)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("scenario", choices=tuple(item.name for item in SCENARIOS))
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--evaluations", type=int, default=2)
    args = parser.parse_args()
    if not args.approved:
        parser.error("--approved is required before executing a native forward pilot")
    if args.evaluations < 2:
        parser.error("at least one cold and one warm evaluation are required")
    if os.environ.get("STAGE9_CLUSTER") != "glamdring":
        raise RuntimeError("Stage-9 native forward pilots are restricted to Glamdring")
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError("mpi4py is required for the 12-rank pilot") from exc
    communicator = MPI.COMM_WORLD
    rank = int(communicator.Get_rank())
    if communicator.Get_size() != 12:
        raise RuntimeError("Stage-9 forward pilots require exactly 12 MPI ranks")
    common = load_common_contract(
        args.project_root.expanduser().resolve() / "contracts" / "common_contract.json"
    )
    setup_started = perf_counter()
    forward = build_native_forward(args.framework, common, args.scenario)
    setup_seconds = perf_counter() - setup_started
    truth = truth_parameters(common, args.scenario)
    timings = []
    spectrum = np.empty(0)
    for _ in range(args.evaluations):
        started = perf_counter()
        spectrum = forward.eclipse_depth(truth)
        timings.append(perf_counter() - started)
    if not np.all(np.isfinite(spectrum)):
        raise RuntimeError("native pilot spectrum is non-finite")
    local: dict[str, Any] = {
        "rank": rank,
        "setup_seconds": setup_seconds,
        "evaluation_seconds": timings,
        "peak_rss_bytes": _rss_bytes(),
        "spectrum_sha256": hashlib.sha256(spectrum.tobytes()).hexdigest(),
    }
    gathered = communicator.gather(local, root=0)
    if rank == 0:
        hashes = {item["spectrum_sha256"] for item in gathered}
        if len(hashes) != 1:
            raise RuntimeError(
                "MPI ranks did not produce identical native truth spectra"
            )
        payload = {
            "schema_version": "1.0",
            "pilot_only": True,
            "framework": args.framework,
            "scenario": args.scenario,
            "mpi_world_size": 12,
            "evaluations_per_rank": args.evaluations,
            "process_tree_peak_rss_upper_bound_bytes": sum(
                item["peak_rss_bytes"] for item in gathered
            ),
            "rank_measurements": gathered,
            "finite_spectrum": True,
            "r100_bin_count": int(spectrum.size),
        }
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    main()
