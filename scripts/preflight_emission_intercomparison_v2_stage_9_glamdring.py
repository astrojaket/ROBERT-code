#!/usr/bin/env python3
"""Run the non-science Stage-9 environment and deployment preflight.

This command never constructs a forward model or evaluates a spectrum. Run it
inside a 12-rank Glamdring addqueue allocation for each framework environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    FRAMEWORKS,
    build_run_matrix,
)


EXPECTED = {
    "robert": {"robert-exoplanets": None},
    "picaso": {"picaso": "4.0"},
    "petitradtrans": {"petitRADTRANS": "3.3.3"},
}
SHARED = {"multinest": "3.10", "pymultinest": "2.12", "mpi4py": "4.1.2"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _version(name: str) -> str:
    if name == "multinest":
        prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
        records = sorted((prefix / "conda-meta").glob("multinest-*.json"))
        if len(records) != 1:
            raise RuntimeError("cannot resolve one MultiNest conda package record")
        return str(json.loads(records[0].read_text(encoding="utf-8"))["version"])
    return importlib.metadata.version(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    if os.environ.get("STAGE9_CLUSTER") != "glamdring":
        raise RuntimeError("Stage-9 preflight is restricted to Glamdring")
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError("mpi4py is required for the 12-rank preflight") from exc
    communicator = MPI.COMM_WORLD
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    mpi_library_version = MPI.Get_library_version()
    if "MPICH" not in mpi_library_version.upper():
        raise RuntimeError(
            "Stage-9 environments must use their pinned Conda MPICH library; "
            f"observed {mpi_library_version}"
        )
    if size != 12:
        raise RuntimeError(
            f"Stage-9 preflight requires exactly 12 MPI ranks, observed {size}"
        )
    hydra = {
        "HYDRA_LAUNCHER": os.environ.get("HYDRA_LAUNCHER"),
        "HYDRA_RMK": os.environ.get("HYDRA_RMK"),
    }
    if hydra != {"HYDRA_LAUNCHER": "fork", "HYDRA_RMK": "user"}:
        raise RuntimeError(
            "Stage-9 preflight requires the pinned single-node Hydra launcher: "
            f"{hydra}"
        )
    threads = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    if any(value != "1" for value in threads.values()):
        raise RuntimeError(f"every numerical thread limit must be one: {threads}")
    project = args.project_root.expanduser().resolve()
    matrix = build_run_matrix()
    if len(matrix) != 72 or len({item.shard_id for item in matrix}) != 12:
        raise RuntimeError("frozen Stage-9 matrix validation failed")
    run_index = json.loads((project / "run_index.json").read_text(encoding="utf-8"))
    if len(run_index) != 72:
        raise RuntimeError("deployed run index does not contain 72 retrievals")
    versions = {name: _version(name) for name in SHARED}
    versions.update({name: _version(name) for name in EXPECTED[args.framework]})
    for name, expected in {**SHARED, **EXPECTED[args.framework]}.items():
        if expected is not None and not (
            versions[name] == expected or versions[name].startswith(expected + ".")
        ):
            raise RuntimeError(f"{name} must be {expected}, observed {versions[name]}")
    required_paths = {
        "STAGE9_PRT_INPUT_DATA": os.environ.get("STAGE9_PRT_INPUT_DATA"),
        "STAGE9_PICASO_CK_DIRECTORY": os.environ.get("STAGE9_PICASO_CK_DIRECTORY"),
        "picaso_refdata": os.environ.get("picaso_refdata"),
        "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
    }
    for name, value in required_paths.items():
        if not value or not Path(value).is_dir():
            raise RuntimeError(f"{name} must name an existing directory")
    for name in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        if not os.access(required_paths[name], os.W_OK):
            raise RuntimeError(f"{name} must be writable")
    common = project / "contracts" / "common_contract.json"
    frozen = project / "contracts" / "stage_9_retrieval_contract.json"
    deployed = json.loads(frozen.read_text(encoding="utf-8"))
    if deployed["common_contract_sha256"] != _sha256(
        ROOT / "docs/data/emission_intercomparison/version_2/common_contract.json"
    ):
        raise RuntimeError(
            "deployed frozen contract points to a different repository common contract"
        )
    if (
        json.loads(common.read_text(encoding="utf-8"))["contract_name"]
        != "wasp17b_emission_intercomparison_v2"
    ):
        raise RuntimeError(
            "deployed common contract is not the Version-2 WASP-17b contract"
        )
    local = {
        "rank": rank,
        "hostname": platform.node(),
        "python": os.path.realpath(sys.executable),
        "versions": versions,
        "mpi_library_version": mpi_library_version,
        "hydra": hydra,
        "threads": threads,
        "paths": required_paths,
    }
    gathered: list[dict[str, Any]] | None = communicator.gather(local, root=0)
    if rank == 0:
        assert gathered is not None
        hostnames = {item["hostname"] for item in gathered}
        if len(hostnames) != 1:
            raise RuntimeError(
                "Stage-9 Glamdring allocation must keep all 12 ranks on one node: "
                f"{sorted(hostnames)}"
            )
        report = {
            "schema_version": "1.0",
            "science_executed": False,
            "framework": args.framework,
            "mpi_world_size": size,
            "single_node": True,
            "ranks": gathered,
            "matrix_retrievals": len(matrix),
            "matrix_shards": len({item.shard_id for item in matrix}),
            "common_contract_sha256": _sha256(common),
            "stage_9_contract_sha256": _sha256(frozen),
        }
        output = project / "integrity" / f"preflight-{args.framework}.json"
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(output)


if __name__ == "__main__":
    main()
