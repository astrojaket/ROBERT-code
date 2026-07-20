#!/usr/bin/env python3
"""Create the complete Stage-9 Glamdring directory and shard layout.

This command performs filesystem setup only.  It never imports a radiative
transfer framework, creates an opacity, evaluates a spectrum, or starts a
retrieval.  Existing run definitions must match byte-for-byte unless
``--verify-only`` is used, in which case nothing is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    FRAMEWORKS,
    SCENARIOS,
    build_run_matrix,
    frozen_contract_payload,
)


COMMON_CONTRACT = (
    ROOT / "docs/data/emission_intercomparison/version_2/common_contract.json"
)
FROZEN_CONTRACT = (
    ROOT
    / "docs/data/emission_intercomparison/version_2/stage_9_retrieval_contract.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _ensure_text(path: Path, text: str, *, verify_only: bool) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(
                f"existing Stage-9 definition differs from frozen content: {path}"
            )
        return
    if verify_only:
        raise RuntimeError(f"required Stage-9 file is missing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_directory(path: Path, *, verify_only: bool) -> None:
    if path.is_dir():
        return
    if path.exists():
        raise RuntimeError(f"Stage-9 directory target is not a directory: {path}")
    if verify_only:
        raise RuntimeError(f"required Stage-9 directory is missing: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _refresh_execution_deployment(
    root: Path,
    *,
    frozen: dict[str, object],
    integrity: dict[str, object],
) -> None:
    """Refresh only pre-science execution metadata after a launcher correction."""

    deployed_contract_path = root / "contracts" / "stage_9_retrieval_contract.json"
    deployed_manifest_path = root / "integrity" / "setup_manifest.json"
    if not deployed_contract_path.is_file() or not deployed_manifest_path.is_file():
        raise RuntimeError(
            "execution refresh requires an existing prepared Stage-9 deployment"
        )
    science_products = tuple(
        path
        for pattern in (
            "injections/**/native_mean.npz",
            "runs/**/result.json",
            "runs/**/result_arrays.npz",
            "runs/**/diagnostic_spectra.npz",
            "runs/**/posterior_summary.json",
            "runs/**/chains/*",
            "pilots/**/*",
            "diagnostics/resource/forward-pilot-*.json",
        )
        for path in root.glob(pattern)
        if path.is_file()
    )
    if science_products:
        raise RuntimeError(
            "refusing execution refresh after Stage-9 science products exist: "
            f"{science_products[0]}"
        )

    deployed_contract = json.loads(
        deployed_contract_path.read_text(encoding="utf-8")
    )
    deployed_science = dict(deployed_contract)
    expected_science = dict(frozen)
    deployed_science.pop("execution", None)
    expected_science.pop("execution", None)
    if deployed_science != expected_science:
        raise RuntimeError(
            "refusing refresh because deployed Stage-9 science content changed"
        )

    deployed_manifest = json.loads(
        deployed_manifest_path.read_text(encoding="utf-8")
    )
    if deployed_manifest.get("stage_9_contract_repository_sha256") != _sha256(
        deployed_contract_path
    ):
        raise RuntimeError(
            "refusing refresh because the deployed contract hash does not match"
        )
    comparable_manifest = dict(deployed_manifest)
    comparable_integrity = dict(integrity)
    comparable_manifest.pop("stage_9_contract_repository_sha256", None)
    comparable_integrity.pop("stage_9_contract_repository_sha256", None)
    if comparable_manifest != comparable_integrity:
        raise RuntimeError(
            "refusing refresh because the deployed setup manifest changed"
        )

    deployed_contract_path.write_text(
        _canonical_json(frozen), encoding="utf-8"
    )
    deployed_manifest_path.write_text(
        _canonical_json(integrity), encoding="utf-8"
    )


def prepare(
    root: Path,
    *,
    verify_only: bool = False,
    refresh_execution_contract: bool = False,
) -> dict[str, object]:
    """Create or verify a complete, idempotent Stage-9 project tree."""

    if verify_only and refresh_execution_contract:
        raise RuntimeError(
            "--verify-only and --refresh-execution-contract are mutually exclusive"
        )
    root = root.expanduser().resolve()
    common = json.loads(COMMON_CONTRACT.read_text(encoding="utf-8"))
    common_sha = _sha256(COMMON_CONTRACT)
    frozen = json.loads(FROZEN_CONTRACT.read_text(encoding="utf-8"))
    expected_frozen = frozen_contract_payload(common_contract_sha256=common_sha)
    if frozen != expected_frozen:
        raise RuntimeError(
            "committed Stage-9 contract does not match the Stage-9 source of truth"
        )
    runs = build_run_matrix()
    integrity = {
        "schema_version": "1.0",
        "common_contract_repository_sha256": common_sha,
        "stage_9_contract_repository_sha256": _sha256(FROZEN_CONTRACT),
        "run_count": len(runs),
        "shard_count": len({run.shard_id for run in runs}),
        "noise_vector_count": 0,
        "injection_mean_count_required": len(FRAMEWORKS) * len(SCENARIOS),
    }
    if refresh_execution_contract:
        _refresh_execution_deployment(root, frozen=frozen, integrity=integrity)

    directories = (
        root,
        root / "contracts",
        root / "injections",
        root / "runs",
        root / "shards",
        root / "logs",
        root / "cache" / "picaso" / "numba",
        root / "cache" / "picaso" / "matplotlib",
        root / "reference" / "picaso",
        root / "reference" / "petitradtrans",
        root / "reference" / "robert",
        root / "diagnostics" / "posterior",
        root / "diagnostics" / "spectral",
        root / "diagnostics" / "resource",
        root / "archive" / "chains",
        root / "integrity",
        root / "tmp",
    )
    for path in directories:
        _ensure_directory(path, verify_only=verify_only)

    _ensure_text(
        root / "contracts" / "common_contract.json",
        _canonical_json(common),
        verify_only=verify_only,
    )
    _ensure_text(
        root / "contracts" / "stage_9_retrieval_contract.json",
        _canonical_json(frozen),
        verify_only=verify_only,
    )

    injection_index: list[dict[str, object]] = []
    for framework in FRAMEWORKS:
        for scenario in SCENARIOS:
            path = root / "injections" / framework / scenario.name
            _ensure_directory(path, verify_only=verify_only)
            injection_index.append(
                {
                    "framework": framework,
                    "scenario": scenario.name,
                    "directory": str(path.relative_to(root)),
                    "required_product": "native_mean.npz",
                    "status": "awaiting_approved_glamdring_generation",
                }
            )

    by_shard: dict[str, list[dict[str, object]]] = {}
    run_index: list[dict[str, object]] = []
    for run in runs:
        run_dir = root / "runs" / run.retriever / run.scenario / run.run_id
        _ensure_directory(run_dir, verify_only=verify_only)
        payload = {
            **run.to_mapping(),
            "schema_version": "2.0.0",
            "stage": 9,
            "track": "track_b_native_retrieval",
            "project_root": str(root),
            "run_directory": str(run_dir),
            "injection_product": str(
                root / "injections" / run.injector / run.scenario / "native_mean.npz"
            ),
            "noise_vector": None,
            "common_contract": str(root / "contracts" / "common_contract.json"),
            "frozen_contract": str(
                root / "contracts" / "stage_9_retrieval_contract.json"
            ),
        }
        _ensure_text(
            run_dir / "run.json", _canonical_json(payload), verify_only=verify_only
        )
        row = {
            "run_id": run.run_id,
            "run_config": str((run_dir / "run.json").relative_to(root)),
            "shard_id": run.shard_id,
            "retriever": run.retriever,
            "scenario": run.scenario,
            "control": run.control,
        }
        run_index.append(row)
        by_shard.setdefault(run.shard_id, []).append(row)

    for shard_id, rows in sorted(by_shard.items()):
        retriever, scenario = shard_id.split("__", maxsplit=1)
        request_gb = {
            "robert": 128 if "grey_" in scenario else 96,
            "picaso": 32,
            "petitradtrans": 64,
        }[retriever]
        shard_payload = {
            "shard_id": shard_id,
            "retriever": retriever,
            "scenario": scenario,
            "run_count": len(rows),
            "mpi_ranks_per_retrieval": 12,
            "threads_per_rank": 1,
            "scheduler_queue": "redwood",
            "preliminary_memory_gb": request_gb,
            "preliminary_walltime": "48:00:00",
            "execution_order": [row["run_config"] for row in rows],
        }
        _ensure_text(
            root / "shards" / f"{shard_id}.json",
            _canonical_json(shard_payload),
            verify_only=verify_only,
        )

    _ensure_text(
        root / "run_index.json", _canonical_json(run_index), verify_only=verify_only
    )
    _ensure_text(
        root / "injections" / "index.json",
        _canonical_json(injection_index),
        verify_only=verify_only,
    )
    if integrity["shard_count"] != len(by_shard):
        raise RuntimeError("Stage-9 shard count changed while preparing deployment")
    _ensure_text(
        root / "integrity" / "setup_manifest.json",
        _canonical_json(integrity),
        verify_only=verify_only,
    )
    return integrity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--refresh-execution-contract", action="store_true")
    args = parser.parse_args()
    summary = prepare(
        args.project_root,
        verify_only=args.verify_only,
        refresh_execution_contract=args.refresh_execution_contract,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
