#!/usr/bin/env python3
"""Stage shared Stage-9 reference trees without downloading any data.

The default symlink mode avoids duplicate multi-gigabyte opacity stores.  A
SHA-256 inventory is written so preflight and archival checks can detect drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise RuntimeError(f"reference source contains no files: {root}")
    return rows


def _stage(source: Path, destination: Path, mode: str) -> None:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"reference source is not a directory: {source}")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            f"refusing to overwrite staged reference target: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    else:
        shutil.copytree(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--picaso-refdata", type=Path, required=True)
    parser.add_argument("--picaso-ck", type=Path, required=True)
    parser.add_argument("--prt-input-data", type=Path, required=True)
    parser.add_argument("--robert-opacity", type=Path, required=True)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    if os.environ.get("STAGE9_CLUSTER") != "glamdring":
        raise RuntimeError("Stage-9 reference staging is restricted to Glamdring")
    project = args.project_root.expanduser().resolve()
    sources = {
        "picaso_refdata": args.picaso_refdata,
        "picaso_resort_rebin": args.picaso_ck,
        "petitradtrans_input_data": args.prt_input_data,
        "robert_opacity": args.robert_opacity,
    }
    targets = {
        "picaso_refdata": project / "reference" / "picaso" / "refdata",
        "picaso_resort_rebin": project / "reference" / "picaso" / "resortrebin",
        "petitradtrans_input_data": project
        / "reference"
        / "petitradtrans"
        / "input_data",
        "robert_opacity": project / "reference" / "robert" / "opacity",
    }
    payload: dict[str, Any] = {"schema_version": "1.0", "mode": args.mode, "trees": {}}
    for name, source in sources.items():
        resolved = source.expanduser().resolve()
        files = _manifest(resolved)
        _stage(resolved, targets[name], args.mode)
        payload["trees"][name] = {
            "source": str(resolved),
            "target": str(targets[name]),
            "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        }
    destination = project / "integrity" / "reference_data_manifest.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
