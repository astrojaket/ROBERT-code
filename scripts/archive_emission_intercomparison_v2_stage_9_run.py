#!/usr/bin/env python3
"""Archive one completed Stage-9 MultiNest chain with integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("archive_directory", type=Path)
    parser.add_argument(
        "--remove-raw-after-verified-archive",
        action="store_true",
        help="delete only chains/ after archive re-open and checksum verification succeeds",
    )
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    chains = run / "chains"
    required = (
        run / "result.json",
        run / "result_arrays.npz",
        run / "posterior_summary.json",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("only completed, compacted Stage-9 runs may be archived")
    summary = json.loads((run / "posterior_summary.json").read_text(encoding="utf-8"))
    if not summary.get("converged"):
        raise RuntimeError(
            "failed or incomplete native restart files must remain in place"
        )
    if not chains.is_dir() or not any(path.is_file() for path in chains.rglob("*")):
        raise RuntimeError("MultiNest chains directory is missing or empty")
    destination = (
        args.archive_directory.expanduser().resolve() / f"{run.name}.chains.tar.gz"
    )
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(chains, arcname="chains", recursive=True)
    with tarfile.open(destination, "r:gz") as archive:
        if not any(item.isfile() for item in archive.getmembers()):
            raise RuntimeError("created chain archive contains no regular files")
    digest = _sha256(destination)
    checksum = destination.with_suffix(destination.suffix + ".sha256")
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    if _sha256(destination) != digest:
        raise RuntimeError("archive checksum changed before cleanup")
    if args.remove_raw_after_verified_archive:
        shutil.rmtree(chains)
    print(
        json.dumps(
            {
                "archive": str(destination),
                "sha256": digest,
                "raw_removed": args.remove_raw_after_verified_archive,
            }
        )
    )


if __name__ == "__main__":
    main()
