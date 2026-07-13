#!/usr/bin/env python3
"""Snapshot NASA's JWST eclipse products and optionally download the tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import urlopen


HOST = "https://exoplanetarchive.ipac.caltech.edu"
TAP = f"{HOST}/TAP/sync"
QUERY = (
    "select * from spectra where spec_type='Eclipse' "
    "and facility like '%James Webb%' order by pl_name,bibcode"
)


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:  # noqa: S310 - fixed trusted host
        return response.read()


def archive_rows() -> list[dict[str, object]]:
    url = TAP + "?" + urlencode({"query": QUERY, "format": "json"})
    rows = json.loads(fetch(url))
    if not isinstance(rows, list):
        raise RuntimeError("NASA TAP response was not a JSON list")
    return rows


def firefly_data_bases() -> tuple[str, ...]:
    html = fetch(f"{HOST}/cgi-bin/atmospheres/nph-firefly?atmospheres").decode()
    match = re.search(r"FF_InitPage \('([^']+)', '([^']+)'", html)
    if match is None:
        raise RuntimeError("could not locate Firefly workspace paths")
    base_url, workspace = match.groups()
    return (
        f"{HOST}{workspace}/atmospheres/tab1/data/",
        f"{HOST}{base_url.rsplit('/tab1', 1)[0]}/data/",
        f"{HOST}{workspace}/atmospheres/data/",
        f"{HOST}{workspace}/data/",
    )


def download_rows(rows: list[dict[str, object]], root: Path) -> list[dict[str, str]]:
    bases = firefly_data_bases()
    manifest = []
    working_base = None
    for row in rows:
        relative = str(row["spec_path"])
        payload = None
        for base in (working_base,) if working_base else bases:
            if base is None:
                continue
            try:
                payload = fetch(base + relative)
                working_base = base
                break
            except Exception:  # Try the alternate public Firefly paths.
                continue
        if payload is None:
            raise RuntimeError(f"could not download archive spectrum: {relative}")
        target = root / "spectra" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        manifest.append(
            {
                "path": str(target.relative_to(root)),
                "sha256": sha256(payload).hexdigest(),
            }
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "jwst_emission_spectra",
    )
    parser.add_argument("--download-spectra", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = archive_rows()
    snapshot = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "query": QUERY,
        "source": TAP,
        "row_count": len(rows),
        "rows": rows,
    }
    (root / "archive_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.download_spectra:
        manifest = download_rows(rows, root)
        (root / "checksums.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"{len(rows)} archive products across {len({row['pl_name'] for row in rows})} targets"
    )


if __name__ == "__main__":
    main()
