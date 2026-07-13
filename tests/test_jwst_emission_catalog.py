"""Integrity tests for the local published JWST emission catalog."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1] / "data" / "jwst_emission_spectra"


def test_catalog_has_dated_scope_and_literature_additions() -> None:
    catalog = yaml.safe_load((ROOT / "catalog.yaml").read_text(encoding="utf-8"))
    publications = catalog["publications"]

    assert catalog["searched_through"] == "2026-07-13"
    assert len(publications) == 35
    assert len({item["target"] for item in publications}) == 27
    assert any(item["target"] == "WASP-121 b" for item in publications)
    assert any(item["target"] == "LHS 3844 b" for item in publications)


def test_downloaded_archive_snapshot_matches_checksums() -> None:
    snapshot = json.loads((ROOT / "archive_snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "checksums.json").read_text(encoding="utf-8"))

    assert snapshot["row_count"] == 74
    assert len({row["pl_name"] for row in snapshot["rows"]}) == 25
    assert len(manifest) == snapshot["row_count"]
    for item in manifest:
        path = ROOT / item["path"]
        assert path.is_file()
        assert sha256(path.read_bytes()).hexdigest() == item["sha256"]
