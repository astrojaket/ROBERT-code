"""Tests for explicit ExoMolOP opacity downloads."""

from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path

import pytest

import robert_exoplanets.opacity.download as download_module
from robert_exoplanets.core import RobertDataError


def test_r1000_downloader_writes_expected_resolution_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic-kta"
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = {
        "products": {
            "H2O": {
                "source_url": "https://example.test/H2O.kta",
                "source_sha256": checksum,
            }
        }
    }
    monkeypatch.setattr(download_module, "bundled_opacity_manifest", lambda: manifest)
    monkeypatch.setattr(download_module, "urlopen", lambda _request: BytesIO(payload))

    paths = download_module.download_exomol_r1000(
        tmp_path / "opacities",
        species=["H2O"],
    )

    expected = tmp_path / "opacities" / "R1000" / "H2O_R1000.kta"
    assert paths == {"H2O": expected}
    assert expected.read_bytes() == payload


def test_r1000_downloader_removes_bad_partial_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "products": {
            "H2O": {
                "source_url": "https://example.test/H2O.kta",
                "source_sha256": hashlib.sha256(b"expected").hexdigest(),
            }
        }
    }
    monkeypatch.setattr(download_module, "bundled_opacity_manifest", lambda: manifest)
    monkeypatch.setattr(download_module, "urlopen", lambda _request: BytesIO(b"wrong"))

    with pytest.raises(RobertDataError, match="checksum"):
        download_module.download_exomol_r1000(
            tmp_path / "opacities",
            species=["H2O"],
        )

    target = tmp_path / "opacities" / "R1000" / "H2O_R1000.kta"
    assert not target.exists()
    assert not target.with_name(f".{target.name}.part").exists()
