"""Tests for the bounded STScI PHOENIX subset acquisition helper."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import pytest

from examples import fetch_wasp77ab_phoenix_subset as fetch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "docs" / "data" / "wasp77ab_phoenix_sources.json"


EXPECTED = {
    "catalog.fits": (
        1_497_600,
        "c5cb54e12ca20f8cba38624c460b743d0783ab92fcc0ad53b8288819d1d4d471",
    ),
    "phoenixm00/phoenixm00_5600.fits": (
        9_901_440,
        "688a5acf69c9ec65a1b5398084d6c62e43ad63d2799588e5a005f715123686ca",
    ),
    "phoenixm00/phoenixm00_5700.fits": (
        9_901_440,
        "373a945f3c16f36473fdca542d5a97c090a16871ad598f7b069fd15bb4bbe027",
    ),
}


def test_official_subset_specs_are_exact() -> None:
    specs = {spec.name: spec for spec in fetch.PHOENIX_FILES}

    assert specs.keys() == EXPECTED.keys()
    assert fetch.PHOENIX_BASE_URL == "https://ssb.stsci.edu/cdbs/grid/phoenix/"
    for name, (size_bytes, digest) in EXPECTED.items():
        assert specs[name].size_bytes == size_bytes
        assert specs[name].sha256 == digest
        assert specs[name].content_url == f"{fetch.PHOENIX_BASE_URL}{name}"


def test_streamed_file_verification_and_atomic_install(tmp_path: Path) -> None:
    payload = b"small PHOENIX fixture\n"
    spec = fetch.PhoenixFileSpec(
        name="catalog.fits",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        role="test fixture",
    )

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def opener(*_args: object, **_kwargs: object) -> Response:
        return Response(payload)

    target = tmp_path / "grid" / "phoenix" / spec.name
    digest = fetch._download_verified(spec, target, opener=opener, chunk_bytes=3)

    assert target.read_bytes() == payload
    assert digest == fetch.LocalDigest(len(payload), spec.sha256)
    assert not tuple(target.parent.glob("*.part"))
    assert fetch.verify_local_file(target, spec, chunk_bytes=2) == digest


def test_invalid_existing_file_is_never_overwritten(tmp_path: Path) -> None:
    payload = b"good"
    spec = fetch.PhoenixFileSpec(
        name="catalog.fits",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        role="test fixture",
    )
    target = tmp_path / "grid" / "phoenix" / spec.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")

    with pytest.raises(fetch.AcquisitionError, match="size mismatch"):
        fetch.acquire_files(tmp_path)
    assert target.read_bytes() == b"bad"


def test_valid_existing_files_are_skipped_without_network(tmp_path: Path, monkeypatch) -> None:
    payload = b"valid fixture"
    spec = fetch.PhoenixFileSpec(
        name="catalog.fits",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        role="test fixture",
    )
    target = tmp_path / "grid" / "phoenix" / spec.name
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    monkeypatch.setattr(fetch, "PHOENIX_FILES", (spec,))

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network access is not allowed for this test")

    records = fetch.acquire_files(tmp_path, opener=fail_if_called)

    assert len(records) == 1
    assert records[0]["status"] == "skipped_valid"
    assert records[0]["sha256"] == spec.sha256


def test_report_and_manifest_have_strict_resource_policy() -> None:
    report = fetch.build_report([], output_directory=Path("/tmp/phoenix-test"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    assert report["resource_policy"]["max_threads"] == 6
    assert report["resource_policy"]["process_rss_limit_bytes"] < 2 * 1024**3
    assert manifest["source"]["official_base_url"] == fetch.PHOENIX_BASE_URL
    assert manifest["resource_policy"]["max_threads"] == 6
    assert manifest["resource_policy"]["process_rss_limit_bytes"] < 2 * 1024**3
    by_name = {entry["name"]: entry for entry in manifest["files"]}
    for name, (size_bytes, digest) in EXPECTED.items():
        assert by_name[name]["size_bytes"] == size_bytes
        assert by_name[name]["sha256"] == digest
        assert by_name[name]["status"] == "not_downloaded"


def test_parser_requires_explicit_output_directory() -> None:
    with pytest.raises(SystemExit):
        fetch.build_parser().parse_args([])


def test_safe_output_directory_rejects_broad_targets() -> None:
    with pytest.raises(ValueError, match="dedicated"):
        fetch._safe_output_directory(Path("/"))
    with pytest.raises(ValueError, match="dedicated"):
        fetch._safe_output_directory(Path.home())
