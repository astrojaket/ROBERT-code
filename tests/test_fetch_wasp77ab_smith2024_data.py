"""Tests for the Smith et al. (2024) data-acquisition script."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest

from examples import fetch_wasp77ab_smith2024_data as fetch


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external_data" / "wasp77ab_smith2024"
SOURCE_MANIFEST = ROOT / "docs" / "data" / "wasp77ab_smith2024_sources.json"


EXPECTED_SHA256 = {
    "cube14v3.pic": "999623a7ba4460aa72ba9bbe284de92e7b5a811ebe09e51d70667084ae592b26",
    "20201214_info.csv": "68f5a80f930e5f1f99cd1861678754f824bf665972d675f617a7d66c4fdad11e",
    "w77_1DRC_FULL.txt": "8f3fcdf119b42e1c1db5b5aba61c8266bad65cf3c279aa237e55346ed9be7a14",
    "w77_1DRC_H2O_ONLY.txt": "f68acda835e0af3e1747504d1e6243560610e5b98ad675917233859a17ace330",
    "w77_1DRC_CO_ONLY.txt": "b8c9fc1496b25d9b4030ac3fafac3623ac1a5e945cf0b2f3161e96d3e8e3c016",
    "w77_pre_nirspec_best_fit_R500K_scaled.txt": "fe1fb7cd37b6dec52173a19ee1fd7ea1529c421946c76d728d0bdcf4285ad34d",
    "NIRSpec_posterior_draws_median_spectrum.txt": "45858658108e62333226e16e13809f304666672803ffcdf089dabe3514b01e5a",
}


def _external_files_or_skip() -> None:
    missing = [
        spec.name for spec in fetch.PRIMARY_FILES if not (EXTERNAL / spec.name).is_file()
    ]
    if missing:
        pytest.skip("Smith et al. primary external files are not installed")


def test_primary_specs_match_official_metadata_and_local_sha256() -> None:
    _external_files_or_skip()
    specs = {spec.name: spec for spec in fetch.PRIMARY_FILES}

    assert set(specs) == set(EXPECTED_SHA256)
    for name, expected_sha256 in EXPECTED_SHA256.items():
        spec = specs[name]
        path = EXTERNAL / name
        digest = fetch.file_digests(path)
        assert digest.size_bytes == spec.size_bytes
        assert digest.md5 == spec.md5
        assert digest.sha256 == expected_sha256
        assert spec.sha256 == expected_sha256
        assert spec.content_url == (
            f"https://zenodo.org/api/records/10382053/files/{name}/content"
        )


def test_optional_sensitivity_specs_have_official_metadata() -> None:
    specs = {spec.name: spec for spec in fetch.SENSITIVITY_NIGHT_FILES}

    assert set(specs) == {
        "cube06v3.pic",
        "20201206_info.csv",
        "cube21v3.pic",
        "20201221_info.csv",
    }
    assert all(spec.optional for spec in specs.values())
    assert specs["cube06v3.pic"].size_bytes == 55_221_925
    assert specs["cube06v3.pic"].md5 == "fa338b902cfa5127fa4707187c2d00f0"
    assert specs["20201206_info.csv"].size_bytes == 5_109
    assert specs["20201206_info.csv"].md5 == "ec2f3cbc0f0d1dc7a4cec599224cd186"
    assert specs["cube21v3.pic"].size_bytes == 50_346_161
    assert specs["cube21v3.pic"].md5 == "53f3c67601475f0f5ffc8d1b9277c304"
    assert specs["20201221_info.csv"].size_bytes == 4_529
    assert specs["20201221_info.csv"].md5 == "6bba581a3eaaa5bdd90286264ad202e0"
    assert all(spec.sha256 is None for spec in specs.values())


def test_valid_primary_files_skip_without_network() -> None:
    _external_files_or_skip()

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network access is not allowed for this test")

    records = fetch.acquire_files(EXTERNAL, opener=fail_if_called)

    assert len(records) == len(fetch.PRIMARY_FILES)
    assert {record["status"] for record in records} == {"skipped_valid"}


def test_streamed_download_verifies_and_installs_atomically(tmp_path: Path) -> None:
    payload = b"small deterministic payload\n"
    spec = fetch.DownloadSpec(
        "tiny.dat",
        size_bytes=len(payload),
        md5=hashlib.md5(payload).hexdigest(),
        role="test fixture",
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def opener(*_args: object, **_kwargs: object) -> Response:
        return Response(payload)

    target = tmp_path / spec.name
    digest = fetch._download_verified(spec, target, opener=opener, chunk_bytes=3)

    assert target.read_bytes() == payload
    assert digest.sha256 == spec.sha256
    assert not tuple(tmp_path.glob("*.part"))
    assert fetch.file_digests(target) == digest


def test_invalid_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "tiny.dat"
    target.write_bytes(b"bad")
    spec = fetch.DownloadSpec(
        "tiny.dat",
        size_bytes=4,
        md5="098f6bcd4621d373cade4e832627b4f6",
        role="test fixture",
    )

    with pytest.raises(fetch.AcquisitionError, match="size mismatch"):
        fetch.verify_local_file(target, spec)
    assert target.read_bytes() == b"bad"


def test_report_and_committed_source_manifest_have_resource_policy() -> None:
    _external_files_or_skip()
    records = fetch.acquire_files(EXTERNAL, opener=lambda *_args, **_kwargs: None)
    report = fetch.build_report(
        records,
        output_directory=EXTERNAL.resolve(),
        include_sensitivity_nights=False,
    )
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    assert report["resource_policy"]["max_threads"] <= 3
    assert report["resource_policy"]["process_rss_limit_bytes"] < 2 * 1024**3
    assert report["acquisition"]["deserializes_archives"] is False
    assert source_manifest["zenodo"]["doi"] == "10.5281/zenodo.10382053"
    assert source_manifest["zenodo"]["license"] == "CC-BY-4.0"
    assert source_manifest["resource_policy"]["max_threads"] <= 3
    assert (
        source_manifest["resource_policy"]["process_rss_limit_bytes"]
        < 2 * 1024**3
    )
    by_name = {entry["name"]: entry for entry in source_manifest["files"]}
    assert by_name["cube14v3.pic"]["sha256"] == EXPECTED_SHA256["cube14v3.pic"]
    assert by_name["cube06v3.pic"]["status"] == "not_requested"


def test_cli_requires_an_explicit_output_directory() -> None:
    with pytest.raises(SystemExit):
        fetch.build_parser().parse_args([])
