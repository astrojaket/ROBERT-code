"""Acquire the Smith et al. (2024) WASP-77Ab data from Zenodo.

The files are external inputs.  This script downloads them one at a time,
streams each response through MD5 and SHA-256 digests, and atomically moves a
verified file into a user-selected directory.  Existing files that pass the
official size and MD5 checks are skipped.  Existing files that fail a check
are never overwritten automatically.

The script does not import numerical libraries or deserialize any archive.
The ``.pic`` cube is acquired as an opaque byte stream.  A later, separately
reviewed loader must verify and safely read it before scientific use.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
ZENODO_RECORD_URL = "https://zenodo.org/records/10382053"
ZENODO_API_FILES_URL = "https://zenodo.org/api/records/10382053/files"
ZENODO_DOI = "10.5281/zenodo.10382053"
LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEFAULT_DATA_DIRECTORY = ROOT / "external_data" / "wasp77ab_smith2024"
DEFAULT_CHUNK_BYTES = 1024 * 1024
MAX_THREADS = 3
PROCESS_RSS_LIMIT_BYTES = 2 * 1024**3 - 1
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)


@dataclass(frozen=True)
class DownloadSpec:
    """Official Zenodo metadata for one external file."""

    name: str
    size_bytes: int
    md5: str
    role: str
    optional: bool = False
    sha256: str | None = None

    @property
    def content_url(self) -> str:
        return f"{ZENODO_API_FILES_URL}/{self.name}/content"


PRIMARY_FILES: tuple[DownloadSpec, ...] = (
    DownloadSpec(
        "cube14v3.pic",
        77_591_894,
        "3fd41560b7dbcc60851ef3a0972c852f",
        "primary pre-eclipse IGRINS reduced cube",
        sha256="999623a7ba4460aa72ba9bbe284de92e7b5a811ebe09e51d70667084ae592b26",
    ),
    DownloadSpec(
        "20201214_info.csv",
        7_261,
        "7c861b1baf8ae87801f95d7af8b2e9c6",
        "primary pre-eclipse observing-condition metadata",
        sha256="68f5a80f930e5f1f99cd1861678754f824bf665972d675f617a7d66c4fdad11e",
    ),
    DownloadSpec(
        "w77_1DRC_FULL.txt",
        13_899_150,
        "dbaa138b7768b727c87ff8f83fcc563f",
        "full one-dimensional retrieval template",
        sha256="8f3fcdf119b42e1c1db5b5aba61c8266bad65cf3c279aa237e55346ed9be7a14",
    ),
    DownloadSpec(
        "w77_1DRC_H2O_ONLY.txt",
        13_899_150,
        "6cf0ec37cbed5db5b18e6d434838fa40",
        "H2O-only one-dimensional retrieval template",
        sha256="f68acda835e0af3e1747504d1e6243560610e5b98ad675917233859a17ace330",
    ),
    DownloadSpec(
        "w77_1DRC_CO_ONLY.txt",
        13_899_150,
        "d4f0c6dfd4bb50a7ca9ca73ea3318eb3",
        "CO-only one-dimensional retrieval template",
        sha256="b8c9fc1496b25d9b4030ac3fafac3623ac1a5e945cf0b2f3161e96d3e8e3c016",
    ),
    DownloadSpec(
        "w77_pre_nirspec_best_fit_R500K_scaled.txt",
        72_672_825,
        "93841125cab0a2c74ae4730ebf36b7dc",
        "pre-eclipse best-fit R=500,000 template scaled for NIRSpec",
        sha256="fe1fb7cd37b6dec52173a19ee1fd7ea1529c421946c76d728d0bdcf4285ad34d",
    ),
    DownloadSpec(
        "NIRSpec_posterior_draws_median_spectrum.txt",
        526_832,
        "e82b61acf01b56c5bc0ea79186680b09",
        "NIRSpec posterior-median spectrum",
        sha256="45858658108e62333226e16e13809f304666672803ffcdf089dabe3514b01e5a",
    ),
)

SENSITIVITY_NIGHT_FILES: tuple[DownloadSpec, ...] = (
    DownloadSpec(
        "cube06v3.pic",
        55_221_925,
        "fa338b902cfa5127fa4707187c2d00f0",
        "post-eclipse 2020-12-06 IGRINS sensitivity cube",
        optional=True,
    ),
    DownloadSpec(
        "20201206_info.csv",
        5_109,
        "ec2f3cbc0f0d1dc7a4cec599224cd186",
        "post-eclipse 2020-12-06 observing-condition metadata",
        optional=True,
    ),
    DownloadSpec(
        "cube21v3.pic",
        50_346_161,
        "53f3c67601475f0f5ffc8d1b9277c304",
        "post-eclipse 2020-12-21 IGRINS sensitivity cube",
        optional=True,
    ),
    DownloadSpec(
        "20201221_info.csv",
        4_529,
        "6bba581a3eaaa5bdd90286264ad202e0",
        "post-eclipse 2020-12-21 observing-condition metadata",
        optional=True,
    ),
)


@dataclass(frozen=True)
class LocalDigest:
    """Measured local file identity."""

    size_bytes: int
    md5: str
    sha256: str


class AcquisitionError(RuntimeError):
    """Raised when a downloaded or existing file fails verification."""


def file_digests(path: Path, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> LocalDigest:
    """Compute size, MD5, and SHA-256 with bounded memory."""

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest_md5 = md5()
    digest_sha256 = sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                size += len(chunk)
                digest_md5.update(chunk)
                digest_sha256.update(chunk)
    except OSError as error:
        raise AcquisitionError(f"cannot read local file: {path}") from error
    return LocalDigest(size, digest_md5.hexdigest(), digest_sha256.hexdigest())


def verify_local_file(
    path: Path,
    spec: DownloadSpec,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> LocalDigest:
    """Verify one local file against official size and MD5 metadata."""

    if not path.is_file():
        raise AcquisitionError(f"expected a regular file: {path}")
    digest = file_digests(path, chunk_bytes=chunk_bytes)
    if digest.size_bytes != spec.size_bytes:
        raise AcquisitionError(
            f"{spec.name} size mismatch: expected {spec.size_bytes}, "
            f"got {digest.size_bytes}"
        )
    if digest.md5 != spec.md5:
        raise AcquisitionError(
            f"{spec.name} MD5 mismatch: expected {spec.md5}, got {digest.md5}"
        )
    if spec.sha256 is not None and digest.sha256 != spec.sha256:
        raise AcquisitionError(
            f"{spec.name} SHA-256 mismatch: expected {spec.sha256}, "
            f"got {digest.sha256}"
        )
    return digest


def _safe_output_directory(path: str | Path) -> Path:
    """Create and return a dedicated output directory.

    The downloader writes only named files below this directory.  Rejecting
    the filesystem root and the user's home directory prevents an accidental
    broad target while still allowing an external data volume.
    """

    value = str(path).strip()
    if not value:
        raise ValueError("output directory must be explicit and non-empty")
    directory = Path(value).expanduser().resolve()
    if directory in {Path("/"), Path.home().resolve()}:
        raise ValueError(
            "output directory must be a dedicated data directory, not a broad root"
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AcquisitionError(
            f"cannot create output directory: {directory}"
        ) from error
    return directory


def _download_verified(
    spec: DownloadSpec,
    target: Path,
    *,
    opener: Callable[..., object] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout_seconds: float = 120.0,
) -> LocalDigest:
    """Stream, verify, and atomically install one Zenodo file."""

    if target.exists():
        raise AcquisitionError(f"download target already exists: {target}")
    temporary_path: Path | None = None
    digest_md5 = md5()
    digest_sha256 = sha256()
    size = 0
    download_opener = urlopen if opener is None else opener
    try:
        response = download_opener(spec.content_url, timeout=timeout_seconds)
        with response as stream:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := stream.read(chunk_bytes):
                    temporary.write(chunk)
                    size += len(chunk)
                    digest_md5.update(chunk)
                    digest_sha256.update(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
        digest = LocalDigest(size, digest_md5.hexdigest(), digest_sha256.hexdigest())
        if digest.size_bytes != spec.size_bytes:
            raise AcquisitionError(
                f"{spec.name} download size mismatch: expected {spec.size_bytes}, "
                f"got {digest.size_bytes}"
            )
        if digest.md5 != spec.md5:
            raise AcquisitionError(
                f"{spec.name} download MD5 mismatch: expected {spec.md5}, "
                f"got {digest.md5}"
            )
        if spec.sha256 is not None and digest.sha256 != spec.sha256:
            raise AcquisitionError(
                f"{spec.name} download SHA-256 mismatch: expected {spec.sha256}, "
                f"got {digest.sha256}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        return digest
    except AcquisitionError:
        raise
    except (OSError, TimeoutError) as error:
        raise AcquisitionError(f"failed to download {spec.name}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _record(
    spec: DownloadSpec,
    target: Path,
    *,
    status: str,
    digest: LocalDigest | None,
) -> dict[str, object]:
    values = {
        "name": spec.name,
        "role": spec.role,
        "optional": spec.optional,
        "content_url": spec.content_url,
        "size_bytes": spec.size_bytes,
        "md5": spec.md5,
        "sha256_expected": spec.sha256,
        "local_path": str(target),
        "status": status,
        "sha256": None if digest is None else digest.sha256,
    }
    return values


def acquire_files(
    output_directory: str | Path,
    *,
    include_sensitivity_nights: bool = False,
    opener: Callable[..., object] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout_seconds: float = 120.0,
) -> tuple[dict[str, object], ...]:
    """Acquire the selected Smith files and return provenance records.

    A valid existing file is reported as ``skipped_valid``.  An invalid
    existing file raises an error, so the user must select a new directory or
    inspect the file before retrying.  The function does not remove unrelated
    files from the output directory.
    """

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    directory = _safe_output_directory(output_directory)
    selected: Iterable[DownloadSpec] = PRIMARY_FILES
    if include_sensitivity_nights:
        selected = (*PRIMARY_FILES, *SENSITIVITY_NIGHT_FILES)
    records: list[dict[str, object]] = []
    for spec in selected:
        target = directory / spec.name
        if target.exists():
            digest = verify_local_file(target, spec, chunk_bytes=chunk_bytes)
            records.append(_record(spec, target, status="skipped_valid", digest=digest))
            continue
        digest = _download_verified(
            spec,
            target,
            opener=opener,
            chunk_bytes=chunk_bytes,
            timeout_seconds=timeout_seconds,
        )
        records.append(_record(spec, target, status="downloaded", digest=digest))
    return tuple(records)


def build_report(
    records: Iterable[Mapping[str, object]],
    *,
    output_directory: Path,
    include_sensitivity_nights: bool,
) -> dict[str, object]:
    """Build a compact, JSON-serialisable acquisition report."""

    record_list = [dict(record) for record in records]
    return {
        "schema_version": 1,
        "record_type": "smith2024_wasp77ab_external_data_acquisition",
        "status": "pass",
        "paper": {
            "authors": "Smith et al. 2024",
            "arxiv": "2312.13069",
            "doi": "10.3847/1538-3881/ad17bf",
        },
        "zenodo": {
            "record_url": ZENODO_RECORD_URL,
            "doi": ZENODO_DOI,
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "files_api_url": ZENODO_API_FILES_URL,
        },
        "acquisition": {
            "output_directory": str(output_directory),
            "include_sensitivity_nights": include_sensitivity_nights,
            "atomic_install": True,
            "skip_valid": True,
            "overwrite_invalid": False,
            "stream_chunk_bytes": DEFAULT_CHUNK_BYTES,
            "deserializes_archives": False,
        },
        "resource_policy": {
            "max_threads": MAX_THREADS,
            "observed_threads": 1,
            "process_rss_limit_bytes": PROCESS_RSS_LIMIT_BYTES,
            "process_rss_limit_comparison": "strictly below 2 GiB",
            "thread_variables": list(THREAD_VARIABLES),
        },
        "files": record_list,
    }


def write_report(path: str | Path, report: Mapping[str, object]) -> Path:
    """Write a JSON report with an atomic replace."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(report, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return destination


def build_parser():
    """Build the command-line parser."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Acquire verified Smith et al. WASP-77Ab Zenodo files."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="dedicated directory for the external files (required)",
    )
    parser.add_argument(
        "--include-sensitivity-nights",
        action="store_true",
        help="also acquire the 2020-12-06 and 2020-12-21 cube and CSV files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional JSON report path (default: OUTPUT_DIR/acquisition_manifest.json)",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=1,
        help="streaming download chunk size in MiB (default: 1)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="per-file network timeout (default: 120)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the explicit, resource-bounded acquisition command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_mib < 1:
        parser.error("--chunk-mib must be at least 1")
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    try:
        output_directory = _safe_output_directory(args.output_dir)
        records = acquire_files(
            output_directory,
            include_sensitivity_nights=args.include_sensitivity_nights,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            timeout_seconds=args.timeout_seconds,
        )
        report = build_report(
            records,
            output_directory=output_directory,
            include_sensitivity_nights=args.include_sensitivity_nights,
        )
        manifest = (
            output_directory / "acquisition_manifest.json"
            if args.manifest is None
            else args.manifest.expanduser().resolve()
        )
        write_report(manifest, report)
    except (AcquisitionError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Verified {len(records)} Smith et al. files; report: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AcquisitionError",
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_DATA_DIRECTORY",
    "DownloadSpec",
    "LICENSE",
    "LICENSE_URL",
    "MAX_THREADS",
    "PRIMARY_FILES",
    "PROCESS_RSS_LIMIT_BYTES",
    "SENSITIVITY_NIGHT_FILES",
    "THREAD_VARIABLES",
    "ZENODO_API_FILES_URL",
    "ZENODO_DOI",
    "ZENODO_RECORD_URL",
    "acquire_files",
    "build_parser",
    "build_report",
    "file_digests",
    "main",
    "verify_local_file",
    "write_report",
]
