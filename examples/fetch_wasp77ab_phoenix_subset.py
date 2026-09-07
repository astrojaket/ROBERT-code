"""Acquire the minimal official STScI PHOENIX subset for WASP-77Ab.

The files are external reference data.  This script downloads each file as a
bounded byte stream, verifies its exact size and SHA-256 digest, and installs
it below ``PYSYN_CDBS/grid/phoenix`` with an atomic rename.  An existing file
is skipped only when it passes both checks.  An invalid existing file is
never overwritten automatically.

The script does not import numerical libraries or read FITS content.  It
only acquires the named files as opaque byte streams.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PHOENIX_BASE_URL = "https://ssb.stsci.edu/cdbs/grid/phoenix/"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "external_data" / "wasp77ab_phoenix_subset"
DEFAULT_CHUNK_BYTES = 1024 * 1024
MAX_THREADS = 6
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


def _clamp_thread_environment() -> None:
    """Clamp numerical thread controls before any optional import occurs."""

    for name in THREAD_VARIABLES:
        raw = os.environ.get(name, str(MAX_THREADS))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = MAX_THREADS
        os.environ[name] = str(min(MAX_THREADS, max(1, value)))


_clamp_thread_environment()


@dataclass(frozen=True)
class PhoenixFileSpec:
    """Official identity and role for one PHOENIX file."""

    name: str
    size_bytes: int
    sha256: str
    role: str

    @property
    def content_url(self) -> str:
        """Return the official STScI file URL."""

        return f"{PHOENIX_BASE_URL}{self.name}"


PHOENIX_FILES: tuple[PhoenixFileSpec, ...] = (
    PhoenixFileSpec(
        name="catalog.fits",
        size_bytes=1_497_600,
        sha256="c5cb54e12ca20f8cba38624c460b743d0783ab92fcc0ad53b8288819d1d4d471",
        role="official STScI PHOENIX grid catalogue",
    ),
    PhoenixFileSpec(
        name="phoenixm00/phoenixm00_5600.fits",
        size_bytes=9_901_440,
        sha256="688a5acf69c9ec65a1b5398084d6c62e43ad63d2799588e5a005f715123686ca",
        role="official STScI PHOENIX [M/H]=0.0, 5600 K spectrum",
    ),
    PhoenixFileSpec(
        name="phoenixm00/phoenixm00_5700.fits",
        size_bytes=9_901_440,
        sha256="373a945f3c16f36473fdca542d5a97c090a16871ad598f7b069fd15bb4bbe027",
        role="official STScI PHOENIX [M/H]=0.0, 5700 K spectrum",
    ),
)


@dataclass(frozen=True)
class LocalDigest:
    """Measured local file identity."""

    size_bytes: int
    sha256: str


class AcquisitionError(RuntimeError):
    """Raised when a PHOENIX file fails acquisition or verification."""


def file_digest(path: Path, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> LocalDigest:
    """Compute size and SHA-256 with bounded memory."""

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise AcquisitionError(f"cannot read local file: {path}") from error
    return LocalDigest(size, digest.hexdigest())


def verify_local_file(
    path: Path,
    spec: PhoenixFileSpec,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> LocalDigest:
    """Verify one local file against the pinned official identity."""

    if not path.is_file():
        raise AcquisitionError(f"expected a regular file: {path}")
    digest = file_digest(path, chunk_bytes=chunk_bytes)
    if digest.size_bytes != spec.size_bytes:
        raise AcquisitionError(
            f"{spec.name} size mismatch: expected {spec.size_bytes}, "
            f"got {digest.size_bytes}"
        )
    if digest.sha256 != spec.sha256:
        raise AcquisitionError(
            f"{spec.name} SHA-256 mismatch: expected {spec.sha256}, "
            f"got {digest.sha256}"
        )
    return digest


def _safe_output_directory(path: str | Path) -> Path:
    """Create a dedicated output root and reject broad filesystem targets."""

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
        raise AcquisitionError(f"cannot create output directory: {directory}") from error
    return directory


def _download_verified(
    spec: PhoenixFileSpec,
    target: Path,
    *,
    opener: Callable[..., object] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout_seconds: float = 120.0,
) -> LocalDigest:
    """Stream one file, verify it, and atomically install it."""

    if target.exists():
        raise AcquisitionError(f"download target already exists: {target}")
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    temporary_path: Path | None = None
    digest = sha256()
    size = 0
    download_opener = urlopen if opener is None else opener
    try:
        response = download_opener(spec.content_url, timeout=timeout_seconds)
        with response as stream:
            target.parent.mkdir(parents=True, exist_ok=True)
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
                    digest.update(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
        measured = LocalDigest(size, digest.hexdigest())
        if measured.size_bytes != spec.size_bytes:
            raise AcquisitionError(
                f"{spec.name} download size mismatch: expected {spec.size_bytes}, "
                f"got {measured.size_bytes}"
            )
        if measured.sha256 != spec.sha256:
            raise AcquisitionError(
                f"{spec.name} download SHA-256 mismatch: expected {spec.sha256}, "
                f"got {measured.sha256}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        return measured
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
    spec: PhoenixFileSpec,
    target: Path,
    *,
    status: str,
    digest: LocalDigest | None,
) -> dict[str, object]:
    """Build one compact provenance record."""

    return {
        "name": spec.name,
        "role": spec.role,
        "content_url": spec.content_url,
        "size_bytes": spec.size_bytes,
        "sha256_expected": spec.sha256,
        "local_path": str(target),
        "status": status,
        "sha256": None if digest is None else digest.sha256,
    }


def acquire_files(
    output_directory: str | Path,
    *,
    opener: Callable[..., object] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout_seconds: float = 120.0,
) -> tuple[dict[str, object], ...]:
    """Acquire the three pinned files below a dedicated PYSYN_CDBS root."""

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    directory = _safe_output_directory(output_directory)
    records: list[dict[str, object]] = []
    for spec in PHOENIX_FILES:
        target = directory / "grid" / "phoenix" / spec.name
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
) -> dict[str, object]:
    """Build a compact JSON-serialisable acquisition report."""

    return {
        "schema_version": 1,
        "record_type": "wasp77ab_stsci_phoenix_subset_acquisition",
        "status": "pass",
        "target": "WASP-77Ab",
        "source": {
            "provider": "Space Telescope Science Institute",
            "official_base_url": PHOENIX_BASE_URL,
            "tree": "grid/phoenix",
            "purpose": "minimal PHOENIX subset for the real ROBERT stellar model",
        },
        "acquisition": {
            "output_directory": str(output_directory),
            "pysyn_cdbs_root": str(output_directory),
            "atomic_install": True,
            "skip_valid": True,
            "overwrite_invalid": False,
            "stream_chunk_bytes": DEFAULT_CHUNK_BYTES,
            "reads_fits_content": False,
        },
        "resource_policy": {
            "max_threads": MAX_THREADS,
            "thread_values": {
                name: int(os.environ[name]) for name in THREAD_VARIABLES
            },
            "thread_variables": list(THREAD_VARIABLES),
            "process_rss_limit_bytes": PROCESS_RSS_LIMIT_BYTES,
            "process_rss_limit_comparison": "strictly below 2 GiB",
        },
        "files": [dict(record) for record in records],
    }


def write_report(path: str | Path, report: Mapping[str, object]) -> Path:
    """Write a report with an atomic replacement."""

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


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit acquisition command parser."""

    parser = argparse.ArgumentParser(
        description="Acquire the official minimal STScI PHOENIX subset for WASP-77Ab."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="dedicated PYSYN_CDBS root for the installed grid/phoenix tree",
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
    """Run the bounded acquisition command."""

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
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            timeout_seconds=args.timeout_seconds,
        )
        report = build_report(records, output_directory=output_directory)
        manifest = (
            output_directory / "acquisition_manifest.json"
            if args.manifest is None
            else args.manifest.expanduser().resolve()
        )
        write_report(manifest, report)
    except (AcquisitionError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Verified {len(records)} PHOENIX files; report: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AcquisitionError",
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_OUTPUT_DIRECTORY",
    "LocalDigest",
    "MAX_THREADS",
    "PHOENIX_BASE_URL",
    "PHOENIX_FILES",
    "PROCESS_RSS_LIMIT_BYTES",
    "PhoenixFileSpec",
    "THREAD_VARIABLES",
    "acquire_files",
    "build_parser",
    "build_report",
    "file_digest",
    "main",
    "verify_local_file",
    "write_report",
]
