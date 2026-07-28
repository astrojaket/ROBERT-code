"""Download checksum-pinned ExoMolOP R=1000 tables used by ROBERT."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import BinaryIO
from urllib.request import Request, urlopen

from robert_exoplanets._data import (
    BUNDLED_K_TABLE_SPECIES,
    bundled_opacity_manifest,
)
from robert_exoplanets.core import RobertDataError, RobertValidationError


def download_exomol_r1000(
    directory: str | Path,
    *,
    species: tuple[str, ...] | list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download the R=1000 parents of ROBERT's bundled molecular tables."""

    selected = BUNDLED_K_TABLE_SPECIES if species is None else tuple(species)
    if not selected or any(item not in BUNDLED_K_TABLE_SPECIES for item in selected):
        raise RobertValidationError(
            "R=1000 downloads must use bundled species: "
            + ", ".join(BUNDLED_K_TABLE_SPECIES)
        )
    if len(set(selected)) != len(selected):
        raise RobertValidationError("download species must not contain duplicates")

    root = Path(directory).expanduser()
    target_directory = root if root.name.casefold() == "r1000" else root / "R1000"
    target_directory.mkdir(parents=True, exist_ok=True)
    products = bundled_opacity_manifest()["products"]
    downloaded: dict[str, Path] = {}
    for molecule in selected:
        product = products[molecule]
        target = target_directory / f"{molecule}_R1000.kta"
        expected_sha256 = str(product["source_sha256"])
        if target.is_file() and _sha256(target) == expected_sha256:
            downloaded[molecule] = target
            continue
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{target} exists but does not match the expected checksum; "
                "pass overwrite=True to replace it"
            )
        temporary = target.with_name(f".{target.name}.part")
        if temporary.exists():
            temporary.unlink()
        request = Request(
            str(product["source_url"]),
            headers={"User-Agent": "robert-exoplanets opacity downloader"},
        )
        try:
            with urlopen(request) as response, temporary.open("wb") as handle:
                _copy_stream(response, handle)
            actual_sha256 = _sha256(temporary)
            if actual_sha256 != expected_sha256:
                raise RobertDataError(
                    f"downloaded {molecule} checksum is {actual_sha256}, "
                    f"expected {expected_sha256}"
                )
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        downloaded[molecule] = target
    return downloaded


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        destination.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("opacity_data") / "ktables_exomol",
        help="opacity root; files are written below its R1000 directory",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        choices=BUNDLED_K_TABLE_SPECIES,
        default=list(BUNDLED_K_TABLE_SPECIES),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    paths = download_exomol_r1000(
        args.directory,
        species=args.species,
        overwrite=args.overwrite,
    )
    for molecule, path in paths.items():
        print(f"{molecule}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
