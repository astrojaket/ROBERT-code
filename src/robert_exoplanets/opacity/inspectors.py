"""Lightweight opacity file and directory inspectors.

Inspectors should read only enough data to build metadata. They must not load
large opacity arrays into memory or perform radiative-transfer calculations.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping

from robert_exoplanets.core import RobertDataError, RobertValidationError

from .metadata import (
    GridCoverage,
    OpacityDatabase,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityMode,
    OpacityStorageFormat,
    SpectralCoverage,
)


_EXOMOL_LINE_LIST_SUFFIXES = {".states", ".trans"}
_EXOMOL_AUXILIARY_SUFFIXES = {".def", ".pf", ".broad"}
_CROSS_SECTION_SUFFIXES = {".xsec", ".sigma", ".cross", ".cross-section", ".cross_section"}
_KTABLE_SUFFIXES = {".kta", ".ktable", ".ktab", ".h5", ".hdf5"}
_NUMERIC_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def inspect_exomol_directory(
    path: str | Path,
    *,
    species: str,
    source: OpacityDataSource | str = OpacityDataSource.EXOMOL,
) -> OpacityDatabase:
    """Inspect an ExoMol/ExoMolOP-style directory without loading arrays.

    The inspector identifies line-list, cross-section, and k-table families
    from file suffixes. Coverage remains unknown unless a later specialized
    reader extracts it from a supported manifest or header.
    """

    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise RobertDataError(f"opacity directory does not exist: {directory}")

    files = tuple(item for item in directory.iterdir() if item.is_file())
    suffixes = {item.suffix.lower() for item in files}
    products: list[OpacityDataProduct] = []
    if _EXOMOL_LINE_LIST_SUFFIXES.issubset(suffixes):
        line_list_files = tuple(
            str(item)
            for item in files
            if item.suffix.lower() in _EXOMOL_LINE_LIST_SUFFIXES | _EXOMOL_AUXILIARY_SUFFIXES
        )
        products.append(
            OpacityDataProduct(
                species=(species,),
                mode=OpacityMode.LINE_BY_LINE,
                source=source,
                storage_format=OpacityStorageFormat.EXOMOL_LINE_LIST,
                path=str(directory),
                metadata={
                    "files": json.dumps(sorted(Path(item).name for item in line_list_files)),
                    "coverage": "unknown",
                },
            )
        )
    for item in sorted(files):
        suffix = item.suffix.lower()
        if suffix in _CROSS_SECTION_SUFFIXES:
            products.append(
                _file_product(
                    item,
                    species=(species,),
                    mode=OpacityMode.OPACITY_SAMPLING,
                    source=source,
                    storage_format=OpacityStorageFormat.EXOMOL_CROSS_SECTION,
                    metadata={"coverage": "unknown"},
                )
            )
        if suffix in _KTABLE_SUFFIXES:
            storage_format = (
                OpacityStorageFormat.NEMESIS_KTA
                if suffix == ".kta"
                else OpacityStorageFormat.EXOMOL_KTABLE
            )
            products.append(
                _file_product(
                    item,
                    species=(species,),
                    mode=OpacityMode.CORRELATED_K,
                    source=OpacityDataSource.EXOMOL_OP if storage_format == OpacityStorageFormat.NEMESIS_KTA else source,
                    storage_format=storage_format,
                    metadata={"coverage": "unknown"},
                )
            )
    if not products:
        raise RobertDataError(f"no recognized ExoMol opacity products found in {directory}")
    return OpacityDatabase(
        products=tuple(products),
        name=f"{species}-exomol-opacity",
        root=str(directory),
        metadata={"source": str(source.value if isinstance(source, OpacityDataSource) else source)},
    )


def inspect_kta_file(
    path: str | Path,
    *,
    species: str | None = None,
    source: OpacityDataSource | str = OpacityDataSource.EXOMOL_OP,
    spectral_coverage: SpectralCoverage | None = None,
    grid_coverage: GridCoverage | None = None,
    g_ordinates: int | None = None,
) -> OpacityDataProduct:
    """Inspect a NEMESIS-style `.kta` correlated-k file.

    ExoMolOP/exo_k products can use the NEMESIS `.kta` storage format, so the
    default source records ExoMolOP while preserving the binary format name.
    Coverage is accepted as explicit metadata until ROBERT has a fully validated
    `.kta` header reader.
    """

    item = Path(path).expanduser()
    inferred_species = species or _species_from_filename(item)
    return _file_product(
        item,
        species=(inferred_species,),
        mode=OpacityMode.CORRELATED_K,
        source=source,
        storage_format=OpacityStorageFormat.NEMESIS_KTA,
        spectral_coverage=spectral_coverage,
        grid_coverage=grid_coverage,
        g_ordinates=g_ordinates,
        metadata={"coverage": "explicit" if spectral_coverage or grid_coverage else "unknown"},
    )


def inspect_hitran_par_file(
    path: str | Path,
    *,
    species: str | None = None,
    max_lines: int | None = None,
) -> OpacityDataProduct:
    """Inspect a HITRAN fixed-width `.par` line list file.

    This reads line centers only, enough to infer line-by-line spectral
    coverage. It does not calculate cross sections or line shapes.
    """

    item = Path(path).expanduser()
    _require_file(item)
    wavenumbers: list[float] = []
    molecule_ids: set[str] = set()
    isotopologue_ids: set[str] = set()
    with item.open("rt", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if max_lines is not None and index >= max_lines:
                break
            if not line.strip():
                continue
            parsed = _parse_hitran_par_line(line)
            if parsed is None:
                continue
            molecule_id, isotopologue_id, wavenumber = parsed
            molecule_ids.add(molecule_id)
            isotopologue_ids.add(isotopologue_id)
            wavenumbers.append(wavenumber)
    if not wavenumbers:
        raise RobertDataError(f"no HITRAN line centers could be read from {item}")
    species_name = species or _species_from_filename(item, fallback=f"HITRAN-{sorted(molecule_ids)[0]}")
    return _file_product(
        item,
        species=(species_name,),
        mode=OpacityMode.LINE_BY_LINE,
        source=OpacityDataSource.HITRAN,
        storage_format=OpacityStorageFormat.HITRAN_PAR,
        spectral_coverage=SpectralCoverage(
            min_value=min(wavenumbers),
            max_value=max(wavenumbers),
            unit="cm^-1",
            n_points=len(wavenumbers),
        ),
        metadata={
            "molecule_ids": ",".join(sorted(molecule_ids)),
            "isotopologue_ids": ",".join(sorted(isotopologue_ids)),
            "inspected_lines": str(len(wavenumbers)),
        },
    )


def inspect_hitran_cia_file(
    path: str | Path,
    *,
    pair: tuple[str, str] | None = None,
) -> OpacityDataProduct:
    """Inspect a HITRAN CIA ASCII file header.

    HITRAN CIA files contain multiple temperature/band sets. This parser
    recognizes header-like lines that start with the collisional pair name and
    extracts spectral and temperature ranges without reading coefficient arrays.
    """

    item = Path(path).expanduser()
    _require_file(item)
    species_pair = pair or _cia_pair_from_filename(item)
    pair_label = "-".join(species_pair)
    wavenumber_min: list[float] = []
    wavenumber_max: list[float] = []
    temperatures: list[float] = []
    header_count = 0
    with item.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not _line_starts_with_pair(line, species_pair):
                continue
            numbers = [float(match.group()) for match in _NUMERIC_PATTERN.finditer(_cia_numeric_tail(line, species_pair))]
            if len(numbers) < 3:
                continue
            header_count += 1
            lower, upper = sorted((numbers[0], numbers[1]))
            wavenumber_min.append(lower)
            wavenumber_max.append(upper)
            temperatures.append(numbers[2])
    spectral = None
    grid = None
    if wavenumber_min and wavenumber_max:
        spectral = SpectralCoverage(
            min_value=min(wavenumber_min),
            max_value=max(wavenumber_max),
            unit="cm^-1",
        )
    if temperatures:
        grid = GridCoverage(
            temperature_min=min(temperatures),
            temperature_max=max(temperatures),
            temperature_unit="K",
            n_temperature=len(set(temperatures)),
        )
    return _file_product(
        item,
        species=species_pair,
        mode=OpacityMode.CIA,
        source=OpacityDataSource.HITRAN,
        storage_format=OpacityStorageFormat.HITRAN_CIA,
        spectral_coverage=spectral,
        grid_coverage=grid,
        cia_pair=species_pair,
        metadata={
            "pair": pair_label,
            "header_sets": str(header_count),
            "coverage": "unknown" if spectral is None else "header",
        },
    )


def inspect_robert_npz_archive(path: str | Path) -> OpacityDatabase:
    """Read ROBERT compressed archive metadata from a `.npz` manifest."""

    from .archive import inspect_robert_npz_archive as inspect_archive

    return inspect_archive(path)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA256 checksum for a local file."""

    item = Path(path).expanduser()
    _require_file(item)
    digest = sha256()
    with item.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_product(
    path: Path,
    *,
    species: Iterable[str],
    mode: OpacityMode | str,
    source: OpacityDataSource | str,
    storage_format: OpacityStorageFormat | str,
    spectral_coverage: SpectralCoverage | None = None,
    grid_coverage: GridCoverage | None = None,
    cia_pair: tuple[str, str] | None = None,
    g_ordinates: int | None = None,
    metadata: Mapping[str, str] | None = None,
) -> OpacityDataProduct:
    _require_file(path)
    return OpacityDataProduct(
        species=tuple(species),
        mode=mode,
        source=source,
        storage_format=storage_format,
        path=str(path),
        spectral_coverage=spectral_coverage,
        grid_coverage=grid_coverage,
        cia_pair=cia_pair,
        g_ordinates=g_ordinates,
        file_size_bytes=path.stat().st_size,
        checksum_sha256=file_sha256(path),
        metadata={} if metadata is None else metadata,
    )


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise RobertDataError(f"opacity file does not exist: {path}")


def _species_from_filename(path: Path, *, fallback: str | None = None) -> str:
    stem = path.name.split(".")[0]
    token = re.split(r"[_\s]", stem)[0].strip()
    if token:
        return token
    if fallback:
        return fallback
    raise RobertValidationError(f"could not infer species from filename: {path.name}")


def _cia_pair_from_filename(path: Path) -> tuple[str, str]:
    stem = path.name.split(".")[0]
    first = re.split(r"[_\s]", stem)[0].strip()
    parts = tuple(part for part in re.split(r"[-–—]", first) if part)
    if len(parts) != 2:
        raise RobertValidationError(f"could not infer CIA pair from filename: {path.name}")
    return (parts[0], parts[1])


def _line_starts_with_pair(line: str, pair: tuple[str, str]) -> bool:
    normalized = line.strip().replace("–", "-").replace("—", "-")
    return normalized.startswith("-".join(pair)) or normalized.startswith(f"{pair[0]} {pair[1]}")


def _cia_numeric_tail(line: str, pair: tuple[str, str]) -> str:
    normalized = line.strip().replace("–", "-").replace("—", "-")
    for prefix in ("-".join(pair), f"{pair[0]} {pair[1]}"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _parse_hitran_par_line(line: str) -> tuple[str, str, float] | None:
    if len(line) < 15:
        return None
    molecule_id = line[0:2].strip()
    isotopologue_id = line[2:3].strip()
    wavenumber_text = line[3:15].strip()
    if not molecule_id or not isotopologue_id or not wavenumber_text:
        return None
    try:
        wavenumber = float(wavenumber_text)
    except ValueError:
        return None
    return molecule_id, isotopologue_id, wavenumber
