"""ROBERT-native opacity archive helpers.

The native fast-read candidate is a directory containing a JSON manifest and
plain NumPy `.npy` arrays. This keeps metadata readable, arrays memory-mappable,
and dependencies minimal. `.npz` archives are also supported as compact exchange
files, but they are not assumed to be the fastest runtime format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertDataError, RobertValidationError

from .metadata import (
    OpacityDatabase,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityStorageFormat,
)


ROBERT_OPACITY_ARCHIVE_VERSION = "0.1"
MANIFEST_FILENAME = "manifest.json"
NPZ_MANIFEST_KEY = "manifest_json"
ARRAY_PREFIX = "array__"


@dataclass(frozen=True)
class RobertOpacityArchive:
    """Loaded ROBERT opacity archive with metadata and arrays."""

    database: OpacityDatabase
    arrays: Mapping[str, NDArray[np.floating]]
    archive_format: OpacityStorageFormat | str
    version: str = ROBERT_OPACITY_ARCHIVE_VERSION
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.version:
            raise RobertValidationError("archive version must not be empty")
        archive_format = (
            self.archive_format
            if isinstance(self.archive_format, OpacityStorageFormat)
            else OpacityStorageFormat(str(self.archive_format))
        )
        arrays: dict[str, NDArray[np.floating]] = {}
        for name, values in self.arrays.items():
            if not name:
                raise RobertValidationError("archive array names must not be empty")
            array = values if isinstance(values, np.memmap) else np.asarray(values)
            if array.size == 0:
                raise RobertValidationError(f"archive array {name!r} must not be empty")
            if not np.issubdtype(array.dtype, np.number):
                raise RobertValidationError(f"archive array {name!r} must be numeric")
            if not np.all(np.isfinite(array)):
                raise RobertValidationError(f"archive array {name!r} must contain only finite values")
            arrays[str(name)] = array
        if not arrays:
            raise RobertValidationError("opacity archive must contain at least one array")
        object.__setattr__(self, "archive_format", archive_format)
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_manifest_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable archive manifest."""

        return _archive_manifest_mapping(
            database=self.database,
            arrays=self.arrays,
            archive_format=self.archive_format,
            version=self.version,
            metadata=self.metadata,
        )


def write_robert_npy_directory(
    path: str | Path,
    *,
    database: OpacityDatabase,
    arrays: Mapping[str, ArrayLike],
    metadata: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> RobertOpacityArchive:
    """Write a ROBERT native directory archive.

    Existing array and manifest files are overwritten only when `overwrite` is
    true. The function does not delete unrelated files.
    """

    directory = Path(path).expanduser()
    archive = RobertOpacityArchive(
        database=_database_with_archive_products(
            database,
            root=str(directory),
            storage_format=OpacityStorageFormat.ROBERT_NPY_DIRECTORY,
            compression="none",
        ),
        arrays=_validated_arrays(arrays),
        archive_format=OpacityStorageFormat.ROBERT_NPY_DIRECTORY,
        metadata={} if metadata is None else metadata,
    )
    if directory.exists() and not directory.is_dir():
        raise RobertDataError(f"archive path exists and is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    manifest_path = directory / MANIFEST_FILENAME
    array_paths = [_array_path(directory, name) for name in archive.arrays]
    existing = [item for item in (manifest_path, *array_paths) if item.exists()]
    if existing and not overwrite:
        raise RobertDataError("archive files already exist; pass overwrite=True to replace them")

    for name, array in archive.arrays.items():
        np.save(_array_path(directory, name), array, allow_pickle=False)
    manifest = archive.to_manifest_mapping()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return archive


def inspect_robert_npy_directory(path: str | Path) -> OpacityDatabase:
    """Read only metadata from a ROBERT native directory archive."""

    manifest = _read_directory_manifest(path)
    return OpacityDatabase.from_mapping(_manifest_database_mapping(manifest))


def load_robert_npy_directory(
    path: str | Path,
    *,
    mmap_mode: str | None = None,
) -> RobertOpacityArchive:
    """Load a ROBERT native directory archive.

    Pass `mmap_mode="r"` to memory-map arrays instead of eagerly reading them
    into RAM.
    """

    directory = Path(path).expanduser()
    manifest = _read_directory_manifest(directory)
    arrays: dict[str, NDArray[np.floating]] = {}
    for name, info in _manifest_arrays(manifest).items():
        filename = str(info["filename"])
        arrays[name] = np.load(directory / filename, mmap_mode=mmap_mode, allow_pickle=False)
    return RobertOpacityArchive(
        database=OpacityDatabase.from_mapping(_manifest_database_mapping(manifest)),
        arrays=arrays,
        archive_format=str(manifest.get("archive_format", OpacityStorageFormat.ROBERT_NPY_DIRECTORY.value)),
        version=str(manifest.get("version", ROBERT_OPACITY_ARCHIVE_VERSION)),
        metadata=_string_mapping(manifest.get("metadata", {}), "metadata"),
    )


def write_robert_npz_archive(
    path: str | Path,
    *,
    database: OpacityDatabase,
    arrays: Mapping[str, ArrayLike],
    compressed: bool = False,
    metadata: Mapping[str, str] | None = None,
) -> RobertOpacityArchive:
    """Write a ROBERT `.npz` archive with manifest and arrays."""

    archive_path = Path(path).expanduser()
    archive_arrays = _validated_arrays(arrays)
    archive = RobertOpacityArchive(
        database=_database_with_archive_products(
            database,
            root=str(archive_path),
            storage_format=OpacityStorageFormat.ROBERT_NPZ,
            compression="zip_deflate" if compressed else "zip_stored",
        ),
        arrays=archive_arrays,
        archive_format=OpacityStorageFormat.ROBERT_NPZ,
        metadata={} if metadata is None else metadata,
    )
    manifest_json = json.dumps(archive.to_manifest_mapping(), sort_keys=True)
    payload = {NPZ_MANIFEST_KEY: np.asarray(manifest_json)}
    payload.update({f"{ARRAY_PREFIX}{name}": array for name, array in archive.arrays.items()})
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(archive_path, **payload)
    else:
        np.savez(archive_path, **payload)
    return archive


def inspect_robert_npz_archive(path: str | Path) -> OpacityDatabase:
    """Read only metadata from a ROBERT `.npz` archive."""

    manifest = _read_npz_manifest(path)
    return OpacityDatabase.from_mapping(_manifest_database_mapping(manifest))


def load_robert_npz_archive(path: str | Path) -> RobertOpacityArchive:
    """Load a ROBERT `.npz` archive and its arrays."""

    manifest = _read_npz_manifest(path)
    archive_path = Path(path).expanduser()
    arrays: dict[str, NDArray[np.floating]] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for name in _manifest_arrays(manifest):
            arrays[name] = np.asarray(archive[f"{ARRAY_PREFIX}{name}"])
    return RobertOpacityArchive(
        database=OpacityDatabase.from_mapping(_manifest_database_mapping(manifest)),
        arrays=arrays,
        archive_format=str(manifest.get("archive_format", OpacityStorageFormat.ROBERT_NPZ.value)),
        version=str(manifest.get("version", ROBERT_OPACITY_ARCHIVE_VERSION)),
        metadata=_string_mapping(manifest.get("metadata", {}), "metadata"),
    )


def _archive_manifest_mapping(
    *,
    database: OpacityDatabase,
    arrays: Mapping[str, NDArray[np.floating]],
    archive_format: OpacityStorageFormat,
    version: str,
    metadata: Mapping[str, str],
) -> dict[str, object]:
    return {
        "format": "robert_opacity_archive",
        "version": version,
        "archive_format": archive_format.value,
        "database": database.to_mapping(),
        "arrays": {
            name: {
                "filename": f"{_safe_array_name(name)}.npy",
                "key": f"{ARRAY_PREFIX}{name}",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
        "metadata": dict(metadata),
    }


def _database_with_archive_products(
    database: OpacityDatabase,
    *,
    root: str,
    storage_format: OpacityStorageFormat,
    compression: str,
) -> OpacityDatabase:
    products = tuple(
        OpacityDataProduct(
            species=product.species,
            mode=product.mode,
            source=OpacityDataSource.ROBERT_ARCHIVE,
            storage_format=storage_format,
            path=root,
            spectral_coverage=product.spectral_coverage,
            grid_coverage=product.grid_coverage,
            cia_pair=product.cia_pair,
            g_ordinates=product.g_ordinates,
            compression=compression,
            native_shape=product.native_shape,
            metadata={
                **dict(product.metadata),
                "converted_from_source": product.source.value,
                "converted_from_format": product.storage_format.value,
            },
        )
        for product in database.products
    )
    return OpacityDatabase(
        products=products,
        name=database.name,
        root=root,
        metadata={**dict(database.metadata), "archive_format": storage_format.value},
    )


def _validated_arrays(arrays: Mapping[str, ArrayLike]) -> dict[str, NDArray[np.floating]]:
    if not arrays:
        raise RobertValidationError("arrays must contain at least one named array")
    validated: dict[str, NDArray[np.floating]] = {}
    for name, values in arrays.items():
        safe_name = _safe_array_name(name)
        array = np.asarray(values)
        if array.size == 0:
            raise RobertValidationError(f"array {name!r} must not be empty")
        if not np.issubdtype(array.dtype, np.number):
            raise RobertValidationError(f"array {name!r} must be numeric")
        if not np.all(np.isfinite(array)):
            raise RobertValidationError(f"array {name!r} must contain only finite values")
        validated[safe_name] = array
    return validated


def _safe_array_name(name: str) -> str:
    safe = str(name).strip()
    if not safe:
        raise RobertValidationError("array names must not be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(character not in allowed for character in safe):
        raise RobertValidationError("array names may contain only letters, numbers, '_' and '-'")
    return safe


def _array_path(directory: Path, name: str) -> Path:
    return directory / f"{_safe_array_name(name)}.npy"


def _read_directory_manifest(path: str | Path) -> Mapping[str, object]:
    directory = Path(path).expanduser()
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RobertDataError(f"ROBERT opacity archive manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_npz_manifest(path: str | Path) -> Mapping[str, object]:
    archive_path = Path(path).expanduser()
    if not archive_path.is_file():
        raise RobertDataError(f"ROBERT opacity archive does not exist: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        if NPZ_MANIFEST_KEY not in archive:
            raise RobertDataError("ROBERT opacity archive must contain manifest_json")
        raw = archive[NPZ_MANIFEST_KEY]
        manifest_json = str(raw.item() if raw.shape == () else raw.tolist())
    return json.loads(manifest_json)


def _manifest_database_mapping(manifest: Mapping[str, object]) -> Mapping[str, object]:
    database = manifest.get("database")
    if database is None:
        # Backward-compatible with early prototype archives that stored the
        # database mapping directly as manifest_json.
        return manifest
    if not isinstance(database, Mapping):
        raise RobertDataError("ROBERT opacity archive database manifest must be a mapping")
    return database


def _manifest_arrays(manifest: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping):
        raise RobertDataError("ROBERT opacity archive manifest must contain arrays")
    converted: dict[str, Mapping[str, object]] = {}
    for name, info in arrays.items():
        if not isinstance(info, Mapping):
            raise RobertDataError("ROBERT opacity archive array entries must be mappings")
        converted[str(name)] = info
    return converted


def _string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise RobertValidationError(f"{name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}
