"""Legacy `.kta` correlated-k reader.

The local HAT-P-32b k-tables were generated from ExoMol/ExoMolOP data and
written by `exo_k` in a `.kta` binary layout. ROBERT reads that format directly
so `.kta` can remain an import format rather than a native dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import RobertDataError, RobertValidationError

from .archive import write_robert_npy_directory, write_robert_npz_archive
from .metadata import (
    GridCoverage,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityDatabase,
    OpacityMode,
    OpacityStorageFormat,
    SpectralCoverage,
)


_INT_DTYPE = np.dtype("<i4")
_FLOAT_DTYPE = np.dtype("<f4")
_FLOAT64 = np.float64
_KDATA_SCALE = 1.0e-20
_DEFAULT_NONFINITE_FILL_VALUE = 1.0e-300


@dataclass(frozen=True)
class KtaHeader:
    """Metadata extracted from a `.kta` file header."""

    path: str
    irec0: int
    n_wavelength: int
    n_pressure: int
    n_temperature: int
    n_g: int
    molecule_id: int
    isotopologue_id: int
    wavelength_min_micron: float
    wavelength_step_micron: float
    fwhm: float
    g_samples: NDArray[np.float64]
    g_weights: NDArray[np.float64]
    pressure_bar: NDArray[np.float64]
    temperature_K: NDArray[np.float64]
    wavelength_micron: NDArray[np.float64]
    wavenumber_cm_inverse: NDArray[np.float64]
    data_offset_bytes: int
    file_size_bytes: int
    checksum_sha256: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("irec0", "n_wavelength", "n_pressure", "n_temperature", "n_g"):
            value = int(getattr(self, name))
            if value < 1:
                raise RobertValidationError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        arrays = {
            "g_samples": (self.g_samples, self.n_g),
            "g_weights": (self.g_weights, self.n_g),
            "pressure_bar": (self.pressure_bar, self.n_pressure),
            "temperature_K": (self.temperature_K, self.n_temperature),
            "wavelength_micron": (self.wavelength_micron, self.n_wavelength),
            "wavenumber_cm_inverse": (self.wavenumber_cm_inverse, self.n_wavelength),
        }
        for name, (values, size) in arrays.items():
            array = np.array(values, dtype=float, copy=True)
            if array.shape != (size,):
                raise RobertValidationError(f"{name} must have shape ({size},)")
            if not np.all(np.isfinite(array)):
                raise RobertValidationError(f"{name} must contain only finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if np.any(self.g_weights < 0.0):
            raise RobertValidationError("g_weights must be non-negative")
        if np.any(self.pressure_bar <= 0.0):
            raise RobertValidationError("pressure_bar values must be positive")
        if np.any(self.temperature_K <= 0.0):
            raise RobertValidationError("temperature_K values must be positive")
        if np.any(self.wavelength_micron <= 0.0) or np.any(self.wavenumber_cm_inverse <= 0.0):
            raise RobertValidationError("spectral coordinates must be positive")
        if self.data_offset_bytes < 1:
            raise RobertValidationError("data_offset_bytes must be positive")
        if self.file_size_bytes < self.expected_file_size_bytes:
            raise RobertDataError("`.kta` file is smaller than expected from its header")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def native_shape(self) -> tuple[int, int, int, int]:
        """ROBERT/exo_k k-coefficient shape: pressure, temperature, wavelength, g."""

        return (self.n_pressure, self.n_temperature, self.n_wavelength, self.n_g)

    @property
    def stored_shape(self) -> tuple[int, int, int, int]:
        """On-disk k-coefficient shape: wavelength, pressure, temperature, g."""

        return (self.n_wavelength, self.n_pressure, self.n_temperature, self.n_g)

    @property
    def n_kcoefficients(self) -> int:
        """Number of stored k-coefficient values."""

        return int(np.prod(self.stored_shape))

    @property
    def expected_file_size_bytes(self) -> int:
        """Expected minimum file size from header and k-coefficient shape."""

        return self.data_offset_bytes + self.n_kcoefficients * _FLOAT_DTYPE.itemsize

    @property
    def spectral_coverage(self) -> SpectralCoverage:
        """Spectral coverage in wavenumber coordinates."""

        return SpectralCoverage(
            min_value=float(np.min(self.wavenumber_cm_inverse)),
            max_value=float(np.max(self.wavenumber_cm_inverse)),
            unit="cm^-1",
            n_points=self.n_wavelength,
        )

    @property
    def grid_coverage(self) -> GridCoverage:
        """Pressure-temperature coverage."""

        return GridCoverage(
            pressure_min=float(np.min(self.pressure_bar)),
            pressure_max=float(np.max(self.pressure_bar)),
            pressure_unit="bar",
            temperature_min=float(np.min(self.temperature_K)),
            temperature_max=float(np.max(self.temperature_K)),
            temperature_unit="K",
            n_pressure=self.n_pressure,
            n_temperature=self.n_temperature,
        )


@dataclass(frozen=True)
class KtaTable:
    """A loaded `.kta` table."""

    header: KtaHeader
    kcoeff: NDArray[np.float64]
    unit: str = "cm^2/molecule"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kcoeff = np.array(self.kcoeff, dtype=float, copy=True)
        if kcoeff.shape != self.header.native_shape:
            raise RobertValidationError("kcoeff shape must be pressure x temperature x wavelength x g")
        if not np.all(np.isfinite(kcoeff)) or np.any(kcoeff < 0.0):
            raise RobertValidationError("kcoeff values must be finite and non-negative")
        if not self.unit:
            raise RobertValidationError("kcoeff unit must not be empty")
        kcoeff.setflags(write=False)
        object.__setattr__(self, "kcoeff", kcoeff)
        object.__setattr__(self, "metadata", dict(self.metadata))


def read_kta_header(
    path: str | Path,
    *,
    checksum: bool = False,
) -> KtaHeader:
    """Read metadata from a `.kta` file without loading k-coefficients."""

    file_path = Path(path).expanduser()
    _require_kta_file(file_path)
    file_size = file_path.stat().st_size
    with file_path.open("rb") as handle:
        first = np.frombuffer(handle.read(2 * _INT_DTYPE.itemsize), dtype=_INT_DTYPE)
        if first.size != 2:
            raise RobertDataError("`.kta` file ended before irec0 and wavelength count")
        irec0 = int(first[0])
        n_wavelength = int(first[1])

        wl_min, dwl, fwhm = _read_float32(handle, 3, "`.kta` spectral header")
        ints = np.frombuffer(handle.read(5 * _INT_DTYPE.itemsize), dtype=_INT_DTYPE)
        if ints.size != 5:
            raise RobertDataError("`.kta` file ended before grid dimensions")
        n_pressure, n_temperature, n_g, molecule_id, isotopologue_id = (int(item) for item in ints)
        _validate_header_dimensions(
            irec0=irec0,
            n_wavelength=n_wavelength,
            n_pressure=n_pressure,
            n_temperature=n_temperature,
            n_g=n_g,
        )

        g_samples = _read_float32(handle, n_g, "`.kta` g samples")
        g_weights = _read_float32(handle, n_g, "`.kta` g weights")
        _unused = _read_float32(handle, 2, "`.kta` unused header floats")
        pressure_bar = _read_float32(handle, n_pressure, "`.kta` pressure grid")
        temperature_K = _read_float32(handle, n_temperature, "`.kta` temperature grid")

        base_records = 12 + 2 * n_g + n_pressure + n_temperature
        wavelength_records = irec0 - 1 - base_records
        if wavelength_records == n_wavelength:
            stored_wavelength = _read_float32(handle, n_wavelength, "`.kta` wavelength grid")
        elif wavelength_records == 0 and float(dwl) >= 0.0:
            stored_wavelength = float(wl_min) + np.arange(n_wavelength, dtype=float) * float(dwl)
        else:
            raise RobertDataError("`.kta` header record count is inconsistent with wavelength grid")

    wavelength_micron = np.array(stored_wavelength[::-1], dtype=float, copy=True)
    wavenumber = 10000.0 / wavelength_micron
    data_offset_bytes = (irec0 - 1) * _FLOAT_DTYPE.itemsize
    return KtaHeader(
        path=str(file_path),
        irec0=irec0,
        n_wavelength=n_wavelength,
        n_pressure=n_pressure,
        n_temperature=n_temperature,
        n_g=n_g,
        molecule_id=int(molecule_id),
        isotopologue_id=int(isotopologue_id),
        wavelength_min_micron=float(wl_min),
        wavelength_step_micron=float(dwl),
        fwhm=float(fwhm),
        g_samples=np.array(g_samples, dtype=float),
        g_weights=np.array(g_weights, dtype=float),
        pressure_bar=np.array(pressure_bar, dtype=float),
        temperature_K=np.array(temperature_K, dtype=float),
        wavelength_micron=wavelength_micron,
        wavenumber_cm_inverse=wavenumber,
        data_offset_bytes=data_offset_bytes,
        file_size_bytes=file_size,
        checksum_sha256=_file_sha256(file_path) if checksum else None,
        metadata={
            "format": "kta_binary",
            "kcoeff_unit": "cm^2/molecule",
            "pressure_unit": "bar",
            "temperature_unit": "K",
            "wavelength_unit": "micron",
            "wavenumber_unit": "cm^-1",
            "byte_order": "little-endian",
        },
    )


def read_kta(
    path: str | Path,
    *,
    checksum: bool = False,
    nonfinite_policy: str = "raise",
    nonfinite_fill_value: float = _DEFAULT_NONFINITE_FILL_VALUE,
) -> KtaTable:
    """Read a `.kta` file into ROBERT's native axis order.

    By default, ROBERT rejects tables with non-finite k-coefficients. For
    incomplete legacy tables, set `nonfinite_policy="floor"` to map NaN or
    infinite k-coefficients to `nonfinite_fill_value` in the returned runtime
    array only. The source file is left unchanged and replacement counts are
    recorded in the returned table metadata.
    """

    header = read_kta_header(path, checksum=checksum)
    stored = np.fromfile(
        header.path,
        dtype=_FLOAT_DTYPE,
        count=header.n_kcoefficients,
        offset=header.data_offset_bytes,
    )
    if stored.size != header.n_kcoefficients:
        raise RobertDataError("`.kta` file ended before all k-coefficients could be read")
    kcoeff = stored.reshape(header.stored_shape)[::-1].transpose(1, 2, 0, 3).astype(_FLOAT64)
    kcoeff *= _KDATA_SCALE
    kcoeff, metadata = _apply_nonfinite_policy(
        kcoeff,
        policy=nonfinite_policy,
        fill_value=nonfinite_fill_value,
    )
    return KtaTable(header=header, kcoeff=kcoeff, metadata=metadata)


def kta_product_from_header(
    header: KtaHeader,
    *,
    species: str,
    source: OpacityDataSource | str = OpacityDataSource.EXOMOL_OP,
) -> OpacityDataProduct:
    """Build ROBERT opacity-product metadata from a `.kta` header."""

    return OpacityDataProduct(
        species=(species,),
        mode=OpacityMode.CORRELATED_K,
        source=source,
        storage_format=OpacityStorageFormat.KTA_BINARY,
        path=header.path,
        spectral_coverage=header.spectral_coverage,
        grid_coverage=header.grid_coverage,
        g_ordinates=header.n_g,
        file_size_bytes=header.file_size_bytes,
        checksum_sha256=header.checksum_sha256,
        native_shape=header.native_shape,
        metadata={
            **dict(header.metadata),
            "irec0": str(header.irec0),
            "molecule_id": str(header.molecule_id),
            "isotopologue_id": str(header.isotopologue_id),
            "fwhm": str(header.fwhm),
        },
    )


def convert_kta_to_robert_archive(
    path: str | Path,
    output_path: str | Path,
    *,
    species: str | None = None,
    source: OpacityDataSource | str = OpacityDataSource.EXOMOL_OP,
    archive: str = "npy",
    compressed: bool = False,
    overwrite: bool = False,
    nonfinite_policy: str = "raise",
    nonfinite_fill_value: float = _DEFAULT_NONFINITE_FILL_VALUE,
) -> OpacityDatabase:
    """Convert a `.kta` file into a ROBERT-native opacity archive."""

    table = read_kta(
        path,
        checksum=True,
        nonfinite_policy=nonfinite_policy,
        nonfinite_fill_value=nonfinite_fill_value,
    )
    species_name = species or _species_from_filename(Path(path))
    product = kta_product_from_header(table.header, species=species_name, source=source)
    product = replace(product, metadata={**dict(product.metadata), **dict(table.metadata)})
    database = OpacityDatabase(
        products=(product,),
        name=f"{species_name}-kta-import",
        root=str(Path(path).expanduser()),
        metadata={"converted_from": "kta_binary", **dict(table.metadata)},
    )
    arrays = {
        "kcoeff": table.kcoeff,
        "pressure_bar": table.header.pressure_bar,
        "temperature_K": table.header.temperature_K,
        "wavenumber_cm-1": table.header.wavenumber_cm_inverse,
        "wavelength_micron": table.header.wavelength_micron,
        "g_samples": table.header.g_samples,
        "g_weights": table.header.g_weights,
    }
    archive_kind = archive.strip().lower()
    if archive_kind in {"npy", "directory", "robert_npy_directory"}:
        written = write_robert_npy_directory(
            output_path,
            database=database,
            arrays=arrays,
            metadata={"source_file": table.header.path},
            overwrite=overwrite,
        )
    elif archive_kind in {"npz", "robert_npz"}:
        written = write_robert_npz_archive(
            output_path,
            database=database,
            arrays=arrays,
            compressed=compressed,
            metadata={"source_file": table.header.path},
        )
    else:
        raise RobertValidationError("archive must be 'npy' or 'npz'")
    return written.database


def _read_float32(handle: object, count: int, name: str) -> NDArray[np.float64]:
    raw = handle.read(count * _FLOAT_DTYPE.itemsize)
    values = np.frombuffer(raw, dtype=_FLOAT_DTYPE)
    if values.size != count:
        raise RobertDataError(f"{name} ended early")
    return values.astype(float)


def _apply_nonfinite_policy(
    kcoeff: NDArray[np.float64],
    *,
    policy: str,
    fill_value: float,
) -> tuple[NDArray[np.float64], dict[str, str]]:
    normalized_policy = policy.strip().lower()
    if normalized_policy in {"strict", "error"}:
        normalized_policy = "raise"
    if normalized_policy not in {"raise", "floor"}:
        raise RobertValidationError("nonfinite_policy must be 'raise' or 'floor'")
    if not np.isfinite(fill_value) or fill_value <= 0.0:
        raise RobertValidationError("nonfinite_fill_value must be finite and positive")

    finite = np.isfinite(kcoeff)
    nonfinite = ~finite
    n_nonfinite = int(np.sum(nonfinite))
    metadata = {
        "kcoeff_nonfinite_policy": normalized_policy,
        "kcoeff_nonfinite_fill_value": f"{float(fill_value):.17g}",
        "kcoeff_nonfinite_replaced": "0",
        "kcoeff_nan_replaced": "0",
        "kcoeff_posinf_replaced": "0",
        "kcoeff_neginf_replaced": "0",
    }
    if n_nonfinite == 0:
        return kcoeff, metadata
    if normalized_policy == "raise":
        raise RobertValidationError(
            f"kcoeff contains {n_nonfinite} non-finite values; "
            "use nonfinite_policy='floor' for incomplete opacity tables"
        )

    sanitized = np.array(kcoeff, dtype=float, copy=True)
    metadata.update(
        {
            "kcoeff_nonfinite_replaced": str(n_nonfinite),
            "kcoeff_nan_replaced": str(int(np.sum(np.isnan(sanitized)))),
            "kcoeff_posinf_replaced": str(int(np.sum(np.isposinf(sanitized)))),
            "kcoeff_neginf_replaced": str(int(np.sum(np.isneginf(sanitized)))),
        }
    )
    sanitized[nonfinite] = float(fill_value)
    return sanitized, metadata


def _require_kta_file(path: Path) -> None:
    if not path.is_file():
        raise RobertDataError(f"`.kta` file does not exist: {path}")
    if path.suffix.lower() != ".kta":
        raise RobertDataError(f"expected a `.kta` file: {path}")


def _validate_header_dimensions(
    *,
    irec0: int,
    n_wavelength: int,
    n_pressure: int,
    n_temperature: int,
    n_g: int,
) -> None:
    if min(irec0, n_wavelength, n_pressure, n_temperature, n_g) < 1:
        raise RobertDataError("`.kta` header dimensions must be positive")


def _species_from_filename(path: Path) -> str:
    token = path.name.split(".")[0].split("_")[0].strip()
    if not token:
        raise RobertValidationError(f"could not infer species from filename: {path.name}")
    return token


def _file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).expanduser().open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
