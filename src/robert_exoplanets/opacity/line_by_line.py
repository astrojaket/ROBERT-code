"""Tabulated line-by-line opacity on a physical wavelength grid.

This module is the file-backed line-by-line counterpart to the correlated-k
and opacity-sampling providers.  It does not calculate line shapes.  It reads
tabulated cross sections with axes ``pressure x temperature x wavelength`` and
interpolates only in pressure, temperature, and log cross section.

The wavelength axis is physical.  It is never sorted into a cumulative
distribution and it is never represented by a g ordinate.  The small
singleton sample axis is present only because the current radiative-transfer
interface expects a final ``g_or_sample`` dimension.  Its value is ``0.5`` and
its weight is one.

HDF5 preparation uses a narrow hyperslab read for the exact requested
wavelength samples.  It supports the petitRADTRANS layout, where ``bin_edges``
is the physical wavenumber grid in cm^-1; the coordinate is converted once to
micron while the opacity array stays file-backed.  NPZ is a compact exchange
format, but ZIP members do not support narrow random reads; NPZ preparation
therefore checks an explicit memory estimate before reading the full
cross-section member.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping
import zipfile

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import (
    PressureGrid,
    RobertCoverageError,
    RobertValidationError,
    SpectralGrid,
)
from robert_exoplanets.core._immutability import immutable_mapping

from ._interpolation import _brackets
from .metadata import pressure_values_in_unit, spectral_grid_values_in_unit


_DEFAULT_CROSS_SECTION_FLOOR = 1.0e-300
_DEFAULT_MAX_MEMORY_BYTES = 1 * 1024**3
_SINGLETON_SAMPLE_VALUE = 0.5
_SINGLETON_SAMPLE_WEIGHT = 1.0


def _require_volume_mixing_ratio(atmosphere: AtmosphereState) -> None:
    """Enforce ROBERT's VMR-only gas-composition contract."""

    if atmosphere.composition_convention != "volume_mixing_ratio":
        raise RobertValidationError(
            "line-by-line opacity requires composition_convention "
            "'volume_mixing_ratio'"
        )


@dataclass(frozen=True)
class LineByLineTable:
    """Metadata and axes for one tabulated line-by-line opacity file.

    The canonical bulk array shape is ``(pressure, temperature, wavelength)``.
    A final singleton dimension is also accepted in source files.  The source
    array is not loaded while this object is constructed.  The table stores
    wavelengths in micron even when the source uses petitRADTRANS ``bin_edges``
    in cm^-1.
    """

    species: str
    path: str | Path
    pressure_bar: ArrayLike
    temperature_K: ArrayLike
    wavelength_micron: ArrayLike
    unit: str = "cm^2/molecule"
    dataset: str = "cross_section"
    metadata: Mapping[str, str] = field(default_factory=dict)
    native_shape: tuple[int, ...] | None = None
    dtype: str = "float64"

    def __post_init__(self) -> None:
        species = str(self.species).strip()
        if not species:
            raise RobertValidationError("line-by-line table species must not be empty")
        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise RobertValidationError(f"line-by-line table does not exist: {path}")

        pressure = _readonly_axis(
            self.pressure_bar,
            "pressure_bar",
            minimum_size=2,
            increasing=True,
        )
        temperature = _readonly_axis(
            self.temperature_K,
            "temperature_K",
            minimum_size=2,
            increasing=True,
        )
        wavelength = _readonly_axis(
            self.wavelength_micron,
            "wavelength_micron",
            minimum_size=1,
            increasing=None,
            allow_endpoint_duplicate=True,
        )
        unit = str(self.unit).strip()
        dataset = str(self.dataset).strip()
        if not unit:
            raise RobertValidationError("line-by-line opacity unit must not be empty")
        if not dataset:
            raise RobertValidationError("line-by-line opacity dataset must not be empty")

        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise RobertValidationError(
                f"line-by-line opacity dtype is invalid: {self.dtype!r}"
            ) from exc
        if dtype.kind not in "fiu":
            raise RobertValidationError(
                "line-by-line opacity dtype must be a real numeric dtype"
            )

        native_shape = None
        if self.native_shape is not None:
            native_shape = tuple(int(value) for value in self.native_shape)
            _validate_native_shape(native_shape, pressure, temperature, wavelength)

        metadata = {str(key): str(value) for key, value in self.metadata.items()}
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "pressure_bar", pressure)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "wavelength_micron", wavelength)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "metadata", immutable_mapping(metadata))
        object.__setattr__(self, "native_shape", native_shape)
        object.__setattr__(self, "dtype", dtype.str)

    @property
    def canonical_shape(self) -> tuple[int, int, int]:
        """Return the canonical pressure, temperature, wavelength shape."""

        return (
            int(self.pressure_bar.size),
            int(self.temperature_K.size),
            int(self.wavelength_micron.size),
        )

    @property
    def native_nbytes(self) -> int | None:
        """Return the on-disk array size when the source shape is known."""

        if self.native_shape is None:
            return None
        return int(np.prod(self.native_shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        species: str,
        dataset: str | None = None,
        checksum: bool = True,
    ) -> "LineByLineTable":
        """Inspect a line-by-line HDF5 table without reading its bulk array.

        Supported canonical datasets are ``pressure_bar``, ``temperature_K``,
        ``wavelength_micron``, and ``cross_section``.  The petitRADTRANS
        layout is also supported: ``p``, ``t``, ``bin_edges`` in cm^-1, and
        ``xsecarr``.  Common aliases are accepted when their units are
        declared in the file.
        """

        h5py = _require_h5py()
        source = _validated_file_path(path, "line-by-line HDF5 table")
        try:
            with h5py.File(source, "r") as handle:
                pressure_name, pressure_dataset = _find_hdf_dataset(
                    handle,
                    ("pressure_bar", "pressure", "p"),
                    "pressure",
                )
                temperature_name, temperature_dataset = _find_hdf_dataset(
                    handle,
                    ("temperature_K", "temperature", "t"),
                    "temperature",
                )
                try:
                    wavelength_name, wavelength_dataset = _find_hdf_dataset(
                        handle,
                        (
                            "wavelength_micron",
                            "wavelength",
                            "wl",
                            "lambda",
                        ),
                        "physical wavelength",
                    )
                    spectral_name = wavelength_name
                    spectral_dataset = wavelength_dataset
                    spectral_kind = "physical_wavelength"
                except RobertValidationError:
                    spectral_name, spectral_dataset = _find_hdf_dataset(
                        handle,
                        (
                            "bin_edges",
                            "wavenumber_cm_inverse",
                            "wavenumber",
                        ),
                        "physical wavelength or wavenumber",
                    )
                    spectral_kind = "physical_wavenumber"
                if dataset is None:
                    dataset_name, cross_section_dataset = _find_hdf_dataset(
                        handle,
                        (
                            "cross_section",
                            "cross_sections",
                            "xsec",
                            "xsecarr",
                            "opacity",
                        ),
                        "cross section",
                    )
                else:
                    dataset_name = str(dataset).strip()
                    if not dataset_name:
                        raise RobertValidationError(
                            "line-by-line HDF5 dataset name must not be empty"
                        )
                    if dataset_name not in handle:
                        raise RobertValidationError(
                            f"line-by-line HDF5 file is missing dataset {dataset_name!r}"
                        )
                    cross_section_dataset = handle[dataset_name]

                pressure_unit, pressure_inferred = _hdf_axis_unit(
                    handle,
                    pressure_dataset,
                    pressure_name,
                    "pressure",
                    canonical_unit="bar",
                )
                temperature_unit, temperature_inferred = _hdf_axis_unit(
                    handle,
                    temperature_dataset,
                    temperature_name,
                    "temperature",
                    canonical_unit="K",
                )
                spectral_unit, spectral_inferred = _hdf_axis_unit(
                    handle,
                    spectral_dataset,
                    spectral_name,
                    "wavelength" if spectral_kind == "physical_wavelength" else "wavenumber",
                    canonical_unit=("micron" if spectral_kind == "physical_wavelength" else "cm^-1"),
                )
                cross_section_unit = _hdf_cross_section_unit(
                    handle,
                    cross_section_dataset,
                    dataset_name,
                )

                pressure = pressure_values_in_unit(
                    np.asarray(pressure_dataset, dtype=float),
                    pressure_unit,
                    "bar",
                )
                temperature = _temperature_values_in_kelvin(
                    np.asarray(temperature_dataset, dtype=float), temperature_unit
                )
                native_shape = tuple(int(value) for value in cross_section_dataset.shape)
                spectral_size = _canonical_spectral_size(native_shape)
                if spectral_kind == "physical_wavelength":
                    wavelength = _wavelength_values_in_micron(
                        np.asarray(spectral_dataset, dtype=float), spectral_unit
                    )
                    if wavelength.size != spectral_size:
                        raise RobertValidationError(
                            "line-by-line wavelength coordinate length must match cross-section spectral shape"
                        )
                    source_spectral_unit = spectral_unit
                    source_spectral_name = spectral_name
                else:
                    wavenumber = _wavenumber_values_in_cm_inverse(
                        np.asarray(spectral_dataset, dtype=float), spectral_unit
                    )
                    if wavenumber.size == spectral_size:
                        source_wavenumber = wavenumber
                        spectral_representation = "grid"
                    elif wavenumber.size == spectral_size + 1:
                        source_wavenumber = 0.5 * (wavenumber[:-1] + wavenumber[1:])
                        source_wavenumber = _readonly_axis(
                            source_wavenumber,
                            "wavenumber_cm_inverse",
                            minimum_size=1,
                            increasing=None,
                        )
                        spectral_representation = "bin_centers_from_edges"
                    else:
                        raise RobertValidationError(
                            "line-by-line bin_edges length must match cross-section spectral size "
                            "or exceed it by one"
                        )
                    wavelength = _wavelength_values_from_wavenumber(source_wavenumber)
                    source_spectral_unit = spectral_unit
                    source_spectral_name = spectral_name
                _validate_native_shape(native_shape, pressure, temperature, wavelength)
                dtype = np.dtype(cross_section_dataset.dtype)
                metadata = _hdf_metadata(handle)
        except RobertValidationError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise RobertValidationError(
                f"could not inspect line-by-line HDF5 table: {source}"
            ) from exc

        metadata.update(
            {
                "source_format": "line_by_line_hdf5",
                "source_path": str(source),
                "storage_strategy": "hdf5_hyperslab",
                "pressure_unit": "bar",
                "temperature_unit": "K",
                "wavelength_unit": "micron",
                "source_spectral_unit": source_spectral_unit,
                "source_spectral_coordinate": spectral_kind,
                "source_spectral_dataset": source_spectral_name,
                "source_spectral_representation": spectral_representation
                if spectral_kind == "physical_wavenumber"
                else "grid",
                "cross_section_unit": cross_section_unit,
                "spectral_coordinate": "physical_wavelength",
                "cross_section_dataset": dataset_name,
                "native_shape": "x".join(str(value) for value in native_shape),
                "dtype": np.dtype(dtype).str,
            }
        )
        inferred = [
            label
            for label, value in (
                ("pressure", pressure_inferred),
                ("temperature", temperature_inferred),
                ("wavelength", spectral_inferred),
            )
            if value
        ]
        if inferred:
            metadata["units_inferred_from_canonical_name"] = ",".join(inferred)
        if checksum:
            metadata["checksum_sha256"] = _file_sha256(source)

        return cls(
            species=species,
            path=source,
            pressure_bar=pressure,
            temperature_K=temperature,
            wavelength_micron=wavelength,
            unit=cross_section_unit,
            dataset=dataset_name,
            metadata=metadata,
            native_shape=native_shape,
            dtype=np.dtype(dtype).str,
        )

    @classmethod
    def from_hdf(
        cls,
        path: str | Path,
        *,
        species: str,
        dataset: str | None = None,
        checksum: bool = True,
    ) -> "LineByLineTable":
        """Alias for :meth:`from_hdf5`."""

        return cls.from_hdf5(
            path,
            species=species,
            dataset=dataset,
            checksum=checksum,
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        species: str,
        dataset: str | None = None,
        checksum: bool = True,
    ) -> "LineByLineTable":
        """Inspect a compact NPZ table without loading its bulk array.

        The preferred keys are ``pressure_bar``, ``temperature_K``,
        ``wavelength_micron``, and ``cross_section``.  Unit keys are optional
        for these canonical names and can be supplied as scalar text arrays:
        ``pressure_unit``, ``temperature_unit``, ``wavelength_unit``, and
        ``cross_section_unit``.
        """

        source = _validated_file_path(path, "line-by-line NPZ table")
        try:
            with np.load(source, allow_pickle=False) as archive:
                pressure_name, pressure = _find_npz_array(
                    archive, ("pressure_bar", "pressure", "p"), "pressure"
                )
                temperature_name, temperature = _find_npz_array(
                    archive, ("temperature_K", "temperature", "t"), "temperature"
                )
                try:
                    spectral_name, spectral_values = _find_npz_array(
                        archive,
                        ("wavelength_micron", "wavelength", "wl", "lambda"),
                        "physical wavelength",
                    )
                    spectral_kind = "physical_wavelength"
                except RobertValidationError:
                    spectral_name, spectral_values = _find_npz_array(
                        archive,
                        (
                            "bin_edges",
                            "wavenumber_cm_inverse",
                            "wavenumber",
                        ),
                        "physical wavelength or wavenumber",
                    )
                    spectral_kind = "physical_wavenumber"
                if dataset is None:
                    dataset_name, _ = _find_npz_array(
                        archive,
                        (
                            "cross_section",
                            "cross_sections",
                            "xsec",
                            "xsecarr",
                            "opacity",
                        ),
                        "cross section",
                    )
                else:
                    dataset_name = str(dataset).strip()
                    if not dataset_name or dataset_name not in archive.files:
                        raise RobertValidationError(
                            f"line-by-line NPZ file is missing dataset {dataset_name!r}"
                        )

                pressure_unit, pressure_inferred = _npz_unit(
                    archive,
                    ("pressure_unit", "pressure_units"),
                    pressure_name,
                    "bar",
                )
                temperature_unit, temperature_inferred = _npz_unit(
                    archive,
                    ("temperature_unit", "temperature_units"),
                    temperature_name,
                    "K",
                )
                spectral_unit, spectral_inferred = _npz_unit(
                    archive,
                    (
                        "wavelength_unit",
                        "wavelength_units",
                        "wavenumber_unit",
                        "wavenumber_units",
                        "bin_edges_unit",
                        "bin_edges_units",
                    ),
                    spectral_name,
                    "micron" if spectral_kind == "physical_wavelength" else "cm^-1",
                )
                cross_section_unit, _ = _npz_unit(
                    archive,
                    ("cross_section_unit", "cross_section_units", "opacity_unit"),
                    dataset_name,
                    "cm^2/molecule",
                )
                pressure_values = pressure_values_in_unit(
                    np.asarray(pressure, dtype=float), pressure_unit, "bar"
                )
                temperature_values = _temperature_values_in_kelvin(
                    np.asarray(temperature, dtype=float), temperature_unit
                )
                native_shape, dtype = _npz_array_header(source, dataset_name)
                spectral_size = _canonical_spectral_size(native_shape)
                if spectral_kind == "physical_wavelength":
                    wavelength_values = _wavelength_values_in_micron(
                        np.asarray(spectral_values, dtype=float), spectral_unit
                    )
                    if wavelength_values.size != spectral_size:
                        raise RobertValidationError(
                            "line-by-line wavelength coordinate length must match cross-section spectral shape"
                        )
                else:
                    wavenumber = _wavenumber_values_in_cm_inverse(
                        np.asarray(spectral_values, dtype=float), spectral_unit
                    )
                    if wavenumber.size == spectral_size:
                        source_wavenumber = wavenumber
                        spectral_representation = "grid"
                    elif wavenumber.size == spectral_size + 1:
                        source_wavenumber = _readonly_axis(
                            0.5 * (wavenumber[:-1] + wavenumber[1:]),
                            "wavenumber_cm_inverse",
                            minimum_size=1,
                            increasing=None,
                        )
                        spectral_representation = "bin_centers_from_edges"
                    else:
                        raise RobertValidationError(
                            "line-by-line bin_edges length must match cross-section spectral size "
                            "or exceed it by one"
                        )
                    wavelength_values = _wavelength_values_from_wavenumber(
                        source_wavenumber
                    )
                _validate_native_shape(
                    native_shape,
                    pressure_values,
                    temperature_values,
                    wavelength_values,
                )
                metadata = _npz_metadata(archive)
        except RobertValidationError:
            raise
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
            raise RobertValidationError(
                f"could not inspect line-by-line NPZ table: {source}"
            ) from exc

        metadata.update(
            {
                "source_format": "line_by_line_npz",
                "source_path": str(source),
                "storage_strategy": "npz_guarded_full_member",
                "pressure_unit": "bar",
                "temperature_unit": "K",
                "wavelength_unit": "micron",
                "source_spectral_unit": spectral_unit,
                "source_spectral_coordinate": spectral_kind,
                "source_spectral_dataset": spectral_name,
                "source_spectral_representation": spectral_representation
                if spectral_kind == "physical_wavenumber"
                else "grid",
                "cross_section_unit": cross_section_unit,
                "spectral_coordinate": "physical_wavelength",
                "cross_section_dataset": dataset_name,
                "native_shape": "x".join(str(value) for value in native_shape),
                "dtype": np.dtype(dtype).str,
            }
        )
        inferred = [
            label
            for label, value in (
                ("pressure", pressure_inferred),
                ("temperature", temperature_inferred),
                ("wavelength", spectral_inferred),
            )
            if value
        ]
        if inferred:
            metadata["units_inferred_from_canonical_name"] = ",".join(inferred)
        if checksum:
            metadata["checksum_sha256"] = _file_sha256(source)

        return cls(
            species=species,
            path=source,
            pressure_bar=pressure_values,
            temperature_K=temperature_values,
            wavelength_micron=wavelength_values,
            unit=cross_section_unit,
            dataset=dataset_name,
            metadata=metadata,
            native_shape=native_shape,
            dtype=np.dtype(dtype).str,
        )

    @classmethod
    def from_numpy_archive(
        cls,
        path: str | Path,
        *,
        species: str,
        dataset: str | None = None,
        checksum: bool = True,
    ) -> "LineByLineTable":
        """Alias for :meth:`from_npz`."""

        return cls.from_npz(
            path,
            species=species,
            dataset=dataset,
            checksum=checksum,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        species: str,
        dataset: str | None = None,
        checksum: bool = True,
    ) -> "LineByLineTable":
        """Load a table by its HDF5 or NPZ suffix."""

        suffix = Path(path).suffix.lower()
        if suffix in {".h5", ".hdf5", ".hdf"}:
            return cls.from_hdf5(
                path,
                species=species,
                dataset=dataset,
                checksum=checksum,
            )
        if suffix == ".npz":
            return cls.from_npz(
                path,
                species=species,
                dataset=dataset,
                checksum=checksum,
            )
        raise RobertValidationError(
            "line-by-line table format must use an HDF5 (.h5/.hdf5) or NPZ (.npz) file"
        )


@dataclass(frozen=True)
class PreparedLineByLineOpacity:
    """Run-specific, exact-wavelength line-by-line preparation state."""

    provider_name: str
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    species: tuple[str, ...]
    g_samples: ArrayLike
    g_weights: ArrayLike
    cache_key: str
    log_cross_sections: Mapping[str, NDArray[np.float64]] = field(repr=False)
    metadata: Mapping[str, str] = field(default_factory=dict)
    spectral_indices: Mapping[str, NDArray[np.int64]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        provider_name = str(self.provider_name).strip()
        cache_key = str(self.cache_key).strip()
        if not provider_name:
            raise RobertValidationError("prepared line-by-line provider name must not be empty")
        if not cache_key:
            raise RobertValidationError("prepared line-by-line cache key must not be empty")
        species = _species_tuple(self.species, "prepared line-by-line species")
        if len(set(species)) != len(species):
            raise RobertValidationError("prepared line-by-line species must be unique")

        g_samples = _readonly_1d(self.g_samples, "line-by-line sample axis")
        g_weights = _readonly_1d(self.g_weights, "line-by-line sample weights")
        if g_samples.shape != (1,) or g_weights.shape != (1,):
            raise RobertValidationError(
                "line-by-line opacity must use one singleton physical-sample axis"
            )
        if not np.isclose(g_samples[0], _SINGLETON_SAMPLE_VALUE):
            raise RobertValidationError(
                "line-by-line singleton sample value must be 0.5"
            )
        if not np.isclose(g_weights[0], _SINGLETON_SAMPLE_WEIGHT):
            raise RobertValidationError(
                "line-by-line singleton sample weight must be one"
            )

        cached: dict[str, NDArray[np.float64]] = {}
        for species_name in species:
            if species_name not in self.log_cross_sections:
                raise RobertValidationError(
                    f"prepared line-by-line opacity is missing species {species_name}"
                )
            values = np.array(self.log_cross_sections[species_name], dtype=float, copy=True)
            if values.ndim != 3 or values.shape[-1] != self.spectral_grid.size:
                raise RobertValidationError(
                    "cached line-by-line cross sections must be pressure x temperature x wavelength"
                )
            if not np.all(np.isfinite(values)):
                raise RobertValidationError(
                    "cached line-by-line log cross sections must be finite"
                )
            values.setflags(write=False)
            cached[species_name] = values

        spectral_indices: dict[str, NDArray[np.int64]] = {}
        for species_name, values in self.spectral_indices.items():
            if species_name not in species:
                raise RobertValidationError(
                    "prepared line-by-line spectral-index species must be prepared"
                )
            indices = np.array(values, dtype=np.int64, copy=True)
            if indices.shape != (self.spectral_grid.size,) or np.any(indices < 0):
                raise RobertValidationError(
                    "prepared line-by-line spectral indices must match the spectral grid"
                )
            indices.setflags(write=False)
            spectral_indices[species_name] = indices

        metadata = {
            str(key): str(value) for key, value in self.metadata.items()
        }
        metadata.setdefault("opacity_mode", "line_by_line")
        metadata.setdefault("spectral_coordinate", "physical_wavelength")
        metadata.setdefault("physical_sample_axis", "singleton")
        object.__setattr__(self, "provider_name", provider_name)
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "g_samples", g_samples)
        object.__setattr__(self, "g_weights", g_weights)
        object.__setattr__(self, "log_cross_sections", immutable_mapping(cached))
        object.__setattr__(self, "spectral_indices", immutable_mapping(spectral_indices))
        object.__setattr__(self, "metadata", immutable_mapping(metadata))

    @property
    def physical_samples(self) -> NDArray[np.float64]:
        """Return the singleton compatibility sample axis."""

        return self.g_samples

    @property
    def sample_axis(self) -> str:
        """Return the physical sample-axis label."""

        return "physical_wavelength"


@dataclass(frozen=True)
class EvaluatedLineByLineOpacity:
    """Line-by-line cross sections evaluated on an atmospheric profile."""

    prepared: PreparedLineByLineOpacity
    kcoeff: ArrayLike
    unit: str = "cm^2/molecule"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.array(self.kcoeff, dtype=float, copy=True)
        expected = (
            len(self.prepared.species),
            self.prepared.pressure_grid.n_layers,
            self.prepared.spectral_grid.size,
            1,
        )
        if values.shape != expected:
            raise RobertValidationError(
                "evaluated line-by-line cross sections must be species x layers x wavelength x 1"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RobertValidationError(
                "evaluated line-by-line cross sections must be finite and non-negative"
            )
        unit = str(self.unit).strip()
        if not unit:
            raise RobertValidationError("evaluated line-by-line unit must not be empty")
        values.setflags(write=False)
        metadata = {str(key): str(value) for key, value in self.metadata.items()}
        metadata.setdefault("opacity_mode", "line_by_line")
        metadata.setdefault("spectral_coordinate", "physical_wavelength")
        metadata.setdefault("physical_sample_axis", "singleton")
        object.__setattr__(self, "kcoeff", values)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "metadata", immutable_mapping(metadata))

    @property
    def physical_samples(self) -> NDArray[np.float64]:
        """Return the singleton compatibility sample axis."""

        return self.prepared.g_samples

    @property
    def sample_axis(self) -> str:
        """Return the physical sample-axis label."""

        return "physical_wavelength"


@dataclass(frozen=True)
class EvaluatedLineByLineMixture:
    """VMR-weighted line-by-line cross section without species arrays."""

    prepared: PreparedLineByLineOpacity
    cross_section: ArrayLike
    unit: str = "cm^2/molecule"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.array(self.cross_section, dtype=float, copy=True)
        expected = (
            self.prepared.pressure_grid.n_layers,
            self.prepared.spectral_grid.size,
        )
        if values.shape != expected:
            raise RobertValidationError(
                "line-by-line mixture cross section must be layers x wavelength"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RobertValidationError(
                "line-by-line mixture cross section must be finite and non-negative"
            )
        unit = str(self.unit).strip()
        if not unit:
            raise RobertValidationError("line-by-line mixture unit must not be empty")
        values.setflags(write=False)
        metadata = {str(key): str(value) for key, value in self.metadata.items()}
        metadata.setdefault("opacity_mode", "line_by_line")
        metadata.setdefault("spectral_coordinate", "physical_wavelength")
        metadata.setdefault("physical_sample_axis", "singleton")
        object.__setattr__(self, "cross_section", values)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "metadata", immutable_mapping(metadata))


@dataclass(frozen=True)
class LineByLineCoverageReport:
    """Coverage result for line-by-line evaluation."""

    valid: bool
    message: str
    species: tuple[str, ...]
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "reasons", immutable_mapping(self.reasons))


@dataclass(frozen=True)
class LineByLineOpacityProvider:
    """File-backed, tabulated line-by-line opacity provider.

    ``prepare`` selects an exact subset of direct physical wavelengths and
    caches only that subset.  HDF5 sources use a narrow hyperslab; NPZ sources
    use a guarded full-member read.  The prepared-slice cache is a bounded,
    thread-safe LRU.  Pressure and temperature are interpolated bilinearly in
    log-pressure, linear temperature, and log cross section.  The strict
    policy is the default; the explicit ``_clip`` policy clamps out-of-range
    atmospheric values to table boundaries.
    """

    tables: Mapping[str, LineByLineTable]
    name: str = "line-by-line-native"
    interpolation: str = "log_pressure_temperature_log_xsec"
    cross_section_floor: float = _DEFAULT_CROSS_SECTION_FLOOR
    max_memory_bytes: int | None = _DEFAULT_MAX_MEMORY_BYTES
    wavelength_bounds_micron: tuple[float, float] | None = None
    native_sampling_stride: int = 1
    max_cached_slices: int = 2
    _prepared_cache: OrderedDict[str, PreparedLineByLineOpacity] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _prepared_cache_bytes: int = field(init=False, repr=False, compare=False)
    _prepared_cache_lock: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise RobertValidationError("line-by-line provider name must not be empty")
        if self.interpolation not in {
            "log_pressure_temperature_log_xsec",
            "log_pressure_temperature_log_xsec_clip",
        }:
            raise RobertValidationError(
                "line-by-line interpolation must be log-pressure/temperature/log-xsec"
            )
        floor = float(self.cross_section_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise RobertValidationError(
                "line-by-line cross_section_floor must be finite and positive"
            )
        limit = _memory_limit(self.max_memory_bytes, "max_memory_bytes")
        bounds = _wavelength_bounds(
            self.wavelength_bounds_micron,
            "wavelength_bounds_micron",
        )
        native_sampling_stride = _sampling_stride(
            self.native_sampling_stride,
            "native_sampling_stride",
        )
        max_cached_slices = _nonnegative_integer(
            self.max_cached_slices,
            "max_cached_slices",
        )
        tables: dict[str, LineByLineTable] = {}
        for key, table in self.tables.items():
            if not isinstance(table, LineByLineTable):
                raise RobertValidationError(
                    "line-by-line provider tables must be LineByLineTable objects"
                )
            species = str(key).strip()
            if not species or species != table.species:
                raise RobertValidationError(
                    "line-by-line table mapping keys must match table species"
                )
            tables[species] = table
        if not tables:
            raise RobertValidationError(
                "line-by-line provider must contain at least one table"
            )
        if len({table.unit for table in tables.values()}) != 1:
            raise RobertValidationError(
                "line-by-line tables in one provider must share a cross-section unit"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tables", immutable_mapping(tables))
        object.__setattr__(self, "cross_section_floor", floor)
        object.__setattr__(self, "max_memory_bytes", limit)
        object.__setattr__(self, "wavelength_bounds_micron", bounds)
        object.__setattr__(self, "native_sampling_stride", native_sampling_stride)
        object.__setattr__(self, "max_cached_slices", max_cached_slices)
        object.__setattr__(self, "_prepared_cache", OrderedDict())
        object.__setattr__(self, "_prepared_cache_bytes", 0)
        object.__setattr__(self, "_prepared_cache_lock", RLock())

    @classmethod
    def from_paths(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "line-by-line-native",
        interpolation: str = "log_pressure_temperature_log_xsec",
        checksum: bool = True,
        max_memory_bytes: int | None = _DEFAULT_MAX_MEMORY_BYTES,
        wavelength_bounds_micron: tuple[float, float] | None = None,
        native_sampling_stride: int = 1,
        max_cached_slices: int = 2,
        cross_section_floor: float = _DEFAULT_CROSS_SECTION_FLOOR,
    ) -> "LineByLineOpacityProvider":
        """Build a provider from species-to-HDF5/NPZ paths."""

        if not paths:
            raise RobertValidationError("line-by-line paths must not be empty")
        tables = {
            str(species): LineByLineTable.from_file(
                path,
                species=str(species),
                checksum=checksum,
            )
            for species, path in paths.items()
        }
        return cls(
            tables=tables,
            name=name,
            interpolation=interpolation,
            cross_section_floor=cross_section_floor,
            max_memory_bytes=max_memory_bytes,
            wavelength_bounds_micron=wavelength_bounds_micron,
            native_sampling_stride=native_sampling_stride,
            max_cached_slices=max_cached_slices,
        )

    @classmethod
    def from_hdf_paths(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "line-by-line-native",
        interpolation: str = "log_pressure_temperature_log_xsec",
        checksum: bool = True,
        max_memory_bytes: int | None = _DEFAULT_MAX_MEMORY_BYTES,
        wavelength_bounds_micron: tuple[float, float] | None = None,
        native_sampling_stride: int = 1,
        max_cached_slices: int = 2,
        cross_section_floor: float = _DEFAULT_CROSS_SECTION_FLOOR,
    ) -> "LineByLineOpacityProvider":
        """Build a provider from HDF5 tables."""

        tables = {
            str(species): LineByLineTable.from_hdf5(
                path,
                species=str(species),
                checksum=checksum,
            )
            for species, path in paths.items()
        }
        return cls(
            tables=tables,
            name=name,
            interpolation=interpolation,
            cross_section_floor=cross_section_floor,
            max_memory_bytes=max_memory_bytes,
            wavelength_bounds_micron=wavelength_bounds_micron,
            native_sampling_stride=native_sampling_stride,
            max_cached_slices=max_cached_slices,
        )

    @classmethod
    def from_npz_paths(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "line-by-line-native",
        interpolation: str = "log_pressure_temperature_log_xsec",
        checksum: bool = True,
        max_memory_bytes: int | None = _DEFAULT_MAX_MEMORY_BYTES,
        wavelength_bounds_micron: tuple[float, float] | None = None,
        native_sampling_stride: int = 1,
        max_cached_slices: int = 2,
        cross_section_floor: float = _DEFAULT_CROSS_SECTION_FLOOR,
    ) -> "LineByLineOpacityProvider":
        """Build a provider from NPZ tables."""

        tables = {
            str(species): LineByLineTable.from_npz(
                path,
                species=str(species),
                checksum=checksum,
            )
            for species, path in paths.items()
        }
        return cls(
            tables=tables,
            name=name,
            interpolation=interpolation,
            cross_section_floor=cross_section_floor,
            max_memory_bytes=max_memory_bytes,
            wavelength_bounds_micron=wavelength_bounds_micron,
            native_sampling_stride=native_sampling_stride,
            max_cached_slices=max_cached_slices,
        )

    @property
    def species(self) -> tuple[str, ...]:
        """Return available species in mapping order."""

        return tuple(self.tables)

    @property
    def cached_slice_count(self) -> int:
        """Return the number of prepared wavelength slices retained in memory."""

        with self._prepared_cache_lock:  # type: ignore[attr-defined]
            return len(self._prepared_cache)

    @property
    def cached_slice_bytes(self) -> int:
        """Return the estimated bytes retained by the prepared-slice cache."""

        with self._prepared_cache_lock:  # type: ignore[attr-defined]
            return int(self._prepared_cache_bytes)

    def clear_prepared_cache(self) -> None:
        """Release all prepared slices held by this provider."""

        with self._prepared_cache_lock:  # type: ignore[attr-defined]
            self._prepared_cache.clear()
            object.__setattr__(self, "_prepared_cache_bytes", 0)

    def native_spectral_grid(
        self,
        *,
        sampling: int | None = None,
        wavelength_bounds_micron: tuple[float, float] | None = None,
        reference_species: str | None = None,
        name: str | None = None,
    ) -> SpectralGrid:
        """Return an ordered exact physical-wavelength intersection.

        Tables can cover different wavelength intervals and can contain
        different numbers of line samples.  The first table is the default
        reference order.  Set ``reference_species`` to select another table;
        every returned reference wavelength must still be present in every
        selected table within the exact-match tolerance.
        """

        sampling = self.native_sampling_stride if sampling is None else _sampling_stride(
            sampling,
            "line-by-line wavelength sampling stride",
        )
        if wavelength_bounds_micron is None:
            wavelength_bounds = self.wavelength_bounds_micron
        else:
            wavelength_bounds = _wavelength_bounds(
                wavelength_bounds_micron,
                "wavelength_bounds_micron",
            )
            if self.wavelength_bounds_micron is not None:
                configured_lower, configured_upper = self.wavelength_bounds_micron
                requested_lower, requested_upper = wavelength_bounds
                if (
                    requested_lower < configured_lower
                    or requested_upper > configured_upper
                ):
                    raise RobertCoverageError(
                        "requested wavelength window is outside the configured "
                        "line-by-line wavelength window"
                    )
        if reference_species is None:
            reference_name = self.species[0]
        else:
            reference_name = str(reference_species).strip()
            if reference_name not in self.tables:
                raise RobertCoverageError(
                    f"missing line-by-line reference species: {reference_name}"
                )
        reference = self.tables[reference_name].wavelength_micron
        selected = np.arange(reference.size, dtype=np.int64)
        if reference.size > 1:
            # Some official pRT tables repeat the first grid sample. pRT drops
            # the earlier copy when it builds its frequency grid.
            selected = selected[np.concatenate((np.diff(reference) != 0.0, [True]))]
        if wavelength_bounds is not None:
            lower, upper = wavelength_bounds
            selected = selected[
                (reference[selected] >= lower) & (reference[selected] <= upper)
            ]
        for species_name, table in self.tables.items():
            if species_name == reference_name:
                continue
            selected = selected[
                _wavelength_intersection_mask(
                    reference[selected], table.wavelength_micron
                )
            ]
        selected = selected[::sampling]
        if selected.size == 0:
            raise RobertCoverageError(
                "requested line-by-line wavelength range is empty"
            )
        grid_metadata = {
            "opacity_mode": "line_by_line",
            "spectral_coordinate": "physical_wavelength",
            "native_sampling_stride": str(sampling),
            "native_sampling_accuracy": "requires_convergence_validation",
            "native_sampling_convergence_validation": "required",
            "native_sampling_validation": "required",
            "wavelength_window_micron": (
                ""
                if wavelength_bounds is None
                else f"{wavelength_bounds[0]:.17g}:{wavelength_bounds[1]:.17g}"
            ),
        }
        return SpectralGrid(
            values=reference[selected],
            unit="micron",
            role="opacity",
            name=name or f"{self.name}-stride-{sampling}",
            metadata=grid_metadata,
        )

    def estimate_memory_bytes(
        self,
        spectral_grid: SpectralGrid,
        species: Iterable[str] | None = None,
    ) -> int:
        """Estimate peak host memory for one preparation.

        HDF5 estimates the selected hyperslab and temporary log/output arrays.
        NPZ estimates the full decoded source member plus selected/log arrays.
        No bulk source array is read by this method.
        """

        selected_species = self._validated_species(species)
        self._validate_wavelength_window(spectral_grid)
        total = 0
        for species_name in selected_species:
            table = self.tables[species_name]
            indices = _exact_wavelength_indices(spectral_grid, table)
            total += _estimate_table_memory_bytes(table, indices.size)
        return int(total)

    def prepare(
        self,
        spectral_grid: SpectralGrid,
        pressure_grid: PressureGrid,
        species: Iterable[str],
        *,
        max_memory_bytes: int | None = None,
    ) -> PreparedLineByLineOpacity:
        """Select exact wavelengths and cache line-by-line cross sections."""

        if not isinstance(spectral_grid, SpectralGrid):
            raise RobertValidationError("spectral_grid must be a SpectralGrid")
        if not isinstance(pressure_grid, PressureGrid):
            raise RobertValidationError("pressure_grid must be a PressureGrid")
        species_tuple = self._validated_species(species)
        memory_limit = self.max_memory_bytes if max_memory_bytes is None else _memory_limit(
            max_memory_bytes, "max_memory_bytes"
        )
        self._validate_wavelength_window(spectral_grid)
        pressure = pressure_values_in_unit(
            pressure_grid.centers,
            pressure_grid.unit,
            "bar",
        )
        clip = self.interpolation.endswith("_clip")
        selected_indices: dict[str, NDArray[np.int64]] = {}
        estimated = 0
        for species_name in species_tuple:
            table = self.tables[species_name]
            _validate_axis_coverage(
                pressure,
                table.pressure_bar,
                "pressure",
                clip=clip,
            )
            indices = _exact_wavelength_indices(spectral_grid, table)
            selected_indices[species_name] = indices
            estimated += _estimate_table_memory_bytes(table, indices.size)
        requested_sampling_stride = _grid_sampling_stride(
            spectral_grid,
            self.native_sampling_stride,
        )
        if memory_limit is not None and estimated > memory_limit:
            raise RobertValidationError(
                "refusing line-by-line preparation: estimated memory "
                f"{estimated / 1024**3:.2f} GiB exceeds max_memory_bytes "
                f"{memory_limit / 1024**3:.2f} GiB; request a narrower exact "
                "wavelength window or use HDF5 instead of NPZ"
            )

        cache_key = _cache_key(
            self.name,
            self.interpolation,
            self.cross_section_floor,
            species_tuple,
            spectral_grid,
            pressure_grid,
            tuple(self.tables[item] for item in species_tuple),
            tuple(selected_indices[item] for item in species_tuple),
            wavelength_bounds_micron=self.wavelength_bounds_micron,
            native_sampling_stride=requested_sampling_stride,
        )
        with self._prepared_cache_lock:  # type: ignore[attr-defined]
            cached_prepared = self._prepared_cache.get(cache_key)
            if cached_prepared is not None:
                self._prepared_cache.move_to_end(cache_key)
                return cached_prepared

        cached: dict[str, NDArray[np.float64]] = {}
        strategies: set[str] = set()
        for species_name in species_tuple:
            table = self.tables[species_name]
            cached[species_name] = _load_selected_log_cross_sections(
                table,
                selected_indices[species_name],
                floor=self.cross_section_floor,
            )
            strategies.add(table.metadata.get("storage_strategy", "unknown"))

        prepared = PreparedLineByLineOpacity(
            provider_name=self.name,
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            species=species_tuple,
            g_samples=np.array([_SINGLETON_SAMPLE_VALUE]),
            g_weights=np.array([_SINGLETON_SAMPLE_WEIGHT]),
            cache_key=cache_key,
            log_cross_sections=cached,
            metadata={
                "opacity_mode": "line_by_line",
                "spectral_coordinate": "physical_wavelength",
                "physical_sample_axis": "singleton",
                "interpolation": self.interpolation,
                "sample_count": str(spectral_grid.size),
                "estimated_memory_bytes": str(estimated),
                "storage_strategy": "+".join(sorted(strategies)),
                "source_read": "exact_narrow_window" if strategies == {"hdf5_hyperslab"} else "guarded_source_read",
                "native_sampling_stride": str(requested_sampling_stride),
                "native_sampling_accuracy": "requires_convergence_validation",
                "native_sampling_convergence_validation": "required",
                "native_sampling_validation": "required",
                "wavelength_window_micron": (
                    ""
                    if self.wavelength_bounds_micron is None
                    else f"{self.wavelength_bounds_micron[0]:.17g}:{self.wavelength_bounds_micron[1]:.17g}"
                ),
            },
            spectral_indices=selected_indices,
        )
        self._cache_prepared(cache_key, prepared, estimated)
        return prepared

    def _validate_wavelength_window(self, spectral_grid: SpectralGrid) -> None:
        """Reject a request outside the source's configured narrow window."""

        if self.wavelength_bounds_micron is None:
            return
        requested = spectral_grid_values_in_unit(spectral_grid, "micron")
        lower, upper = self.wavelength_bounds_micron
        inside = (requested >= lower) & (requested <= upper)
        if not np.all(inside):
            raise RobertCoverageError(
                "requested wavelengths are outside the configured line-by-line "
                "wavelength window"
            )

    def _cache_prepared(
        self,
        cache_key: str,
        prepared: PreparedLineByLineOpacity,
        estimated_bytes: int,
    ) -> None:
        """Insert one immutable prepared slice into the bounded LRU cache."""

        if self.max_cached_slices == 0:
            return
        # The default cache byte limit is the same 1 GiB limit used for one
        # preparation.  This prevents several legal narrow slices from
        # silently creating an unbounded resident set.
        cache_limit = (
            _DEFAULT_MAX_MEMORY_BYTES
            if self.max_memory_bytes is None
            else self.max_memory_bytes
        )
        if cache_limit is not None and estimated_bytes > cache_limit:
            return
        with self._prepared_cache_lock:  # type: ignore[attr-defined]
            existing = self._prepared_cache.get(cache_key)
            if existing is not None:
                self._prepared_cache.move_to_end(cache_key)
                return
            while self._prepared_cache and (
                len(self._prepared_cache) >= self.max_cached_slices
                or (
                    cache_limit is not None
                    and self._prepared_cache_bytes + estimated_bytes > cache_limit
                )
            ):
                _, evicted = self._prepared_cache.popitem(last=False)
                object.__setattr__(
                    self,
                    "_prepared_cache_bytes",
                    self._prepared_cache_bytes
                    - int(evicted.metadata.get("estimated_memory_bytes", "0")),
                )
            self._prepared_cache[cache_key] = prepared
            object.__setattr__(
                self,
                "_prepared_cache_bytes",
                self._prepared_cache_bytes + int(estimated_bytes),
            )

    def coverage(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedLineByLineOpacity,
    ) -> LineByLineCoverageReport:
        """Check pressure and temperature coverage for an atmosphere."""

        if not isinstance(prepared, PreparedLineByLineOpacity):
            raise RobertValidationError(
                "line-by-line provider requires prepared line-by-line opacity"
            )
        reasons: dict[str, str] = {}
        pressure = pressure_values_in_unit(
            atmosphere.pressure_grid.centers,
            atmosphere.pressure_grid.unit,
            "bar",
        )
        clip = self.interpolation.endswith("_clip")
        for species_name in prepared.species:
            table = self.tables.get(species_name)
            if table is None:
                reasons[species_name] = "missing line-by-line table"
                continue
            if species_name not in atmosphere.composition:
                reasons[species_name] = "atmosphere is missing species composition"
                continue
            try:
                _validate_axis_coverage(
                    pressure,
                    table.pressure_bar,
                    "pressure",
                    clip=clip,
                )
                _validate_axis_coverage(
                    atmosphere.temperature,
                    table.temperature_K,
                    "temperature",
                    clip=clip,
                )
            except RobertCoverageError as exc:
                reasons[species_name] = str(exc)
        valid = not reasons
        return LineByLineCoverageReport(
            valid=valid,
            message="covered" if valid else "line-by-line coverage is incomplete",
            species=prepared.species,
            reasons=reasons,
        )

    def evaluate(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedLineByLineOpacity,
    ) -> EvaluatedLineByLineOpacity:
        """Interpolate prepared line-by-line cross sections on an atmosphere."""

        _require_volume_mixing_ratio(atmosphere)
        if not isinstance(prepared, PreparedLineByLineOpacity):
            raise RobertValidationError(
                "line-by-line provider requires prepared line-by-line opacity"
            )
        report = self.coverage(atmosphere, prepared)
        if not report.valid:
            detail = "; ".join(
                f"{species}: {reason}" for species, reason in report.reasons.items()
            )
            raise RobertCoverageError(f"{report.message}: {detail}")

        pressure = pressure_values_in_unit(
            atmosphere.pressure_grid.centers,
            atmosphere.pressure_grid.unit,
            "bar",
        )
        clip = self.interpolation.endswith("_clip")
        values = np.empty(
            (
                len(prepared.species),
                atmosphere.n_layers,
                prepared.spectral_grid.size,
                1,
            ),
            dtype=float,
        )
        for species_index, species_name in enumerate(prepared.species):
            table = self.tables[species_name]
            log_values = _interpolate_log_cross_sections(
                prepared.log_cross_sections[species_name],
                pressure,
                atmosphere.temperature,
                table.pressure_bar,
                table.temperature_K,
                clip=clip,
            )
            try:
                values[species_index, :, :, 0] = np.exp(log_values)
            except FloatingPointError as exc:  # pragma: no cover - NumPy defaults vary
                raise RobertValidationError(
                    "line-by-line cross-section interpolation overflowed"
                ) from exc
        if not np.all(np.isfinite(values)):
            raise RobertValidationError(
                "evaluated line-by-line cross sections must be finite"
            )
        return EvaluatedLineByLineOpacity(
            prepared=prepared,
            kcoeff=values,
            unit=self.tables[prepared.species[0]].unit,
            metadata={
                "opacity_mode": "line_by_line",
                "spectral_coordinate": "physical_wavelength",
                "physical_sample_axis": "singleton",
                "interpolation": self.interpolation,
                "native_sampling_stride": str(
                    prepared.metadata.get("native_sampling_stride", "1")
                ),
                "native_sampling_accuracy": str(
                    prepared.metadata.get(
                        "native_sampling_accuracy",
                        "requires_convergence_validation",
                    )
                ),
            },
        )

    def evaluate_mixture(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedLineByLineOpacity,
    ) -> EvaluatedLineByLineMixture:
        """Evaluate and directly VMR-sum line-by-line cross sections.

        This method is deliberately fused.  It interpolates one species,
        adds its VMR-weighted contribution to the mixture, and then releases
        that temporary array before it reads the next species.  It therefore
        does not retain a species-by-layer-by-wavelength opacity cube.
        """

        _require_volume_mixing_ratio(atmosphere)
        if not isinstance(prepared, PreparedLineByLineOpacity):
            raise RobertValidationError(
                "line-by-line provider requires prepared line-by-line opacity"
            )
        report = self.coverage(atmosphere, prepared)
        if not report.valid:
            detail = "; ".join(
                f"{species}: {reason}" for species, reason in report.reasons.items()
            )
            raise RobertCoverageError(f"{report.message}: {detail}")

        pressure = pressure_values_in_unit(
            atmosphere.pressure_grid.centers,
            atmosphere.pressure_grid.unit,
            "bar",
        )
        clip = self.interpolation.endswith("_clip")
        mixture = np.zeros(
            (atmosphere.n_layers, prepared.spectral_grid.size),
            dtype=float,
        )
        for species_name in prepared.species:
            table = self.tables[species_name]
            log_values = _interpolate_log_cross_sections(
                prepared.log_cross_sections[species_name],
                pressure,
                atmosphere.temperature,
                table.pressure_bar,
                table.temperature_K,
                clip=clip,
            )
            species_cross_section = np.empty_like(log_values)
            try:
                np.exp(log_values, out=species_cross_section)
            except FloatingPointError as exc:  # pragma: no cover - NumPy defaults vary
                raise RobertValidationError(
                    "line-by-line cross-section interpolation overflowed"
                ) from exc
            mixture += species_cross_section * np.asarray(
                atmosphere.composition[species_name],
                dtype=float,
            )[:, None]
        if not np.all(np.isfinite(mixture)):
            raise RobertValidationError(
                "fused line-by-line cross sections must be finite"
            )
        return EvaluatedLineByLineMixture(
            prepared=prepared,
            cross_section=mixture,
            unit=self.tables[prepared.species[0]].unit,
            metadata={
                "opacity_mode": "line_by_line",
                "spectral_coordinate": "physical_wavelength",
                "physical_sample_axis": "singleton",
                "species_combination": "vmr_weighted_direct_sum",
                "assembly_backend": "fused_low_memory_direct_sum",
                "species_arrays_retained": "false",
                "native_sampling_stride": str(
                    prepared.metadata.get("native_sampling_stride", "1")
                ),
                "native_sampling_accuracy": str(
                    prepared.metadata.get(
                        "native_sampling_accuracy",
                        "requires_convergence_validation",
                    )
                ),
            },
        )

    def _validated_species(self, species: Iterable[str] | None) -> tuple[str, ...]:
        values = self.species if species is None else _species_tuple(species, "species")
        if len(set(values)) != len(values):
            raise RobertValidationError("line-by-line species must not contain duplicates")
        missing = tuple(item for item in values if item not in self.tables)
        if missing:
            raise RobertCoverageError(
                "missing line-by-line tables for species: " + ", ".join(missing)
            )
        return values


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RobertValidationError(
            "loading line-by-line HDF5 opacity requires h5py"
        ) from exc
    return h5py


def _validated_file_path(path: str | Path, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RobertValidationError(f"{label} does not exist: {source}")
    return source


def _readonly_axis(
    values: ArrayLike,
    name: str,
    *,
    minimum_size: int,
    increasing: bool | None,
    allow_endpoint_duplicate: bool = False,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size < minimum_size:
        raise RobertValidationError(
            f"{name} must contain at least {minimum_size} values"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise RobertValidationError(f"{name} must be finite and positive")
    if array.size > 1:
        difference = np.diff(array)
        if increasing is True and not np.all(difference > 0.0):
            raise RobertValidationError(f"{name} must be strictly increasing")
        if increasing is False and not np.all(difference < 0.0):
            raise RobertValidationError(f"{name} must be strictly decreasing")
        if increasing is None:
            strictly_monotonic = np.all(difference > 0.0) or np.all(difference < 0.0)
            duplicate = np.flatnonzero(difference == 0.0)
            endpoint_duplicate = (
                allow_endpoint_duplicate
                and duplicate.size == 1
                and duplicate[0] in {0, difference.size - 1}
                and (
                    difference.size == 1
                    or np.all(difference[difference != 0.0] > 0.0)
                    or np.all(difference[difference != 0.0] < 0.0)
                )
            )
            if not strictly_monotonic and not endpoint_duplicate:
                raise RobertValidationError(f"{name} must be strictly monotonic")
    array.setflags(write=False)
    return array


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must be one-dimensional and finite")
    array.setflags(write=False)
    return array


def _validate_native_shape(
    native_shape: tuple[int, ...],
    pressure: NDArray[np.float64],
    temperature: NDArray[np.float64],
    wavelength: NDArray[np.float64],
) -> None:
    if any(value < 1 for value in native_shape):
        raise RobertValidationError(
            "line-by-line cross-section shape dimensions must be positive"
        )
    canonical = (pressure.size, temperature.size, wavelength.size)
    if native_shape not in {canonical, (*canonical, 1)}:
        raise RobertValidationError(
            "line-by-line cross-section shape must be pressure x temperature x wavelength "
            "or pressure x temperature x wavelength x 1"
        )


def _canonical_spectral_size(native_shape: tuple[int, ...]) -> int:
    """Return the spectral length from a 3-D or singleton-4-D source shape."""

    if len(native_shape) == 3:
        return int(native_shape[-1])
    if len(native_shape) == 4 and native_shape[-1] == 1:
        return int(native_shape[-2])
    raise RobertValidationError(
        "line-by-line cross-section shape must be pressure x temperature x wavelength "
        "or pressure x temperature x wavelength x 1"
    )


def _find_hdf_dataset(handle, names: tuple[str, ...], label: str):
    for name in names:
        if name in handle:
            dataset = handle[name]
            if not hasattr(dataset, "shape"):
                raise RobertValidationError(
                    f"line-by-line HDF5 {label} dataset {name!r} is not an array"
                )
            return name, dataset
    raise RobertValidationError(
        "line-by-line HDF5 file is missing " + label + " dataset"
    )


def _hdf_attribute(dataset, name: str) -> str:
    if name not in dataset.attrs:
        return ""
    return _text_value(dataset.attrs[name])


def _hdf_axis_unit(
    handle,
    dataset,
    dataset_name: str,
    role: str,
    *,
    canonical_unit: str,
) -> tuple[str, bool]:
    value = _hdf_attribute(dataset, "units")
    if not value:
        for name in (f"{role}_unit", f"{role}_units"):
            if name in handle.attrs:
                value = _text_value(handle.attrs[name])
                break
    if value:
        return value, False
    if dataset_name in {
        "pressure_bar",
        "temperature_K",
        "wavelength_micron",
        "bin_edges",
        "wavenumber_cm_inverse",
    }:
        return canonical_unit, True
    raise RobertValidationError(
        f"line-by-line HDF5 {role} dataset must declare its units"
    )


def _hdf_cross_section_unit(handle, dataset, dataset_name: str) -> str:
    value = _hdf_attribute(dataset, "units")
    if not value:
        for name in ("cross_section_unit", "cross_section_units", "opacity_unit"):
            if name in handle.attrs:
                value = _text_value(handle.attrs[name])
                break
    if not value:
        raise RobertValidationError(
            f"line-by-line HDF5 cross-section dataset {dataset_name!r} must declare its units"
        )
    return value


def _hdf_metadata(handle) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in handle.attrs.items():
        text = _text_value(value)
        if text:
            metadata[str(key)] = text
    return metadata


def _find_npz_array(archive, names: tuple[str, ...], label: str):
    for name in names:
        if name in archive.files:
            return name, archive[name]
    raise RobertValidationError("line-by-line NPZ file is missing " + label + " array")


def _npz_unit(
    archive,
    names: tuple[str, ...],
    dataset_name: str,
    canonical_unit: str,
) -> tuple[str, bool]:
    for name in names:
        if name in archive.files:
            value = _text_value(archive[name])
            if not value:
                raise RobertValidationError(
                    f"line-by-line NPZ unit array {name!r} must not be empty"
                )
            return value, False
    if dataset_name in {
        "pressure_bar",
        "temperature_K",
        "wavelength_micron",
    }:
        return canonical_unit, True
    return canonical_unit, True


def _npz_metadata(archive) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if "metadata_json" in archive.files:
        raw = _text_value(archive["metadata_json"])
        if raw:
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RobertValidationError(
                    "line-by-line NPZ metadata_json must contain valid JSON"
                ) from exc
            if not isinstance(decoded, Mapping):
                raise RobertValidationError(
                    "line-by-line NPZ metadata_json must contain an object"
                )
            metadata.update({str(key): str(value) for key, value in decoded.items()})
    return metadata


def _npz_array_header(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    member_name = f"{key}.npy"
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if member_name not in archive.namelist():
                raise RobertValidationError(
                    f"line-by-line NPZ file is missing array member {member_name!r}"
                )
            with archive.open(member_name, "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                elif version == (3, 0):
                    # NumPy 2.x does not expose a public 3.0 helper.  The
                    # private reader has the same header contract and keeps
                    # this metadata-only path independent of NumPy version.
                    reader = getattr(np.lib.format, "_read_array_header", None)
                    if reader is None:
                        raise RobertValidationError(
                            "NPZ format 3.0 headers are unsupported by this NumPy version"
                        )
                    shape, _, dtype = reader(stream, version)
                else:
                    raise RobertValidationError(
                        f"unsupported NPZ array header version for {member_name!r}"
                    )
    except RobertValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise RobertValidationError(
            f"could not inspect line-by-line NPZ array header: {path}"
        ) from exc
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _text_value(value: object) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace").strip()
    return str(item).strip()


def _temperature_values_in_kelvin(values: ArrayLike, unit: str) -> NDArray[np.float64]:
    normalized = str(unit).strip().lower()
    if normalized not in {"k", "kelvin"}:
        raise RobertValidationError(
            "line-by-line temperature axes must use kelvin"
        )
    return _readonly_axis(values, "temperature_K", minimum_size=2, increasing=True)


def _wavelength_values_in_micron(
    values: ArrayLike,
    unit: str,
) -> NDArray[np.float64]:
    normalized = str(unit).strip().lower().replace("μ", "u")
    factors = {
        "micron": 1.0,
        "microns": 1.0,
        "um": 1.0,
        "u m": 1.0,
        "nm": 1.0e-3,
        "nanometer": 1.0e-3,
        "nanometers": 1.0e-3,
        "angstrom": 1.0e-4,
        "angstroms": 1.0e-4,
        "a": 1.0e-4,
        "m": 1.0e6,
        "meter": 1.0e6,
        "meters": 1.0e6,
        "cm": 1.0e4,
    }
    if normalized not in factors:
        raise RobertValidationError(
            "line-by-line wavelength axes must use a physical length unit"
        )
    return _readonly_axis(
        np.asarray(values, dtype=float) * factors[normalized],
        "wavelength_micron",
        minimum_size=1,
        increasing=None,
    )


def _wavenumber_values_in_cm_inverse(
    values: ArrayLike,
    unit: str,
) -> NDArray[np.float64]:
    normalized = str(unit).strip().lower().replace("²", "2")
    if normalized not in {
        "cm^-1",
        "cm-1",
        "1/cm",
        "wavenumber",
        "wavenumbers",
        "cm inverse",
    }:
        raise RobertValidationError(
            "line-by-line wavenumber axes must use inverse centimetres"
        )
    return _readonly_axis(
        values,
        "wavenumber_cm_inverse",
        minimum_size=1,
        increasing=None,
        allow_endpoint_duplicate=True,
    )


def _wavelength_values_from_wavenumber(
    wavenumber_cm_inverse: NDArray[np.float64],
) -> NDArray[np.float64]:
    wavelength = 10000.0 / np.asarray(wavenumber_cm_inverse, dtype=float)
    return _readonly_axis(
        wavelength,
        "wavelength_micron",
        minimum_size=1,
        increasing=None,
        allow_endpoint_duplicate=True,
    )


def _species_tuple(values: Iterable[str], name: str) -> tuple[str, ...]:
    result = tuple(str(item).strip() for item in values)
    if not result or any(not item for item in result):
        raise RobertValidationError(f"{name} must contain non-empty names")
    return result


def _sampling_stride(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RobertValidationError(f"{name} must be a positive integer")
    stride = int(value)
    if stride < 1:
        raise RobertValidationError(f"{name} must be a positive integer")
    return stride


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RobertValidationError(f"{name} must be a non-negative integer")
    integer = int(value)
    if integer < 0:
        raise RobertValidationError(f"{name} must be a non-negative integer")
    return integer


def _wavelength_bounds(
    value: tuple[float, float] | None,
    name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RobertValidationError(
            f"{name} must contain two finite, positive, increasing values"
        ) from exc
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or lower >= upper
    ):
        raise RobertValidationError(
            f"{name} must contain two finite, positive, increasing values"
        )
    return (lower, upper)


def _grid_sampling_stride(spectral_grid: SpectralGrid, default: int) -> int:
    value = spectral_grid.metadata.get("native_sampling_stride")
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RobertValidationError(
            "spectral-grid native_sampling_stride metadata must be a positive integer"
        ) from exc
    if str(value).strip() != str(parsed) or parsed < 1:
        raise RobertValidationError(
            "spectral-grid native_sampling_stride metadata must be a positive integer"
        )
    return parsed


def _memory_limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RobertValidationError(f"{name} must be a positive integer or None")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise RobertValidationError(f"{name} must be a positive integer or None") from exc
    if limit < 1:
        raise RobertValidationError(f"{name} must be a positive integer or None")
    return limit


def _exact_wavelength_indices(
    spectral_grid: SpectralGrid,
    table: LineByLineTable,
) -> NDArray[np.int64]:
    requested = spectral_grid_values_in_unit(spectral_grid, "micron")
    source = table.wavelength_micron
    if source.size == 1:
        indices = np.zeros(requested.size, dtype=np.int64)
        if not np.allclose(
            source[0], requested, rtol=2.0e-10, atol=1.0e-12
        ):
            raise RobertCoverageError(
                "line-by-line wavelength grid must be an exact subset of the source grid"
            )
        indices.setflags(write=False)
        return indices

    if source[0] < source[-1]:
        sorted_source = source
        sorted_to_source = np.arange(source.size, dtype=np.int64)
    else:
        sorted_source = source[::-1]
        sorted_to_source = np.arange(source.size - 1, -1, -1, dtype=np.int64)
    insertion = np.searchsorted(sorted_source, requested, side="left")
    right = np.clip(insertion, 0, sorted_source.size - 1)
    left = np.clip(insertion - 1, 0, sorted_source.size - 1)
    choose_left = np.abs(sorted_source[left] - requested) <= np.abs(
        sorted_source[right] - requested
    )
    selected_sorted = np.where(choose_left, left, right)
    indices = sorted_to_source[selected_sorted].astype(np.int64)
    if not np.allclose(
        source[indices], requested, rtol=2.0e-10, atol=1.0e-12
    ):
        raise RobertCoverageError(
            "line-by-line wavelength grid must be an exact subset of the source grid"
        )
    indices.setflags(write=False)
    return indices


def _wavelength_intersection_mask(
    requested: NDArray[np.float64],
    source: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Return which requested wavelengths are exact samples in ``source``."""

    requested = np.asarray(requested, dtype=float)
    if requested.size == 0:
        return np.zeros(0, dtype=bool)
    if source[0] < source[-1]:
        sorted_source = source
    else:
        sorted_source = source[::-1]
    insertion = np.searchsorted(sorted_source, requested, side="left")
    right = np.clip(insertion, 0, sorted_source.size - 1)
    left = np.clip(insertion - 1, 0, sorted_source.size - 1)
    nearest = np.where(
        np.abs(sorted_source[left] - requested)
        <= np.abs(sorted_source[right] - requested),
        sorted_source[left],
        sorted_source[right],
    )
    return np.isclose(nearest, requested, rtol=2.0e-10, atol=1.0e-12)


def _validate_axis_coverage(
    requested: ArrayLike,
    grid: NDArray[np.float64],
    name: str,
    *,
    clip: bool,
) -> None:
    values = np.asarray(requested, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RobertValidationError(f"{name} values must be finite and positive")
    if not clip and (np.any(values < grid[0]) or np.any(values > grid[-1])):
        raise RobertCoverageError(
            f"{name} values are outside line-by-line table coverage"
        )


def _estimate_table_memory_bytes(table: LineByLineTable, spectral_count: int) -> int:
    if spectral_count < 1:
        raise RobertValidationError("line-by-line spectral sample count must be positive")
    selected_elements = (
        table.pressure_bar.size * table.temperature_K.size * int(spectral_count)
    )
    selected_bytes = int(selected_elements) * np.dtype(np.float64).itemsize
    if table.metadata.get("source_format") == "line_by_line_npz":
        native_bytes = table.native_nbytes
        if native_bytes is None:
            raise RobertValidationError(
                "NPZ line-by-line table must declare its native shape before preparation"
            )
        # np.load decodes the complete ZIP member.  Include the decoded source,
        # the float64 selected copy, and the log-space preparation array.
        return int(native_bytes + selected_bytes * 2)
    # HDF5 reads a selected hyperslab, then keeps source, log, and output-sized
    # temporaries during preparation.  This is deliberately conservative.
    return int(selected_bytes * 3)


def _load_selected_log_cross_sections(
    table: LineByLineTable,
    indices: NDArray[np.int64],
    *,
    floor: float,
) -> NDArray[np.float64]:
    strategy = table.metadata.get("source_format", "")
    if strategy == "line_by_line_hdf5":
        values = _read_hdf_selected(table, indices)
    elif strategy == "line_by_line_npz":
        values = _read_npz_selected(table, indices)
    else:
        raise RobertValidationError(
            "line-by-line table metadata must identify an HDF5 or NPZ source format"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise RobertValidationError(
            "selected line-by-line cross sections must be finite and non-negative"
        )
    result = np.log(np.maximum(values, floor))
    if not np.all(np.isfinite(result)):
        raise RobertValidationError(
            "selected line-by-line log cross sections must be finite"
        )
    result.setflags(write=False)
    return result


def _read_hdf_selected(
    table: LineByLineTable,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    h5py = _require_h5py()
    order = np.argsort(indices, kind="stable")
    sorted_indices = np.asarray(indices[order], dtype=np.int64)
    inverse = np.argsort(order, kind="stable")
    try:
        with h5py.File(table.path, "r") as handle:
            dataset = handle[table.dataset]
            if dataset.ndim == 3:
                values = np.asarray(dataset[:, :, sorted_indices], dtype=float)
            elif dataset.ndim == 4 and dataset.shape[-1] == 1:
                values = np.asarray(
                    dataset[:, :, sorted_indices, 0],
                    dtype=float,
                )
            else:  # pragma: no cover - table validation catches this first
                raise RobertValidationError(
                    "line-by-line HDF5 cross-section array has an invalid shape"
                )
    except RobertValidationError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RobertValidationError(
            f"could not read line-by-line HDF5 cross sections from {table.path}"
        ) from exc
    return np.asarray(values[:, :, inverse], dtype=float)


def _read_npz_selected(
    table: LineByLineTable,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    try:
        with np.load(table.path, allow_pickle=False) as archive:
            source = np.asarray(archive[table.dataset], dtype=float)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise RobertValidationError(
            f"could not read line-by-line NPZ cross sections from {table.path}"
        ) from exc
    canonical = _canonical_cross_sections(source, table)
    return np.asarray(canonical[:, :, indices], dtype=float)


def _canonical_cross_sections(
    values: NDArray[np.float64],
    table: LineByLineTable,
) -> NDArray[np.float64]:
    if values.shape == table.canonical_shape:
        return values
    if values.shape == (*table.canonical_shape, 1):
        return values[:, :, :, 0]
    raise RobertValidationError(
        "line-by-line cross-section array shape does not match its table axes"
    )


def _interpolate_log_cross_sections(
    log_values: NDArray[np.float64],
    pressure: NDArray[np.float64],
    temperature: NDArray[np.float64],
    pressure_grid: NDArray[np.float64],
    temperature_grid: NDArray[np.float64],
    *,
    clip: bool,
) -> NDArray[np.float64]:
    requested_log_pressure = np.log(pressure)
    log_pressure_grid = np.log(pressure_grid)
    requested_temperature = np.asarray(temperature, dtype=float)
    if clip:
        requested_log_pressure = np.clip(
            requested_log_pressure,
            log_pressure_grid[0],
            log_pressure_grid[-1],
        )
        requested_temperature = np.clip(
            requested_temperature,
            temperature_grid[0],
            temperature_grid[-1],
        )
    p0, p1, wp = _brackets(requested_log_pressure, log_pressure_grid)
    t0, t1, wt = _brackets(requested_temperature, temperature_grid)
    wp = wp[:, None]
    wt = wt[:, None]
    lower = (1.0 - wt) * log_values[p0, t0, :] + wt * log_values[p0, t1, :]
    upper = (1.0 - wt) * log_values[p1, t0, :] + wt * log_values[p1, t1, :]
    return (1.0 - wp) * lower + wp * upper


def _cache_key(
    name: str,
    interpolation: str,
    floor: float,
    species: tuple[str, ...],
    spectral_grid: SpectralGrid,
    pressure_grid: PressureGrid,
    tables: tuple[LineByLineTable, ...],
    indices: tuple[NDArray[np.int64], ...],
    *,
    wavelength_bounds_micron: tuple[float, float] | None = None,
    native_sampling_stride: int = 1,
) -> str:
    digest = sha256()
    for text in (
        name,
        interpolation,
        repr(float(floor)),
        *species,
        spectral_grid.unit,
        pressure_grid.unit,
        repr(wavelength_bounds_micron),
        str(native_sampling_stride),
    ):
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    for array in (
        spectral_grid.values,
        pressure_grid.edges,
        pressure_grid.centers,
        *indices,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    for table in tables:
        digest.update(str(table.path).encode("utf-8"))
        digest.update(str(table.metadata.get("checksum_sha256", "")).encode("utf-8"))
        digest.update(table.dataset.encode("utf-8"))
        try:
            stat = table.path.stat()
        except OSError:
            stat = None
        if stat is not None:
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "EvaluatedLineByLineMixture",
    "EvaluatedLineByLineOpacity",
    "LineByLineCoverageReport",
    "LineByLineOpacityProvider",
    "LineByLineTable",
    "PreparedLineByLineOpacity",
]
