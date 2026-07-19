"""Correlated-k opacity evaluation on native grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping

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

from .archive import load_robert_npy_directory, load_robert_npz_archive
from .kta import KtaTable, read_kta
from .metadata import pressure_values_in_unit, spectral_grid_values_in_unit


@dataclass(frozen=True)
class CorrelatedKTable:
    """One species' correlated-k table in ROBERT native axis order."""

    species: str
    pressure_bar: NDArray[np.float64]
    temperature_K: NDArray[np.float64]
    wavenumber_cm_inverse: NDArray[np.float64]
    g_samples: NDArray[np.float64]
    g_weights: NDArray[np.float64]
    kcoeff: NDArray[np.float64]
    wavelength_micron: NDArray[np.float64] | None = None
    unit: str = "cm^2/molecule"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.species:
            raise RobertValidationError("correlated-k table species must not be empty")
        pressure = _readonly_1d(self.pressure_bar, "pressure_bar")
        temperature = _readonly_1d(self.temperature_K, "temperature_K")
        wavenumber = _readonly_1d(self.wavenumber_cm_inverse, "wavenumber_cm_inverse")
        g_samples = _readonly_1d(self.g_samples, "g_samples")
        g_weights = _readonly_1d(self.g_weights, "g_weights")
        if np.any(pressure <= 0.0):
            raise RobertValidationError("pressure_bar values must be positive")
        if np.any(temperature <= 0.0):
            raise RobertValidationError("temperature_K values must be positive")
        if np.any(wavenumber <= 0.0):
            raise RobertValidationError("wavenumber_cm_inverse values must be positive")
        if not _is_strictly_monotonic_or_single(pressure):
            raise RobertValidationError("pressure_bar must be strictly monotonic")
        if not _is_strictly_monotonic_or_single(temperature):
            raise RobertValidationError("temperature_K must be strictly monotonic")
        if not _is_strictly_monotonic_or_single(wavenumber):
            raise RobertValidationError(
                "wavenumber_cm_inverse must be strictly monotonic"
            )
        if g_samples.shape != g_weights.shape:
            raise RobertValidationError(
                "g_samples and g_weights must have the same shape"
            )
        if np.any(g_weights < 0.0):
            raise RobertValidationError("g_weights must be non-negative")
        g_weight_sum = float(np.sum(g_weights))
        if g_weight_sum <= 0.0 or not np.isfinite(g_weight_sum):
            raise RobertValidationError("g_weights must have a finite positive sum")
        if not np.isclose(g_weight_sum, 1.0, rtol=1.0e-6, atol=1.0e-8):
            raise RobertValidationError("g_weights must sum to one")
        normalized_g_weights = np.array(
            g_weights / g_weight_sum, dtype=float, copy=True
        )
        normalized_g_weights.setflags(write=False)

        wavelength = None
        if self.wavelength_micron is not None:
            wavelength = _readonly_1d(self.wavelength_micron, "wavelength_micron")
            if wavelength.shape != wavenumber.shape:
                raise RobertValidationError(
                    "wavelength_micron must match wavenumber grid shape"
                )
            if np.any(wavelength <= 0.0):
                raise RobertValidationError("wavelength_micron values must be positive")
        else:
            wavelength = 10000.0 / wavenumber
            wavelength.setflags(write=False)

        kcoeff = np.array(self.kcoeff, dtype=float, copy=True)
        expected_shape = (
            pressure.size,
            temperature.size,
            wavenumber.size,
            g_samples.size,
        )
        if kcoeff.shape != expected_shape:
            raise RobertValidationError(
                "kcoeff shape must be pressure x temperature x wavelength x g"
            )
        if not np.all(np.isfinite(kcoeff)) or np.any(kcoeff < 0.0):
            raise RobertValidationError("kcoeff values must be finite and non-negative")
        kcoeff.setflags(write=False)
        if not self.unit:
            raise RobertValidationError("correlated-k unit must not be empty")

        object.__setattr__(self, "pressure_bar", pressure)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "wavenumber_cm_inverse", wavenumber)
        object.__setattr__(self, "wavelength_micron", wavelength)
        object.__setattr__(self, "g_samples", g_samples)
        object.__setattr__(self, "g_weights", normalized_g_weights)
        object.__setattr__(self, "kcoeff", kcoeff)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @classmethod
    def from_kta(cls, species: str, table: KtaTable) -> "CorrelatedKTable":
        """Build a correlated-k table from a loaded `.kta` table."""

        return cls(
            species=species,
            pressure_bar=table.header.pressure_bar,
            temperature_K=table.header.temperature_K,
            wavenumber_cm_inverse=table.header.wavenumber_cm_inverse,
            wavelength_micron=table.header.wavelength_micron,
            g_samples=table.header.g_samples,
            g_weights=table.header.g_weights,
            kcoeff=table.kcoeff,
            unit=table.unit,
            metadata={
                "source_format": "kta_binary",
                "source_path": table.header.path,
                "checksum_sha256": ""
                if table.header.checksum_sha256 is None
                else table.header.checksum_sha256,
                **dict(table.metadata),
            },
        )

    @classmethod
    def from_robert_archive(
        cls,
        path: str | Path,
        *,
        species: str | None = None,
        mmap_mode: str | None = "r",
    ) -> "CorrelatedKTable":
        """Load one correlated-k table from a ROBERT native archive."""

        archive_path = Path(path).expanduser()
        if archive_path.is_dir():
            archive = load_robert_npy_directory(archive_path, mmap_mode=mmap_mode)
        else:
            archive = load_robert_npz_archive(archive_path)
        if len(archive.database.products) != 1:
            raise RobertValidationError(
                "only single-product correlated-k archives are currently supported"
            )
        product = archive.database.products[0]
        species_name = species or product.species[0]
        arrays = archive.arrays
        required = (
            "kcoeff",
            "pressure_bar",
            "temperature_K",
            "wavenumber_cm-1",
            "g_samples",
            "g_weights",
        )
        missing = tuple(name for name in required if name not in arrays)
        if missing:
            raise RobertValidationError(
                f"ROBERT opacity archive is missing arrays: {', '.join(missing)}"
            )
        wavelength = arrays.get("wavelength_micron")
        return cls(
            species=species_name,
            pressure_bar=arrays["pressure_bar"],
            temperature_K=arrays["temperature_K"],
            wavenumber_cm_inverse=arrays["wavenumber_cm-1"],
            wavelength_micron=None if wavelength is None else wavelength,
            g_samples=arrays["g_samples"],
            g_weights=arrays["g_weights"],
            kcoeff=arrays["kcoeff"],
            unit=str(product.metadata.get("kcoeff_unit", "cm^2/molecule")),
            metadata={
                "source_format": product.storage_format.value,
                "source_path": "" if product.path is None else product.path,
                **dict(product.metadata),
            },
        )

    @classmethod
    def from_petitradtrans_hdf(
        cls,
        path: str | Path,
        *,
        species: str,
    ) -> "CorrelatedKTable":
        """Load a petitRADTRANS correlated-k HDF5 table without pRT.

        petitRADTRANS stores coefficients in pressure, temperature,
        wavenumber, and g-ordinate order, which is ROBERT's native order.
        Units are read from the file and are never inferred silently.
        """

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RobertValidationError(
                "loading petitRADTRANS HDF5 opacity requires h5py"
            ) from exc

        source = Path(path).expanduser()
        required = ("p", "t", "bin_centers", "samples", "weights", "kcoeff")
        try:
            with h5py.File(source, "r") as handle:
                missing = tuple(name for name in required if name not in handle)
                if missing:
                    raise RobertValidationError(
                        "petitRADTRANS opacity file is missing datasets: "
                        + ", ".join(missing)
                    )
                unit = str(handle["kcoeff"].attrs.get("units", "")).strip()
                if not unit:
                    raise RobertValidationError(
                        "petitRADTRANS kcoeff dataset must declare its units"
                    )
                pressure = np.asarray(handle["p"], dtype=float)
                temperature = np.asarray(handle["t"], dtype=float)
                wavenumber = np.asarray(handle["bin_centers"], dtype=float)
                g_samples = np.asarray(handle["samples"], dtype=float)
                g_weights = np.asarray(handle["weights"], dtype=float)
                kcoeff = np.asarray(handle["kcoeff"], dtype=float)
                doi = _decode_hdf_scalar(handle.get("DOI"))
                method = _decode_hdf_scalar(handle.get("method"))
        except OSError as exc:
            raise RobertValidationError(
                f"could not read petitRADTRANS opacity file: {source}"
            ) from exc

        return cls(
            species=species,
            pressure_bar=pressure,
            temperature_K=temperature,
            wavenumber_cm_inverse=wavenumber,
            wavelength_micron=10000.0 / wavenumber,
            g_samples=g_samples,
            g_weights=g_weights,
            kcoeff=kcoeff,
            unit=unit,
            metadata={
                "source_format": "petitradtrans_hdf5",
                "source_path": str(source.resolve()),
                "doi": doi,
                "correlated_k_method": method,
            },
        )

    @classmethod
    def from_exomol_cross_section_hdf(
        cls,
        path: str | Path,
        *,
        species: str,
        spectral_grid: SpectralGrid,
        g_points: int = 8,
        checksum: bool = True,
    ) -> "CorrelatedKTable":
        """Correlate real ExoMolOP cross sections within target bins.

        At every source pressure and temperature, the native cross sections in
        each target wavelength bin are sorted into an empirical cumulative
        distribution. Native samples are weighted by their wavelength-cell
        widths so the distribution represents a linear-wavelength top-hat
        integral. Gauss-Legendre nodes sample that distribution. This changes
        only the spectral representation; it does not synthesize or otherwise
        modify the source molecular cross sections.
        """

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RobertValidationError(
                "loading ExoMol cross sections requires h5py"
            ) from exc
        if (
            isinstance(g_points, bool)
            or int(g_points) != g_points
            or int(g_points) < 2
        ):
            raise RobertValidationError("g_points must be an integer of at least two")
        if spectral_grid.bin_edges is None:
            raise RobertValidationError(
                "ExoMol cross-section correlation requires spectral bin edges"
            )
        source = Path(path).expanduser().resolve()
        edge_grid = SpectralGrid(
            values=spectral_grid.bin_edges,
            unit=spectral_grid.unit,
            role="internal",
        )
        target_wavelength = spectral_grid_values_in_unit(
            spectral_grid,
            "micron",
        )
        target_edges = spectral_grid_values_in_unit(edge_grid, "micron")
        nodes, weights = np.polynomial.legendre.leggauss(int(g_points))
        g_samples = 0.5 * (nodes + 1.0)
        g_weights = 0.5 * weights
        try:
            with h5py.File(source, "r") as handle:
                required = {"p", "t", "bin_edges", "xsecarr"}
                missing = sorted(required - set(handle))
                if missing:
                    raise RobertValidationError(
                        "ExoMol cross-section HDF is missing datasets: "
                        + ", ".join(missing)
                    )
                pressure = np.asarray(handle["p"], dtype=float)
                temperature = np.asarray(handle["t"], dtype=float)
                source_wavenumber = np.asarray(handle["bin_edges"], dtype=float)
                cross_sections = handle["xsecarr"]
                expected = (
                    pressure.size,
                    temperature.size,
                    source_wavenumber.size,
                )
                if cross_sections.shape != expected:
                    raise RobertValidationError(
                        "ExoMol xsecarr shape must be pressure x temperature x wavenumber"
                    )
                if np.any(np.diff(source_wavenumber) <= 0.0):
                    raise RobertValidationError(
                        "ExoMol cross-section wavenumber grid must increase"
                    )
                unit = str(cross_sections.attrs.get("units", "")).strip()
                if unit != "cm^2/molecule":
                    raise RobertValidationError(
                        "ExoMol cross sections must declare cm^2/molecule units"
                    )
                coefficients = np.empty(
                    (
                        pressure.size,
                        temperature.size,
                        spectral_grid.size,
                        int(g_points),
                    ),
                    dtype=float,
                )
                for index, (left, right) in enumerate(
                    zip(target_edges[:-1], target_edges[1:], strict=True)
                ):
                    lower = min(10000.0 / left, 10000.0 / right)
                    upper = max(10000.0 / left, 10000.0 / right)
                    start = int(
                        np.searchsorted(source_wavenumber, lower, side="left")
                    )
                    stop = int(
                        np.searchsorted(source_wavenumber, upper, side="right")
                    )
                    if stop - start < int(g_points):
                        raise RobertCoverageError(
                            f"target bin {index} contains fewer than {g_points} ExoMol samples"
                        )
                    native = np.asarray(
                        cross_sections[:, :, start:stop],
                        dtype=float,
                    )[:, :, ::-1]
                    native_wavelength = (10000.0 / source_wavenumber[start:stop])[::-1]
                    bin_lower = min(left, right)
                    bin_upper = max(left, right)
                    cell_edges = np.concatenate(
                        (
                            [bin_lower],
                            0.5 * (native_wavelength[:-1] + native_wavelength[1:]),
                            [bin_upper],
                        )
                    )
                    sample_weights = np.diff(cell_edges)
                    if np.any(sample_weights <= 0.0):
                        raise RobertValidationError(
                            "ExoMol wavelength cells must have positive widths"
                        )
                    quantiles = _weighted_quantiles(
                        native,
                        sample_weights,
                        g_samples,
                    )
                    coefficients[:, :, index, :] = np.transpose(
                        quantiles,
                        (1, 2, 0),
                    )
                doi = _decode_hdf_scalar(handle.get("DOI"))
                line_list = _decode_hdf_scalar(handle.get("key_iso_ll"))
        except OSError as exc:
            raise RobertValidationError(
                f"could not read ExoMol cross sections: {source}"
            ) from exc
        coefficients = np.maximum(coefficients, 1.0e-300)
        return cls(
            species=species,
            pressure_bar=pressure,
            temperature_K=temperature,
            wavenumber_cm_inverse=10000.0 / target_wavelength,
            wavelength_micron=target_wavelength,
            g_samples=g_samples,
            g_weights=g_weights,
            kcoeff=coefficients,
            unit=unit,
            metadata={
                "source_format": "exomol_cross_section_hdf5",
                "source_path": str(source),
                "checksum_sha256": _path_sha256(source) if checksum else "",
                "doi": doi,
                "line_list": line_list,
                "spectral_preparation": (
                    "wavelength_weighted_empirical_target_bin_correlated_k"
                ),
                "g_points": str(int(g_points)),
            },
        )


@dataclass(frozen=True)
class PreparedCorrelatedKOpacity:
    """Run-specific correlated-k preparation state."""

    provider_name: str
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    species: tuple[str, ...]
    g_samples: NDArray[np.float64]
    g_weights: NDArray[np.float64]
    cache_key: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    spectral_indices: Mapping[str, NDArray[np.int64]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.provider_name:
            raise RobertValidationError("provider_name must not be empty")
        species = tuple(str(item) for item in self.species)
        if not species or any(not item for item in species):
            raise RobertValidationError(
                "prepared correlated-k opacity species must be non-empty"
            )
        g_samples = _readonly_1d(self.g_samples, "g_samples")
        g_weights = _readonly_1d(self.g_weights, "g_weights")
        if g_samples.shape != g_weights.shape:
            raise RobertValidationError(
                "g_samples and g_weights must have the same shape"
            )
        if not self.cache_key:
            raise RobertValidationError("cache_key must not be empty")
        spectral_indices: dict[str, NDArray[np.int64]] = {}
        for species_name, values in self.spectral_indices.items():
            if species_name not in species:
                raise RobertValidationError(
                    "prepared spectral-index species must be present in prepared species"
                )
            indices = np.array(values, dtype=np.int64, copy=True)
            if indices.shape != (self.spectral_grid.size,) or np.any(indices < 0):
                raise RobertValidationError(
                    "prepared spectral indices must match the spectral grid and be non-negative"
                )
            indices.setflags(write=False)
            spectral_indices[species_name] = indices
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "g_samples", g_samples)
        object.__setattr__(self, "g_weights", g_weights)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(
            self, "spectral_indices", immutable_mapping(spectral_indices)
        )


@dataclass(frozen=True)
class EvaluatedCorrelatedKOpacity:
    """Evaluated correlated-k coefficients on a model atmosphere."""

    prepared: PreparedCorrelatedKOpacity
    kcoeff: NDArray[np.float64]
    unit: str = "cm^2/molecule"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kcoeff = np.array(self.kcoeff, dtype=float, copy=True)
        expected_shape = (
            len(self.prepared.species),
            self.prepared.pressure_grid.n_layers,
            self.prepared.spectral_grid.size,
            self.prepared.g_weights.size,
        )
        if kcoeff.shape != expected_shape:
            raise RobertValidationError(
                "kcoeff shape must be species x layers x wavelength x g"
            )
        if not np.all(np.isfinite(kcoeff)) or np.any(kcoeff < 0.0):
            raise RobertValidationError(
                "evaluated kcoeff values must be finite and non-negative"
            )
        if not self.unit:
            raise RobertValidationError("evaluated correlated-k unit must not be empty")
        kcoeff.setflags(write=False)
        object.__setattr__(self, "kcoeff", kcoeff)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class CorrelatedKCoverageReport:
    """Coverage result for correlated-k native-grid evaluation."""

    valid: bool
    message: str
    species: tuple[str, ...]
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "reasons", immutable_mapping(self.reasons))


@dataclass(frozen=True)
class CorrelatedKOpacityProvider:
    """Correlated-k opacity provider on a fixed native spectral grid.

    The default mode performs exact pressure, temperature, and spectral lookup
    for benchmark cases. The optional interpolation mode keeps the spectral
    grid exact and interpolates log(k) bilinearly in log-pressure and
    temperature.
    """

    tables: Mapping[str, CorrelatedKTable]
    name: str = "correlated-k-native"
    interpolation: str = "exact"
    cache_log_kcoeff: bool = True
    _log_kcoeff: Mapping[str, NDArray[np.float64]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("correlated-k provider name must not be empty")
        if self.interpolation not in {
            "exact",
            "log_pressure_temperature_log_k",
            "log_pressure_temperature_log_k_clip",
        }:
            raise RobertValidationError(
                "interpolation must be 'exact', 'log_pressure_temperature_log_k', "
                "or 'log_pressure_temperature_log_k_clip'"
            )
        tables = {str(species): table for species, table in self.tables.items()}
        if not tables:
            raise RobertValidationError(
                "correlated-k provider must contain at least one table"
            )
        for species, table in tables.items():
            if species != table.species:
                raise RobertValidationError(
                    "correlated-k table mapping key must match table species"
                )
        if not isinstance(self.cache_log_kcoeff, bool):
            raise RobertValidationError("cache_log_kcoeff must be a boolean")
        _validate_common_g_grid(tuple(tables.values()))
        log_kcoeff: dict[str, NDArray[np.float64]] = {}
        if self.cache_log_kcoeff and self.interpolation != "exact":
            for species, table in tables.items():
                values = np.log(np.maximum(table.kcoeff, 1.0e-300))
                values.setflags(write=False)
                log_kcoeff[species] = values
        object.__setattr__(self, "tables", immutable_mapping(tables))
        object.__setattr__(self, "_log_kcoeff", immutable_mapping(log_kcoeff))

    @classmethod
    def from_kta_paths(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "correlated-k-native",
        interpolation: str = "exact",
        nonfinite_policy: str = "raise",
        nonfinite_fill_value: float = 1.0e-300,
        cache_log_kcoeff: bool = True,
    ) -> "CorrelatedKOpacityProvider":
        """Build a provider from species-to-`.kta` file paths."""

        tables = {
            str(species): CorrelatedKTable.from_kta(
                str(species),
                read_kta(
                    path,
                    checksum=True,
                    nonfinite_policy=nonfinite_policy,
                    nonfinite_fill_value=nonfinite_fill_value,
                ),
            )
            for species, path in paths.items()
        }
        return cls(
            tables=tables,
            name=name,
            interpolation=interpolation,
            cache_log_kcoeff=cache_log_kcoeff,
        )

    @classmethod
    def from_exomol_kta_directory(
        cls,
        directory: str | Path,
        *,
        species: Iterable[str] | None = None,
        resolution: str | int | None = None,
        filename_pattern: str = "*.kta",
        name: str = "exomol-correlated-k",
        interpolation: str = "exact",
        nonfinite_policy: str = "raise",
        nonfinite_fill_value: float = 1.0e-300,
        cache_log_kcoeff: bool = True,
    ) -> "CorrelatedKOpacityProvider":
        """Discover precomputed ExoMol/exo_k KTA tables in a user directory.

        Species names are inferred from the filename token before the first
        underscore. If multiple files map to one species, callers must choose
        explicitly with :meth:`from_kta_paths` so resolution/isotopologue
        selection is never implicit. When ``resolution`` is supplied, ROBERT
        accepts a parent containing an ``R<value>`` subdirectory, the
        resolution directory itself, or a flat directory containing matching
        ``*_<resolution>.kta`` files. Only that requested resolution is loaded.
        """

        root = Path(directory).expanduser()
        resolution_name = _normalize_kta_resolution(resolution)
        if resolution_name is not None:
            resolution_directory = root / resolution_name
            if resolution_directory.is_dir():
                root = resolution_directory
            elif (
                root.name.casefold() != resolution_name.casefold()
                and not any(root.glob(f"*_{resolution_name}.kta"))
            ):
                raise RobertValidationError(
                    "ExoMol opacity root does not contain the requested "
                    f"resolution directory {resolution_name}: {root}"
                )
        if not root.is_dir():
            raise RobertValidationError(
                f"ExoMol opacity directory does not exist: {root}"
            )
        pattern = str(filename_pattern).strip()
        if not pattern:
            raise RobertValidationError("filename_pattern must not be empty")
        discovered: dict[str, Path] = {}
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path.suffix.lower() != ".kta":
                continue
            if resolution_name is not None and not path.stem.casefold().endswith(
                f"_{resolution_name}".casefold()
            ):
                continue
            inferred = _species_name_from_opacity_path(path)
            key = inferred.casefold()
            if key in discovered:
                raise RobertValidationError(
                    f"multiple KTA files discovered for {inferred}; use from_kta_paths for explicit selection"
                )
            discovered[key] = path
        if not discovered:
            raise RobertValidationError(
                f"no KTA opacity tables matched {pattern!r} under {root}"
            )

        if species is None:
            selected_paths = {
                _species_name_from_opacity_path(path): path
                for path in discovered.values()
            }
        else:
            requested = tuple(str(item).strip() for item in species)
            if not requested or any(not item for item in requested):
                raise RobertValidationError("species must contain non-empty names")
            if len({item.casefold() for item in requested}) != len(requested):
                raise RobertValidationError("species must not contain duplicates")
            missing = tuple(
                item for item in requested if item.casefold() not in discovered
            )
            if missing:
                raise RobertValidationError(
                    f"missing ExoMol KTA opacity tables for species: {', '.join(missing)}"
                )
            selected_paths = {item: discovered[item.casefold()] for item in requested}
        return cls.from_kta_paths(
            selected_paths,
            name=name,
            interpolation=interpolation,
            nonfinite_policy=nonfinite_policy,
            nonfinite_fill_value=nonfinite_fill_value,
            cache_log_kcoeff=cache_log_kcoeff,
        )

    @classmethod
    def from_exok_paths(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "exo-k-correlated-k",
        interpolation: str = "exact",
        nonfinite_policy: str = "raise",
        nonfinite_fill_value: float = 1.0e-300,
        remove_zeros: bool = True,
        zero_deltalog_min_value: float = 10.0,
        cache_log_kcoeff: bool = True,
    ) -> "CorrelatedKOpacityProvider":
        """Load precomputed molecular opacity formats supported by ``exo_k``."""

        from .exok import load_correlated_k_table_with_exok

        tables = {
            str(species): load_correlated_k_table_with_exok(
                path,
                species=str(species),
                nonfinite_policy=nonfinite_policy,
                nonfinite_fill_value=nonfinite_fill_value,
                remove_zeros=remove_zeros,
                zero_deltalog_min_value=zero_deltalog_min_value,
            )
            for species, path in paths.items()
        }
        return cls(
            tables=tables,
            name=name,
            interpolation=interpolation,
            cache_log_kcoeff=cache_log_kcoeff,
        )

    @classmethod
    def from_robert_archives(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "correlated-k-native",
        interpolation: str = "exact",
        cache_log_kcoeff: bool = True,
    ) -> "CorrelatedKOpacityProvider":
        """Build a provider from species-to-ROBERT archive paths."""

        tables = {
            str(species): CorrelatedKTable.from_robert_archive(
                path, species=str(species)
            )
            for species, path in paths.items()
        }
        return cls(
            tables=tables,
            name=name,
            interpolation=interpolation,
            cache_log_kcoeff=cache_log_kcoeff,
        )

    @property
    def species(self) -> tuple[str, ...]:
        """Available opacity species."""

        return tuple(self.tables)

    def bin_to_spectral_grid(
        self,
        spectral_grid: SpectralGrid,
        *,
        backend: str = "exo_k",
        num: int = 300,
        use_rebin: bool = False,
        remove_zeros: bool = True,
        zero_deltalog_min_value: float = 10.0,
    ) -> "CorrelatedKOpacityProvider":
        """Return a provider binned to requested spectral bins.

        Correlated-k distributions are recompressed within each bin by
        ``exo_k``; individual g ordinates are never wavelength-interpolated.
        """

        normalized = backend.strip().lower().replace("-", "_")
        if normalized not in {"exo_k", "exok"}:
            raise RobertValidationError(
                "correlated-k spectral binning backend must be 'exo_k'"
            )
        from .exok import bin_correlated_k_table_with_exok

        tables = {
            species: bin_correlated_k_table_with_exok(
                table,
                spectral_grid,
                num=num,
                use_rebin=use_rebin,
                remove_zeros=remove_zeros,
                zero_deltalog_min_value=zero_deltalog_min_value,
            )
            for species, table in self.tables.items()
        }
        return CorrelatedKOpacityProvider(
            tables=tables,
            name=f"{self.name}-exo-k-binned",
            interpolation=self.interpolation,
            cache_log_kcoeff=self.cache_log_kcoeff,
        )

    def prepare(
        self,
        spectral_grid: SpectralGrid,
        pressure_grid: PressureGrid,
        species: Iterable[str],
    ) -> PreparedCorrelatedKOpacity:
        """Prepare correlated-k opacity for one model configuration."""

        species_tuple = tuple(str(item) for item in species)
        if not species_tuple or any(not item for item in species_tuple):
            raise RobertValidationError(
                "species must contain at least one non-empty name"
            )
        missing = tuple(item for item in species_tuple if item not in self.tables)
        if missing:
            raise RobertCoverageError(
                f"missing correlated-k opacity tables for species: {', '.join(missing)}"
            )

        reference = self.tables[species_tuple[0]]
        spectral_indices: dict[str, NDArray[np.int64]] = {}
        for species_name in species_tuple:
            table = self.tables[species_name]
            spectral_indices[species_name] = _exact_spectral_indices(
                spectral_grid, table
            )
            if self.interpolation == "exact":
                _exact_indices(
                    pressure_values_in_unit(
                        pressure_grid.centers, pressure_grid.unit, "bar"
                    ),
                    table.pressure_bar,
                    "pressure",
                )
            elif self.interpolation == "log_pressure_temperature_log_k":
                _validate_within_grid(
                    pressure_values_in_unit(
                        pressure_grid.centers, pressure_grid.unit, "bar"
                    ),
                    table.pressure_bar,
                    "pressure",
                )
            if not np.array_equal(
                reference.g_samples, table.g_samples
            ) or not np.array_equal(
                reference.g_weights,
                table.g_weights,
            ):
                raise RobertCoverageError(
                    "correlated-k tables must share one g grid for this provider"
                )

        return PreparedCorrelatedKOpacity(
            provider_name=self.name,
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            species=species_tuple,
            g_samples=reference.g_samples,
            g_weights=reference.g_weights,
            cache_key=_correlated_k_cache_key(
                self.name,
                self.interpolation,
                species_tuple,
                spectral_grid,
                pressure_grid,
                tuple(self.tables[item] for item in species_tuple),
            ),
            metadata={
                "interpolation": self.interpolation,
                "log_kcoeff_cache": "enabled" if self._log_kcoeff else "disabled",
            },
            spectral_indices=spectral_indices,
        )

    def coverage(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedCorrelatedKOpacity,
    ) -> CorrelatedKCoverageReport:
        """Check whether this provider can evaluate `atmosphere`."""

        reasons: dict[str, str] = {}
        for species_name in prepared.species:
            table = self.tables.get(species_name)
            if table is None:
                reasons[species_name] = "missing correlated-k table"
                continue
            if species_name not in atmosphere.composition:
                reasons[species_name] = "atmosphere is missing species composition"
                continue
            try:
                if _prepared_interpolation(prepared) == "exact":
                    _lookup_indices(
                        atmosphere,
                        prepared.spectral_grid,
                        table,
                        spectral_index=prepared.spectral_indices.get(species_name),
                    )
                else:
                    _validate_interpolation_coverage(
                        atmosphere,
                        prepared.spectral_grid,
                        table,
                        clip=_prepared_interpolation(prepared).endswith("_clip"),
                        check_spectral=species_name not in prepared.spectral_indices,
                    )
            except RobertCoverageError as exc:
                reasons[species_name] = str(exc)
        valid = not reasons
        return CorrelatedKCoverageReport(
            valid=valid,
            message="covered"
            if valid
            else "correlated-k native-grid coverage is incomplete",
            species=prepared.species,
            reasons=reasons,
        )

    def evaluate(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedCorrelatedKOpacity,
    ) -> EvaluatedCorrelatedKOpacity:
        """Evaluate species k-coefficients on `atmosphere`."""

        report = self.coverage(atmosphere, prepared)
        if not report.valid:
            detail = "; ".join(
                f"{species}: {reason}" for species, reason in report.reasons.items()
            )
            raise RobertCoverageError(f"{report.message}: {detail}")

        values = np.empty(
            (
                len(prepared.species),
                atmosphere.n_layers,
                prepared.spectral_grid.size,
                prepared.g_weights.size,
            ),
            dtype=float,
        )
        interpolation = _prepared_interpolation(prepared)
        for species_index, species_name in enumerate(prepared.species):
            table = self.tables[species_name]
            if interpolation == "exact":
                pressure_index, temperature_index, spectral_index = _lookup_indices(
                    atmosphere,
                    prepared.spectral_grid,
                    table,
                    spectral_index=prepared.spectral_indices.get(species_name),
                )
                values[species_index] = table.kcoeff[
                    pressure_index[:, None],
                    temperature_index[:, None],
                    spectral_index[None, :],
                    :,
                ]
            else:
                values[species_index] = _interpolate_pressure_temperature_log_k(
                    atmosphere,
                    prepared.spectral_grid,
                    table,
                    clip=interpolation.endswith("_clip"),
                    log_kcoeff=self._log_kcoeff.get(species_name),
                    spectral_index=prepared.spectral_indices.get(species_name),
                )

        return EvaluatedCorrelatedKOpacity(
            prepared=prepared,
            kcoeff=values,
            metadata={"interpolation": interpolation},
        )


def _lookup_indices(
    atmosphere: AtmosphereState,
    spectral_grid: SpectralGrid,
    table: CorrelatedKTable,
    *,
    spectral_index: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    pressure = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "bar",
    )
    pressure_index = _exact_indices(pressure, table.pressure_bar, "pressure")
    temperature_index = _exact_indices(
        atmosphere.temperature, table.temperature_K, "temperature"
    )
    if spectral_index is None:
        spectral_index = _exact_spectral_indices(spectral_grid, table)
    return pressure_index, temperature_index, spectral_index


def _decode_hdf_scalar(dataset: object | None) -> str:
    if dataset is None:
        return ""
    values = np.asarray(dataset)
    if values.size == 0:
        return ""
    value = values.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _path_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_quantiles(
    values: NDArray[np.float64],
    sample_weights: NDArray[np.float64],
    quantiles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return quantile x pressure x temperature weighted empirical values."""

    order = np.argsort(values, axis=-1)
    sorted_values = np.take_along_axis(values, order, axis=-1)
    weights = np.broadcast_to(sample_weights, values.shape)
    sorted_weights = np.take_along_axis(weights, order, axis=-1)
    positions = np.cumsum(sorted_weights, axis=-1) - 0.5 * sorted_weights
    positions /= np.sum(sorted_weights, axis=-1, keepdims=True)
    flattened_values = sorted_values.reshape(-1, sorted_values.shape[-1])
    flattened_positions = positions.reshape(-1, positions.shape[-1])
    output = np.empty((quantiles.size, flattened_values.shape[0]), dtype=float)
    for index, (position, value) in enumerate(
        zip(flattened_positions, flattened_values, strict=True)
    ):
        output[:, index] = np.interp(
            quantiles,
            position,
            value,
            left=value[0],
            right=value[-1],
        )
    return output.reshape((quantiles.size, *values.shape[:-1]))


def _prepared_interpolation(prepared: PreparedCorrelatedKOpacity) -> str:
    return str(
        prepared.metadata.get("interpolation", prepared.metadata.get("mode", "exact"))
    )


def _validate_interpolation_coverage(
    atmosphere: AtmosphereState,
    spectral_grid: SpectralGrid,
    table: CorrelatedKTable,
    *,
    clip: bool = False,
    check_spectral: bool = True,
) -> None:
    pressure = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "bar",
    )
    if not clip:
        _validate_within_grid(pressure, table.pressure_bar, "pressure")
        _validate_within_grid(
            atmosphere.temperature, table.temperature_K, "temperature"
        )
    if check_spectral:
        _exact_spectral_indices(spectral_grid, table)


def _interpolate_pressure_temperature_log_k(
    atmosphere: AtmosphereState,
    spectral_grid: SpectralGrid,
    table: CorrelatedKTable,
    *,
    k_floor: float = 1.0e-300,
    clip: bool = False,
    log_kcoeff: NDArray[np.float64] | None = None,
    spectral_index: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    _validate_interpolation_coverage(
        atmosphere,
        spectral_grid,
        table,
        clip=clip,
        check_spectral=spectral_index is None,
    )
    pressure = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "bar",
    )
    if spectral_index is None:
        spectral_index = _exact_spectral_indices(spectral_grid, table)
    pressure_lower, pressure_upper, pressure_weight = _bracket_indices(
        np.log10(pressure),
        np.log10(table.pressure_bar),
        "pressure",
        clip=clip,
    )
    temperature_lower, temperature_upper, temperature_weight = _bracket_indices(
        atmosphere.temperature,
        table.temperature_K,
        "temperature",
        clip=clip,
    )
    log_k = (
        np.log(np.maximum(table.kcoeff, k_floor)) if log_kcoeff is None else log_kcoeff
    )
    output = np.empty(
        (atmosphere.n_layers, spectral_index.size, table.g_weights.size), dtype=float
    )
    for layer_index in range(atmosphere.n_layers):
        wp = pressure_weight[layer_index]
        wt = temperature_weight[layer_index]
        p0 = pressure_lower[layer_index]
        p1 = pressure_upper[layer_index]
        t0 = temperature_lower[layer_index]
        t1 = temperature_upper[layer_index]
        values = (
            (1.0 - wp) * (1.0 - wt) * log_k[p0, t0, spectral_index, :]
            + wp * (1.0 - wt) * log_k[p1, t0, spectral_index, :]
            + (1.0 - wp) * wt * log_k[p0, t1, spectral_index, :]
            + wp * wt * log_k[p1, t1, spectral_index, :]
        )
        output[layer_index] = np.exp(values)
    return output


def _exact_spectral_indices(
    spectral_grid: SpectralGrid,
    table: CorrelatedKTable,
) -> NDArray[np.int64]:
    spectral_values = spectral_grid_values_in_unit(spectral_grid, "cm^-1")
    return _exact_indices(spectral_values, table.wavenumber_cm_inverse, "spectral")


def _exact_indices(
    values: ArrayLike,
    grid: ArrayLike,
    label: str,
    *,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-12,
) -> NDArray[np.int64]:
    requested = np.asarray(values, dtype=float)
    native = np.asarray(grid, dtype=float)
    indices: list[int] = []
    for value in requested:
        matches = np.where(np.isclose(native, value, rtol=rtol, atol=atol))[0]
        if matches.size != 1:
            raise RobertCoverageError(
                f"{label} value {value:g} is not on the native correlated-k grid"
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def _validate_within_grid(
    values: ArrayLike,
    grid: ArrayLike,
    label: str,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> None:
    requested = np.asarray(values, dtype=float)
    native = np.asarray(grid, dtype=float)
    lower = min(float(native[0]), float(native[-1]))
    upper = max(float(native[0]), float(native[-1]))
    if np.any(requested < lower - atol - rtol * abs(lower)) or np.any(
        requested > upper + atol + rtol * abs(upper)
    ):
        raise RobertCoverageError(
            f"{label} values are outside the native correlated-k grid"
        )


def _bracket_indices(
    values: ArrayLike,
    grid: ArrayLike,
    label: str,
    *,
    clip: bool = False,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    requested = np.asarray(values, dtype=float)
    native = np.asarray(grid, dtype=float)
    if native.size == 1:
        if not clip:
            _validate_within_grid(requested, native, label)
        zeros = np.zeros(requested.shape, dtype=np.int64)
        weights = np.zeros(requested.shape, dtype=float)
        return zeros, zeros, weights
    increasing_grid = native if native[0] < native[-1] else native[::-1]
    if clip:
        requested = np.clip(requested, increasing_grid[0], increasing_grid[-1])
    else:
        _validate_within_grid(requested, native, label)
    upper = np.searchsorted(increasing_grid, requested, side="left")
    upper = np.clip(upper, 1, increasing_grid.size - 1)
    lower = upper - 1
    exact_upper = np.isclose(
        requested, increasing_grid[upper], rtol=1.0e-12, atol=1.0e-12
    )
    lower = np.where(exact_upper, upper, lower)
    span = increasing_grid[upper] - increasing_grid[lower]
    weight = np.zeros(requested.shape, dtype=float)
    nonzero = span != 0.0
    weight[nonzero] = (requested[nonzero] - increasing_grid[lower][nonzero]) / span[
        nonzero
    ]
    if native[0] > native[-1]:
        lower = native.size - 1 - lower
        upper = native.size - 1 - upper
    return lower.astype(np.int64), upper.astype(np.int64), weight


def _validate_common_g_grid(tables: tuple[CorrelatedKTable, ...]) -> None:
    reference = tables[0]
    for table in tables[1:]:
        if not np.array_equal(
            reference.g_samples, table.g_samples
        ) or not np.array_equal(
            reference.g_weights,
            table.g_weights,
        ):
            raise RobertValidationError("all correlated-k tables must share one g grid")


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _is_strictly_monotonic_or_single(array: NDArray[np.float64]) -> bool:
    if array.size == 1:
        return True
    diff = np.diff(array)
    return bool(np.all(diff > 0.0) or np.all(diff < 0.0))


def _species_name_from_opacity_path(path: Path) -> str:
    species = path.stem.split("_", maxsplit=1)[0].strip()
    if not species:
        raise RobertValidationError(
            f"could not infer a species name from opacity file: {path}"
        )
    return species


def _normalize_kta_resolution(resolution: str | int | None) -> str | None:
    if resolution is None:
        return None
    value = str(resolution).strip().upper()
    if value.startswith("R"):
        value = value[1:]
    if not value.isdigit() or int(value) <= 0:
        raise RobertValidationError(
            "KTA resolution must be a positive integer or 'R<integer>'"
        )
    return f"R{int(value)}"


def _correlated_k_cache_key(
    provider_name: str,
    interpolation: str,
    species: tuple[str, ...],
    spectral_grid: SpectralGrid,
    pressure_grid: PressureGrid,
    tables: tuple[CorrelatedKTable, ...],
) -> str:
    payload = "|".join(
        (
            provider_name,
            interpolation,
            ",".join(species),
            spectral_grid.unit,
            np.array2string(spectral_grid.values, precision=16),
            pressure_grid.unit,
            np.array2string(pressure_grid.centers, precision=16),
            np.array2string(tables[0].g_weights, precision=16),
            ",".join(
                f"{table.species}:{table.metadata.get('checksum_sha256', '')}:"
                f"{table.metadata.get('spectral_preparation', 'native')}:"
                f"{table.metadata.get('source_path', '')}"
                for table in tables
            ),
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
