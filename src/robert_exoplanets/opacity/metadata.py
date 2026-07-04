"""Opacity data metadata and coverage contracts.

These objects describe opacity inputs without loading large opacity arrays.
They are intentionally separate from radiative transfer so file formats,
coverage checks, and future compression can evolve behind a stable boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import PressureGrid, RobertCoverageError, RobertValidationError, SpectralGrid


class OpacityMode(str, Enum):
    """Numerical mode represented by an opacity data product."""

    CORRELATED_K = "correlated_k"
    OPACITY_SAMPLING = "opacity_sampling"
    LINE_BY_LINE = "line_by_line"
    CIA = "cia"


class OpacityDataSource(str, Enum):
    """External or ROBERT-native source of opacity data."""

    EXOMOL = "exomol"
    EXOMOL_OP = "exomol_op"
    HITRAN = "hitran"
    HITEMP = "hitemp"
    NEMESIS = "nemesis"
    ROBERT_ARCHIVE = "robert_archive"
    UNKNOWN = "unknown"


class OpacityStorageFormat(str, Enum):
    """On-disk format used by an opacity data product."""

    EXOMOL_LINE_LIST = "exomol_line_list"
    EXOMOL_CROSS_SECTION = "exomol_cross_section"
    EXOMOL_KTABLE = "exomol_ktable"
    HITRAN_PAR = "hitran_par"
    HITRAN_CIA = "hitran_cia"
    NEMESIS_KTA = "nemesis_kta"
    ROBERT_NPY_DIRECTORY = "robert_npy_directory"
    ROBERT_NPZ = "robert_npz"
    UNKNOWN = "unknown"


def _enum_value(value: str | Enum, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise RobertValidationError(f"{name} must be one of: {allowed}") from exc


def _species_tuple(species: Iterable[str], name: str = "species") -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in species)
    if not values:
        raise RobertValidationError(f"{name} must contain at least one species")
    if any(not item for item in values):
        raise RobertValidationError(f"{name} names must not be empty")
    return values


def _readonly_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SpectralCoverage:
    """Closed spectral interval covered by an opacity product."""

    min_value: float
    max_value: float
    unit: str = "cm^-1"
    n_points: int | None = None
    resolving_power: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        min_value = float(self.min_value)
        max_value = float(self.max_value)
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            raise RobertValidationError("spectral coverage bounds must be finite")
        if min_value >= max_value:
            raise RobertValidationError("spectral coverage min_value must be smaller than max_value")
        if not self.unit:
            raise RobertValidationError("spectral coverage unit must not be empty")
        if self.n_points is not None and self.n_points < 1:
            raise RobertValidationError("spectral coverage n_points must be positive")
        if self.resolving_power is not None and self.resolving_power <= 0.0:
            raise RobertValidationError("spectral coverage resolving_power must be positive")
        object.__setattr__(self, "min_value", min_value)
        object.__setattr__(self, "max_value", max_value)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def contains(self, spectral_grid: SpectralGrid) -> bool:
        """Return whether `spectral_grid` lies inside this interval."""

        values = spectral_grid_values_in_unit(spectral_grid, self.unit)
        return bool(np.min(values) >= self.min_value and np.max(values) <= self.max_value)

    def to_mapping(self) -> dict[str, object]:
        """Return JSON-serializable coverage metadata."""

        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "n_points": self.n_points,
            "resolving_power": self.resolving_power,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SpectralCoverage":
        """Build coverage metadata from a JSON-like mapping."""

        return cls(
            min_value=float(values["min_value"]),
            max_value=float(values["max_value"]),
            unit=str(values.get("unit", "cm^-1")),
            n_points=_optional_int(values.get("n_points")),
            resolving_power=_optional_float(values.get("resolving_power")),
            metadata=_string_mapping(values.get("metadata", {}), "spectral_coverage.metadata"),
        )


@dataclass(frozen=True)
class GridCoverage:
    """Pressure and temperature coverage for tabulated opacity data."""

    pressure_min: float | None = None
    pressure_max: float | None = None
    pressure_unit: str = "bar"
    temperature_min: float | None = None
    temperature_max: float | None = None
    temperature_unit: str = "K"
    n_pressure: int | None = None
    n_temperature: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pressure_min = _optional_float(self.pressure_min)
        pressure_max = _optional_float(self.pressure_max)
        temperature_min = _optional_float(self.temperature_min)
        temperature_max = _optional_float(self.temperature_max)
        if (pressure_min is None) != (pressure_max is None):
            raise RobertValidationError("pressure coverage must provide both min and max")
        if pressure_min is not None and pressure_min >= pressure_max:
            raise RobertValidationError("pressure coverage min must be smaller than max")
        if (temperature_min is None) != (temperature_max is None):
            raise RobertValidationError("temperature coverage must provide both min and max")
        if temperature_min is not None and temperature_min >= temperature_max:
            raise RobertValidationError("temperature coverage min must be smaller than max")
        if not self.pressure_unit:
            raise RobertValidationError("pressure coverage unit must not be empty")
        if not self.temperature_unit:
            raise RobertValidationError("temperature coverage unit must not be empty")
        if self.n_pressure is not None and self.n_pressure < 1:
            raise RobertValidationError("n_pressure must be positive")
        if self.n_temperature is not None and self.n_temperature < 1:
            raise RobertValidationError("n_temperature must be positive")
        object.__setattr__(self, "pressure_min", pressure_min)
        object.__setattr__(self, "pressure_max", pressure_max)
        object.__setattr__(self, "temperature_min", temperature_min)
        object.__setattr__(self, "temperature_max", temperature_max)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def has_pressure(self) -> bool:
        """Whether pressure coverage is known."""

        return self.pressure_min is not None and self.pressure_max is not None

    @property
    def has_temperature(self) -> bool:
        """Whether temperature coverage is known."""

        return self.temperature_min is not None and self.temperature_max is not None

    def contains_pressure_grid(self, pressure_grid: PressureGrid) -> bool:
        """Return whether `pressure_grid` lies inside the pressure coverage."""

        if not self.has_pressure:
            return False
        pressure = pressure_values_in_unit(pressure_grid.centers, pressure_grid.unit, self.pressure_unit)
        return bool(np.min(pressure) >= self.pressure_min and np.max(pressure) <= self.pressure_max)

    def contains_temperature(self, temperature: ArrayLike) -> bool:
        """Return whether all temperatures lie inside the temperature coverage."""

        if not self.has_temperature:
            return False
        values = _readonly_float_array(temperature, "temperature")
        if _temperature_unit(self.temperature_unit) != "K":
            raise RobertValidationError("only Kelvin temperature coverage is currently supported")
        return bool(np.min(values) >= self.temperature_min and np.max(values) <= self.temperature_max)

    def to_mapping(self) -> dict[str, object]:
        """Return JSON-serializable grid coverage metadata."""

        return {
            "pressure_min": self.pressure_min,
            "pressure_max": self.pressure_max,
            "pressure_unit": self.pressure_unit,
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "temperature_unit": self.temperature_unit,
            "n_pressure": self.n_pressure,
            "n_temperature": self.n_temperature,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "GridCoverage":
        """Build grid coverage metadata from a JSON-like mapping."""

        return cls(
            pressure_min=_optional_float(values.get("pressure_min")),
            pressure_max=_optional_float(values.get("pressure_max")),
            pressure_unit=str(values.get("pressure_unit", "bar")),
            temperature_min=_optional_float(values.get("temperature_min")),
            temperature_max=_optional_float(values.get("temperature_max")),
            temperature_unit=str(values.get("temperature_unit", "K")),
            n_pressure=_optional_int(values.get("n_pressure")),
            n_temperature=_optional_int(values.get("n_temperature")),
            metadata=_string_mapping(values.get("metadata", {}), "grid_coverage.metadata"),
        )


@dataclass(frozen=True)
class OpacityDataProduct:
    """Metadata for one opacity data product."""

    species: tuple[str, ...]
    mode: OpacityMode | str
    source: OpacityDataSource | str
    storage_format: OpacityStorageFormat | str
    path: str | None = None
    spectral_coverage: SpectralCoverage | None = None
    grid_coverage: GridCoverage | None = None
    cia_pair: tuple[str, str] | None = None
    g_ordinates: int | None = None
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None
    compression: str | None = None
    native_shape: tuple[int, ...] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = _species_tuple(self.species)
        mode = _enum_value(self.mode, OpacityMode, "mode")
        source = _enum_value(self.source, OpacityDataSource, "source")
        storage_format = _enum_value(self.storage_format, OpacityStorageFormat, "storage_format")
        cia_pair = None
        if self.cia_pair is not None:
            cia_pair = _species_tuple(self.cia_pair, "cia_pair")
            if len(cia_pair) != 2:
                raise RobertValidationError("cia_pair must contain exactly two species")
        if mode == OpacityMode.CIA and cia_pair is None and len(species) == 2:
            cia_pair = (species[0], species[1])
        if self.g_ordinates is not None and self.g_ordinates < 1:
            raise RobertValidationError("g_ordinates must be positive")
        if self.file_size_bytes is not None and self.file_size_bytes < 0:
            raise RobertValidationError("file_size_bytes must be non-negative")
        native_shape = None
        if self.native_shape is not None:
            native_shape = tuple(int(item) for item in self.native_shape)
            if any(item < 1 for item in native_shape):
                raise RobertValidationError("native_shape values must be positive")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "storage_format", storage_format)
        object.__setattr__(self, "cia_pair", cia_pair)
        object.__setattr__(self, "native_shape", native_shape)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def product_id(self) -> str:
        """Stable human-readable product identifier."""

        parts = [self.source.value, self.storage_format.value, self.mode.value, "+".join(self.species)]
        if self.path:
            parts.append(Path(self.path).name)
        return ":".join(parts)

    def coverage_reasons(
        self,
        *,
        spectral_grid: SpectralGrid | None = None,
        pressure_grid: PressureGrid | None = None,
        temperature: ArrayLike | None = None,
    ) -> tuple[str, ...]:
        """Return coverage failure reasons for this product."""

        reasons: list[str] = []
        if spectral_grid is not None:
            if self.spectral_coverage is None:
                reasons.append("spectral coverage is unknown")
            elif not self.spectral_coverage.contains(spectral_grid):
                reasons.append("spectral grid is outside opacity coverage")
        if pressure_grid is not None:
            if self.grid_coverage is None or not self.grid_coverage.has_pressure:
                reasons.append("pressure coverage is unknown")
            elif not self.grid_coverage.contains_pressure_grid(pressure_grid):
                reasons.append("pressure grid is outside opacity coverage")
        if temperature is not None:
            if self.grid_coverage is None or not self.grid_coverage.has_temperature:
                reasons.append("temperature coverage is unknown")
            elif not self.grid_coverage.contains_temperature(temperature):
                reasons.append("temperature profile is outside opacity coverage")
        return tuple(reasons)

    def to_mapping(self) -> dict[str, object]:
        """Return JSON-serializable product metadata."""

        return {
            "species": list(self.species),
            "mode": self.mode.value,
            "source": self.source.value,
            "storage_format": self.storage_format.value,
            "path": self.path,
            "spectral_coverage": None
            if self.spectral_coverage is None
            else self.spectral_coverage.to_mapping(),
            "grid_coverage": None if self.grid_coverage is None else self.grid_coverage.to_mapping(),
            "cia_pair": None if self.cia_pair is None else list(self.cia_pair),
            "g_ordinates": self.g_ordinates,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "compression": self.compression,
            "native_shape": None if self.native_shape is None else list(self.native_shape),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "OpacityDataProduct":
        """Build product metadata from a JSON-like mapping."""

        spectral = values.get("spectral_coverage")
        grid = values.get("grid_coverage")
        cia_pair = values.get("cia_pair")
        native_shape = values.get("native_shape")
        return cls(
            species=tuple(str(item) for item in _sequence(values["species"], "species")),
            mode=str(values["mode"]),
            source=str(values["source"]),
            storage_format=str(values["storage_format"]),
            path=None if values.get("path") is None else str(values["path"]),
            spectral_coverage=None
            if spectral is None
            else SpectralCoverage.from_mapping(_mapping(spectral, "spectral_coverage")),
            grid_coverage=None if grid is None else GridCoverage.from_mapping(_mapping(grid, "grid_coverage")),
            cia_pair=None if cia_pair is None else tuple(str(item) for item in _sequence(cia_pair, "cia_pair")),
            g_ordinates=_optional_int(values.get("g_ordinates")),
            file_size_bytes=_optional_int(values.get("file_size_bytes")),
            checksum_sha256=None
            if values.get("checksum_sha256") is None
            else str(values["checksum_sha256"]),
            compression=None if values.get("compression") is None else str(values["compression"]),
            native_shape=None
            if native_shape is None
            else tuple(int(item) for item in _sequence(native_shape, "native_shape")),
            metadata=_string_mapping(values.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class OpacityCoverageReport:
    """Coverage validation result for an opacity database query."""

    valid: bool
    message: str
    requested_species: tuple[str, ...]
    covered_species: tuple[str, ...]
    missing_species: tuple[str, ...]
    reasons: Mapping[str, str] = field(default_factory=dict)
    product_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_species", tuple(self.requested_species))
        object.__setattr__(self, "covered_species", tuple(self.covered_species))
        object.__setattr__(self, "missing_species", tuple(self.missing_species))
        object.__setattr__(self, "reasons", dict(self.reasons))
        object.__setattr__(self, "product_ids", tuple(self.product_ids))


@dataclass(frozen=True)
class OpacityDatabase:
    """Collection of inspectable opacity data products."""

    products: tuple[OpacityDataProduct, ...]
    name: str = "opacity-database"
    root: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.products:
            raise RobertValidationError("opacity database must contain at least one product")
        if not self.name:
            raise RobertValidationError("opacity database name must not be empty")
        object.__setattr__(self, "products", tuple(self.products))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def species(self) -> tuple[str, ...]:
        """Species present in the database, preserving first-seen order."""

        values: list[str] = []
        for product in self.products:
            for species in product.species:
                if species not in values:
                    values.append(species)
        return tuple(values)

    def products_for_species(
        self,
        species: str,
        *,
        mode: OpacityMode | str | None = None,
    ) -> tuple[OpacityDataProduct, ...]:
        """Return products that can contribute opacity for one species."""

        mode_value = None if mode is None else _enum_value(mode, OpacityMode, "mode")
        return tuple(
            product
            for product in self.products
            if species in product.species and (mode_value is None or product.mode == mode_value)
        )

    def coverage(
        self,
        *,
        species: Iterable[str],
        mode: OpacityMode | str | None = None,
        spectral_grid: SpectralGrid | None = None,
        pressure_grid: PressureGrid | None = None,
        temperature: ArrayLike | None = None,
    ) -> OpacityCoverageReport:
        """Check whether the database covers one model request."""

        requested = _species_tuple(species)
        covered: list[str] = []
        missing: list[str] = []
        reasons: dict[str, str] = {}
        product_ids: list[str] = []
        for species_name in requested:
            products = self.products_for_species(species_name, mode=mode)
            if not products:
                missing.append(species_name)
                reasons[species_name] = "no opacity product for requested species and mode"
                continue
            valid_product = None
            product_messages: list[str] = []
            for product in products:
                product_reasons = product.coverage_reasons(
                    spectral_grid=spectral_grid,
                    pressure_grid=pressure_grid,
                    temperature=temperature,
                )
                if not product_reasons:
                    valid_product = product
                    break
                product_messages.append(f"{product.product_id}: {', '.join(product_reasons)}")
            if valid_product is None:
                missing.append(species_name)
                reasons[species_name] = "; ".join(product_messages)
            else:
                covered.append(species_name)
                product_ids.append(valid_product.product_id)

        valid = not missing
        message = "covered" if valid else "opacity coverage is incomplete"
        return OpacityCoverageReport(
            valid=valid,
            message=message,
            requested_species=requested,
            covered_species=tuple(covered),
            missing_species=tuple(missing),
            reasons=reasons,
            product_ids=tuple(product_ids),
        )

    def validate_coverage(self, **kwargs: object) -> OpacityCoverageReport:
        """Return coverage or raise `RobertCoverageError`."""

        report = self.coverage(**kwargs)
        if not report.valid:
            detail = "; ".join(f"{key}: {value}" for key, value in report.reasons.items())
            raise RobertCoverageError(f"{report.message}: {detail}")
        return report

    def to_mapping(self) -> dict[str, object]:
        """Return JSON-serializable database metadata."""

        return {
            "name": self.name,
            "root": self.root,
            "products": [product.to_mapping() for product in self.products],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "OpacityDatabase":
        """Build database metadata from a JSON-like mapping."""

        products = tuple(
            OpacityDataProduct.from_mapping(_mapping(item, "product"))
            for item in _sequence(values["products"], "products")
        )
        return cls(
            products=products,
            name=str(values.get("name", "opacity-database")),
            root=None if values.get("root") is None else str(values["root"]),
            metadata=_string_mapping(values.get("metadata", {}), "metadata"),
        )


def spectral_grid_values_in_unit(spectral_grid: SpectralGrid, unit: str) -> NDArray[np.float64]:
    """Return spectral grid values converted into `unit` when supported."""

    source = _spectral_unit(spectral_grid.unit)
    target = _spectral_unit(unit)
    values = np.array(spectral_grid.values, dtype=float, copy=True)
    if source == target:
        values.setflags(write=False)
        return values
    if source == "micron" and target == "cm^-1":
        if np.any(values <= 0.0):
            raise RobertValidationError("wavelength values must be positive for wavenumber conversion")
        converted = 10000.0 / values
        converted.setflags(write=False)
        return converted
    if source == "cm^-1" and target == "micron":
        if np.any(values <= 0.0):
            raise RobertValidationError("wavenumber values must be positive for wavelength conversion")
        converted = 10000.0 / values
        converted.setflags(write=False)
        return converted
    raise RobertValidationError(f"cannot convert spectral grid from {spectral_grid.unit!r} to {unit!r}")


def pressure_values_in_unit(values: ArrayLike, source_unit: str, target_unit: str) -> NDArray[np.float64]:
    """Return pressure values converted between common opacity-grid units."""

    source = _pressure_unit(source_unit)
    target = _pressure_unit(target_unit)
    pressure = _readonly_float_array(values, "pressure")
    pressure_bar = pressure * _pressure_to_bar_factor(source)
    converted = pressure_bar / _pressure_to_bar_factor(target)
    converted.setflags(write=False)
    return converted


def _spectral_unit(unit: str) -> str:
    normalized = unit.strip().lower().replace("μ", "u")
    if normalized in {"micron", "microns", "um", "wavelength_micron"}:
        return "micron"
    if normalized in {"cm^-1", "cm-1", "1/cm", "wavenumber", "wavenumber_cm-1"}:
        return "cm^-1"
    raise RobertValidationError(f"unsupported spectral unit: {unit}")


def _pressure_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized in {"bar", "bars"}:
        return "bar"
    if normalized in {"mbar", "millibar", "millibars"}:
        return "mbar"
    if normalized in {"pa", "pascal", "pascals"}:
        return "pa"
    if normalized in {"atm", "atmosphere", "atmospheres"}:
        return "atm"
    raise RobertValidationError(f"unsupported pressure unit: {unit}")


def _pressure_to_bar_factor(unit: str) -> float:
    if unit == "bar":
        return 1.0
    if unit == "mbar":
        return 1.0e-3
    if unit == "pa":
        return 1.0e-5
    if unit == "atm":
        return 1.01325
    raise RobertValidationError(f"unsupported pressure unit: {unit}")


def _temperature_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized in {"k", "kelvin"}:
        return "K"
    raise RobertValidationError(f"unsupported temperature unit: {unit}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not np.isfinite(converted):
        raise RobertValidationError("optional float values must be finite when provided")
    return converted


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    converted = int(value)
    return converted


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RobertValidationError(f"{name} must be a mapping")
    return value


def _string_mapping(value: object, name: str) -> dict[str, str]:
    mapping = _mapping(value, name)
    return {str(key): str(item) for key, item in mapping.items()}


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RobertValidationError(f"{name} must be a sequence")
    return value
