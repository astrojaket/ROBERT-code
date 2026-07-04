"""Correlated-k opacity evaluation on native grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, RobertCoverageError, RobertValidationError, SpectralGrid

from .archive import load_robert_npy_directory, load_robert_npz_archive
from .kta import NemesisKTable, read_kta
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
            raise RobertValidationError("wavenumber_cm_inverse must be strictly monotonic")
        if g_samples.shape != g_weights.shape:
            raise RobertValidationError("g_samples and g_weights must have the same shape")
        if np.any(g_weights < 0.0):
            raise RobertValidationError("g_weights must be non-negative")
        g_weight_sum = float(np.sum(g_weights))
        if g_weight_sum <= 0.0 or not np.isfinite(g_weight_sum):
            raise RobertValidationError("g_weights must have a finite positive sum")
        if not np.isclose(g_weight_sum, 1.0, rtol=1.0e-6, atol=1.0e-8):
            raise RobertValidationError("g_weights must sum to one")
        normalized_g_weights = np.array(g_weights / g_weight_sum, dtype=float, copy=True)
        normalized_g_weights.setflags(write=False)

        wavelength = None
        if self.wavelength_micron is not None:
            wavelength = _readonly_1d(self.wavelength_micron, "wavelength_micron")
            if wavelength.shape != wavenumber.shape:
                raise RobertValidationError("wavelength_micron must match wavenumber grid shape")
            if np.any(wavelength <= 0.0):
                raise RobertValidationError("wavelength_micron values must be positive")
        else:
            wavelength = 10000.0 / wavenumber
            wavelength.setflags(write=False)

        kcoeff = np.asarray(self.kcoeff, dtype=float)
        expected_shape = (pressure.size, temperature.size, wavenumber.size, g_samples.size)
        if kcoeff.shape != expected_shape:
            raise RobertValidationError("kcoeff shape must be pressure x temperature x wavelength x g")
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
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_nemesis(cls, species: str, table: NemesisKTable) -> "CorrelatedKTable":
        """Build a correlated-k table from a loaded NEMESIS `.kta` table."""

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
                "source_format": "nemesis_kta",
                "source_path": table.header.path,
                "checksum_sha256": "" if table.header.checksum_sha256 is None else table.header.checksum_sha256,
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
            raise RobertValidationError("only single-product correlated-k archives are currently supported")
        product = archive.database.products[0]
        species_name = species or product.species[0]
        arrays = archive.arrays
        required = ("kcoeff", "pressure_bar", "temperature_K", "wavenumber_cm-1", "g_samples", "g_weights")
        missing = tuple(name for name in required if name not in arrays)
        if missing:
            raise RobertValidationError(f"ROBERT opacity archive is missing arrays: {', '.join(missing)}")
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

    def __post_init__(self) -> None:
        if not self.provider_name:
            raise RobertValidationError("provider_name must not be empty")
        species = tuple(str(item) for item in self.species)
        if not species or any(not item for item in species):
            raise RobertValidationError("prepared correlated-k opacity species must be non-empty")
        g_samples = _readonly_1d(self.g_samples, "g_samples")
        g_weights = _readonly_1d(self.g_weights, "g_weights")
        if g_samples.shape != g_weights.shape:
            raise RobertValidationError("g_samples and g_weights must have the same shape")
        if not self.cache_key:
            raise RobertValidationError("cache_key must not be empty")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "g_samples", g_samples)
        object.__setattr__(self, "g_weights", g_weights)
        object.__setattr__(self, "metadata", dict(self.metadata))


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
            raise RobertValidationError("kcoeff shape must be species x layers x wavelength x g")
        if not np.all(np.isfinite(kcoeff)) or np.any(kcoeff < 0.0):
            raise RobertValidationError("evaluated kcoeff values must be finite and non-negative")
        if not self.unit:
            raise RobertValidationError("evaluated correlated-k unit must not be empty")
        kcoeff.setflags(write=False)
        object.__setattr__(self, "kcoeff", kcoeff)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class CorrelatedKCoverageReport:
    """Coverage result for correlated-k native-grid evaluation."""

    valid: bool
    message: str
    species: tuple[str, ...]
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "reasons", dict(self.reasons))


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

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("correlated-k provider name must not be empty")
        if self.interpolation not in {"exact", "log_pressure_temperature_log_k"}:
            raise RobertValidationError(
                "interpolation must be 'exact' or 'log_pressure_temperature_log_k'"
            )
        tables = {str(species): table for species, table in self.tables.items()}
        if not tables:
            raise RobertValidationError("correlated-k provider must contain at least one table")
        for species, table in tables.items():
            if species != table.species:
                raise RobertValidationError("correlated-k table mapping key must match table species")
        _validate_common_g_grid(tuple(tables.values()))
        object.__setattr__(self, "tables", tables)

    @classmethod
    def from_kta_paths(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "correlated-k-native",
        interpolation: str = "exact",
        nonfinite_policy: str = "raise",
        nonfinite_fill_value: float = 1.0e-300,
    ) -> "CorrelatedKOpacityProvider":
        """Build a provider from species-to-`.kta` file paths."""

        tables = {
            str(species): CorrelatedKTable.from_nemesis(
                str(species),
                read_kta(
                    path,
                    nonfinite_policy=nonfinite_policy,
                    nonfinite_fill_value=nonfinite_fill_value,
                ),
            )
            for species, path in paths.items()
        }
        return cls(tables=tables, name=name, interpolation=interpolation)

    @classmethod
    def from_robert_archives(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "correlated-k-native",
        interpolation: str = "exact",
    ) -> "CorrelatedKOpacityProvider":
        """Build a provider from species-to-ROBERT archive paths."""

        tables = {
            str(species): CorrelatedKTable.from_robert_archive(path, species=str(species))
            for species, path in paths.items()
        }
        return cls(tables=tables, name=name, interpolation=interpolation)

    @property
    def species(self) -> tuple[str, ...]:
        """Available opacity species."""

        return tuple(self.tables)

    def prepare(
        self,
        spectral_grid: SpectralGrid,
        pressure_grid: PressureGrid,
        species: Iterable[str],
    ) -> PreparedCorrelatedKOpacity:
        """Prepare correlated-k opacity for one model configuration."""

        species_tuple = tuple(str(item) for item in species)
        if not species_tuple or any(not item for item in species_tuple):
            raise RobertValidationError("species must contain at least one non-empty name")
        missing = tuple(item for item in species_tuple if item not in self.tables)
        if missing:
            raise RobertCoverageError(f"missing correlated-k opacity tables for species: {', '.join(missing)}")

        reference = self.tables[species_tuple[0]]
        for species_name in species_tuple:
            table = self.tables[species_name]
            _exact_spectral_indices(spectral_grid, table)
            if self.interpolation == "exact":
                _exact_indices(
                    pressure_values_in_unit(pressure_grid.centers, pressure_grid.unit, "bar"),
                    table.pressure_bar,
                    "pressure",
                )
            else:
                _validate_within_grid(
                    pressure_values_in_unit(pressure_grid.centers, pressure_grid.unit, "bar"),
                    table.pressure_bar,
                    "pressure",
                )
            if not np.array_equal(reference.g_samples, table.g_samples) or not np.array_equal(
                reference.g_weights,
                table.g_weights,
            ):
                raise RobertCoverageError("correlated-k tables must share one g grid for this provider")

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
                reference,
            ),
            metadata={"interpolation": self.interpolation},
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
                    _lookup_indices(atmosphere, prepared.spectral_grid, table)
                else:
                    _validate_interpolation_coverage(atmosphere, prepared.spectral_grid, table)
            except RobertCoverageError as exc:
                reasons[species_name] = str(exc)
        valid = not reasons
        return CorrelatedKCoverageReport(
            valid=valid,
            message="covered" if valid else "correlated-k native-grid coverage is incomplete",
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
            detail = "; ".join(f"{species}: {reason}" for species, reason in report.reasons.items())
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
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    pressure = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "bar",
    )
    pressure_index = _exact_indices(pressure, table.pressure_bar, "pressure")
    temperature_index = _exact_indices(atmosphere.temperature, table.temperature_K, "temperature")
    spectral_index = _exact_spectral_indices(spectral_grid, table)
    return pressure_index, temperature_index, spectral_index


def _prepared_interpolation(prepared: PreparedCorrelatedKOpacity) -> str:
    return str(prepared.metadata.get("interpolation", prepared.metadata.get("mode", "exact")))


def _validate_interpolation_coverage(
    atmosphere: AtmosphereState,
    spectral_grid: SpectralGrid,
    table: CorrelatedKTable,
) -> None:
    pressure = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "bar",
    )
    _validate_within_grid(pressure, table.pressure_bar, "pressure")
    _validate_within_grid(atmosphere.temperature, table.temperature_K, "temperature")
    _exact_spectral_indices(spectral_grid, table)


def _interpolate_pressure_temperature_log_k(
    atmosphere: AtmosphereState,
    spectral_grid: SpectralGrid,
    table: CorrelatedKTable,
    *,
    k_floor: float = 1.0e-300,
) -> NDArray[np.float64]:
    _validate_interpolation_coverage(atmosphere, spectral_grid, table)
    pressure = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "bar",
    )
    spectral_index = _exact_spectral_indices(spectral_grid, table)
    pressure_lower, pressure_upper, pressure_weight = _bracket_indices(
        np.log10(pressure),
        np.log10(table.pressure_bar),
        "pressure",
    )
    temperature_lower, temperature_upper, temperature_weight = _bracket_indices(
        atmosphere.temperature,
        table.temperature_K,
        "temperature",
    )
    log_k = np.log(np.maximum(table.kcoeff, k_floor))
    output = np.empty((atmosphere.n_layers, spectral_index.size, table.g_weights.size), dtype=float)
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
            raise RobertCoverageError(f"{label} value {value:g} is not on the native correlated-k grid")
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
        raise RobertCoverageError(f"{label} values are outside the native correlated-k grid")


def _bracket_indices(
    values: ArrayLike,
    grid: ArrayLike,
    label: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    requested = np.asarray(values, dtype=float)
    native = np.asarray(grid, dtype=float)
    if native.size == 1:
        _validate_within_grid(requested, native, label)
        zeros = np.zeros(requested.shape, dtype=np.int64)
        weights = np.zeros(requested.shape, dtype=float)
        return zeros, zeros, weights
    _validate_within_grid(requested, native, label)
    increasing_grid = native if native[0] < native[-1] else native[::-1]
    upper = np.searchsorted(increasing_grid, requested, side="left")
    upper = np.clip(upper, 1, increasing_grid.size - 1)
    lower = upper - 1
    exact_upper = np.isclose(requested, increasing_grid[upper], rtol=1.0e-12, atol=1.0e-12)
    lower = np.where(exact_upper, upper, lower)
    span = increasing_grid[upper] - increasing_grid[lower]
    weight = np.zeros(requested.shape, dtype=float)
    nonzero = span != 0.0
    weight[nonzero] = (requested[nonzero] - increasing_grid[lower][nonzero]) / span[nonzero]
    if native[0] > native[-1]:
        lower = native.size - 1 - lower
        upper = native.size - 1 - upper
    return lower.astype(np.int64), upper.astype(np.int64), weight


def _validate_common_g_grid(tables: tuple[CorrelatedKTable, ...]) -> None:
    reference = tables[0]
    for table in tables[1:]:
        if not np.array_equal(reference.g_samples, table.g_samples) or not np.array_equal(
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


def _correlated_k_cache_key(
    provider_name: str,
    interpolation: str,
    species: tuple[str, ...],
    spectral_grid: SpectralGrid,
    pressure_grid: PressureGrid,
    reference: CorrelatedKTable,
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
            np.array2string(reference.g_weights, precision=16),
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
