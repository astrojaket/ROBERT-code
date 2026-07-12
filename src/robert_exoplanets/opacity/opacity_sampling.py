"""Beta opacity-sampling cross sections on a shared native wavelength grid.

This module reads the pressure-temperature cross-section grids published by
ExoMolOP in TauREx HDF5 format.  It deliberately performs *opacity sampling*:
the same physical wavenumber samples are retained for every species and their
optical depths can therefore be added directly.  No sorting into a cumulative
opacity distribution and no random-overlap approximation is involved.

Opacity sampling is a functional beta backend. Correlated-k remains ROBERT's
validated default for retrievals while target-specific sampling convergence is
still being developed.
"""

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

from .inspectors import file_sha256
from .metadata import pressure_values_in_unit, spectral_grid_values_in_unit

try:  # pragma: no cover - availability depends on the environment.
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range

_NUMBA_AVAILABLE = njit is not None


@dataclass(frozen=True)
class OpacitySamplingTable:
    """Metadata and axes for one file-backed molecular cross-section grid."""

    species: str
    path: str | Path
    pressure_bar: ArrayLike
    temperature_K: ArrayLike
    wavenumber_cm_inverse: ArrayLike
    unit: str = "cm^2/molecule"
    dataset: str = "xsecarr"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.species:
            raise RobertValidationError("opacity-sampling table species must not be empty")
        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise RobertValidationError(f"opacity-sampling table does not exist: {path}")
        pressure = _readonly_positive_axis(self.pressure_bar, "pressure_bar")
        temperature = _readonly_positive_axis(self.temperature_K, "temperature_K")
        wavenumber = _readonly_positive_axis(
            self.wavenumber_cm_inverse, "wavenumber_cm_inverse"
        )
        if not self.unit:
            raise RobertValidationError("opacity-sampling unit must not be empty")
        if not self.dataset:
            raise RobertValidationError("opacity-sampling dataset must not be empty")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "pressure_bar", pressure)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "wavenumber_cm_inverse", wavenumber)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @classmethod
    def from_exomol_hdf(
        cls,
        path: str | Path,
        *,
        species: str,
        checksum: bool = True,
    ) -> "OpacitySamplingTable":
        """Inspect an ExoMolOP/TauREx cross-section HDF5 file.

        The bulk cross-section array remains file-backed until ``prepare`` so
        constructing a table never allocates hundreds of megabytes.
        """

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RobertValidationError(
                "loading ExoMol cross sections requires h5py"
            ) from exc

        source = Path(path).expanduser().resolve()
        required = ("p", "t", "bin_edges", "xsecarr")
        try:
            with h5py.File(source, "r") as handle:
                missing = tuple(name for name in required if name not in handle)
                if missing:
                    raise RobertValidationError(
                        "ExoMol cross-section file is missing datasets: "
                        + ", ".join(missing)
                    )
                pressure = np.asarray(handle["p"], dtype=float)
                temperature = np.asarray(handle["t"], dtype=float)
                wavenumber = np.asarray(handle["bin_edges"], dtype=float)
                xsec = handle["xsecarr"]
                expected = (pressure.size, temperature.size, wavenumber.size)
                if xsec.shape != expected:
                    raise RobertValidationError(
                        "ExoMol xsecarr shape must be pressure x temperature x wavenumber"
                    )
                unit = str(xsec.attrs.get("units", "")).strip()
                if not unit:
                    raise RobertValidationError(
                        "ExoMol xsecarr dataset must declare its units"
                    )
                pressure_unit = str(handle["p"].attrs.get("units", "")).lower()
                temperature_unit = str(handle["t"].attrs.get("units", "")).lower()
                spectral_unit = str(handle["bin_edges"].attrs.get("units", "")).lower()
                if pressure_unit != "bar" or temperature_unit not in {"k", "kelvin"}:
                    raise RobertValidationError(
                        "ExoMol cross sections must use bar and kelvin axes"
                    )
                if "wavenumber" not in spectral_unit:
                    raise RobertValidationError(
                        "ExoMol bin_edges must declare a wavenumber axis"
                    )
                doi = _decode_hdf_text(handle.get("DOI"))
                line_list = _decode_hdf_text(handle.get("key_iso_ll"))
                molecule = _decode_hdf_text(handle.get("mol_name"))
        except OSError as exc:
            raise RobertValidationError(
                f"could not read ExoMol cross-section file: {source}"
            ) from exc

        return cls(
            species=species,
            path=source,
            pressure_bar=pressure,
            temperature_K=temperature,
            wavenumber_cm_inverse=wavenumber,
            unit=unit,
            metadata={
                "source_format": "exomol_taurex_hdf5",
                "source_path": str(source),
                "checksum_sha256": file_sha256(source) if checksum else "",
                "doi": doi,
                "line_list": line_list,
                "molecule": molecule,
            },
        )


@dataclass(frozen=True)
class PreparedOpacitySampling:
    """Cached, run-specific opacity-sampling cross sections."""

    provider_name: str
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    species: tuple[str, ...]
    g_samples: ArrayLike
    g_weights: ArrayLike
    cache_key: str
    log_cross_sections: Mapping[str, NDArray[np.float64]] = field(repr=False)
    metadata: Mapping[str, str] = field(default_factory=dict)
    stacked_log_cross_sections: NDArray[np.float64] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.provider_name or not self.cache_key:
            raise RobertValidationError("prepared opacity-sampling identifiers must not be empty")
        species = tuple(str(item) for item in self.species)
        if not species or any(not item for item in species):
            raise RobertValidationError("prepared opacity-sampling species must be non-empty")
        g_samples = _readonly_1d(self.g_samples, "g_samples")
        g_weights = _readonly_1d(self.g_weights, "g_weights")
        if g_samples.shape != (1,) or g_weights.shape != (1,) or g_weights[0] != 1.0:
            raise RobertValidationError(
                "opacity sampling must use one unit-weight compatibility ordinate"
            )
        arrays = []
        for species_name in species:
            if species_name not in self.log_cross_sections:
                raise RobertValidationError(
                    f"prepared opacity sampling is missing cached species {species_name}"
                )
            values = np.asarray(self.log_cross_sections[species_name], dtype=float)
            if values.ndim != 3 or values.shape[-1] != self.spectral_grid.size:
                raise RobertValidationError(
                    "cached cross sections must be pressure x temperature x spectral"
                )
            if not np.all(np.isfinite(values)):
                raise RobertValidationError("cached log cross sections must be finite")
            arrays.append(values)
        stacked = np.ascontiguousarray(np.stack(arrays, axis=0))
        stacked.setflags(write=False)
        cached = {name: stacked[index] for index, name in enumerate(species)}
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "g_samples", g_samples)
        object.__setattr__(self, "g_weights", g_weights)
        object.__setattr__(self, "log_cross_sections", immutable_mapping(cached))
        object.__setattr__(self, "stacked_log_cross_sections", stacked)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class EvaluatedOpacitySampling:
    """Sampled cross sections evaluated on an atmospheric profile."""

    prepared: PreparedOpacitySampling
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
                "sampled cross sections must be species x layers x wavelength x 1"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RobertValidationError(
                "evaluated sampled cross sections must be finite and non-negative"
            )
        if not self.unit:
            raise RobertValidationError("evaluated opacity-sampling unit must not be empty")
        values.setflags(write=False)
        object.__setattr__(self, "kcoeff", values)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class EvaluatedOpacitySamplingMixture:
    """VMR-weighted sampled cross section without species-resolved arrays."""

    prepared: PreparedOpacitySampling
    cross_section: ArrayLike
    unit: str = "cm^2/molecule"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.cross_section, dtype=float)
        expected = (
            self.prepared.pressure_grid.n_layers,
            self.prepared.spectral_grid.size,
        )
        if values.shape != expected:
            raise RobertValidationError(
                "mixture cross section must be layers x wavelength"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RobertValidationError(
                "mixture cross section must be finite and non-negative"
            )
        values.setflags(write=False)
        object.__setattr__(self, "cross_section", values)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class OpacitySamplingCoverageReport:
    """Coverage result for opacity-sampling evaluation."""

    valid: bool
    message: str
    species: tuple[str, ...]
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "reasons", immutable_mapping(self.reasons))


@dataclass(frozen=True)
class OpacitySamplingProvider:
    """File-backed ExoMol cross-section opacity-sampling provider.

    ``prepare`` selects a shared set of physical wavelengths and caches only
    those samples.  ``evaluate`` then performs bilinear interpolation in
    log-pressure, temperature, and log cross section without file I/O.
    """

    tables: Mapping[str, OpacitySamplingTable]
    name: str = "exomol-opacity-sampling"
    interpolation: str = "log_pressure_temperature_log_xsec"
    cross_section_floor: float = 1.0e-300

    def __post_init__(self) -> None:
        tables = {str(key): value for key, value in self.tables.items()}
        if not tables or any(key != value.species for key, value in tables.items()):
            raise RobertValidationError(
                "opacity-sampling table keys must match at least one table species"
            )
        if not self.name:
            raise RobertValidationError("opacity-sampling provider name must not be empty")
        if self.interpolation not in {
            "log_pressure_temperature_log_xsec",
            "log_pressure_temperature_log_xsec_clip",
        }:
            raise RobertValidationError(
                "opacity-sampling interpolation must be log-pressure/temperature/log-xsec"
            )
        floor = float(self.cross_section_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise RobertValidationError("cross_section_floor must be finite and positive")
        if len({table.unit for table in tables.values()}) != 1:
            raise RobertValidationError(
                "opacity-sampling tables in one provider must share a unit"
            )
        object.__setattr__(self, "tables", immutable_mapping(tables))
        object.__setattr__(self, "cross_section_floor", floor)

    @classmethod
    def from_exomol_paths(
        cls,
        paths: Mapping[str, str | Path],
        *,
        name: str = "exomol-opacity-sampling",
        interpolation: str = "log_pressure_temperature_log_xsec",
        checksum: bool = True,
    ) -> "OpacitySamplingProvider":
        """Build a provider from species-to-ExoMolOP HDF5 paths."""

        return cls(
            tables={
                str(species): OpacitySamplingTable.from_exomol_hdf(
                    path, species=str(species), checksum=checksum
                )
                for species, path in paths.items()
            },
            name=name,
            interpolation=interpolation,
        )

    @property
    def species(self) -> tuple[str, ...]:
        """Available opacity species."""

        return tuple(self.tables)

    def native_spectral_grid(
        self,
        *,
        sampling: int = 1,
        wavelength_bounds_micron: tuple[float, float] | None = None,
        name: str | None = None,
    ) -> SpectralGrid:
        """Return a shared native grid, retaining every ``sampling``-th point."""

        if not isinstance(sampling, int) or isinstance(sampling, bool) or sampling < 1:
            raise RobertValidationError("opacity sampling stride must be a positive integer")
        reference = next(iter(self.tables.values())).wavenumber_cm_inverse
        for table in self.tables.values():
            if not np.array_equal(reference, table.wavenumber_cm_inverse):
                raise RobertCoverageError(
                    "opacity-sampling tables must share an identical wavenumber grid"
                )
        selected = np.arange(reference.size, dtype=np.int64)
        if wavelength_bounds_micron is not None:
            lower, upper = (float(value) for value in wavelength_bounds_micron)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or lower >= upper:
                raise RobertValidationError(
                    "wavelength_bounds_micron must be finite, positive, and increasing"
                )
            wavelength = 10000.0 / reference
            selected = selected[(wavelength >= lower) & (wavelength <= upper)]
        selected = selected[::sampling]
        if selected.size == 0:
            raise RobertCoverageError("requested opacity-sampling wavelength range is empty")
        wavelength = (10000.0 / reference[selected])[::-1]
        return SpectralGrid.from_array(
            wavelength,
            unit="micron",
            role="opacity",
            name=name or f"{self.name}-stride-{sampling}",
        )

    def prepare(
        self,
        spectral_grid: SpectralGrid,
        pressure_grid: PressureGrid,
        species: Iterable[str],
    ) -> PreparedOpacitySampling:
        """Select and cache cross sections for one model configuration."""

        species_tuple = tuple(str(item) for item in species)
        if not species_tuple or any(not item for item in species_tuple):
            raise RobertValidationError("opacity-sampling species must be non-empty")
        missing = tuple(item for item in species_tuple if item not in self.tables)
        if missing:
            raise RobertCoverageError(
                f"missing opacity-sampling tables for species: {', '.join(missing)}"
            )
        pressure = pressure_values_in_unit(
            pressure_grid.centers, pressure_grid.unit, "bar"
        )
        cached: dict[str, NDArray[np.float64]] = {}
        indices_by_species: dict[str, NDArray[np.int64]] = {}
        for species_name in species_tuple:
            table = self.tables[species_name]
            _validate_axis_coverage(
                pressure,
                table.pressure_bar,
                "pressure",
                clip=self.interpolation.endswith("_clip"),
            )
            indices = _exact_spectral_indices(spectral_grid, table)
            indices_by_species[species_name] = indices
            cached[species_name] = _load_selected_log_cross_sections(
                table, indices, floor=self.cross_section_floor
            )
        return PreparedOpacitySampling(
            provider_name=self.name,
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            species=species_tuple,
            g_samples=np.array([0.5]),
            g_weights=np.array([1.0]),
            cache_key=_cache_key(
                self.name,
                self.interpolation,
                species_tuple,
                spectral_grid,
                pressure_grid,
                tuple(self.tables[item] for item in species_tuple),
                tuple(indices_by_species[item] for item in species_tuple),
            ),
            log_cross_sections=cached,
            metadata={
                "opacity_mode": "opacity_sampling",
                "maturity": "beta",
                "interpolation": self.interpolation,
                "sample_count": str(spectral_grid.size),
                "species_combination": "direct_optical_depth_sum",
            },
        )

    def coverage(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedOpacitySampling,
    ) -> OpacitySamplingCoverageReport:
        """Check pressure, temperature, and composition coverage."""

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
                reasons[species_name] = "missing opacity-sampling table"
                continue
            if species_name not in atmosphere.composition:
                reasons[species_name] = "atmosphere is missing species composition"
                continue
            try:
                _validate_axis_coverage(pressure, table.pressure_bar, "pressure", clip=clip)
                _validate_axis_coverage(
                    atmosphere.temperature,
                    table.temperature_K,
                    "temperature",
                    clip=clip,
                )
            except RobertCoverageError as exc:
                reasons[species_name] = str(exc)
        valid = not reasons
        return OpacitySamplingCoverageReport(
            valid=valid,
            message="covered" if valid else "opacity-sampling coverage is incomplete",
            species=prepared.species,
            reasons=reasons,
        )

    def evaluate(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedOpacitySampling,
    ) -> EvaluatedOpacitySampling:
        """Interpolate cached sampled cross sections onto ``atmosphere``."""

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
            values[species_index, :, :, 0] = np.exp(log_values)
        return EvaluatedOpacitySampling(
            prepared=prepared,
            kcoeff=values,
            unit=next(iter(self.tables.values())).unit,
            metadata={
                "opacity_mode": "opacity_sampling",
                "maturity": "beta",
                "interpolation": self.interpolation,
            },
        )

    def evaluate_mixture(
        self,
        atmosphere: AtmosphereState,
        prepared: PreparedOpacitySampling,
    ) -> EvaluatedOpacitySamplingMixture:
        """Evaluate the VMR-weighted mixture in one compiled direct-sum kernel."""

        report = self.coverage(atmosphere, prepared)
        if not report.valid:
            detail = "; ".join(
                f"{species}: {reason}" for species, reason in report.reasons.items()
            )
            raise RobertCoverageError(f"{report.message}: {detail}")
        reference = self.tables[prepared.species[0]]
        for species_name in prepared.species[1:]:
            table = self.tables[species_name]
            if not np.array_equal(table.pressure_bar, reference.pressure_bar) or not np.array_equal(
                table.temperature_K, reference.temperature_K
            ):
                raise RobertValidationError(
                    "fused opacity-sampling mixtures require shared pressure-temperature grids"
                )
        pressure = pressure_values_in_unit(
            atmosphere.pressure_grid.centers,
            atmosphere.pressure_grid.unit,
            "bar",
        )
        log_pressure_grid = np.log(reference.pressure_bar)
        requested_log_pressure = np.log(pressure)
        temperature = np.asarray(atmosphere.temperature, dtype=float)
        if self.interpolation.endswith("_clip"):
            requested_log_pressure = np.clip(
                requested_log_pressure,
                log_pressure_grid[0],
                log_pressure_grid[-1],
            )
            temperature = np.clip(
                temperature,
                reference.temperature_K[0],
                reference.temperature_K[-1],
            )
        p0, p1, wp = _brackets(requested_log_pressure, log_pressure_grid)
        t0, t1, wt = _brackets(temperature, reference.temperature_K)
        vmr = np.ascontiguousarray(
            np.stack(
                [atmosphere.composition[name] for name in prepared.species],
                axis=0,
            )
        )
        if _NUMBA_AVAILABLE:
            mixture = _mixture_cross_section_kernel(
                prepared.stacked_log_cross_sections,
                vmr,
                p0,
                p1,
                wp,
                t0,
                t1,
                wt,
            )
            backend = "numba"
        else:
            mixture = _mixture_cross_section_numpy(
                prepared.stacked_log_cross_sections,
                vmr,
                p0,
                p1,
                wp,
                t0,
                t1,
                wt,
            )
            backend = "numpy"
        return EvaluatedOpacitySamplingMixture(
            prepared=prepared,
            cross_section=mixture,
            unit=reference.unit,
            metadata={
                "opacity_mode": "opacity_sampling",
                "maturity": "beta",
                "species_combination": "fused_vmr_weighted_direct_sum",
                "backend": backend,
            },
        )


def _load_selected_log_cross_sections(
    table: OpacitySamplingTable,
    indices: NDArray[np.int64],
    *,
    floor: float,
) -> NDArray[np.float64]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RobertValidationError("loading ExoMol cross sections requires h5py") from exc
    order = np.argsort(indices)
    sorted_indices = indices[order]
    inverse = np.argsort(order)
    try:
        with h5py.File(table.path, "r") as handle:
            selected = np.asarray(handle[table.dataset][:, :, sorted_indices], dtype=float)
    except OSError as exc:
        raise RobertValidationError(
            f"could not read ExoMol cross sections from {table.path}"
        ) from exc
    selected = selected[:, :, inverse]
    if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
        raise RobertValidationError("ExoMol cross sections must be finite and non-negative")
    result = np.log(np.maximum(selected, floor))
    result.setflags(write=False)
    return result


def _interpolate_log_cross_sections(
    log_values: NDArray[np.float64],
    pressure: NDArray[np.float64],
    temperature: NDArray[np.float64],
    pressure_grid: NDArray[np.float64],
    temperature_grid: NDArray[np.float64],
    *,
    clip: bool,
) -> NDArray[np.float64]:
    log_pressure_grid = np.log(pressure_grid)
    requested_log_pressure = np.log(pressure)
    if clip:
        requested_log_pressure = np.clip(
            requested_log_pressure, log_pressure_grid[0], log_pressure_grid[-1]
        )
        temperature = np.clip(temperature, temperature_grid[0], temperature_grid[-1])
    p0, p1, wp = _brackets(requested_log_pressure, log_pressure_grid)
    t0, t1, wt = _brackets(temperature, temperature_grid)
    wp = wp[:, None]
    wt = wt[:, None]
    lower = (1.0 - wt) * log_values[p0, t0, :] + wt * log_values[p0, t1, :]
    upper = (1.0 - wt) * log_values[p1, t0, :] + wt * log_values[p1, t1, :]
    return (1.0 - wp) * lower + wp * upper


def _mixture_cross_section_numpy(
    log_cross_sections,
    vmr,
    p0,
    p1,
    wp,
    t0,
    t1,
    wt,
):
    mixture = np.zeros((vmr.shape[1], log_cross_sections.shape[-1]), dtype=float)
    for species_index in range(vmr.shape[0]):
        values = log_cross_sections[species_index]
        lower = (
            (1.0 - wt[:, None]) * values[p0, t0, :]
            + wt[:, None] * values[p0, t1, :]
        )
        upper = (
            (1.0 - wt[:, None]) * values[p1, t0, :]
            + wt[:, None] * values[p1, t1, :]
        )
        interpolated = (1.0 - wp[:, None]) * lower + wp[:, None] * upper
        mixture += vmr[species_index, :, None] * np.exp(interpolated)
    return mixture


if _NUMBA_AVAILABLE:

    @njit(parallel=True)
    def _mixture_cross_section_kernel(
        log_cross_sections,
        vmr,
        p0,
        p1,
        wp,
        t0,
        t1,
        wt,
    ):
        n_species = log_cross_sections.shape[0]
        n_layers = vmr.shape[1]
        n_spectral = log_cross_sections.shape[3]
        mixture = np.zeros((n_layers, n_spectral), dtype=np.float64)
        for flat_index in prange(n_layers * n_spectral):
            layer = flat_index // n_spectral
            spectral = flat_index - layer * n_spectral
            total = 0.0
            for species in range(n_species):
                values = log_cross_sections[species]
                lower = (
                    (1.0 - wt[layer]) * values[p0[layer], t0[layer], spectral]
                    + wt[layer] * values[p0[layer], t1[layer], spectral]
                )
                upper = (
                    (1.0 - wt[layer]) * values[p1[layer], t0[layer], spectral]
                    + wt[layer] * values[p1[layer], t1[layer], spectral]
                )
                log_value = (1.0 - wp[layer]) * lower + wp[layer] * upper
                total += vmr[species, layer] * np.exp(log_value)
            mixture[layer, spectral] = total
        return mixture

else:

    def _mixture_cross_section_kernel(*args):
        raise RobertValidationError("compiled opacity sampling requires numba")


def _brackets(
    values: NDArray[np.float64], grid: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    upper = np.searchsorted(grid, values, side="right")
    upper = np.clip(upper, 1, grid.size - 1).astype(np.int64)
    lower = upper - 1
    weight = (values - grid[lower]) / (grid[upper] - grid[lower])
    exact_low = values == grid[0]
    exact_high = values == grid[-1]
    lower[exact_low] = upper[exact_low] = 0
    lower[exact_high] = upper[exact_high] = grid.size - 1
    weight[exact_low | exact_high] = 0.0
    return lower, upper, weight


def _exact_spectral_indices(
    spectral_grid: SpectralGrid, table: OpacitySamplingTable
) -> NDArray[np.int64]:
    requested = spectral_grid_values_in_unit(spectral_grid, "cm^-1")
    source = table.wavenumber_cm_inverse
    insertion = np.searchsorted(source, requested)
    clipped = np.clip(insertion, 0, source.size - 1)
    previous = np.clip(insertion - 1, 0, source.size - 1)
    choose_previous = np.abs(source[previous] - requested) < np.abs(
        source[clipped] - requested
    )
    indices = np.where(choose_previous, previous, clipped).astype(np.int64)
    if not np.allclose(source[indices], requested, rtol=2.0e-10, atol=1.0e-10):
        raise RobertCoverageError(
            "opacity-sampling spectral grid must be an exact subset of the source grid"
        )
    indices.setflags(write=False)
    return indices


def _validate_axis_coverage(
    requested: ArrayLike,
    grid: NDArray[np.float64],
    name: str,
    *,
    clip: bool,
) -> None:
    values = np.asarray(requested, dtype=float)
    if not clip and (np.any(values < grid[0]) or np.any(values > grid[-1])):
        raise RobertCoverageError(
            f"{name} values are outside opacity-sampling table coverage"
        )


def _cache_key(
    name: str,
    interpolation: str,
    species: tuple[str, ...],
    spectral_grid: SpectralGrid,
    pressure_grid: PressureGrid,
    tables: tuple[OpacitySamplingTable, ...],
    indices: tuple[NDArray[np.int64], ...],
) -> str:
    digest = sha256()
    for text in (name, interpolation, *species, spectral_grid.unit, pressure_grid.unit):
        digest.update(text.encode("utf-8"))
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
    return digest.hexdigest()


def _decode_hdf_text(dataset: object | None) -> str:
    if dataset is None:
        return ""
    values = np.asarray(dataset)
    if values.size == 0:
        return ""
    value = values.reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _readonly_positive_axis(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = _readonly_1d(values, name)
    if array.size < 2 or np.any(array <= 0.0) or not np.all(np.diff(array) > 0.0):
        raise RobertValidationError(f"{name} must be positive and strictly increasing")
    return array


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must be one-dimensional and finite")
    array.setflags(write=False)
    return array
