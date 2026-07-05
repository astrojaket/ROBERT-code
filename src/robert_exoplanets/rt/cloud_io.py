"""Interchange readers for external cloud optical-property products."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import PressureGrid, RobertDataError, RobertValidationError, SpectralGrid
from robert_exoplanets.opacity import pressure_values_in_unit

from .clouds import CloudOpticalProperties

_PRESSURE_ALIASES = ("pressure_bar", "pressure", "pressure_pa", "pressure_mbar")
_WAVELENGTH_ALIASES = ("wavelength_micron", "wavelength_um", "wavelength")
_TAU_ALIASES = ("extinction_tau", "tau_ext", "tau", "opd")
_SSA_ALIASES = ("single_scattering_albedo", "omega0", "ssa")
_ASYMMETRY_ALIASES = ("asymmetry_factor", "g", "cosbar")


def load_cloud_optical_properties_npz(
    path: str | Path,
    *,
    name: str | None = None,
    pressure_unit: str = "bar",
    wavelength_unit: str = "micron",
) -> CloudOpticalProperties:
    """Load dense cloud optical properties from a `.npz` interchange file.

    Required arrays are pressure centers, wavelength, and extinction optical
    depth. Single-scattering albedo and asymmetry factor are optional and
    default to zero. Common PICASO/Virga-style aliases such as `tau_ext`,
    `omega0`, and `g` are accepted.
    """

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise RobertDataError(f"cloud optical-property file does not exist: {file_path}")
    with np.load(file_path, allow_pickle=False) as archive:
        keys = tuple(str(key) for key in archive.files)
        pressure_key = _first_present(keys, _PRESSURE_ALIASES, "pressure")
        wavelength_key = _first_present(keys, _WAVELENGTH_ALIASES, "wavelength")
        tau_key = _first_present(keys, _TAU_ALIASES, "extinction_tau")
        pressure_values = np.array(archive[pressure_key], dtype=float, copy=True)
        wavelength_values = np.array(archive[wavelength_key], dtype=float, copy=True)
        tau = np.array(archive[tau_key], dtype=float, copy=True)
        ssa = _optional_array_from_archive(archive, keys, _SSA_ALIASES, default=0.0)
        asymmetry = _optional_array_from_archive(archive, keys, _ASYMMETRY_ALIASES, default=0.0)
        pressure_edges = _optional_pressure_edges_from_archive(archive, keys)

    resolved_pressure_unit = _unit_from_pressure_key(pressure_key, pressure_unit)
    if pressure_edges is None:
        pressure_edges = _edges_from_centers(pressure_values)
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_values,
        unit=resolved_pressure_unit,
        name=f"{file_path.stem} pressure",
    )
    spectral_grid = SpectralGrid.from_array(
        wavelength_values,
        unit=wavelength_unit,
        role="cloud_optical_properties",
        name=f"{file_path.stem} wavelength",
    )
    return CloudOpticalProperties(
        name=name or file_path.stem,
        extinction_tau=tau,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=ssa,
        asymmetry_factor=asymmetry,
        metadata={
            "source_format": "npz_cloud_optical_properties",
            "source_path": str(file_path),
            "pressure_key": pressure_key,
            "wavelength_key": wavelength_key,
            "tau_key": tau_key,
        },
    )


def load_cloud_optical_properties_csv(
    path: str | Path,
    *,
    name: str | None = None,
    pressure_unit: str = "bar",
    wavelength_unit: str = "micron",
) -> CloudOpticalProperties:
    """Load long-table cloud optical properties from a CSV file.

    The CSV must contain one row per pressure/wavelength cell. Accepted column
    aliases include `pressure_bar`, `wavelength_micron`, `tau_ext`, `omega0`,
    and `g`.
    """

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise RobertDataError(f"cloud optical-property CSV does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RobertDataError("cloud optical-property CSV is missing a header row")
        pressure_column = _first_present(reader.fieldnames, _PRESSURE_ALIASES, "pressure")
        wavelength_column = _first_present(reader.fieldnames, _WAVELENGTH_ALIASES, "wavelength")
        tau_column = _first_present(reader.fieldnames, _TAU_ALIASES, "extinction_tau")
        ssa_column = _first_optional(reader.fieldnames, _SSA_ALIASES)
        asymmetry_column = _first_optional(reader.fieldnames, _ASYMMETRY_ALIASES)
        rows = tuple(reader)

    if not rows:
        raise RobertDataError("cloud optical-property CSV is empty")
    pressure_values = np.array([_float_from_row(row, pressure_column) for row in rows], dtype=float)
    wavelength_values = np.array([_float_from_row(row, wavelength_column) for row in rows], dtype=float)
    pressures = np.unique(pressure_values)
    wavelengths = np.unique(wavelength_values)
    tau = np.full((pressures.size, wavelengths.size), np.nan, dtype=float)
    ssa = np.full_like(tau, 0.0)
    asymmetry = np.full_like(tau, 0.0)
    pressure_lookup = {float(value): index for index, value in enumerate(pressures)}
    wavelength_lookup = {float(value): index for index, value in enumerate(wavelengths)}
    filled = np.zeros_like(tau, dtype=bool)

    for row in rows:
        pressure = _float_from_row(row, pressure_column)
        wavelength = _float_from_row(row, wavelength_column)
        i_pressure = pressure_lookup[pressure]
        i_wavelength = wavelength_lookup[wavelength]
        if filled[i_pressure, i_wavelength]:
            raise RobertDataError("cloud optical-property CSV contains duplicate pressure/wavelength cells")
        tau[i_pressure, i_wavelength] = _float_from_row(row, tau_column)
        if ssa_column is not None:
            ssa[i_pressure, i_wavelength] = _float_from_row(row, ssa_column)
        if asymmetry_column is not None:
            asymmetry[i_pressure, i_wavelength] = _float_from_row(row, asymmetry_column)
        filled[i_pressure, i_wavelength] = True

    if not np.all(filled):
        raise RobertDataError("cloud optical-property CSV is missing pressure/wavelength cells")
    resolved_pressure_unit = _unit_from_pressure_key(pressure_column, pressure_unit)
    pressure_grid = PressureGrid(
        edges=_edges_from_centers(pressures),
        centers=pressures,
        unit=resolved_pressure_unit,
        name=f"{file_path.stem} pressure",
    )
    spectral_grid = SpectralGrid.from_array(
        wavelengths,
        unit=wavelength_unit,
        role="cloud_optical_properties",
        name=f"{file_path.stem} wavelength",
    )
    return CloudOpticalProperties(
        name=name or file_path.stem,
        extinction_tau=tau,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=ssa,
        asymmetry_factor=asymmetry,
        metadata={
            "source_format": "csv_cloud_optical_properties",
            "source_path": str(file_path),
            "pressure_column": pressure_column,
            "wavelength_column": wavelength_column,
            "tau_column": tau_column,
        },
    )


def write_cloud_optical_properties_npz(
    cloud: CloudOpticalProperties,
    path: str | Path,
    *,
    pressure_unit: str = "bar",
) -> Path:
    """Write cloud optical properties to the compact ROBERT/PICASO bridge format."""

    file_path = Path(path).expanduser()
    pressure = pressure_values_in_unit(
        cloud.pressure_grid.centers,
        cloud.pressure_grid.unit,
        pressure_unit,
    )
    pressure_edges = pressure_values_in_unit(
        cloud.pressure_grid.edges,
        cloud.pressure_grid.unit,
        pressure_unit,
    )
    pressure_key = f"pressure_{_normalized_pressure_unit_for_key(pressure_unit)}"
    edge_key = f"pressure_edges_{_normalized_pressure_unit_for_key(pressure_unit)}"
    np.savez(
        file_path,
        **{
            pressure_key: pressure,
            edge_key: pressure_edges,
            "wavelength_micron": cloud.spectral_grid.values,
            "extinction_tau": cloud.extinction_tau,
            "single_scattering_albedo": cloud.single_scattering_albedo,
            "asymmetry_factor": cloud.asymmetry_factor,
        },
    )
    return file_path


def _first_present(keys: tuple[str, ...] | list[str], aliases: tuple[str, ...], label: str) -> str:
    for alias in aliases:
        if alias in keys:
            return alias
    raise RobertDataError(f"cloud optical-property data are missing {label}; expected one of {aliases}")


def _first_optional(keys: tuple[str, ...] | list[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in keys:
            return alias
    return None


def _optional_array_from_archive(
    archive: Mapping[str, NDArray[np.float64]],
    keys: tuple[str, ...],
    aliases: tuple[str, ...],
    *,
    default: float,
) -> NDArray[np.float64] | float:
    key = _first_optional(keys, aliases)
    if key is None:
        return default
    return np.array(archive[key], dtype=float, copy=True)


def _optional_pressure_edges_from_archive(
    archive: Mapping[str, NDArray[np.float64]],
    keys: tuple[str, ...],
) -> NDArray[np.float64] | None:
    key = _first_optional(keys, ("pressure_edges_bar", "pressure_edges", "pressure_edges_pa", "pressure_edges_mbar"))
    if key is None:
        return None
    return np.array(archive[key], dtype=float, copy=True)


def _unit_from_pressure_key(key: str, fallback: str) -> str:
    normalized = key.strip().lower()
    if normalized.endswith("_pa"):
        return "pa"
    if normalized.endswith("_mbar"):
        return "mbar"
    if normalized.endswith("_bar"):
        return "bar"
    return fallback


def _edges_from_centers(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.array(centers, dtype=float, copy=True)
    if values.ndim != 1 or values.size == 0:
        raise RobertValidationError("pressure centers must be a non-empty one-dimensional array")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise RobertValidationError("pressure centers must be finite and positive")
    if values.size > 1 and not (np.all(np.diff(values) > 0.0) or np.all(np.diff(values) < 0.0)):
        raise RobertValidationError("pressure centers must be strictly monotonic")
    log_centers = np.log(values)
    if values.size == 1:
        delta = 1.0
        edges = np.exp(np.array([log_centers[0] - 0.5 * delta, log_centers[0] + 0.5 * delta]))
    else:
        inner_edges = 0.5 * (log_centers[:-1] + log_centers[1:])
        first_edge = log_centers[0] - (inner_edges[0] - log_centers[0])
        last_edge = log_centers[-1] + (log_centers[-1] - inner_edges[-1])
        edges = np.exp(np.concatenate(([first_edge], inner_edges, [last_edge])))
    edges.setflags(write=False)
    return edges


def _float_from_row(row: Mapping[str, str], column: str) -> float:
    try:
        return float(row[column])
    except (TypeError, ValueError, KeyError) as exc:
        raise RobertDataError(f"invalid numeric value in cloud optical-property column {column!r}") from exc


def _normalized_pressure_unit_for_key(unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized in {"bar", "bars"}:
        return "bar"
    if normalized in {"mbar", "millibar", "millibars"}:
        return "mbar"
    if normalized in {"pa", "pascal", "pascals"}:
        return "pa"
    raise RobertValidationError(f"unsupported pressure unit: {unit}")
