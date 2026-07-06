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
_PICASO_PRESSURE_ALIASES = ("pressure_bar", "pressure", "pressure_pa", "pressure_mbar")
_PICASO_LAYER_ALIASES = ("lvl", "level", "layer", "nlayer")
_PICASO_WAVELENGTH_ALIASES = ("wavelength_micron", "wavelength_um", "wavelength", "micron")
_PICASO_WAVENUMBER_ALIASES = ("wavenumber", "wno", "wn", "cm^-1", "cm-1")
_PICASO_WAVE_INDEX_ALIASES = ("wv", "w", "wave", "nwave")
_PICASO_SSA_ALIASES = ("w0", "omega0", "single_scattering_albedo", "ssa")
_PICASO_ASYMMETRY_ALIASES = ("g0", "asymmetry_factor", "g", "cosbar")


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


def load_picaso_cloud_optical_properties(
    path: str | Path,
    *,
    name: str | None = None,
    pressure_grid: PressureGrid | None = None,
    spectral_grid: SpectralGrid | None = None,
    pressure_path: str | Path | None = None,
    wave_grid_path: str | Path | None = None,
    pressure_unit: str = "bar",
) -> CloudOpticalProperties:
    """Load a PICASO/Virga-style whitespace-delimited cloud table.

    PICASO `.cld` files contain one row per layer/wavelength cell and must
    provide extinction optical depth (`opd`), single-scattering albedo (`w0`),
    and asymmetry factor (`g0`). Files generated from Virga through PICASO may
    contain physical `pressure` and `wavenumber` columns. The bundled PICASO
    reference files often contain non-physical layer and wavelength-bin indices
    instead; for those files, pass the paired pressure table and
    `wave_EGP.dat` wavelength grid through `pressure_path` and `wave_grid_path`.
    """

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise RobertDataError(f"PICASO cloud file does not exist: {file_path}")
    table = _read_named_numeric_table(file_path, "PICASO cloud file")
    columns = _normalized_columns(table.dtype.names or ())

    pressure_column = _first_optional_normalized(columns, _PICASO_PRESSURE_ALIASES)
    if pressure_column is None:
        layer_column = _first_present_normalized(columns, _PICASO_LAYER_ALIASES, "layer")
        layer_is_pressure = False
    else:
        layer_column = pressure_column
        layer_is_pressure = True

    wavelength_column = _first_optional_normalized(columns, _PICASO_WAVELENGTH_ALIASES)
    wavenumber_column = _first_optional_normalized(columns, _PICASO_WAVENUMBER_ALIASES)
    if wavelength_column is not None:
        spectral_column = wavelength_column
        spectral_kind = "wavelength"
    elif wavenumber_column is not None:
        spectral_column = wavenumber_column
        spectral_kind = "wavenumber"
    else:
        spectral_column = _first_present_normalized(columns, _PICASO_WAVE_INDEX_ALIASES, "wavelength index")
        spectral_kind = "index"

    tau_column = _first_present_normalized(columns, _TAU_ALIASES, "extinction optical depth")
    ssa_column = _first_present_normalized(columns, _PICASO_SSA_ALIASES, "single-scattering albedo")
    asymmetry_column = _first_present_normalized(columns, _PICASO_ASYMMETRY_ALIASES, "asymmetry factor")

    layer_values = np.unique(_table_column(table, layer_column))
    spectral_values = np.unique(_table_column(table, spectral_column))
    tau = np.full((layer_values.size, spectral_values.size), np.nan, dtype=float)
    ssa = np.full_like(tau, np.nan)
    asymmetry = np.full_like(tau, np.nan)
    filled = np.zeros_like(tau, dtype=bool)
    layer_lookup = {float(value): index for index, value in enumerate(layer_values)}
    spectral_lookup = {float(value): index for index, value in enumerate(spectral_values)}

    layer_raw = _table_column(table, layer_column)
    spectral_raw = _table_column(table, spectral_column)
    tau_raw = _table_column(table, tau_column)
    ssa_raw = _table_column(table, ssa_column)
    asymmetry_raw = _table_column(table, asymmetry_column)
    for layer, spectral, tau_value, ssa_value, asymmetry_value in zip(
        layer_raw,
        spectral_raw,
        tau_raw,
        ssa_raw,
        asymmetry_raw,
        strict=True,
    ):
        i_layer = layer_lookup[float(layer)]
        i_spectral = spectral_lookup[float(spectral)]
        if filled[i_layer, i_spectral]:
            raise RobertDataError("PICASO cloud file contains duplicate layer/wavelength cells")
        tau[i_layer, i_spectral] = tau_value
        ssa[i_layer, i_spectral] = ssa_value
        asymmetry[i_layer, i_spectral] = asymmetry_value
        filled[i_layer, i_spectral] = True

    if not np.all(filled):
        raise RobertDataError("PICASO cloud file is missing layer/wavelength cells")

    resolved_pressure_grid = _resolve_picaso_pressure_grid(
        layer_values,
        layer_column=layer_column,
        layer_is_pressure=layer_is_pressure,
        pressure_grid=pressure_grid,
        pressure_path=pressure_path,
        pressure_unit=pressure_unit,
        name=file_path.stem,
    )
    resolved_spectral_grid, tau, ssa, asymmetry = _resolve_picaso_spectral_grid(
        spectral_values,
        spectral_kind=spectral_kind,
        spectral_grid=spectral_grid,
        wave_grid_path=wave_grid_path,
        tau=tau,
        ssa=ssa,
        asymmetry=asymmetry,
        name=file_path.stem,
    )

    metadata = {
        "source_format": "picaso_cld_cloud_optical_properties",
        "source_path": str(file_path),
        "layer_column": layer_column,
        "spectral_column": spectral_column,
        "spectral_kind": spectral_kind,
        "tau_column": tau_column,
        "ssa_column": ssa_column,
        "asymmetry_column": asymmetry_column,
        "layer_coordinate": "pressure" if layer_is_pressure else "index",
    }
    if pressure_path is not None:
        metadata["pressure_path"] = str(Path(pressure_path).expanduser())
    if wave_grid_path is not None:
        metadata["wave_grid_path"] = str(Path(wave_grid_path).expanduser())

    return CloudOpticalProperties(
        name=name or file_path.stem,
        extinction_tau=tau,
        spectral_grid=resolved_spectral_grid,
        pressure_grid=resolved_pressure_grid,
        single_scattering_albedo=ssa,
        asymmetry_factor=asymmetry,
        metadata=metadata,
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


def _first_present_normalized(columns: Mapping[str, str], aliases: tuple[str, ...], label: str) -> str:
    key = _first_optional_normalized(columns, aliases)
    if key is None:
        raise RobertDataError(f"PICASO cloud data are missing {label}; expected one of {aliases}")
    return key


def _first_optional_normalized(columns: Mapping[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        key = columns.get(_normalize_column_name(alias))
        if key is not None:
            return key
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


def _read_named_numeric_table(path: Path, label: str) -> NDArray[np.float64]:
    try:
        table = np.genfromtxt(path, names=True, dtype=float, encoding=None)
    except ValueError as exc:
        raise RobertDataError(f"could not parse {label}: {path}") from exc
    if table.dtype.names is None:
        raise RobertDataError(f"{label} is missing a header row: {path}")
    table = np.atleast_1d(table)
    if table.size == 0:
        raise RobertDataError(f"{label} is empty: {path}")
    return table


def _normalized_columns(names: tuple[str, ...]) -> dict[str, str]:
    columns: dict[str, str] = {}
    for name in names:
        columns.setdefault(_normalize_column_name(name), name)
    return columns


def _normalize_column_name(name: str) -> str:
    return "".join(character for character in name.strip().lower() if character.isalnum())


def _table_column(table: NDArray[np.float64], column: str) -> NDArray[np.float64]:
    values = np.array(table[column], dtype=float, copy=True)
    if not np.all(np.isfinite(values)):
        raise RobertDataError(f"PICASO cloud column {column!r} contains non-finite values")
    return values


def _resolve_picaso_pressure_grid(
    layer_values: NDArray[np.float64],
    *,
    layer_column: str,
    layer_is_pressure: bool,
    pressure_grid: PressureGrid | None,
    pressure_path: str | Path | None,
    pressure_unit: str,
    name: str,
) -> PressureGrid:
    if pressure_grid is not None:
        if pressure_grid.n_layers != layer_values.size:
            raise RobertValidationError("pressure_grid has a different number of layers than the PICASO cloud file")
        return pressure_grid

    if layer_is_pressure:
        resolved_unit = _unit_from_pressure_key(layer_column, pressure_unit)
        return PressureGrid(
            edges=_edges_from_centers(layer_values),
            centers=layer_values,
            unit=resolved_unit,
            name=f"{name} pressure",
        )

    if pressure_path is None:
        raise RobertDataError(
            "PICASO cloud file uses layer indices; pass pressure_grid or pressure_path to attach physical pressures"
        )
    return _pressure_grid_from_picaso_table(
        Path(pressure_path).expanduser(),
        n_layers=layer_values.size,
        pressure_unit=pressure_unit,
        name=f"{name} pressure",
    )


def _pressure_grid_from_picaso_table(
    path: Path,
    *,
    n_layers: int,
    pressure_unit: str,
    name: str,
) -> PressureGrid:
    if not path.exists():
        raise RobertDataError(f"PICASO pressure table does not exist: {path}")
    table = _read_named_numeric_table(path, "PICASO pressure table")
    columns = _normalized_columns(table.dtype.names or ())
    pressure_column = _first_present_normalized(columns, _PICASO_PRESSURE_ALIASES, "pressure")
    pressure_values = _table_column(table, pressure_column)
    resolved_unit = _unit_from_pressure_key(pressure_column, pressure_unit)
    if pressure_values.size == n_layers + 1:
        edges = np.array(pressure_values, dtype=float, copy=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
    elif pressure_values.size == n_layers:
        centers = np.array(pressure_values, dtype=float, copy=True)
        edges = _edges_from_centers(centers)
    else:
        raise RobertDataError(
            "PICASO pressure table must contain either n_layers or n_layers + 1 pressure values"
        )
    return PressureGrid(edges=edges, centers=centers, unit=resolved_unit, name=name)


def _resolve_picaso_spectral_grid(
    spectral_values: NDArray[np.float64],
    *,
    spectral_kind: str,
    spectral_grid: SpectralGrid | None,
    wave_grid_path: str | Path | None,
    tau: NDArray[np.float64],
    ssa: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    name: str,
) -> tuple[SpectralGrid, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if spectral_grid is not None:
        if spectral_grid.size != spectral_values.size:
            raise RobertValidationError("spectral_grid has a different size than the PICASO cloud file")
        return spectral_grid, tau, ssa, asymmetry

    if spectral_kind == "wavelength":
        wavelength = np.array(spectral_values, dtype=float, copy=True)
    elif spectral_kind == "wavenumber":
        if np.any(spectral_values <= 0.0):
            raise RobertDataError("PICASO wavenumber values must be positive")
        wavelength = 10000.0 / spectral_values
    else:
        if wave_grid_path is None:
            raise RobertDataError(
                "PICASO cloud file uses wavelength-bin indices; pass spectral_grid or wave_grid_path"
            )
        wavelength = _wavelength_values_from_picaso_wave_grid(
            Path(wave_grid_path).expanduser(),
            expected_size=spectral_values.size,
        )

    order = np.argsort(wavelength)
    wavelength = np.array(wavelength[order], dtype=float, copy=True)
    return (
        SpectralGrid.from_array(
            wavelength,
            unit="micron",
            role="cloud_optical_properties",
            name=f"{name} wavelength",
        ),
        tau[:, order],
        ssa[:, order],
        asymmetry[:, order],
    )


def _wavelength_values_from_picaso_wave_grid(path: Path, *, expected_size: int) -> NDArray[np.float64]:
    if not path.exists():
        raise RobertDataError(f"PICASO wave grid does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
    columns = {_normalize_column_name(name): index for index, name in enumerate(header)}
    wavelength_column = _first_optional_normalized_index(columns, _PICASO_WAVELENGTH_ALIASES)
    if wavelength_column is not None:
        wavelength = _load_text_column(path, wavelength_column)
    else:
        wavenumber_column = _first_present_normalized_index(
            columns,
            _PICASO_WAVENUMBER_ALIASES,
            "wavelength or wavenumber",
        )
        wavenumber = _load_text_column(path, wavenumber_column)
        if np.any(wavenumber <= 0.0):
            raise RobertDataError("PICASO wave-grid wavenumber values must be positive")
        wavelength = 10000.0 / wavenumber
    if wavelength.size != expected_size:
        raise RobertDataError("PICASO wave grid size does not match the cloud file wavelength count")
    return wavelength


def _first_present_normalized_index(columns: Mapping[str, int], aliases: tuple[str, ...], label: str) -> int:
    index = _first_optional_normalized_index(columns, aliases)
    if index is None:
        raise RobertDataError(f"PICASO wave grid is missing {label}; expected one of {aliases}")
    return index


def _first_optional_normalized_index(columns: Mapping[str, int], aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        index = columns.get(_normalize_column_name(alias))
        if index is not None:
            return index
    return None


def _load_text_column(path: Path, column_index: int) -> NDArray[np.float64]:
    try:
        values = np.loadtxt(path, skiprows=1, usecols=(column_index,), dtype=float)
    except ValueError as exc:
        raise RobertDataError(f"could not parse PICASO wave-grid column {column_index} from {path}") from exc
    values = np.atleast_1d(np.array(values, dtype=float, copy=True))
    if not np.all(np.isfinite(values)):
        raise RobertDataError(f"PICASO wave-grid column {column_index} contains non-finite values")
    return values


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
