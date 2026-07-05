"""Continuum and scattering-extinction optical-depth helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import PressureGrid, RobertValidationError, SpectralGrid
from robert_exoplanets.opacity import pressure_values_in_unit, spectral_grid_values_in_unit

from .optical_depth import GasOpticalDepth

BOLTZMANN_CONSTANT_J_K = 1.380649e-23
AMAGAT_MOLECULES_CM3 = 2.68675e19

DEFAULT_CIA_PAIR_ORDER = (
    "H2-H2_equilibrium",
    "H2-He_equilibrium",
    "H2-H2_normal",
    "H2-He_normal",
    "H2-N2",
    "N2-CH4",
    "N2-N2",
    "CH4-CH4",
    "H2-CH4",
)


@dataclass(frozen=True)
class LayerOpticalDepth:
    """Named optical-depth contribution on layer and spectral axes."""

    name: str
    tau: ArrayLike
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    kind: str = "extinction"
    unit: str = "dimensionless"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("layer optical-depth name must not be empty")
        if not self.kind:
            raise RobertValidationError("layer optical-depth kind must not be empty")
        if not self.unit:
            raise RobertValidationError("layer optical-depth unit must not be empty")
        tau = _readonly_array(
            self.tau,
            "tau",
            (self.pressure_grid.n_layers, self.spectral_grid.size),
        )
        if np.any(tau < 0.0):
            raise RobertValidationError("layer optical depth must be non-negative")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def cumulative_tau_from_top(self) -> NDArray[np.float64]:
        """Return cumulative optical depth from the top of the atmosphere."""

        order = _top_to_bottom_order(self.pressure_grid)
        cumulative = np.cumsum(self.tau[order], axis=0)
        return _restore_layer_order(cumulative, order)


@dataclass(frozen=True)
class CiaTable:
    """CIA coefficient table in ROBERT's pair, temperature, wavenumber order."""

    wavenumber_cm_inverse: ArrayLike
    temperature_K: ArrayLike
    k_cia: ArrayLike
    pair_order: tuple[str, ...] = DEFAULT_CIA_PAIR_ORDER
    unit: str = "cm^-1 amagat^-2"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavenumber = _readonly_1d(self.wavenumber_cm_inverse, "wavenumber_cm_inverse")
        temperature = _readonly_1d(self.temperature_K, "temperature_K")
        k_cia = np.array(self.k_cia, dtype=float, copy=True)
        if np.any(wavenumber < 0.0):
            raise RobertValidationError("CIA wavenumber values must be non-negative")
        if wavenumber.size > 1 and not np.all(np.diff(wavenumber) > 0.0):
            raise RobertValidationError("CIA wavenumber grid must be strictly increasing")
        if np.any(temperature <= 0.0):
            raise RobertValidationError("CIA temperature values must be positive")
        if temperature.size > 1 and not np.all(np.diff(temperature) > 0.0):
            raise RobertValidationError("CIA temperature grid must be strictly increasing")
        if k_cia.shape != (len(self.pair_order), temperature.size, wavenumber.size):
            raise RobertValidationError("k_cia must have shape pair x temperature x wavenumber")
        if not np.all(np.isfinite(k_cia)) or np.any(k_cia < 0.0):
            raise RobertValidationError("CIA coefficients must be finite and non-negative")
        if not self.unit:
            raise RobertValidationError("CIA coefficient unit must not be empty")

        k_cia.setflags(write=False)
        object.__setattr__(self, "wavenumber_cm_inverse", wavenumber)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "k_cia", k_cia)
        object.__setattr__(self, "pair_order", tuple(self.pair_order))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_pairs(self) -> int:
        """Number of CIA pairs stored in the table."""

        return len(self.pair_order)


def read_cia_table(
    path: str | Path,
    *,
    dnu: float = 10.0,
    n_pairs: int = 9,
    endian: str | None = None,
) -> CiaTable:
    """Read an unformatted binary CIA table.

    The file format stores one float64 temperature record followed by one
    float32 coefficient record. Coefficients are stored by wavenumber,
    temperature, then CIA pair.
    """

    table_path = Path(path).expanduser()
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    if dnu <= 0.0 or not np.isfinite(dnu):
        raise RobertValidationError("CIA wavenumber spacing dnu must be finite and positive")
    if n_pairs < 1:
        raise RobertValidationError("CIA table must contain at least one pair")

    endians = (endian,) if endian is not None else ("<", ">")
    last_error: Exception | None = None
    for byte_order in endians:
        if byte_order not in {"<", ">"}:
            raise RobertValidationError("CIA table endian must be '<', '>', or None")
        try:
            return _read_cia_table_with_endian(
                table_path,
                dnu=float(dnu),
                n_pairs=int(n_pairs),
                endian=byte_order,
            )
        except RobertValidationError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RobertValidationError("could not read CIA table")


def cia_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    cia_table: CiaTable,
    *,
    normal_hydrogen: bool = True,
    name: str = "H2-H2/H2-He CIA",
) -> LayerOpticalDepth:
    """Compute CIA optical depth using a uniform-layer path estimate.

    ROBERT does not yet carry a full height grid here, so this helper derives
    the path length from the hydrostatic column density and ideal-gas number
    density at the layer pressure and temperature. The approximation is
    recorded in metadata.
    """

    atmosphere = gas_optical_depth.atmosphere
    wavenumber = spectral_grid_values_in_unit(gas_optical_depth.spectral_grid, "cm^-1")
    pressure_pa = pressure_values_in_unit(
        atmosphere.pressure_grid.centers,
        atmosphere.pressure_grid.unit,
        "pa",
    )
    temperature = atmosphere.temperature
    number_density_m3 = pressure_pa / (BOLTZMANN_CONSTANT_J_K * temperature)
    path_length_m = gas_optical_depth.layer_column_density_molecules_m2 / number_density_m3
    xlen_cm = path_length_m * 100.0
    total_amount_cm2 = gas_optical_depth.layer_column_density_molecules_m2 * 1.0e-4
    amagat_density = total_amount_cm2 / xlen_cm / AMAGAT_MOLECULES_CM3
    tau_path = xlen_cm * amagat_density**2

    if not np.all(np.isfinite(tau_path)) or np.any(tau_path < 0.0):
        raise RobertValidationError("CIA path calculation produced invalid values")

    h2 = _composition_profile(atmosphere.composition, "H2", atmosphere.n_layers)
    he = _composition_profile(atmosphere.composition, "He", atmosphere.n_layers)
    n2 = _composition_profile(atmosphere.composition, "N2", atmosphere.n_layers)
    ch4 = _composition_profile(atmosphere.composition, "CH4", atmosphere.n_layers)

    h2_h2_index = 2 if normal_hydrogen else 0
    h2_he_index = 3 if normal_hydrogen else 1
    pair_terms = (
        (h2_h2_index, h2 * h2),
        (h2_he_index, h2 * he),
        (4, h2 * n2),
        (5, n2 * ch4),
        (6, n2 * n2),
        (7, ch4 * ch4),
        (8, h2 * ch4),
    )

    tau = np.zeros((atmosphere.n_layers, gas_optical_depth.spectral_grid.size), dtype=float)
    active_pairs: list[str] = []
    for layer_index in range(atmosphere.n_layers):
        coefficients = _interpolate_cia_coefficients(
            cia_table,
            temperature[layer_index],
            wavenumber,
        )
        layer_coeff = np.zeros_like(wavenumber)
        for pair_index, mixing_factor in pair_terms:
            if pair_index >= cia_table.n_pairs:
                if mixing_factor[layer_index] > 0.0:
                    raise RobertValidationError(
                        "CIA table does not contain a required active pair"
                    )
                continue
            if mixing_factor[layer_index] <= 0.0:
                continue
            layer_coeff += coefficients[pair_index] * mixing_factor[layer_index]
            active_pairs.append(cia_table.pair_order[pair_index])
        tau[layer_index] = layer_coeff * tau_path[layer_index]

    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError("CIA optical-depth calculation produced invalid values")
    return LayerOpticalDepth(
        name=name,
        tau=tau,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        kind="absorption_continuum",
        metadata={
            "source_format": str(cia_table.metadata.get("source_format", "cia_binary")),
            "source_path": str(cia_table.metadata.get("source_path", "")),
            "path_model": "uniform_layer_ideal_gas_from_hydrostatic_column",
            "hydrogen_spin_state": "normal" if normal_hydrogen else "equilibrium",
            "active_pairs": ",".join(sorted(set(active_pairs))),
        },
    )


def rayleigh_scattering_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    *,
    default_h2_fraction_of_h2_he: float = 0.864,
    name: str = "H2/He Rayleigh scattering",
) -> LayerOpticalDepth:
    """Compute H2/He Rayleigh scattering extinction optical depth.

    The calculation applies an H2/He refractivity model to the H2+He column
    when those species are present in the atmosphere.
    """

    default_fraction = float(default_h2_fraction_of_h2_he)
    if not np.isfinite(default_fraction) or not 0.0 <= default_fraction <= 1.0:
        raise RobertValidationError("default_h2_fraction_of_h2_he must be in [0, 1]")

    atmosphere = gas_optical_depth.atmosphere
    wavelength_micron = spectral_grid_values_in_unit(gas_optical_depth.spectral_grid, "micron")
    h2 = _composition_profile(atmosphere.composition, "H2", atmosphere.n_layers)
    he = _composition_profile(atmosphere.composition, "He", atmosphere.n_layers)
    h2_he_fraction = h2 + he
    if np.any(h2_he_fraction > 0.0):
        h2_fraction = np.divide(
            h2,
            h2_he_fraction,
            out=np.full_like(h2, default_fraction),
            where=h2_he_fraction > 0.0,
        )
        rayleigh_column = gas_optical_depth.layer_column_density_molecules_m2 * h2_he_fraction
    else:
        h2_fraction = np.full(atmosphere.n_layers, default_fraction, dtype=float)
        rayleigh_column = gas_optical_depth.layer_column_density_molecules_m2

    lambda_m = wavelength_micron * 1.0e-6
    inverse_micron = 1.0 / wavelength_micron
    ah2 = 13.58e-5
    bh2 = 7.52e-3
    ahe = 3.48e-5
    bhe = 2.30e-3
    refractivity_h2 = ah2 * (1.0 + bh2 * inverse_micron**2)
    refractivity_he = ahe * (1.0 + bhe * inverse_micron**2)
    n0 = 1.01325e5 / (BOLTZMANN_CONSTANT_J_K * 273.15)
    faniso = 1.0

    mixed_refractivity = (
        h2_fraction[:, None] * refractivity_h2[None, :]
        + (1.0 - h2_fraction[:, None]) * refractivity_he[None, :]
    )
    cross_section_m2 = (
        32.0
        * np.pi**3
        * mixed_refractivity**2
        * faniso
        / (3.0 * (n0 * lambda_m[None, :] ** 2) ** 2)
    )
    tau = cross_section_m2 * rayleigh_column[:, None]
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError("Rayleigh optical-depth calculation produced invalid values")

    return LayerOpticalDepth(
        name=name,
        tau=tau,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        kind="scattering_extinction",
        metadata={
            "source": "H2/He refractivity model",
            "column_model": "H2/He fraction of hydrostatic layer column",
            "scattering_source_function": "not_included",
        },
    )


def _read_cia_table_with_endian(
    path: Path,
    *,
    dnu: float,
    n_pairs: int,
    endian: str,
) -> CiaTable:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        temperature_record = _read_fortran_record(handle, endian=endian, file_size=file_size)
        coefficients_record = _read_fortran_record(handle, endian=endian, file_size=file_size)

    if len(temperature_record) % 8 != 0:
        raise RobertValidationError("CIA temperature record is not float64-aligned")
    temperature = np.frombuffer(temperature_record, dtype=np.dtype(f"{endian}f8")).astype(float)
    coefficients = np.frombuffer(coefficients_record, dtype=np.dtype(f"{endian}f4")).astype(float)
    denominator = n_pairs * temperature.size
    if denominator <= 0 or coefficients.size % denominator != 0:
        raise RobertValidationError("CIA coefficient record has inconsistent dimensions")
    n_wavenumber = coefficients.size // denominator
    wavenumber = np.linspace(0.0, dnu * (n_wavenumber - 1), n_wavenumber)
    k_cia = coefficients.reshape(n_wavenumber, temperature.size, n_pairs).transpose(2, 1, 0)
    pair_order = DEFAULT_CIA_PAIR_ORDER[:n_pairs]
    if len(pair_order) != n_pairs:
        pair_order = tuple(f"pair_{index}" for index in range(n_pairs))
    return CiaTable(
        wavenumber_cm_inverse=wavenumber,
        temperature_K=temperature,
        k_cia=k_cia,
        pair_order=pair_order,
        metadata={
            "source_format": "fortran_unformatted_cia",
            "source_path": str(path),
            "dnu_cm^-1": f"{dnu:g}",
            "endian": endian,
        },
    )


def _read_fortran_record(handle, *, endian: str, file_size: int) -> bytes:
    prefix = handle.read(4)
    if len(prefix) != 4:
        raise RobertValidationError("unexpected end of CIA table while reading record marker")
    (record_size,) = struct.unpack(f"{endian}i", prefix)
    if record_size <= 0 or record_size > file_size - 8:
        raise RobertValidationError("invalid CIA Fortran record size")
    record = handle.read(record_size)
    suffix = handle.read(4)
    if len(record) != record_size or len(suffix) != 4:
        raise RobertValidationError("unexpected end of CIA table while reading record")
    (trailer_size,) = struct.unpack(f"{endian}i", suffix)
    if trailer_size != record_size:
        raise RobertValidationError("CIA Fortran record markers do not match")
    return record


def _interpolate_cia_coefficients(
    table: CiaTable,
    temperature_k: float,
    wavenumber_cm_inverse: NDArray[np.float64],
) -> NDArray[np.float64]:
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RobertValidationError("CIA interpolation temperature must be finite and positive")
    if temperature <= table.temperature_K[0]:
        native = table.k_cia[:, 0, :]
    elif temperature >= table.temperature_K[-1]:
        native = table.k_cia[:, -1, :]
    else:
        upper = int(np.searchsorted(table.temperature_K, temperature))
        lower = upper - 1
        fraction = (temperature - table.temperature_K[lower]) / (
            table.temperature_K[upper] - table.temperature_K[lower]
        )
        native = (1.0 - fraction) * table.k_cia[:, lower, :] + fraction * table.k_cia[:, upper, :]

    coefficients = np.vstack(
        [
            np.interp(
                wavenumber_cm_inverse,
                table.wavenumber_cm_inverse,
                native[pair_index],
                left=0.0,
                right=0.0,
            )
            for pair_index in range(table.n_pairs)
        ]
    )
    coefficients.setflags(write=False)
    return coefficients


def _composition_profile(
    composition: Mapping[str, NDArray[np.float64]],
    species: str,
    n_layers: int,
) -> NDArray[np.float64]:
    if species not in composition:
        profile = np.zeros(n_layers, dtype=float)
        profile.setflags(write=False)
        return profile
    profile = np.array(composition[species], dtype=float, copy=True)
    if profile.shape != (n_layers,):
        raise RobertValidationError(f"{species} composition must match pressure grid layers")
    if not np.all(np.isfinite(profile)) or np.any(profile < 0.0):
        raise RobertValidationError(f"{species} composition must be finite and non-negative")
    profile.setflags(write=False)
    return profile


def _top_to_bottom_order(pressure_grid: PressureGrid) -> NDArray[np.int64]:
    pressure = pressure_values_in_unit(pressure_grid.centers, pressure_grid.unit, "pa")
    return np.argsort(pressure).astype(np.int64)


def _restore_layer_order(
    values_in_top_to_bottom_order: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    restored = np.empty_like(values_in_top_to_bottom_order)
    restored[order] = values_in_top_to_bottom_order
    restored.setflags(write=False)
    return restored


def _readonly_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size < 1:
        raise RobertValidationError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_array(
    values: ArrayLike,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != shape:
        raise RobertValidationError(f"{name} has incorrect shape")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
