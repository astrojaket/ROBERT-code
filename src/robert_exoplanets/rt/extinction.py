"""Continuum and scattering-extinction optical-depth helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.resources import as_file, files
from pathlib import Path
import struct
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import (
    PressureGrid,
    RobertCoverageError,
    RobertValidationError,
    SpectralGrid,
)
from robert_exoplanets.core._immutability import immutable_mapping
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
    phase_function_moments: ArrayLike | None = None
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
        phase_moments = None
        if self.phase_function_moments is not None:
            phase_moments = _readonly_phase_function_moments(
                self.phase_function_moments,
                (self.pressure_grid.n_layers, self.spectral_grid.size),
            )
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "phase_function_moments", phase_moments)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

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
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def n_pairs(self) -> int:
        """Number of CIA pairs stored in the table."""

        return len(self.pair_order)

    @classmethod
    def from_petitradtrans_hdf(
        cls,
        path: str | Path,
        *,
        collision_pair: str,
    ) -> "CiaTable":
        """Load one H2-H2 or H2-He petitRADTRANS CIA HDF5 table.

        The pRT ``alpha`` values are absorption coefficients at unit amagat
        density product. They are placed in both equilibrium/normal hydrogen
        slots because these pRT tables do not encode a separate spin isomer.
        """

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RobertValidationError(
                "loading petitRADTRANS HDF5 CIA requires h5py"
            ) from exc

        normalized_pair = collision_pair.strip().upper().replace("_", "-")
        pair_indices = {"H2-H2": (0, 2), "H2-HE": (1, 3)}
        if normalized_pair not in pair_indices:
            raise RobertValidationError("collision_pair must be 'H2-H2' or 'H2-He'")
        source = Path(path).expanduser()
        required = ("t", "wavenumbers", "alpha", "mol_name", "mol_mass")
        try:
            with h5py.File(source, "r") as handle:
                missing = tuple(name for name in required if name not in handle)
                if missing:
                    raise RobertValidationError(
                        "petitRADTRANS CIA file is missing datasets: " + ", ".join(missing)
                    )
                temperature = np.asarray(handle["t"], dtype=float)
                wavenumber = np.asarray(handle["wavenumbers"], dtype=float)
                alpha = np.asarray(handle["alpha"], dtype=float)
                declared_unit = str(handle["alpha"].attrs.get("units", "")).strip()
                molecules = "+".join(_decode_hdf_strings(handle["mol_name"]))
                molar_masses = ",".join(
                    f"{value:.17g}" for value in np.asarray(handle["mol_mass"], dtype=float)
                )
                doi = _decode_hdf_first(handle.get("DOI"))
        except OSError as exc:
            raise RobertValidationError(
                f"could not read petitRADTRANS CIA file: {source}"
            ) from exc
        if declared_unit != "cm^-1":
            raise RobertValidationError(
                "petitRADTRANS CIA alpha dataset must declare units of cm^-1"
            )
        if alpha.shape != (temperature.size, wavenumber.size):
            raise RobertValidationError(
                "petitRADTRANS CIA alpha must have temperature x wavenumber shape"
            )
        k_cia = np.zeros((len(DEFAULT_CIA_PAIR_ORDER), temperature.size, wavenumber.size), dtype=float)
        for index in pair_indices[normalized_pair]:
            k_cia[index] = alpha
        return cls(
            wavenumber_cm_inverse=wavenumber,
            temperature_K=temperature,
            k_cia=k_cia,
            pair_order=DEFAULT_CIA_PAIR_ORDER,
            unit="cm^-1 amagat^-2",
            metadata={
                "source_format": "petitradtrans_cia_hdf5",
                "source_path": str(source.resolve()),
                "source_alpha_unit": declared_unit,
                "collision_pair": collision_pair,
                "molecules": molecules,
                "molar_masses_amu": molar_masses,
                "doi": doi,
                "hydrogen_spin_state": "not_separated_in_source",
            },
        )


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


def load_nemesispy_cia_table() -> CiaTable:
    """Load ROBERT's vendored NemesisPy v1.0.1 CIA reference table."""

    resource = files("robert_exoplanets").joinpath(
        "data/cia/exocia_hitran12_200-3800K.tab"
    )
    with as_file(resource) as path:
        table = read_cia_table(path)
    return replace(
        table,
        metadata={
            **dict(table.metadata),
            "source_project": "NEMESISPY",
            "source_tag": "v1.0.1",
            "source_commit": "a883805fcd402eab341308f39715670b0ae74cb8",
            "checksum_sha256": "5b519f02f98b205f20628ee5ec7f2829528d0bd356b449c4221ba8b2ef86ea0e",
            "license": "BSD-3-Clause",
        },
    )


def cia_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    cia_table: CiaTable,
    *,
    normal_hydrogen: bool = True,
    temperature_extrapolation: str = "raise",
    spectral_extrapolation: str = "raise",
    coefficient_interpolation: str = "linear",
    name: str = "H2-H2/H2-He CIA",
) -> LayerOpticalDepth:
    """Compute CIA optical depth using a uniform-layer path estimate.

    ROBERT does not yet carry a full height grid here, so this helper derives
    the path length from the hydrostatic column density and ideal-gas number
    density at the layer pressure and temperature. The approximation is
    recorded in metadata.
    """

    if temperature_extrapolation not in {"raise", "clip"}:
        raise RobertValidationError("temperature_extrapolation must be 'raise' or 'clip'")
    if spectral_extrapolation not in {"raise", "zero"}:
        raise RobertValidationError("spectral_extrapolation must be 'raise' or 'zero'")
    if coefficient_interpolation not in {"linear", "log"}:
        raise RobertValidationError("coefficient_interpolation must be 'linear' or 'log'")

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
            temperature_extrapolation=temperature_extrapolation,
            spectral_extrapolation=spectral_extrapolation,
            coefficient_interpolation=coefficient_interpolation,
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
            "temperature_extrapolation": temperature_extrapolation,
            "spectral_extrapolation": spectral_extrapolation,
            "coefficient_interpolation": coefficient_interpolation,
            "active_pairs": ",".join(sorted(set(active_pairs))),
        },
    )


def rayleigh_scattering_optical_depth(
    gas_optical_depth: GasOpticalDepth,
    *,
    default_h2_fraction_of_h2_he: float | None = None,
    name: str = "H2/He Rayleigh scattering",
) -> LayerOpticalDepth:
    """Compute H2/He Rayleigh scattering extinction optical depth.

    The calculation applies separate H2 and He refractivity models and adds
    their number-weighted molecular cross-sections.  Cross-sections, rather
    than refractivities, are additive for an ideal gas mixture. An atmosphere
    without explicit H2 and He raises unless the caller deliberately supplies
    a fallback H2 fraction for an assumed H2/He background.
    """

    default_fraction = None
    if default_h2_fraction_of_h2_he is not None:
        default_fraction = float(default_h2_fraction_of_h2_he)
        if not np.isfinite(default_fraction) or not 0.0 <= default_fraction <= 1.0:
            raise RobertValidationError("default_h2_fraction_of_h2_he must be in [0, 1]")

    atmosphere = gas_optical_depth.atmosphere
    wavelength_micron = spectral_grid_values_in_unit(gas_optical_depth.spectral_grid, "micron")
    h2 = _composition_profile(atmosphere.composition, "H2", atmosphere.n_layers)
    he = _composition_profile(atmosphere.composition, "He", atmosphere.n_layers)
    h2_he_fraction = h2 + he
    composition_fallback = "none"
    if not np.any(h2_he_fraction > 0.0):
        opacity_free_species = tuple(
            name.strip()
            for name in atmosphere.metadata.get("opacity_free_species", "").split(",")
            if name.strip()
        )
        opacity_free_background = any(
            name in atmosphere.composition
            and np.any(np.asarray(atmosphere.composition[name]) > 0.0)
            for name in opacity_free_species
        )
        if default_fraction is None and not opacity_free_background:
            raise RobertValidationError(
                "Rayleigh scattering requires explicit H2/He composition or an explicit "
                "default_h2_fraction_of_h2_he"
            )
        if default_fraction is not None:
            h2 = np.full(atmosphere.n_layers, default_fraction, dtype=float)
            he = 1.0 - h2
            composition_fallback = "explicit_H2/He"
        else:
            composition_fallback = "zero_H2/He_with_opacity_free_background"

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

    common_factor = (
        32.0
        * np.pi**3
        * faniso
        / (3.0 * (n0 * lambda_m[None, :] ** 2) ** 2)
    )
    cross_section_h2_m2 = common_factor * refractivity_h2[None, :] ** 2
    cross_section_he_m2 = common_factor * refractivity_he[None, :] ** 2
    tau = gas_optical_depth.layer_column_density_molecules_m2[:, None] * (
        h2[:, None] * cross_section_h2_m2 + he[:, None] * cross_section_he_m2
    )
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise RobertValidationError("Rayleigh optical-depth calculation produced invalid values")

    return LayerOpticalDepth(
        name=name,
        tau=tau,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        kind="scattering_extinction",
        phase_function_moments=np.repeat(
            np.array([1.0, 0.0, 0.5, 0.0, 0.0])[:, None],
            gas_optical_depth.spectral_grid.size,
            axis=1,
        ),
        metadata={
            "source": "H2/He refractivity model",
            "column_model": "species VMR times hydrostatic layer column",
            "mixture_rule": "number-weighted sum of molecular cross-sections",
            "composition_fallback": composition_fallback,
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
    *,
    temperature_extrapolation: str,
    spectral_extrapolation: str,
    coefficient_interpolation: str,
) -> NDArray[np.float64]:
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RobertValidationError("CIA interpolation temperature must be finite and positive")
    if temperature < table.temperature_K[0] or temperature > table.temperature_K[-1]:
        if temperature_extrapolation == "raise":
            raise RobertCoverageError(
                "atmosphere temperature is outside the CIA table temperature grid"
            )
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
        lower_values = table.k_cia[:, lower, :]
        upper_values = table.k_cia[:, upper, :]
        if coefficient_interpolation == "linear":
            native = (1.0 - fraction) * lower_values + fraction * upper_values
        else:
            both_positive = (lower_values > 0.0) & (upper_values > 0.0)
            native = (1.0 - fraction) * lower_values + fraction * upper_values
            native[both_positive] = np.exp(
                (1.0 - fraction) * np.log(lower_values[both_positive])
                + fraction * np.log(upper_values[both_positive])
            )

    requested_min = float(np.min(wavenumber_cm_inverse))
    requested_max = float(np.max(wavenumber_cm_inverse))
    if (
        requested_min < float(table.wavenumber_cm_inverse[0])
        or requested_max > float(table.wavenumber_cm_inverse[-1])
    ) and spectral_extrapolation == "raise":
        raise RobertCoverageError("requested spectrum is outside the CIA table wavenumber grid")
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


def _decode_hdf_strings(dataset: object) -> tuple[str, ...]:
    values = np.asarray(dataset)
    decoded = []
    for value in values.reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def _decode_hdf_first(dataset: object | None) -> str:
    if dataset is None:
        return ""
    values = _decode_hdf_strings(dataset)
    return "" if not values else values[0]


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


def _readonly_phase_function_moments(
    values: ArrayLike,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape == (5, shape[1]):
        array = np.repeat(array[:, None, :], shape[0], axis=1)
    elif array.shape != (5, shape[0], shape[1]):
        raise RobertValidationError(
            "phase_function_moments must have shape (5, spectral) or "
            "(5, layer, spectral)"
        )
    if not np.all(np.isfinite(array)):
        raise RobertValidationError("phase_function_moments must contain only finite values")
    limits = (2.0 * np.arange(5) + 1.0)[:, None, None]
    if np.any(np.abs(array) > limits * (1.0 + 1.0e-10)):
        raise RobertValidationError("phase_function_moments exceed physical bounds")
    if not np.allclose(array[0], 1.0, rtol=0.0, atol=1.0e-12):
        raise RobertValidationError("phase_function_moments[0] must equal one")
    array.setflags(write=False)
    return array
