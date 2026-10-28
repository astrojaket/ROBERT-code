"""Optional FastChem gas and condensate diagnostics.

The result includes FastChem's fictitious condensate formula-unit densities,
gas species densities, and a mass-equivalent conversion. The conversion does
not turn formula units into aerosol particle counts or specify particle sizes.
In rainout mode, condensate mass values describe the local FastChem
condensation solution; they do not describe condensate retained in a cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from importlib import import_module, metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import (
    PressureGrid,
    RobertConfigError,
    RobertValidationError,
)
from robert_exoplanets.core._immutability import immutable_mapping


FastChemCondensationMode = Literal["equilibrium", "rainout"]

_PRESSURE_TO_BAR = {
    "pa": 1.0e-5,
    "pascal": 1.0e-5,
    "pascals": 1.0e-5,
    "bar": 1.0,
    "bars": 1.0,
    "mbar": 1.0e-3,
    "millibar": 1.0e-3,
    "millibars": 1.0e-3,
    "atm": 1.01325,
    "atmosphere": 1.01325,
    "atmospheres": 1.01325,
}

# 2022 CODATA recommended unified atomic mass unit (NIST):
# https://physics.nist.gov/cuu/pdf/RevModPhys.97.025002.pdf
_ATOMIC_MASS_UNIT_KG = 1.66053906892e-27
_CM3_PER_M3 = 1.0e6
_BOLTZMANN_J_K = 1.380649e-23
_FASTCHEM_PRESSURE_CLOSURE_RTOL = 1.0e-3


def _readonly_float_vector(
    values: ArrayLike,
    name: str,
    size: int,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.shape != (size,):
        raise RobertValidationError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise RobertValidationError(f"{name} must be positive")
    if non_negative and np.any(array < 0.0):
        raise RobertValidationError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array


def _readonly_int_vector(
    values: ArrayLike,
    name: str,
    size: int,
) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape != (size,):
        raise RobertValidationError(f"{name} must have shape ({size},)")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RobertValidationError(f"{name} must contain integers") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise RobertValidationError(f"{name} must contain finite integers")
    array = np.array(numeric, dtype=np.int64, copy=True)
    if np.any(array < 0):
        raise RobertValidationError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array


def _readonly_int_matrix(
    values: ArrayLike,
    name: str,
    shape: tuple[int, int],
) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape != shape:
        raise RobertValidationError(f"{name} must have shape {shape}")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RobertValidationError(f"{name} must contain integers") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise RobertValidationError(f"{name} must contain finite integers")
    array = np.array(numeric, dtype=np.int64, copy=True)
    if np.any(array < 0):
        raise RobertValidationError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array


def _readonly_float_matrix(
    values: ArrayLike,
    name: str,
    shape: tuple[int, int],
    *,
    non_negative: bool = False,
    unit_interval: bool = False,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 2 or array.shape != shape:
        raise RobertValidationError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    if non_negative and np.any(array < 0.0):
        raise RobertValidationError(f"{name} must be non-negative")
    if unit_interval and (np.any(array < 0.0) or np.any(array > 1.0)):
        raise RobertValidationError(f"{name} must lie in [0, 1]")
    array.setflags(write=False)
    return array


def _pressure_values_bar(pressure_grid: PressureGrid) -> NDArray[np.float64]:
    unit = str(pressure_grid.unit).strip().lower()
    try:
        factor = _PRESSURE_TO_BAR[unit]
    except KeyError as exc:
        raise RobertValidationError(
            f"unsupported pressure unit for FastChem condensate diagnostics: "
            f"{pressure_grid.unit}"
        ) from exc
    pressure_bar = np.array(pressure_grid.centers * factor, dtype=float, copy=True)
    if not np.all(np.isfinite(pressure_bar)) or np.any(pressure_bar <= 0.0):
        raise RobertValidationError("FastChem pressures must be finite and positive")
    pressure_bar.setflags(write=False)
    return pressure_bar


def _load_pyfastchem() -> tuple[Any, str]:
    """Load the optional PyFastChem package only when this feature is used."""

    try:
        api = import_module("pyfastchem")
    except Exception as exc:  # pragma: no cover - environment-specific import path.
        raise RobertConfigError(
            "FastChem condensate diagnostics require the optional pyfastchem package"
        ) from exc
    try:
        version = importlib_metadata.version("pyfastchem")
    except (
        importlib_metadata.PackageNotFoundError
    ):  # pragma: no cover - unusual install.
        version = str(getattr(api, "__version__", "unknown"))
    return api, version


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RobertConfigError(f"could not read FastChem input file: {path}") from exc
    return digest.hexdigest()


def _condensate_stoichiometry_from_file(
    path: Path,
    *,
    element_symbols: tuple[str, ...],
    condensate_symbols: tuple[str, ...],
) -> NDArray[np.float64]:
    """Read FastChem's elemental coefficients for its loaded condensates."""

    parsed: dict[str, dict[str, int]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RobertConfigError(
            f"could not read FastChem condensate species table: {path}"
        ) from exc

    for line_number, line in enumerate(lines, start=1):
        data = line.split("#", maxsplit=1)[0].strip()
        if not data or ":" not in data:
            continue
        left, right = data.split(":", maxsplit=1)
        fields = left.split()
        species = fields[0] if fields else ""
        tokens = right.split()
        if not species or not tokens or len(tokens) % 2:
            raise RobertConfigError(
                "invalid FastChem condensate stoichiometry at "
                f"{path}:{line_number}"
            )
        coefficients: dict[str, int] = {}
        for element, raw_count in zip(tokens[::2], tokens[1::2], strict=True):
            try:
                count = int(raw_count)
            except ValueError as exc:
                raise RobertConfigError(
                    "invalid FastChem stoichiometric coefficient at "
                    f"{path}:{line_number}"
                ) from exc
            if count <= 0 or element in coefficients:
                raise RobertConfigError(
                    "invalid FastChem condensate stoichiometry at "
                    f"{path}:{line_number}"
                )
            coefficients[element] = count
        if species in parsed:
            raise RobertConfigError(
                f"duplicate condensate in FastChem species table: {species}"
            )
        parsed[species] = coefficients

    element_indices = {symbol: index for index, symbol in enumerate(element_symbols)}
    stoichiometry = np.zeros(
        (len(condensate_symbols), len(element_symbols)), dtype=float
    )
    for row, species in enumerate(condensate_symbols):
        try:
            coefficients = parsed[species]
        except KeyError as exc:
            raise RobertConfigError(
                "FastChem did not expose stoichiometry for loaded condensate "
                f"{species!r} in {path}"
            ) from exc
        for element, count in coefficients.items():
            try:
                column = element_indices[element]
            except KeyError as exc:
                raise RobertConfigError(
                    f"FastChem condensate {species!r} uses unknown element {element!r}"
                ) from exc
            stoichiometry[row, column] = count
    if np.any(np.sum(stoichiometry, axis=1) <= 0.0):
        raise RobertConfigError("FastChem condensate stoichiometry must be positive")
    stoichiometry.setflags(write=False)
    return stoichiometry


@dataclass(frozen=True)
class FastChemCondensateResult:
    """Immutable gas and condensate diagnostics from one FastChem calculation.

    ``condensate_number_density_cm3`` has shape ``(n_layers, n_condensates)``.
    Its values are FastChem's reported fictitious condensate number densities
    in cm^-3. They are formula-unit densities, not aerosol particle counts.

    ``gas_species_number_density_cm3`` contains all gas species from the same
    FastChem calculation. Molecular weights use u per species. The mass
    densities use the 2022 CODATA unified atomic mass unit and are in kg/m^3.
    ``condensate_mass_fraction_of_total`` is each selected condensate's mass
    divided by gas mass plus the mass of all condensates in the FastChem
    calculation. ``condensate_mass_mixing_ratio_gas`` is each selected
    condensate's mass divided by gas mass. Both ratios are kg/kg.

    In rainout mode, condensate mass values are local mass equivalents from
    FastChem's rainout chemistry. They are not a cloud-retention profile.

    ``element_condensation_degree`` has shape ``(n_layers, n_elements)`` and
    contains unitless fractions. All layer arrays use the input pressure-grid
    order, including rainout results.
    """

    pressure_grid: PressureGrid
    temperature_k: ArrayLike
    mode: FastChemCondensationMode
    condensate_symbols: tuple[str, ...]
    condensate_names: tuple[str, ...]
    condensate_number_density_cm3: ArrayLike
    condensate_molecular_weight_u: ArrayLike
    all_condensate_symbols: tuple[str, ...]
    all_condensate_number_density_cm3: ArrayLike
    all_condensate_molecular_weight_u: ArrayLike
    gas_species_symbols: tuple[str, ...]
    gas_species_number_density_cm3: ArrayLike
    gas_species_molecular_weight_u: ArrayLike
    element_symbols: tuple[str, ...]
    element_condensation_degree: ArrayLike
    element_conserved: ArrayLike
    fastchem_flag: ArrayLike
    chemistry_iterations: ArrayLike
    condensation_iterations: ArrayLike
    total_iterations: ArrayLike
    return_flag: int
    metadata: Mapping[str, str] = field(default_factory=dict)
    gas_number_density_cm3: NDArray[np.float64] = field(init=False)
    gas_mean_molecular_weight_u: NDArray[np.float64] = field(init=False)
    gas_mass_density_kg_m3: NDArray[np.float64] = field(init=False)
    total_condensate_mass_density_kg_m3: NDArray[np.float64] = field(init=False)
    condensate_mass_density_kg_m3: NDArray[np.float64] = field(init=False)
    condensate_mass_fraction_of_total: NDArray[np.float64] = field(init=False)
    condensate_mass_mixing_ratio_gas: NDArray[np.float64] = field(init=False)
    total_condensate_mass_fraction_of_total: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("equilibrium", "rainout"):
            raise RobertValidationError(
                "FastChem condensation mode must be 'equilibrium' or 'rainout'"
            )
        n_layers = self.pressure_grid.n_layers
        temperature = _readonly_float_vector(
            self.temperature_k, "temperature_k", n_layers, positive=True
        )
        condensate_symbols = tuple(
            str(value).strip() for value in self.condensate_symbols
        )
        condensate_names = tuple(str(value).strip() for value in self.condensate_names)
        all_condensate_symbols = tuple(
            str(value).strip() for value in self.all_condensate_symbols
        )
        gas_species_symbols = tuple(
            str(value).strip() for value in self.gas_species_symbols
        )
        element_symbols = tuple(str(value).strip() for value in self.element_symbols)
        if not condensate_symbols or any(not value for value in condensate_symbols):
            raise RobertValidationError(
                "condensate_symbols must contain non-empty symbols"
            )
        if len(set(condensate_symbols)) != len(condensate_symbols):
            raise RobertValidationError("condensate_symbols must be unique")
        if not all_condensate_symbols or any(
            not value for value in all_condensate_symbols
        ):
            raise RobertValidationError(
                "all_condensate_symbols must contain non-empty symbols"
            )
        if len(set(all_condensate_symbols)) != len(all_condensate_symbols):
            raise RobertValidationError("all_condensate_symbols must be unique")
        if any(value not in all_condensate_symbols for value in condensate_symbols):
            raise RobertValidationError(
                "selected condensates must exist in all_condensate_symbols"
            )
        if len(condensate_names) != len(condensate_symbols) or any(
            not value for value in condensate_names
        ):
            raise RobertValidationError(
                "condensate_names must be non-empty and match condensate_symbols"
            )
        if not element_symbols or any(not value for value in element_symbols):
            raise RobertValidationError(
                "element_symbols must contain non-empty symbols"
            )
        if len(set(element_symbols)) != len(element_symbols):
            raise RobertValidationError("element_symbols must be unique")
        if not gas_species_symbols or any(not value for value in gas_species_symbols):
            raise RobertValidationError("gas_species_symbols must be non-empty")
        if len(set(gas_species_symbols)) != len(gas_species_symbols):
            raise RobertValidationError("gas_species_symbols must be unique")

        number_density = _readonly_float_matrix(
            self.condensate_number_density_cm3,
            "condensate_number_density_cm3",
            (n_layers, len(condensate_symbols)),
            non_negative=True,
        )
        condensate_weights = _readonly_float_vector(
            self.condensate_molecular_weight_u,
            "condensate_molecular_weight_u",
            len(condensate_symbols),
            positive=True,
        )
        all_condensate_number_density = _readonly_float_matrix(
            self.all_condensate_number_density_cm3,
            "all_condensate_number_density_cm3",
            (n_layers, len(all_condensate_symbols)),
            non_negative=True,
        )
        all_condensate_weights = _readonly_float_vector(
            self.all_condensate_molecular_weight_u,
            "all_condensate_molecular_weight_u",
            len(all_condensate_symbols),
            positive=True,
        )
        gas_number_density_by_species = _readonly_float_matrix(
            self.gas_species_number_density_cm3,
            "gas_species_number_density_cm3",
            (n_layers, len(gas_species_symbols)),
            non_negative=True,
        )
        gas_weights = _readonly_float_vector(
            self.gas_species_molecular_weight_u,
            "gas_species_molecular_weight_u",
            len(gas_species_symbols),
            positive=True,
        )
        selected_indices = np.array(
            [all_condensate_symbols.index(symbol) for symbol in condensate_symbols],
            dtype=np.int64,
        )
        if not np.array_equal(
            number_density, all_condensate_number_density[:, selected_indices]
        ) or not np.array_equal(
            condensate_weights, all_condensate_weights[selected_indices]
        ):
            raise RobertValidationError(
                "selected condensate diagnostics must match the full FastChem output"
            )
        element_degree = _readonly_float_matrix(
            self.element_condensation_degree,
            "element_condensation_degree",
            (n_layers, len(element_symbols)),
            unit_interval=True,
        )
        element_conserved = _readonly_int_matrix(
            self.element_conserved,
            "element_conserved",
            (n_layers, len(element_symbols)),
        )
        if np.any(element_conserved != 1):
            raise RobertValidationError(
                "FastChem element conservation failed for one or more layers"
            )
        fastchem_flag = _readonly_int_vector(
            self.fastchem_flag, "fastchem_flag", n_layers
        )
        chemistry_iterations = _readonly_int_vector(
            self.chemistry_iterations, "chemistry_iterations", n_layers
        )
        condensation_iterations = _readonly_int_vector(
            self.condensation_iterations, "condensation_iterations", n_layers
        )
        total_iterations = _readonly_int_vector(
            self.total_iterations, "total_iterations", n_layers
        )
        if int(self.return_flag) != 0 or np.any(fastchem_flag != 0):
            raise RobertValidationError(
                "FastChem condensate result must contain only successful status flags"
            )

        gas_number_density = np.sum(gas_number_density_by_species, axis=1)
        gas_mass_u_cm3 = np.sum(
            gas_number_density_by_species * gas_weights[None, :], axis=1
        )
        all_condensate_mass_u_cm3 = np.sum(
            all_condensate_number_density * all_condensate_weights[None, :], axis=1
        )
        if np.any(gas_number_density <= 0.0) or np.any(gas_mass_u_cm3 <= 0.0):
            raise RobertValidationError(
                "FastChem gas number and mass densities must be positive"
            )
        pressure_bar = _pressure_values_bar(self.pressure_grid)
        gas_pressure_pa = (
            gas_number_density * _CM3_PER_M3 * _BOLTZMANN_J_K * temperature
        )
        input_pressure_pa = pressure_bar * 1.0e5
        pressure_error = np.abs(gas_pressure_pa / input_pressure_pa - 1.0)
        if np.any(pressure_error > _FASTCHEM_PRESSURE_CLOSURE_RTOL):
            raise RobertValidationError(
                "FastChem gas number densities do not reproduce the input pressure; "
                f"maximum relative error is {float(np.max(pressure_error)):.6g}"
            )
        mass_conversion = _ATOMIC_MASS_UNIT_KG * _CM3_PER_M3
        gas_mass_density = gas_mass_u_cm3 * mass_conversion
        all_condensate_mass_density = all_condensate_mass_u_cm3 * mass_conversion
        total_mass_density = gas_mass_density + all_condensate_mass_density
        selected_mass_density = (
            number_density * condensate_weights[None, :] * mass_conversion
        )
        if (
            not np.all(np.isfinite(total_mass_density))
            or np.any(total_mass_density <= 0.0)
        ):
            raise RobertValidationError(
                "FastChem total gas and condensate mass density must be finite and positive"
            )

        gas_mean_molecular_weight = gas_mass_u_cm3 / gas_number_density
        condensate_mass_fraction = selected_mass_density / total_mass_density[:, None]
        condensate_mass_mixing_ratio = selected_mass_density / gas_mass_density[:, None]
        total_condensate_mass_fraction = (
            all_condensate_mass_density / total_mass_density
        )
        selected_mass_sum = np.sum(selected_mass_density, axis=1)
        if np.any(condensate_mass_fraction > 1.0) or np.any(
            total_condensate_mass_fraction > 1.0
        ):
            raise RobertValidationError(
                "FastChem condensate mass fractions must lie in [0, 1]"
            )
        if np.any(
            selected_mass_sum
            > all_condensate_mass_density
            + 1.0e-12 * np.maximum(all_condensate_mass_density, 1.0e-300)
        ):
            raise RobertValidationError(
                "selected condensate mass exceeds the full FastChem condensate mass"
            )
        derived_arrays = {
            "gas_number_density_cm3": gas_number_density,
            "gas_mean_molecular_weight_u": gas_mean_molecular_weight,
            "gas_mass_density_kg_m3": gas_mass_density,
            "total_condensate_mass_density_kg_m3": all_condensate_mass_density,
            "condensate_mass_density_kg_m3": selected_mass_density,
            "condensate_mass_fraction_of_total": condensate_mass_fraction,
            "condensate_mass_mixing_ratio_gas": condensate_mass_mixing_ratio,
            "total_condensate_mass_fraction_of_total": total_condensate_mass_fraction,
        }
        for name, values in derived_arrays.items():
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise RobertValidationError(f"{name} must be finite and non-negative")
            values.setflags(write=False)

        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "condensate_symbols", condensate_symbols)
        object.__setattr__(self, "condensate_names", condensate_names)
        object.__setattr__(self, "all_condensate_symbols", all_condensate_symbols)
        object.__setattr__(self, "gas_species_symbols", gas_species_symbols)
        object.__setattr__(self, "element_symbols", element_symbols)
        object.__setattr__(self, "condensate_number_density_cm3", number_density)
        object.__setattr__(self, "condensate_molecular_weight_u", condensate_weights)
        object.__setattr__(
            self,
            "all_condensate_number_density_cm3",
            all_condensate_number_density,
        )
        object.__setattr__(
            self, "all_condensate_molecular_weight_u", all_condensate_weights
        )
        object.__setattr__(
            self,
            "gas_species_number_density_cm3",
            gas_number_density_by_species,
        )
        object.__setattr__(self, "gas_species_molecular_weight_u", gas_weights)
        object.__setattr__(self, "element_condensation_degree", element_degree)
        object.__setattr__(self, "element_conserved", element_conserved)
        object.__setattr__(self, "fastchem_flag", fastchem_flag)
        object.__setattr__(self, "chemistry_iterations", chemistry_iterations)
        object.__setattr__(self, "condensation_iterations", condensation_iterations)
        object.__setattr__(self, "total_iterations", total_iterations)
        object.__setattr__(self, "return_flag", int(self.return_flag))
        for name, values in derived_arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class _FastChemRuntime:
    api: Any
    fastchem: Any
    version: str
    condensate_symbols: tuple[str, ...]
    condensate_names: tuple[str, ...]
    condensate_indices: NDArray[np.int64]
    all_condensate_symbols: tuple[str, ...]
    all_condensate_molecular_weights_u: NDArray[np.float64]
    gas_species_symbols: tuple[str, ...]
    gas_species_molecular_weights_u: NDArray[np.float64]
    element_symbols: tuple[str, ...]
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class FastChemCondensateDiagnostics:
    """Opt-in FastChem equilibrium or rainout condensate diagnostics.

    ``fastchem_path`` must contain the standard FastChem tree with
    ``input/element_abundances/asplund_2009.dat``,
    ``input/logK/logK.dat``, and ``input/logK/logK_condensates.dat``. FastChem
    uses its abundance file as supplied. This diagnostic does not change
    metallicity or C/O and does not connect to ROBERT's gas chemistry or cloud
    models.

    For rainout mode, pressure and temperature pairs are passed to FastChem in
    decreasing pressure order (deep to high altitude), because FastChem carries
    depleted elemental abundances from each input layer to the next. Returned
    arrays are restored to the input ``PressureGrid`` order.
    """

    fastchem_path: str | Path
    mode: FastChemCondensationMode = "equilibrium"
    condensate_symbols: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in ("equilibrium", "rainout"):
            raise RobertValidationError(
                "FastChem condensation mode must be 'equilibrium' or 'rainout'"
            )
        path = Path(self.fastchem_path).expanduser()
        symbols = self.condensate_symbols
        if symbols is not None:
            symbols = tuple(str(value).strip() for value in symbols)
            if not symbols or any(not value for value in symbols):
                raise RobertValidationError(
                    "condensate_symbols must be None or a non-empty tuple"
                )
            if len(set(symbols)) != len(symbols):
                raise RobertValidationError("condensate_symbols must be unique")
        object.__setattr__(self, "fastchem_path", path)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "condensate_symbols", symbols)

    def evaluate(
        self,
        pressure_grid: PressureGrid,
        temperature_k: ArrayLike,
    ) -> FastChemCondensateResult:
        """Calculate raw condensate diagnostics on the supplied layer grid."""

        temperature = _readonly_float_vector(
            temperature_k, "temperature_k", pressure_grid.n_layers, positive=True
        )
        pressure_bar = _pressure_values_bar(pressure_grid)
        runtime = self._runtime

        if self.mode == "rainout":
            input_order = np.argsort(-pressure_bar, kind="stable")
        else:
            # Equilibrium condensation solves each T-P point independently.
            input_order = np.arange(pressure_grid.n_layers, dtype=int)
        restore_order = np.argsort(input_order, kind="stable")

        input_data = runtime.api.FastChemInput()
        input_data.pressure = pressure_bar[input_order].tolist()
        input_data.temperature = temperature[input_order].tolist()
        input_data.equilibrium_condensation = self.mode == "equilibrium"
        input_data.rainout_condensation = self.mode == "rainout"

        output_data = runtime.api.FastChemOutput()
        try:
            return_flag = int(runtime.fastchem.calcDensities(input_data, output_data))
        except Exception as exc:
            raise RobertValidationError(
                "FastChem condensate calculation failed while evaluating the T-P grid"
            ) from exc
        success_flag = int(getattr(runtime.api, "FASTCHEM_SUCCESS", 0))
        if return_flag != success_flag:
            raise RobertValidationError(
                f"FastChem condensate calculation returned status {return_flag}"
            )

        n_layers = pressure_grid.n_layers
        layer_flags = _output_int_vector(output_data, "fastchem_flag", n_layers)
        if np.any(layer_flags != success_flag):
            raise RobertValidationError(
                "FastChem condensate calculation failed for one or more layers: "
                f"status flags {layer_flags.tolist()}"
            )

        element_conserved = _output_int_matrix(
            output_data,
            "element_conserved",
            (n_layers, len(runtime.element_symbols)),
        )
        if np.any(element_conserved != 1):
            raise RobertValidationError(
                "FastChem element conservation failed for one or more layers"
            )

        gas_number_density = _output_float_matrix(
            output_data,
            "number_densities",
            (n_layers, int(runtime.fastchem.getGasSpeciesNumber())),
            non_negative=True,
        )

        full_number_density = _output_float_matrix(
            output_data,
            "number_densities_cond",
            (n_layers, int(runtime.fastchem.getCondSpeciesNumber())),
            non_negative=True,
        )
        selected_number_density = full_number_density[:, runtime.condensate_indices]
        element_degree = _output_float_matrix(
            output_data,
            "element_cond_degree",
            (n_layers, len(runtime.element_symbols)),
            unit_interval=True,
        )
        chemistry_iterations = _output_int_vector(
            output_data, "nb_chemistry_iterations", n_layers
        )
        condensation_iterations = _output_int_vector(
            output_data, "nb_cond_iterations", n_layers
        )
        total_iterations = _output_int_vector(output_data, "nb_iterations", n_layers)

        return FastChemCondensateResult(
            pressure_grid=pressure_grid,
            temperature_k=temperature,
            mode=self.mode,
            condensate_symbols=runtime.condensate_symbols,
            condensate_names=runtime.condensate_names,
            condensate_number_density_cm3=selected_number_density[restore_order],
            condensate_molecular_weight_u=(
                runtime.all_condensate_molecular_weights_u[
                    runtime.condensate_indices
                ]
            ),
            all_condensate_symbols=runtime.all_condensate_symbols,
            all_condensate_number_density_cm3=full_number_density[restore_order],
            all_condensate_molecular_weight_u=(
                runtime.all_condensate_molecular_weights_u
            ),
            gas_species_symbols=runtime.gas_species_symbols,
            gas_species_number_density_cm3=gas_number_density[restore_order],
            gas_species_molecular_weight_u=(
                runtime.gas_species_molecular_weights_u
            ),
            element_symbols=runtime.element_symbols,
            element_condensation_degree=element_degree[restore_order],
            element_conserved=element_conserved[restore_order],
            fastchem_flag=layer_flags[restore_order],
            chemistry_iterations=chemistry_iterations[restore_order],
            condensation_iterations=condensation_iterations[restore_order],
            total_iterations=total_iterations[restore_order],
            return_flag=return_flag,
            metadata=runtime.metadata,
        )

    @cached_property
    def _runtime(self) -> _FastChemRuntime:
        files = self._input_files()
        api, version = _load_pyfastchem()
        try:
            fastchem = api.FastChem(
                str(files["element_abundances"]),
                str(files["gas_equilibrium"]),
                str(files["condensate_equilibrium"]),
                0,
            )
        except Exception as exc:
            raise RobertConfigError(
                "FastChem could not load the configured abundance and equilibrium "
                f"tables under {self.fastchem_path}"
            ) from exc

        try:
            condensate_count = int(fastchem.getCondSpeciesNumber())
            gas_species_count = int(fastchem.getGasSpeciesNumber())
            element_count = int(fastchem.getElementNumber())
        except Exception as exc:
            raise RobertConfigError(
                "FastChem did not expose condensate and element species metadata"
            ) from exc
        if condensate_count < 1:
            raise RobertConfigError(
                "FastChem loaded no condensate species from "
                f"{files['condensate_equilibrium']}"
            )
        if element_count < 1:
            raise RobertConfigError("FastChem loaded no chemical elements")
        if gas_species_count < 1:
            raise RobertConfigError("FastChem loaded no gas species")

        all_symbols = tuple(
            str(fastchem.getCondSpeciesSymbol(index)).strip()
            for index in range(condensate_count)
        )
        all_names = tuple(
            str(fastchem.getCondSpeciesName(index)).strip()
            for index in range(condensate_count)
        )
        element_symbols = tuple(
            str(fastchem.getElementSymbol(index)).strip()
            for index in range(element_count)
        )
        gas_species_symbols = tuple(
            str(fastchem.getGasSpeciesSymbol(index)).strip()
            for index in range(gas_species_count)
        )
        if any(not value for value in (*all_symbols, *all_names, *element_symbols)):
            raise RobertConfigError(
                "FastChem returned an empty species or element label"
            )
        if any(not value for value in gas_species_symbols):
            raise RobertConfigError("FastChem returned an empty gas species label")
        if len(set(all_symbols)) != len(all_symbols):
            raise RobertConfigError("FastChem returned duplicate condensate symbols")
        if len(set(element_symbols)) != len(element_symbols):
            raise RobertConfigError("FastChem returned duplicate element symbols")
        if len(set(gas_species_symbols)) != len(gas_species_symbols):
            raise RobertConfigError("FastChem returned duplicate gas species symbols")

        try:
            gas_species_weights = np.array(
                [
                    fastchem.getGasSpeciesWeight(index)
                    for index in range(gas_species_count)
                ],
                dtype=float,
            )
            element_weights = np.array(
                [fastchem.getElementWeight(index) for index in range(element_count)],
                dtype=float,
            )
        except Exception as exc:
            raise RobertConfigError(
                "FastChem did not expose gas species and element atomic weights"
            ) from exc
        gas_species_weights = _readonly_float_vector(
            gas_species_weights,
            "gas_species_molecular_weight_u",
            gas_species_count,
            positive=True,
        )
        element_weights = _readonly_float_vector(
            element_weights,
            "element_atomic_weight_u",
            element_count,
            positive=True,
        )

        stoichiometry = _condensate_stoichiometry_from_file(
            files["condensate_equilibrium"],
            element_symbols=element_symbols,
            condensate_symbols=all_symbols,
        )
        stoichiometric_weights = stoichiometry @ element_weights
        if not np.all(np.isfinite(stoichiometric_weights)) or np.any(
            stoichiometric_weights <= 0.0
        ):
            raise RobertConfigError(
                "FastChem condensate formula weights must be finite and positive"
            )

        get_condensate_weight = getattr(fastchem, "getCondSpeciesWeight", None)
        if get_condensate_weight is None:
            backend_condensate_weights = np.zeros(condensate_count, dtype=float)
        else:
            try:
                backend_condensate_weights = np.array(
                    [
                        get_condensate_weight(index)
                        for index in range(condensate_count)
                    ],
                    dtype=float,
                )
            except Exception as exc:
                raise RobertConfigError(
                    "FastChem condensate weight lookup failed"
                ) from exc
        if (
            backend_condensate_weights.shape != (condensate_count,)
            or not np.all(np.isfinite(backend_condensate_weights))
            or np.any(backend_condensate_weights < 0.0)
        ):
            raise RobertConfigError(
                "FastChem returned invalid condensate molecular weights"
            )
        api_has_weight = backend_condensate_weights > 0.0
        if np.any(api_has_weight) and not np.allclose(
            backend_condensate_weights[api_has_weight],
            stoichiometric_weights[api_has_weight],
            rtol=1.0e-6,
            atol=1.0e-8,
        ):
            raise RobertConfigError(
                "FastChem condensate weights do not match table stoichiometry"
            )
        if np.all(api_has_weight):
            condensate_weights = backend_condensate_weights
            condensate_weight_source = "pyfastchem_getCondSpeciesWeight"
        else:
            # FastChem 3.1.2 exposes this method but reports zero for each
            # condensate. In that case, derive formula weights from the loaded
            # stoichiometry and the same FastChem object's atomic weights.
            condensate_weights = stoichiometric_weights
            condensate_weight_source = (
                "loaded_condensate_stoichiometry_times_fastchem_element_weights"
            )
        condensate_weights = _readonly_float_vector(
            condensate_weights,
            "all_condensate_molecular_weight_u",
            condensate_count,
            positive=True,
        )
        gas_species_weights.setflags(write=False)

        if self.condensate_symbols is None:
            selected_indices = np.arange(condensate_count, dtype=np.int64)
        else:
            unknown = int(getattr(api, "FASTCHEM_UNKNOWN_SPECIES", 9999999))
            indices: list[int] = []
            for symbol in self.condensate_symbols:
                index = int(fastchem.getCondSpeciesIndex(symbol))
                if index == unknown or index < 0 or index >= condensate_count:
                    raise RobertConfigError(
                        f"FastChem does not know condensate species {symbol!r}"
                    )
                indices.append(index)
            if len(set(indices)) != len(indices):
                raise RobertConfigError(
                    "FastChem condensate symbols resolve to duplicate species indices"
                )
            selected_indices = np.asarray(indices, dtype=np.int64)
        selected_indices.setflags(write=False)

        metadata = {
            "fastchem_backend": "pyfastchem",
            "fastchem_version": version,
            "fastchem_path": str(Path(self.fastchem_path).expanduser().resolve()),
            "fastchem_element_abundance_table": str(files["element_abundances"]),
            "fastchem_element_abundance_sha256": _sha256_file(
                files["element_abundances"]
            ),
            "fastchem_gas_equilibrium_table": str(files["gas_equilibrium"]),
            "fastchem_gas_equilibrium_sha256": _sha256_file(files["gas_equilibrium"]),
            "fastchem_condensate_equilibrium_table": str(
                files["condensate_equilibrium"]
            ),
            "fastchem_condensate_equilibrium_sha256": _sha256_file(
                files["condensate_equilibrium"]
            ),
            "fastchem_condensation_mode": self.mode,
            "fastchem_condensate_number_density_unit": "cm^-3",
            "fastchem_gas_number_density_unit": "cm^-3",
            "fastchem_species_molecular_weight_unit": "u",
            "fastchem_mass_density_unit": "kg m^-3",
            "fastchem_mass_fraction_unit": "kg/kg",
            "fastchem_condensate_weight_source": condensate_weight_source,
            "fastchem_atomic_mass_unit_kg": f"{_ATOMIC_MASS_UNIT_KG:.14g}",
            "fastchem_condensate_mass_fraction_denominator": (
                "gas_mass_plus_all_fastchem_condensate_mass"
            ),
            "fastchem_condensate_mass_mixing_ratio_denominator": "gas_mass",
            "fastchem_condensate_mass_interpretation": (
                "local_rainout_mass_equivalent_not_cloud_retention"
                if self.mode == "rainout"
                else "local_equilibrium_condensate_mass_equivalent"
            ),
            "fastchem_pressure_closure_relative_tolerance": (
                f"{_FASTCHEM_PRESSURE_CLOSURE_RTOL:.8g}"
            ),
            "fastchem_rainout_input_order": (
                "decreasing_pressure_deep_to_upper"
                if self.mode == "rainout"
                else "input_order_independent_equilibrium_points"
            ),
            "fastchem_output_layer_order": "input_pressure_grid_order",
            "fastchem_selected_condensates": ",".join(
                all_symbols[index] for index in selected_indices
            ),
        }
        return _FastChemRuntime(
            api=api,
            fastchem=fastchem,
            version=version,
            condensate_symbols=tuple(all_symbols[index] for index in selected_indices),
            condensate_names=tuple(all_names[index] for index in selected_indices),
            condensate_indices=selected_indices,
            all_condensate_symbols=all_symbols,
            all_condensate_molecular_weights_u=condensate_weights,
            gas_species_symbols=gas_species_symbols,
            gas_species_molecular_weights_u=gas_species_weights,
            element_symbols=element_symbols,
            metadata=immutable_mapping(metadata),
        )

    def _input_files(self) -> Mapping[str, Path]:
        root = Path(self.fastchem_path).expanduser()
        files = {
            "element_abundances": (
                root / "input" / "element_abundances" / "asplund_2009.dat"
            ),
            "gas_equilibrium": root / "input" / "logK" / "logK.dat",
            "condensate_equilibrium": (
                root / "input" / "logK" / "logK_condensates.dat"
            ),
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise RobertConfigError(
                "FastChem condensate diagnostics require these input files: "
                + ", ".join(missing)
            )
        return files


def _output_int_vector(
    output_data: Any, field_name: str, size: int
) -> NDArray[np.int64]:
    try:
        values = getattr(output_data, field_name)
    except Exception as exc:
        raise RobertValidationError(f"FastChem output is missing {field_name}") from exc
    return _readonly_int_vector(values, field_name, size)


def _output_int_matrix(
    output_data: Any,
    field_name: str,
    shape: tuple[int, int],
) -> NDArray[np.int64]:
    try:
        values = getattr(output_data, field_name)
    except Exception as exc:
        raise RobertValidationError(f"FastChem output is missing {field_name}") from exc
    return _readonly_int_matrix(values, field_name, shape)


def _output_float_matrix(
    output_data: Any,
    field_name: str,
    shape: tuple[int, int],
    *,
    non_negative: bool = False,
    unit_interval: bool = False,
) -> NDArray[np.float64]:
    try:
        values = getattr(output_data, field_name)
    except Exception as exc:
        raise RobertValidationError(f"FastChem output is missing {field_name}") from exc
    return _readonly_float_matrix(
        values,
        field_name,
        shape,
        non_negative=non_negative,
        unit_interval=unit_interval,
    )


__all__ = [
    "FastChemCondensationMode",
    "FastChemCondensateDiagnostics",
    "FastChemCondensateResult",
]
