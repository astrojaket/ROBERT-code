"""Optional FastChem condensate diagnostics.

This module returns raw FastChem condensate number densities and elemental
condensation degrees. It does not convert condensate number densities to mass
fractions or aerosol particle number densities.
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


@dataclass(frozen=True)
class FastChemCondensateResult:
    """Immutable raw condensate diagnostics from one FastChem calculation.

    ``condensate_number_density_cm3`` has shape ``(n_layers, n_condensates)``.
    Its values are FastChem's reported fictitious condensate number densities
    in cm^-3. They are not aerosol particle counts. No mass conversion is
    applied.

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
    element_symbols: tuple[str, ...]
    element_condensation_degree: ArrayLike
    fastchem_flag: ArrayLike
    chemistry_iterations: ArrayLike
    condensation_iterations: ArrayLike
    total_iterations: ArrayLike
    return_flag: int
    metadata: Mapping[str, str] = field(default_factory=dict)

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
        element_symbols = tuple(str(value).strip() for value in self.element_symbols)
        if not condensate_symbols or any(not value for value in condensate_symbols):
            raise RobertValidationError(
                "condensate_symbols must contain non-empty symbols"
            )
        if len(set(condensate_symbols)) != len(condensate_symbols):
            raise RobertValidationError("condensate_symbols must be unique")
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

        number_density = _readonly_float_matrix(
            self.condensate_number_density_cm3,
            "condensate_number_density_cm3",
            (n_layers, len(condensate_symbols)),
            non_negative=True,
        )
        element_degree = _readonly_float_matrix(
            self.element_condensation_degree,
            "element_condensation_degree",
            (n_layers, len(element_symbols)),
            unit_interval=True,
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
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "condensate_symbols", condensate_symbols)
        object.__setattr__(self, "condensate_names", condensate_names)
        object.__setattr__(self, "element_symbols", element_symbols)
        object.__setattr__(self, "condensate_number_density_cm3", number_density)
        object.__setattr__(self, "element_condensation_degree", element_degree)
        object.__setattr__(self, "fastchem_flag", fastchem_flag)
        object.__setattr__(self, "chemistry_iterations", chemistry_iterations)
        object.__setattr__(self, "condensation_iterations", condensation_iterations)
        object.__setattr__(self, "total_iterations", total_iterations)
        object.__setattr__(self, "return_flag", int(self.return_flag))
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class _FastChemRuntime:
    api: Any
    fastchem: Any
    version: str
    condensate_symbols: tuple[str, ...]
    condensate_names: tuple[str, ...]
    condensate_indices: NDArray[np.int64]
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
            element_symbols=runtime.element_symbols,
            element_condensation_degree=element_degree[restore_order],
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
        if any(not value for value in (*all_symbols, *all_names, *element_symbols)):
            raise RobertConfigError(
                "FastChem returned an empty species or element label"
            )
        if len(set(all_symbols)) != len(all_symbols):
            raise RobertConfigError("FastChem returned duplicate condensate symbols")
        if len(set(element_symbols)) != len(element_symbols):
            raise RobertConfigError("FastChem returned duplicate element symbols")

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
