"""Temperature profile parameterizations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import (
    PressureGrid,
    RobertConfigError,
    RobertDataError,
    RobertValidationError,
)


class TemperatureProfile(Protocol):
    """Protocol shared by temperature profile parameterizations."""

    name: str
    unit: str

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
    ) -> NDArray[np.float64]:
        """Evaluate temperature on a pressure grid."""


def _readonly_1d_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise RobertValidationError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _pressure_unit_scale_to_bar(unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"bar", "bars"}:
        return 1.0
    if normalized in {"pa", "pascal", "pascals"}:
        return 1.0e-5
    if normalized in {"atm", "atmosphere", "atmospheres"}:
        return 1.01325
    raise RobertValidationError(f"unsupported pressure unit: {unit}")


def _convert_pressure(
    pressure: NDArray[np.float64],
    from_unit: str,
    to_unit: str,
) -> NDArray[np.float64]:
    pressure_bar = pressure * _pressure_unit_scale_to_bar(from_unit)
    converted = pressure_bar / _pressure_unit_scale_to_bar(to_unit)
    converted.setflags(write=False)
    return converted


@dataclass(frozen=True)
class IsothermalTemperatureProfile:
    """Layer-constant temperature profile.

    The profile can either hold a fixed temperature or read one parameter from
    the provided parameter mapping.
    """

    temperature: float | None = None
    parameter_name: str = "temperature"
    name: str = "isothermal"
    unit: str = "K"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("temperature profile name must not be empty")
        if not self.unit:
            raise RobertValidationError("temperature unit must not be empty")
        if not self.parameter_name:
            raise RobertValidationError("temperature parameter name must not be empty")
        if self.temperature is not None:
            value = float(self.temperature)
            if not np.isfinite(value) or value <= 0.0:
                raise RobertValidationError("temperature must be finite and positive")
            object.__setattr__(self, "temperature", value)

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        if self.temperature is not None:
            return ()
        return (self.parameter_name,)

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
    ) -> NDArray[np.float64]:
        """Return a positive layer temperature array."""

        if self.temperature is None:
            try:
                value = float(parameters[self.parameter_name])
            except KeyError as exc:
                raise RobertConfigError(
                    f"missing required temperature parameter: {self.parameter_name}"
                ) from exc
        else:
            value = self.temperature

        if not np.isfinite(value) or value <= 0.0:
            raise RobertValidationError("temperature must be finite and positive")

        temperature = np.full(pressure_grid.n_layers, value, dtype=float)
        temperature.setflags(write=False)
        return temperature


@dataclass(frozen=True)
class TabulatedTemperatureProfile:
    """Temperature profile interpolated from tabulated pressure-temperature data.

    Interpolation is performed linearly in `log10(pressure)`. The tabulated
    pressures may be supplied in any order; they are stored internally in
    increasing pressure order.
    """

    pressure: NDArray[np.float64]
    temperature: NDArray[np.float64]
    pressure_unit: str = "bar"
    unit: str = "K"
    name: str = "tabulated"
    extrapolation: str = "raise"
    source_path: Path | None = None
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("temperature profile name must not be empty")
        if not self.unit:
            raise RobertValidationError("temperature unit must not be empty")
        if self.unit != "K":
            raise RobertValidationError("tabulated temperature profiles currently require unit='K'")
        if not self.pressure_unit:
            raise RobertValidationError("pressure_unit must not be empty")
        if self.extrapolation not in {"raise", "clip"}:
            raise RobertValidationError("extrapolation must be 'raise' or 'clip'")

        pressure = _readonly_1d_float_array(self.pressure, "tabulated pressure")
        temperature = _readonly_1d_float_array(self.temperature, "tabulated temperature")
        if pressure.shape != temperature.shape:
            raise RobertValidationError("tabulated pressure and temperature must have matching shapes")
        if np.any(pressure <= 0.0):
            raise RobertValidationError("tabulated pressure values must be positive")
        if np.any(temperature <= 0.0):
            raise RobertValidationError("tabulated temperature values must be positive")

        order = np.argsort(pressure)
        sorted_pressure = np.array(pressure[order], dtype=float, copy=True)
        sorted_temperature = np.array(temperature[order], dtype=float, copy=True)
        if sorted_pressure.size > 1 and np.any(np.diff(sorted_pressure) <= 0.0):
            raise RobertValidationError("tabulated pressure values must be unique")

        sorted_pressure.setflags(write=False)
        sorted_temperature.setflags(write=False)
        object.__setattr__(self, "pressure", sorted_pressure)
        object.__setattr__(self, "temperature", sorted_temperature)
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        pressure_column: str = "pressure_bar",
        temperature_column: str = "temperature_K",
        pressure_unit: str = "bar",
        unit: str = "K",
        name: str = "tabulated",
        extrapolation: str = "raise",
    ) -> "TabulatedTemperatureProfile":
        """Load a tabulated temperature profile from a CSV file."""

        csv_path = Path(path).expanduser()
        if not csv_path.exists():
            raise RobertDataError(f"temperature profile CSV does not exist: {csv_path}")

        pressure_values: list[float] = []
        temperature_values: list[float] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RobertDataError("temperature profile CSV is missing a header row")
            missing = {pressure_column, temperature_column}.difference(reader.fieldnames)
            if missing:
                missing_columns = ", ".join(sorted(missing))
                raise RobertDataError(
                    f"temperature profile CSV is missing required columns: {missing_columns}"
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    pressure_values.append(float(row[pressure_column]))
                    temperature_values.append(float(row[temperature_column]))
                except (TypeError, ValueError) as exc:
                    raise RobertDataError(
                        f"invalid numeric value in temperature profile CSV row {row_number}"
                    ) from exc

        return cls(
            pressure=np.asarray(pressure_values, dtype=float),
            temperature=np.asarray(temperature_values, dtype=float),
            pressure_unit=pressure_unit,
            unit=unit,
            name=name,
            extrapolation=extrapolation,
            source_path=csv_path,
            metadata={
                "pressure_column": pressure_column,
                "temperature_column": temperature_column,
                "interpolation": "linear in log10(pressure)",
                "extrapolation": extrapolation,
            },
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        return ()

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
    ) -> NDArray[np.float64]:
        """Interpolate the tabulated profile onto pressure-grid layer centers."""

        source_pressure = _convert_pressure(
            self.pressure,
            from_unit=self.pressure_unit,
            to_unit=pressure_grid.unit,
        )
        target_pressure = np.array(pressure_grid.centers, dtype=float, copy=True)
        source_min = float(source_pressure[0])
        source_max = float(source_pressure[-1])

        if self.extrapolation == "raise":
            if np.any(target_pressure < source_min) or np.any(target_pressure > source_max):
                raise RobertValidationError(
                    "pressure grid extends outside tabulated temperature profile coverage"
                )
            interpolation_pressure = target_pressure
        else:
            interpolation_pressure = np.clip(target_pressure, source_min, source_max)

        interpolated = np.interp(
            np.log10(interpolation_pressure),
            np.log10(source_pressure),
            self.temperature,
        )
        if not np.all(np.isfinite(interpolated)) or np.any(interpolated <= 0.0):
            raise RobertValidationError("tabulated temperature interpolation produced invalid values")
        interpolated.setflags(write=False)
        return interpolated
