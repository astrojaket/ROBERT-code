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


def _parameter_value(
    parameters: Mapping[str, float],
    parameter_name: str,
    *,
    context: str,
) -> float:
    try:
        value = float(parameters[parameter_name])
    except KeyError as exc:
        raise RobertConfigError(f"missing required {context} parameter: {parameter_name}") from exc
    except (TypeError, ValueError) as exc:
        raise RobertValidationError(f"{context} parameter must be numeric: {parameter_name}") from exc
    if not np.isfinite(value):
        raise RobertValidationError(f"{context} parameter must be finite: {parameter_name}")
    return value


def _validate_parameter_name(parameter_name: str, *, context: str) -> None:
    if not parameter_name:
        raise RobertValidationError(f"{context} parameter name must not be empty")


def _validate_temperature_array(
    temperature: NDArray[np.float64],
    *,
    context: str,
) -> NDArray[np.float64]:
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise RobertValidationError(f"{context} produced invalid temperatures")
    temperature.setflags(write=False)
    return temperature


def _natural_cubic_spline_interpolate(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate a natural cubic spline through increasing one-dimensional knots."""

    if x.size == 2:
        return np.interp(target, x, y)

    spacing = np.diff(x)
    matrix = np.zeros((x.size, x.size), dtype=float)
    rhs = np.zeros(x.size, dtype=float)
    matrix[0, 0] = 1.0
    matrix[-1, -1] = 1.0
    for index in range(1, x.size - 1):
        matrix[index, index - 1] = spacing[index - 1]
        matrix[index, index] = 2.0 * (spacing[index - 1] + spacing[index])
        matrix[index, index + 1] = spacing[index]
        rhs[index] = 6.0 * (
            (y[index + 1] - y[index]) / spacing[index]
            - (y[index] - y[index - 1]) / spacing[index - 1]
        )
    second_derivatives = np.linalg.solve(matrix, rhs)

    interval = np.searchsorted(x, target, side="right") - 1
    interval = np.clip(interval, 0, x.size - 2)
    h = x[interval + 1] - x[interval]
    left = x[interval + 1] - target
    right = target - x[interval]
    return (
        second_derivatives[interval] * left**3 / (6.0 * h)
        + second_derivatives[interval + 1] * right**3 / (6.0 * h)
        + (y[interval] - second_derivatives[interval] * h**2 / 6.0) * left / h
        + (y[interval + 1] - second_derivatives[interval + 1] * h**2 / 6.0) * right / h
    )


_E2_NODES, _E2_WEIGHTS = np.polynomial.legendre.leggauss(64)
_E2_QUADRATURE_POINTS = 0.5 * (_E2_NODES + 1.0)
_E2_QUADRATURE_WEIGHTS = 0.5 * _E2_WEIGHTS


def _exponential_integral_e2(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate E2(x) with fixed Gauss-Legendre quadrature on [0, 1]."""

    x = np.asarray(values, dtype=float)
    if np.any(x < 0.0):
        raise RobertValidationError("E2 arguments must be non-negative")
    exponent = -x[:, np.newaxis] / _E2_QUADRATURE_POINTS[np.newaxis, :]
    with np.errstate(under="ignore"):
        integrand = np.exp(exponent)
    result = np.sum(integrand * _E2_QUADRATURE_WEIGHTS[np.newaxis, :], axis=1)
    return np.asarray(result, dtype=float)


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


@dataclass(frozen=True)
class SplineTemperatureProfile:
    """Natural-cubic spline temperature profile in log10 pressure.

    The spline knots can either hold fixed temperatures or read one retrieval
    parameter per knot. Knot pressures may be supplied in any order and are
    stored internally in increasing pressure order.
    """

    knot_pressure: NDArray[np.float64]
    knot_temperature: NDArray[np.float64] | None = None
    parameter_names: tuple[str, ...] | None = None
    pressure_unit: str = "bar"
    unit: str = "K"
    name: str = "spline"
    extrapolation: str = "raise"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("temperature profile name must not be empty")
        if self.unit != "K":
            raise RobertValidationError("spline temperature profiles currently require unit='K'")
        if not self.pressure_unit:
            raise RobertValidationError("pressure_unit must not be empty")
        _pressure_unit_scale_to_bar(self.pressure_unit)
        if self.extrapolation not in {"raise", "clip"}:
            raise RobertValidationError("extrapolation must be 'raise' or 'clip'")

        pressure = _readonly_1d_float_array(self.knot_pressure, "spline knot pressure")
        if pressure.size < 2:
            raise RobertValidationError("spline profiles require at least two pressure knots")
        if np.any(pressure <= 0.0):
            raise RobertValidationError("spline knot pressure values must be positive")

        order = np.argsort(pressure)
        sorted_pressure = np.array(pressure[order], dtype=float, copy=True)
        if np.any(np.diff(sorted_pressure) <= 0.0):
            raise RobertValidationError("spline knot pressure values must be unique")

        if self.knot_temperature is not None:
            temperature = _readonly_1d_float_array(self.knot_temperature, "spline knot temperature")
            if temperature.shape != pressure.shape:
                raise RobertValidationError(
                    "spline knot pressure and temperature must have matching shapes"
                )
            if np.any(temperature <= 0.0):
                raise RobertValidationError("spline knot temperature values must be positive")
            sorted_temperature = np.array(temperature[order], dtype=float, copy=True)
            sorted_temperature.setflags(write=False)
            names = ()
        else:
            if self.parameter_names is None:
                names = tuple(f"temperature_{index}" for index in range(sorted_pressure.size))
            else:
                input_names = tuple(self.parameter_names)
                if len(input_names) != pressure.size:
                    raise RobertValidationError(
                        "spline parameter_names must match the number of pressure knots"
                    )
                for parameter_name in input_names:
                    _validate_parameter_name(parameter_name, context="spline temperature")
                if len(set(input_names)) != len(input_names):
                    raise RobertValidationError("spline parameter_names must be unique")
                names = tuple(input_names[int(index)] for index in order)
            sorted_temperature = None

        sorted_pressure.setflags(write=False)
        object.__setattr__(self, "knot_pressure", sorted_pressure)
        object.__setattr__(self, "knot_temperature", sorted_temperature)
        object.__setattr__(self, "parameter_names", names)

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        if self.knot_temperature is not None:
            return ()
        return tuple(self.parameter_names or ())

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
    ) -> NDArray[np.float64]:
        """Interpolate the spline onto pressure-grid layer centers."""

        source_pressure = _convert_pressure(
            self.knot_pressure,
            from_unit=self.pressure_unit,
            to_unit=pressure_grid.unit,
        )
        target_pressure = np.array(pressure_grid.centers, dtype=float, copy=True)
        source_min = float(source_pressure[0])
        source_max = float(source_pressure[-1])

        if self.extrapolation == "raise":
            if np.any(target_pressure < source_min) or np.any(target_pressure > source_max):
                raise RobertValidationError(
                    "pressure grid extends outside spline temperature profile coverage"
                )
            interpolation_pressure = target_pressure
        else:
            interpolation_pressure = np.clip(target_pressure, source_min, source_max)

        if self.knot_temperature is None:
            knot_temperature = np.asarray(
                [
                    _parameter_value(
                        parameters,
                        parameter_name,
                        context="spline temperature",
                    )
                    for parameter_name in self.parameter_names or ()
                ],
                dtype=float,
            )
            if np.any(knot_temperature <= 0.0):
                raise RobertValidationError("spline temperature parameters must be positive")
        else:
            knot_temperature = self.knot_temperature

        interpolated = _natural_cubic_spline_interpolate(
            np.log10(source_pressure),
            knot_temperature,
            np.log10(interpolation_pressure),
        )
        return _validate_temperature_array(interpolated, context="spline temperature profile")


@dataclass(frozen=True)
class MadhusudhanSeager2009TemperatureProfile:
    """Madhusudhan & Seager (2009) piecewise parametric P-T profile.

    The transition-pressure parameters are interpreted as log10 pressure in
    `pressure_unit`, matching the retrieval convention used by the HAT-P-32b
    NemesisPy emission example.
    """

    pressure_unit: str = "bar"
    reference_pressure: float | None = None
    p1_parameter_name: str = "P1"
    p2_parameter_name: str = "P2"
    p3_parameter_name: str = "P3"
    t0_parameter_name: str = "T0"
    alpha1_parameter_name: str = "alpha1"
    alpha2_parameter_name: str = "alpha2"
    name: str = "madhusudhan_seager_2009"
    unit: str = "K"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("temperature profile name must not be empty")
        if self.unit != "K":
            raise RobertValidationError(
                "Madhusudhan-Seager temperature profiles currently require unit='K'"
            )
        if not self.pressure_unit:
            raise RobertValidationError("pressure_unit must not be empty")
        _pressure_unit_scale_to_bar(self.pressure_unit)
        for parameter_name in self.required_parameters():
            _validate_parameter_name(parameter_name, context="Madhusudhan-Seager temperature")
        if self.reference_pressure is not None:
            reference_pressure = float(self.reference_pressure)
            if not np.isfinite(reference_pressure) or reference_pressure <= 0.0:
                raise RobertValidationError("reference_pressure must be finite and positive")
            object.__setattr__(self, "reference_pressure", reference_pressure)

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        return (
            self.p1_parameter_name,
            self.p2_parameter_name,
            self.p3_parameter_name,
            self.t0_parameter_name,
            self.alpha1_parameter_name,
            self.alpha2_parameter_name,
        )

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
    ) -> NDArray[np.float64]:
        """Evaluate the piecewise Madhusudhan-Seager profile."""

        pressure = _convert_pressure(
            np.array(pressure_grid.centers, dtype=float, copy=True),
            from_unit=pressure_grid.unit,
            to_unit=self.pressure_unit,
        )
        p0 = float(np.min(pressure) if self.reference_pressure is None else self.reference_pressure)
        p1 = 10.0 ** _parameter_value(
            parameters,
            self.p1_parameter_name,
            context="Madhusudhan-Seager temperature",
        )
        p2 = 10.0 ** _parameter_value(
            parameters,
            self.p2_parameter_name,
            context="Madhusudhan-Seager temperature",
        )
        p3 = 10.0 ** _parameter_value(
            parameters,
            self.p3_parameter_name,
            context="Madhusudhan-Seager temperature",
        )
        t0 = _parameter_value(
            parameters,
            self.t0_parameter_name,
            context="Madhusudhan-Seager temperature",
        )
        alpha1 = _parameter_value(
            parameters,
            self.alpha1_parameter_name,
            context="Madhusudhan-Seager temperature",
        )
        alpha2 = _parameter_value(
            parameters,
            self.alpha2_parameter_name,
            context="Madhusudhan-Seager temperature",
        )

        if p1 <= p0:
            raise RobertValidationError("P1 must be deeper than the profile reference pressure")
        if p3 <= p1:
            raise RobertValidationError("P3 must be deeper than P1")
        if p2 <= 0.0:
            raise RobertValidationError("P2 must be positive after log-pressure conversion")
        if t0 <= 0.0:
            raise RobertValidationError("T0 must be positive")
        if alpha1 <= 0.0 or alpha2 <= 0.0:
            raise RobertValidationError("alpha1 and alpha2 must be positive")

        t2 = ((1.0 / alpha1) * np.log10(p1 / p0)) ** 2
        t2 -= ((1.0 / alpha2) * np.log10(p1 / p2)) ** 2
        t2 += t0
        t3 = ((1.0 / alpha2) * np.log10(p3 / p2)) ** 2 + t2
        middle_temperature = ((1.0 / alpha2) * np.log10(pressure / p2)) ** 2 + t2
        upper_temperature = ((1.0 / alpha1) * np.log10(pressure / p0)) ** 2 + t0
        temperature = np.where(
            pressure >= p3,
            t3,
            np.where(pressure >= p1, middle_temperature, upper_temperature),
        )
        return _validate_temperature_array(
            np.asarray(temperature, dtype=float),
            context="Madhusudhan-Seager temperature profile",
        )


@dataclass(frozen=True)
class ParmentierGuillot2014TemperatureProfile:
    """Dual-visible-channel irradiated P-T profile used for PG14-style retrievals.

    This implementation follows the compact Parmentier & Guillot style form
    used by the local NemesisPy HAT-P-32b temperature engine: pressure is
    converted to Pa, `kappa_IR` is interpreted as m2 kg-1, and optical depth is
    `tau = kappa_IR * pressure / gravity`.
    """

    gravity: float | None = None
    internal_temperature: float | None = 0.0
    kappa_ir_parameter_name: str = "kappa_IR"
    gamma1_parameter_name: str = "gamma1"
    gamma2_parameter_name: str = "gamma2"
    irradiation_temperature_parameter_name: str = "T_irr"
    alpha_parameter_name: str = "alpha"
    gravity_parameter_name: str = "gravity"
    internal_temperature_parameter_name: str = "T_int"
    name: str = "parmentier_guillot_2014"
    unit: str = "K"

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("temperature profile name must not be empty")
        if self.unit != "K":
            raise RobertValidationError(
                "Parmentier-Guillot temperature profiles currently require unit='K'"
            )
        for parameter_name in self.required_parameters():
            _validate_parameter_name(parameter_name, context="Parmentier-Guillot temperature")
        if self.gravity is not None:
            gravity = float(self.gravity)
            if not np.isfinite(gravity) or gravity <= 0.0:
                raise RobertValidationError("gravity must be finite and positive")
            object.__setattr__(self, "gravity", gravity)
        if self.internal_temperature is not None:
            internal_temperature = float(self.internal_temperature)
            if not np.isfinite(internal_temperature) or internal_temperature < 0.0:
                raise RobertValidationError("internal_temperature must be finite and non-negative")
            object.__setattr__(self, "internal_temperature", internal_temperature)

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        names = [
            self.kappa_ir_parameter_name,
            self.gamma1_parameter_name,
            self.gamma2_parameter_name,
            self.irradiation_temperature_parameter_name,
            self.alpha_parameter_name,
        ]
        if self.gravity is None:
            names.append(self.gravity_parameter_name)
        if self.internal_temperature is None:
            names.append(self.internal_temperature_parameter_name)
        return tuple(names)

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
    ) -> NDArray[np.float64]:
        """Evaluate the dual-visible-channel irradiated temperature profile."""

        kappa_ir = _parameter_value(
            parameters,
            self.kappa_ir_parameter_name,
            context="Parmentier-Guillot temperature",
        )
        gamma1 = _parameter_value(
            parameters,
            self.gamma1_parameter_name,
            context="Parmentier-Guillot temperature",
        )
        gamma2 = _parameter_value(
            parameters,
            self.gamma2_parameter_name,
            context="Parmentier-Guillot temperature",
        )
        irradiation_temperature = _parameter_value(
            parameters,
            self.irradiation_temperature_parameter_name,
            context="Parmentier-Guillot temperature",
        )
        alpha = _parameter_value(
            parameters,
            self.alpha_parameter_name,
            context="Parmentier-Guillot temperature",
        )
        gravity = (
            float(self.gravity)
            if self.gravity is not None
            else _parameter_value(
                parameters,
                self.gravity_parameter_name,
                context="Parmentier-Guillot temperature",
            )
        )
        internal_temperature = (
            float(self.internal_temperature)
            if self.internal_temperature is not None
            else _parameter_value(
                parameters,
                self.internal_temperature_parameter_name,
                context="Parmentier-Guillot temperature",
            )
        )

        if kappa_ir <= 0.0:
            raise RobertValidationError("kappa_IR must be positive")
        if gamma1 <= 0.0 or gamma2 <= 0.0:
            raise RobertValidationError("gamma1 and gamma2 must be positive")
        if irradiation_temperature <= 0.0:
            raise RobertValidationError("T_irr must be positive")
        if internal_temperature < 0.0:
            raise RobertValidationError("T_int must be non-negative")
        if gravity <= 0.0:
            raise RobertValidationError("gravity must be positive")
        if alpha < 0.0 or alpha > 1.0:
            raise RobertValidationError("alpha must be between 0 and 1")

        pressure_pa = _convert_pressure(
            np.array(pressure_grid.centers, dtype=float, copy=True),
            from_unit=pressure_grid.unit,
            to_unit="Pa",
        )
        tau = kappa_ir * pressure_pa / gravity
        eta1 = _parmentier_guillot_eta(gamma1, tau)
        eta2 = _parmentier_guillot_eta(gamma2, tau)
        temperature_fourth = (3.0 * internal_temperature**4 / 4.0) * (2.0 / 3.0 + tau)
        temperature_fourth += (3.0 * irradiation_temperature**4 / 4.0) * (
            (1.0 - alpha) * eta1 + alpha * eta2
        )
        if not np.all(np.isfinite(temperature_fourth)) or np.any(temperature_fourth <= 0.0):
            raise RobertValidationError("Parmentier-Guillot profile produced invalid T^4 values")
        temperature = np.power(temperature_fourth, 0.25)
        return _validate_temperature_array(
            np.asarray(temperature, dtype=float),
            context="Parmentier-Guillot temperature profile",
        )


def _parmentier_guillot_eta(gamma: float, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    argument = gamma * tau
    return (
        2.0 / 3.0
        + 2.0 / (3.0 * gamma) * (1.0 + (argument / 2.0 - 1.0) * np.exp(-argument))
        + (2.0 * gamma / 3.0)
        * (1.0 - 0.5 * tau * tau)
        * _exponential_integral_e2(argument)
    )
