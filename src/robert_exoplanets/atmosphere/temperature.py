"""Temperature profile parameterizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import PressureGrid, RobertConfigError, RobertValidationError


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
