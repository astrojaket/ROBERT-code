"""Atmosphere construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from robert_exoplanets.core import PressureGrid

from .chemistry import ConstantChemistry
from .state import AtmosphereState
from .temperature import TemperatureProfile


@dataclass(frozen=True)
class AtmosphereBuilder:
    """Build an `AtmosphereState` from simple v0.3 components."""

    pressure_grid: PressureGrid
    temperature_profile: TemperatureProfile
    chemistry_model: ConstantChemistry
    mean_molecular_weight: float = 2.3

    @property
    def species(self) -> tuple[str, ...]:
        """Species produced by the chemistry model."""

        return self.chemistry_model.species

    def build(self, parameters: Mapping[str, float] | None = None) -> AtmosphereState:
        """Evaluate temperature and chemistry into a complete atmosphere state."""

        parameter_values = {} if parameters is None else parameters
        temperature = self.temperature_profile.evaluate(parameter_values, self.pressure_grid)
        composition = self.chemistry_model.evaluate(
            parameter_values,
            self.pressure_grid,
            temperature,
        )
        return AtmosphereState(
            pressure_grid=self.pressure_grid,
            temperature=temperature,
            composition=composition,
            mean_molecular_weight=self.mean_molecular_weight,
        )
