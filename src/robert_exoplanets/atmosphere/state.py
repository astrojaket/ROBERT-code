"""Atmospheric state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import PressureGrid, RobertValidationError


def _readonly_layer_array(
    values: ArrayLike,
    name: str,
    n_layers: int,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim == 0:
        array = np.full(n_layers, float(array), dtype=float)
    if array.ndim != 1:
        raise RobertValidationError(f"{name} must be one-dimensional")
    if array.shape != (n_layers,):
        raise RobertValidationError(f"{name} must match pressure grid layers")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AtmosphereState:
    """Evaluated one-dimensional atmospheric state for one parameter vector."""

    pressure_grid: PressureGrid
    temperature: NDArray[np.float64]
    composition: Mapping[str, ArrayLike]
    mean_molecular_weight: ArrayLike
    temperature_unit: str = "K"
    composition_convention: str = "volume_mixing_ratio"
    mean_molecular_weight_unit: str = "amu"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_layers = self.pressure_grid.n_layers
        temperature = _readonly_layer_array(self.temperature, "temperature", n_layers)
        if np.any(temperature <= 0.0):
            raise RobertValidationError("temperature values must be positive")

        composition: dict[str, NDArray[np.float64]] = {}
        for species, values in self.composition.items():
            if not species:
                raise RobertValidationError("composition species names must not be empty")
            profile = _readonly_layer_array(values, f"{species} composition", n_layers)
            if np.any(profile < 0.0):
                raise RobertValidationError("composition values must be non-negative")
            composition[str(species)] = profile
        if not composition:
            raise RobertValidationError("composition must contain at least one species")

        mean_molecular_weight = _readonly_layer_array(
            self.mean_molecular_weight,
            "mean molecular weight",
            n_layers,
        )
        if np.any(mean_molecular_weight <= 0.0):
            raise RobertValidationError("mean molecular weight values must be positive")
        if not self.temperature_unit:
            raise RobertValidationError("temperature_unit must not be empty")
        if not self.composition_convention:
            raise RobertValidationError("composition_convention must not be empty")
        if not self.mean_molecular_weight_unit:
            raise RobertValidationError("mean_molecular_weight_unit must not be empty")

        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "composition", composition)
        object.__setattr__(self, "mean_molecular_weight", mean_molecular_weight)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_layers(self) -> int:
        """Number of atmospheric layers."""

        return self.pressure_grid.n_layers

    @property
    def species(self) -> tuple[str, ...]:
        """Composition species in insertion order."""

        return tuple(self.composition)
