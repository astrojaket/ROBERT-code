"""Simple chemistry parameterizations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import PressureGrid, RobertValidationError


@dataclass(frozen=True)
class ConstantChemistry:
    """Layer-constant trace composition.

    This model repeats declared mixing ratios through every atmospheric layer.
    It is a fixture-style chemistry model for wiring tests and examples, not an
    equilibrium or disequilibrium chemistry calculation.
    """

    mixing_ratios: Mapping[str, float]
    name: str = "constant-chemistry"
    convention: str = "volume_mixing_ratio"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mixing_ratios:
            raise RobertValidationError("mixing_ratios must contain at least one species")
        if not self.name:
            raise RobertValidationError("chemistry name must not be empty")
        if not self.convention:
            raise RobertValidationError("composition convention must not be empty")

        ratios: dict[str, float] = {}
        for species, value in self.mixing_ratios.items():
            if not species:
                raise RobertValidationError("species names must not be empty")
            ratio = float(value)
            if not np.isfinite(ratio) or ratio < 0.0:
                raise RobertValidationError("mixing ratios must be finite and non-negative")
            ratios[str(species)] = ratio

        if self.convention == "volume_mixing_ratio" and sum(ratios.values()) > 1.0:
            raise RobertValidationError("volume mixing ratios must sum to no more than one")

        object.__setattr__(self, "mixing_ratios", ratios)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def species(self) -> tuple[str, ...]:
        """Species produced by this chemistry model."""

        return tuple(self.mixing_ratios)

    def required_parameters(self) -> tuple[str, ...]:
        """Return required parameter names."""

        return ()

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
        temperature: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Return layer-constant composition profiles."""

        if temperature.shape != pressure_grid.centers.shape:
            raise RobertValidationError("temperature must match pressure grid layers")

        profiles: dict[str, NDArray[np.float64]] = {}
        for species, ratio in self.mixing_ratios.items():
            profile = np.full(pressure_grid.n_layers, ratio, dtype=float)
            profile.setflags(write=False)
            profiles[species] = profile
        return profiles
