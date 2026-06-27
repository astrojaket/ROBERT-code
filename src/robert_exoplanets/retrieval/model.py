"""Placeholder emission model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EmissionModel:
    """Stub emission model for wiring retrieval workflows.

    The current implementation returns a deterministic smooth spectrum that is
    useful for tests and examples only. It is not a physical emission model.
    """

    name: str = "stub-gray-emission"

    def evaluate(
        self,
        wavelength: NDArray[np.float64],
        parameters: Mapping[str, float],
    ) -> NDArray[np.float64]:
        """Return a deterministic placeholder spectrum."""

        baseline = float(parameters.get("baseline", 1.0))
        slope = float(parameters.get("slope", 0.0))
        centered_wavelength = wavelength - np.mean(wavelength)
        return baseline + slope * centered_wavelength

