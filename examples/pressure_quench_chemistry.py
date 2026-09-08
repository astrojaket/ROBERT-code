"""Run compact molecular and Taylor-grouped pressure-quench examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets import (
    PressureGrid,
    PressureQuenchChemistry,
)


@dataclass(frozen=True)
class AnalyticEquilibriumChemistry:
    """Deterministic example base; this is not a science chemistry solver."""

    profiles: Mapping[str, NDArray[np.float64]]
    name: str = "analytic-example-equilibrium"
    convention: str = "volume_mixing_ratio"
    metadata: Mapping[str, str] = field(
        default_factory=lambda: {"purpose": "runnable-example"}
    )

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def required_parameters(self) -> tuple[str, ...]:
        return ()

    def evaluate(self, parameters, pressure_grid, temperature):
        del parameters, pressure_grid, temperature
        return {
            species: np.array(profile, dtype=float, copy=True)
            for species, profile in self.profiles.items()
        }


def main() -> None:
    pressure = PressureGrid.from_log_centers(1.0e-4, 10.0, 6, unit="bar")
    temperature = np.full(pressure.n_layers, 1200.0)
    base = AnalyticEquilibriumChemistry(
        {
            "H2O": np.geomspace(8.0e-4, 3.0e-4, pressure.n_layers),
            "CO": np.geomspace(2.0e-4, 7.0e-4, pressure.n_layers),
            "CO2": np.geomspace(1.0e-7, 1.0e-4, pressure.n_layers),
            "CH4": np.geomspace(1.0e-8, 1.0e-4, pressure.n_layers),
            "NH3": np.geomspace(1.0e-9, 1.0e-5, pressure.n_layers),
            "H2": np.full(pressure.n_layers, 0.998),
        }
    )

    molecular = PressureQuenchChemistry.molecular(
        base,
        {"CO2": "log_Pq_CO2"},
    )
    molecular_profiles = molecular.evaluate(
        {"log_Pq_CO2": -1.5},
        pressure,
        temperature,
    )

    grouped = PressureQuenchChemistry.taylor_2026_hot_jupiter_element_grouped(
        base
    )
    grouped_profiles = grouped.evaluate(
        {"log_Pq_C": -1.0, "log_Pq_N": -2.0},
        pressure,
        temperature,
    )

    print("Pressure centers (bar):", pressure.centers)
    print("Molecular CO2 quench:", molecular_profiles["CO2"])
    print("Taylor carbon-group CH4 quench:", grouped_profiles["CH4"])
    print("Taylor nitrogen-group NH3 quench:", grouped_profiles["NH3"])


if __name__ == "__main__":
    main()
