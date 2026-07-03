"""Plot ROBERT temperature profile families on one pressure grid."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    IsothermalTemperatureProfile,
    MadhusudhanSeager2009TemperatureProfile,
    ParmentierGuillot2014TemperatureProfile,
    PressureGrid,
    SplineTemperatureProfile,
)


def main() -> Path:
    """Create a profile-family diagnostic plot and return its output path."""

    pressure_grid = PressureGrid.logspace(
        min_pressure=1.0e-6,
        max_pressure=100.0,
        n_layers=120,
        unit="bar",
        name="temperature-profile-family diagnostic grid",
    )

    profiles = [
        (
            "Isothermal",
            IsothermalTemperatureProfile(temperature=1600.0),
            {},
        ),
        (
            "Spline",
            SplineTemperatureProfile(
                knot_pressure=np.array([1.0e-6, 1.0e-3, 1.0e-1, 10.0, 100.0]),
                knot_temperature=np.array([1200.0, 1350.0, 1650.0, 2050.0, 2200.0]),
            ),
            {},
        ),
        (
            "Madhusudhan-Seager 2009",
            MadhusudhanSeager2009TemperatureProfile(),
            {
                "P1": -2.0,
                "P2": -3.0,
                "P3": 0.3,
                "T0": 1200.0,
                "alpha1": 0.3,
                "alpha2": 0.3,
            },
        ),
        (
            "Parmentier-Guillot 2014",
            ParmentierGuillot2014TemperatureProfile(
                gravity=4.3,
                internal_temperature=200.0,
            ),
            {
                "kappa_IR": 0.02,
                "gamma1": 0.5,
                "gamma2": 1.5,
                "T_irr": 1500.0,
                "alpha": 0.5,
            },
        ),
    ]

    output_path = Path(__file__).resolve().parent / "outputs" / "temperature_profile_families.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    colors = ["#111111", "#1f77b4", "#2ca02c", "#d62728"]
    for color, (label, profile, parameters) in zip(colors, profiles):
        temperature = profile.evaluate(parameters, pressure_grid)
        ax.plot(
            temperature,
            pressure_grid.centers,
            color=color,
            linewidth=1.8,
            label=label,
        )

    ax.set_title("ROBERT P-T Profile Families")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim(float(max(pressure_grid.centers)), float(min(pressure_grid.centers)))
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    main()
