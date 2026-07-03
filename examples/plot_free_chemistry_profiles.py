"""Plot ROBERT free-chemistry VMR profiles on one pressure grid."""

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
    BackgroundGasMixture,
    CompositionMeanMolecularWeight,
    FreeChemistry,
    PressureGrid,
)


def main() -> Path:
    """Create a free-chemistry diagnostic plot and return its output path."""

    pressure_grid = PressureGrid.logspace(
        min_pressure=1.0e-6,
        max_pressure=100.0,
        n_layers=100,
        unit="bar",
        name="free-chemistry diagnostic grid",
    )
    chemistry = FreeChemistry(
        active_species=("H2O", "CO", "CO2"),
        fixed_mixing_ratios={"CO2": 1.0e-5},
        parameter_names={"H2O": "log_H2O", "CO": "log_CO"},
        parameter_mode="log10",
        background=BackgroundGasMixture.hydrogen_helium(h2_fraction=0.8547),
    )
    composition = chemistry.evaluate(
        {"log_H2O": -3.0, "log_CO": -4.0},
        pressure_grid,
        np.full(pressure_grid.n_layers, 1500.0),
    )
    mean_molecular_weight = CompositionMeanMolecularWeight().evaluate(
        composition,
        pressure_grid,
    )

    output_path = Path(__file__).resolve().parent / "outputs" / "free_chemistry_profiles.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    colors = {
        "H2O": "#1f77b4",
        "CO": "#2ca02c",
        "CO2": "#d62728",
        "H2": "#111111",
        "He": "#9467bd",
    }
    for species, profile in composition.items():
        ax.plot(
            profile,
            pressure_grid.centers,
            color=colors.get(species, "#555555"),
            linewidth=1.8,
            label=species,
        )

    ax.set_title("ROBERT Free-Chemistry Profiles")
    ax.set_xlabel("Volume Mixing Ratio")
    ax.set_ylabel("Pressure [bar]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-6, 1.0)
    ax.set_ylim(float(max(pressure_grid.centers)), float(min(pressure_grid.centers)))
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.text(
        0.97,
        0.04,
        f"Mean molecular weight: {mean_molecular_weight[0]:.3f} amu",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        ha="right",
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    main()
