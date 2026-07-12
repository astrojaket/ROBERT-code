"""Plot the published native WASP-69b eclipse spectrum by dataset."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from robert_exoplanets import load_schlawin2024_wasp69b

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wasp69b_schlawin2024"
OUTPUT = Path(__file__).resolve().parent / "outputs" / "wasp69b_schlawin2024_data.png"


def main() -> None:
    collection = load_schlawin2024_wasp69b(DATA)
    colors = {
        "f322w2": "#4c78a8",
        "avg": "#72b7b2",
        "f444w": "#f58518",
        "lrs": "#e45756",
    }
    labels = {
        "f322w2": "NIRCam/F322W2",
        "avg": "NIRCam overlap average",
        "f444w": "NIRCam/F444W",
        "lrs": "MIRI/LRS",
    }
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for dataset in collection.datasets:
        observation = dataset.observation
        axis.errorbar(
            observation.wavelength,
            observation.flux * 1.0e6,
            yerr=observation.uncertainty * 1.0e6,
            fmt="o",
            ms=2.6 if dataset.name != "lrs" else 4.0,
            lw=0.6,
            capsize=0,
            color=colors[dataset.name],
            label=f"{labels[dataset.name]} ({observation.n_points})",
        )
    axis.set(
        xlabel="Wavelength [micron]",
        ylabel="Eclipse depth [ppm]",
        title="WASP-69b published JWST emission spectrum (280 native bins)",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
