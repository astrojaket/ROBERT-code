"""Plot the external HAT-P-32b emission benchmark against blackbody references."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import load_emission_benchmark_csv

DEFAULT_BENCHMARK_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "HAT-P-32b"
    / "emission"
    / "emission_R1000.csv"
)


def benchmark_csv_path() -> Path:
    """Return the configured external HAT-P-32b benchmark CSV path."""

    configured = os.environ.get("HAT_P_32B_EMISSION_CSV")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_BENCHMARK_CSV


def main() -> Path:
    """Create the HAT-P-32b diagnostic plot and return its output path."""

    csv_path = benchmark_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            "HAT-P-32b benchmark CSV was not found. Set HAT_P_32B_EMISSION_CSV "
            f"or place the file at {DEFAULT_BENCHMARK_CSV}."
        )

    benchmark = load_emission_benchmark_csv(csv_path, name="HAT-P-32b emission R1000")
    output_path = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_benchmark.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(
        benchmark.wavelength_micron,
        benchmark.eclipse_depth * 1.0e6,
        color="#111111",
        linewidth=1.6,
        label="External emission benchmark",
    )

    reference_styles = {
        "bb_1200K": ("#1f77b4", "--", "Blackbody 1200 K"),
        "bb_1500K": ("#d62728", "--", "Blackbody 1500 K"),
    }
    for label, values in benchmark.references.items():
        color, linestyle, display_label = reference_styles.get(
            label,
            ("#666666", ":", label.replace("_", " ")),
        )
        ax.plot(
            benchmark.wavelength_micron,
            values * 1.0e6,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=display_label,
        )

    median_depth_ppm = float(np.median(benchmark.eclipse_depth) * 1.0e6)
    ax.axhline(
        median_depth_ppm,
        color="#2ca02c",
        linewidth=1.3,
        linestyle=":",
        label="Median benchmark depth",
    )

    ax.set_title("HAT-P-32b External Emission Benchmark")
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Eclipse depth [ppm]")
    ax.set_xlim(float(np.min(benchmark.wavelength_micron)), float(np.max(benchmark.wavelength_micron)))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.04,
        f"Source: {csv_path.name}",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#333333",
        ha="right",
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    main()
