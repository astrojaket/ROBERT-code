"""Plot the external HAT-P-32b P-T profile on a ROBERT pressure grid."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from robert_exoplanets import PressureGrid, TabulatedTemperatureProfile

DEFAULT_PT_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "PTprofiles-Teq_1800-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_1.8-TiOVO_false-daysideavg-w_mu_area.csv"
)


def pt_csv_path() -> Path:
    """Return the configured external HAT-P-32b P-T profile path."""

    configured = os.environ.get("HAT_P_32B_PT_CSV")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_PT_CSV


def main() -> Path:
    """Create the HAT-P-32b P-T diagnostic plot and return its output path."""

    csv_path = pt_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            "HAT-P-32b P-T CSV was not found. Set HAT_P_32B_PT_CSV "
            f"or place the file at {DEFAULT_PT_CSV}."
        )

    profile = TabulatedTemperatureProfile.from_csv(csv_path, name="HAT-P-32b external PT")
    pressure_grid = PressureGrid.logspace(
        min_pressure=3.0e-6,
        max_pressure=100.0,
        n_layers=100,
        unit="bar",
        name="HAT-P-32b ROBERT diagnostic grid",
    )
    interpolated_temperature = profile.evaluate({}, pressure_grid)

    output_path = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_pt_profile.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    ax.plot(
        profile.temperature,
        profile.pressure,
        color="#111111",
        linewidth=1.8,
        label="External profile",
    )
    ax.scatter(
        interpolated_temperature,
        pressure_grid.centers,
        s=14,
        color="#1f77b4",
        alpha=0.85,
        label="ROBERT layer centers",
    )

    ax.set_title("HAT-P-32b P-T Profile")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim(float(max(profile.pressure)), float(min(profile.pressure)))
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.98,
        0.03,
        "Source: external HAT-P-32b PT CSV",
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
