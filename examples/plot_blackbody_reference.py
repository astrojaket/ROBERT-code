"""Plot a blackbody eclipse-depth reference curve."""

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
    blackbody_eclipse_depth,
    blackbody_eclipse_depth_spectrum,
)

R_JUPITER_M = 7.1492e7
R_SUN_M = 6.957e8

def main() -> Path:
    """Create the diagnostic blackbody plot and return its path."""

    output_path = Path(__file__).resolve().parent / "outputs" / "blackbody_reference.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wavelength_micron = np.linspace(1.0, 15.0, 500)
    observed_wavelength_micron = np.linspace(5.0, 12.0, 8)

    planet_temperature_k = 1800.0
    star_temperature_k = 6200.0
    planet_radius_m = 1.8 * R_JUPITER_M
    star_radius_m = 1.2 * R_SUN_M

    blackbody = blackbody_eclipse_depth_spectrum(
        wavelength_micron,
        planet_temperature_k=planet_temperature_k,
        star_temperature_k=star_temperature_k,
        planet_radius_m=planet_radius_m,
        star_radius_m=star_radius_m,
    )
    observed_reference = blackbody_eclipse_depth(
        observed_wavelength_micron,
        planet_temperature_k=planet_temperature_k,
        star_temperature_k=star_temperature_k,
        planet_radius_m=planet_radius_m,
        star_radius_m=star_radius_m,
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(
        blackbody.spectral_grid.values,
        blackbody.values * 1.0e6,
        color="#1f77b4",
        linewidth=2.2,
        label="Blackbody eclipse estimate",
    )
    ax.scatter(
        observed_wavelength_micron,
        observed_reference * 1.0e6,
        color="#111111",
        s=34,
        zorder=3,
        label="Observation grid samples",
    )

    ax.set_title("Blackbody Reference for a Hot-Jupiter Emission Spectrum")
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("Eclipse depth [ppm]")
    ax.set_xlim(1.0, 15.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.04,
        "Fiducial: T_p=1800 K, T_star=6200 K, R_p=1.8 R_J, R_star=1.2 R_sun",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#333333",
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    main()
