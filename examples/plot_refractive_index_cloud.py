"""Inspect a selectable optical-constant material and its Mie cloud optics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    OpticalConstantsCatalog,
    SpectralGrid,
    lognormal_mie_optics,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "optical_constants" / "exo_skryer"
OUTPUT = Path(__file__).resolve().parent / "outputs" / "refractive_index_cloud"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="MgSiO3")
    parser.add_argument("--radius-micron", type=float, default=0.3)
    parser.add_argument("--geometric-stddev", type=float, default=1.5)
    parser.add_argument("--density-kg-m3", type=float, default=3200.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    catalog = OpticalConstantsCatalog(CATALOG)
    refractive_index = catalog.load(args.material)
    lower = max(0.3, float(refractive_index.wavelength_micron[0]))
    upper = min(30.0, float(refractive_index.wavelength_micron[-1]))
    wavelength = np.geomspace(lower, upper, 300)
    spectral_grid = SpectralGrid.from_array(
        wavelength,
        unit="micron",
        role="cloud_optical_properties",
        name=f"{args.material} cloud grid",
    )
    optics = lognormal_mie_optics(
        refractive_index,
        spectral_grid,
        effective_radius_micron=args.radius_micron,
        geometric_stddev=args.geometric_stddev,
        particle_density_kg_m3=args.density_kg_m3,
        quadrature_points=16,
    )
    complex_index = refractive_index.evaluate(wavelength)

    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].semilogx(wavelength, complex_index.real)
    axes[0, 0].set_ylabel("Real index n")
    axes[0, 1].loglog(wavelength, complex_index.imag)
    axes[0, 1].set_ylabel("Imaginary index k")
    axes[1, 0].loglog(wavelength, optics.mass_extinction_m2_kg, label="extinction")
    axes[1, 0].loglog(wavelength, optics.mass_scattering_m2_kg, label="scattering")
    axes[1, 0].set_ylabel(r"Mass coefficient (m$^2$ kg$^{-1}$)")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].semilogx(wavelength, optics.single_scattering_albedo, label="single-scattering albedo")
    axes[1, 1].semilogx(wavelength, optics.asymmetry_factor, label="asymmetry")
    axes[1, 1].set_ylabel("Dimensionless")
    axes[1, 1].legend(frameon=False)
    for axis in axes[1]:
        axis.set_xlabel("Wavelength (micron)")
    figure.suptitle(
        f"{args.material}: r_eff={args.radius_micron:g} micron, "
        f"sigma_g={args.geometric_stddev:g}"
    )
    figure.tight_layout()
    path = args.output / f"{args.material}_mie_cloud.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Available materials: {', '.join(catalog.materials)}")
    print(f"Source SHA-256: {refractive_index.metadata['source_sha256']}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
