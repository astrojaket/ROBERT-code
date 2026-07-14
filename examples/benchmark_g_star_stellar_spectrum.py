"""Benchmark a Sun-like PHOENIX spectrum against the blackbody fallback.

Set ``PYSYN_CDBS`` to an STScI Synphot reference-data root containing
``grid/phoenix`` before running this script. The benchmark uses a G2V-like
photosphere (5778 K, log(g)=4.44, solar metallicity), applies the same stellar
normalization used by ROBERT eclipse-depth forward models, and reports the
effect on a 1500 K Jupiter-radius planet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    SpectralGrid,
    Star,
    planck_radiance_wavelength,
    prepare_stellar_spectrum,
)

JUPITER_RADIUS_M = 7.1492e7
SOLAR_RADIUS_M = 6.957e8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/outputs/g_star_stellar_spectrum"),
    )
    parser.add_argument("--bins", type=int, default=1000)
    args = parser.parse_args()
    if args.bins < 10:
        parser.error("--bins must be at least 10")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    edges = np.geomspace(0.3, 20.0, args.bins + 1)
    wavelength = np.sqrt(edges[:-1] * edges[1:])
    grid = SpectralGrid(
        values=wavelength,
        bin_edges=edges,
        unit="micron",
        role="benchmark",
        name="G2V PHOENIX benchmark grid",
    )
    star = Star(
        name="Sun-like G2V benchmark",
        radius_m=SOLAR_RADIUS_M,
        effective_temperature_k=5778.0,
        log_g_cgs=4.44,
        metallicity_dex=0.0,
    )

    started = time.perf_counter()
    phoenix = prepare_stellar_spectrum(star, grid, model="phoenix")
    phoenix_seconds = time.perf_counter() - started
    started = time.perf_counter()
    blackbody = prepare_stellar_spectrum(star, grid, model="blackbody")
    blackbody_seconds = time.perf_counter() - started

    ratio = phoenix.values / blackbody.values
    planet_radiance = planck_radiance_wavelength(wavelength, 1500.0)
    area_ratio = (JUPITER_RADIUS_M / SOLAR_RADIUS_M) ** 2
    phoenix_eclipse = planet_radiance / phoenix.values * area_ratio
    blackbody_eclipse = planet_radiance / blackbody.values * area_ratio
    eclipse_difference_ppm = (phoenix_eclipse - blackbody_eclipse) * 1.0e6

    comparison = (wavelength >= 0.5) & (wavelength <= 15.0)
    jwst = (wavelength >= 2.4) & (wavelength <= 12.0)
    max_difference_index = np.flatnonzero(jwst)[
        np.argmax(np.abs(eclipse_difference_ppm[jwst]))
    ]
    report = {
        "schema_version": 1,
        "benchmark": "sun_like_g2v_phoenix_vs_blackbody",
        "star": {
            "effective_temperature_k": 5778.0,
            "log_g_cgs": 4.44,
            "metallicity_dex": 0.0,
            "radius_m": SOLAR_RADIUS_M,
        },
        "planet_reference": {
            "temperature_k": 1500.0,
            "radius_m": JUPITER_RADIUS_M,
        },
        "spectral_grid": {
            "minimum_micron": float(edges[0]),
            "maximum_micron": float(edges[-1]),
            "bins": args.bins,
            "sampling": "logarithmic_flux_conserving_bins",
        },
        "timing_seconds": {
            "phoenix_prepare": phoenix_seconds,
            "blackbody_prepare": blackbody_seconds,
        },
        "metrics": {
            "phoenix_to_blackbody_ratio_min_0p5_15um": float(
                np.min(ratio[comparison])
            ),
            "phoenix_to_blackbody_ratio_max_0p5_15um": float(
                np.max(ratio[comparison])
            ),
            "phoenix_to_blackbody_rms_fraction_0p5_15um": float(
                np.sqrt(np.mean(np.square(ratio[comparison] - 1.0)))
            ),
            "maximum_absolute_eclipse_difference_ppm_2p4_12um": float(
                np.max(np.abs(eclipse_difference_ppm[jwst]))
            ),
            "median_absolute_eclipse_difference_ppm_2p4_12um": float(
                np.median(np.abs(eclipse_difference_ppm[jwst]))
            ),
            "wavelength_of_maximum_eclipse_difference_micron": float(
                wavelength[max_difference_index]
            ),
        },
        "phoenix_metadata": dict(phoenix.metadata),
    }
    report_path = output / "g_star_stellar_spectrum_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    axes[0].loglog(
        wavelength,
        np.pi * phoenix.values,
        label="PHOENIX G2V surface flux",
        linewidth=1.0,
    )
    axes[0].loglog(
        wavelength,
        np.pi * blackbody.values,
        label="5778 K blackbody",
        linewidth=1.0,
        linestyle="--",
    )
    axes[0].set_ylabel(r"$F_\lambda$ [W m$^{-3}$]")
    axes[0].legend(frameon=False)
    axes[1].semilogx(wavelength, ratio, linewidth=0.9)
    axes[1].axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("PHOENIX / blackbody")
    axes[2].semilogx(wavelength, eclipse_difference_ppm, linewidth=0.9)
    axes[2].axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Eclipse change [ppm]")
    axes[2].set_xlabel("Wavelength [micron]")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
        axis.set_xlim(edges[0], edges[-1])
    figure.suptitle("Sun-like G-star stellar-spectrum benchmark")
    figure.tight_layout()
    figure.savefig(output / "g_star_stellar_spectrum_benchmark.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
