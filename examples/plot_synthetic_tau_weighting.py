"""Plot synthetic gas optical-depth and weighting diagnostics."""

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
    AtmosphereState,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
)


def main() -> Path:
    """Create a synthetic gas-tau diagnostic plot and return its output path."""

    pressure_grid = PressureGrid.logspace(
        min_pressure=1.0e-5,
        max_pressure=10.0,
        n_layers=56,
        unit="bar",
        name="synthetic tau diagnostic grid",
    )
    pressure_scale = (
        np.log10(pressure_grid.centers) - np.log10(np.min(pressure_grid.centers))
    ) / (
        np.log10(np.max(pressure_grid.centers)) - np.log10(np.min(pressure_grid.centers))
    )
    temperature = 900.0 + 850.0 * pressure_scale**0.75

    spectral_grid, table = _synthetic_h2o_table(pressure_grid, temperature)
    provider = CorrelatedKOpacityProvider({"H2O": table})
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        composition={"H2O": np.full(pressure_grid.n_layers, 8.0e-4)},
        mean_molecular_weight=np.full(pressure_grid.n_layers, 2.3),
        metadata={"example": "synthetic tau weighting"},
    )
    prepared = provider.prepare(spectral_grid, pressure_grid, species=("H2O",))
    evaluated = provider.evaluate(atmosphere, prepared)
    gas_tau = assemble_gas_optical_depth(atmosphere, evaluated, gravity_m_s2=9.0)

    wavelength = spectral_grid.values
    pressure = pressure_grid.centers
    layer_tau = gas_tau.g_weighted_layer_tau()
    cumulative_tau = gas_tau.g_weighted_cumulative_tau_from_top()
    weighting = gas_tau.layer_transmission_weighting(normalize=True)
    mean_weighting = np.mean(weighting, axis=1)
    if np.max(mean_weighting) > 0.0:
        mean_weighting = mean_weighting / np.max(mean_weighting)

    output_path = Path(__file__).resolve().parent / "outputs" / "synthetic_tau_weighting.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_tau, ax_profile) = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    mesh = ax_tau.pcolormesh(
        wavelength,
        pressure,
        np.log10(np.maximum(layer_tau, 1.0e-300)),
        shading="auto",
        cmap="magma",
    )
    if np.nanmin(cumulative_tau) <= 1.0 <= np.nanmax(cumulative_tau):
        ax_tau.contour(
            wavelength,
            pressure,
            np.log10(np.maximum(cumulative_tau, 1.0e-300)),
            levels=[0.0],
            colors="white",
            linewidths=1.2,
        )
    ax_tau.set_title("Layer Gas Optical Depth")
    ax_tau.set_xlabel("Wavelength [micron]")
    ax_tau.set_ylabel("Pressure [bar]")
    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    ax_tau.set_ylim(float(np.max(pressure)), float(np.min(pressure)))
    colorbar = fig.colorbar(mesh, ax=ax_tau, pad=0.02)
    colorbar.set_label("log10 g-weighted layer tau")

    ax_profile.plot(
        temperature,
        pressure,
        color="#111111",
        linewidth=1.9,
        label="P-T profile",
    )
    ax_profile.set_title("Averaged Weighting vs P-T")
    ax_profile.set_xlabel("Temperature [K]")
    ax_profile.set_ylabel("Pressure [bar]")
    ax_profile.set_yscale("log")
    ax_profile.set_ylim(float(np.max(pressure)), float(np.min(pressure)))
    ax_profile.grid(alpha=0.25, which="both")

    ax_weight = ax_profile.twiny()
    ax_weight.fill_betweenx(
        pressure,
        0.0,
        mean_weighting,
        color="#2a9d8f",
        alpha=0.25,
    )
    ax_weight.plot(
        mean_weighting,
        pressure,
        color="#2a9d8f",
        linewidth=1.8,
        label="Mean weighting",
    )
    ax_weight.set_xlabel("Normalized Mean Weighting")
    ax_weight.set_xlim(0.0, 1.05)

    lines = ax_profile.get_lines() + ax_weight.get_lines()
    labels = [line.get_label() for line in lines]
    ax_profile.legend(lines, labels, frameon=False, loc="lower right", fontsize=8.5)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def _synthetic_h2o_table(
    pressure_grid: PressureGrid,
    temperature: np.ndarray,
) -> tuple[SpectralGrid, CorrelatedKTable]:
    wavelength = np.geomspace(1.0, 10.0, 96)
    wavenumber = 10000.0 / wavelength
    g_samples = np.array([0.10, 0.35, 0.65, 0.90])
    g_weights = np.array([0.20, 0.30, 0.30, 0.20])

    pressure_factor = (pressure_grid.centers[:, None, None, None] / 1.0e-3) ** 0.18
    temperature_factor = (temperature[None, :, None, None] / 1300.0) ** 0.65
    wave = wavelength[None, None, :, None]
    band_shape = (
        0.25
        + 1.8 * np.exp(-0.5 * (np.log(wave / 1.4) / 0.12) ** 2)
        + 2.4 * np.exp(-0.5 * (np.log(wave / 2.7) / 0.14) ** 2)
        + 3.5 * np.exp(-0.5 * (np.log(wave / 6.3) / 0.18) ** 2)
    )
    g_factor = np.array([0.18, 0.65, 1.8, 5.0])[None, None, None, :]
    kcoeff = (2.0e-26 + 5.0e-24 * pressure_factor * temperature_factor * band_shape) * g_factor

    spectral_grid = SpectralGrid.from_array(wavelength, unit="micron", role="opacity")
    table = CorrelatedKTable(
        species="H2O",
        pressure_bar=pressure_grid.centers,
        temperature_K=temperature,
        wavenumber_cm_inverse=wavenumber,
        wavelength_micron=wavelength,
        g_samples=g_samples,
        g_weights=g_weights,
        kcoeff=kcoeff,
        metadata={"example": "synthetic h2o opacity"},
    )
    return spectral_grid, table


if __name__ == "__main__":
    main()
