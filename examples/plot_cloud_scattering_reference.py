"""Plot a synthetic cloud/aerosol scattering reference calculation."""

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
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    grey_cloud_deck,
    power_law_haze,
    solve_clear_sky_emission,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cloud_scattering_reference"


def main() -> Path:
    """Run a synthetic cloudy thermal-emission reference case and plot it."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spectral_grid = SpectralGrid.from_array(
        np.geomspace(0.7, 12.0, 360),
        unit="micron",
        role="opacity",
    )
    pressure_grid = PressureGrid.logspace(
        min_pressure=1.0e-5,
        max_pressure=10.0,
        n_layers=48,
        unit="bar",
        name="synthetic cloud reference",
    )
    gas_tau = _zero_gas_tau(pressure_grid, spectral_grid)
    cloud_deck = grey_cloud_deck(
        pressure_grid,
        spectral_grid,
        cloud_top_pressure=3.0e-2,
        cloud_top_pressure_unit="bar",
        optical_depth=1.2,
        name="synthetic grey condensate deck",
        single_scattering_albedo=0.55,
        asymmetry_factor=0.35,
    )
    haze = power_law_haze(
        pressure_grid,
        spectral_grid,
        optical_depth_at_reference=0.25,
        reference_wavelength_micron=1.0,
        slope=-4.0,
        name="synthetic optical haze",
        single_scattering_albedo=0.9,
        asymmetry_factor=0.2,
    )
    cloud_terms = [cloud_deck, haze]

    clear = solve_clear_sky_emission(gas_tau, bottom_boundary="blackbody")
    extinction_only = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="blackbody",
        additional_optical_depths=cloud_terms,
    )
    two_stream = solve_clear_sky_emission(
        gas_tau,
        bottom_boundary="blackbody",
        additional_optical_depths=cloud_terms,
        multiple_scattering_backend="two_stream",
    )

    output_path = OUTPUT_DIR / "cloud_scattering_two_stream_reference.png"
    _plot_reference(output_path, pressure_grid, clear, extinction_only, two_stream)
    print(f"Wrote {output_path}")
    return output_path


def _plot_reference(
    output_path: Path,
    pressure_grid: PressureGrid,
    clear,
    extinction_only,
    two_stream,
) -> None:
    wavelength = clear.radiance.spectral_grid.values
    clear_radiance = clear.radiance.values
    extinction_ratio = extinction_only.radiance.values / clear_radiance
    two_stream_ratio = two_stream.radiance.values / clear_radiance
    physical_tau = np.sum(extinction_only.extinction_optical_depth, axis=0)[:, 0]
    effective_tau = np.sum(two_stream.total_optical_depth, axis=0)[:, 0]
    cloud_contribution = np.mean(two_stream.normalized_layer_contribution(), axis=1)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.6, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0, 0.9]},
    )
    ax_ratio, ax_tau, ax_profile = axes

    ax_ratio.plot(wavelength, extinction_ratio, color="#4c78a8", linewidth=1.8, label="Extinction only")
    ax_ratio.plot(wavelength, two_stream_ratio, color="#f58518", linewidth=1.8, label="Two-stream reference")
    ax_ratio.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.45)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Wavelength [micron]")
    ax_ratio.set_ylabel("Cloudy / clear thermal radiance")
    ax_ratio.set_title("Cloudy Emission Ratio")
    ax_ratio.grid(alpha=0.25, which="both")
    ax_ratio.legend(frameon=False)

    ax_tau.plot(wavelength, physical_tau, color="#4c78a8", linewidth=1.8, label="Physical extinction tau")
    ax_tau.plot(wavelength, effective_tau, color="#f58518", linewidth=1.8, label="Two-stream effective tau")
    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    ax_tau.set_xlabel("Wavelength [micron]")
    ax_tau.set_ylabel("Column optical depth")
    ax_tau.set_title("Cloud/Aerosol Tau")
    ax_tau.grid(alpha=0.25, which="both")
    ax_tau.legend(frameon=False)

    if np.max(cloud_contribution) > 0.0:
        cloud_contribution = cloud_contribution / np.max(cloud_contribution)
    ax_profile.fill_betweenx(
        pressure_grid.centers,
        0.0,
        cloud_contribution,
        color="#54a24b",
        alpha=0.25,
    )
    ax_profile.plot(cloud_contribution, pressure_grid.centers, color="#54a24b", linewidth=1.9)
    ax_profile.set_yscale("log")
    ax_profile.set_ylim(float(np.max(pressure_grid.centers)), float(np.min(pressure_grid.centers)))
    ax_profile.set_xlabel("Normalized contribution")
    ax_profile.set_ylabel("Pressure [bar]")
    ax_profile.set_title("Mean Contribution")
    ax_profile.grid(alpha=0.25, which="both")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _zero_gas_tau(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
):
    pressure_scale = (
        np.log10(pressure_grid.centers) - np.log10(np.min(pressure_grid.centers))
    ) / (
        np.log10(np.max(pressure_grid.centers)) - np.log10(np.min(pressure_grid.centers))
    )
    temperature = 850.0 + 850.0 * pressure_scale**0.7
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        composition={
            "H2O": np.full(pressure_grid.n_layers, 1.0e-12),
            "H2": np.full(pressure_grid.n_layers, 0.84),
            "He": np.full(pressure_grid.n_layers, 0.16),
        },
        mean_molecular_weight=2.3,
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="synthetic-zero-opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="cloud-scattering-reference",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure_grid.n_layers, spectral_grid.size, 1)),
        unit="m^2/molecule",
    )
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


if __name__ == "__main__":
    main()
