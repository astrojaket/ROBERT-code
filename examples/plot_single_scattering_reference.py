"""Plot a synthetic single-scattering reference calculation."""

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
    DirectStellarBeam,
    EvaluatedCorrelatedKOpacity,
    LayerOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SingleScatteringSource,
    SpectralGrid,
    assemble_gas_optical_depth,
    nemesis_lobatto_phase_geometry,
    solve_clear_sky_emission,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "single_scattering_reference"
R_SUN_M = 6.957e8
R_JUP_M = 7.1492e7
AU_M = 1.495978707e11


def main() -> Path:
    """Run and plot a visible-wavelength single-scattering reference case."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wavelength = np.linspace(0.4, 2.2, 320)
    spectral_grid = SpectralGrid.from_array(wavelength, unit="micron", role="opacity")
    pressure_grid = _pressure_grid()
    gas_tau = _gas_tau(pressure_grid, spectral_grid)
    scattering_tau = _synthetic_rayleigh_tau(pressure_grid, wavelength)
    scattering = LayerOpticalDepth(
        name="synthetic H2/He Rayleigh",
        tau=scattering_tau,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        kind="scattering_extinction",
        metadata={"reference_case": "lambda^-4 synthetic Rayleigh"},
    )
    beam = DirectStellarBeam.blackbody(
        spectral_grid,
        star_temperature_k=6000.0,
        star_radius_m=R_SUN_M,
        semi_major_axis_m=0.05 * AU_M,
    )
    scattering_source = SingleScatteringSource(
        stellar_beam=beam,
        phase_function="rayleigh",
    )

    phase_results = {}
    for phase in (0.0, 90.0, 180.0):
        geometry = nemesis_lobatto_phase_geometry(phase_angle_deg=phase, n_mu=4)
        phase_results[phase] = solve_clear_sky_emission(
            gas_tau,
            geometry=geometry,
            bottom_boundary="none",
            additional_optical_depths=[scattering],
            scattering_source=scattering_source,
            planet_radius_m=1.1 * R_JUP_M,
            star_radius_m=R_SUN_M,
            star_temperature_k=6000.0,
        )

    output_path = OUTPUT_DIR / "single_scattering_phase_reference.png"
    _plot_reference(output_path, pressure_grid, phase_results)
    print(f"Wrote {output_path}")
    return output_path


def _plot_reference(
    output_path: Path,
    pressure_grid: PressureGrid,
    phase_results: dict[float, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    ax_spectrum, ax_profile = axes

    colors = {0.0: "#4c78a8", 90.0: "#f58518", 180.0: "#54a24b"}
    for phase, result in phase_results.items():
        if result.eclipse_depth is None:
            raise RuntimeError("single-scattering reference did not include eclipse depth")
        ax_spectrum.plot(
            result.eclipse_depth.spectral_grid.values,
            result.eclipse_depth.values * 1.0e6,
            color=colors[phase],
            linewidth=1.7,
            label=f"Phase {phase:.0f} deg",
        )
    ax_spectrum.set_xlabel("Wavelength [micron]")
    ax_spectrum.set_ylabel("Reflected-light depth [ppm]")
    ax_spectrum.set_title("Single-Scattering Phase Reference")
    ax_spectrum.grid(alpha=0.25)
    ax_spectrum.legend(frameon=False)

    dayside = phase_results[180.0]
    if dayside.scattering_layer_contribution_radiance is None:
        raise RuntimeError("single-scattering reference did not include scattering diagnostics")
    profile = np.mean(dayside.scattering_layer_contribution_radiance, axis=1)
    if np.max(profile) > 0.0:
        profile = profile / np.max(profile)
    ax_profile.plot(profile, pressure_grid.centers, color="#54a24b", linewidth=1.9)
    ax_profile.fill_betweenx(pressure_grid.centers, 0.0, profile, color="#54a24b", alpha=0.22)
    ax_profile.set_yscale("log")
    ax_profile.set_ylim(float(np.max(pressure_grid.centers)), float(np.min(pressure_grid.centers)))
    ax_profile.set_xlabel("Normalized contribution")
    ax_profile.set_ylabel("Pressure [bar]")
    ax_profile.set_title("Dayside Scattering Contribution")
    ax_profile.grid(alpha=0.25, which="both")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _pressure_grid() -> PressureGrid:
    centers = np.logspace(-5.0, 0.0, 20)
    log_centers = np.log(centers)
    inner_edges = 0.5 * (log_centers[:-1] + log_centers[1:])
    first_edge = log_centers[0] - (inner_edges[0] - log_centers[0])
    last_edge = log_centers[-1] + (log_centers[-1] - inner_edges[-1])
    edges = np.exp(np.concatenate(([first_edge], inner_edges, [last_edge])))
    return PressureGrid(edges=edges, centers=centers, unit="bar", name="synthetic scattering")


def _synthetic_rayleigh_tau(
    pressure_grid: PressureGrid,
    wavelength_micron: np.ndarray,
) -> np.ndarray:
    layer_weights = pressure_grid.centers / np.sum(pressure_grid.centers)
    vertical_tau = 0.18 * (0.5 / wavelength_micron) ** 4
    return layer_weights[:, None] * vertical_tau[None, :]


def _gas_tau(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
):
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.full(pressure_grid.n_layers, 100.0),
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
        cache_key="single-scattering-reference",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure_grid.n_layers, spectral_grid.size, 1)),
        unit="m^2/molecule",
    )
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


if __name__ == "__main__":
    main()
