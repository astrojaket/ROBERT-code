"""Compare molecular correlated-k emission between ROBERT and PICASO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    CorrelatedKOpacityProvider,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    load_robert_npz_archive,
    solve_emission,
)
from robert_exoplanets.diagnostics.benchmark_style import (
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
)

ROOT = Path(__file__).resolve().parent
OPACITY_DIR = ROOT / "data" / "hat_p_32b" / "opacities"
PICASO_RUNNER = ROOT / "run_picaso_grey_rt_reference.py"
OUTPUT_DIR = ROOT / "outputs" / "molecular_emission_picaso"
SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
VMR = {
    "H2O": 5.0e-4,
    "CO": 3.0e-4,
    "CO2": 1.0e-7,
    "CH4": 1.0e-8,
    "NH3": 1.0e-7,
    "HCN": 1.0e-8,
}


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picaso-python",
        type=Path,
        default=Path(
            os.environ.get(
                "ROBERT_PICASO_PYTHON",
                "/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python",
            )
        ),
    )
    parser.add_argument("--opacity-dir", type=Path, default=OPACITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provider, spectral_grid = _opacity_provider(args.opacity_dir)
    atmosphere = _atmosphere(spectral_grid)
    geometry = gauss_legendre_disk_geometry(6)

    components: dict[str, np.ndarray] = {}
    for species in ("H2O", "CO", "CO2", "CH4"):
        _, result = _run_robert(provider, atmosphere, spectral_grid, (species,), geometry)
        components[species] = np.asarray(result.radiance.values)
    gas_tau, robert = _run_robert(provider, atmosphere, spectral_grid, SPECIES, geometry)

    input_path = args.output_dir / "molecular_emission_shared_tau.npz"
    picaso_path = args.output_dir / "molecular_emission_picaso_toon.npz"
    zero_scattering = np.zeros_like(gas_tau.total_tau)
    np.savez_compressed(
        input_path,
        wavelength_micron=spectral_grid.values,
        pressure_edges_bar=atmosphere.pressure_grid.edges,
        temperature_level_k=atmosphere.temperature_edges,
        extinction_tau=gas_tau.total_tau,
        single_scattering_albedo=zero_scattering,
        asymmetry_factor=zero_scattering,
        emission_mu=geometry.emission_angle_cosines,
        g_weights=gas_tau.g_weights,
    )
    _run_picaso(args.picaso_python, input_path, picaso_path)
    with np.load(picaso_path, allow_pickle=False) as reference:
        picaso_point = np.asarray(reference["point_radiance_w_m2_m_sr"], dtype=float)
    picaso_disk = np.tensordot(
        geometry.emission_angle_weights,
        picaso_point,
        axes=(0, 0),
    )
    robert_disk = np.asarray(robert.radiance.values)
    point_relative = (np.asarray(robert.point_radiance) - picaso_point) / picaso_point
    disk_relative = (robert_disk - picaso_disk) / picaso_disk

    report = {
        "schema_version": 1,
        "comparison": "ROBERT_vs_PICASO_identical_molecular_correlated_k_optical_depth",
        "opacity_source": "bundled_ExoMolOP_exo_k_HAT_P_32b_archives",
        "species": list(SPECIES),
        "volume_mixing_ratios": VMR,
        "n_layers": atmosphere.n_layers,
        "n_wavelength": spectral_grid.size,
        "n_g_ordinates": int(gas_tau.g_weights.size),
        "n_emission_angles": geometry.n_points,
        "pressure_range_bar": [
            float(np.min(atmosphere.pressure_grid.edges)),
            float(np.max(atmosphere.pressure_grid.edges)),
        ],
        "temperature_range_k": [
            float(np.min(atmosphere.temperature_edges)),
            float(np.max(atmosphere.temperature_edges)),
        ],
        "metrics": {
            "max_abs_disk_relative_difference": float(np.max(np.abs(disk_relative))),
            "median_disk_relative_difference": float(np.median(disk_relative)),
            "rms_disk_relative_difference": float(np.sqrt(np.mean(disk_relative**2))),
            "max_abs_point_relative_difference": float(np.max(np.abs(point_relative))),
            "max_abs_relative_difference_by_mu": np.max(
                np.abs(point_relative), axis=1
            ).tolist(),
        },
        "controlled_quantities": {
            "molecular_tau": "identical layer-wavelength-g cube",
            "g_weights": "identical",
            "pressure_edges": "identical",
            "temperature_edges": "identical",
            "bottom_boundary": "blackbody",
            "top_boundary": "zero incoming thermal radiation",
        },
    }
    json_path = args.output_dir / "molecular_emission_picaso_comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = args.output_dir / "molecular_emission_picaso_comparison.png"
    _plot(
        plot_path,
        np.asarray(spectral_grid.values),
        atmosphere,
        robert_disk,
        picaso_disk,
        disk_relative,
        components,
        robert.normalized_layer_contribution(),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(json.dumps(report, indent=2))
    return report


def _opacity_provider(
    opacity_dir: Path,
) -> tuple[CorrelatedKOpacityProvider, SpectralGrid]:
    paths = {species: opacity_dir / f"{species}.robert-opacity.npz" for species in SPECIES}
    provider = CorrelatedKOpacityProvider.from_robert_archives(
        paths,
        name="molecular-emission-PICASO-comparison",
        interpolation="log_pressure_temperature_log_k_clip",
    )
    reference = load_robert_npz_archive(paths["H2O"])
    spectral_grid = SpectralGrid.from_array(
        reference.arrays["wavelength_micron"],
        unit="micron",
        role="opacity",
        name="HAT-P-32b ExoMolOP molecular comparison",
    )
    return provider, spectral_grid


def _atmosphere(spectral_grid: SpectralGrid) -> AtmosphereState:
    del spectral_grid
    pressure_edges = np.geomspace(1.0e-5, 100.0, 81)
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=np.sqrt(pressure_edges[:-1] * pressure_edges[1:]),
        unit="bar",
        name="molecular-emission comparison pressure",
    )
    log_fraction = (
        np.log10(pressure_edges) - np.log10(pressure_edges[0])
    ) / (np.log10(pressure_edges[-1]) - np.log10(pressure_edges[0]))
    temperature_edges = 800.0 + 1000.0 * log_fraction**0.7
    temperature_edges[-2:] = 1800.0
    temperature = 0.5 * (temperature_edges[:-1] + temperature_edges[1:])
    composition = {
        species: np.full(pressure_grid.n_layers, abundance)
        for species, abundance in VMR.items()
    }
    composition.update(
        {
            "H2": np.full(pressure_grid.n_layers, 0.849),
            "He": np.full(pressure_grid.n_layers, 0.15),
        }
    )
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=temperature_edges,
        composition=composition,
        mean_molecular_weight=2.3,
        metadata={"profile": "controlled_nonisothermal_molecular_comparison"},
    )


def _run_robert(
    provider: CorrelatedKOpacityProvider,
    atmosphere: AtmosphereState,
    spectral_grid: SpectralGrid,
    species: tuple[str, ...],
    geometry: Any,
) -> tuple[Any, Any]:
    prepared = provider.prepare(spectral_grid, atmosphere.pressure_grid, species)
    opacity = provider.evaluate(atmosphere, prepared)
    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=4.35,
        gas_combination="random_overlap" if len(species) > 1 else "sum_by_g",
    )
    result = solve_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="blackbody",
    )
    return gas_tau, result


def _run_picaso(python: Path, input_path: Path, output_path: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "picaso-numba-cache"),
    )
    completed = subprocess.run(
        [
            str(python),
            str(PICASO_RUNNER),
            str(input_path),
            str(output_path),
            "--method",
            "toon",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PICASO molecular reference failed:\n{completed.stdout}\n{completed.stderr}"
        )


def _plot(
    output_path: Path,
    wavelength: np.ndarray,
    atmosphere: AtmosphereState,
    robert: np.ndarray,
    picaso: np.ndarray,
    relative: np.ndarray,
    components: dict[str, np.ndarray],
    contribution: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    ax_spectrum, ax_residual, ax_components, ax_contribution = axes.flat
    ax_spectrum.plot(wavelength, picaso, color=REFERENCE_COLOR, lw=2.0, label="PICASO Toon")
    ax_spectrum.plot(wavelength, robert, color=ROBERT_COLOR, ls="--", lw=1.8, label="ROBERT")
    ax_spectrum.set(
        ylabel="Disk radiance [W m$^{-2}$ m$^{-1}$ sr$^{-1}$]",
        title="Identical six-molecule correlated-k optical depth",
    )
    ax_spectrum.legend(frameon=False)

    ax_residual.plot(wavelength, 100.0 * relative, color=RESIDUAL_COLOR)
    ax_residual.axhline(0.0, color=REFERENCE_COLOR, lw=0.8)
    ax_residual.set(
        ylabel="(ROBERT - PICASO) / PICASO [%]",
        title="RT-only residual",
    )

    for species, values in components.items():
        ax_components.plot(wavelength, values, lw=1.3, label=species)
    ax_components.plot(wavelength, robert, color=ROBERT_COLOR, lw=2.0, label="six gases")
    ax_components.set(
        xlabel="Wavelength [micron]",
        ylabel="Disk radiance [W m$^{-2}$ m$^{-1}$ sr$^{-1}$]",
        title="Molecular spectral structure",
    )
    ax_components.legend(frameon=False, ncol=2)

    image = ax_contribution.pcolormesh(
        wavelength,
        atmosphere.pressure_grid.centers,
        contribution,
        shading="auto",
        cmap="magma",
    )
    ax_contribution.set_yscale("log")
    ax_contribution.invert_yaxis()
    ax_contribution.set(
        xlabel="Wavelength [micron]",
        ylabel="Pressure [bar]",
        title="ROBERT normalized emission contribution",
    )
    fig.colorbar(image, ax=ax_contribution, label="Normalized contribution")
    for axis in (ax_spectrum, ax_residual, ax_components):
        axis.set_xlabel("Wavelength [micron]")
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
