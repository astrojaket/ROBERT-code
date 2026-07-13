"""Compare realistic HAT-P-32b molecular+CIA emission with PICASO RT."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .hat_p_32b_fastchem_config import (
        OBSERVATION_NPZ,
        make_model_config,
        reference_map_parameters,
    )
else:
    from hat_p_32b_fastchem_config import (
        OBSERVATION_NPZ,
        make_model_config,
        reference_map_parameters,
    )
from robert_exoplanets import (
    assemble_gas_optical_depth,
    build_parameterized_clear_sky_emission_model,
    cia_optical_depth,
    load_emission_observation_npz,
    planck_radiance_wavelength,
    solve_clear_sky_emission,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "hat_p_32b_emission_picaso"
PICASO_RUNNER = ROOT / "run_picaso_grey_rt_reference.py"
PICASO_PYTHON = Path(
    os.environ.get("ROBERT_PICASO_PYTHON", "/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python")
)


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observation = load_emission_observation_npz(
        OBSERVATION_NPZ,
        instrument="JWST/NIRSpec G395H",
    )
    model = build_parameterized_clear_sky_emission_model(
        make_model_config(n_layers=80, include_rayleigh=False),
        spectral_grid=observation.spectral_grid,
    )
    parameters = reference_map_parameters()
    atmosphere = model.atmosphere_builder.build(parameters)
    opacity = model.opacity_provider.evaluate(atmosphere, model.prepared_opacity)
    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=model.gravity_m_s2,
        gas_combination=model.config.gas_combination,
    )
    if model.cia_table is None:
        raise RuntimeError("HAT-P-32b comparison model must include CIA")
    cia = cia_optical_depth(
        gas_tau,
        model.cia_table,
        normal_hydrogen=model.config.cia_normal_hydrogen,
        temperature_extrapolation=model.config.cia_temperature_extrapolation,
        spectral_extrapolation=model.config.cia_spectral_extrapolation,
    )
    without_cia = solve_clear_sky_emission(
        gas_tau,
        geometry=model.geometry,
        bottom_boundary="blackbody",
        planet_radius_m=model.planet.radius_m,
        star_radius_m=model.star.radius_m,
        star_temperature_k=model.star.effective_temperature_k,
    )
    with_cia = solve_clear_sky_emission(
        gas_tau,
        geometry=model.geometry,
        bottom_boundary="blackbody",
        additional_optical_depths=[cia],
        planet_radius_m=model.planet.radius_m,
        star_radius_m=model.star.radius_m,
        star_temperature_k=model.star.effective_temperature_k,
    )

    total_tau = gas_tau.total_tau + np.asarray(cia.tau)[:, :, None]
    top_to_bottom = np.argsort(atmosphere.pressure_grid.centers)
    pressure_edges_top_to_bottom = np.asarray(atmosphere.pressure_grid.edges)[::-1]
    temperature_edges_top_to_bottom = np.asarray(atmosphere.temperature_edges)[::-1]
    shared_path = OUTPUT_DIR / "hat_p_32b_molecular_cia_shared_tau.npz"
    picaso_path = OUTPUT_DIR / "hat_p_32b_molecular_cia_picaso.npz"
    np.savez_compressed(
        shared_path,
        wavelength_micron=observation.spectral_grid.values,
        pressure_edges_bar=pressure_edges_top_to_bottom,
        temperature_level_k=temperature_edges_top_to_bottom,
        extinction_tau=total_tau[top_to_bottom],
        single_scattering_albedo=np.zeros_like(total_tau),
        asymmetry_factor=np.zeros_like(total_tau),
        emission_mu=model.geometry.emission_angle_cosines,
        g_weights=gas_tau.g_weights,
    )
    _run_picaso(shared_path, picaso_path)
    with np.load(picaso_path, allow_pickle=False) as archive:
        picaso_point = np.asarray(archive["point_radiance_w_m2_m_sr"], dtype=float)
    picaso_radiance = np.tensordot(
        model.geometry.emission_angle_weights,
        picaso_point,
        axes=(0, 0),
    )
    robert_radiance = np.asarray(with_cia.radiance.values)
    disk_relative = (robert_radiance - picaso_radiance) / picaso_radiance
    point_relative = (np.asarray(with_cia.point_radiance) - picaso_point) / picaso_point
    stellar_radiance = planck_radiance_wavelength(
        observation.spectral_grid.values,
        model.star.effective_temperature_k,
    )
    area_ratio = (model.planet.radius_m / model.star.radius_m) ** 2
    picaso_eclipse = picaso_radiance / stellar_radiance * area_ratio
    if with_cia.eclipse_depth is None or without_cia.eclipse_depth is None:
        raise RuntimeError("eclipse-depth diagnostics were not produced")

    report = {
        "schema_version": 1,
        "comparison": "HAT_P_32b_FastChem_molecular_CIA_ROBERT_vs_PICASO_RT",
        "parameters": {key: float(value) for key, value in parameters.items()},
        "n_layers": atmosphere.n_layers,
        "n_wavelength": observation.spectral_grid.size,
        "n_g_ordinates": int(gas_tau.g_weights.size),
        "species": list(model.config.opacity_species),
        "cia": str(cia.metadata.get("source_project", cia.name)),
        "metrics": {
            "max_abs_disk_relative_difference": float(np.max(np.abs(disk_relative))),
            "rms_disk_relative_difference": float(np.sqrt(np.mean(disk_relative**2))),
            "median_disk_relative_difference": float(np.median(disk_relative)),
            "max_abs_point_relative_difference": float(np.max(np.abs(point_relative))),
            "max_cia_eclipse_depth_change_ppm": float(
                np.max(
                    np.abs(
                        np.asarray(with_cia.eclipse_depth.values)
                        - np.asarray(without_cia.eclipse_depth.values)
                    )
                )
                * 1.0e6
            ),
        },
    }
    json_path = OUTPUT_DIR / "hat_p_32b_emission_picaso_comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = OUTPUT_DIR / "hat_p_32b_emission_picaso_comparison.png"
    _plot(
        plot_path,
        np.asarray(observation.spectral_grid.values),
        observation.flux,
        observation.uncertainty,
        np.asarray(with_cia.eclipse_depth.values),
        np.asarray(without_cia.eclipse_depth.values),
        picaso_eclipse,
        disk_relative,
        atmosphere.pressure_grid.centers,
        with_cia.normalized_layer_contribution(),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(json.dumps(report, indent=2))
    return report


def _run_picaso(input_path: Path, output_path: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "picaso-numba-cache"),
    )
    completed = subprocess.run(
        [
            str(PICASO_PYTHON),
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
        raise RuntimeError(f"PICASO reference failed:\n{completed.stdout}\n{completed.stderr}")


def _plot(
    output_path: Path,
    wavelength: np.ndarray,
    observed: np.ndarray,
    uncertainty: np.ndarray,
    robert: np.ndarray,
    without_cia: np.ndarray,
    picaso: np.ndarray,
    relative: np.ndarray,
    pressure: np.ndarray,
    contribution: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    ax_spectrum, ax_residual, ax_cia, ax_contribution = axes.flat
    ax_spectrum.errorbar(
        wavelength,
        observed * 1.0e6,
        yerr=uncertainty * 1.0e6,
        fmt=".",
        ms=3,
        color="#888888",
        alpha=0.6,
        label="JWST observation",
    )
    ax_spectrum.plot(wavelength, picaso * 1.0e6, color="#222222", lw=2.0, label="PICASO")
    ax_spectrum.plot(wavelength, robert * 1.0e6, color="#e45756", ls="--", lw=1.8, label="ROBERT")
    ax_spectrum.set(ylabel="Eclipse depth [ppm]", title="HAT-P-32b FastChem molecular+CIA emission")
    ax_spectrum.legend(frameon=False)

    ax_residual.plot(wavelength, relative * 100.0, color="#4c78a8")
    ax_residual.axhline(0.0, color="#222222", lw=0.8)
    ax_residual.set(ylabel="(ROBERT - PICASO) / PICASO [%]", title="Identical-opacity RT residual")

    ax_cia.plot(wavelength, (robert - without_cia) * 1.0e6, color="#72b7b2")
    ax_cia.axhline(0.0, color="#222222", lw=0.8)
    ax_cia.set(xlabel="Wavelength [micron]", ylabel="CIA eclipse-depth change [ppm]", title="H2-H2/H2-He CIA effect")

    image = ax_contribution.pcolormesh(
        wavelength,
        pressure,
        contribution,
        shading="auto",
        cmap="magma",
    )
    ax_contribution.set_yscale("log")
    ax_contribution.invert_yaxis()
    ax_contribution.set(xlabel="Wavelength [micron]", ylabel="Pressure [bar]", title="ROBERT molecular+CIA contribution")
    fig.colorbar(image, ax=ax_contribution, label="Normalized contribution")
    for axis in (ax_spectrum, ax_residual):
        axis.set_xlabel("Wavelength [micron]")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
