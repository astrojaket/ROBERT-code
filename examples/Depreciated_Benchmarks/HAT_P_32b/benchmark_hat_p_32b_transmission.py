"""Benchmark absorption-dominated HAT-P-32b spherical transmission."""

from __future__ import annotations

import json
import os
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
    hydrostatic_path_geometry,
    load_emission_observation_npz,
    solve_absorption_transmission,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_transmission"


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observation = load_emission_observation_npz(OBSERVATION_NPZ, instrument="JWST/NIRSpec G395H")
    model = build_parameterized_clear_sky_emission_model(
        make_model_config(n_layers=80, include_rayleigh=False),
        spectral_grid=observation.spectral_grid,
    )
    atmosphere = model.atmosphere_builder.build(reference_map_parameters())
    opacity = model.opacity_provider.evaluate(atmosphere, model.prepared_opacity)
    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=model.gravity_m_s2,
        gas_combination=model.config.gas_combination,
    )
    if model.cia_table is None:
        raise RuntimeError("transmission benchmark requires CIA")
    cia = cia_optical_depth(
        gas_tau,
        model.cia_table,
        normal_hydrogen=model.config.cia_normal_hydrogen,
        temperature_extrapolation=model.config.cia_temperature_extrapolation,
        spectral_extrapolation=model.config.cia_spectral_extrapolation,
    )
    path = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=model.gravity_m_s2,
        reference_radius_m=model.planet.radius_m,
        reference_pressure=10.0,
        reference_pressure_unit="bar",
    )
    gas_only = solve_absorption_transmission(
        gas_tau,
        path,
        star_radius_m=model.star.radius_m,
        impact_quadrature_order=12,
    )
    molecular_cia = solve_absorption_transmission(
        gas_tau,
        path,
        star_radius_m=model.star.radius_m,
        additional_optical_depths=[cia],
        impact_quadrature_order=12,
    )
    convergence = {
        order: solve_absorption_transmission(
            gas_tau,
            path,
            star_radius_m=model.star.radius_m,
            additional_optical_depths=[cia],
            impact_quadrature_order=order,
        )
        for order in (2, 4, 8, 16)
    }
    reference = np.asarray(convergence[16].transit_depth.values)
    convergence_ppm = {
        str(order): float(
            np.max(np.abs(np.asarray(result.transit_depth.values) - reference)) * 1.0e6
        )
        for order, result in convergence.items()
    }
    cia_change = (
        np.asarray(molecular_cia.transit_depth.values)
        - np.asarray(gas_only.transit_depth.values)
    )
    report = {
        "schema_version": 1,
        "benchmark": "HAT_P_32b_absorption_spherical_transmission",
        "reference_radius_pressure_bar": 10.0,
        "reference_radius_m": model.planet.radius_m,
        "star_radius_m": model.star.radius_m,
        "n_layers": atmosphere.n_layers,
        "n_wavelength": observation.spectral_grid.size,
        "n_g_ordinates": int(gas_tau.g_weights.size),
        "species": list(model.config.opacity_species),
        "metrics": {
            "transit_depth_min_ppm": float(np.min(molecular_cia.transit_depth.values) * 1.0e6),
            "transit_depth_max_ppm": float(np.max(molecular_cia.transit_depth.values) * 1.0e6),
            "spectral_modulation_ppm": float(np.ptp(molecular_cia.transit_depth.values) * 1.0e6),
            "max_cia_change_ppm": float(np.max(np.abs(cia_change)) * 1.0e6),
            "impact_quadrature_convergence_max_difference_ppm": convergence_ppm,
        },
        "scope": {
            "scattering": "extinction_only_no_scattered_light_return",
            "refraction": "not_included",
            "stellar_limb_darkening": "not_included",
        },
    }
    json_path = OUTPUT_DIR / "hat_p_32b_transmission_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = OUTPUT_DIR / "hat_p_32b_transmission_benchmark.png"
    _plot(
        plot_path,
        np.asarray(observation.spectral_grid.values),
        gas_only,
        molecular_cia,
        convergence,
        model.planet.radius_m,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(json.dumps(report, indent=2))
    return report


def _plot(output_path, wavelength, gas_only, molecular_cia, convergence, reference_radius) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    ax_depth, ax_altitude, ax_cia, ax_contribution = axes.flat
    ax_depth.plot(wavelength, np.asarray(gas_only.transit_depth.values) * 1.0e6, label="molecules")
    ax_depth.plot(
        wavelength,
        np.asarray(molecular_cia.transit_depth.values) * 1.0e6,
        ls="--",
        label="molecules + CIA",
    )
    ax_depth.set(ylabel="Transit depth [ppm]", title="HAT-P-32b spherical transmission")
    ax_depth.legend(frameon=False)

    altitude_km = (np.asarray(molecular_cia.effective_radius_m) - reference_radius) / 1.0e3
    ax_altitude.plot(wavelength, altitude_km, color="#e45756")
    ax_altitude.set(ylabel="Effective altitude above 10-bar radius [km]", title="Effective transit radius")

    reference = np.asarray(convergence[16].transit_depth.values)
    for order in (2, 4, 8):
        ax_cia.plot(
            wavelength,
            (np.asarray(convergence[order].transit_depth.values) - reference) * 1.0e6,
            label=f"order {order} - order 16",
        )
    ax_cia.set(xlabel="Wavelength [micron]", ylabel="Transit-depth difference [ppm]", title="Impact quadrature convergence")
    ax_cia.legend(frameon=False)

    contribution = np.asarray(molecular_cia.annulus_area_contribution_m2)
    normalization = np.sum(contribution, axis=0, keepdims=True)
    normalized = np.divide(contribution, normalization, out=np.zeros_like(contribution), where=normalization > 0.0)
    midpoint = 0.5 * (
        np.asarray(molecular_cia.impact_radius_edges_m[:-1])
        + np.asarray(molecular_cia.impact_radius_edges_m[1:])
    )
    altitude_midpoint = (midpoint - reference_radius) / 1.0e3
    image = ax_contribution.pcolormesh(wavelength, altitude_midpoint, normalized, shading="auto", cmap="magma")
    ax_contribution.set(xlabel="Wavelength [micron]", ylabel="Tangent altitude [km]", title="Normalized annulus contribution")
    fig.colorbar(image, ax=ax_contribution, label="Normalized area contribution")
    for axis in (ax_depth, ax_altitude):
        axis.set_xlabel("Wavelength [micron]")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
