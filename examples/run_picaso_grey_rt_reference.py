"""Run a PICASO low-level thermal solver for a ROBERT validation payload.

This script is intentionally isolated from the ROBERT package environment. Run
it with a Python environment containing PICASO; the comparison harness invokes
it as a subprocess so PICASO remains an optional external reference code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Shared RT input payload (.npz)")
    parser.add_argument("output", type=Path, help="PICASO reference output (.npz)")
    parser.add_argument(
        "--method",
        choices=("toon", "sh4"),
        default="toon",
        help="PICASO thermal multiple-scattering solver",
    )
    args = parser.parse_args()

    from picaso.fluxes import get_thermal_1d, get_thermal_SH

    with np.load(args.input, allow_pickle=False) as payload:
        wavelength_micron = np.asarray(payload["wavelength_micron"], dtype=float)
        pressure_edges_bar = np.asarray(payload["pressure_edges_bar"], dtype=float)
        temperature_level_k = np.asarray(payload["temperature_level_k"], dtype=float)
        extinction_tau = np.asarray(payload["extinction_tau"], dtype=float)
        single_scattering_albedo = np.asarray(payload["single_scattering_albedo"], dtype=float)
        asymmetry_factor = np.asarray(payload["asymmetry_factor"], dtype=float)
        emission_mu = np.asarray(payload["emission_mu"], dtype=float)
        g_weights = (
            np.asarray(payload["g_weights"], dtype=float)
            if "g_weights" in payload
            else np.array([1.0])
        )

    nlevel = pressure_edges_bar.size
    nwavelength = wavelength_micron.size
    if extinction_tau.ndim == 2:
        extinction_tau = extinction_tau[:, :, None]
    if single_scattering_albedo.ndim == 2:
        single_scattering_albedo = single_scattering_albedo[:, :, None]
    if asymmetry_factor.ndim == 2:
        asymmetry_factor = asymmetry_factor[:, :, None]
    expected_shape = (nlevel - 1, nwavelength, g_weights.size)
    for name, values in (
        ("extinction_tau", extinction_tau),
        ("single_scattering_albedo", single_scattering_albedo),
        ("asymmetry_factor", asymmetry_factor),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}")
    if temperature_level_k.shape != (nlevel,):
        raise ValueError("temperature_level_k must have one value per pressure edge")

    # PICASO's Toon implementation divides by scattering-coupling terms, so
    # its own high-level optics path replaces exactly zero albedo by 1e-10.
    if g_weights.shape != (extinction_tau.shape[-1],):
        raise ValueError("g_weights must have one value per correlated-k ordinate")
    if np.any(g_weights < 0.0) or not np.isclose(np.sum(g_weights), 1.0):
        raise ValueError("g_weights must be non-negative and sum to one")
    picaso_albedo = np.maximum(single_scattering_albedo, 1.0e-10)
    wavenumber_cm = 1.0e4 / wavelength_micron
    picaso_pressure_dyne_cm2 = pressure_edges_bar * 1.0e6
    # get_thermal_1d normally extrapolates a small thermal optical depth above
    # its first positive pressure. Setting the formal top level to zero imposes
    # Rutten's no-incident-radiation boundary at tau=0 for this controlled test.
    picaso_pressure_dyne_cm2[0] = 0.0
    point_flux_cgs = np.zeros((emission_mu.size, 1, nwavelength), dtype=float)
    level_fluxes = None
    for g_index, g_weight in enumerate(g_weights):
        tau_g = extinction_tau[:, :, g_index]
        albedo_g = picaso_albedo[:, :, g_index]
        asymmetry_g = asymmetry_factor[:, :, g_index]
        if args.method == "toon":
            point_g, level_g = get_thermal_1d(
                nlevel,
                wavenumber_cm,
                nwavelength,
                emission_mu.size,
                1,
                temperature_level_k,
                tau_g,
                albedo_g,
                asymmetry_g,
                picaso_pressure_dyne_cm2,
                emission_mu[:, None],
                np.zeros(nwavelength, dtype=float),
                1,
                np.zeros(nwavelength, dtype=float),
                0,
            )
        else:
            cumulative_tau = np.vstack(
                (np.zeros((1, nwavelength), dtype=float), np.cumsum(tau_g, axis=0))
            )
            point_g, level_g = get_thermal_SH(
                nlevel,
                wavenumber_cm,
                nwavelength,
                emission_mu.size,
                1,
                temperature_level_k,
                tau_g,
                cumulative_tau,
                albedo_g,
                asymmetry_g,
                tau_g,
                cumulative_tau,
                albedo_g,
                albedo_g,
                asymmetry_g,
                picaso_pressure_dyne_cm2,
                emission_mu[:, None],
                np.zeros(nwavelength, dtype=float),
                4,
                1,
            )
        point_flux_cgs += g_weight * point_g
        level_g_array = np.asarray(level_g, dtype=float)
        level_fluxes = (
            g_weight * level_g_array
            if level_fluxes is None
            else level_fluxes + g_weight * level_g_array
        )

    if args.method == "toon":
        solver = "picaso.fluxes.get_thermal_1d"
        method = "Toon_1989_source_function_two_stream"
    else:
        solver = "picaso.fluxes.get_thermal_SH"
        method = "Rooney_et_al_P3_SH4_Henyey_Greenstein"

    # get_thermal_1d returns 2*pi*I_lambda in erg s^-1 cm^-2 cm^-1.
    # One CGS wavelength-density unit is 0.1 W m^-2 m^-1.
    point_radiance_si = point_flux_cgs[:, 0, :] * (0.1 / (2.0 * np.pi))
    metadata = {
        "solver": solver,
        "method": method,
        "picaso_zero_albedo_floor": "1e-10",
        "point_output_conversion": "erg/s/cm2/cm/(2*pi)*0.1_to_W/m2/m/sr",
        "hard_surface": "true_blackbody",
        "top_boundary": "zero_incident_thermal_radiation_at_tau_zero",
        "correlated_k_g_ordinates": str(g_weights.size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wavelength_micron=wavelength_micron,
        emission_mu=emission_mu,
        point_radiance_w_m2_m_sr=point_radiance_si,
        level_flux_diagnostics_cgs=level_fluxes,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


if __name__ == "__main__":
    main()
