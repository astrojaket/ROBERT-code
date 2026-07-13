"""Independently evaluate a shared cloud contract with Virga and PICASO.

This runner must be executed with the external PICASO/Virga Python. It does
not import ROBERT and deliberately reimplements the validation gas-opacity
contract so no optical-depth array is shared between frameworks.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Shared physical contract (.npz)")
    parser.add_argument("output", type=Path, help="PICASO/Virga output (.npz)")
    args = parser.parse_args()

    from picaso.fluxes import get_thermal_SH
    from virga.calc_mie import calc_new_mieff

    with np.load(args.input, allow_pickle=False) as payload:
        wavelength = np.asarray(payload["wavelength_micron"], dtype=float)
        refractive_n = np.asarray(payload["refractive_index_n"], dtype=float)
        refractive_k = np.asarray(payload["refractive_index_k"], dtype=float)
        radius_cm = np.asarray(payload["radius_cm"], dtype=float)
        radius_upper_cm = np.asarray(payload["radius_upper_cm"], dtype=float)
        number_weights = np.asarray(payload["radius_number_weights"], dtype=float)
        pressure_edges = np.asarray(payload["pressure_edges_bar"], dtype=float)
        temperature_level = np.asarray(payload["temperature_level_k"], dtype=float)
        condensate_mass_fraction = np.asarray(
            payload["condensate_mass_fraction"], dtype=float
        )
        gas_mass_fractions = np.asarray(payload["gas_mass_fractions"], dtype=float)
        gas_species = tuple(str(value) for value in payload["gas_species"].tolist())
        gravity = float(payload["gravity_m_s2"])
        particle_density = float(payload["particle_density_kg_m3"])
        emission_mu = np.asarray(payload["emission_mu"], dtype=float)
        emission_weights = np.asarray(payload["emission_weights"], dtype=float)
        planet_radius_m = float(payload["planet_radius_m"])
        star_radius_m = float(payload["star_radius_m"])
        star_temperature_k = float(payload["star_temperature_k"])

    qext, qsca, g_qsca = calc_new_mieff(
        wavelength,
        refractive_n,
        refractive_k,
        radius_cm,
        radius_upper_cm,
        fort_calc_mie=False,
    )
    mie = _integrate_discrete_population(
        qext,
        qsca,
        g_qsca,
        radius_cm,
        number_weights,
        particle_density,
    )
    gas_tau = _validation_gas_tau(
        wavelength,
        pressure_edges,
        temperature_level,
        gas_species,
        gas_mass_fractions,
        gravity,
    )
    layer_mass = np.diff(pressure_edges) * 1.0e5 / gravity
    cloud_tau = (
        layer_mass[:, None]
        * condensate_mass_fraction[:, None]
        * mie["mass_extinction_m2_kg"][None, :]
    )
    cloud_scattering_tau = cloud_tau * mie["single_scattering_albedo"][None, :]
    total_tau = gas_tau + cloud_tau
    total_albedo = np.divide(
        cloud_scattering_tau,
        total_tau,
        out=np.zeros_like(total_tau),
        where=total_tau > 0.0,
    )
    total_asymmetry = np.repeat(
        mie["asymmetry_factor"][None, :], cloud_tau.shape[0], axis=0
    )

    cloudy_point = _solve_picaso_sh4(
        get_thermal_SH,
        wavelength,
        pressure_edges,
        temperature_level,
        total_tau,
        total_albedo,
        total_asymmetry,
        emission_mu,
    )
    cloud_free_point = _solve_picaso_sh4(
        get_thermal_SH,
        wavelength,
        pressure_edges,
        temperature_level,
        gas_tau,
        np.zeros_like(gas_tau),
        np.zeros_like(gas_tau),
        emission_mu,
    )
    cloudy_disk = np.tensordot(emission_weights, cloudy_point, axes=(0, 0))
    cloud_free_disk = np.tensordot(emission_weights, cloud_free_point, axes=(0, 0))
    stellar = _planck_radiance(wavelength, star_temperature_k)
    area_ratio = (planet_radius_m / star_radius_m) ** 2

    metadata = {
        "picaso_version": _version("picaso"),
        "virga_version": _version("virga-exo", fallback="virga"),
        "cloud_mie_solver": "virga.calc_mie.calc_new_mieff_PyMieScatt",
        "thermal_solver": "picaso.fluxes.get_thermal_SH_stream4",
        "phase_function": "Henyey_Greenstein_from_Virga_asymmetry",
        "delta_m": "false",
        "gas_opacity": "analytic_validation_contract_v1_not_science_opacity",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wavelength_micron=wavelength,
        mie_qext=qext,
        mie_qsca=qsca,
        mie_g_qsca=g_qsca,
        mass_extinction_m2_kg=mie["mass_extinction_m2_kg"],
        mass_scattering_m2_kg=mie["mass_scattering_m2_kg"],
        single_scattering_albedo=mie["single_scattering_albedo"],
        asymmetry_factor=mie["asymmetry_factor"],
        gas_tau=gas_tau,
        cloud_tau=cloud_tau,
        total_tau=total_tau,
        cloudy_point_radiance_w_m2_m_sr=cloudy_point,
        cloud_free_point_radiance_w_m2_m_sr=cloud_free_point,
        cloudy_disk_radiance_w_m2_m_sr=cloudy_disk,
        cloud_free_disk_radiance_w_m2_m_sr=cloud_free_disk,
        cloudy_eclipse_depth=cloudy_disk / stellar * area_ratio,
        cloud_free_eclipse_depth=cloud_free_disk / stellar * area_ratio,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def _integrate_discrete_population(
    qext: np.ndarray,
    qsca: np.ndarray,
    g_qsca: np.ndarray,
    radius_cm: np.ndarray,
    weights: np.ndarray,
    density_kg_m3: float,
) -> dict[str, np.ndarray]:
    radius_m = radius_cm * 1.0e-2
    area = np.pi * radius_m**2
    mass = (4.0 / 3.0) * np.pi * density_kg_m3 * radius_m**3
    mean_mass = np.sum(weights * mass)
    extinction = np.sum(qext * (weights * area)[None, :], axis=1)
    scattering = np.sum(qsca * (weights * area)[None, :], axis=1)
    scattering_g = np.sum(g_qsca * (weights * area)[None, :], axis=1)
    mass_extinction = extinction / mean_mass
    mass_scattering = scattering / mean_mass
    albedo = np.divide(
        mass_scattering,
        mass_extinction,
        out=np.zeros_like(mass_extinction),
        where=mass_extinction > 0.0,
    )
    asymmetry = np.divide(
        scattering_g,
        scattering,
        out=np.zeros_like(scattering_g),
        where=scattering > 0.0,
    )
    return {
        "mass_extinction_m2_kg": mass_extinction,
        "mass_scattering_m2_kg": mass_scattering,
        "single_scattering_albedo": albedo,
        "asymmetry_factor": asymmetry,
    }


def _validation_gas_tau(
    wavelength: np.ndarray,
    pressure_edges: np.ndarray,
    temperature_level: np.ndarray,
    species: tuple[str, ...],
    mass_fractions: np.ndarray,
    gravity: float,
) -> np.ndarray:
    """Independent implementation of validation contract v1.

    The smooth bands are deliberately not retrieval or science opacity.
    """

    pressure = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    temperature = 0.5 * (temperature_level[:-1] + temperature_level[1:])
    opacity = np.zeros((pressure.size, wavelength.size))
    bands = {
        "H2O": ((1.4, 0.13, 1.7), (1.9, 0.12, 1.5), (2.7, 0.15, 2.4), (6.3, 0.18, 3.2)),
        "CO": ((4.7, 0.09, 2.8),),
        "CO2": ((4.3, 0.08, 3.5), (9.4, 0.15, 0.7)),
        "CH4": ((3.3, 0.10, 2.5), (7.7, 0.14, 2.0)),
    }
    baselines = {"H2O": 0.12, "CO": 0.015, "CO2": 0.02, "CH4": 0.02}
    log_wave = np.log(wavelength)
    pressure_scale = np.clip(1.0 + 0.06 * np.log10(pressure / 0.1), 0.65, 1.35)
    temperature_scale = (temperature / 1200.0) ** 0.3
    state_scale = pressure_scale * temperature_scale
    for index, name in enumerate(species):
        spectrum = np.full(wavelength.size, baselines[name])
        for center, width, strength in bands[name]:
            spectrum += strength * np.exp(
                -0.5 * ((log_wave - np.log(center)) / width) ** 2
            )
        opacity += mass_fractions[:, index, None] * state_scale[:, None] * spectrum
    opacity += (
        2.0e-4
        * pressure[:, None]
        * (1200.0 / temperature[:, None]) ** 0.5
        * (wavelength[None, :] / 2.0) ** 0.25
    )
    layer_mass = np.diff(pressure_edges) * 1.0e5 / gravity
    return layer_mass[:, None] * opacity


def _solve_picaso_sh4(
    solver,
    wavelength: np.ndarray,
    pressure_edges: np.ndarray,
    temperature_level: np.ndarray,
    tau: np.ndarray,
    albedo: np.ndarray,
    asymmetry: np.ndarray,
    emission_mu: np.ndarray,
) -> np.ndarray:
    nlevel = pressure_edges.size
    nwave = wavelength.size
    cumulative = np.vstack((np.zeros((1, nwave)), np.cumsum(tau, axis=0)))
    pressure_cgs = pressure_edges * 1.0e6
    pressure_cgs[0] = 0.0
    point, _ = solver(
        nlevel,
        1.0e4 / wavelength,
        nwave,
        emission_mu.size,
        1,
        temperature_level,
        tau,
        cumulative,
        np.maximum(albedo, 1.0e-10),
        asymmetry,
        tau,
        cumulative,
        np.maximum(albedo, 1.0e-10),
        np.maximum(albedo, 1.0e-10),
        asymmetry,
        pressure_cgs,
        emission_mu[:, None],
        np.zeros(nwave),
        4,
        1,
    )
    return np.asarray(point[:, 0, :], dtype=float) * (0.1 / (2.0 * np.pi))


def _planck_radiance(wavelength_micron: np.ndarray, temperature_k: float) -> np.ndarray:
    wavelength_m = wavelength_micron * 1.0e-6
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    exponent = h * c / (wavelength_m * k * temperature_k)
    return 2.0 * h * c**2 / wavelength_m**5 / np.expm1(exponent)


def _version(distribution: str, *, fallback: str | None = None) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        if fallback is None:
            return "unknown"
        try:
            return importlib.metadata.version(fallback)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"


if __name__ == "__main__":
    main()
