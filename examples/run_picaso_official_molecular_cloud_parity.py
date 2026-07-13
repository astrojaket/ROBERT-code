"""Run the official PICASO molecular-opacity cloud benchmark externally.

This script is intentionally independent of ROBERT.  It reads only the shared
physical contract, queries the official PICASO SQLite opacity database, computes
Virga Mie optics, and runs PICASO emission and transmission calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np


G_SI = 6.67430e-11
SPECIES = ("H2O", "CO", "CO2", "CH4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opacity-db", type=Path, required=True)
    parser.add_argument("--resample", type=int, default=1)
    args = parser.parse_args()
    if args.resample < 1:
        raise ValueError("resample must be positive")
    evaluate(args.input, args.output, args.opacity_db, args.resample)


def evaluate(contract_path: Path, output_path: Path, opacity_db: Path, resample: int) -> None:
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi
    from picaso.optics import compute_opacity
    from virga.calc_mie import calc_new_mieff

    with np.load(contract_path, allow_pickle=False) as payload:
        contract = {name: np.array(payload[name], copy=True) for name in payload.files}

    started = perf_counter()
    qext, qsca, g_qsca = calc_new_mieff(
        contract["wavelength_micron"],
        contract["refractive_index_n"],
        contract["refractive_index_k"],
        contract["radius_cm"],
        contract["radius_upper_cm"],
        fort_calc_mie=False,
    )
    mie = _integrate_population(
        qext,
        qsca,
        g_qsca,
        contract["radius_cm"],
        contract["radius_number_weights"],
        float(contract["particle_density_kg_m3"]),
    )
    mie_seconds = perf_counter() - started

    started = perf_counter()
    opacity = jdi.opannection(
        filename_db=str(opacity_db),
        wave_range=[
            float(contract["wavelength_micron"][0]),
            float(contract["wavelength_micron"][-1]),
        ],
        resample=resample,
        verbose=False,
    )
    opacity_seconds = perf_counter() - started

    cloudy_case = _build_case(jdi, u, pd, contract, mie, cloudy=True)
    started = perf_counter()
    cloudy = cloudy_case.spectrum(
        opacity,
        calculation="thermal",
        full_output=True,
        as_dict=False,
    )
    cloudy_thermal_seconds = perf_counter() - started
    atmosphere = cloudy["full_output"]
    contributions = compute_opacity(
        atmosphere,
        opacity,
        ngauss=1,
        stream=4,
        delta_eddington=False,
        raman=2,
        return_mode=True,
    )
    gas_tau = np.asarray(atmosphere.taugas[:, :, 0], dtype=float)
    cloud_tau = np.asarray(atmosphere.taucld[:, :, 0], dtype=float)

    clear_case = _build_case(jdi, u, pd, contract, mie, cloudy=False)
    started = perf_counter()
    clear = clear_case.spectrum(opacity, calculation="thermal", full_output=False)
    clear_thermal_seconds = perf_counter() - started

    star_radius_cm = float(contract["star_radius_m"]) * 100.0
    cloudy_case.inputs["star"]["radius"] = star_radius_cm
    cloudy_case.inputs["star"]["radius_unit"] = "cm"
    clear_case.inputs["star"]["radius"] = star_radius_cm
    clear_case.inputs["star"]["radius_unit"] = "cm"
    started = perf_counter()
    cloudy_transmission = cloudy_case.spectrum(
        opacity, calculation="transmission", full_output=False
    )
    cloudy_transmission_seconds = perf_counter() - started
    started = perf_counter()
    clear_transmission = clear_case.spectrum(
        opacity, calculation="transmission", full_output=False
    )
    clear_transmission_seconds = perf_counter() - started

    wavelength = 1.0e4 / np.asarray(cloudy["wavenumber"], dtype=float)
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    thermal_cloudy = np.asarray(cloudy["thermal"], dtype=float)[order]
    thermal_clear = np.asarray(clear["thermal"], dtype=float)[order]
    stellar_flux = np.pi * _planck_radiance(wavelength, float(contract["star_temperature_k"])) * 10.0
    area_ratio = (float(contract["planet_radius_m"]) / float(contract["star_radius_m"])) ** 2

    species_tau = np.stack(
        [np.asarray(contributions[name], dtype=float)[:, order] for name in SPECIES],
        axis=0,
    )
    continuum_names = tuple(
        name for name in ("H2H2", "H2He") if name in contributions
    )
    continuum_tau = (
        np.sum(
            np.stack(
                [np.asarray(contributions[name], dtype=float)[:, order] for name in continuum_names],
                axis=0,
            ),
            axis=0,
        )
        if continuum_names
        else np.zeros_like(gas_tau[:, order])
    )
    metadata = {
        "picaso_version": importlib.metadata.version("picaso"),
        "virga_version": _distribution_version("virga-exo", "virga"),
        "opacity_database": str(opacity_db.resolve()),
        "opacity_database_size_bytes": opacity_db.stat().st_size,
        "opacity_database_sha256": _sha256(opacity_db),
        "opacity_resample_stride": resample,
        "native_wavelength_count": int(wavelength.size),
        "molecular_species": list(SPECIES),
        "continuum_species": list(continuum_names),
        "molecular_query": "PICASO log-opacity bilinear P-T interpolation",
        "cloud_mie_solver": "virga.calc_mie.calc_new_mieff_PyMieScatt",
        "thermal_solver": "PICASO SH4",
        "transmission_solver": "PICASO spherical extinction",
        "phase_closure": "Henyey-Greenstein, delta-M disabled",
        "timing_seconds": {
            "mie": mie_seconds,
            "opacity_connection": opacity_seconds,
            "cloudy_thermal": cloudy_thermal_seconds,
            "clear_thermal": clear_thermal_seconds,
            "cloudy_transmission": cloudy_transmission_seconds,
            "clear_transmission": clear_transmission_seconds,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        wavelength_micron=wavelength,
        cloud_wavelength_micron=contract["wavelength_micron"],
        mie_qext=qext,
        mie_qsca=qsca,
        mie_g_qsca=g_qsca,
        mass_extinction_m2_kg=mie["mass_extinction_m2_kg"],
        single_scattering_albedo=mie["single_scattering_albedo"],
        asymmetry_factor=mie["asymmetry_factor"],
        gas_tau=gas_tau[:, order],
        molecular_tau_by_species=species_tau,
        continuum_tau=continuum_tau,
        cloud_tau=cloud_tau[:, order],
        cloudy_thermal_flux_cgs=thermal_cloudy,
        clear_thermal_flux_cgs=thermal_clear,
        cloudy_eclipse_depth=thermal_cloudy / stellar_flux * area_ratio,
        clear_eclipse_depth=thermal_clear / stellar_flux * area_ratio,
        cloudy_transit_depth=np.asarray(cloudy_transmission["transit_depth"], dtype=float)[order],
        clear_transit_depth=np.asarray(clear_transmission["transit_depth"], dtype=float)[order],
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def _build_case(jdi, u, pd, contract, mie, *, cloudy: bool):
    pressure_edges = contract["pressure_edges_bar"]
    pressure_layer = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    case = jdi.inputs(calculation="browndwarf")
    planet_radius = float(contract["planet_radius_m"])
    gravity = float(contract["gravity_m_s2"])
    planet_mass = gravity * planet_radius**2 / G_SI
    case.gravity(
        mass=planet_mass,
        mass_unit=u.kg,
        radius=planet_radius,
        radius_unit=u.m,
    )
    gas_vmr_layer = np.asarray(contract["gas_vmr"], dtype=float)
    profile = {
        "pressure": pressure_edges,
        "temperature": contract["temperature_level_k"],
    }
    log_edges = np.log(pressure_edges)
    log_layers = np.log(pressure_layer)
    for index, name in enumerate(SPECIES):
        profile[name] = np.interp(
            log_edges,
            log_layers,
            gas_vmr_layer[:, index],
            left=gas_vmr_layer[0, index],
            right=gas_vmr_layer[-1, index],
        )
    profile["H2"] = np.full(pressure_edges.size, float(contract["h2_vmr"]))
    profile["He"] = 1.0 - profile["H2"] - sum(profile[name] for name in SPECIES)
    case.atmosphere(df=pd.DataFrame(profile), verbose=False)
    if cloudy:
        layer_mass = np.diff(pressure_edges) * 1.0e5 / gravity
        cloud_tau = (
            layer_mass[:, None]
            * contract["condensate_mass_fraction"][:, None]
            * mie["mass_extinction_m2_kg"][None, :]
        )
        rows = []
        for layer_index, pressure in enumerate(pressure_layer):
            for wave_index, wavelength in enumerate(contract["wavelength_micron"]):
                rows.append(
                    (
                        pressure,
                        1.0e4 / wavelength,
                        cloud_tau[layer_index, wave_index],
                        mie["single_scattering_albedo"][wave_index],
                        mie["asymmetry_factor"][wave_index],
                    )
                )
        case.clouds(
            df=pd.DataFrame(
                rows,
                columns=("pressure", "wavenumber", "opd", "w0", "g0"),
            )
        )
    case.approx(
        rt_method="SH",
        stream=4,
        delta_eddington=False,
        raman="none",
        query="interp",
        p_reference=float(pressure_edges[-1]),
        w_single_rayleigh="off",
        w_multi_rayleigh="off",
        psingle_rayleigh="off",
    )
    return case


def _integrate_population(qext, qsca, g_qsca, radius_cm, weights, density):
    radius_m = radius_cm * 1.0e-2
    area = np.pi * radius_m**2
    mass = (4.0 / 3.0) * np.pi * density * radius_m**3
    mean_mass = np.sum(weights * mass)
    extinction = np.sum(qext * (weights * area)[None, :], axis=1)
    scattering = np.sum(qsca * (weights * area)[None, :], axis=1)
    scattering_g = np.sum(g_qsca * (weights * area)[None, :], axis=1)
    mass_extinction = extinction / mean_mass
    mass_scattering = scattering / mean_mass
    return {
        "mass_extinction_m2_kg": mass_extinction,
        "single_scattering_albedo": np.divide(
            mass_scattering,
            mass_extinction,
            out=np.zeros_like(mass_extinction),
            where=mass_extinction > 0.0,
        ),
        "asymmetry_factor": np.divide(
            scattering_g,
            scattering,
            out=np.zeros_like(scattering),
            where=scattering > 0.0,
        ),
    }


def _planck_radiance(wavelength_micron, temperature):
    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    exponent = h * c / (wavelength_m * k * temperature)
    return 2.0 * h * c**2 / wavelength_m**5 / np.expm1(exponent)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_version(primary: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(primary)
    except importlib.metadata.PackageNotFoundError:
        return importlib.metadata.version(fallback)


if __name__ == "__main__":
    main()
