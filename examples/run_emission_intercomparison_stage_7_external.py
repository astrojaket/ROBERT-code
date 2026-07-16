"""Run process-isolated Stage-7 absorbing-cloud cases in PICASO or pRT."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from typing import Callable

import numpy as np

_CACHE_ROOT = Path(tempfile.gettempdir()) / "robert-emission-intercomparison-stage7"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("OMPI_MCA_btl", "self")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _planck_radiance(
    wavelength_micron: np.ndarray, temperature_k: np.ndarray
) -> np.ndarray:
    from scipy.constants import c, h, k

    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    temperature = np.asarray(temperature_k, dtype=float)[:, None]
    exponent = h * c / (wavelength_m[None, :] * k * temperature)
    return 2.0 * h * c**2 / wavelength_m[None, :] ** 5 / np.expm1(exponent)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = np.sum(array, axis=-2, keepdims=True)
    return np.divide(array, total, out=np.zeros_like(array), where=total > 0.0)


def _formal_contribution(
    wavelength_micron: np.ndarray,
    temperature_edges_k: np.ndarray,
    layer_tau: np.ndarray,
    g_weights: np.ndarray,
    emission_mu: np.ndarray,
    disk_weights: np.ndarray,
) -> np.ndarray:
    """Return the Stage-1 pure-absorption layer source decomposition."""

    tau = np.asarray(layer_tau, dtype=float)
    if tau.ndim == 2:
        tau = tau[:, :, None]
    weights = np.asarray(g_weights, dtype=float)
    weights /= np.sum(weights)
    source = _planck_radiance(wavelength_micron, temperature_edges_k)
    layers = np.zeros(tau.shape[:2])
    bottom = np.zeros(tau.shape[1])
    for mu, weight in zip(emission_mu, disk_weights, strict=True):
        slant = tau / float(mu)
        before = np.zeros_like(slant)
        before[1:] = np.cumsum(slant[:-1], axis=0)
        transmission = np.exp(-before)
        escape = -np.expm1(-slant)
        small = np.abs(slant) < 1.0e-5
        linear = np.empty_like(slant)
        value = slant[small]
        linear[small] = (
            value / 2.0 - value**2 / 3.0 + value**3 / 8.0 - value**4 / 30.0
        )
        linear[~small] = (
            escape[~small] - slant[~small] * np.exp(-slant[~small])
        ) / slant[~small]
        emitted = (
            source[:-1, :, None] * escape
            + (source[1:, :, None] - source[:-1, :, None]) * linear
        )
        layers += float(weight) * np.sum(
            transmission * emitted * weights[None, None, :], axis=-1
        )
        bottom += (
            float(weight)
            * source[-1]
            * np.sum(np.exp(-np.sum(slant, axis=0)) * weights[None, :], axis=-1)
        )
    layers[-1] += bottom
    return _normalize(layers)


def _shared_total_tau(contract: dict[str, np.ndarray]) -> np.ndarray:
    profile_index = np.asarray(contract["profile_index"], dtype=int)
    cloud_index = np.asarray(contract["case_cloud_index"], dtype=int)
    return (
        np.asarray(contract["shared_gas_tau"], dtype=float)[profile_index]
        + np.asarray(contract["cloud_extinction_tau"], dtype=float)[cloud_index]
    )


def _run_shared_picaso(
    contract: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from picaso.fluxes import get_thermal_1d

    wavelength = contract["wavelength_micron"]
    pressure = contract["pressure_edges_bar"].copy() * 1.0e6
    pressure[0] = 0.0
    total_tau = _shared_total_tau(contract)
    flux = np.empty((contract["case_id"].size, wavelength.size))
    contribution = np.empty(
        (contract["case_id"].size, total_tau.shape[1], wavelength.size)
    )
    runtime = np.empty(contract["case_id"].size)
    for case_index, tau in enumerate(total_tau):
        started = perf_counter()
        point, _ = get_thermal_1d(
            pressure.size,
            1.0e4 / wavelength,
            wavelength.size,
            contract["emission_mu"].size,
            1,
            contract["temperature_edges_k"][case_index],
            tau,
            np.zeros_like(tau),
            np.zeros_like(tau),
            pressure,
            contract["emission_mu"][:, None],
            np.zeros(wavelength.size),
            1,
            np.zeros(wavelength.size),
            0,
        )
        runtime[case_index] = perf_counter() - started
        intensity = point[:, 0, :] * (0.1 / (2.0 * np.pi))
        flux[case_index] = np.pi * np.sum(
            contract["disk_weights"][:, None] * intensity, axis=0
        )
        contribution[case_index] = _formal_contribution(
            wavelength,
            contract["temperature_edges_k"][case_index],
            tau,
            np.array([1.0]),
            contract["emission_mu"],
            contract["disk_weights"],
        )
    return flux, contribution, runtime, total_tau


def _run_shared_prt(
    contract: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.constants import c
    from petitRADTRANS.radtrans import fcore

    wavelength = contract["wavelength_micron"]
    frequencies = c / (wavelength * 1.0e-6)
    total_tau = _shared_total_tau(contract)
    flux = np.empty((contract["case_id"].size, wavelength.size))
    contribution = np.empty(
        (contract["case_id"].size, total_tau.shape[1], wavelength.size)
    )
    runtime = np.empty(contract["case_id"].size)
    for case_index, tau in enumerate(total_tau):
        cumulative = np.concatenate(
            (np.zeros((1, wavelength.size)), np.cumsum(tau, axis=0)), axis=0
        )
        started = perf_counter()
        flux_nu_cgs, _ = fcore.compute_ck_flux(
            frequencies,
            contract["temperature_edges_k"][case_index],
            np.array([1.0]),
            contract["emission_mu"],
            contract["legendre_weights"],
            cumulative.T[None, :, None, :],
            0,
        )
        runtime[case_index] = perf_counter() - started
        flux[case_index] = flux_nu_cgs * 1.0e-3 * c / (wavelength * 1.0e-6) ** 2
        contribution[case_index] = _formal_contribution(
            wavelength,
            contract["temperature_edges_k"][case_index],
            tau,
            np.array([1.0]),
            contract["emission_mu"],
            contract["disk_weights"],
        )
    return flux, contribution, runtime, total_tau


def _cloud_dataframe(pd, contract: dict[str, np.ndarray], cloud_index: int):
    pressure = contract["pressure_centers_bar"]
    wavelength = contract["wavelength_micron"]
    tau = contract["cloud_extinction_tau"][cloud_index]
    rows = [
        (pressure[layer], 1.0e4 / wavelength[wave], tau[layer, wave], 0.0, 0.0)
        for layer in range(pressure.size)
        for wave in range(wavelength.size)
    ]
    return pd.DataFrame(rows, columns=("pressure", "wavenumber", "opd", "w0", "g0"))


def _run_native_picaso(
    contract: dict[str, np.ndarray], reference: Path, database: Path, resample: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    os.environ["picaso_refdata"] = str(reference.resolve())
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi

    opacity = jdi.opannection(
        filename_db=str(database),
        wave_range=[0.5, 12.0],
        resample=resample,
        verbose=False,
    )
    pressure = contract["pressure_edges_bar"]
    names = ("H2", "He", "H2O", "CO", "CO2", "CH4")
    output_flux: list[np.ndarray] = []
    output_contribution: list[np.ndarray] = []
    output_cloud_tau: list[np.ndarray] = []
    runtime: list[float] = []
    wavelength = None
    maximum_rayleigh_tau = 0.0
    for case_index, cloud_index in enumerate(contract["case_cloud_index"]):
        profile_index = int(contract["profile_index"][case_index])
        case = jdi.inputs(calculation="browndwarf")
        case.gravity(gravity=15.0, gravity_unit=u.m / u.s**2)
        vmr = contract["gas_vmr"][case_index]
        profile = {
            "pressure": pressure,
            "temperature": contract["temperature_edges_by_profile_k"][profile_index],
            **{name: np.full(pressure.size, vmr[index]) for index, name in enumerate(names)},
        }
        case.atmosphere(df=pd.DataFrame(profile), verbose=False)
        case.approx(
            rt_method="SH",
            stream=4,
            delta_eddington=False,
            raman="none",
            query="interp",
            w_single_rayleigh="off",
            w_multi_rayleigh="off",
            psingle_rayleigh="off",
        )
        if int(cloud_index) != 0:
            case.clouds(df=_cloud_dataframe(pd, contract, int(cloud_index)))
        started = perf_counter()
        result = case.spectrum(opacity, calculation="thermal", full_output=True)
        runtime.append(perf_counter() - started)
        native_wavelength = 1.0e4 / np.asarray(result["wavenumber"], dtype=float)
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        output_flux.append(np.asarray(result["thermal"], dtype=float)[order] * 0.1)
        full = result["full_output"]
        gas_tau = np.asarray(full["taugas"], dtype=float)
        cloud_tau = np.asarray(full["taucld"], dtype=float)
        rayleigh_tau = np.asarray(full["tauray"], dtype=float)
        maximum_rayleigh_tau = max(maximum_rayleigh_tau, float(np.max(np.abs(rayleigh_tau))))
        native_total = gas_tau + cloud_tau
        output_contribution.append(
            _formal_contribution(
                native_wavelength,
                contract["temperature_edges_by_profile_k"][profile_index],
                native_total,
                np.asarray(opacity.gauss_wts, dtype=float),
                contract["emission_mu"],
                contract["disk_weights"],
            )[:, order]
        )
        weights = np.asarray(opacity.gauss_wts, dtype=float)
        weights /= np.sum(weights)
        if cloud_tau.ndim == 3:
            cloud_mean = np.sum(cloud_tau * weights[None, None, :], axis=-1)
        else:
            cloud_mean = cloud_tau
        output_cloud_tau.append(cloud_mean[:, order])
    return (
        np.asarray(wavelength),
        np.asarray(output_flux),
        np.asarray(output_contribution),
        np.asarray(output_cloud_tau),
        np.asarray(runtime),
        maximum_rayleigh_tau,
    )


def _mass_fractions(vmr: np.ndarray) -> tuple[dict[str, float], float]:
    names = ("H2", "He", "H2O", "CO", "CO2", "CH4")
    masses = np.array([2.01588, 4.002602, 18.01528, 28.0101, 44.0095, 16.04246])
    mean = float(np.sum(vmr * masses))
    fractions = vmr * masses / mean
    return dict(zip(names, fractions, strict=True)), mean


def _prt_cloud_opacity_function(
    contract: dict[str, np.ndarray], cloud_index: int, gravity_cgs: float
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    kind = str(contract["cloud_kind"][cloud_index])
    if kind == "clear":
        return lambda wavelength, pressure: np.zeros((wavelength.size, pressure.size))
    if kind == "power_law_deck":
        tau_ref = float(contract["cloud_optical_depth_at_reference"][cloud_index])
        top = float(contract["cloud_top_pressure_bar"][cloud_index])
        slope = float(contract["cloud_extinction_slope"][cloud_index])
        reference = float(contract["cloud_reference_wavelength_micron"])
        bottom = float(contract["pressure_edges_bar"][-1])

        def power_law(wavelength: np.ndarray, pressure: np.ndarray) -> np.ndarray:
            spectral = tau_ref * (np.asarray(wavelength) / reference) ** slope
            vertical = np.where(
                np.asarray(pressure) >= top,
                gravity_cgs / (np.asarray(pressure) * 1.0e6 * np.log(bottom / top)),
                0.0,
            )
            return spectral[:, None] * vertical[None, :]

        return power_law

    source_edges = np.asarray(contract["archived_pressure_edges_bar"], dtype=float)
    source_pressure = np.sqrt(source_edges[:-1] * source_edges[1:])
    source_wavelength = np.asarray(contract["archived_wavelength_micron"], dtype=float)
    source_tau = np.asarray(contract["archived_extinction_tau"], dtype=float)
    source_opacity = (
        source_tau
        * gravity_cgs
        / (np.diff(source_edges)[:, None] * 1.0e6)
    )

    def archived(wavelength: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        requested_wavelength = np.asarray(wavelength, dtype=float)
        requested_pressure = np.asarray(pressure, dtype=float)
        spectral = np.empty((source_pressure.size, requested_wavelength.size))
        tiny = np.finfo(float).tiny
        for layer_index, values in enumerate(source_opacity):
            spectral[layer_index] = np.exp(
                np.interp(
                    np.log(requested_wavelength),
                    np.log(source_wavelength),
                    np.log(np.maximum(values, tiny)),
                    left=np.log(max(values[0], tiny)),
                    right=np.log(max(values[-1], tiny)),
                )
            )
        output = np.empty((requested_wavelength.size, requested_pressure.size))
        for wave_index in range(requested_wavelength.size):
            output[wave_index] = np.exp(
                np.interp(
                    np.log(requested_pressure),
                    np.log(source_pressure),
                    np.log(np.maximum(spectral[:, wave_index], tiny)),
                    left=np.log(max(spectral[0, wave_index], tiny)),
                    right=np.log(max(spectral[-1, wave_index], tiny)),
                )
            )
        return output

    return archived


def _run_native_prt(
    contract: dict[str, np.ndarray], input_data: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from petitRADTRANS.radtrans import Radtrans

    line_species = (
        "H2O__POKAZATEL",
        "CO__HITEMP",
        "CO2__UCL-4000",
        "CH4__YT34to10",
    )
    pressure = contract["prt_pressure_bar"]
    atmosphere = Radtrans(
        pressures=pressure,
        wavelength_boundaries=np.array([0.5, 12.0]),
        line_species=list(line_species),
        gas_continuum_contributors=[
            "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
            "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
        ],
        rayleigh_species=[],
        cloud_species=[],
        scattering_in_emission=False,
        anisotropic_cloud_scattering=False,
        emission_angle_grid=np.vstack(
            (contract["emission_mu"], contract["legendre_weights"])
        ),
        path_input_data=str(input_data),
    )
    fractions, mean_molar_mass = _mass_fractions(contract["gas_vmr"][0])
    mass_fractions = {
        line: np.full(pressure.size, fractions[name])
        for line, name in zip(line_species, ("H2O", "CO", "CO2", "CH4"), strict=True)
    }
    mass_fractions["H2"] = np.full(pressure.size, fractions["H2"])
    mass_fractions["He"] = np.full(pressure.size, fractions["He"])
    flux: list[np.ndarray] = []
    contribution: list[np.ndarray] = []
    cloud_tau: list[np.ndarray] = []
    runtime: list[float] = []
    wavelength = None
    gravity_cgs = 1500.0
    layer_pressure_cgs = np.diff(contract["pressure_edges_bar"]) * 1.0e6
    for case_index, cloud_index in enumerate(contract["case_cloud_index"]):
        profile_index = int(contract["profile_index"][case_index])
        opacity_function = _prt_cloud_opacity_function(
            contract, int(cloud_index), gravity_cgs
        )
        started = perf_counter()
        result = atmosphere.calculate_flux(
            temperatures=contract["temperature_cells_by_profile_k"][profile_index],
            mass_fractions=mass_fractions,
            mean_molar_masses=np.full(pressure.size, mean_molar_mass),
            reference_gravity=gravity_cgs,
            additional_absorption_opacities_function=opacity_function,
            additional_scattering_opacities_function=None,
            frequencies_to_wavelengths=True,
            return_contribution=True,
        )
        runtime.append(perf_counter() - started)
        native_wavelength = np.asarray(result[0], dtype=float) * 1.0e4
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        flux.append(np.asarray(result[1], dtype=float)[order] * 0.1)
        native_contribution = np.asarray(result[2]["emission_contribution"], dtype=float)
        if native_contribution.shape[0] != pressure.size:
            native_contribution = native_contribution.T
        contribution.append(_normalize(native_contribution[:, order]))
        node_opacity = opacity_function(native_wavelength, pressure).T
        cloud_tau.append(
            (node_opacity * layer_pressure_cgs[:, None] / gravity_cgs)[:, order]
        )
    return (
        np.asarray(wavelength),
        np.asarray(flux),
        np.asarray(contribution),
        np.asarray(cloud_tau),
        np.asarray(runtime),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("picaso", "petitradtrans"))
    parser.add_argument("track", choices=("shared", "native"))
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--picaso-reference", type=Path)
    parser.add_argument("--picaso-database", type=Path)
    parser.add_argument("--picaso-resample", type=int, default=50)
    args = parser.parse_args()
    contract = _load(args.contract)
    started = perf_counter()
    maximum_rayleigh_tau = 0.0
    if args.track == "shared":
        flux, contribution, runtime, _ = (
            _run_shared_picaso(contract)
            if args.model == "picaso"
            else _run_shared_prt(contract)
        )
        wavelength = contract["wavelength_micron"]
        cloud_tau = contract["cloud_extinction_tau"][contract["case_cloud_index"]]
    elif args.model == "picaso":
        if args.picaso_reference is None or args.picaso_database is None:
            parser.error("native PICASO requires reference and database paths")
        (
            wavelength,
            flux,
            contribution,
            cloud_tau,
            runtime,
            maximum_rayleigh_tau,
        ) = _run_native_picaso(
            contract,
            args.picaso_reference,
            args.picaso_database,
            args.picaso_resample,
        )
    else:
        if args.input_data is None:
            parser.error("native pRT requires --input-data")
        wavelength, flux, contribution, cloud_tau, runtime = _run_native_prt(
            contract, args.input_data
        )
    metadata = {
        "model": args.model,
        "track": args.track,
        "version": importlib.metadata.version(
            "picaso" if args.model == "picaso" else "petitRADTRANS"
        ),
        "python": os.sys.executable,
        "omega0": 0.0,
        "scattering_in_emission": False,
        "rayleigh_scattering": "explicitly_off",
        "delta_m": False,
        "maximum_native_rayleigh_tau": maximum_rayleigh_tau,
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": contract["case_id"],
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "normalized_contribution": contribution,
        "cloud_extinction_tau": cloud_tau,
        "pressure_bar": contract["pressure_centers_bar"],
        "runtime_s": runtime,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    np.savez_compressed(args.output, **payload)


if __name__ == "__main__":
    main()
