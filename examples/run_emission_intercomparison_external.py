"""Run shared-tau or native-opacity emission cases in PICASO or pRT 3."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

_CACHE_ROOT = Path(tempfile.gettempdir()) / "robert-emission-intercomparison"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("OMPI_MCA_btl", "self")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _planck_radiance_wavelength(
    wavelength_micron: np.ndarray, temperature_k: np.ndarray
) -> np.ndarray:
    """Return Planck radiance on temperature-by-wavelength axes."""

    from scipy.constants import c, h, k

    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    temperature = np.asarray(temperature_k, dtype=float)[:, None]
    exponent = h * c / (wavelength_m[None, :] * k * temperature)
    return (
        2.0
        * h
        * c**2
        / wavelength_m[None, :] ** 5
        / np.expm1(exponent)
    )


def _normalize_contribution(values: np.ndarray) -> np.ndarray:
    contribution = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = np.sum(contribution, axis=0, keepdims=True)
    return np.divide(
        contribution,
        total,
        out=np.zeros_like(contribution),
        where=total > 0.0,
    )


def _absorbing_formal_contribution(
    wavelength_micron: np.ndarray,
    temperature_edges_k: np.ndarray,
    layer_tau: np.ndarray,
    g_weights: np.ndarray,
    emission_mu: np.ndarray,
    disk_weights: np.ndarray,
) -> np.ndarray:
    """Return an exact pure-absorption contribution from supplied native tau.

    PICASO exposes its native total optical depths but not source-decomposed
    SH4 layer fluxes.  Stage 4 therefore applies the independently implemented
    absorbing formal solution used in Track A to those native PICASO optical
    depths.  Stage 1 already validates this RT limit across all three codes.
    """

    tau = np.asarray(layer_tau, dtype=float)
    if tau.ndim == 2:
        tau = tau[:, :, None]
    if tau.ndim != 3:
        raise ValueError("PICASO native optical depth must be layer by wavelength by g")
    weights = np.asarray(g_weights, dtype=float)
    weights = weights / np.sum(weights)
    if weights.shape != (tau.shape[2],):
        raise ValueError("PICASO g weights do not match native optical depth")
    source = _planck_radiance_wavelength(
        wavelength_micron, np.asarray(temperature_edges_k, dtype=float)
    )
    layer_contribution = np.zeros(tau.shape[:2], dtype=float)
    bottom_contribution = np.zeros(tau.shape[1], dtype=float)
    for mu, point_weight in zip(emission_mu, disk_weights, strict=True):
        slant_tau = tau / float(mu)
        cumulative_before = np.zeros_like(slant_tau)
        cumulative_before[1:] = np.cumsum(slant_tau[:-1], axis=0)
        transmission_before = np.exp(-cumulative_before)
        escape = -np.expm1(-slant_tau)
        small = np.abs(slant_tau) < 1.0e-5
        linear_weight = np.empty_like(slant_tau)
        value = slant_tau[small]
        linear_weight[small] = (
            value / 2.0
            - value**2 / 3.0
            + value**3 / 8.0
            - value**4 / 30.0
        )
        linear_weight[~small] = (
            escape[~small]
            - slant_tau[~small] * np.exp(-slant_tau[~small])
        ) / slant_tau[~small]
        emitted = (
            source[:-1, :, None] * escape
            + (source[1:, :, None] - source[:-1, :, None]) * linear_weight
        )
        layer_contribution += float(point_weight) * np.sum(
            transmission_before * emitted * weights[None, None, :], axis=-1
        )
        bottom_contribution += float(point_weight) * source[-1] * np.sum(
            np.exp(-np.sum(slant_tau, axis=0)) * weights[None, :], axis=-1
        )
    layer_contribution[-1] += bottom_contribution
    return _normalize_contribution(layer_contribution)


def _shared_picaso(contract: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from picaso.fluxes import get_thermal_1d

    wavelength = contract["wavelength_micron"]
    wavenumber = 1.0e4 / wavelength
    mu = contract["emission_mu"]
    disk_weights = contract["disk_weights"]
    pressure = contract["pressure_edges_bar"].copy() * 1.0e6
    pressure[0] = 0.0
    total_tau = contract["component_tau"].sum(axis=1)
    flux = np.empty((total_tau.shape[0], wavelength.size))
    runtime = np.empty(total_tau.shape[0])
    for case_index, tau in enumerate(total_tau):
        started = perf_counter()
        point, _ = get_thermal_1d(
            pressure.size,
            wavenumber,
            wavelength.size,
            mu.size,
            1,
            contract["temperature_edges_k"][case_index],
            tau,
            np.full_like(tau, 1.0e-10),
            np.zeros_like(tau),
            pressure,
            mu[:, None],
            np.zeros(wavelength.size),
            1,
            np.zeros(wavelength.size),
            0,
        )
        runtime[case_index] = perf_counter() - started
        intensity = point[:, 0, :] * (0.1 / (2.0 * np.pi))
        flux[case_index] = np.pi * np.sum(
            disk_weights[:, None] * intensity, axis=0
        )
    return flux, runtime


def _shared_prt(contract: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.constants import c
    from petitRADTRANS.radtrans import fcore

    wavelength = contract["wavelength_micron"]
    frequencies = c / (wavelength * 1.0e-6)
    mu = contract["emission_mu"]
    angle_weights = contract["legendre_weights"]
    total_tau = contract["component_tau"].sum(axis=1)
    flux = np.empty((total_tau.shape[0], wavelength.size))
    runtime = np.empty(total_tau.shape[0])
    for case_index, tau in enumerate(total_tau):
        cumulative = np.concatenate(
            (
                np.zeros((1, wavelength.size)),
                np.cumsum(tau, axis=0),
            ),
            axis=0,
        )
        optical_depth = cumulative.T[None, :, None, :]
        started = perf_counter()
        flux_nu_cgs, _ = fcore.compute_ck_flux(
            frequencies,
            contract["temperature_edges_k"][case_index],
            np.array([1.0]),
            mu,
            angle_weights,
            optical_depth,
            0,
        )
        runtime[case_index] = perf_counter() - started
        flux[case_index] = flux_nu_cgs * 1.0e-3 * c / (wavelength * 1.0e-6) ** 2
    return flux, runtime


def _mass_fractions(vmr: np.ndarray) -> tuple[dict[str, float], float]:
    names = ("H2", "He", "H2O", "CO", "CO2", "CH4")
    masses = np.array([2.01588, 4.002602, 18.01528, 28.0101, 44.0095, 16.04246])
    mean_molar_mass = float(np.sum(vmr * masses))
    fractions = vmr * masses / mean_molar_mass
    return dict(zip(names, fractions, strict=True)), mean_molar_mass


def _native_prt(
    contract: dict[str, np.ndarray], input_data: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from petitRADTRANS.radtrans import Radtrans

    line_species = (
        "H2O__POKAZATEL",
        "CO__HITEMP",
        "CO2__UCL-4000",
        "CH4__YT34to10",
    )
    cia_species = (
        "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
        "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
    )
    pressure = contract.get("prt_pressure_bar", contract["pressure_edges_bar"])
    temperatures = contract.get(
        "temperature_cells_k", contract["temperature_edges_k"]
    )
    return_contribution = bool(contract.get("native_return_contribution", False))
    atmosphere = Radtrans(
        pressures=pressure,
        wavelength_boundaries=np.array([0.5, 12.0]),
        line_species=list(line_species),
        gas_continuum_contributors=(list(cia_species) if bool(contract["native_include_cia"]) else []),
        scattering_in_emission=False,
        emission_angle_grid=np.vstack(
            (contract["emission_mu"], contract["legendre_weights"])
        ),
        path_input_data=str(input_data),
    )
    output_flux = []
    output_contribution = []
    runtime = []
    wavelength = None
    for case_index, vmr in enumerate(contract["gas_vmr"]):
        fractions, mean_molar_mass = _mass_fractions(vmr)
        mass_fractions = {
            line: np.full(pressure.size, fractions[name])
            for line, name in zip(line_species, ("H2O", "CO", "CO2", "CH4"), strict=True)
        }
        mass_fractions["H2"] = np.full(pressure.size, fractions["H2"])
        mass_fractions["He"] = np.full(pressure.size, fractions["He"])
        started = perf_counter()
        result = atmosphere.calculate_flux(
            temperatures=temperatures[case_index],
            mass_fractions=mass_fractions,
            mean_molar_masses=np.full(pressure.size, mean_molar_mass),
            reference_gravity=1500.0,
            frequencies_to_wavelengths=True,
            return_contribution=return_contribution,
        )
        runtime.append(perf_counter() - started)
        native_wavelength = np.asarray(result[0], dtype=float) * 1.0e4
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        output_flux.append(np.asarray(result[1], dtype=float)[order] * 0.1)
        if return_contribution:
            contribution = np.asarray(
                result[2]["emission_contribution"], dtype=float
            )
            if contribution.shape[0] != pressure.size:
                contribution = contribution.T
            if contribution.shape != (pressure.size, native_wavelength.size):
                raise ValueError(
                    "unexpected pRT contribution shape "
                    f"{contribution.shape}; expected {(pressure.size, native_wavelength.size)}"
                )
            output_contribution.append(_normalize_contribution(contribution[:, order]))
    contribution_array = (
        np.asarray(output_contribution)
        if return_contribution
        else np.empty((len(output_flux), 0, 0))
    )
    return wavelength, np.asarray(output_flux), np.asarray(runtime), contribution_array


def _native_picaso(
    contract: dict[str, np.ndarray], reference: Path, database: Path, resample: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return_contribution = bool(contract.get("native_return_contribution", False))
    output_flux = []
    output_contribution = []
    runtime = []
    wavelength = None
    for case_index, vmr in enumerate(contract["gas_vmr"]):
        case = jdi.inputs(calculation="browndwarf")
        case.gravity(gravity=15.0, gravity_unit=u.m / u.s**2)
        profile = {
            "pressure": pressure,
            "temperature": contract["temperature_edges_k"][case_index],
            "H2": np.full(pressure.size, vmr[0]),
            "He": np.full(pressure.size, vmr[1]),
            "H2O": np.full(pressure.size, vmr[2]),
            "CO": np.full(pressure.size, vmr[3]),
            "CO2": np.full(pressure.size, vmr[4]),
            "CH4": np.full(pressure.size, vmr[5]),
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
        started = perf_counter()
        result = case.spectrum(
            opacity,
            calculation="thermal",
            full_output=return_contribution,
        )
        runtime.append(perf_counter() - started)
        wavelength = 1.0e4 / np.asarray(result["wavenumber"], dtype=float)
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        output_flux.append(np.asarray(result["thermal"], dtype=float)[order] * 0.1)
        if return_contribution:
            full_output = result["full_output"]
            native_tau = np.asarray(full_output["taugas"], dtype=float)
            native_tau += np.asarray(full_output["taucld"], dtype=float)
            native_tau += np.asarray(full_output["tauray"], dtype=float)
            contribution = _absorbing_formal_contribution(
                1.0e4 / np.asarray(result["wavenumber"], dtype=float),
                contract["temperature_edges_k"][case_index],
                native_tau,
                np.asarray(opacity.gauss_wts, dtype=float),
                contract["emission_mu"],
                contract["disk_weights"],
            )
            output_contribution.append(contribution[:, order])
    contribution_array = (
        np.asarray(output_contribution)
        if return_contribution
        else np.empty((len(output_flux), 0, 0))
    )
    return wavelength, np.asarray(output_flux), np.asarray(runtime), contribution_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("picaso", "petitradtrans"))
    parser.add_argument("mode", choices=("shared", "native"))
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--picaso-reference", type=Path)
    parser.add_argument("--picaso-database", type=Path)
    parser.add_argument("--picaso-resample", type=int, default=50)
    args = parser.parse_args()
    contract = _load(args.contract)
    if args.mode == "shared":
        flux, runtime = (
            _shared_picaso(contract)
            if args.model == "picaso"
            else _shared_prt(contract)
        )
        wavelength = contract["wavelength_micron"]
    elif args.model == "picaso":
        if args.picaso_reference is None or args.picaso_database is None:
            parser.error("PICASO native mode requires reference and database paths")
        wavelength, flux, runtime, contribution = _native_picaso(
            contract,
            args.picaso_reference,
            args.picaso_database,
            args.picaso_resample,
        )
    else:
        if args.input_data is None:
            parser.error("pRT native mode requires --input-data")
        wavelength, flux, runtime, contribution = _native_prt(contract, args.input_data)
    metadata = {
        "model": args.model,
        "mode": args.mode,
        "version": importlib.metadata.version(
            "picaso" if args.model == "picaso" else "petitRADTRANS"
        ),
        "python": os.sys.executable,
        "native_include_cia": (
            bool(contract["native_include_cia"])
            if "native_include_cia" in contract
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "case_id": contract["case_id"],
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "runtime_s": runtime,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    if args.mode == "native" and bool(
        contract.get("native_return_contribution", False)
    ):
        output["pressure_bar"] = contract.get(
            "prt_pressure_bar", contract["pressure_centers_bar"]
        )
        output["normalized_contribution"] = contribution
    np.savez_compressed(
        args.output,
        **output,
    )


if __name__ == "__main__":
    main()
