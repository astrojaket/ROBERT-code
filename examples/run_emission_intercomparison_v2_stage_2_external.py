"""Run one isolated Version-2 Stage-2 PICASO or pRT worker."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Any

import numpy as np


PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PICASO_REFERENCE = Path("/Users/jaketaylor/Dropbox/picaso-v4/reference")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = np.sum(array, axis=-2, keepdims=True)
    return np.divide(array, total, out=np.zeros_like(array), where=total > 0.0)


def _planck_radiance(
    wavelength_micron: np.ndarray, temperature_k: np.ndarray
) -> np.ndarray:
    from scipy.constants import c, h, k

    wavelength_m = np.asarray(wavelength_micron) * 1.0e-6
    temperature = np.asarray(temperature_k)[:, None]
    exponent = h * c / (wavelength_m[None, :] * k * temperature)
    return 2.0 * h * c**2 / wavelength_m[None, :] ** 5 / np.expm1(exponent)


def _formal_contribution(
    wavelength_micron: np.ndarray,
    temperature_edges_k: np.ndarray,
    layer_tau: np.ndarray,
    g_weights: np.ndarray,
    emission_mu: np.ndarray,
    disk_weights: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Return a labelled absorbing-formal vertical diagnostic."""

    tau = np.asarray(layer_tau, dtype=float)
    if tau.ndim == 2:
        tau = tau[:, :, None]
    weights = np.asarray(g_weights, dtype=float)
    weights /= np.sum(weights)
    source = _planck_radiance(wavelength_micron, temperature_edges_k)
    layers = np.zeros(tau.shape[:2])
    bottom = np.zeros(tau.shape[1])
    for mu, point_weight in zip(emission_mu, disk_weights, strict=True):
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
        layers += float(point_weight) * np.sum(
            transmission * emitted * weights[None, None, :], axis=-1
        )
        bottom += (
            float(point_weight)
            * source[-1]
            * np.sum(np.exp(-np.sum(slant, axis=0)) * weights[None, :], axis=-1)
        )
    layers[-1] += bottom
    return _normalize(layers) if normalize else layers


def _molecular_only_get_opacities(opacity: Any) -> None:
    """Exclude CIA/Rayleigh while retaining H2/He in PICASO's MMW state."""

    original = opacity.get_opacities

    def molecular_only(atmosphere: Any, exclude_mol: Any = 1) -> Any:
        atmosphere.continuum_molecules = []
        atmosphere.rayleigh_molecules = []
        return original(atmosphere, exclude_mol=exclude_mol)

    opacity.get_opacities = molecular_only


def _restore_resort_rebin_absolute_vmr(opacity: Any) -> None:
    """Restore the absolute line-gas VMR discarded by PICASO's mixer."""

    original = opacity.mix_my_opacities_gasesfly

    def absolute_vmr_mixer(atmosphere: Any, exclude_mol: Any = 1) -> Any:
        result = original(atmosphere, exclude_mol=exclude_mol)
        total_vmr = np.sum(
            [
                atmosphere.layer["mixingratios"][molecule].values
                for molecule in atmosphere.molecules
                if exclude_mol == 1 or exclude_mol[molecule] == 1
            ],
            axis=0,
        )
        opacity.molecular_opa *= total_vmr[:, None, None]
        return result

    opacity.mix_my_opacities_gasesfly = absolute_vmr_mixer


def _validate_picaso_environment() -> dict[str, str]:
    required = {
        "picaso_refdata": PICASO_REFERENCE,
        "NUMBA_CACHE_DIR": None,
        "MPLCONFIGDIR": None,
    }
    resolved: dict[str, str] = {}
    for name, expected in required.items():
        raw = os.environ.get(name)
        if not raw:
            raise RuntimeError(f"PICASO requires explicit {name}")
        path = Path(raw)
        if expected is not None and path.resolve() != expected.resolve():
            raise RuntimeError(f"{name} must be exactly {expected}")
        if name != "picaso_refdata" and (not path.is_dir() or not os.access(path, os.W_OK)):
            raise RuntimeError(f"{name} must name an existing writable directory")
        resolved[name] = raw
    return resolved


def _picaso(
    contract: dict[str, np.ndarray],
    *,
    ck_directory: Path,
) -> dict[str, np.ndarray]:
    _validate_picaso_environment()
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi

    species = [str(value) for value in contract["species_name"]]
    preload = list(dict.fromkeys(species))
    opacity = jdi.opannection(
        method="resortrebin",
        ck_db=str(ck_directory),
        preload_gases=preload,
        wave_range=[0.3, 12.0],
        verbose=False,
    )
    _restore_resort_rebin_absolute_vmr(opacity)
    _molecular_only_get_opacities(opacity)
    output_flux: list[np.ndarray] = []
    output_native_probe_flux: list[np.ndarray] = []
    output_tau: list[np.ndarray] = []
    output_contribution: list[np.ndarray] = []
    timings: list[float] = []
    maximum_rayleigh_tau = 0.0
    maximum_cloud_tau = 0.0
    wavelength = np.empty(0)
    gas_names = [str(value) for value in contract["gas_name"]]
    for case_index, molecule in enumerate(species):
        profile = {
            "pressure": contract["pressure_edges_bar"],
            "temperature": contract["temperature_edges_k"][case_index],
        }
        for gas_index, gas_name in enumerate(gas_names):
            value = float(contract["gas_vmr"][case_index, gas_index])
            if value > 0.0:
                profile[gas_name] = np.full(contract["pressure_edges_bar"].size, value)
        case = jdi.inputs(calculation="browndwarf")
        case.gravity(
            gravity=float(contract["gravity_m_s2"]), gravity_unit=u.m / u.s**2
        )
        case.atmosphere(df=pd.DataFrame(profile), verbose=False)
        case.approx(
            rt_method="SH",
            stream=4,
            delta_eddington=False,
            raman="none",
            w_single_rayleigh="off",
            w_multi_rayleigh="off",
            psingle_rayleigh="off",
        )
        started = perf_counter()
        result = case.spectrum(opacity, calculation="thermal", full_output=True)
        timings.append(perf_counter() - started)
        native_wavelength = 1.0e4 / np.asarray(result["wavenumber"], dtype=float)
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        output_native_probe_flux.append(
            np.asarray(result["thermal"], dtype=float)[order] * 0.1
        )
        full = result["full_output"]
        gas_tau = np.asarray(full["taugas"], dtype=float)
        rayleigh_tau = np.asarray(full["tauray"], dtype=float)
        cloud_tau = np.asarray(full["taucld"], dtype=float)
        maximum_rayleigh_tau = max(
            maximum_rayleigh_tau, float(np.max(np.abs(rayleigh_tau)))
        )
        maximum_cloud_tau = max(maximum_cloud_tau, float(np.max(np.abs(cloud_tau))))
        gas_tau = gas_tau[:, order]
        output_tau.append(gas_tau)
        formal_layers = _formal_contribution(
            wavelength,
            contract["temperature_edges_k"][case_index],
            gas_tau,
            np.asarray(opacity.gauss_wts, dtype=float),
            contract["emission_mu"],
            contract["disk_weights"],
            normalize=False,
        )
        output_flux.append(np.pi * np.sum(formal_layers, axis=0))
        output_contribution.append(_normalize(formal_layers))
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(output_flux),
        "native_framework_probe_flux_w_m2_m": np.asarray(output_native_probe_flux),
        "layer_tau": np.asarray(output_tau, dtype=np.float32),
        "normalized_vertical_diagnostic": np.asarray(
            output_contribution, dtype=np.float32
        ),
        "runtime_s": np.asarray(timings),
        "g_weights": np.asarray(opacity.gauss_wts, dtype=float),
        "maximum_abs_rayleigh_tau": np.array(maximum_rayleigh_tau),
        "maximum_abs_cloud_tau": np.array(maximum_cloud_tau),
    }


def _mass_fractions(
    vmr: np.ndarray, masses: np.ndarray, names: list[str]
) -> tuple[dict[str, float], float]:
    mean = float(np.sum(vmr * masses))
    fractions = vmr * masses / mean
    return dict(zip(names, fractions, strict=True)), mean


def _petitradtrans_native(
    contract: dict[str, np.ndarray], input_data: Path
) -> dict[str, np.ndarray]:
    from petitRADTRANS.radtrans import Radtrans

    line_names = {
        "H2O": "H2O__POKAZATEL",
        "CO": "CO__HITEMP",
        "CO2": "CO2__UCL-4000",
        "CH4": "CH4__YT34to10",
    }
    gas_names = [str(value) for value in contract["gas_name"]]
    species = [str(value) for value in contract["species_name"]]
    output_flux: list[np.ndarray] = []
    output_contribution: list[np.ndarray] = []
    timings: list[float] = []
    wavelength = np.empty(0)
    for case_index, molecule in enumerate(species):
        line = line_names[molecule]
        atmosphere = Radtrans(
            pressures=contract["pressure_centers_bar"],
            wavelength_boundaries=np.array([0.3, 12.1]),
            line_species=[line],
            scattering_in_emission=False,
            emission_angle_grid=np.vstack(
                (contract["emission_mu"], contract["legendre_weights"])
            ),
            path_input_data=str(input_data),
        )
        fractions, mean = _mass_fractions(
            contract["gas_vmr"][case_index], contract["gas_mass_u"], gas_names
        )
        started = perf_counter()
        result = atmosphere.calculate_flux(
            temperatures=contract["temperature_cells_k"][case_index],
            mass_fractions={
                line: np.full(contract["pressure_centers_bar"].size, fractions[molecule]),
                "H2": np.full(contract["pressure_centers_bar"].size, fractions["H2"]),
                "He": np.full(contract["pressure_centers_bar"].size, fractions["He"]),
            },
            mean_molar_masses=np.full(contract["pressure_centers_bar"].size, mean),
            reference_gravity=float(contract["gravity_m_s2"]) * 100.0,
            frequencies_to_wavelengths=True,
            return_contribution=True,
        )
        timings.append(perf_counter() - started)
        native_wavelength = np.asarray(result[0], dtype=float) * 1.0e4
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        output_flux.append(np.asarray(result[1], dtype=float)[order] * 0.1)
        contribution = np.asarray(result[2]["emission_contribution"], dtype=float)
        if contribution.shape[0] != contract["pressure_centers_bar"].size:
            contribution = contribution.T
        output_contribution.append(_normalize(contribution[:, order]))
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(output_flux),
        "normalized_vertical_diagnostic": np.asarray(
            output_contribution, dtype=np.float32
        ),
        "runtime_s": np.asarray(timings),
    }


def _petitradtrans_shared(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from scipy.constants import c
    from petitRADTRANS.radtrans import fcore

    wavelength = contract["shared_wavelength_micron"]
    frequency = c / (wavelength * 1.0e-6)
    flux = []
    contribution = []
    timings = []
    for case_index, tau in enumerate(contract["shared_layer_tau"]):
        cumulative = np.concatenate(
            (np.zeros((1, wavelength.size)), np.cumsum(tau, axis=0)), axis=0
        )
        started = perf_counter()
        flux_nu_cgs, _ = fcore.compute_ck_flux(
            frequency,
            contract["temperature_edges_k"][case_index],
            np.array([1.0]),
            contract["emission_mu"],
            contract["legendre_weights"],
            cumulative.T[None, :, None, :],
            0,
        )
        timings.append(perf_counter() - started)
        flux.append(flux_nu_cgs * 1.0e-3 * c / (wavelength * 1.0e-6) ** 2)
        contribution.append(
            _formal_contribution(
                wavelength,
                contract["temperature_edges_k"][case_index],
                tau,
                np.array([1.0]),
                contract["emission_mu"],
                contract["disk_weights"],
            )
        )
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "normalized_vertical_diagnostic": np.asarray(contribution, dtype=np.float32),
        "runtime_s": np.asarray(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("picaso_ck", "petitradtrans_native", "petitradtrans_shared"),
    )
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--picaso-ck-directory", type=Path)
    parser.add_argument("--prt-input-data", type=Path)
    args = parser.parse_args()
    expected_python = PICASO_PYTHON if args.mode == "picaso_ck" else PRT_PYTHON
    if os.path.realpath(sys.executable) != os.path.realpath(expected_python):
        raise RuntimeError(f"{args.mode} must run with {expected_python}")
    contract = _load(args.contract)
    started = perf_counter()
    if args.mode == "picaso_ck":
        if args.picaso_ck_directory is None:
            parser.error("picaso_ck requires --picaso-ck-directory")
        output = _picaso(
            contract,
            ck_directory=args.picaso_ck_directory,
        )
        package = "picaso"
        limitations = [
            "PICASO native resort-rebin opacity is used with an independently labelled absorbing-formal RT reference because the exact-omega0=0 native thermal probe is pathological; the raw native probe is retained separately."
        ]
    elif args.mode == "petitradtrans_native":
        if args.prt_input_data is None:
            parser.error("petitradtrans_native requires --prt-input-data")
        output = _petitradtrans_native(contract, args.prt_input_data)
        package = "petitRADTRANS"
        limitations = [
            "Stable pRT native layer optical-depth tensors are not exposed by the supported high-level flux interface; spectra and native emission contributions are retained."
        ]
    else:
        output = _petitradtrans_shared(contract)
        package = "petitRADTRANS"
        limitations = []
    metadata = {
        "mode": args.mode,
        "python": os.path.realpath(sys.executable),
        "package": package,
        "version": importlib.metadata.version(package),
        "platform": platform.platform(),
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cia_enabled": False,
        "cloud_enabled": False,
        "known_warnings": (
            ["Optional Vega spectrum absent; Version 2 uses an explicit blackbody star"]
            if package == "picaso"
            else []
        ),
        "limitations": limitations,
    }
    if package == "picaso":
        metadata["picaso_environment"] = _validate_picaso_environment()
        metadata["absolute_line_vmr_restored_after_resort_rebin"] = True
    output["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()
