"""Run one isolated Version-2 Stage-3 PICASO or stable-pRT worker."""

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
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")
PRT_LINE_SPECIES = {
    "H2O": "H2O__POKAZATEL",
    "CO": "CO__HITEMP",
    "CO2": "CO2__UCL-4000",
    "CH4": "CH4__YT34to10",
}
PRT_CIA_SPECIES = {
    "H2-H2": "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
    "H2-He": "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
}


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
    """Return the labelled absorbing-formal vertical diagnostic."""

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


def _validate_contract(contract: dict[str, np.ndarray]) -> None:
    required = {
        "case_id",
        "profile_name",
        "gas_name",
        "gas_mass_u",
        "gas_vmr",
        "mean_molecular_weight_u",
        "molecular_species_name",
        "molecular_species_active",
        "pressure_edges_bar",
        "pressure_centers_bar",
        "temperature_edges_k",
        "temperature_cells_k",
        "gravity_m_s2",
        "emission_mu",
        "legendre_weights",
        "disk_weights",
        "include_h2_h2_cia",
        "include_h2_he_cia",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"Stage-3 contract is missing fields: {', '.join(missing)}")
    case_count = contract["case_id"].size
    gas_names = [str(value) for value in contract["gas_name"]]
    if set(gas_names) != {"H2", "He", *MOLECULAR_SPECIES}:
        raise ValueError("Stage-3 contract must contain exactly H2, He, and four absorbers")
    if contract["gas_vmr"].shape != (case_count, len(gas_names)):
        raise ValueError("gas_vmr must have shape case by six gases")
    if not np.all(contract["gas_vmr"] > 0.0):
        raise ValueError("all six frozen Stage-3 gases must be present in every case")
    if not np.all(contract["gas_vmr"] == contract["gas_vmr"][[0]]):
        raise ValueError("Stage-3 cases must use one identical frozen composition")
    if not np.all(
        np.isclose(
            np.sum(contract["gas_vmr"], axis=1), 1.0, rtol=0.0, atol=5.0e-16
        )
    ):
        raise ValueError("each frozen Stage-3 composition must sum exactly to one")
    allowed_profiles = {"isothermal", "pg14_non_inverted"}
    if not set(contract["profile_name"].tolist()) <= allowed_profiles:
        raise ValueError("Stage 3 supports only isothermal and pg14_non_inverted")
    n_cells = contract["pressure_centers_bar"].size
    if n_cells not in {40, 80, 160}:
        raise ValueError("Stage-3 pressure grid must have 40, 80, or 160 cells")
    if contract["pressure_edges_bar"].shape != (n_cells + 1,):
        raise ValueError("pressure edges must have one more entry than pressure cells")
    if contract["temperature_edges_k"].shape != (case_count, n_cells + 1):
        raise ValueError("temperature_edges_k has an unexpected shape")
    if contract["temperature_cells_k"].shape != (case_count, n_cells):
        raise ValueError("temperature_cells_k has an unexpected shape")
    for name in ("include_h2_h2_cia", "include_h2_he_cia"):
        if contract[name].shape != (case_count,):
            raise ValueError(f"{name} must have one value per case")
    if tuple(contract["molecular_species_name"].tolist()) != MOLECULAR_SPECIES:
        raise ValueError("Stage 3 must use exactly the four frozen molecular absorbers")
    if contract["molecular_species_active"].shape != (
        case_count,
        len(MOLECULAR_SPECIES),
    ) or not np.all(contract["molecular_species_active"]):
        raise ValueError("all four molecular absorbers must be active in every case")
    if contract["mean_molecular_weight_u"].shape != (case_count,) or not np.all(
        contract["mean_molecular_weight_u"]
        == contract["mean_molecular_weight_u"][[0]]
    ):
        raise ValueError("one frozen mean molecular weight must be used in every case")


def _requested_cia_pairs(
    contract: dict[str, np.ndarray], case_index: int
) -> tuple[str, ...]:
    pairs = []
    if bool(contract["include_h2_h2_cia"][case_index]):
        pairs.append("H2-H2")
    if bool(contract["include_h2_he_cia"][case_index]):
        pairs.append("H2-He")
    return tuple(pairs)


def _picaso_pair_name(pair: Any) -> str | None:
    values = tuple(str(value) for value in pair)
    if values == ("H2", "H2"):
        return "H2-H2"
    if len(values) == 2 and set(values) == {"H2", "He"}:
        return "H2-He"
    return None


def _filter_picaso_continuum(
    opacity: Any, requested_pairs: tuple[str, ...], observed_pairs: set[str]
) -> None:
    """Keep only the requested H2-H2/H2-He CIA pairs and disable Rayleigh."""

    original = opacity.get_opacities
    requested = set(requested_pairs)

    def selected_continuum(atmosphere: Any, exclude_mol: Any = 1) -> Any:
        retained = []
        for pair in atmosphere.continuum_molecules:
            name = _picaso_pair_name(pair)
            if name in requested:
                retained.append(pair)
                observed_pairs.add(name)
        atmosphere.continuum_molecules = retained
        atmosphere.rayleigh_molecules = []
        return original(atmosphere, exclude_mol=exclude_mol)

    opacity.get_opacities = selected_continuum


def _validate_picaso_environment() -> dict[str, str]:
    reference = os.environ.get("picaso_refdata")
    numba_cache = os.environ.get("NUMBA_CACHE_DIR")
    matplotlib_cache = os.environ.get("MPLCONFIGDIR")
    if reference is None or Path(reference).resolve() != PICASO_REFERENCE.resolve():
        raise RuntimeError(f"picaso_refdata must be exactly {PICASO_REFERENCE}")
    if not numba_cache or not matplotlib_cache:
        raise RuntimeError("PICASO requires explicit NUMBA_CACHE_DIR and MPLCONFIGDIR")
    for raw_path, label in (
        (numba_cache, "NUMBA_CACHE_DIR"),
        (matplotlib_cache, "MPLCONFIGDIR"),
    ):
        path = Path(raw_path)
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise RuntimeError(f"{label} must name an existing writable directory")
    return {
        "picaso_refdata": reference,
        "NUMBA_CACHE_DIR": numba_cache,
        "MPLCONFIGDIR": matplotlib_cache,
    }


def _picaso(
    contract: dict[str, np.ndarray],
    *,
    representation: str,
    ck_directory: Path,
    sampling_database: Path | None,
    sampling_resample: int,
) -> dict[str, np.ndarray]:
    _validate_picaso_environment()
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi

    if representation == "correlated_k_resort_rebin":
        opacity = jdi.opannection(
            method="resortrebin",
            ck_db=str(ck_directory),
            preload_gases=list(MOLECULAR_SPECIES),
            wave_range=[0.79, 12.1],
            verbose=False,
        )
    else:
        if sampling_database is None:
            raise ValueError("opacity sampling requires a database")
        opacity = jdi.opannection(
            filename_db=str(sampling_database),
            wave_range=[0.79, 12.1],
            resample=sampling_resample,
            verbose=False,
        )
    output_flux: list[np.ndarray] = []
    output_native_probe_flux: list[np.ndarray] = []
    output_tau: list[np.ndarray] = []
    output_contribution: list[np.ndarray] = []
    output_pairs_json: list[str] = []
    timings: list[float] = []
    maximum_rayleigh_tau = 0.0
    maximum_cloud_tau = 0.0
    wavelength = np.empty(0)
    gas_names = [str(value) for value in contract["gas_name"]]
    base_get_opacities = opacity.get_opacities
    for case_index in range(contract["case_id"].size):
        requested_pairs = _requested_cia_pairs(contract, case_index)
        observed_pairs: set[str] = set()
        opacity.get_opacities = base_get_opacities
        _filter_picaso_continuum(opacity, requested_pairs, observed_pairs)
        profile: dict[str, Any] = {
            "pressure": contract["pressure_edges_bar"],
            "temperature": contract["temperature_edges_k"][case_index],
        }
        for gas_index, gas_name in enumerate(gas_names):
            profile[gas_name] = np.full(
                contract["pressure_edges_bar"].size,
                float(contract["gas_vmr"][case_index, gas_index]),
            )
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
        if observed_pairs != set(requested_pairs):
            raise RuntimeError(
                "PICASO continuum selection mismatch: "
                f"requested {sorted(requested_pairs)}, observed {sorted(observed_pairs)}"
            )
        output_pairs_json.append(json.dumps(sorted(observed_pairs)))
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
        "native_continuum_pairs_json": np.asarray(output_pairs_json),
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

    gas_names = [str(value) for value in contract["gas_name"]]
    output_flux: list[np.ndarray] = []
    output_contribution: list[np.ndarray] = []
    output_pairs_json: list[str] = []
    timings: list[float] = []
    wavelength = np.empty(0)
    atmospheres: dict[tuple[bool, bool], Any] = {}
    for case_index in range(contract["case_id"].size):
        h2_h2 = bool(contract["include_h2_h2_cia"][case_index])
        h2_he = bool(contract["include_h2_he_cia"][case_index])
        key = (h2_h2, h2_he)
        requested_pairs = _requested_cia_pairs(contract, case_index)
        if key not in atmospheres:
            atmospheres[key] = Radtrans(
                pressures=contract["pressure_centers_bar"],
                wavelength_boundaries=np.array([0.79, 12.1]),
                line_species=list(PRT_LINE_SPECIES.values()),
                gas_continuum_contributors=[
                    PRT_CIA_SPECIES[pair] for pair in requested_pairs
                ],
                scattering_in_emission=False,
                emission_angle_grid=np.vstack(
                    (contract["emission_mu"], contract["legendre_weights"])
                ),
                path_input_data=str(input_data),
            )
        atmosphere = atmospheres[key]
        fractions, _computed_mean = _mass_fractions(
            contract["gas_vmr"][case_index], contract["gas_mass_u"], gas_names
        )
        declared_mean = float(contract["mean_molecular_weight_u"][case_index])
        mass_fractions = {
            line: np.full(
                contract["pressure_centers_bar"].size, fractions[molecule]
            )
            for molecule, line in PRT_LINE_SPECIES.items()
        }
        mass_fractions["H2"] = np.full(
            contract["pressure_centers_bar"].size, fractions["H2"]
        )
        mass_fractions["He"] = np.full(
            contract["pressure_centers_bar"].size, fractions["He"]
        )
        started = perf_counter()
        result = atmosphere.calculate_flux(
            temperatures=contract["temperature_cells_k"][case_index],
            mass_fractions=mass_fractions,
            mean_molar_masses=np.full(
                contract["pressure_centers_bar"].size, declared_mean
            ),
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
        output_pairs_json.append(json.dumps(list(requested_pairs)))
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(output_flux),
        "normalized_vertical_diagnostic": np.asarray(
            output_contribution, dtype=np.float32
        ),
        "runtime_s": np.asarray(timings),
        "native_continuum_pairs_json": np.asarray(output_pairs_json),
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


def _expected_python(mode: str) -> Path:
    return PICASO_PYTHON if mode.startswith("picaso") else PRT_PYTHON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "picaso_ck",
            "picaso_sampling",
            "petitradtrans_native",
            "petitradtrans_shared",
        ),
    )
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--picaso-ck-directory", type=Path)
    parser.add_argument("--picaso-sampling-database", type=Path)
    parser.add_argument("--picaso-sampling-resample", type=int, default=50)
    parser.add_argument("--prt-input-data", type=Path)
    args = parser.parse_args()
    expected_python = _expected_python(args.mode)
    if os.path.realpath(sys.executable) != os.path.realpath(expected_python):
        raise RuntimeError(f"{args.mode} must run with {expected_python}")
    contract = _load(args.contract)
    _validate_contract(contract)
    started = perf_counter()
    if args.mode == "picaso_ck":
        if args.picaso_ck_directory is None:
            parser.error("picaso_ck requires --picaso-ck-directory")
        output = _picaso(
            contract,
            representation="correlated_k_resort_rebin",
            ck_directory=args.picaso_ck_directory,
            sampling_database=None,
            sampling_resample=args.picaso_sampling_resample,
        )
        package = "picaso"
        representation = "primary_correlated_k_resort_rebin"
        limitations = [
            "PICASO native taugas combines the four molecular absorbers and requested CIA; a separate native component tensor is not exposed by this supported high-level path.",
            "PICASO exact-omega0=0 native thermal probes are retained as pathological capability evidence; the separately labelled absorbing-formal flux is the scientific product.",
            "PICASO vertical arrays are absorbing-formal diagnostics applied to native taugas, not native SH contribution definitions.",
        ]
    elif args.mode == "picaso_sampling":
        if args.picaso_sampling_database is None:
            parser.error("picaso_sampling requires --picaso-sampling-database")
        output = _picaso(
            contract,
            representation="opacity_sampling",
            ck_directory=Path("."),
            sampling_database=args.picaso_sampling_database,
            sampling_resample=args.picaso_sampling_resample,
        )
        package = "picaso"
        representation = "secondary_opacity_sampling_unsmoothed"
        limitations = [
            "PICASO native taugas combines the four molecular absorbers and requested CIA; a separate native component tensor is not exposed by this supported high-level path.",
            "PICASO exact-omega0=0 native thermal probes are retained as pathological capability evidence; the separately labelled absorbing-formal flux is the scientific product.",
            "PICASO opacity sampling is a secondary unsmoothed representation diagnostic and is not interchangeable with correlated-k.",
        ]
    elif args.mode == "petitradtrans_native":
        if args.prt_input_data is None:
            parser.error("petitradtrans_native requires --prt-input-data")
        output = _petitradtrans_native(contract, args.prt_input_data)
        package = "petitRADTRANS"
        representation = "native_correlated_k_with_selected_cia"
        limitations = [
            "Stable pRT does not expose native layer optical-depth or separate CIA tensors through the supported high-level flux interface; spectra and native emission contributions are retained."
        ]
    else:
        output = _petitradtrans_shared(contract)
        package = "petitRADTRANS"
        representation = "track_a_identical_mean_tau"
        limitations = []
    requested_pairs_by_case = {
        str(case_id): list(_requested_cia_pairs(contract, index))
        for index, case_id in enumerate(contract["case_id"])
    }
    metadata: dict[str, Any] = {
        "mode": args.mode,
        "representation": representation,
        "python": os.path.realpath(sys.executable),
        "expected_python": str(expected_python),
        "interpreter_matches_expected": True,
        "package": package,
        "version": importlib.metadata.version(package),
        "platform": platform.platform(),
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "molecular_species_always_enabled": list(MOLECULAR_SPECIES),
        "requested_cia_pairs_by_case": requested_pairs_by_case,
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "cloud_enabled": False,
        "known_warnings": (
            [
                "Optional Vega spectrum absent; Version 2 uses an explicit blackbody star.",
                "Exact-zero cloud/Rayleigh arrays can emit harmless invalid-divide warnings; retained arrays verify both remain zero.",
            ]
            if package == "picaso"
            else []
        ),
        "limitations": limitations,
    }
    if package == "picaso":
        metadata["picaso_environment"] = _validate_picaso_environment()
    output["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()
