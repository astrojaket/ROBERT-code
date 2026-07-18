"""Run one isolated Version-2 Stage-7 PICASO or stable-pRT worker."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE_3_WORKER_PATH = HERE / "run_emission_intercomparison_v2_stage_3_external.py"
SPEC = importlib.util.spec_from_file_location(
    "emission_v2_stage_3_worker", STAGE_3_WORKER_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {STAGE_3_WORKER_PATH}")
stage_3_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_3_worker)

PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
GAS_NAMES = ("H2", "He", "H2O", "CO", "CO2", "CH4")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _validate_contract(contract: dict[str, np.ndarray], *, shared: bool) -> None:
    required = {
        "case_id",
        "profile_name",
        "profile_index",
        "case_cloud_index",
        "gas_name",
        "gas_mass_u",
        "gas_vmr",
        "mean_molecular_weight_u",
        "pressure_edges_bar",
        "pressure_centers_bar",
        "temperature_edges_k",
        "temperature_cells_k",
        "gravity_m_s2",
        "emission_mu",
        "legendre_weights",
        "disk_weights",
        "cloud_label",
        "cloud_kind",
        "cloud_input_wavelength_micron",
        "cloud_input_extinction_tau",
        "cloud_single_scattering_albedo",
    }
    if shared:
        required |= {"shared_wavelength_micron", "shared_layer_tau"}
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"Stage-7 contract is missing fields: {', '.join(missing)}")
    count = contract["case_id"].size
    n_cells = contract["pressure_centers_bar"].size
    if n_cells not in {40, 80, 160}:
        raise ValueError("Stage-7 pressure grid must have 40, 80, or 160 cells")
    if set(contract["gas_name"].tolist()) != set(GAS_NAMES):
        raise ValueError("Stage 7 requires exactly H2, He, H2O, CO, CO2, and CH4")
    if contract["gas_vmr"].shape != (count, 6):
        raise ValueError("gas_vmr must have case by six-gas shape")
    if not np.allclose(np.sum(contract["gas_vmr"], axis=-1), 1.0, rtol=0.0, atol=5e-16):
        raise ValueError("every Stage-7 composition must sum to one")
    if not np.all(contract["include_h2_h2_cia"]) or not np.all(
        contract["include_h2_he_cia"]
    ):
        raise ValueError("Stage 7 fixes both H2-H2 and H2-He CIA on")
    if np.any(contract["cloud_single_scattering_albedo"] != 0.0):
        raise ValueError("Stage 7 requires exact omega0=0")
    expected_cloud_shape = (
        contract["cloud_label"].size,
        n_cells,
        contract["cloud_input_wavelength_micron"].size,
    )
    if contract["cloud_input_extinction_tau"].shape != expected_cloud_shape:
        raise ValueError("cloud input extinction has an unexpected shape")


def _cloud_dataframe(pd: Any, contract: dict[str, np.ndarray], cloud_index: int) -> Any:
    pressure = contract["pressure_centers_bar"]
    wavelength = contract["cloud_input_wavelength_micron"]
    extinction = contract["cloud_input_extinction_tau"][cloud_index]
    rows = [
        (pressure[layer], 1.0e4 / wavelength[wave], extinction[layer, wave], 0.0, 0.0)
        for layer in range(pressure.size)
        for wave in range(wavelength.size)
    ]
    return pd.DataFrame(rows, columns=("pressure", "wavenumber", "opd", "w0", "g0"))


def _picaso_native(
    contract: dict[str, np.ndarray], ck_directory: Path
) -> dict[str, np.ndarray]:
    stage_3_worker._validate_picaso_environment()
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi

    opacity = jdi.opannection(
        method="resortrebin",
        ck_db=str(ck_directory),
        preload_gases=list(stage_3_worker.MOLECULAR_SPECIES),
        wave_range=[0.3, 12.0],
        verbose=False,
    )
    stage_3_worker._restore_resort_rebin_absolute_vmr(opacity)
    base_get_opacities = opacity.get_opacities
    gas_names = [str(value) for value in contract["gas_name"]]
    flux: list[np.ndarray] = []
    native_probe: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    timings: list[float] = []
    gas_tau_by_profile: dict[int, np.ndarray] = {}
    cloud_tau_by_cloud: dict[int, np.ndarray] = {}
    wavelength = np.empty(0)
    maximum_rayleigh_tau = 0.0
    maximum_input_omega0 = 0.0
    for case_index, cloud_index_value in enumerate(contract["case_cloud_index"]):
        cloud_index = int(cloud_index_value)
        profile_index = int(contract["profile_index"][case_index])
        observed_pairs: set[str] = set()
        opacity.get_opacities = base_get_opacities
        stage_3_worker._filter_picaso_continuum(
            opacity, ("H2-H2", "H2-He"), observed_pairs
        )
        profile: dict[str, Any] = {
            "pressure": contract["pressure_edges_bar"],
            "temperature": contract["temperature_edges_k"][case_index],
        }
        for gas_index, gas_name in enumerate(gas_names):
            profile[gas_name] = np.full(
                contract["pressure_edges_bar"].size,
                contract["gas_vmr"][case_index, gas_index],
            )
        case = jdi.inputs(calculation="browndwarf")
        case.gravity(gravity=float(contract["gravity_m_s2"]), gravity_unit=u.m / u.s**2)
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
        if str(contract["cloud_kind"][cloud_index]) != "clear":
            cloud_frame = _cloud_dataframe(pd, contract, cloud_index)
            maximum_input_omega0 = max(
                maximum_input_omega0, float(np.max(np.abs(cloud_frame["w0"])))
            )
            case.clouds(df=cloud_frame)
        started = perf_counter()
        result = case.spectrum(opacity, calculation="thermal", full_output=True)
        timings.append(perf_counter() - started)
        if observed_pairs != {"H2-H2", "H2-He"}:
            raise RuntimeError("PICASO did not retain both frozen CIA pairs")
        native_wavelength = 1.0e4 / np.asarray(result["wavenumber"], dtype=float)
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        native_probe.append(np.asarray(result["thermal"], dtype=float)[order] * 0.1)
        full = result["full_output"]
        gas_tau = np.asarray(full["taugas"], dtype=float)[:, order]
        cloud_tau = np.asarray(full["taucld"], dtype=float)
        if cloud_tau.ndim == 3:
            cloud_tau = cloud_tau[:, order]
        else:
            cloud_tau = cloud_tau[:, order]
        rayleigh_tau = np.asarray(full["tauray"], dtype=float)
        maximum_rayleigh_tau = max(
            maximum_rayleigh_tau, float(np.max(np.abs(rayleigh_tau)))
        )
        gas_tau_by_profile.setdefault(profile_index, gas_tau.astype(np.float32))
        cloud_tau_by_cloud.setdefault(cloud_index, cloud_tau.astype(np.float32))
        total_tau = gas_tau + cloud_tau
        layers = stage_3_worker._formal_contribution(
            wavelength,
            contract["temperature_edges_k"][case_index],
            total_tau,
            np.asarray(opacity.gauss_wts, dtype=float),
            contract["emission_mu"],
            contract["disk_weights"],
            normalize=False,
        )
        flux.append(np.pi * np.sum(layers, axis=0))
        vertical.append(stage_3_worker._normalize(layers).astype(np.float32))

    profile_indices = sorted(gas_tau_by_profile)
    cloud_indices = sorted(cloud_tau_by_cloud)
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "native_framework_probe_flux_w_m2_m": np.asarray(native_probe),
        "normalized_vertical_diagnostic": np.asarray(vertical),
        "runtime_s": np.asarray(timings),
        "g_weights": np.asarray(opacity.gauss_wts, dtype=float),
        "native_gas_tau_profile_index": np.asarray(profile_indices),
        "native_gas_tau_by_profile": np.asarray(
            [gas_tau_by_profile[index] for index in profile_indices]
        ),
        "native_cloud_tau_cloud_index": np.asarray(cloud_indices),
        "native_cloud_tau_by_cloud": np.asarray(
            [cloud_tau_by_cloud[index] for index in cloud_indices]
        ),
        "maximum_abs_rayleigh_tau": np.array(maximum_rayleigh_tau),
        "maximum_abs_input_cloud_omega0": np.array(maximum_input_omega0),
    }


def _interpolate_positive_or_zero(
    source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray
) -> np.ndarray:
    output = np.empty(target_x.size)
    for index, value in enumerate(target_x):
        if value <= source_x[0]:
            output[index] = source_y[0]
        elif value >= source_x[-1]:
            output[index] = source_y[-1]
        else:
            right = int(np.searchsorted(source_x, value, side="right"))
            left = right - 1
            fraction = np.log(value / source_x[left]) / np.log(source_x[right] / source_x[left])
            if source_y[left] > 0.0 and source_y[right] > 0.0:
                output[index] = np.exp(
                    (1.0 - fraction) * np.log(source_y[left])
                    + fraction * np.log(source_y[right])
                )
            else:
                output[index] = (1.0 - fraction) * source_y[left] + fraction * source_y[right]
    return output


def _prt_cloud_opacity_function(
    contract: dict[str, np.ndarray], cloud_index: int
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    input_wavelength = contract["cloud_input_wavelength_micron"]
    input_pressure = contract["pressure_centers_bar"]
    layer_pressure_cgs = np.diff(contract["pressure_edges_bar"]) * 1.0e6
    gravity_cgs = float(contract["gravity_m_s2"]) * 100.0
    layer_tau = contract["cloud_input_extinction_tau"][cloud_index]
    opacity = layer_tau * gravity_cgs / layer_pressure_cgs[:, None]

    def callback(wavelength: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        requested_wavelength = np.asarray(wavelength, dtype=float)
        requested_pressure = np.asarray(pressure, dtype=float)
        spectral = np.stack(
            [
                _interpolate_positive_or_zero(input_wavelength, row, requested_wavelength)
                for row in opacity
            ]
        )
        if np.array_equal(requested_pressure, input_pressure):
            return spectral.T
        output = np.empty((requested_wavelength.size, requested_pressure.size))
        for wave_index in range(requested_wavelength.size):
            output[wave_index] = _interpolate_positive_or_zero(
                input_pressure, spectral[:, wave_index], requested_pressure
            )
        return output

    return callback


def _petitradtrans_native(
    contract: dict[str, np.ndarray], input_data: Path
) -> dict[str, np.ndarray]:
    from petitRADTRANS.radtrans import Radtrans

    atmosphere = Radtrans(
        pressures=contract["pressure_centers_bar"],
        wavelength_boundaries=np.array([0.3, 12.1]),
        line_species=list(stage_3_worker.PRT_LINE_SPECIES.values()),
        gas_continuum_contributors=list(stage_3_worker.PRT_CIA_SPECIES.values()),
        rayleigh_species=[],
        cloud_species=[],
        scattering_in_emission=False,
        anisotropic_cloud_scattering=False,
        emission_angle_grid=np.vstack(
            (contract["emission_mu"], contract["legendre_weights"])
        ),
        path_input_data=str(input_data),
    )
    gas_names = [str(value) for value in contract["gas_name"]]
    masses = contract["gas_mass_u"]
    flux: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    timings: list[float] = []
    wavelength = np.empty(0)
    for case_index, cloud_index_value in enumerate(contract["case_cloud_index"]):
        vmr = contract["gas_vmr"][case_index]
        mean_molecular_weight = float(contract["mean_molecular_weight_u"][case_index])
        fractions = vmr * masses / mean_molecular_weight
        mass_fractions = {
            line: np.full(
                contract["pressure_centers_bar"].size,
                fractions[gas_names.index(molecule)],
            )
            for molecule, line in stage_3_worker.PRT_LINE_SPECIES.items()
        }
        mass_fractions["H2"] = np.full(
            contract["pressure_centers_bar"].size, fractions[gas_names.index("H2")]
        )
        mass_fractions["He"] = np.full(
            contract["pressure_centers_bar"].size, fractions[gas_names.index("He")]
        )
        opacity_function = _prt_cloud_opacity_function(
            contract, int(cloud_index_value)
        )
        started = perf_counter()
        result = atmosphere.calculate_flux(
            temperatures=contract["temperature_cells_k"][case_index],
            mass_fractions=mass_fractions,
            mean_molar_masses=np.full(
                contract["pressure_centers_bar"].size, mean_molecular_weight
            ),
            reference_gravity=float(contract["gravity_m_s2"]) * 100.0,
            additional_absorption_opacities_function=opacity_function,
            additional_scattering_opacities_function=None,
            frequencies_to_wavelengths=True,
            return_contribution=True,
        )
        timings.append(perf_counter() - started)
        native_wavelength = np.asarray(result[0], dtype=float) * 1.0e4
        order = np.argsort(native_wavelength)
        wavelength = native_wavelength[order]
        flux.append(np.asarray(result[1], dtype=float)[order] * 0.1)
        contribution = np.asarray(result[2]["emission_contribution"], dtype=float)
        if contribution.shape[0] != contract["pressure_centers_bar"].size:
            contribution = contribution.T
        vertical.append(stage_3_worker._normalize(contribution[:, order]).astype(np.float32))
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "normalized_vertical_diagnostic": np.asarray(vertical),
        "runtime_s": np.asarray(timings),
        "native_layer_tau_supported": np.array(False),
    }


def _expected_python(mode: str) -> Path:
    return PICASO_PYTHON if mode == "picaso_native" else PRT_PYTHON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("picaso_native", "petitradtrans_native", "petitradtrans_shared")
    )
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--picaso-ck-directory", type=Path)
    parser.add_argument("--prt-input-data", type=Path)
    args = parser.parse_args()
    expected = _expected_python(args.mode)
    if os.path.realpath(sys.executable) != os.path.realpath(expected):
        raise RuntimeError(f"{args.mode} must run with {expected}")
    contract = _load(args.contract)
    _validate_contract(contract, shared=args.mode == "petitradtrans_shared")
    started = perf_counter()
    if args.mode == "picaso_native":
        if args.picaso_ck_directory is None:
            parser.error("picaso_native requires --picaso-ck-directory")
        output = _picaso_native(contract, args.picaso_ck_directory)
        package = "picaso"
        representation = "native_cloud_table_plus_resort_rebin_gas_absorbing_formal"
        limitations = [
            "Exact-omega0=0 native thermal output is capability evidence separate from the absorbing-formal scientific spectrum.",
            "Absorbing-formal vertical diagnostics are not native SH contribution functions.",
        ]
    elif args.mode == "petitradtrans_native":
        if args.prt_input_data is None:
            parser.error("petitradtrans_native requires --prt-input-data")
        output = _petitradtrans_native(contract, args.prt_input_data)
        package = "petitRADTRANS"
        representation = "native_additional_absorption_callback"
        limitations = [
            "The supported stable-pRT high-level flux interface exposes no stable native layer optical-depth tensor; none is fabricated."
        ]
    else:
        output = stage_3_worker._petitradtrans_shared(contract)
        package = "petitRADTRANS"
        representation = "track_a_identical_gas_plus_cloud_mean_tau"
        limitations = []
    metadata: dict[str, Any] = {
        "mode": args.mode,
        "representation": representation,
        "python": os.path.realpath(sys.executable),
        "expected_python": str(expected),
        "interpreter_matches_expected": True,
        "package": package,
        "version": importlib.metadata.version(package),
        "platform": platform.platform(),
        "wall_time_s": perf_counter() - started,
        "peak_rss_bytes": stage_3_worker._peak_rss_bytes(),
        "molecular_species_always_enabled": list(stage_3_worker.MOLECULAR_SPECIES),
        "fixed_cia_pairs": ["H2-H2", "H2-He"],
        "cloud_single_scattering_albedo": 0.0,
        "scattering_enabled": False,
        "rayleigh_enabled": False,
        "delta_m": False,
        "limitations": limitations,
        "known_warnings": (
            [
                "Optional Vega spectrum absent; Version 2 uses an explicit blackbody star.",
                "Exact-zero cloud/Rayleigh calculations may emit invalid-divide warnings; exact zeros and warnings are retained.",
            ]
            if package == "picaso"
            else []
        ),
    }
    if package == "picaso":
        metadata["picaso_environment"] = stage_3_worker._validate_picaso_environment()
        metadata["absolute_line_vmr_restored_after_resort_rebin"] = True
    output["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()
