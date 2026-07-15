"""Independent official-PICASO reference for the shared deck/haze contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    from examples.run_picaso_official_molecular_cloud_parity import (
        _build_case,
        _planck_radiance,
        _sha256,
    )
except ModuleNotFoundError:
    from run_picaso_official_molecular_cloud_parity import (
        _build_case,
        _planck_radiance,
        _sha256,
    )


CASES = ("clear", "deck", "haze", "deck_haze")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opacity-db", type=Path, required=True)
    parser.add_argument("--resample", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.contract, args.output, args.opacity_db, args.resample)


def evaluate(contract_path: Path, output_path: Path, opacity_db: Path, resample: int) -> None:
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi

    with np.load(contract_path, allow_pickle=False) as archive:
        contract = {name: np.array(archive[name], copy=True) for name in archive.files}

    started = perf_counter()
    opacity = jdi.opannection(
        filename_db=str(opacity_db),
        wave_range=[1.0, 12.0],
        resample=resample,
        verbose=False,
    )
    connection_seconds = perf_counter() - started
    outputs = {}
    timing = {}
    wavelength = None
    for case_name in CASES:
        case = _build_case(jdi, u, pd, contract, {}, cloudy=False)
        if case_name != "clear":
            case.clouds(df=_cloud_dataframe(pd, contract, case_name))
        started = perf_counter()
        thermal = case.spectrum(
            opacity,
            calculation="thermal",
            full_output=True,
            as_dict=False,
        )
        thermal_seconds = perf_counter() - started
        case.inputs["star"]["radius"] = float(contract["star_radius_m"]) * 100.0
        case.inputs["star"]["radius_unit"] = "cm"
        started = perf_counter()
        transmission = case.spectrum(opacity, calculation="transmission", full_output=False)
        transmission_seconds = perf_counter() - started
        case_wavelength = 1.0e4 / np.asarray(thermal["wavenumber"], dtype=float)
        order = np.argsort(case_wavelength)
        wavelength = case_wavelength[order]
        thermal_flux = np.asarray(thermal["thermal"], dtype=float)[order]
        stellar_flux = (
            np.pi
            * _planck_radiance(wavelength, float(contract["star_temperature_k"]))
            * 10.0
        )
        area_ratio = (
            float(contract["planet_radius_m"]) / float(contract["star_radius_m"])
        ) ** 2
        outputs[f"{case_name}_eclipse_depth"] = thermal_flux / stellar_flux * area_ratio
        outputs[f"{case_name}_transit_depth"] = np.asarray(
            transmission["transit_depth"], dtype=float
        )[order]
        atmosphere = thermal["full_output"]
        outputs[f"{case_name}_gas_tau"] = np.asarray(
            atmosphere.taugas[:, :, 0], dtype=float
        )[:, order]
        outputs[f"{case_name}_cloud_tau"] = np.asarray(
            atmosphere.taucld[:, :, 0], dtype=float
        )[:, order]
        timing[case_name] = {
            "thermal": thermal_seconds,
            "transmission": transmission_seconds,
        }

    metadata = {
        "picaso_version": importlib.metadata.version("picaso"),
        "opacity_database_sha256": _sha256(opacity_db),
        "opacity_resample_stride": resample,
        "opacity_connection_seconds": connection_seconds,
        "timing_seconds": timing,
        "cloud_contract": "exact_layer_optical_depth_from_shared_physical_parameters",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        wavelength_micron=wavelength,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **outputs,
    )


def _cloud_dataframe(pd, contract, case_name: str):
    pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    pressure = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    wavelength = np.asarray(contract["wavelength_micron"], dtype=float)
    deck_tau = np.zeros((pressure.size, wavelength.size))
    if case_name in {"deck", "deck_haze"}:
        layer_low = np.minimum(pressure_edges[:-1], pressure_edges[1:])
        layer_high = np.maximum(pressure_edges[:-1], pressure_edges[1:])
        overlap_low = np.maximum(layer_low, float(contract["deck_top_pressure_bar"]))
        overlap = np.where(
            layer_high > overlap_low,
            np.log(layer_high / overlap_low),
            0.0,
        )
        deck_tau[:] = (
            float(contract["deck_optical_depth"]) * overlap / np.sum(overlap)
        )[:, None]
    haze_tau = np.zeros_like(deck_tau)
    if case_name in {"haze", "deck_haze"}:
        layer_mass = (
            np.diff(pressure_edges)
            * 1.0e5
            / float(contract["gravity_m_s2"])
        )
        mass_extinction_m2_kg = (
            float(contract["haze_mass_extinction_cm2_g"])
            * 0.1
            * (wavelength / float(contract["haze_reference_wavelength_micron"]))
            ** float(contract["haze_slope"])
        )
        haze_tau = layer_mass[:, None] * mass_extinction_m2_kg[None, :]
    total_tau = deck_tau + haze_tau
    omega = np.divide(
        haze_tau,
        total_tau,
        out=np.zeros_like(total_tau),
        where=total_tau > 0.0,
    )
    rows = [
        (pressure[layer], 1.0e4 / wavelength[wave], total_tau[layer, wave], omega[layer, wave], 0.0)
        for layer in range(pressure.size)
        for wave in range(wavelength.size)
    ]
    return pd.DataFrame(rows, columns=("pressure", "wavenumber", "opd", "w0", "g0"))


if __name__ == "__main__":
    main()
