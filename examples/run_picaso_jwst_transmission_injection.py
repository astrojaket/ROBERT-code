"""Generate one independent PICASO deck/haze transmission spectrum.

This runner intentionally imports no ROBERT package code. It executes inside
the isolated PICASO environment and consumes only a numerical physical
contract written by the parent benchmark.
"""

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
        _sha256,
    )
    from examples.run_picaso_shared_deck_haze import _cloud_dataframe
except ModuleNotFoundError:
    from run_picaso_official_molecular_cloud_parity import _build_case, _sha256
    from run_picaso_shared_deck_haze import _cloud_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opacity-db", type=Path, required=True)
    parser.add_argument("--resample", type=int, default=5)
    parser.add_argument("--include-opacity-components", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.contract,
        args.output,
        args.opacity_db,
        args.resample,
        include_opacity_components=args.include_opacity_components,
    )


def evaluate(
    contract_path: Path,
    output_path: Path,
    opacity_db: Path,
    resample: int,
    *,
    include_opacity_components: bool = False,
) -> None:
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi
    from picaso.optics import compute_opacity

    with np.load(contract_path, allow_pickle=False) as archive:
        contract = {name: np.array(archive[name], copy=True) for name in archive.files}

    started = perf_counter()
    opacity = jdi.opannection(
        filename_db=str(opacity_db),
        wave_range=[
            float(contract["observation_bin_edges_micron"][0]),
            float(contract["observation_bin_edges_micron"][-1]),
        ],
        resample=resample,
        verbose=False,
    )
    opacity_seconds = perf_counter() - started

    case = _build_case(jdi, u, pd, contract, {}, cloudy=False)
    case.clouds(df=_cloud_dataframe(pd, contract, "deck_haze"))
    started = perf_counter()
    thermal = case.spectrum(
        opacity,
        calculation="thermal",
        full_output=include_opacity_components,
        as_dict=False,
    )
    thermal_seconds = perf_counter() - started
    case.inputs["star"]["radius"] = float(contract["star_radius_m"]) * 100.0
    case.inputs["star"]["radius_unit"] = "cm"
    started = perf_counter()
    transmission = case.spectrum(
        opacity,
        calculation="transmission",
        full_output=False,
    )
    transmission_seconds = perf_counter() - started

    wavelength = 1.0e4 / np.asarray(thermal["wavenumber"], dtype=float)
    order = np.argsort(wavelength)
    metadata = {
        "generator": "PICASO",
        "picaso_version": importlib.metadata.version("picaso"),
        "opacity_database": str(opacity_db.resolve()),
        "opacity_database_sha256": _sha256(opacity_db),
        "opacity_resample_stride": resample,
        "native_wavelength_count": int(wavelength.size),
        "transmission_solver": "PICASO spherical extinction",
        "cloud_contract": "shared finite deck plus power-law haze",
        "opacity_components_exported": include_opacity_components,
        "timing_seconds": {
            "opacity_connection": opacity_seconds,
            "thermal_grid_query": thermal_seconds,
            "transmission": transmission_seconds,
        },
    }
    components = {}
    if include_opacity_components:
        atmosphere = thermal["full_output"]
        contributions = compute_opacity(
            atmosphere,
            opacity,
            ngauss=1,
            stream=4,
            delta_eddington=False,
            raman=2,
            return_mode=True,
        )
        species = ("H2O", "CO", "CO2", "CH4")
        continuum_names = tuple(
            name for name in ("H2H2", "H2He") if name in contributions
        )
        components["molecular_tau_by_species"] = np.stack(
            [np.asarray(contributions[name], dtype=float)[:, order] for name in species],
            axis=0,
        )
        components["continuum_tau"] = np.sum(
            np.stack(
                [
                    np.asarray(contributions[name], dtype=float)[:, order]
                    for name in continuum_names
                ],
                axis=0,
            ),
            axis=0,
        )
        metadata["molecular_species"] = list(species)
        metadata["continuum_species"] = list(continuum_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        wavelength_micron=wavelength[order],
        transit_depth=np.asarray(transmission["transit_depth"], dtype=float)[order],
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **components,
    )


if __name__ == "__main__":
    main()
