"""Independent stable-pRT reference for the shared finite deck/haze contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMPI_MCA_btl", "self")

import numpy as np
import petitRADTRANS
from petitRADTRANS.radtrans import Radtrans


LINE_SPECIES = {
    "H2O": "H2O__POKAZATEL",
    "CO": "CO__HITEMP",
    "CO2": "CO2__UCL-4000",
    "CH4": "CH4__YT34to10",
}
CIA_SPECIES = [
    "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
    "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
]
MOLAR_MASS = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.01528,
    "CO": 28.0101,
    "CO2": 44.0095,
    "CH4": 16.0425,
}
CASES = ("clear", "deck", "haze", "deck_haze")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-data", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.contract, args.output, args.input_data)


def evaluate(contract_path: Path, output_path: Path, input_data: Path) -> None:
    with np.load(contract_path, allow_pickle=False) as archive:
        contract = {name: np.array(archive[name], copy=True) for name in archive.files}

    pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    pressure = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    temperature = 0.5 * (
        np.asarray(contract["temperature_level_k"][:-1], dtype=float)
        + np.asarray(contract["temperature_level_k"][1:], dtype=float)
    )
    volume = {
        name: np.asarray(contract["gas_vmr"][:, index], dtype=float)
        for index, name in enumerate(LINE_SPECIES)
    }
    volume["H2"] = np.full(pressure.size, float(contract["h2_vmr"]))
    volume["He"] = np.asarray(contract["he_vmr"], dtype=float)
    mean_molar_mass = sum(volume[name] * MOLAR_MASS[name] for name in volume)
    mass_fraction = {
        LINE_SPECIES.get(name, name): volume[name] * MOLAR_MASS[name] / mean_molar_mass
        for name in volume
    }

    started = perf_counter()
    atmosphere = Radtrans(
        pressures=pressure,
        wavelength_boundaries=np.array([1.0, 12.0]),
        line_species=list(LINE_SPECIES.values()),
        gas_continuum_contributors=CIA_SPECIES,
        rayleigh_species=["H2", "He"],
        scattering_in_emission=True,
        path_input_data=str(input_data),
    )
    construction_seconds = perf_counter() - started

    gravity = float(contract["gravity_m_s2"])
    deck_active = pressure >= float(contract["deck_top_pressure_bar"])
    if np.count_nonzero(deck_active) < 2:
        raise ValueError("the pRT deck contract requires at least two active nodes")
    # pRT integrates opacity between pressure nodes (its top node has zero
    # overlying column). A constant opacity below the requested top therefore
    # gives the requested integrated tau without importing ROBERT layer edges.
    active_pressure = pressure[deck_active]
    active_column_g_cm2 = (
        (active_pressure[-1] - active_pressure[0]) * 1.0e6 / (gravity * 100.0)
    )
    deck_kappa_cm2_g = np.zeros(pressure.size)
    deck_kappa_cm2_g[deck_active] = (
        float(contract["deck_optical_depth"]) / active_column_g_cm2
    )

    def callbacks(case: str):
        include_deck = case in {"deck", "deck_haze"}
        include_haze = case in {"haze", "deck_haze"}

        def haze(wavelength_micron, pressure_bar):
            del pressure_bar
            opacity = float(contract["haze_mass_extinction_cm2_g"]) * (
                np.asarray(wavelength_micron, dtype=float)
                / float(contract["haze_reference_wavelength_micron"])
            ) ** float(contract["haze_slope"])
            return np.repeat(opacity[:, None], pressure.size, axis=1)

        def deck(wavelength_micron, pressure_bar):
            del pressure_bar
            return np.repeat(
                deck_kappa_cm2_g[None, :], len(wavelength_micron), axis=0
            )

        absorption = deck if include_deck else None
        scattering = haze if include_haze else None
        return absorption, scattering

    output: dict[str, np.ndarray] = {
        "pressure_bar": pressure,
        "temperature_k": temperature,
    }
    timings = {}
    wavelength_cm = None
    for case in CASES:
        absorption, scattering = callbacks(case)
        common = dict(
            temperatures=temperature,
            mass_fractions=mass_fraction,
            mean_molar_masses=mean_molar_mass,
            reference_gravity=gravity * 100.0,
            additional_absorption_opacities_function=absorption,
            additional_scattering_opacities_function=scattering,
            frequencies_to_wavelengths=True,
        )
        started = perf_counter()
        emission = atmosphere.calculate_flux(**common)
        emission_seconds = perf_counter() - started
        started = perf_counter()
        transmission = atmosphere.calculate_transit_radii(
            **common,
            reference_pressure=float(contract["reference_pressure_bar"]),
            planet_radius=float(contract["planet_radius_m"]) * 100.0,
            variable_gravity=True,
        )
        transmission_seconds = perf_counter() - started
        wavelength_cm = np.asarray(emission[0], dtype=float)
        output[f"{case}_flux_cgs_per_cm"] = np.asarray(emission[1], dtype=float)
        output[f"{case}_transit_radius_cm"] = np.asarray(transmission[1], dtype=float)
        timings[case] = {
            "emission": emission_seconds,
            "transmission": transmission_seconds,
        }

    metadata = {
        "petitradtrans_version": petitRADTRANS.__version__,
        "cases": list(CASES),
        "cloud_contract": "native_pRT_continuous_deck_plus_well_mixed_isotropic_haze",
        "construction_seconds": construction_seconds,
        "timing_seconds": timings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        wavelength_cm=wavelength_cm,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **output,
    )


if __name__ == "__main__":
    main()
