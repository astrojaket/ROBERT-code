"""Run the isolated stable petitRADTRANS 3 reference and save raw outputs."""

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

LINE_SPECIES = "H2O__POKAZATEL"
CIA_SPECIES = [
    "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
    "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
]
MASS_FRACTIONS = {"H2": 0.740, "He": 0.259, "H2O": 0.001}
MOLAR_MASSES = {"H2": 2.01588, "He": 4.002602, "H2O": 18.01528}
GRAVITY_CGS = 1500.0
PLANET_RADIUS_CM = 1.0e10
REFERENCE_PRESSURE_BAR = 0.01
WAVELENGTH_BOUNDARIES_MICRON = np.array([0.3, 12.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--emission-only", action="store_true")
    args = parser.parse_args()
    pressure = np.geomspace(1.0e-5, 100.0, args.layers)
    temperature = 900.0 + 900.0 * (np.log10(pressure) + 5.0) / 7.0
    mean_molar_mass = 1.0 / sum(
        MASS_FRACTIONS[species] / MOLAR_MASSES[species] for species in MASS_FRACTIONS
    )
    mass_fractions = {
        LINE_SPECIES: np.full(args.layers, MASS_FRACTIONS["H2O"]),
        "H2": np.full(args.layers, MASS_FRACTIONS["H2"]),
        "He": np.full(args.layers, MASS_FRACTIONS["He"]),
    }
    mean_molar_masses = np.full(args.layers, mean_molar_mass)

    start = perf_counter()
    atmosphere = Radtrans(
        pressures=pressure,
        wavelength_boundaries=WAVELENGTH_BOUNDARIES_MICRON,
        line_species=[LINE_SPECIES],
        gas_continuum_contributors=CIA_SPECIES,
        scattering_in_emission=False,
        path_input_data=str(args.input_data),
    )
    construct_s = perf_counter() - start

    def emission(return_opacities: bool = False):
        return atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=GRAVITY_CGS,
            frequencies_to_wavelengths=True,
            return_opacities=return_opacities,
        )

    def transmission(return_opacities: bool = False):
        return atmosphere.calculate_transit_radii(
            temperatures=temperature,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=GRAVITY_CGS,
            reference_pressure=REFERENCE_PRESSURE_BAR,
            planet_radius=PLANET_RADIUS_CM,
            variable_gravity=False,
            frequencies_to_wavelengths=True,
            return_opacities=return_opacities,
        )

    if args.emission_only:
        wavelength, flux, _ = emission(False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            n_layers=np.array(args.layers),
            wavelength_cm=np.asarray(wavelength, dtype=float),
            flux_cgs_per_cm=np.asarray(flux, dtype=float),
        )
        return

    emission_first_s, emission_result = _time_first(lambda: emission(True))
    emission_steady_s = _time_steady(emission)
    transmission_first_s, transmission_result = _time_first(lambda: transmission(True))
    transmission_steady_s = _time_steady(transmission)
    wavelength_cm = np.asarray(emission_result[0], dtype=float)
    if not np.allclose(wavelength_cm, np.asarray(transmission_result[0], dtype=float)):
        raise RuntimeError("stable pRT emission and transmission wavelength grids differ")

    metadata = {
        "petitradtrans_version": petitRADTRANS.__version__,
        "n_layers": args.layers,
        "n_wavelength": int(wavelength_cm.size),
        "wavelength_micron": [float(wavelength_cm[0] * 1.0e4), float(wavelength_cm[-1] * 1.0e4)],
        "line_species": LINE_SPECIES,
        "cia_species": CIA_SPECIES,
        "timings": {
            "construct_and_load_s": construct_s,
            "emission_first_s": emission_first_s,
            "emission_steady_median_s": emission_steady_s,
            "transmission_first_s": transmission_first_s,
            "transmission_steady_median_s": transmission_steady_s,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata)),
        pressure_bar=pressure,
        temperature_K=temperature,
        mean_molar_mass_amu=np.array(mean_molar_mass),
        wavelength_cm=wavelength_cm,
        flux_cgs_per_cm=np.asarray(emission_result[1], dtype=float),
        transit_radius_cm=np.asarray(transmission_result[1], dtype=float),
        emission_opacities=np.asarray(emission_result[2]["opacities"], dtype=float),
        emission_optical_depths=np.asarray(emission_result[2]["optical_depths"], dtype=float),
        transmission_opacities=np.asarray(transmission_result[2]["opacities"], dtype=float),
    )
    print(json.dumps(metadata, indent=2))


def _time_first(function):
    start = perf_counter()
    result = function()
    return perf_counter() - start, result


def _time_steady(function, repeats: int = 7) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


if __name__ == "__main__":
    main()
