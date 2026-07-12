"""Run pRT3 with the six ExoMolOP cross-section tables used by ROBERT."""

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

LINE_SPECIES = [
    "H2O__POKAZATEL",
    "CO__Li2015",
    "CO2__UCL-4000",
    "CH4__YT34to10",
    "NH3__CoYuTe",
    "HCN__Harris",
]
CORRELATED_K_LINE_SPECIES = [
    "H2O__POKAZATEL",
    "CO__HITEMP",
    "CO2__UCL-4000",
    "CH4__YT34to10",
    "NH3__CoYuTe",
    "HCN__Harris",
]
SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
MASS_FRACTIONS = {
    "H2": 0.73654,
    "He": 0.259,
    "H2O": 0.001,
    "CO": 0.003,
    "CO2": 0.0003,
    "CH4": 0.0001,
    "NH3": 0.00003,
    "HCN": 0.00003,
}
MOLAR_MASSES = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.01528,
    "CO": 28.0101,
    "CO2": 44.0095,
    "CH4": 16.04246,
    "NH3": 17.03052,
    "HCN": 27.0253,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sampling", type=int, default=15)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--mode", choices=("opacity-sampling", "correlated-k"), default="opacity-sampling")
    args = parser.parse_args()
    pressure = np.geomspace(1.0e-5, 100.0, args.layers)
    temperature = 900.0 + 900.0 * (np.log10(pressure) + 5.0) / 7.0
    mean_molar_mass = 1.0 / sum(
        MASS_FRACTIONS[name] / MOLAR_MASSES[name] for name in MASS_FRACTIONS
    )
    line_species = LINE_SPECIES if args.mode == "opacity-sampling" else CORRELATED_K_LINE_SPECIES
    mass_fractions = {
        line: np.full(args.layers, MASS_FRACTIONS[species])
        for line, species in zip(line_species, SPECIES, strict=True)
    }
    mass_fractions.update(
        {name: np.full(args.layers, MASS_FRACTIONS[name]) for name in ("H2", "He")}
    )

    start = perf_counter()
    atmosphere = Radtrans(
        pressures=pressure,
        wavelength_boundaries=np.array([1.0, 12.0]),
        line_species=line_species,
        line_opacity_mode="lbl" if args.mode == "opacity-sampling" else "c-k",
        line_by_line_opacity_sampling=(
            args.sampling if args.mode == "opacity-sampling" else 1
        ),
        scattering_in_emission=False,
        path_input_data=str(args.input_data),
    )
    construction_s = perf_counter() - start

    def calculate():
        return atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=mass_fractions,
            mean_molar_masses=np.full(args.layers, mean_molar_mass),
            reference_gravity=1500.0,
            frequencies_to_wavelengths=True,
        )

    start = perf_counter()
    wavelength_cm, flux_cgs_per_cm, _ = calculate()
    first_s = perf_counter() - start
    durations = []
    for _ in range(5):
        start = perf_counter()
        calculate()
        durations.append(perf_counter() - start)
    metadata = {
        "petitradtrans_version": petitRADTRANS.__version__,
        "opacity_mode": args.mode,
        "line_by_line_opacity_sampling": (
            args.sampling if args.mode == "opacity-sampling" else None
        ),
        "n_layers": args.layers,
        "n_wavelength": int(np.asarray(wavelength_cm).size),
        "line_species": line_species,
        "timings": {
            "construct_and_load_s": construction_s,
            "emission_first_s": first_s,
            "emission_steady_median_s": float(np.median(durations)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata)),
        pressure_bar=pressure,
        temperature_K=temperature,
        mean_molar_mass_amu=np.array(mean_molar_mass),
        wavelength_cm=np.asarray(wavelength_cm),
        flux_cgs_per_cm=np.asarray(flux_cgs_per_cm),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
