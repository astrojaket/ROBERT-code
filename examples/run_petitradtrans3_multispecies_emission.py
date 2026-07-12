"""Run the stable-pRT3 six-molecule+CIA thermal-emission reference."""

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
    "CO__HITEMP",
    "CO2__UCL-4000",
    "CH4__YT34to10",
    "NH3__CoYuTe",
    "HCN__Harris",
]
CIA_SPECIES = [
    "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
    "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
]
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
LINE_TO_COMPOSITION = dict(
    zip(LINE_SPECIES, ("H2O", "CO", "CO2", "CH4", "NH3", "HCN"), strict=True)
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--flux-only", action="store_true")
    args = parser.parse_args()
    n_layers = args.layers
    pressure = np.geomspace(1.0e-5, 100.0, n_layers)
    temperature = 900.0 + 900.0 * (np.log10(pressure) + 5.0) / 7.0
    mean_molar_mass = 1.0 / sum(
        MASS_FRACTIONS[species] / MOLAR_MASSES[species] for species in MASS_FRACTIONS
    )
    mass_fractions = {
        line: np.full(n_layers, MASS_FRACTIONS[species])
        for line, species in LINE_TO_COMPOSITION.items()
    }
    mass_fractions["H2"] = np.full(n_layers, MASS_FRACTIONS["H2"])
    mass_fractions["He"] = np.full(n_layers, MASS_FRACTIONS["He"])
    mean_molar_masses = np.full(n_layers, mean_molar_mass)

    start = perf_counter()
    atmosphere = Radtrans(
        pressures=pressure,
        wavelength_boundaries=np.array([0.3, 12.0]),
        line_species=LINE_SPECIES,
        gas_continuum_contributors=CIA_SPECIES,
        scattering_in_emission=False,
        path_input_data=str(args.input_data),
    )
    construction_s = perf_counter() - start

    def calculate(*, diagnostics: bool = False):
        return atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=1500.0,
            frequencies_to_wavelengths=True,
            return_contribution=diagnostics,
            return_opacities=diagnostics,
        )

    start = perf_counter()
    reference = calculate(diagnostics=not args.flux_only)
    first_s = perf_counter() - start
    steady = []
    for _ in range(7):
        start = perf_counter()
        calculate()
        steady.append(perf_counter() - start)
    extra = reference[2]
    volume_fractions = {
        species: (MASS_FRACTIONS[species] / MOLAR_MASSES[species]) * mean_molar_mass
        for species in MASS_FRACTIONS
    }
    metadata = {
        "petitradtrans_version": petitRADTRANS.__version__,
        "n_layers": n_layers,
        "line_species": LINE_SPECIES,
        "cia_species": CIA_SPECIES,
        "mass_fractions": MASS_FRACTIONS,
        "volume_fractions": volume_fractions,
        "mean_molar_mass_amu": mean_molar_mass,
        "diagnostic_keys": sorted(extra),
        "timings": {
            "construct_and_load_s": construction_s,
            "emission_first_with_diagnostics_s": first_s,
            "emission_steady_median_s": float(np.median(steady)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.flux_only:
        np.savez_compressed(
            args.output,
            metadata_json=np.array(json.dumps(metadata)),
            pressure_bar=pressure,
            temperature_K=temperature,
            wavelength_cm=np.asarray(reference[0], dtype=float),
            flux_cgs_per_cm=np.asarray(reference[1], dtype=float),
        )
        print(json.dumps(metadata, indent=2))
        return
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata)),
        pressure_bar=pressure,
        temperature_K=temperature,
        wavelength_cm=np.asarray(reference[0], dtype=float),
        flux_cgs_per_cm=np.asarray(reference[1], dtype=float),
        absorption_opacities=np.asarray(extra["opacities"], dtype=float),
        optical_depths=np.asarray(extra["optical_depths"], dtype=float),
        contribution=np.asarray(extra["emission_contribution"], dtype=float),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
