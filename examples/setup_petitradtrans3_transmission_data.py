"""Download the stable-pRT opacity set for 0.3--12 micron transmission tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.radtrans import Radtrans

ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = ROOT / "opacity_data" / "petitRADTRANS" / "input_data"
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
DEFAULT_LINE_FILES = {
    "CO/12C-16O": "12C-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5",
    "CO2/12C-16O2": "12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
    "CH4/12C-1H4": "12C-1H4__YT34to10.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
    "NH3/14N-1H3": "14N-1H3__CoYuTe.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
    "HCN/1H-12C-14N": "1H-12C-14N__Harris.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
}


def main() -> None:
    line_root = INPUT_DATA / "opacities" / "lines" / "correlated_k"
    for relative_directory, filename in DEFAULT_LINE_FILES.items():
        petitradtrans_config_parser.set_default_file(
            str(line_root / relative_directory / filename),
            path_input_data=str(INPUT_DATA),
        )
    atmosphere = Radtrans(
        pressures=np.geomspace(1.0e-5, 100.0, 80),
        wavelength_boundaries=np.array([0.3, 12.0]),
        line_species=LINE_SPECIES,
        gas_continuum_contributors=CIA_SPECIES,
        rayleigh_species=["H2", "He"],
        scattering_in_emission=False,
        path_input_data=str(INPUT_DATA),
    )
    print(f"petitRADTRANS input data: {INPUT_DATA}")
    print(f"line species: {', '.join(LINE_SPECIES)}")
    print(f"loaded frequencies: {atmosphere.frequencies.size}")


if __name__ == "__main__":
    main()
