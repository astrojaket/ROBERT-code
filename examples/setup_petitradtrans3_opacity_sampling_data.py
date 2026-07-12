"""Create pRT3-compatible local copies of the downloaded ExoMol tables."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

TABLES = {
    "H2O": ("1H2-16O", "1H2-16O__POKAZATEL.R15000_0.3-50mu.xsec.TauREx.h5", 18.01528),
    "CO": ("12C-16O", "12C-16O__Li2015.R15000_0.3-50mu.xsec.TauREx.h5", 28.0101),
    "CO2": ("12C-16O2", "12C-16O2__UCL-4000.R15000_0.3-50mu.xsec.TauREx.h5", 44.0095),
    "CH4": ("12C-1H4", "12C-1H4__YT34to10.R15000_0.3-50mu.xsec.TauREx.h5", 16.04246),
    "NH3": ("14N-1H3", "14N-1H3__CoYuTe.R15000_0.3-50mu.xsec.TauREx.h5", 17.03052),
    "HCN": ("1H-12C-14N", "1H-12C-14N__Harris.R15000_0.3-50mu.xsec.TauREx.h5", 27.0253),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=Path("opacity_data/exomol_xsec"))
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=Path("opacity_data/exomol_pRT/input_data"),
    )
    args = parser.parse_args()
    root = args.destination / "opacities" / "lines" / "line_by_line"
    for species, (isotopologue, filename, molar_mass) in TABLES.items():
        source = args.source / f"{species}.h5"
        destination = root / species / isotopologue / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        with h5py.File(destination, "r+") as handle:
            if "mol_mass" not in handle:
                handle.create_dataset("mol_mass", data=np.array([molar_mass]))
        print(destination)


if __name__ == "__main__":
    main()
