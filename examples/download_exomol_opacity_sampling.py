"""Download and verify the six ExoMolOP opacity-sampling cross sections."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

FILES = {
    "H2O": (
        "https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL__R15000_0.3-50mu.xsec.TauREx.h5",
        "ba9d455db93809661aee79e7d322fde19da0236af1c1a9bb68f5b24735efb7fb",
    ),
    "CO": (
        "https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.R15000_0.3-50mu.xsec.TauREx.h5",
        "4f66cfca115eeb76fd8580bdf10f616db038d928b4af9d18ddc0c584a7116596",
    ),
    "CO2": (
        "https://www.exomol.com/db/CO2/12C-16O2/UCL-4000/12C-16O2__UCL-4000.R15000_0.3-50mu.xsec.TauREx.h5",
        "de25847fada9879ee11dc9bfd031b57e8747bbef108d05c242b62c2cfe82b314",
    ),
    "CH4": (
        "https://www.exomol.com/db/CH4/12C-1H4/YT34to10/12C-1H4__YT34to10.R15000_0.3-50mu.xsec.TauREx.h5",
        "87fe3971038a3b1975a341cf2ef8632d76ec612b093f812479833072c738d11d",
    ),
    "NH3": (
        "https://www.exomol.com/db/NH3/14N-1H3/CoYuTe/14N-1H3__CoYuTe.R15000_0.3-50mu.xsec.TauREx.h5",
        "e5069a45aabffad377e26cb8025ae051ed64c3109ab31d41cff9081461bba8e7",
    ),
    "HCN": (
        "https://www.exomol.com/db/HCN/1H-12C-14N/Harris/1H-12C-14N__Harris.R15000_0.3-50mu.xsec.TauREx.h5",
        "9d0a50c7a08a788d52c8947787bdd6be59acd9a00960ee59d3f15fba979fb3df",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("opacity_data/exomol_xsec"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for species, (url, expected) in FILES.items():
        destination = args.directory / f"{species}.h5"
        if not destination.is_file() and args.verify_only:
            raise FileNotFoundError(destination)
        if not destination.is_file():
            partial = destination.with_suffix(".h5.part")
            print(f"downloading {species} from ExoMol")
            urllib.request.urlretrieve(url, partial)
            partial.replace(destination)
        actual = _sha256(destination)
        if actual != expected:
            raise RuntimeError(
                f"checksum mismatch for {species}: expected {expected}, got {actual}"
            )
        records[species] = {
            "path": str(destination),
            "url": url,
            "sha256": actual,
            "bytes": destination.stat().st_size,
        }
    print(json.dumps(records, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
