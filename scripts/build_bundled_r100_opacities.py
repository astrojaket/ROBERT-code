#!/usr/bin/env python
"""Build ROBERT's bundled R=100 opacity set from ExoMolOP R=1000 KTA files."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from robert_exoplanets.opacity import read_kta_header
from robert_exoplanets.opacity.exok import _import_exok
from robert_exoplanets.validation import constant_resolving_power_grid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src" / "robert_exoplanets" / "data" / "opacities" / "R100"
SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
SOURCE_SHA256 = {
    "H2O": "71c38634b391141177ae2a029f2fac00d78fd172fe52f4f10da361e975971d9f",
    "CO": "f87d35061fa21a11c68e84a6dc2276f3d89e31f9321dc97b47b2c288766894d9",
    "CO2": "b4bc9f544115dd490ef545e03693cc758cd59195c853503920efef98287879a7",
    "CH4": "6c83b1d7b7cc93007fa90a6c8bdc673638dc7cfabae343219396cdf2e879778d",
    "NH3": "6fcb97aee63318df3c9c3ad7e2c646d5a5b8ba8284c66259b01b87caa8e911ac",
    "HCN": "203c6b802d07becda6cee9328ab6abb4ad6b8cafed9d27f3014b06e1604f6b54",
}
UPSTREAM = {
    "H2O": {
        "isotopologue": "1H2-16O",
        "line_list": "POKAZATEL",
        "filename": "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.NEMESIS.kta",
    },
    "CO": {
        "isotopologue": "12C-16O",
        "line_list": "Li2015",
        "filename": "12C-16O__Li2015.R1000_0.3-50mu.ktable.NEMESIS.kta",
    },
    "CO2": {
        "isotopologue": "12C-16O2",
        "line_list": "UCL-4000",
        "filename": "12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.NEMESIS.kta",
    },
    "CH4": {
        "isotopologue": "12C-1H4",
        "line_list": "YT34to10",
        "filename": "12C-1H4__YT34to10.R1000_0.3-50mu.ktable.NEMESIS.kta",
    },
    "NH3": {
        "isotopologue": "14N-1H3",
        "line_list": "CoYuTe",
        "filename": "14N-1H3__CoYuTe.R1000_0.3-50mu.ktable.NEMESIS.kta",
    },
    "HCN": {
        "isotopologue": "1H-12C-14N",
        "line_list": "Harris",
        "filename": "1H-12C-14N__Harris.R1000_0.3-50mu.ktable.NEMESIS.kta",
    },
}


def build_opacities(
    source_directory: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate the complete bundled opacity set and return its manifest."""

    sources = {
        species: source_directory / f"{species}_R1000.kta" for species in SPECIES
    }
    for species, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {species} source table: {path}")
        checksum = _sha256(path)
        if checksum != SOURCE_SHA256[species]:
            raise ValueError(
                f"{species} source checksum is {checksum}, expected {SOURCE_SHA256[species]}"
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    targets = {species: output_directory / f"{species}_R100.kta" for species in SPECIES}
    existing = tuple(path for path in targets.values() if path.exists())
    if existing and not overwrite:
        raise FileExistsError(
            f"{len(existing)} opacity outputs already exist; pass --overwrite to replace them"
        )

    exok = _import_exok()
    wavelength_grid = constant_resolving_power_grid(0.3, 15.0, 100.0)
    wavenumber_edges = np.sort(10000.0 / wavelength_grid.edges_micron)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    g_samples = (nodes + 1.0) / 2.0
    g_weights = weights / 2.0
    products: dict[str, object] = {}

    for species in SPECIES:
        source = sources[species]
        native = exok.Ktable(
            filename=str(source),
            mol=species,
            p_unit="bar",
            kdata_unit="cm^2/molecule",
            remove_zeros=False,
        )
        nonfinite = ~np.isfinite(native.kdata)
        nonfinite_count = int(np.sum(nonfinite))
        if nonfinite_count:
            native.kdata[nonfinite] = 1.0e-300
        zero_count = int(np.sum(native.kdata == 0.0))
        binned = native.bin_down_cp(
            wnedges=wavenumber_edges,
            weights=g_weights,
            ggrid=g_samples,
            num=300,
            use_rebin=False,
            remove_zeros=True,
        )

        target = targets[species]
        temporary = target.with_suffix(".tmp.kta")
        if temporary.exists():
            temporary.unlink()
        binned.write_nemesis(str(temporary))
        temporary.replace(target)
        header = read_kta_header(target, checksum=True)
        if header.native_shape != (22, 27, wavelength_grid.n_bins, 8):
            raise RuntimeError(
                f"{species} output shape is {header.native_shape}, expected "
                f"(22, 27, {wavelength_grid.n_bins}, 8)"
            )
        products[species] = {
            **UPSTREAM[species],
            "source_path_convention": f"{species}_R1000.kta",
            "source_sha256": SOURCE_SHA256[species],
            "source_url": _source_url(species),
            "source_nonfinite_replaced": nonfinite_count,
            "source_zeros_seen": zero_count,
            "output_filename": target.name,
            "output_sha256": header.checksum_sha256,
            "output_size_bytes": header.file_size_bytes,
            "native_shape": list(header.native_shape),
        }
        del native, binned
        gc.collect()

    manifest: dict[str, object] = {
        "name": "ROBERT bundled molecular opacity",
        "resolution": "R100",
        "wavelength_min_micron": 0.3,
        "wavelength_max_micron": 15.0,
        "spectral_bins": wavelength_grid.n_bins,
        "effective_resolving_power": wavelength_grid.effective_resolving_power,
        "pressure_points": 22,
        "pressure_min_bar": 1.0e-5,
        "pressure_max_bar": 100.0,
        "temperature_points": 27,
        "temperature_min_k": 100.0,
        "temperature_max_k": 3400.0,
        "g_points": 8,
        "g_samples": g_samples.tolist(),
        "g_weights": g_weights.tolist(),
        "source_resolution": "R1000",
        "source_wavelength_range_micron": [0.3, 50.0],
        "spectral_binning": {
            "software": "exo_k",
            "software_version": str(exok.__version__),
            "method": "Ktable.bin_down_cp",
            "num": 300,
            "use_rebin": False,
            "remove_zeros": True,
        },
        "source_database": "ExoMolOP",
        "source_database_doi": "10.1051/0004-6361/202038350",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "changes": (
            "ROBERT truncated the ExoMolOP 0.3-50 micron R=1000 tables to "
            "0.3-15 micron, spectrally recompressed them to R=100, and "
            "recompressed each correlated-k distribution to eight Gauss points."
        ),
        "products": products,
    }
    manifest_path = output_directory / "provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _source_url(species: str) -> str:
    details = UPSTREAM[species]
    return (
        "https://www.exomol.com/db/"
        f"{species}/{details['isotopologue']}/{details['line_list']}/"
        f"{details['filename']}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory",
        type=Path,
        required=True,
        help="directory containing SPECIES_R1000.kta source files",
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_opacities(
        args.source_directory.expanduser().resolve(),
        args.output_directory.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    total = sum(
        int(item["output_size_bytes"]) for item in manifest["products"].values()
    )
    print(
        f"Wrote {len(SPECIES)} R=100 opacity tables "
        f"({total / 1024**2:.1f} MiB) to {args.output_directory}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
