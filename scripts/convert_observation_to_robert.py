#!/usr/bin/env python3
"""Convert a named-column eclipse-spectrum table to ROBERT NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

from robert_exoplanets.retrieval import convert_emission_observation_table


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("input", type=Path, help="CSV/ECSV/whitespace table")
    argument_parser.add_argument("output", type=Path, help="output .npz path")
    argument_parser.add_argument("--wavelength-column", default="wavelength")
    argument_parser.add_argument("--flux-column", default="flux")
    argument_parser.add_argument("--uncertainty-column", default="uncertainty")
    argument_parser.add_argument("--bin-low-column")
    argument_parser.add_argument("--bin-high-column")
    argument_parser.add_argument(
        "--delimiter",
        choices=("whitespace", "comma", "tab"),
        default="whitespace",
    )
    argument_parser.add_argument(
        "--wavelength-unit",
        choices=("micron", "nm", "angstrom", "m"),
        default="micron",
    )
    argument_parser.add_argument(
        "--flux-unit",
        choices=("eclipse_depth", "fraction", "percent", "ppm"),
        default="eclipse_depth",
    )
    argument_parser.add_argument("--instrument", help="for example JWST/NIRSpec-G395H")
    argument_parser.add_argument("--overwrite", action="store_true")
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    delimiters = {"whitespace": None, "comma": ",", "tab": "\t"}
    output = convert_emission_observation_table(
        args.input,
        args.output,
        wavelength_column=args.wavelength_column,
        flux_column=args.flux_column,
        uncertainty_column=args.uncertainty_column,
        bin_low_column=args.bin_low_column,
        bin_high_column=args.bin_high_column,
        delimiter=delimiters[args.delimiter],
        wavelength_input_unit=args.wavelength_unit,
        flux_input_unit=args.flux_unit,
        instrument=args.instrument,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
