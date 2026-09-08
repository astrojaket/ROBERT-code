#!/usr/bin/env python3
"""Run and inspect a configured ROBERT forward model from Python."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robert_exoplanets.io.configured_tasks import (
    describe_config,
    load_observations,
    prepare_opacity,
    run_forward_task,
)
from robert_exoplanets.io.task_config import (
    initialize_task_directories,
    load_task_config,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configurations/quickstart.yaml"),
        help="schema-version-2 ROBERT YAML configuration",
    )
    argument_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="resolve and validate the YAML without loading scientific data",
    )
    argument_parser.add_argument(
        "--prepare-opacity",
        action="store_true",
        help="prepare the configured opacity cache before evaluating the model",
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    config = load_task_config(args.config)
    print(describe_config(config))
    if args.validate_only:
        return

    initialize_task_directories(config)
    if args.prepare_opacity:
        observations = load_observations(config)
        prepare_opacity(config, observations)

    output = run_forward_task(config, args.config)
    print(f"\nForward product: {output}")
    with np.load(output, allow_pickle=False) as archive:
        for dataset in config.observations.datasets:
            wavelength = archive[f"{dataset}_wavelength_micron"]
            model = archive[f"{dataset}_model"]
            print(
                f"{dataset}: {model.size} bins, "
                f"{wavelength.min():.4g}-{wavelength.max():.4g} micron"
            )
        parameters = sorted(
            key.removeprefix("parameter_")
            for key in archive.files
            if key.startswith("parameter_")
        )
    print("Parameters: " + ", ".join(parameters))


if __name__ == "__main__":
    main()
