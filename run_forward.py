#!/usr/bin/env python3
"""Validate, prepare opacity for, or run a ROBERT forward model from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from robert_exoplanets.io.configured_tasks import (
    describe_config,
    load_observations,
    prepare_opacity,
    run_forward_task,
)
from robert_exoplanets.io.task_config import initialize_task_directories, load_task_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="ROBERT YAML configuration")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate-only", action="store_true")
    actions.add_argument("--initialize", action="store_true")
    actions.add_argument("--prepare-opacity", action="store_true")
    args = parser.parse_args()

    config = load_task_config(args.config)
    print(describe_config(config), flush=True)
    if args.validate_only:
        return
    initialize_task_directories(config)
    if args.initialize:
        print("Configured directories created.", flush=True)
        return
    if args.prepare_opacity:
        prepare_opacity(config, load_observations(config))
        print("Opacity preparation complete.", flush=True)
        return
    output = run_forward_task(config, args.config)
    print(f"Forward model written to {output}", flush=True)


if __name__ == "__main__":
    main()
