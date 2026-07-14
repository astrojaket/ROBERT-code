#!/usr/bin/env python3
"""Validate, prepare, smoke-test, or run OE/UltraNest/MultiNest from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from robert_exoplanets.io.configured_tasks import (
    describe_config,
    load_observations,
    prepare_opacity,
    run_retrieval_task,
    run_smoke_task,
)
from robert_exoplanets.io.task_config import initialize_task_directories, load_task_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="ROBERT YAML configuration")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate-only", action="store_true", help="validate and print the resolved run")
    actions.add_argument("--initialize", action="store_true", help="create configured output, cache, and scratch directories")
    actions.add_argument("--prepare-opacity", action="store_true", help="prepare per-dataset opacity caches, then exit")
    actions.add_argument("--smoke-only", action="store_true", help="evaluate one prior-midpoint likelihood, then exit")
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
    if args.smoke_only:
        print(run_smoke_task(config, args.config), flush=True)
        return
    run_retrieval_task(config, args.config)


if __name__ == "__main__":
    main()
