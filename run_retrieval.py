#!/usr/bin/env python3
"""Validate, prepare, smoke-test, or run OE/UltraNest/MultiNest from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from robert_exoplanets.io.configured_tasks import (
    describe_config,
    load_observations,
    prepare_opacity,
    run_retrieval_task,
    run_smoke_task,
)
from robert_exoplanets.io.task_config import initialize_task_directories, load_task_config
from robert_exoplanets.retrieval.samplers.callback_supervisor import (
    supervise_retrieval_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="ROBERT YAML configuration")
    parser.add_argument(
        "--supervise-callback-failures",
        action="store_true",
        help="run a serial MultiNest child and stop it if a Python callback fails",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate-only", action="store_true", help="validate and print the resolved run")
    actions.add_argument("--initialize", action="store_true", help="create configured output, cache, and scratch directories")
    actions.add_argument("--prepare-opacity", action="store_true", help="prepare per-dataset opacity caches, then exit")
    actions.add_argument("--smoke-only", action="store_true", help="evaluate one prior-midpoint likelihood, then exit")
    args = parser.parse_args()

    if args.supervise_callback_failures:
        if args.validate_only or args.initialize or args.prepare_opacity or args.smoke_only:
            parser.error("--supervise-callback-failures applies only to retrieval runs")
        config = load_task_config(args.config)
        if "multinest" not in config.sampler.engine:
            parser.error("--supervise-callback-failures requires a MultiNest workflow")
        if config.runtime.mpi_processes not in {"auto", 1}:
            parser.error("callback supervision requires runtime.mpi_processes=auto or 1")
        nested_output_dir = (
            config.outputs.directory / "multinest"
            if config.sampler.engine == "multinest"
            else config.outputs.directory / "nested_sampling"
        )
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.expanduser().resolve()),
        ]
        return supervise_retrieval_command(
            command,
            fallback_output_dir=nested_output_dir,
        )

    config = load_task_config(args.config)
    print(describe_config(config), flush=True)
    if args.validate_only:
        return 0
    initialize_task_directories(config)
    if args.initialize:
        print("Configured directories created.", flush=True)
        return 0
    if args.prepare_opacity:
        prepare_opacity(config, load_observations(config))
        print("Opacity preparation complete.", flush=True)
        return 0
    if args.smoke_only:
        print(run_smoke_task(config, args.config), flush=True)
        return 0
    run_retrieval_task(config, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
