#!/usr/bin/env python3
"""Generate fit diagnostics and plots from one configured ROBERT forward run."""

from __future__ import annotations

import argparse
from pathlib import Path

from robert_exoplanets.io.configured_tasks import build_problem, load_observations
from robert_exoplanets.io.task_config import initialize_task_directories, load_task_config
from robert_exoplanets.postprocessing import postprocess_forward_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--forward-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--style", help="Matplotlib style name or .mplstyle path")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), dest="image_format")
    parser.add_argument("--dpi", type=int)
    parser.add_argument(
        "--color",
        action="append",
        default=[],
        metavar="DATASET=COLOR",
        help="override a dataset colour; repeat as needed",
    )
    args = parser.parse_args()

    config = load_task_config(args.config)
    initialize_task_directories(config)
    problem = build_problem(config, load_observations(config))
    source = args.forward_file or config.outputs.directory / "forward_model.npz"
    destination = args.output_dir or config.outputs.directory / "plots" / "forward"
    colors = {**config.plotting.dataset_colors, **_assignments(args.color)}
    diagnostics = postprocess_forward_output(
        problem,
        source,
        plot_dir=destination,
        dataset_colors=colors,
        style=args.style or config.plotting.style,
        image_format=args.image_format or config.plotting.image_format,
        dpi=args.dpi or config.plotting.dpi,
    )
    print(
        f"reduced chi-squared={diagnostics['reduced_chi_squared']}; "
        f"plots={destination}",
        flush=True,
    )


def _assignments(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise SystemExit(f"--color expects DATASET=COLOR, received {value!r}")
        output[key] = item
    return output


if __name__ == "__main__":
    main()
