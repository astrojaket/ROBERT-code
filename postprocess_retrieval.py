#!/usr/bin/env python3
"""Generate fit diagnostics and plots from one configured ROBERT retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from robert_exoplanets.io.configured_tasks import build_problem, load_observations
from robert_exoplanets.io.task_config import initialize_task_directories, load_task_config
from robert_exoplanets.postprocessing import (
    discover_retrieval_result_directories,
    postprocess_retrieval_output,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--result-dir",
        type=Path,
        action="append",
        help="specific phase directory; repeat as needed (default: discover all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="plot root (default: <configured outputs>/plots)",
    )
    parser.add_argument("--style", help="Matplotlib style name or .mplstyle path")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), dest="image_format")
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--color",
        action="append",
        default=[],
        metavar="DATASET=COLOR",
        help="override a dataset colour; repeat as needed",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="PARAMETER=LABEL",
        help="override a parameter display label; repeat as needed",
    )
    args = parser.parse_args()

    config = load_task_config(args.config)
    initialize_task_directories(config)
    problem = build_problem(config, load_observations(config))
    result_dirs = tuple(args.result_dir or ()) or discover_retrieval_result_directories(
        config.outputs.directory
    )
    if not result_dirs:
        parser.error(f"no completed retrieval results found under {config.outputs.directory}")
    plot_root = (args.output_dir or config.outputs.directory / "plots").expanduser()
    colors = {**config.plotting.dataset_colors, **_assignments(args.color, "--color")}
    labels = {
        parameter.name: parameter.label
        for parameter in config.parameters
        if parameter.label is not None
    }
    labels.update(config.plotting.parameter_labels)
    labels.update(_assignments(args.label, "--label"))
    for result_dir in result_dirs:
        phase = Path(result_dir).name
        diagnostics = postprocess_retrieval_output(
            problem,
            result_dir,
            plot_dir=plot_root / phase,
            parameter_labels=labels,
            dataset_colors=colors,
            style=args.style or config.plotting.style,
            image_format=args.image_format or config.plotting.image_format,
            dpi=args.dpi or config.plotting.dpi,
            max_posterior_samples=(
                args.max_samples or config.plotting.max_posterior_samples
            ),
        )
        print(
            f"{phase}: reduced chi-squared={diagnostics['reduced_chi_squared']}; "
            f"plots={plot_root / phase}",
            flush=True,
        )


def _assignments(values: list[str], option: str) -> dict[str, str]:
    output = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise SystemExit(f"{option} expects NAME=VALUE, received {value!r}")
        output[key] = item
    return output


if __name__ == "__main__":
    main()
