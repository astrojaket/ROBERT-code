#!/usr/bin/env python3
"""Compare the clear and Mie WASP-69b UltraNest/MultiNest/OE benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np

from robert_exoplanets.io.configured_tasks import build_problem, load_observations
from robert_exoplanets.io.task_config import initialize_task_directories, load_task_config
from robert_exoplanets.postprocessing import (
    posterior_summary,
    postprocess_retrieval_output,
)


METHOD_ORDER = (
    "ultranest",
    "multinest",
    "optimal_estimation",
    "optimal_estimation_to_ultranest",
    "optimal_estimation_to_multinest",
)
METHOD_LABELS = {
    "ultranest": "UltraNest",
    "multinest": "MultiNest",
    "optimal_estimation": "OE",
    "optimal_estimation_to_ultranest": "OE → UltraNest",
    "optimal_estimation_to_multinest": "OE → MultiNest",
}
METHOD_COLORS = {
    "ultranest": "#20639b",
    "multinest": "#ef5675",
    "optimal_estimation": "#2ca25f",
    "optimal_estimation_to_ultranest": "#ffa600",
    "optimal_estimation_to_multinest": "#7a5195",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--style", default="default")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--method-color",
        action="append",
        default=[],
        metavar="ENGINE=COLOR",
        help="override a method colour; repeat as needed",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="compare completed runs instead of requiring the full ten-run matrix",
    )
    args = parser.parse_args()

    project = args.project_dir.expanduser().resolve()
    destination = (args.output_dir or project / "benchmark_comparison").expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    colors = {**METHOD_COLORS, **_assignments(args.method_color)}
    records = _load_records(project)
    expected = {(science, engine) for science in ("clear", "mie_catalog") for engine in METHOD_ORDER}
    present = {(record["science_model"], record["engine"]) for record in records}
    missing = sorted(expected - present)
    if missing and not args.allow_incomplete:
        parser.error(
            "benchmark is incomplete: "
            + ", ".join(f"{science}/{engine}" for science, engine in missing)
        )
    if not records:
        parser.error(f"no completed benchmark runs found beneath {project}")

    _write_summary(records, destination)
    _plot_runtime(records, destination, colors, args.style, args.format, args.dpi)
    _plot_fit_statistics(
        records, destination, colors, args.style, args.format, args.dpi
    )
    _plot_evidence(records, destination, colors, args.style, args.format, args.dpi)
    for science in sorted({record["science_model"] for record in records}):
        _plot_parameters(
            [record for record in records if record["science_model"] == science],
            destination,
            colors,
            args.style,
            args.format,
            args.dpi,
        )
    print(f"Benchmark comparison written to {destination}", flush=True)


def _load_records(project: Path) -> list[dict[str, object]]:
    records = []
    for config_path in sorted(project.glob("*/configuration.yaml")):
        config = load_task_config(config_path)
        engine = config.sampler.engine
        if engine not in METHOD_ORDER:
            continue
        result_dir = _primary_result_directory(config.outputs.directory, engine)
        result_path = result_dir / "result.json"
        arrays_path = result_dir / "result_arrays.npz"
        if not result_path.is_file() or not arrays_path.is_file():
            continue
        plot_dir = config.outputs.directory / "plots" / result_dir.name
        statistics_path = plot_dir / "fit_statistics.json"
        if not statistics_path.is_file():
            initialize_task_directories(config)
            problem = build_problem(config, load_observations(config))
            postprocess_retrieval_output(
                problem,
                result_dir,
                plot_dir=plot_dir,
                parameter_labels={
                    parameter.name: config.plotting.parameter_labels.get(
                        parameter.name, parameter.label or parameter.name
                    )
                    for parameter in config.parameters
                },
                dataset_colors=config.plotting.dataset_colors,
                style=config.plotting.style,
                image_format=config.plotting.image_format,
                dpi=config.plotting.dpi,
                max_posterior_samples=config.plotting.max_posterior_samples,
            )
        result = _read_json(result_path)
        statistics = _read_json(statistics_path)
        with np.load(arrays_path, allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
        names = tuple(str(name) for name in result["parameter_names"])
        posterior = posterior_summary(names, arrays)
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        elapsed = _optional_float(metadata.get("inference_elapsed_seconds"))
        ranks = _submission_ranks(config_path.parent / "submit.sbatch")
        records.append(
            {
                "run_name": config.run.name,
                "science_model": "clear" if config.clouds.model == "none" else "mie_catalog",
                "engine": engine,
                "method_label": METHOD_LABELS[engine],
                "result_directory": str(result_dir.resolve()),
                "converged": bool(result.get("converged")),
                "inference_elapsed_seconds": elapsed,
                "requested_mpi_ranks": ranks,
                "inference_core_hours": (
                    elapsed * ranks / 3600.0 if elapsed is not None and ranks else None
                ),
                "best_fit_log_likelihood": result.get("best_fit_log_likelihood"),
                "chi_squared": statistics.get("chi_squared"),
                "reduced_chi_squared": statistics.get("reduced_chi_squared"),
                "aic": statistics.get("aic"),
                "aicc": statistics.get("aicc"),
                "bic": statistics.get("bic"),
                "log_evidence": result.get("log_evidence"),
                "log_evidence_error": result.get("log_evidence_error"),
                "number_likelihood_calls": metadata.get("ncall"),
                "posterior": posterior,
                "parameter_names": names,
            }
        )
    return sorted(
        records,
        key=lambda record: (
            str(record["science_model"]),
            METHOD_ORDER.index(str(record["engine"])),
        ),
    )


def _primary_result_directory(output: Path, engine: str) -> Path:
    if engine in {"ultranest", "multinest"}:
        return output / engine
    if engine == "optimal_estimation":
        return output / "optimal_estimation"
    return output / "nested_sampling"


def _write_summary(records: list[dict[str, object]], output: Path) -> None:
    serializable = [
        {key: value for key, value in record.items() if key not in {"posterior", "parameter_names"}}
        for record in records
    ]
    (output / "benchmark_summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    fields = tuple(serializable[0])
    with (output / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable)


def _plot_runtime(records, output, colors, style, image_format, dpi) -> None:
    plt = _pyplot()
    with plt.style.context(style):
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for axis, metric, label in (
            (axes[0], "inference_elapsed_seconds", "Inference elapsed time (hours)"),
            (axes[1], "inference_core_hours", "Inference allocation (core-hours)"),
        ):
            labels = []
            values = []
            bar_colors = []
            for record in records:
                value = record[metric]
                if value is None:
                    continue
                labels.append(
                    f"{record['science_model']}\n{record['method_label']}"
                )
                values.append(float(value) / 3600.0 if metric.endswith("seconds") else float(value))
                bar_colors.append(colors[str(record["engine"])])
            axis.bar(range(len(values)), values, color=bar_colors)
            axis.set_xticks(range(len(values)), labels, rotation=55, ha="right")
            axis.set_ylabel(label)
            axis.set_yscale("log")
        figure.suptitle("WASP-69b sampler benchmark runtime")
        figure.tight_layout()
        figure.savefig(output / f"benchmark_runtime.{image_format}", dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_fit_statistics(records, output, colors, style, image_format, dpi) -> None:
    plt = _pyplot()
    with plt.style.context(style):
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for axis, metric, label in (
            (axes[0], "reduced_chi_squared", r"Reduced $\chi^2$"),
            (axes[1], "best_fit_log_likelihood", "Best-fit log likelihood"),
        ):
            labels = []
            values = []
            bar_colors = []
            for record in records:
                value = record[metric]
                if value is None:
                    continue
                labels.append(f"{record['science_model']}\n{record['method_label']}")
                values.append(float(value))
                bar_colors.append(colors[str(record["engine"])])
            axis.bar(range(len(values)), values, color=bar_colors)
            axis.set_xticks(range(len(values)), labels, rotation=55, ha="right")
            axis.set_ylabel(label)
        figure.suptitle("WASP-69b fit-statistic comparison")
        figure.tight_layout()
        figure.savefig(
            output / f"benchmark_fit_statistics.{image_format}",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def _plot_evidence(records, output, colors, style, image_format, dpi) -> None:
    nested = [record for record in records if record["log_evidence"] is not None]
    if not nested:
        return
    plt = _pyplot()
    with plt.style.context(style):
        figure, axis = plt.subplots(figsize=(11, 5.5))
        labels = [f"{r['science_model']}\n{r['method_label']}" for r in nested]
        values = [float(r["log_evidence"]) for r in nested]
        errors = [float(r["log_evidence_error"] or 0.0) for r in nested]
        axis.errorbar(
            range(len(nested)),
            values,
            yerr=errors,
            fmt="none",
            ecolor="0.2",
            capsize=3,
        )
        axis.scatter(
            range(len(nested)),
            values,
            c=[colors[str(r["engine"])] for r in nested],
            s=60,
        )
        axis.set_xticks(range(len(nested)), labels, rotation=55, ha="right")
        axis.set_ylabel("Log evidence")
        axis.set_title("Nested-sampler evidence comparison")
        figure.tight_layout()
        figure.savefig(output / f"benchmark_evidence.{image_format}", dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_parameters(records, output, colors, style, image_format, dpi) -> None:
    if not records:
        return
    names = tuple(records[0]["parameter_names"])
    shared = tuple(
        name
        for name in names
        if all(name in tuple(record["parameter_names"]) for record in records)
    )
    plt = _pyplot()
    rows = math.ceil(len(shared) / 3)
    with plt.style.context(style):
        figure, axes = plt.subplots(rows, 3, figsize=(13, 3.5 * rows))
        flat_axes = np.atleast_1d(axes).ravel()
        for parameter_index, name in enumerate(shared):
            axis = flat_axes[parameter_index]
            for method_index, record in enumerate(records):
                quantiles = record["posterior"]["quantiles_16_50_84"][name]
                lower, median, upper = (float(value) for value in quantiles)
                axis.errorbar(
                    method_index,
                    median,
                    yerr=[[median - lower], [upper - median]],
                    fmt="o",
                    color=colors[str(record["engine"])],
                    capsize=3,
                )
            axis.set_xticks(
                range(len(records)),
                [str(record["method_label"]) for record in records],
                rotation=55,
                ha="right",
            )
            axis.set_ylabel(name)
        for axis in flat_axes[len(shared) :]:
            axis.set_visible(False)
        science = str(records[0]["science_model"])
        figure.suptitle(f"{science}: shared-parameter constraints")
        figure.tight_layout()
        figure.savefig(
            output / f"benchmark_parameters_{science}.{image_format}",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def _submission_ranks(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(r"^#SBATCH\s+--ntasks=(\d+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return None if match is None else int(match.group(1))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _optional_float(value: object) -> float | None:
    return None if value in (None, "") else float(value)


def _assignments(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or key not in METHOD_ORDER or not item:
            raise SystemExit(f"--method-color expects ENGINE=COLOR, received {value!r}")
        output[key] = item
    return output


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    main()
