#!/usr/bin/env python3
"""Plot Stage-9 posterior overlays and five-seed coverage counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    FRAMEWORKS,
    NOISE_TIERS_PPM,
    SCENARIOS,
    parameter_definitions,
)


COLORS = {"robert": "#0072B2", "picaso": "#D55E00", "petitradtrans": "#009E73"}
LINESTYLES = {"robert": "-", "picaso": "--", "petitradtrans": ":"}


def _load_runs(project: Path) -> list[dict[str, Any]]:
    rows = json.loads((project / "run_index.json").read_text(encoding="utf-8"))
    runs = []
    for row in rows:
        config = json.loads((project / row["run_config"]).read_text(encoding="utf-8"))
        result = Path(config["run_directory"]) / "result_arrays.npz"
        summary = Path(config["run_directory"]) / "posterior_summary.json"
        if result.exists() and summary.exists():
            config["result_arrays"] = result
            config["posterior_summary"] = summary
            runs.append(config)
    return runs


def _weighted_histogram(
    path: Path, parameter_index: int, edges: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)[:, parameter_index]
        weights = (
            np.asarray(archive["weights"], dtype=float)
            if "weights" in archive.files
            else np.ones(samples.size)
        )
    histogram, _ = np.histogram(samples, bins=edges, weights=weights, density=True)
    return histogram


def posterior_pdf(project: Path, runs: list[dict[str, Any]], output: Path) -> None:
    """Write overlays pooled with equal weight per completed noise realization."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for scenario in SCENARIOS:
            definitions = parameter_definitions(scenario)
            for tier in NOISE_TIERS_PPM:
                for index, parameter in enumerate(definitions):
                    edges = np.linspace(parameter.lower, parameter.upper, 61)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    fig, axis = plt.subplots(
                        figsize=(9.5, 5.5), constrained_layout=True
                    )
                    plotted = 0
                    for injector in FRAMEWORKS:
                        for retriever in FRAMEWORKS:
                            selected = [
                                run
                                for run in runs
                                if run["scenario"] == scenario.name
                                and run["noise_ppm"] == tier
                                and run["injector"] == injector
                                and run["retriever"] == retriever
                            ]
                            if not selected:
                                continue
                            pooled = np.mean(
                                [
                                    _weighted_histogram(
                                        run["result_arrays"], index, edges
                                    )
                                    for run in selected
                                ],
                                axis=0,
                            )
                            axis.plot(
                                centers,
                                pooled,
                                color=COLORS[retriever],
                                ls=LINESTYLES[injector],
                                lw=1.4,
                                alpha=0.85,
                                label=f"{injector} -> {retriever} (n={len(selected)})",
                            )
                            plotted += 1
                    axis.axvline(
                        parameter.truth, color="black", lw=1.2, alpha=0.7, label="truth"
                    )
                    axis.set(
                        xlabel=f"{parameter.label}{' [' + parameter.unit + ']' if parameter.unit else ''}",
                        ylabel="posterior density",
                        title=f"{scenario.name} | {tier} ppm | {parameter.name}",
                    )
                    axis.grid(alpha=0.2)
                    if plotted:
                        axis.legend(fontsize=7, ncol=3)
                    else:
                        axis.text(
                            0.5,
                            0.5,
                            "No completed retrievals",
                            transform=axis.transAxes,
                            ha="center",
                        )
                    pdf.savefig(fig)
                    plt.close(fig)


def coverage_products(
    project: Path, runs: list[dict[str, Any]], csv_path: Path, plot_path: Path
) -> None:
    """Report coverage as x/5; do not interpret five seeds as precise calibration."""

    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for tier in NOISE_TIERS_PPM:
            for injector in FRAMEWORKS:
                for retriever in FRAMEWORKS:
                    selected = [
                        run
                        for run in runs
                        if run["scenario"] == scenario.name
                        and run["noise_ppm"] == tier
                        and run["injector"] == injector
                        and run["retriever"] == retriever
                        and run["noise_id"].startswith("seed_")
                    ]
                    summaries = [
                        json.loads(
                            Path(run["posterior_summary"]).read_text(encoding="utf-8")
                        )
                        for run in selected
                    ]
                    for parameter in parameter_definitions(scenario):
                        for interval in (68, 95):
                            covered = sum(
                                bool(
                                    summary["credible_intervals"][parameter.name][
                                        f"truth_covered_{interval}"
                                    ]
                                )
                                for summary in summaries
                            )
                            completed = len(summaries)
                            lower = (
                                0.0
                                if covered == 0
                                else float(
                                    beta.ppf(0.025, covered, completed - covered + 1)
                                )
                            )
                            upper = (
                                1.0
                                if covered == completed
                                else float(
                                    beta.ppf(0.975, covered + 1, completed - covered)
                                )
                            )
                            biases = [
                                summary["credible_intervals"][parameter.name][
                                    "median_bias"
                                ]
                                for summary in summaries
                            ]
                            standardized = [
                                summary["credible_intervals"][parameter.name][
                                    "median_bias_posterior_sigma"
                                ]
                                for summary in summaries
                                if summary["credible_intervals"][parameter.name][
                                    "median_bias_posterior_sigma"
                                ]
                                is not None
                            ]
                            records.append(
                                {
                                    "scenario": scenario.name,
                                    "noise_ppm": tier,
                                    "injector": injector,
                                    "retriever": retriever,
                                    "parameter": parameter.name,
                                    "credible_interval_percent": interval,
                                    "covered": covered,
                                    "completed": completed,
                                    "frozen_denominator": 5,
                                    "coverage_fraction": covered / completed
                                    if completed
                                    else "",
                                    "binomial_95_lower": lower if completed else "",
                                    "binomial_95_upper": upper if completed else "",
                                    "mean_median_bias": float(np.mean(biases))
                                    if biases
                                    else "",
                                    "mean_bias_posterior_sigma": (
                                        float(np.mean(standardized))
                                        if standardized
                                        else ""
                                    ),
                                }
                            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    complete = [
        item
        for item in records
        if item["completed"] == 5 and item["credible_interval_percent"] == 68
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, scenario in zip(axes.flat, SCENARIOS, strict=True):
        selected = [item for item in complete if item["scenario"] == scenario.name]
        if selected:
            values = np.asarray([item["covered"] for item in selected], dtype=float)
            axis.hist(
                values,
                bins=np.arange(-0.5, 6.0, 1.0),
                color="#56B4E9",
                edgecolor="black",
            )
        axis.set(
            xticks=range(6),
            xlabel="truth within central 68% interval (x/5)",
            ylabel="parameter/pair/tier cells",
            title=scenario.name,
        )
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Stage 9 five-seed coverage counts (descriptive, not precision calibration)"
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    runs = _load_runs(project)
    output = project / "diagnostics" / "posterior"
    posterior_pdf(project, runs, output / "posterior_comparisons.pdf")
    coverage_products(
        project,
        runs,
        output / "coverage_counts.csv",
        output / "coverage_counts.png",
    )
    print(f"posterior diagnostics used {len(runs)} completed retrievals")


if __name__ == "__main__":
    main()
