#!/usr/bin/env python3
"""Plot Stage-9 posterior overlays and single-run truth-recovery diagnostics."""

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
    """Write posterior overlays for completed directed retrievals."""

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


def truth_recovery_products(
    runs: list[dict[str, Any]], csv_path: Path, plot_path: Path
) -> None:
    """Report per-run truth inclusion and bias without claiming coverage."""

    records: list[dict[str, Any]] = []
    for run in runs:
        if run["noise_id"] != "mean":
            continue
        summary = json.loads(Path(run["posterior_summary"]).read_text(encoding="utf-8"))
        intervals = summary["credible_intervals"]
        for parameter in parameter_definitions(run["scenario"]):
            values = intervals[parameter.name]
            records.append(
                {
                    "run_id": run["run_id"],
                    "scenario": run["scenario"],
                    "noise_ppm": run["noise_ppm"],
                    "injector": run["injector"],
                    "retriever": run["retriever"],
                    "parameter": parameter.name,
                    "truth": parameter.truth,
                    "posterior_median": values["q50"],
                    "median_bias": values["median_bias"],
                    "median_bias_posterior_sigma": values[
                        "median_bias_posterior_sigma"
                    ],
                    "truth_within_central_68": values["truth_covered_68"],
                    "truth_within_central_95": values["truth_covered_95"],
                    "central_68_width": values["q84"] - values["q16"],
                    "central_95_width": values["q975"] - values["q025"],
                }
            )
    fieldnames = [
        "run_id",
        "scenario",
        "noise_ppm",
        "injector",
        "retriever",
        "parameter",
        "truth",
        "posterior_median",
        "median_bias",
        "median_bias_posterior_sigma",
        "truth_within_central_68",
        "truth_within_central_95",
        "central_68_width",
        "central_95_width",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, scenario in zip(axes.flat, SCENARIOS, strict=True):
        values = np.asarray(
            [
                item["median_bias_posterior_sigma"]
                for item in records
                if item["scenario"] == scenario.name
                and item["median_bias_posterior_sigma"] is not None
            ],
            dtype=float,
        )
        if values.size:
            axis.hist(values, bins=25, color="#56B4E9", edgecolor="black")
        axis.axvline(0.0, color="black", lw=1.2)
        axis.axvline(-1.0, color="black", lw=1.0, ls="--", alpha=0.6)
        axis.axvline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
        axis.set(
            xlabel="posterior median bias / posterior standard deviation",
            ylabel="completed parameter/run cells",
            title=scenario.name,
        )
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Stage 9 single-realization truth-recovery diagnostics")
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
    truth_recovery_products(
        runs,
        output / "truth_recovery.csv",
        output / "truth_recovery.png",
    )
    print(f"posterior diagnostics used {len(runs)} completed retrievals")


if __name__ == "__main__":
    main()
