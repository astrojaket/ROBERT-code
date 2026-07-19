#!/usr/bin/env python3
"""Plot Stage-9 native injection and posterior spectral comparisons."""

from __future__ import annotations

import argparse
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
)


COLORS = {"robert": "#0072B2", "picaso": "#D55E00", "petitradtrans": "#009E73"}


def _injection(
    project: Path, framework: str, scenario: str
) -> tuple[np.ndarray, np.ndarray]:
    path = project / "injections" / framework / scenario / "native_mean.npz"
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["wavelength_micron"], dtype=float),
            np.asarray(archive["eclipse_depth"], dtype=float),
        )


def plot_injections(project: Path, output: Path) -> None:
    fig, axes = plt.subplots(
        4, 2, figsize=(13, 15), sharex="col", constrained_layout=True
    )
    for row, scenario in enumerate(SCENARIOS):
        spectra = {}
        wavelength = None
        for framework in FRAMEWORKS:
            wavelength, spectra[framework] = _injection(
                project, framework, scenario.name
            )
            axes[row, 0].plot(
                wavelength,
                spectra[framework] * 1.0e6,
                color=COLORS[framework],
                label=framework,
            )
        reference = spectra["robert"]
        for framework in ("picaso", "petitradtrans"):
            axes[row, 1].plot(
                wavelength,
                (spectra[framework] - reference) * 1.0e6,
                color=COLORS[framework],
                label=f"{framework} - robert",
            )
        axes[row, 0].set(ylabel="eclipse depth [ppm]", title=scenario.name)
        axes[row, 1].set(
            ylabel="difference [ppm]", title=f"{scenario.name} differences"
        )
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("wavelength [micron]")
    axes[0, 0].legend()
    axes[0, 1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _completed_mean_runs(project: Path) -> list[dict[str, Any]]:
    rows = json.loads((project / "run_index.json").read_text(encoding="utf-8"))
    completed = []
    for row in rows:
        config = json.loads((project / row["run_config"]).read_text(encoding="utf-8"))
        spectra = Path(config["run_directory"]) / "diagnostic_spectra.npz"
        if config["noise_id"] == "mean" and spectra.exists():
            config["diagnostic_spectra"] = spectra
            completed.append(config)
    return completed


def plot_retrieval_spectra(project: Path, output: Path) -> None:
    runs = _completed_mean_runs(project)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for scenario in SCENARIOS:
            for tier in NOISE_TIERS_PPM:
                selected = [
                    run
                    for run in runs
                    if run["scenario"] == scenario.name and run["noise_ppm"] == tier
                ]
                fig, axes = plt.subplots(
                    3, 3, figsize=(14, 10), sharex=True, constrained_layout=True
                )
                for axis, injector in zip(axes[:, 0], FRAMEWORKS, strict=True):
                    for retriever in FRAMEWORKS:
                        match = next(
                            (
                                run
                                for run in selected
                                if run["injector"] == injector
                                and run["retriever"] == retriever
                            ),
                            None,
                        )
                        if match is None:
                            continue
                        with np.load(
                            match["diagnostic_spectra"], allow_pickle=False
                        ) as archive:
                            wavelength = np.asarray(
                                archive["wavelength_micron"], dtype=float
                            )
                            injection = np.asarray(
                                archive["injection_eclipse_depth"], dtype=float
                            )
                            median = np.asarray(
                                archive["posterior_median_eclipse_depth"], dtype=float
                            )
                        axis.plot(wavelength, injection * 1.0e6, color="black", lw=1.0)
                        axis.plot(
                            wavelength,
                            median * 1.0e6,
                            color=COLORS[retriever],
                            label=retriever,
                        )
                    axis.set_ylabel(f"inj {injector}\nppm")
                    axis.grid(alpha=0.2)
                for axis, injector in zip(axes[:, 1], FRAMEWORKS, strict=True):
                    for retriever in FRAMEWORKS:
                        match = next(
                            (
                                run
                                for run in selected
                                if run["injector"] == injector
                                and run["retriever"] == retriever
                            ),
                            None,
                        )
                        if match is None:
                            continue
                        with np.load(
                            match["diagnostic_spectra"], allow_pickle=False
                        ) as archive:
                            wavelength = np.asarray(
                                archive["wavelength_micron"], dtype=float
                            )
                            residual = np.asarray(
                                archive["posterior_median_eclipse_depth"], dtype=float
                            ) - np.asarray(
                                archive["injection_eclipse_depth"], dtype=float
                            )
                        axis.plot(
                            wavelength,
                            residual * 1.0e6,
                            color=COLORS[retriever],
                            label=retriever,
                        )
                    axis.axhline(0.0, color="black", lw=0.7)
                    axis.grid(alpha=0.2)
                # The third column shows best-fit minus injection for the same cells.
                for axis, injector in zip(axes[:, 2], FRAMEWORKS, strict=True):
                    for retriever in FRAMEWORKS:
                        match = next(
                            (
                                run
                                for run in selected
                                if run["injector"] == injector
                                and run["retriever"] == retriever
                            ),
                            None,
                        )
                        if match is None:
                            continue
                        with np.load(
                            match["diagnostic_spectra"], allow_pickle=False
                        ) as archive:
                            wavelength = np.asarray(
                                archive["wavelength_micron"], dtype=float
                            )
                            residual = np.asarray(
                                archive["best_fit_eclipse_depth"], dtype=float
                            ) - np.asarray(
                                archive["injection_eclipse_depth"], dtype=float
                            )
                        axis.plot(
                            wavelength,
                            residual * 1.0e6,
                            color=COLORS[retriever],
                            label=retriever,
                        )
                    axis.axhline(0.0, color="black", lw=0.7)
                    axis.grid(alpha=0.2)
                axes[0, 0].set_title("injection and posterior median")
                axes[0, 1].set_title("posterior median - injection [ppm]")
                axes[0, 2].set_title("best fit - injection [ppm]")
                for axis in axes[-1]:
                    axis.set_xlabel("wavelength [micron]")
                handles, labels = axes[0, 0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="outside upper center", ncol=3)
                fig.suptitle(
                    f"{scenario.name} | noiseless means labelled at {tier} ppm"
                )
                if not selected:
                    axes[1, 1].text(
                        0.5, 0.5, "No completed mean retrievals", ha="center"
                    )
                pdf.savefig(fig)
                plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    output = project / "diagnostics" / "spectral"
    plot_injections(project, output / "native_injection_comparisons.png")
    plot_retrieval_spectra(project, output / "retrieval_spectral_comparisons.pdf")
    print(output)


if __name__ == "__main__":
    main()
