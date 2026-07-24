#!/usr/bin/env python3
"""Create Stage-9 spectrum, input-TP, and molecular-posterior comparison pages.

Each page fixes one scenario, uncertainty tier, and injection framework. It
compares the two directed retrieval frameworks using only compact production
products and the analytic PG14 TP mapping. No opacity or forward-spectrum
calculation is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_intercomparison_v2_stage_9_native import (  # noqa: E402
    atmospheric_state,
    load_common_contract,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    FRAMEWORKS,
    NOISE_TIERS_PPM,
    SCENARIOS,
    parameter_definitions,
)


DISPLAY_NAMES = {
    "robert": "ROBERT",
    "picaso": "PICASO",
    "petitradtrans": "petitRADTRANS",
}
MODEL_COLORS = {
    "robert": "#9370DB",
    "picaso": "#009E73",
    "petitradtrans": "#E69F00",
}
DATA_COLOR = "#202020"
MOLECULES = ("H2O", "CO", "CO2", "CH4")


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    order = np.argsort(values)
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, ordered))


def _completed_runs(project: Path) -> list[dict[str, Any]]:
    rows = json.loads((project / "run_index.json").read_text(encoding="utf-8"))
    completed = []
    for row in rows:
        run = json.loads((project / row["run_config"]).read_text(encoding="utf-8"))
        run_dir = Path(run["run_directory"])
        products = {
            "result": run_dir / "result.json",
            "arrays": run_dir / "result_arrays.npz",
            "spectra": run_dir / "diagnostic_spectra.npz",
            "summary": run_dir / "posterior_summary.json",
        }
        if all(path.is_file() and path.stat().st_size > 0 for path in products.values()):
            run.update(products)
            completed.append(run)
    return completed


def _posterior(
    run: Mapping[str, Any],
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
    result = json.loads(Path(run["result"]).read_text(encoding="utf-8"))
    names = tuple(str(name) for name in result["parameter_names"])
    with np.load(run["arrays"], allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        weights = (
            np.asarray(archive["weights"], dtype=float)
            if "weights" in archive.files
            else np.ones(samples.shape[0], dtype=float)
        )
    weights /= np.sum(weights)
    return names, samples, weights


def _plot_page(
    runs: list[dict[str, Any]],
    *,
    common: Mapping[str, Any],
    scenario: str,
    tier: int,
    injector: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(17.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    spectrum_axis = fig.add_subplot(grid[0, 0])
    tp_axis = fig.add_subplot(grid[1, 0])
    molecule_axes = (
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    )
    selected = sorted(
        (
            run
            for run in runs
            if run["scenario"] == scenario
            and int(run["noise_ppm"]) == tier
            and run["injector"] == injector
        ),
        key=lambda run: FRAMEWORKS.index(run["retriever"]),
    )
    if not selected:
        for axis in (spectrum_axis, tp_axis, *molecule_axes):
            axis.text(
                0.5,
                0.5,
                "No completed retrieval",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        return fig

    with np.load(selected[0]["spectra"], allow_pickle=False) as archive:
        wavelength = np.asarray(archive["wavelength_micron"], dtype=float)
        injection_spectrum = np.asarray(
            archive["injection_eclipse_depth"], dtype=float
        )
    spectrum_axis.errorbar(
        wavelength,
        injection_spectrum * 1.0e6,
        yerr=np.full(wavelength.size, float(tier)),
        fmt="o",
        ms=1.9,
        color=DATA_COLOR,
        ecolor=DATA_COLOR,
        elinewidth=0.55,
        capsize=0.0,
        alpha=0.72,
        label=f"{tier} ppm data generated with {DISPLAY_NAMES[injector]}",
        zorder=4,
    )
    for run in selected:
        retriever = str(run["retriever"])
        color = MODEL_COLORS[retriever]
        with np.load(run["spectra"], allow_pickle=False) as archive:
            best = np.asarray(archive["best_fit_eclipse_depth"], dtype=float)
        spectrum_axis.fill_between(
            wavelength,
            best * 1.0e6 - tier,
            best * 1.0e6 + tier,
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )
        spectrum_axis.plot(
            wavelength,
            best * 1.0e6,
            color=color,
            lw=1.45,
            label=f"best-fitting spectrum from {DISPLAY_NAMES[retriever]}",
        )
    spectrum_axis.set(
        xlabel="wavelength [micron]",
        ylabel="eclipse depth [ppm]",
        title="Spectrum",
    )
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.legend(fontsize=8)

    pressure = np.asarray(
        next(
            item["centers_bar"]
            for item in common["pressure_grids"]
            if item["n_cells"] == 80
        ),
        dtype=float,
    )
    definitions = parameter_definitions(scenario)
    truth = {item.name: item.truth for item in definitions}
    input_tp = atmospheric_state(common, scenario, truth).temperature_cells_k
    tp_axis.plot(
        input_tp,
        pressure,
        color=DATA_COLOR,
        lw=1.5,
        ls="--",
        label=f"input TP from {DISPLAY_NAMES[injector]}",
    )
    posterior_cache: dict[
        str, tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]
    ] = {}
    for run in selected:
        retriever = str(run["retriever"])
        color = MODEL_COLORS[retriever]
        result = json.loads(Path(run["result"]).read_text(encoding="utf-8"))
        best_parameters = {
            str(name): float(value)
            for name, value in result["best_fit_parameters"].items()
        }
        best_tp = atmospheric_state(
            common, scenario, best_parameters
        ).temperature_cells_k
        tp_axis.plot(
            best_tp,
            pressure,
            color=color,
            lw=1.5,
            label=f"best-fitting {DISPLAY_NAMES[retriever]} TP",
        )
        posterior_cache[retriever] = _posterior(run)
    tp_axis.set_yscale("log")
    tp_axis.invert_yaxis()
    tp_axis.set(
        xlabel="temperature [K]",
        ylabel="pressure [bar]",
        title="Best-fitting TP profiles versus input",
    )
    tp_axis.grid(alpha=0.2)
    tp_axis.legend(fontsize=8)

    definition_by_name = {item.name: item for item in definitions}
    for molecule, axis in zip(MOLECULES, molecule_axes, strict=True):
        parameter_name = f"log10_vmr_{molecule}"
        parameter = definition_by_name[parameter_name]
        lower = parameter.truth
        upper = parameter.truth
        for names, samples, weights in posterior_cache.values():
            index = names.index(parameter_name)
            lower = min(
                lower,
                _weighted_quantile(samples[:, index], weights, 0.0025),
            )
            upper = max(
                upper,
                _weighted_quantile(samples[:, index], weights, 0.9975),
            )
        span = max(upper - lower, 1.0e-6)
        edges = np.linspace(lower - 0.05 * span, upper + 0.05 * span, 70)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for retriever, (names, samples, weights) in posterior_cache.items():
            index = names.index(parameter_name)
            density, _ = np.histogram(
                samples[:, index], bins=edges, weights=weights, density=True
            )
            color = MODEL_COLORS[retriever]
            axis.fill_between(centers, density, color=color, alpha=0.14)
            axis.plot(
                centers,
                density,
                color=color,
                lw=1.5,
                label=DISPLAY_NAMES[retriever],
            )
        axis.axvline(
            parameter.truth,
            color=DATA_COLOR,
            lw=1.1,
            ls="--",
            label="input",
        )
        axis.set(
            xlabel=f"log10 VMR({molecule})",
            ylabel="posterior density",
            title=f"{molecule} posterior",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    fig.suptitle(
        f"{scenario} | {tier} ppm data generated with {DISPLAY_NAMES[injector]}",
        fontsize=14,
    )
    return fig


def plot_big_comparison(
    project: Path,
    *,
    scenario_filter: str | None = None,
    tier_filter: int | None = None,
    injector_filter: str | None = None,
) -> tuple[Path, int]:
    project = project.expanduser().resolve()
    runs = _completed_runs(project)
    common = load_common_contract(project / "contracts" / "common_contract.json")
    output = project / "diagnostics" / "big_comparison"
    output.mkdir(parents=True, exist_ok=True)
    stem = scenario_filter or "all_scenarios"
    pdf_path = output / f"{stem}_big_comparison.pdf"
    pages = 0
    with PdfPages(pdf_path) as pdf:
        for scenario_item in SCENARIOS:
            scenario = scenario_item.name
            if scenario_filter is not None and scenario != scenario_filter:
                continue
            for tier in NOISE_TIERS_PPM:
                if tier_filter is not None and tier != tier_filter:
                    continue
                for injector in FRAMEWORKS:
                    if injector_filter is not None and injector != injector_filter:
                        continue
                    fig = _plot_page(
                        runs,
                        common=common,
                        scenario=scenario,
                        tier=tier,
                        injector=injector,
                    )
                    pdf.savefig(fig)
                    png_dir = output / scenario / f"{tier:03d}ppm"
                    png_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        png_dir / f"data_generated_with_{injector}.png", dpi=180
                    )
                    plt.close(fig)
                    pages += 1
    return pdf_path, pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--scenario",
        choices=tuple(item.name for item in SCENARIOS),
    )
    parser.add_argument("--noise-ppm", type=int, choices=NOISE_TIERS_PPM)
    parser.add_argument("--injector", choices=FRAMEWORKS)
    args = parser.parse_args()
    output, pages = plot_big_comparison(
        args.project_root,
        scenario_filter=args.scenario,
        tier_filter=args.noise_ppm,
        injector_filter=args.injector,
    )
    print(f"{output} ({pages} pages)")


if __name__ == "__main__":
    main()
