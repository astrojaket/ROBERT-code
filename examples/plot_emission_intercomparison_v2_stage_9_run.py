#!/usr/bin/env python3
"""Plot saved spectral, TP, and posterior diagnostics for one Stage-9 run.

This is a read-only post-processing command. It consumes compact retrieval
products and evaluates only the analytic PG14 temperature parameterization; it
does not load opacities, evaluate a forward spectrum, or resume a retrieval.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_intercomparison_v2_stage_9_native import (  # noqa: E402
    atmospheric_state,
    load_common_contract,
)
MODEL_COLORS = {
    "robert": "#9370DB",  # mediumpurple
    "petitradtrans": "#DDA0DD",  # plum
    "picaso": "#36454F",  # charcoal
}
DATA_COLOR = "#202020"
DISPLAY_NAMES = {
    "robert": "ROBERT",
    "picaso": "PICASO",
    "petitradtrans": "petitRADTRANS",
}


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, sorted_values))


def _parameter_metadata(
    run: Mapping[str, Any],
    names: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    configured = run.get("parameters")
    if not isinstance(configured, list):
        raise RuntimeError("run config must provide a parameters list")
    by_name = {
        str(item["name"]): item
        for item in configured
        if isinstance(item, dict) and "name" in item
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(
            "run config has no plotting metadata for fitted parameters: "
            + ", ".join(missing)
        )
    return tuple(by_name[name] for name in names)


def _parameter_label(item: Mapping[str, Any]) -> str:
    label = str(item.get("label") or item["name"])
    unit = item.get("unit")
    return f"{label} [{unit}]" if unit else label


def _reference_value(item: Mapping[str, Any]) -> float | None:
    value = item.get("reference_value", item.get("truth"))
    return None if value is None else float(value)


def _posterior(
    run: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    Mapping[str, Any],
]:
    run_dir = Path(run["run_directory"])
    result_path = run_dir / "result.json"
    arrays_path = run_dir / "result_arrays.npz"
    spectra_path = run_dir / "diagnostic_spectra.npz"
    tp_path = run_dir / "diagnostic_tp.npz"
    chemistry_path = run_dir / "diagnostic_chemistry.npz"
    summary_path = run_dir / "posterior_summary.json"
    required = (
        result_path,
        arrays_path,
        spectra_path,
        tp_path,
        chemistry_path,
        summary_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "individual diagnostics require a completed production run; missing: "
            + ", ".join(missing)
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    names = tuple(str(name) for name in result["parameter_names"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        weights = (
            np.asarray(archive["weights"], dtype=float)
            if "weights" in archive.files
            else np.ones(samples.shape[0], dtype=float)
        )
    if samples.ndim != 2 or samples.shape[1] != len(names):
        raise RuntimeError("saved posterior dimensions do not match parameter names")
    if weights.shape != (samples.shape[0],):
        raise RuntimeError("saved posterior weights do not match samples")
    if (
        not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.sum(weights) <= 0.0
    ):
        raise RuntimeError("saved posterior samples or weights are invalid")
    weights = weights / np.sum(weights)
    return names, samples, weights, result


def _plot_spectrum(run: Mapping[str, Any], output: Path) -> None:
    path = Path(run["run_directory"]) / "diagnostic_spectra.npz"
    with np.load(path, allow_pickle=False) as archive:
        wavelength = np.asarray(archive["wavelength_micron"], dtype=float)
        observed = np.asarray(archive["observed_eclipse_depth"], dtype=float)
        uncertainty = np.asarray(
            archive["observational_uncertainty_eclipse_depth"],
            dtype=float,
        )
        best = np.asarray(archive["best_fit_eclipse_depth"], dtype=float)
        required = (
            "posterior_spectrum_q16_eclipse_depth",
            "posterior_spectrum_q84_eclipse_depth",
        )
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise RuntimeError(
                "true posterior spectral envelope is missing; run the Stage-9 "
                "posterior-envelope backfill for this retrieval"
            )
        spectrum_q16 = np.asarray(archive[required[0]], dtype=float)
        spectrum_q84 = np.asarray(archive[required[1]], dtype=float)
    sigma_ppm = float(run["noise_ppm"])
    injector = str(run["injector"])
    retriever = str(run["retriever"])
    injector_label = DISPLAY_NAMES.get(injector, injector)
    retriever_label = DISPLAY_NAMES.get(retriever, retriever)
    retriever_color = MODEL_COLORS[retriever]
    fig, (spectrum, residual) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
        constrained_layout=True,
    )
    data_artist = spectrum.errorbar(
        wavelength,
        observed * 1.0e6,
        yerr=uncertainty * 1.0e6,
        fmt="o",
        ms=2.2,
        color=DATA_COLOR,
        ecolor=DATA_COLOR,
        elinewidth=0.65,
        capsize=0.0,
        alpha=0.75,
        label=f"{sigma_ppm:g} ppm data generated with {injector_label}",
    )
    uncertainty_artist = spectrum.fill_between(
        wavelength,
        spectrum_q16 * 1.0e6,
        spectrum_q84 * 1.0e6,
        color=retriever_color,
        alpha=0.18,
        linewidth=0.0,
        label="central 68% spectral posterior",
    )
    (best_artist,) = spectrum.plot(
        wavelength,
        best * 1.0e6,
        color=retriever_color,
        lw=1.5,
        label=f"best-fitting spectrum from {retriever_label}",
    )
    spectrum.set_ylabel("eclipse depth [ppm]")
    spectrum.legend(
        handles=(data_artist, best_artist, uncertainty_artist),
        labels=(
            f"{sigma_ppm:g} ppm data generated with {injector_label}",
            f"best-fitting spectrum from {retriever_label}",
            "central 68% spectral posterior",
        ),
    )
    spectrum.grid(alpha=0.2)

    residual.axhspan(
        -sigma_ppm, sigma_ppm, color=DATA_COLOR, alpha=0.08
    )
    best_residual = (best - observed) * 1.0e6
    residual.fill_between(
        wavelength,
        (spectrum_q16 - observed) * 1.0e6,
        (spectrum_q84 - observed) * 1.0e6,
        color=retriever_color,
        alpha=0.18,
        linewidth=0.0,
        label="central 68% spectral posterior",
    )
    residual.plot(
        wavelength,
        best_residual,
        color=retriever_color,
        lw=1.35,
        label=f"{retriever_label} best fit − data",
    )
    residual.axhline(0.0, color="black", lw=0.7)
    residual.set(xlabel="wavelength [micron]", ylabel="residual [ppm]")
    residual.legend(fontsize=8, ncol=3)
    residual.grid(alpha=0.2)
    fig.suptitle(str(run["run_id"]))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_temperature_pressure(
    run: Mapping[str, Any],
    output: Path,
) -> None:
    with np.load(
        Path(run["run_directory"]) / "diagnostic_tp.npz",
        allow_pickle=False,
    ) as archive:
        pressure = np.asarray(archive["pressure_bar"], dtype=float)
        best_profile = np.asarray(archive["best_fit_temperature_k"], dtype=float)
        lower_profile = np.asarray(
            archive["posterior_temperature_q16_k"], dtype=float
        )
        upper_profile = np.asarray(
            archive["posterior_temperature_q84_k"], dtype=float
        )
    configured = run.get("parameters")
    references = (
        {
            str(item["name"]): _reference_value(item)
            for item in configured
            if isinstance(item, dict) and "name" in item
        }
        if isinstance(configured, list)
        else {}
    )
    truth_profile = None
    if references and all(value is not None for value in references.values()):
        common = load_common_contract(run["common_contract"])
        truth_profile = atmospheric_state(
            common,
            str(run["scenario"]),
            {name: float(value) for name, value in references.items()},
        ).temperature_cells_k
    retriever = str(run.get("retriever", "robert"))
    injector = str(run.get("injector", ""))
    retriever_color = MODEL_COLORS.get(retriever, "#9370DB")
    injector_color = MODEL_COLORS.get(injector, "#202020")

    fig, axis = plt.subplots(figsize=(7.2, 8.2), constrained_layout=True)
    axis.fill_betweenx(
        pressure,
        lower_profile,
        upper_profile,
        color=retriever_color,
        alpha=0.18,
        linewidth=0.0,
        label=f"{DISPLAY_NAMES.get(retriever, retriever)} central 68% TP posterior",
    )
    axis.plot(
        best_profile,
        pressure,
        color=retriever_color,
        lw=1.5,
        label=f"best-fitting {DISPLAY_NAMES.get(retriever, retriever)} TP",
    )
    if truth_profile is not None:
        axis.plot(
            truth_profile,
            pressure,
            color=injector_color,
            lw=1.4,
            ls="--",
            label=f"reference TP from {DISPLAY_NAMES.get(injector, injector)}",
        )
    axis.set_yscale("log")
    axis.invert_yaxis()
    axis.set(
        xlabel="temperature [K]",
        ylabel="pressure [bar]",
        title=(
            f"{run['run_id']}\nbest-fitting TP and 1σ envelope"
            + (" versus reference TP" if truth_profile is not None else "")
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _posterior_limits(
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    references: tuple[float | None, ...],
    best: NDArray[np.float64],
) -> list[tuple[float, float]]:
    limits = []
    for index, reference in enumerate(references):
        lower = min(
            _weighted_quantile(samples[:, index], weights, 0.0025),
            best[index],
        )
        upper = max(
            _weighted_quantile(samples[:, index], weights, 0.9975),
            best[index],
        )
        if reference is not None:
            lower = min(lower, reference)
            upper = max(upper, reference)
        span = upper - lower
        if span <= 0.0:
            span = max(abs(lower), 1.0) * 0.1
        limits.append((lower - 0.04 * span, upper + 0.04 * span))
    return limits


def _plot_corner(
    run: Mapping[str, Any],
    names: tuple[str, ...],
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    result: Mapping[str, Any],
    output: Path,
) -> None:
    metadata = _parameter_metadata(run, names)
    references = tuple(_reference_value(item) for item in metadata)
    best = np.asarray(
        [float(result["best_fit_parameters"][name]) for name in names], dtype=float
    )
    retriever = str(run.get("retriever", "robert"))
    injector = str(run.get("injector", ""))
    retriever_color = MODEL_COLORS.get(retriever, "#9370DB")
    injector_color = MODEL_COLORS.get(injector, "#202020")
    posterior_cmap = LinearSegmentedColormap.from_list(
        f"{retriever}_posterior",
        ("#FFFFFF", retriever_color),
    )
    limits = _posterior_limits(samples, weights, references, best)
    dimension = len(names)
    size = max(9.0, 1.65 * dimension)
    fig, axes = plt.subplots(
        dimension,
        dimension,
        figsize=(size, size),
        squeeze=False,
        constrained_layout=True,
    )
    for row in range(dimension):
        for column in range(dimension):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            if row == column:
                axis.hist(
                    samples[:, column],
                    bins=45,
                    range=limits[column],
                    weights=weights,
                    density=True,
                    color=retriever_color,
                    alpha=0.75,
                )
                if references[column] is not None:
                    axis.axvline(
                        references[column],
                        color=injector_color,
                        lw=0.9,
                        ls="--",
                    )
                axis.axvline(best[column], color=retriever_color, lw=1.1)
                axis.set_yticks([])
            else:
                axis.hist2d(
                    samples[:, column],
                    samples[:, row],
                    bins=36,
                    range=(limits[column], limits[row]),
                    weights=weights,
                    cmap=posterior_cmap,
                    cmin=np.finfo(float).tiny,
                )
                if references[column] is not None:
                    axis.axvline(
                        references[column],
                        color=injector_color,
                        lw=0.7,
                        ls="--",
                    )
                if references[row] is not None:
                    axis.axhline(
                        references[row],
                        color=injector_color,
                        lw=0.7,
                        ls="--",
                    )
                if references[column] is not None and references[row] is not None:
                    axis.plot(
                        references[column],
                        references[row],
                        marker="s",
                        ms=2.5,
                        color=injector_color,
                    )
                axis.axvline(best[column], color=retriever_color, lw=0.8)
                axis.axhline(best[row], color=retriever_color, lw=0.8)
                axis.plot(
                    best[column],
                    best[row],
                    marker="D",
                    ms=2.5,
                    color=retriever_color,
                )
            axis.set_xlim(limits[column])
            if row != column:
                axis.set_ylim(limits[row])
            axis.tick_params(labelsize=6)
            if row < dimension - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(_parameter_label(metadata[column]), fontsize=7)
                axis.tick_params(axis="x", labelrotation=45)
            if column > 0 or row == column:
                axis.set_yticklabels([])
            else:
                axis.set_ylabel(_parameter_label(metadata[row]), fontsize=7)
    handles = [
        Line2D(
            [0],
            [0],
            color=retriever_color,
            lw=1.3,
            marker="D",
            ms=3,
            label="best fit",
        ),
    ]
    if any(value is not None for value in references):
        handles.append(
            Line2D(
                [0],
                [0],
                color=injector_color,
                lw=1.0,
                ls="--",
                marker="s",
                ms=3,
                label="reference",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
    )
    fig.suptitle(f"{run['run_id']} posterior", fontsize=11)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_individual_run(
    run_config: Path,
    *,
    output: Path | None = None,
    max_tp_draws: int = 5000,
) -> Path:
    del max_tp_draws  # Retained for command-line compatibility.
    config_path = run_config.expanduser().resolve()
    run = json.loads(config_path.read_text(encoding="utf-8"))
    destination = (
        Path(run["run_directory"]) / "plots"
        if output is None
        else output.expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    names, samples, weights, result = _posterior(run)
    _plot_spectrum(run, destination / "spectrum_fit.png")
    _plot_temperature_pressure(
        run,
        destination / "temperature_pressure.png",
    )
    _plot_corner(
        run,
        names,
        samples,
        weights,
        result,
        destination / "posterior_corner.png",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-tp-draws",
        type=int,
        default=5000,
        help="deprecated compatibility option; saved exact TP quantiles are used",
    )
    args = parser.parse_args()
    output = plot_individual_run(
        args.run_config,
        output=args.output,
        max_tp_draws=args.max_tp_draws,
    )
    print(output)


if __name__ == "__main__":
    main()
