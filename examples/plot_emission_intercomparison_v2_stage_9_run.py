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
    parameter_definitions,
)


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
    summary_path = run_dir / "posterior_summary.json"
    required = (result_path, arrays_path, spectra_path, summary_path)
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


def _parameter_mapping(
    names: tuple[str, ...], values: NDArray[np.float64]
) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(names)}


def _posterior_median(
    names: tuple[str, ...],
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> dict[str, float]:
    return {
        name: _weighted_quantile(samples[:, index], weights, 0.5)
        for index, name in enumerate(names)
    }


def _systematic_posterior_indices(
    weights: NDArray[np.float64], maximum: int
) -> NDArray[np.int64]:
    count = min(int(maximum), weights.size)
    positions = (np.arange(count, dtype=float) + 0.5) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left")


def _plot_spectrum(run: Mapping[str, Any], output: Path) -> None:
    path = Path(run["run_directory"]) / "diagnostic_spectra.npz"
    with np.load(path, allow_pickle=False) as archive:
        wavelength = np.asarray(archive["wavelength_micron"], dtype=float)
        injection = np.asarray(archive["injection_eclipse_depth"], dtype=float)
        best = np.asarray(archive["best_fit_eclipse_depth"], dtype=float)
        median = np.asarray(
            archive["posterior_median_eclipse_depth"], dtype=float
        )
    sigma_ppm = float(run["noise_ppm"])
    fig, (spectrum, residual) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
        constrained_layout=True,
    )
    spectrum.plot(
        wavelength,
        injection * 1.0e6,
        color="black",
        lw=1.2,
        label="injection / noiseless data",
    )
    spectrum.plot(
        wavelength, median * 1.0e6, color="#0072B2", lw=1.3, label="posterior median"
    )
    spectrum.plot(
        wavelength, best * 1.0e6, color="#D55E00", lw=1.0, label="best fit"
    )
    spectrum.set_ylabel("eclipse depth [ppm]")
    spectrum.legend()
    spectrum.grid(alpha=0.2)

    residual.axhspan(
        -sigma_ppm,
        sigma_ppm,
        color="0.85",
        alpha=0.7,
        label=f"±{sigma_ppm:g} ppm likelihood σ",
    )
    residual.plot(
        wavelength,
        (median - injection) * 1.0e6,
        color="#0072B2",
        lw=1.2,
        label="median − injection",
    )
    residual.plot(
        wavelength,
        (best - injection) * 1.0e6,
        color="#D55E00",
        lw=1.0,
        label="best − injection",
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
    names: tuple[str, ...],
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    result: Mapping[str, Any],
    output: Path,
    *,
    max_draws: int,
) -> None:
    common = load_common_contract(run["common_contract"])
    pressure = np.asarray(
        next(
            item["centers_bar"]
            for item in common["pressure_grids"]
            if item["n_cells"] == 80
        ),
        dtype=float,
    )
    definitions = parameter_definitions(run["scenario"])
    truth = {item.name: item.truth for item in definitions}
    median = _posterior_median(names, samples, weights)
    best = {
        str(name): float(value)
        for name, value in result["best_fit_parameters"].items()
    }
    indices = _systematic_posterior_indices(weights, max_draws)
    profiles = np.asarray(
        [
            atmospheric_state(
                common,
                str(run["scenario"]),
                _parameter_mapping(names, samples[index]),
            ).temperature_cells_k
            for index in indices
        ],
        dtype=float,
    )
    q025, q16, q50, q84, q975 = np.quantile(
        profiles, (0.025, 0.16, 0.5, 0.84, 0.975), axis=0
    )
    truth_profile = atmospheric_state(
        common, str(run["scenario"]), truth
    ).temperature_cells_k
    median_profile = atmospheric_state(
        common, str(run["scenario"]), median
    ).temperature_cells_k
    best_profile = atmospheric_state(
        common, str(run["scenario"]), best
    ).temperature_cells_k

    fig, axis = plt.subplots(figsize=(7.2, 8.2), constrained_layout=True)
    axis.fill_betweenx(
        pressure, q025, q975, color="#56B4E9", alpha=0.18, label="central 95%"
    )
    axis.fill_betweenx(
        pressure, q16, q84, color="#56B4E9", alpha=0.35, label="central 68%"
    )
    axis.plot(q50, pressure, color="#0072B2", lw=1.5, label="profile median")
    axis.plot(
        median_profile,
        pressure,
        color="#0072B2",
        lw=1.0,
        ls=":",
        label="median-parameter profile",
    )
    axis.plot(best_profile, pressure, color="#D55E00", lw=1.1, label="best fit")
    axis.plot(truth_profile, pressure, color="black", lw=1.2, ls="--", label="truth")
    axis.set_yscale("log")
    axis.invert_yaxis()
    axis.set(
        xlabel="temperature [K]",
        ylabel="pressure [bar]",
        title=f"{run['run_id']}\nTP posterior from {indices.size} deterministic weighted draws",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _posterior_limits(
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    truths: NDArray[np.float64],
) -> list[tuple[float, float]]:
    limits = []
    for index, truth in enumerate(truths):
        lower = min(_weighted_quantile(samples[:, index], weights, 0.0025), truth)
        upper = max(_weighted_quantile(samples[:, index], weights, 0.9975), truth)
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
    output: Path,
) -> None:
    definitions = parameter_definitions(run["scenario"])
    if tuple(item.name for item in definitions) != names:
        raise RuntimeError("posterior parameter order differs from frozen Stage-9 order")
    truths = np.asarray([item.truth for item in definitions], dtype=float)
    limits = _posterior_limits(samples, weights, truths)
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
                    color="#56B4E9",
                    alpha=0.75,
                )
                axis.axvline(truths[column], color="black", lw=0.9, ls="--")
                axis.set_yticks([])
            else:
                axis.hist2d(
                    samples[:, column],
                    samples[:, row],
                    bins=36,
                    range=(limits[column], limits[row]),
                    weights=weights,
                    cmap="Blues",
                    cmin=np.finfo(float).tiny,
                )
                axis.axvline(truths[column], color="black", lw=0.7, ls="--")
                axis.axhline(truths[row], color="black", lw=0.7, ls="--")
                axis.plot(
                    truths[column], truths[row], marker="s", ms=2.5, color="black"
                )
            axis.set_xlim(limits[column])
            if row != column:
                axis.set_ylim(limits[row])
            axis.tick_params(labelsize=6)
            if row < dimension - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(definitions[column].label, fontsize=7)
                axis.tick_params(axis="x", labelrotation=45)
            if column > 0 or row == column:
                axis.set_yticklabels([])
            else:
                axis.set_ylabel(definitions[row].label, fontsize=7)
    fig.suptitle(f"{run['run_id']} posterior; dashed lines show truth", fontsize=11)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_individual_run(
    run_config: Path,
    *,
    output: Path | None = None,
    max_tp_draws: int = 5_000,
) -> Path:
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
        names,
        samples,
        weights,
        result,
        destination / "temperature_pressure.png",
        max_draws=max_tp_draws,
    )
    _plot_corner(run, names, samples, weights, destination / "posterior_corner.png")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tp-draws", type=int, default=5_000)
    args = parser.parse_args()
    if args.max_tp_draws < 1:
        parser.error("--max-tp-draws must be positive")
    output = plot_individual_run(
        args.run_config,
        output=args.output,
        max_tp_draws=args.max_tp_draws,
    )
    print(output)


if __name__ == "__main__":
    main()
