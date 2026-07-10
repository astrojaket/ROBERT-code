"""Plot ROBERT and NemesisPy HAT-P-32b FastChem/Madhu retrieval products."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    build_parameterized_clear_sky_emission_model,
    load_emission_observation_npz,
)

if __package__:
    from .hat_p_32b_fastchem_config import (
        OBSERVATION_NPZ,
        OPACITY_SPECIES,
        REFERENCE_POSTERIOR_NPZ,
        RESULTS_DIR,
        make_model_config,
        reference_map_parameters,
        retrieval_parameters,
    )
else:
    from hat_p_32b_fastchem_config import (
        OBSERVATION_NPZ,
        OPACITY_SPECIES,
        REFERENCE_POSTERIOR_NPZ,
        RESULTS_DIR,
        make_model_config,
        reference_map_parameters,
        retrieval_parameters,
    )

DEFAULT_ROBERT_RESULT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "hat_p_32b_fastchem_comparison"
    / "optimal_estimation"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "hat_p_32b_fastchem_comparison"
    / "plots"
)

NEMESIS_COLOR = "#3b6fb6"
ROBERT_COLOR = "#d75f00"
DATA_COLOR = "#202124"
REFERENCE_COLOR = "#7a5195"


def main() -> dict[str, object]:
    args = _parser().parse_args()
    result_dir = Path(args.robert_result_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    result_summary, result_arrays = _load_robert_result(result_dir)
    parameters = {str(key): float(value) for key, value in result_summary["best_fit_parameters"].items()}
    observation = load_emission_observation_npz(
        OBSERVATION_NPZ,
        instrument="JWST/NIRSpec G395H",
    )
    model = build_parameterized_clear_sky_emission_model(
        make_model_config(
            pressure_top_bar=args.pressure_top_bar,
            pressure_bottom_bar=args.pressure_bottom_bar,
            n_layers=args.layers,
        ),
        spectral_grid=observation.spectral_grid,
    )
    robert_spectrum = model(parameters)
    reference_parameters = reference_map_parameters()
    robert_at_nemesis_map = model(reference_parameters)
    robert_atmosphere = model.atmosphere_builder.build(parameters)
    reference_atmosphere = model.atmosphere_builder.build(reference_parameters)

    with np.load(OBSERVATION_NPZ, allow_pickle=False) as archive:
        nemesis_map_spectrum = np.asarray(archive["MAP"], dtype=float)
    _plot_spectrum(
        output_dir / "spectrum-residual-comparison.png",
        observation,
        nemesis_map_spectrum,
        robert_at_nemesis_map.values,
        robert_spectrum.values,
    )
    _plot_temperature(
        output_dir / "temperature-profile-comparison.png",
        robert_atmosphere.pressure_grid.centers,
        robert_atmosphere.temperature,
        reference_atmosphere.temperature,
    )
    _plot_composition(
        output_dir / "vmr-profile-comparison.png",
        robert_atmosphere.pressure_grid.centers,
        robert_atmosphere.composition,
    )
    _plot_parameters(
        output_dir / "parameter-comparison.png",
        result_summary,
        result_arrays,
    )

    residual = observation.flux - robert_spectrum.values
    comparison = nemesis_map_spectrum - robert_spectrum.values
    metrics = {
        "robert_result_dir": str(result_dir),
        "robert_method": result_summary["method"],
        "robert_converged": bool(result_summary["converged"]),
        "robert_best_fit_log_likelihood": float(result_summary["best_fit_log_likelihood"]),
        "robert_likelihood_calls": _optional_int(result_summary["metadata"].get("ncall")),
        "robert_best_fit_chi_square": float(
            np.sum(np.square(residual / observation.uncertainty))
        ),
        "robert_vs_nemesis_map_rms_ppm": float(
            np.sqrt(np.mean(np.square(comparison))) * 1.0e6
        ),
        "pressure_top_bar": args.pressure_top_bar,
        "pressure_bottom_bar": args.pressure_bottom_bar,
        "n_layers": args.layers,
        "plots": sorted(path.name for path in output_dir.glob("*.png")),
    }
    metrics_path = output_dir / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def _plot_spectrum(path, observation, nemesis_map, robert_reference, robert_best) -> None:
    wavelength = observation.wavelength
    fig, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
        constrained_layout=True,
    )
    axis.errorbar(
        wavelength,
        observation.flux * 1.0e6,
        yerr=observation.uncertainty * 1.0e6,
        fmt=".",
        color=DATA_COLOR,
        ecolor="#9a9a9a",
        markersize=3.5,
        linewidth=0.7,
        label="G395H data",
    )
    axis.plot(wavelength, nemesis_map * 1.0e6, color=NEMESIS_COLOR, lw=1.8, label="NemesisPy MAP")
    axis.plot(
        wavelength,
        robert_reference * 1.0e6,
        color=REFERENCE_COLOR,
        lw=1.2,
        ls="--",
        label="ROBERT at NemesisPy MAP parameters",
    )
    axis.plot(wavelength, robert_best * 1.0e6, color=ROBERT_COLOR, lw=1.8, label="ROBERT best fit")
    axis.set_ylabel("Eclipse depth [ppm]")
    axis.legend(frameon=False, fontsize=9, ncol=2)
    axis.grid(alpha=0.2)

    residual_axis.axhline(0.0, color="#777777", lw=0.8)
    residual_axis.plot(
        wavelength,
        (robert_reference - nemesis_map) * 1.0e6,
        color=REFERENCE_COLOR,
        lw=1.2,
        ls="--",
        label="ROBERT(ref) − NemesisPy",
    )
    residual_axis.plot(
        wavelength,
        (robert_best - nemesis_map) * 1.0e6,
        color=ROBERT_COLOR,
        lw=1.5,
        label="ROBERT(best) − NemesisPy",
    )
    residual_axis.set_xlabel("Wavelength [µm]")
    residual_axis.set_ylabel("Difference [ppm]")
    residual_axis.grid(alpha=0.2)
    residual_axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_temperature(path: Path, pressure_bar, robert_temperature, robert_reference) -> None:
    source = RESULTS_DIR / "quench_study_emission_TP_band.npz"
    with np.load(source, allow_pickle=False) as archive:
        pressure = np.asarray(archive["pressure_bar"], dtype=float)
        nemesis_map = np.asarray(archive["T_map"], dtype=float)
        lower = np.asarray(archive["T_1lo"], dtype=float)
        upper = np.asarray(archive["T_1hi"], dtype=float)
    fig, axis = plt.subplots(figsize=(6.2, 6.4), constrained_layout=True)
    axis.fill_betweenx(pressure, lower, upper, color=NEMESIS_COLOR, alpha=0.18, label="NemesisPy 68%")
    axis.plot(nemesis_map, pressure, color=NEMESIS_COLOR, lw=1.8, label="NemesisPy MAP")
    axis.plot(
        robert_reference,
        pressure_bar,
        color=REFERENCE_COLOR,
        lw=1.2,
        ls="--",
        label="ROBERT at NemesisPy MAP",
    )
    axis.plot(robert_temperature, pressure_bar, color=ROBERT_COLOR, lw=1.8, label="ROBERT best fit")
    axis.set_yscale("log")
    axis.set_ylim(np.max(pressure), np.min(pressure))
    axis.set_xlabel("Temperature [K]")
    axis.set_ylabel("Pressure [bar]")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_composition(path: Path, pressure_bar, composition) -> None:
    source = RESULTS_DIR / "quench_study_emission_TP_VMR_band.npz"
    with np.load(source, allow_pickle=False) as archive:
        nemesis_pressure = np.asarray(archive["pressure_bar"], dtype=float)
        gas_names = tuple(str(name) for name in archive["gas_names"])
        vmr_map = np.asarray(archive["VMR_map"], dtype=float)
        vmr_lower = np.asarray(archive["VMR_1lo"], dtype=float)
        vmr_upper = np.asarray(archive["VMR_1hi"], dtype=float)
    selected = tuple(species for species in OPACITY_SPECIES if species in gas_names)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.4), sharey=True, constrained_layout=True)
    for axis, species in zip(axes.flat, selected, strict=True):
        index = gas_names.index(species)
        axis.fill_betweenx(
            nemesis_pressure,
            vmr_lower[:, index],
            vmr_upper[:, index],
            color=NEMESIS_COLOR,
            alpha=0.18,
        )
        axis.plot(vmr_map[:, index], nemesis_pressure, color=NEMESIS_COLOR, lw=1.5, label="NemesisPy MAP")
        axis.plot(composition[species], pressure_bar, color=ROBERT_COLOR, lw=1.6, label="ROBERT best fit")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(np.max(nemesis_pressure), np.min(nemesis_pressure))
        axis.set_title(species)
        axis.set_xlabel("VMR")
        axis.grid(alpha=0.18, which="both")
    axes[0, 0].set_ylabel("Pressure [bar]")
    axes[1, 0].set_ylabel("Pressure [bar]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_parameters(path: Path, result_summary, result_arrays) -> None:
    with np.load(REFERENCE_POSTERIOR_NPZ, allow_pickle=False) as archive:
        nemesis_names = tuple(str(name) for name in archive["names"])
        nemesis_samples = np.asarray(archive["samples_raw"], dtype=float)
        nemesis_map = np.asarray(archive["truths_map_raw"], dtype=float)
    names = retrieval_parameters().names
    nemesis_quantiles = np.quantile(nemesis_samples, [0.16, 0.5, 0.84], axis=0)
    robert_center, robert_lower, robert_upper = _robert_parameter_intervals(
        names,
        result_summary,
        result_arrays,
    )
    robert_label = "ROBERT posterior"
    if result_summary["method"] == "ultranest" and not result_summary["converged"]:
        robert_label += " (not converged)"
    fig, axes = plt.subplots(4, 2, figsize=(9.4, 10.2), constrained_layout=True)
    for axis, name, center, lower, upper in zip(
        axes.flat,
        names,
        robert_center,
        robert_lower,
        robert_upper,
        strict=True,
    ):
        reference_index = nemesis_names.index(name)
        nemesis_median = nemesis_quantiles[1, reference_index]
        axis.errorbar(
            [0.0],
            [nemesis_median],
            yerr=[[nemesis_median - nemesis_quantiles[0, reference_index]], [nemesis_quantiles[2, reference_index] - nemesis_median]],
            fmt="o",
            color=NEMESIS_COLOR,
            capsize=3,
            label="NemesisPy posterior",
        )
        axis.scatter([0.0], [nemesis_map[reference_index]], marker="x", color=DATA_COLOR, zorder=3)
        axis.errorbar(
            [1.0],
            [center],
            yerr=[[center - lower], [upper - center]],
            fmt="o",
            color=ROBERT_COLOR,
            capsize=3,
            label=robert_label,
        )
        axis.set_xticks([0.0, 1.0], ["NemesisPy", "ROBERT"])
        axis.set_title(name)
        axis.grid(alpha=0.18, axis="y")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _robert_parameter_intervals(names, summary, arrays):
    center = np.array([summary["best_fit_parameters"][name] for name in names], dtype=float)
    if summary["method"] == "ultranest" and "samples" in arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        weights = np.asarray(arrays.get("weights", np.ones(samples.shape[0])), dtype=float)
        quantiles = np.vstack(
            [_weighted_quantile(samples[:, index], weights, (0.16, 0.5, 0.84)) for index in range(len(names))]
        ).T
        return quantiles[1], quantiles[0], quantiles[2]
    covariance = np.asarray(arrays["covariance"], dtype=float)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return center, center - sigma, center + sigma


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.sum(sorted_weights)
    return np.interp(quantiles, cumulative, sorted_values)


def _optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def _load_robert_result(result_dir: Path):
    summary_path = result_dir / "result.json"
    arrays_path = result_dir / "result_arrays.npz"
    if not summary_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"ROBERT result files are missing under {result_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    return summary, arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robert-result-dir", default=str(DEFAULT_ROBERT_RESULT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pressure-top-bar", type=float, default=1.0e-6)
    parser.add_argument("--pressure-bottom-bar", type=float, default=100.0)
    parser.add_argument("--layers", type=int, default=100)
    return parser


if __name__ == "__main__":
    main()
