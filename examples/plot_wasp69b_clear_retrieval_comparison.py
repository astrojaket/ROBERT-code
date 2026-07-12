"""Plot the exploratory ROBERT WASP-69b clear retrieval and paper comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import retrieve_wasp69b_clear_native_modes as retrieval


OUTPUT = retrieval.OUTPUT
PARAMETER_LABELS = (
    r"$\log_{10}(Z/Z_\odot)$",
    "C/O",
    r"$\kappa_{\rm IR}$",
    r"$\gamma_1$",
    r"$\gamma_2$",
    r"$T_{\rm irr}$",
    r"$\alpha$",
)
COLORS = {
    "f322w2": "#20639b",
    "f444w": "#ef5675",
    "lrs": "#2ca25f",
}
PAPER = {
    "citation": "Schlawin et al. (2024), AJ 168, 104",
    "doi": "10.3847/1538-3881/ad58e0",
    "model": "CHIMERA homogeneous one-region clear, full 2-12 micron spectrum",
    "metallicity_16_50_84": [1.28, 1.30, 1.32],
    "CtoO_16_50_84": [0.10, 0.11, 0.13],
    "reduced_chi_squared": 14.5,
    "log_evidence": -588.0,
}


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] = (0.16, 0.5, 0.84),
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, sorted_values)


def _profile_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            _weighted_quantile(values[:, index], weights)
            for index in range(values.shape[1])
        ]
    )


def _load(output: Path):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    products = np.load(output / "posterior_products.npz")
    samples = np.asarray(products["samples"], dtype=float)
    weights = np.asarray(products["weights"], dtype=float)
    weights /= weights.sum()
    return summary, products, samples, weights


def _plot_spectrum(problem, products, summary, output: Path) -> None:
    best = problem.model_spectra(summary["best_fit"])
    figure, (spectrum_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    for dataset in problem.observations.datasets:
        name = dataset.name
        observation = dataset.observation
        color = COLORS[name]
        q16 = products[f"{name}_q16"] * 1.0e6
        q50 = products[f"{name}_q50"] * 1.0e6
        q84 = products[f"{name}_q84"] * 1.0e6
        spectrum_axis.fill_between(
            observation.wavelength, q16, q84, color=color, alpha=0.2
        )
        spectrum_axis.plot(observation.wavelength, q50, color=color, linewidth=1.4)
        spectrum_axis.errorbar(
            observation.wavelength,
            observation.flux * 1.0e6,
            yerr=observation.uncertainty * 1.0e6,
            fmt=".",
            color=color,
            markersize=3,
            alpha=0.75,
            label=observation.instrument,
        )
        residual = (observation.flux - best[name].values) / observation.uncertainty
        residual_axis.plot(
            observation.wavelength, residual, ".", color=color, markersize=3
        )
    spectrum_axis.set_ylabel("Eclipse depth (ppm)")
    spectrum_axis.set_title("WASP-69b clear one-region fit (exploratory early stop)")
    spectrum_axis.legend(fontsize=8, ncol=3)
    spectrum_axis.text(
        0.02,
        0.96,
        f"ROBERT: $\\chi^2_\\nu$={summary['reduced_chi_squared']:.2f} (274 bins)\n"
        f"Schlawin+ clear: $\\chi^2_\\nu$={PAPER['reduced_chi_squared']:.1f} (280 bins)",
        transform=spectrum_axis.transAxes,
        va="top",
        fontsize=9,
    )
    residual_axis.axhline(0.0, color="0.2", linewidth=0.8)
    residual_axis.axhline(1.0, color="0.6", linewidth=0.6, linestyle="--")
    residual_axis.axhline(-1.0, color="0.6", linewidth=0.6, linestyle="--")
    residual_axis.set_ylabel(r"Residual / $\sigma$")
    residual_axis.set_xlabel("Wavelength (micron)")
    figure.tight_layout()
    figure.savefig(output / "comparison_spectrum_residuals.png", dpi=200)
    plt.close(figure)


def _plot_corner(samples: np.ndarray, weights: np.ndarray, output: Path) -> None:
    figure = corner.corner(
        samples,
        weights=weights,
        labels=PARAMETER_LABELS,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".3g",
        smooth=1.0,
        smooth1d=1.0,
        plot_contours=False,
        color="#20639b",
    )
    figure.suptitle(
        "WASP-69b ROBERT posterior — exploratory; weighted ESS = 2.45",
        y=1.01,
        fontsize=13,
    )
    figure.savefig(output / "comparison_corner.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_composition_and_temperature(
    problem,
    samples: np.ndarray,
    weights: np.ndarray,
    output: Path,
) -> None:
    builder = next(iter(problem.forward_model.models.values())).atmosphere_builder
    temperatures = []
    composition: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        state = builder.build(problem.parameter_mapping(sample))
        temperatures.append(state.temperature)
        for species, profile in state.composition.items():
            if species in retrieval.workflow.SPECIES:
                composition.setdefault(species, []).append(profile)
    pressure = builder.pressure_grid.centers
    temperature_q = _profile_quantiles(np.asarray(temperatures), weights)

    figure, (temperature_axis, chemistry_axis) = plt.subplots(1, 2, figsize=(12, 6))
    temperature_axis.fill_betweenx(
        pressure,
        temperature_q[0],
        temperature_q[2],
        color="#20639b",
        alpha=0.22,
        label="16–84%",
    )
    temperature_axis.plot(
        temperature_q[1], pressure, color="#20639b", linewidth=1.8, label="median"
    )
    temperature_axis.set_xscale("linear")
    temperature_axis.set_yscale("log")
    temperature_axis.invert_yaxis()
    temperature_axis.set_xlabel("Temperature (K)")
    temperature_axis.set_ylabel("Pressure (bar)")
    temperature_axis.set_title("Temperature–pressure profile")
    temperature_axis.legend(fontsize=8)

    species_colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(composition)))
    for color, (species, profiles) in zip(species_colors, composition.items()):
        quantiles = _profile_quantiles(np.asarray(profiles), weights)
        chemistry_axis.fill_betweenx(
            pressure, quantiles[0], quantiles[2], color=color, alpha=0.12
        )
        chemistry_axis.plot(
            quantiles[1], pressure, color=color, linewidth=1.3, label=species
        )
    chemistry_axis.set_xscale("log")
    chemistry_axis.set_yscale("log")
    chemistry_axis.invert_yaxis()
    chemistry_axis.set_xlabel("Volume mixing ratio")
    chemistry_axis.set_ylabel("Pressure (bar)")
    chemistry_axis.set_title("Equilibrium chemistry profiles")
    chemistry_axis.legend(fontsize=8, ncol=2)
    figure.suptitle("Exploratory weighted profiles (ESS = 2.45)")
    figure.tight_layout()
    figure.savefig(output / "comparison_atmosphere_profiles.png", dpi=200)
    plt.close(figure)


def _plot_paper_parameters(
    samples: np.ndarray,
    weights: np.ndarray,
    summary: dict[str, object],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, index, key, label in (
        (axes[0], 0, "metallicity_16_50_84", r"$\log_{10}(Z/Z_\odot)$"),
        (axes[1], 1, "CtoO_16_50_84", "C/O"),
    ):
        axis.hist(
            samples[:, index],
            bins=35,
            weights=weights,
            density=True,
            color="#20639b",
            alpha=0.55,
            label="ROBERT",
        )
        paper_q = PAPER[key]
        axis.axvspan(paper_q[0], paper_q[2], color="#ef5675", alpha=0.2)
        axis.axvline(paper_q[1], color="#ef5675", linewidth=1.8, label="Schlawin+")
        robert_q = summary["posterior_16_50_84"][
            "metallicity" if index == 0 else "CtoO"
        ]
        axis.axvline(robert_q[1], color="#20639b", linewidth=1.8)
        axis.set_xlabel(label)
        axis.set_ylabel("Weighted density")
        axis.legend(fontsize=8)
    axes[2].bar(
        ("ROBERT\nPG14", "Schlawin+\nRCE grid"),
        (summary["reduced_chi_squared"], PAPER["reduced_chi_squared"]),
        color=("#20639b", "#ef5675"),
        alpha=0.75,
    )
    axes[2].set_ylabel(r"Reduced $\chi^2$")
    axes[2].set_title("Clear one-region fit quality")
    figure.suptitle("Published clear-model comparison (models are not identical)")
    figure.tight_layout()
    figure.savefig(output / "comparison_published_parameters.png", dpi=200)
    plt.close(figure)


def _plot_convergence(debug_path: Path, output: Path) -> None:
    pattern = re.compile(
        r"iteration=(?P<iteration>\d+), ncalls=(?P<ncall>\d+).*?"
        r"logz=(?P<logz>[-+0-9.eE]+).*?Lmin=(?P<lmin>[-+0-9.eE]+), "
        r"Lmax=(?P<lmax>[-+0-9.eE]+)"
    )
    records: dict[int, tuple[int, float, float, float]] = {}
    for line in debug_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        iteration = int(match["iteration"])
        try:
            record = (
                int(match["ncall"]),
                float(match["logz"]),
                float(match["lmin"]),
                float(match["lmax"]),
            )
        except ValueError:
            continue
        records.setdefault(iteration, record)
    ordered = [records[key] for key in sorted(records)]
    values = np.asarray(ordered, dtype=float)
    tail = values[(values[:, 0] >= 10_000) & (values[:, 1] >= 1_800)]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(tail[:, 0], tail[:, 1], label="accumulated logZ", linewidth=1.5)
    axis.plot(tail[:, 0], tail[:, 2], label="minimum live logL", linewidth=1.2)
    axis.plot(tail[:, 0], tail[:, 3], label="best live logL", linewidth=1.2)
    axis.axvline(20049, color="0.3", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Likelihood evaluations")
    axis.set_ylabel("Log value")
    axis.set_title("UltraNest exploration history (high-likelihood tail)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "comparison_convergence.png", dpi=200)
    plt.close(figure)


def _write_comparison(
    summary: dict[str, object],
    weights: np.ndarray,
    output: Path,
) -> None:
    robert = summary["posterior_16_50_84"]
    effective_sample_size = float(1.0 / np.sum(weights**2))
    comparison = {
        "schema_version": 1,
        "run_classification": "exploratory_forced_early_stop",
        "posterior_effective_sample_size": effective_sample_size,
        "maximum_posterior_weight": float(np.max(weights)),
        "robert": {
            "dataset_selection": summary["selection"],
            "n_points": summary["n_points"],
            "metallicity_16_50_84": robert["metallicity"],
            "CtoO_16_50_84": robert["CtoO"],
            "reduced_chi_squared": summary["reduced_chi_squared"],
            "best_fit": summary["best_fit"],
        },
        "published": PAPER,
        "median_differences_robert_minus_published": {
            "metallicity_dex": robert["metallicity"][1]
            - PAPER["metallicity_16_50_84"][1],
            "CtoO": robert["CtoO"][1] - PAPER["CtoO_16_50_84"][1],
            "reduced_chi_squared": summary["reduced_chi_squared"]
            - PAPER["reduced_chi_squared"],
        },
        "comparability": {
            "spectrum": "near-full published native spectrum; six overlap-average bins excluded",
            "temperature_profile": "ROBERT PG14 analytic versus published interpolated EGP RCE grid",
            "opacities_and_rt": "ROBERT ExoMol/petitRADTRANS correlated-k versus CHIMERA",
            "evidence": "not comparable because likelihood/model definitions differ and ROBERT was force-stopped",
            "posterior_warning": "ESS is too small for production credible intervals",
        },
    }
    (output / "published_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary, products, samples, weights = _load(args.output)
    observations = retrieval.native_mode_observations()
    problem = retrieval.build_native_mode_problem(observations)
    _plot_spectrum(problem, products, summary, args.output)
    _plot_corner(samples, weights, args.output)
    _plot_composition_and_temperature(problem, samples, weights, args.output)
    _plot_paper_parameters(samples, weights, summary, args.output)
    _plot_convergence(args.output / "ultranest" / "debug.log", args.output)
    _write_comparison(summary, weights, args.output)
    print(f"Wrote WASP-69b comparison products to {args.output}")


if __name__ == "__main__":
    main()
