"""Plot Version-2 Stage-6 composition responses and convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "docs/data/emission_intercomparison/version_2"
PROFILES = ("pg14_non_inverted", "pg14_inverted")
MODELS = ("robert", "picaso", "petitradtrans")
SPECIES = ("H2O", "CO", "CO2", "CH4")
LABELS = {"robert": "ROBERT", "picaso": "PICASO 4.0 CK", "petitradtrans": "stable pRT"}
COLORS = {"robert": "#1565c0", "picaso": "#ef6c00", "petitradtrans": "#2e7d32"}
SPECIES_COLORS = {"H2O": "#1565c0", "CO": "#6a1b9a", "CO2": "#ef6c00", "CH4": "#2e7d32"}


def _main(model: str, profile: str, species: str = "H2O", n_cells: int = 80):
    path = DATA / f"stage_6_main_{profile}_{species}_{n_cells}_cells_{model}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _response(model: str, profile: str, n_cells: int = 80):
    path = DATA / f"stage_6_response_{profile}_{model}_{n_cells}_cells.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def plot_spectra_and_jacobians(output: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    center_index = 2
    for row, profile in enumerate(PROFILES):
        for model in MODELS:
            baseline = _main(model, profile)
            baseline_index = np.flatnonzero(baseline["perturbation_sign"] == 0)[0]
            axes[row, 0].plot(
                baseline["r100_centers_micron"],
                baseline["r100_eclipse_depth"][baseline_index] * 1e6,
                color=COLORS[model],
                lw=1.15,
                label=LABELS[model],
            )
            response = _response(model, profile)
            for column, species in enumerate(SPECIES, start=1):
                species_index = SPECIES.index(species)
                axes[row, column].plot(
                    response["r100_centers_micron"],
                    response["eclipse_jacobian_r100_ppm_dex"][species_index, center_index],
                    color=COLORS[model],
                    lw=1.05,
                )
        axes[row, 0].set_ylabel(
            f"{profile.replace('_', ' ')}\nEclipse depth (ppm)"
        )
        for axis in axes[row, 1:]:
            axis.axhline(0.0, color="0.5", lw=0.6)
            axis.set_ylabel("ppm dex$^{-1}$")
        for axis in axes[row]:
            axis.set_xscale("log")
            axis.grid(alpha=0.18)
    axes[0, 0].set_title("Baseline spectrum")
    for column, species in enumerate(SPECIES, start=1):
        axes[0, column].set_title(f"{species} Jacobian at 0.01 bar")
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (micron)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pressure_responses(output: Path) -> None:
    arrays = _response("robert", "pg14_non_inverted")
    wavelength = arrays["r100_centers_micron"]
    centers = arrays["perturbation_centers_bar"]
    extent = [wavelength[0], wavelength[-1], np.log10(centers[-1]), np.log10(centers[0])]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for column, species in enumerate(SPECIES):
        signed = arrays["eclipse_jacobian_r100_ppm_dex"][column]
        normalized = arrays["normalized_absolute_response_r100"][column]
        scale = np.percentile(np.abs(signed), 99)
        signed_image = axes[0, column].imshow(
            signed,
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
        )
        normalized_image = axes[1, column].imshow(
            normalized,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[0, column].set_title(species)
        fig.colorbar(signed_image, ax=axes[0, column], label="ppm dex$^{-1}$")
        fig.colorbar(normalized_image, ax=axes[1, column], label="Normalized |J|")
        for row in range(2):
            axes[row, column].set_xscale("log")
            axes[row, column].set_xlabel("Wavelength (micron)")
    axes[0, 0].set_ylabel("Signed response\nlog10 pressure (bar)")
    axes[1, 0].set_ylabel("Normalized response\nlog10 pressure (bar)")
    fig.suptitle("ROBERT PG14 non-inverted localized composition response", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cross_species_fractions(output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    for row, profile in enumerate(PROFILES):
        for column, model in enumerate(MODELS):
            arrays = _response(model, profile)
            for index, species in enumerate(SPECIES):
                axes[row, column].plot(
                    arrays["r100_centers_micron"],
                    arrays["cross_species_sensitivity_fraction_r100"][index],
                    color=SPECIES_COLORS[species],
                    lw=1.1,
                    label=species,
                )
            axes[row, column].set_xscale("log")
            axes[row, column].set_ylim(-0.02, 1.02)
            axes[row, column].grid(alpha=0.18)
            axes[row, column].set_title(f"{profile.replace('_', ' ')} — {LABELS[model]}")
            axes[row, column].set_xlabel("Wavelength (micron)")
    axes[0, 0].set_ylabel("Own/cross-species |J| fraction")
    axes[1, 0].set_ylabel("Own/cross-species |J| fraction")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _centroid(weights: np.ndarray, centers: np.ndarray) -> np.ndarray:
    denominator = np.sum(weights, axis=0)
    normalized = np.divide(
        weights,
        denominator,
        out=np.zeros_like(weights),
        where=denominator != 0,
    )
    return np.sum(normalized * np.log10(centers)[:, None], axis=0)


def plot_diagnostic_comparison(output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    profile = "pg14_inverted"
    for axis, model in zip(axes, MODELS, strict=True):
        arrays = _response(model, profile)
        centers = arrays["perturbation_centers_bar"]
        contribution = _centroid(arrays["stage_4_contribution_projection_r100"], centers)
        temperature = _centroid(arrays["stage_5_temperature_response_r100"], centers)
        composition = arrays["response_centroid_r100_log10_bar"][0]
        wavelength = arrays["r100_centers_micron"]
        axis.plot(wavelength, contribution, color="0.25", ls="--", label="Stage 4 contribution")
        axis.plot(wavelength, temperature, color="#8e24aa", label="Stage 5 temperature")
        axis.plot(wavelength, composition, color=COLORS[model], label="Stage 6 H2O composition")
        axis.set_xscale("log")
        axis.grid(alpha=0.18)
        axis.set_title(LABELS[model])
        axis.set_xlabel("Wavelength (micron)")
    axes[0].set_ylabel("Diagnostic centroid log10 pressure (bar)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Distinct source-contribution, temperature-response, and composition-response diagnostics")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_convergence_and_differences(output: Path) -> None:
    report = json.loads((DATA / "stage_6_report.json").read_text())
    convergence = report["vertical_and_r100_spectral_convergence"]
    jacobian = [
        convergence[model]["80_to_160"]["jacobian"][
            "p95_abs_difference_over_pair_peak"
        ]
        * 100
        for model in MODELS
    ]
    response = [
        convergence[model]["80_to_160"]["response"]["profile_total_variation_p95"]
        for model in MODELS
    ]
    observed = report["observed_gate_values"]
    pair_jacobian = [
        observed["track_a_primary_p95_abs_jacobian_difference_over_pair_peak"] * 100,
        observed["track_a_80_to_160_p95_abs_jacobian_difference_over_pair_peak"] * 100,
    ]
    pair_response = [
        observed["track_a_primary_response_total_variation_p95"],
        observed["track_a_80_to_160_response_total_variation_p95"],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    model_x = np.arange(len(MODELS))
    pair_x = np.arange(2)
    axes[0, 0].bar(model_x, jacobian, color=[COLORS[model] for model in MODELS])
    axes[0, 1].bar(model_x, response, color=[COLORS[model] for model in MODELS])
    axes[1, 0].bar(pair_x, pair_jacobian, color=["#5e35b1", "#9575cd"])
    axes[1, 1].bar(pair_x, pair_response, color=["#5e35b1", "#9575cd"])
    for axis in axes[0]:
        axis.set_xticks(model_x, [LABELS[model] for model in MODELS], rotation=12)
    for axis in axes[1]:
        axis.set_xticks(pair_x, ["Primary Track A", "Track A 80→160"], rotation=8)
    axes[0, 0].set_ylabel("Native 80→160 Jacobian p95 (%)")
    axes[0, 1].set_ylabel("Native 80→160 response TV p95")
    axes[1, 0].set_ylabel("Matched Track-A Jacobian p95 (%)")
    axes[1, 1].set_ylabel("Matched Track-A response TV p95")
    axes[1, 0].axhline(5.0, color="#c62828", ls="--", label="Frozen limit")
    axes[1, 1].axhline(0.08, color="#c62828", ls="--", label="Frozen limit")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        for patch in axis.patches:
            axis.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                f"{patch.get_height():.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_spectra_and_jacobians(args.output / "stage6-spectra-composition-jacobians.png")
    plot_pressure_responses(args.output / "stage6-pressure-responses.png")
    plot_cross_species_fractions(args.output / "stage6-own-cross-species.png")
    plot_diagnostic_comparison(args.output / "stage6-diagnostic-comparison.png")
    plot_convergence_and_differences(args.output / "stage6-convergence-differences.png")


if __name__ == "__main__":
    main()
