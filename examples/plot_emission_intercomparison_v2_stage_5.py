"""Plot Version-2 Stage-5 spectra, Jacobians, responses, and convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "docs/data/emission_intercomparison/version_2"
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
MODELS = ("robert", "picaso", "petitradtrans")
LABELS = {"robert": "ROBERT", "picaso": "PICASO 4.0 CK", "petitradtrans": "stable pRT"}
COLORS = {"robert": "#1565c0", "picaso": "#ef6c00", "petitradtrans": "#2e7d32"}


def _load_model(model: str, profile: str, n_cells: int = 80) -> dict[str, np.ndarray]:
    path = (
        DATA / f"stage_5_robert_{n_cells}_cells.npz"
        if model == "robert"
        else DATA / f"stage_5_{model}_{profile}_{n_cells}_cells.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if model == "robert":
        profile_index = PROFILES.index(profile)
        count = arrays["case_id"].size
        selected = arrays["profile_index"] == profile_index
        output = {}
        for name, value in arrays.items():
            if value.ndim and value.shape[0] == count:
                output[name] = value[selected]
            elif value.ndim and value.shape[0] == 3 and name not in {"gas_name"}:
                output[name] = value[profile_index : profile_index + 1]
            else:
                output[name] = value
        return output
    return arrays


def plot_spectra_and_jacobians(output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    center_index = 2  # 0.01 bar
    for row, profile in enumerate(PROFILES):
        for model in MODELS:
            arrays = _load_model(model, profile)
            baseline = np.flatnonzero(arrays["perturbation_sign"] == 0)[0]
            axes[row, 0].plot(
                arrays["r100_centers_micron"],
                arrays["r100_eclipse_depth"][baseline] * 1e6,
                color=COLORS[model],
                lw=1.3,
                label=LABELS[model],
            )
            axes[row, 1].plot(
                arrays["r100_centers_micron"],
                arrays["eclipse_jacobian_r100_ppm_k"][0, center_index],
                color=COLORS[model],
                lw=1.3,
                label=LABELS[model],
            )
        axes[row, 0].set_ylabel(f"{profile.replace('_', ' ')}\nEclipse depth (ppm)")
        axes[row, 1].set_ylabel("d eclipse / dT (ppm K$^{-1}$)")
        axes[row, 1].axhline(0, color="0.45", lw=0.7)
        for axis in axes[row]:
            axis.set_xscale("log")
            axis.grid(alpha=0.2)
    axes[0, 0].set_title("Native-framework baseline spectra on R=100 grid")
    axes[0, 1].set_title("Signed localized Jacobian at 0.01 bar")
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (micron)")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pressure_responses(output: Path) -> None:
    profile = "pg14_non_inverted"
    robert = _load_model("robert", profile)
    wavelength = robert["r100_centers_micron"]
    centers = robert["perturbation_centers_bar"]
    signed = robert["eclipse_jacobian_r100_ppm_k"][0]
    normalized = robert["normalized_absolute_response_r100"][0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    extent = [wavelength[0], wavelength[-1], np.log10(centers[-1]), np.log10(centers[0])]
    scale = np.percentile(np.abs(signed), 99)
    image = axes[0].imshow(
        signed,
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
    )
    fig.colorbar(image, ax=axes[0], label="ppm K$^{-1}$")
    image = axes[1].imshow(
        normalized,
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(image, ax=axes[1], label="Normalized |response|")
    wavelength_index = int(np.argmin(np.abs(wavelength - 4.5)))
    for model in MODELS:
        arrays = _load_model(model, profile)
        axes[2].plot(
            arrays["normalized_absolute_response_r100"][0, :, wavelength_index],
            np.log10(centers),
            marker="o",
            color=COLORS[model],
            label=LABELS[model],
        )
    for axis in axes[:2]:
        axis.set_xscale("log")
        axis.set_xlabel("Wavelength (micron)")
    axes[0].set_ylabel("log10 pressure (bar)")
    axes[0].set_title("ROBERT signed response")
    axes[1].set_title("ROBERT normalized absolute response")
    axes[2].set_title(f"Framework profiles at {wavelength[wavelength_index]:.2f} micron")
    axes[2].set_xlabel("Normalized |response|")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_contribution_relation(output: Path) -> None:
    profile = "pg14_inverted"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS, strict=True):
        arrays = _load_model(model, profile)
        response = arrays["normalized_absolute_response_r100"][0]
        if model == "robert":
            stage4_path = DATA / "stage_4_robert_pg14_inverted_80_cells.npz"
        else:
            stage4_path = DATA / f"stage_4_{model}_80_cells.npz"
        with np.load(stage4_path, allow_pickle=False) as stage4:
            contribution = np.asarray(stage4["normalized_vertical_r100"])
            if contribution.shape[0] > 1:
                contribution = contribution[PROFILES.index(profile)]
            else:
                contribution = contribution[0]
            pressure = np.asarray(stage4["pressure_centers_bar"])
        kernels = np.exp(
            -0.5
            * (
                np.log10(pressure[None, :] / arrays["perturbation_centers_bar"][:, None])
                / float(arrays["localization_sigma_dex"])
            )
            ** 2
        )
        projected = kernels @ contribution
        projected = np.divide(
            projected,
            np.sum(projected, axis=0, keepdims=True),
            out=np.zeros_like(projected),
            where=np.sum(projected, axis=0, keepdims=True) != 0,
        )
        response_centroid = np.sum(response * np.log10(arrays["perturbation_centers_bar"])[:, None], axis=0)
        contribution_centroid = np.sum(projected * np.log10(arrays["perturbation_centers_bar"])[:, None], axis=0)
        axis.plot(arrays["r100_centers_micron"], response_centroid, color=COLORS[model], label="Temperature response")
        axis.plot(arrays["r100_centers_micron"], contribution_centroid, color="0.25", ls="--", label="Projected Stage-4 contribution")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
        axis.set_title(LABELS[model])
        axis.set_xlabel("Wavelength (micron)")
    axes[0].set_ylabel("Response centroid log10 pressure (bar)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_convergence(output: Path) -> None:
    report = json.loads((DATA / "stage_5_report.json").read_text())
    labels = [LABELS[model] for model in MODELS]
    p95 = [
        report["vertical_and_r100_spectral_convergence"][model]["80_to_160"]["jacobian"][
            "p95_abs_difference_over_pair_peak"
        ]
        * 100
        for model in MODELS
    ]
    tv = [
        report["vertical_and_r100_spectral_convergence"][model]["80_to_160"]["response"][
            "profile_total_variation_p95"
        ]
        for model in MODELS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(3)
    axes[0].bar(x, p95, color=[COLORS[m] for m in MODELS])
    axes[1].bar(x, tv, color=[COLORS[m] for m in MODELS])
    for axis, values in zip(axes, (p95, tv), strict=True):
        axis.set_xticks(x, labels, rotation=15)
        axis.grid(axis="y", alpha=0.2)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("80-to-160 Jacobian p95 change (%)")
    axes[1].set_ylabel("80-to-160 response TV p95")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_spectra_and_jacobians(args.output / "stage5-spectra-jacobians.png")
    plot_pressure_responses(args.output / "stage5-pressure-responses.png")
    plot_contribution_relation(args.output / "stage5-contribution-response.png")
    plot_convergence(args.output / "stage5-convergence.png")


if __name__ == "__main__":
    main()
