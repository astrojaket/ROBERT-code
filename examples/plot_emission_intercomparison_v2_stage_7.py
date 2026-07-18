"""Plot local Version-2 Stage-7 absorbing-cloud products with manuscript style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from robert_exoplanets.diagnostics.benchmark_style import (
    PURPLE_DARK,
    PURPLE_LIGHT,
    REFERENCE_COLOR,
    ROBERT_COLOR,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTS = (
    REPOSITORY
    / "examples/outputs/emission_intercomparison/version_2/stage_7/products"
)
PROFILE = "pg14_non_inverted"
CLOUD = "deck_tau1_top10mbar_slope+0"
MODELS = ("robert", "picaso", "petitradtrans")
LABELS = {
    "robert": "ROBERT",
    "picaso": "PICASO 4.0 CK",
    "petitradtrans": "stable pRT",
}
MODEL_STYLE = {
    "robert": {"color": ROBERT_COLOR, "ls": "-", "marker": None},
    "picaso": {"color": "#17131a", "ls": "--", "marker": "o"},
    "petitradtrans": {"color": "#68616b", "ls": ":", "marker": "s"},
}
SIGNED_PURPLE = LinearSegmentedColormap.from_list(
    "signed_purple_neutral", (REFERENCE_COLOR, "#f7f5f8", ROBERT_COLOR)
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _resolution(products: Path, n_cells: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    root = (
        products
        if (products / "native_contract.npz").is_file()
        else products / "matrix" / f"{n_cells}_cells"
    )
    contract = _load(root / "native_contract.npz")
    outputs = {
        f"track_b_{model}": _load(root / f"track_b_{model}.npz")
        for model in MODELS
    }
    outputs.update(
        {
            f"track_a_{model}": _load(root / f"track_a_{model}.npz")
            for model in ("robert", "petitradtrans")
        }
    )
    return contract, outputs


def _case(contract: dict[str, np.ndarray], profile: str, cloud: str) -> int:
    cloud_index = int(np.flatnonzero(contract["cloud_label"] == cloud)[0])
    profile_by_case = contract["profile_name"][contract["profile_index"]]
    return int(
        np.flatnonzero(
            (profile_by_case == profile)
            & (contract["case_cloud_index"] == cloud_index)
        )[0]
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spectra_and_effects(products: Path, output: Path) -> None:
    contract, arrays = _resolution(products, 80)
    cloudy = _case(contract, PROFILE, CLOUD)
    clear = _case(contract, PROFILE, "clear")
    wavelength = arrays["track_b_robert"]["r100_centers_micron"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    for model in MODELS:
        values = arrays[f"track_b_{model}"]
        style = MODEL_STYLE[model]
        axes[0].plot(
            wavelength,
            values["r100_eclipse_depth"][clear] * 1.0e6,
            color=style["color"],
            ls=style["ls"],
            lw=0.9,
            alpha=0.45,
        )
        axes[0].plot(
            wavelength,
            values["r100_eclipse_depth"][cloudy] * 1.0e6,
            label=f"{LABELS[model]} cloudy",
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
            markevery=42,
            ms=2.6,
            lw=1.25,
        )
        axes[1].plot(
            wavelength,
            values["r100_cloud_effect_eclipse_ppm"][cloudy],
            label=LABELS[model],
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
            markevery=42,
            ms=2.6,
            lw=1.15,
        )
    axes[1].axhline(0.0, color=REFERENCE_COLOR, lw=0.7, ls="--")
    axes[0].set_ylabel("Eclipse depth (ppm)")
    axes[1].set_ylabel("Cloudy − clear (ppm)")
    axes[1].set_xlabel("Wavelength (micron)")
    axes[0].set_title("PG14 non-inverted: clear and tau=1, 10-mbar grey cloud")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(0.3, 12.0)
        axis.grid(alpha=0.16)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    _save(fig, output / "stage7-full-domain-spectra-cloud-effects.png")


def plot_cloud_placement(products: Path, output: Path) -> None:
    contract, _ = _resolution(products, 80)
    cloud_index = int(np.flatnonzero(contract["cloud_label"] == CLOUD)[0])
    wavelength = contract["cloud_input_wavelength_micron"]
    pressure = contract["pressure_centers_bar"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, index, title in (
        (axes[0], cloud_index, "Fractional 10-mbar deck placement"),
        (
            axes[1],
            int(np.flatnonzero(contract["cloud_label"] == "archived_virga_mie_extinction")[0]),
            "Archived physical extinction placement",
        ),
    ):
        image = axis.pcolormesh(
            wavelength,
            pressure,
            contract["cloud_input_extinction_tau"][index],
            shading="auto",
            cmap="Purples",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.3, 12.0)
        axis.set_title(title)
        axis.set_xlabel("Wavelength (micron)")
        fig.colorbar(image, ax=axis, label="Layer extinction tau")
    axes[0].set_ylabel("Pressure (bar; increasing downward)")
    axes[0].set_ylim(100.0, 1.0e-5)
    _save(fig, output / "stage7-cloud-placement-extinction-heatmaps.png")


def plot_vertical_cloud_effect(products: Path, output: Path) -> None:
    contract, arrays = _resolution(products, 80)
    cloudy = _case(contract, PROFILE, CLOUD)
    clear = _case(contract, PROFILE, "clear")
    values = arrays["track_b_robert"]
    wavelength = values["r100_centers_micron"]
    pressure = contract["pressure_centers_bar"]
    signed = (
        values["r100_normalized_vertical_diagnostic"][cloudy]
        - values["r100_normalized_vertical_diagnostic"][clear]
    )
    normalized = values["r100_normalized_cloud_effect"][cloudy]
    scale = float(np.percentile(np.abs(signed), 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    signed_image = axes[0].pcolormesh(
        wavelength,
        pressure,
        signed,
        shading="auto",
        cmap=SIGNED_PURPLE,
        vmin=-scale,
        vmax=scale,
    )
    normalized_image = axes[1].pcolormesh(
        wavelength,
        pressure,
        normalized,
        shading="auto",
        cmap="Purples",
        vmin=0.0,
    )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.3, 12.0)
        axis.set_xlabel("Wavelength (micron)")
    axes[0].set_ylabel("Pressure (bar; increasing downward)")
    axes[0].set_ylim(100.0, 1.0e-5)
    axes[0].set_title("Signed contribution redistribution")
    axes[1].set_title("Normalized absolute cloud effect")
    fig.colorbar(signed_image, ax=axes[0], label="Cloudy − clear normalized contribution")
    fig.colorbar(normalized_image, ax=axes[1], label="Normalized |effect|")
    _save(fig, output / "stage7-signed-normalized-vertical-cloud-effect.png")


def plot_track_differences(products: Path, output: Path) -> None:
    contract, arrays = _resolution(products, 80)
    cloudy = _case(contract, PROFILE, CLOUD)
    wavelength = arrays["track_b_robert"]["r100_centers_micron"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    shared = (
        arrays["track_a_robert"]["r100_cloud_effect_eclipse_ppm"][cloudy]
        - arrays["track_a_petitradtrans"]["r100_cloud_effect_eclipse_ppm"][cloudy]
    )
    axes[0].plot(wavelength, shared, color=PURPLE_DARK, lw=1.15, label="ROBERT − stable pRT")
    for model in ("picaso", "petitradtrans"):
        difference = (
            arrays["track_b_robert"]["r100_cloud_effect_eclipse_ppm"][cloudy]
            - arrays[f"track_b_{model}"]["r100_cloud_effect_eclipse_ppm"][cloudy]
        )
        style = MODEL_STYLE[model]
        axes[1].plot(
            wavelength,
            difference,
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
            markevery=42,
            ms=2.6,
            label=f"ROBERT − {LABELS[model]}",
        )
    for axis in axes:
        axis.axhline(0.0, color=REFERENCE_COLOR, lw=0.7, ls="--")
        axis.set_xscale("log")
        axis.set_xlim(0.3, 12.0)
        axis.set_ylabel("Cloud-effect difference (ppm)")
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_title("Track A: genuinely identical gas + cloud tau")
    axes[1].set_title("Track B: native attribution only")
    axes[1].set_xlabel("Wavelength (micron)")
    _save(fig, output / "stage7-track-a-track-b-differences.png")


def plot_band_windows(products: Path, output: Path) -> None:
    contract, arrays = _resolution(products, 80)
    cloudy = _case(contract, PROFILE, CLOUD)
    wavelength = arrays["track_b_robert"]["r100_centers_micron"]
    report_path = products / "stage_7_report.json"
    names = (
        list(
            json.loads(report_path.read_text()).get(
                "predeclared_band_windows_micron", {}
            )
        )
        if report_path.is_file()
        else []
    )
    if not names:
        names = ["optical", "near_ir_water_band", "near_ir_window", "methane_band", "co_co2_band", "mid_ir_water_band", "mid_ir_window"]
    bounds = {
        "optical": (0.3, 0.8),
        "near_ir_water_band": (1.35, 1.55),
        "near_ir_window": (2.0, 2.3),
        "methane_band": (3.1, 3.6),
        "co_co2_band": (4.2, 5.0),
        "mid_ir_water_band": (5.5, 7.5),
        "mid_ir_window": (8.0, 10.0),
    }
    x = np.arange(len(names))
    width = 0.25
    fig, axis = plt.subplots(figsize=(12, 5.2))
    for model_index, model in enumerate(MODELS):
        effect = arrays[f"track_b_{model}"]["r100_cloud_effect_eclipse_ppm"][cloudy]
        means = [
            float(np.mean(effect[(wavelength >= bounds[name][0]) & (wavelength <= bounds[name][1])]))
            for name in names
        ]
        style = MODEL_STYLE[model]
        axis.bar(
            x + (model_index - 1) * width,
            means,
            width,
            color=style["color"],
            edgecolor="black",
            linewidth=0.45,
            hatch=(None, "//", "..")[model_index],
            label=LABELS[model],
        )
    axis.axhline(0.0, color=REFERENCE_COLOR, lw=0.7)
    axis.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=8)
    axis.set_ylabel("Mean cloudy − clear eclipse depth (ppm)")
    axis.set_title("Predeclared band/window cloud effects")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.16)
    _save(fig, output / "stage7-band-window-diagnostics.png")


def plot_convergence(products: Path, output: Path) -> None:
    loaded = {resolution: _resolution(products, resolution) for resolution in (40, 80, 160)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    colors = {40: PURPLE_LIGHT, 80: ROBERT_COLOR, 160: PURPLE_DARK}
    effects: dict[int, np.ndarray] = {}
    wavelength = np.empty(0)
    for resolution, (contract, arrays) in loaded.items():
        cloudy = _case(contract, PROFILE, CLOUD)
        wavelength = arrays["track_b_robert"]["r100_centers_micron"]
        effects[resolution] = arrays["track_b_robert"]["r100_cloud_effect_eclipse_ppm"][cloudy]
        axes[0].plot(
            wavelength,
            effects[resolution],
            color=colors[resolution],
            lw=1.05,
            ls=(":", "--", "-")[(40, 80, 160).index(resolution)],
            label=f"{resolution} cells",
        )
    axes[1].plot(wavelength, effects[40] - effects[80], color=PURPLE_LIGHT, ls=":", label="40 − 80")
    axes[1].plot(wavelength, effects[80] - effects[160], color=PURPLE_DARK, ls="--", label="80 − 160")
    for axis in axes:
        axis.axhline(0.0, color=REFERENCE_COLOR, lw=0.7)
        axis.set_xscale("log")
        axis.set_xlim(0.3, 12.0)
        axis.set_xlabel("Wavelength (micron)")
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Cloud effect (ppm)")
    axes[1].set_ylabel("Convergence difference (ppm)")
    axes[0].set_title("ROBERT native vertical-grid ladder")
    axes[1].set_title("Signed spectral convergence")
    _save(fig, output / "stage7-40-80-160-convergence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--product-root", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument(
        "--representative-pilot",
        action="store_true",
        help="plot one warm pilot directory and omit unavailable 40/80/160 convergence",
    )
    args = parser.parse_args()
    products = args.product_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_spectra_and_effects(products, output)
    plot_cloud_placement(products, output)
    plot_vertical_cloud_effect(products, output)
    plot_track_differences(products, output)
    plot_band_windows(products, output)
    if not args.representative_pilot:
        plot_convergence(products, output)


if __name__ == "__main__":
    main()
