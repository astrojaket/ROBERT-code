"""Plot the controlled Version-2 Stage-8 grey-aerosol comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-stage8-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets.diagnostics.benchmark_style import (
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCT_ROOT = ROOT / "examples/outputs/emission_intercomparison/version_2/stage_8/controlled_study"
FRAMEWORK_STYLE = {
    "robert": (ROBERT_COLOR, "-"),
    "picaso": (REFERENCE_COLOR, "--"),
    "petitradtrans": ("#777078", ":"),
}
STATE_STYLE = {
    "clear": (0.45, ":"),
    "absorbing_cloud": (0.72, "--"),
    "scattering_cloud": (1.0, "-"),
}


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _primary_index(arrays: dict[str, np.ndarray]) -> int:
    return int(
        np.flatnonzero(
            (arrays["profile"] == "pg14_non_inverted") & (arrays["cells"] == 80)
        )[0]
    )


def plot_primary(arrays: dict[str, np.ndarray], output: Path) -> None:
    wavelength = arrays["r100_centers_micron"]
    shard = _primary_index(arrays)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.5), sharex=True, constrained_layout=True)
    for framework_index, framework_value in enumerate(arrays["framework"]):
        framework = str(framework_value)
        color, framework_line = FRAMEWORK_STYLE[framework]
        for state_index, state_value in enumerate(arrays["state"]):
            state = str(state_value)
            alpha, state_line = STATE_STYLE[state]
            line = framework_line if state == "scattering_cloud" else state_line
            axes[0].plot(
                wavelength,
                arrays["r100_eclipse_depth"][framework_index, shard, state_index] * 1.0e6,
                color=color,
                linestyle=line,
                alpha=alpha,
                label=f"{framework} {state.replace('_', ' ')}",
            )
        axes[1].plot(
            wavelength,
            arrays["r100_scattering_increment_eclipse_ppm"][framework_index, shard],
            color=color,
            linestyle=framework_line,
            label=framework,
        )
    axes[0].set(ylabel="Eclipse depth (ppm)", xscale="log", title="Controlled isotropic grey-aerosol emission")
    axes[1].axhline(0.0, color=RESIDUAL_COLOR, linewidth=0.8, alpha=0.45)
    axes[1].set(xlabel="Wavelength (micron)", ylabel="Scattering - absorbing (ppm)", xscale="log")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].legend(frameon=False, fontsize=8, ncol=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_convergence(arrays: dict[str, np.ndarray], output: Path) -> None:
    wavelength = arrays["r100_centers_micron"]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True, constrained_layout=True)
    for framework_index, framework_value in enumerate(arrays["framework"]):
        framework = str(framework_value)
        color, _ = FRAMEWORK_STYLE[framework]
        axis = axes[framework_index]
        for cells, linestyle in ((40, ":"), (80, "-"), (160, "--")):
            shard = int(
                np.flatnonzero(
                    (arrays["profile"] == "pg14_non_inverted")
                    & (arrays["cells"] == cells)
                )[0]
            )
            axis.plot(
                wavelength,
                arrays["r100_scattering_increment_eclipse_ppm"][framework_index, shard],
                color=color,
                linestyle=linestyle,
                label=f"{cells} cells",
            )
        axis.axhline(0.0, color=RESIDUAL_COLOR, linewidth=0.7, alpha=0.4)
        axis.set(xscale="log", ylabel=f"{framework}\nincrement (ppm)")
        axis.legend(frameon=False, fontsize=8, ncol=3)
    axes[0].set_title("Vertical-grid convergence of the scattering increment")
    axes[-1].set_xlabel("Wavelength (micron)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-root", type=Path, default=DEFAULT_PRODUCT_ROOT)
    args = parser.parse_args()
    arrays = _load(args.product_root / "controlled_study_arrays.npz")
    plot_primary(arrays, args.product_root / "controlled_primary.png")
    plot_convergence(arrays, args.product_root / "controlled_convergence.png")


if __name__ == "__main__":
    main()
