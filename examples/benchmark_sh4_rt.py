"""Benchmark ROBERT Toon and SH4 thermal scattering against PICASO."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from compare_grey_cloud_rt_picaso import (
    GreyRTCase,
    _disk_quadrature,
    _run_picaso,
)
from robert_exoplanets import planck_radiance_wavelength
from robert_exoplanets.rt.sh4 import solve_thermal_sh4
from robert_exoplanets.rt.toon import solve_thermal_two_stream

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "sh4_rt_benchmark"


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picaso-python",
        type=Path,
        default=Path(
            os.environ.get(
                "ROBERT_PICASO_PYTHON",
                "/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python",
            )
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wavelength = np.geomspace(1.0, 12.0, 80)
    pressure_edges = np.geomspace(1.0e-5, 10.0, 161)
    temperature = np.linspace(600.0, 1500.0, pressure_edges.size)
    temperature[-2:] = 1500.0
    extinction_tau = np.full((pressure_edges.size - 1, wavelength.size), 2.0 / 160.0)
    mu, disk_weight = _disk_quadrature(6)
    cases = (
        GreyRTCase("isotropic_w090", temperature, extinction_tau, 0.9, 0.0, 1.0e-4),
        GreyRTCase("forward_w090_g060", temperature, extinction_tau, 0.9, 0.6, 1.0e-4),
    )

    comparisons: dict[str, dict[str, Any]] = {}
    plotted: dict[str, np.ndarray] = {}
    for case in cases:
        level_planck = _level_planck(wavelength, case.temperature_level_k)
        toon = _robert_point("toon", case, level_planck, mu)
        sh4 = _robert_point("sh4", case, level_planck, mu)
        picaso_toon = _run_picaso(
            args.picaso_python,
            case,
            wavelength,
            pressure_edges,
            mu,
            args.output_dir,
            method="toon",
        )
        picaso_sh4 = _run_picaso(
            args.picaso_python,
            case,
            wavelength,
            pressure_edges,
            mu,
            args.output_dir,
            method="sh4",
        )
        comparisons[case.name] = {
            "toon": _accuracy(toon, picaso_toon, disk_weight),
            "sh4": _accuracy(sh4, picaso_sh4, disk_weight),
        }
        if case.name == "forward_w090_g060":
            plotted = {
                "toon": toon,
                "sh4": sh4,
                "picaso_toon": picaso_toon,
                "picaso_sh4": picaso_sh4,
            }

    speed = _speed_benchmark(mu)
    report = {
        "schema_version": 1,
        "comparison": "ROBERT_Toon_SH4_vs_PICASO_matched_grey_thermal_RT",
        "phase_function_conventions": {
            "toon": "hemispheric_mean",
            "sh4": "four_term_Henyey_Greenstein_without_delta_M_for_matched_reference",
            "robert_science_default": "SH4_Henyey_Greenstein_with_delta_M",
        },
        "accuracy": comparisons,
        "speed": speed,
    }
    json_path = args.output_dir / "sh4_rt_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = args.output_dir / "sh4_rt_benchmark.png"
    _plot(plot_path, wavelength, mu, disk_weight, plotted, speed)
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(json.dumps(report, indent=2))
    return report


def _level_planck(wavelength: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    return np.vstack(
        [planck_radiance_wavelength(wavelength, value) for value in temperature]
    )


def _robert_point(
    method: str,
    case: GreyRTCase,
    level_planck: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    tau = case.extinction_tau[:, :, None]
    omega = np.full_like(tau, case.single_scattering_albedo)
    asymmetry = np.full_like(tau, case.asymmetry_factor)
    if method == "toon":
        result = solve_thermal_two_stream(
            tau,
            omega,
            asymmetry,
            level_planck,
            mu,
            bottom_planck_radiance=level_planck[-1],
        )
    else:
        result = solve_thermal_sh4(
            tau,
            omega,
            asymmetry,
            level_planck,
            mu,
            bottom_planck_radiance=level_planck[-1],
            delta_m=False,
            source_quadrature_order=8,
        )
    return np.asarray(result.point_radiance[:, :, 0])


def _accuracy(
    robert: np.ndarray,
    picaso: np.ndarray,
    disk_weight: np.ndarray,
) -> dict[str, Any]:
    relative = (robert - picaso) / np.maximum(np.abs(picaso), np.finfo(float).tiny)
    robert_disk = np.tensordot(disk_weight, robert, axes=(0, 0))
    picaso_disk = np.tensordot(disk_weight, picaso, axes=(0, 0))
    disk_relative = (robert_disk - picaso_disk) / np.maximum(
        np.abs(picaso_disk), np.finfo(float).tiny
    )
    return {
        "max_abs_point_relative_difference": float(np.max(np.abs(relative))),
        "max_abs_disk_relative_difference": float(np.max(np.abs(disk_relative))),
        "median_disk_relative_difference": float(np.median(disk_relative)),
        "max_abs_relative_difference_by_mu": np.max(np.abs(relative), axis=1).tolist(),
    }


def _speed_benchmark(mu: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(20260711)
    nlayer, nwavelength, ng = 64, 900, 4
    wavelength = np.geomspace(1.0, 12.0, nwavelength)
    temperature = np.linspace(650.0, 1550.0, nlayer + 1)
    planck = _level_planck(wavelength, temperature)
    tau = rng.uniform(0.002, 0.05, size=(nlayer, nwavelength, ng))
    omega = rng.uniform(0.5, 0.95, size=tau.shape)
    asymmetry = rng.uniform(0.0, 0.8, size=tau.shape)
    bottom = planck[-1]

    def toon() -> None:
        solve_thermal_two_stream(
            tau,
            omega,
            asymmetry,
            planck,
            mu,
            bottom_planck_radiance=bottom,
        )

    def sh4() -> None:
        solve_thermal_sh4(
            tau,
            omega,
            asymmetry,
            planck,
            mu,
            bottom_planck_radiance=bottom,
            delta_m=True,
        )

    toon_seconds = _median_seconds(toon, repeats=3)
    sh4_seconds = _median_seconds(sh4, repeats=3)
    return {
        "shape": {
            "n_layers": nlayer,
            "n_wavelength": nwavelength,
            "n_correlated_k_g": ng,
            "n_emission_angles": int(mu.size),
        },
        "median_seconds": {"toon": toon_seconds, "sh4": sh4_seconds},
        "sh4_to_toon_runtime_ratio": sh4_seconds / toon_seconds,
        "repeats": 3,
    }


def _median_seconds(function: Callable[[], None], *, repeats: int) -> float:
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _plot(
    output_path: Path,
    wavelength: np.ndarray,
    mu: np.ndarray,
    disk_weight: np.ndarray,
    values: dict[str, np.ndarray],
    speed: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    ax_spectrum, ax_residual, ax_angle, ax_speed = axes.flat
    colors = {"toon": "#e45756", "sh4": "#4c78a8"}
    for method in ("toon", "sh4"):
        picaso = values[f"picaso_{method}"]
        robert = values[method]
        picaso_disk = np.tensordot(disk_weight, picaso, axes=(0, 0))
        robert_disk = np.tensordot(disk_weight, robert, axes=(0, 0))
        ax_spectrum.plot(wavelength, picaso_disk, color=colors[method], label=f"PICASO {method.upper()}")
        ax_spectrum.plot(
            wavelength,
            robert_disk,
            color=colors[method],
            linestyle="--",
            label=f"ROBERT {method.upper()}",
        )
        disk_relative = (robert_disk - picaso_disk) / picaso_disk
        ax_residual.plot(wavelength, 100.0 * disk_relative, color=colors[method], label=method.upper())
        point_relative = (robert - picaso) / picaso
        ax_angle.plot(mu, 100.0 * np.max(np.abs(point_relative), axis=1), "o-", color=colors[method], label=method.upper())

    ax_spectrum.set(xscale="log", yscale="log", xlabel="Wavelength [micron]", ylabel="Disk radiance [W m$^{-2}$ m$^{-1}$ sr$^{-1}$]", title="Forward-scattering grey atmosphere")
    ax_spectrum.legend(frameon=False)
    ax_residual.axhline(0.0, color="#222222", linewidth=0.8)
    ax_residual.set(xscale="log", xlabel="Wavelength [micron]", ylabel="(ROBERT - PICASO) / PICASO [%]", title="Matched-solver spectral residual")
    ax_residual.legend(frameon=False)
    ax_angle.set(xlabel=r"Emission cosine $\mu$", ylabel="Maximum point-radiance error [%]", yscale="log", title="Angular reconstruction error")
    ax_angle.legend(frameon=False)

    timing = speed["median_seconds"]
    bars = ax_speed.bar(["Toon", "SH4"], [timing["toon"], timing["sh4"]], color=[colors["toon"], colors["sh4"]])
    ax_speed.bar_label(bars, fmt="%.3f s", padding=3)
    ax_speed.set(ylabel="Median wall time [s]", title="64 layers × 900 wavelengths × 4 g ordinates")
    for axis in axes.flat:
        axis.grid(alpha=0.25, which="both")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
