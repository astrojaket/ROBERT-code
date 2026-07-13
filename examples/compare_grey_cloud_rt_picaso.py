"""Compare controlled grey-cloud thermal RT cases between ROBERT and PICASO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    CloudOpticalProperties,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    planck_radiance_wavelength,
    solve_emission,
)
from robert_exoplanets.rt.toon import solve_thermal_two_stream
from robert_exoplanets.diagnostics.benchmark_style import (
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "grey_cloud_rt_picaso"
PICASO_RUNNER = Path(__file__).with_name("run_picaso_grey_rt_reference.py")


@dataclass(frozen=True)
class GreyRTCase:
    name: str
    temperature_level_k: np.ndarray
    extinction_tau: np.ndarray
    single_scattering_albedo: float
    asymmetry_factor: float
    relative_tolerance: float


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

    wavelength_micron = np.geomspace(1.0, 12.0, 80)
    pressure_edges_bar = np.geomspace(1.0e-5, 10.0, 161)
    emission_mu, emission_weight = _disk_quadrature(6)
    cases = _cases(pressure_edges_bar.size - 1, wavelength_micron.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in cases:
        robert_point, robert_disk = _run_robert(
            case,
            wavelength_micron,
            pressure_edges_bar,
            emission_mu,
            emission_weight,
        )
        picaso_point = _run_picaso(
            args.picaso_python,
            case,
            wavelength_micron,
            pressure_edges_bar,
            emission_mu,
            args.output_dir,
        )
        picaso_disk = np.tensordot(emission_weight, picaso_point, axes=(0, 0))
        report = _comparison_report(case, robert_point, picaso_point, robert_disk, picaso_disk)
        reports.append(report)
        spectra[case.name] = (robert_disk, picaso_disk)

    result = {
        "schema_version": 1,
        "comparison": "ROBERT_vs_PICASO_controlled_grey_thermal_RT",
        "picaso_python": str(args.picaso_python),
        "shared_inputs": {
            "n_layers": int(pressure_edges_bar.size - 1),
            "n_wavelength": int(wavelength_micron.size),
            "wavelength_min_micron": float(wavelength_micron.min()),
            "wavelength_max_micron": float(wavelength_micron.max()),
            "n_emission_angles": int(emission_mu.size),
            "bottom_boundary": "blackbody_hard_surface",
            "gas_optical_depth": "zero",
        },
        "cases": reports,
        "all_cases_pass": all(item["passed"] for item in reports),
    }
    json_path = args.output_dir / "grey_cloud_rt_picaso_comparison.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_path = args.output_dir / "grey_cloud_rt_picaso_comparison.png"
    _plot(plot_path, wavelength_micron, spectra)
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(json.dumps(result, indent=2))
    return result


def _cases(nlayer: int, nwavelength: int) -> tuple[GreyRTCase, ...]:
    isothermal = np.full(nlayer + 1, 1000.0)
    gradient = np.linspace(600.0, 1500.0, nlayer + 1)
    gradient[-2:] = 1500.0
    thin_tau = np.full((nlayer, nwavelength), 2.0 / nlayer)
    return (
        GreyRTCase("isothermal_absorbing", isothermal, thin_tau, 0.0, 0.0, 2.0e-5),
        GreyRTCase("gradient_absorbing", gradient, thin_tau, 0.0, 0.0, 1.0e-2),
        GreyRTCase("gradient_scattering_w050_g000", gradient, thin_tau, 0.5, 0.0, 5.0e-2),
        GreyRTCase("gradient_scattering_w090_g000", gradient, thin_tau, 0.9, 0.0, 5.0e-2),
        GreyRTCase("gradient_scattering_w090_g060", gradient, thin_tau, 0.9, 0.6, 5.0e-2),
    )


def _run_robert(
    case: GreyRTCase,
    wavelength_micron: np.ndarray,
    pressure_edges_bar: np.ndarray,
    emission_mu: np.ndarray,
    emission_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if case.single_scattering_albedo > 0.0:
        level_planck = np.vstack(
            [planck_radiance_wavelength(wavelength_micron, value) for value in case.temperature_level_k]
        )
        bottom_planck = planck_radiance_wavelength(
            wavelength_micron,
            float(case.temperature_level_k[-1]),
        )
        two_stream = solve_thermal_two_stream(
            case.extinction_tau[:, :, None],
            np.full((*case.extinction_tau.shape, 1), case.single_scattering_albedo),
            np.full((*case.extinction_tau.shape, 1), case.asymmetry_factor),
            level_planck,
            emission_mu,
            bottom_planck_radiance=bottom_planck,
        )
        point = np.asarray(two_stream.point_radiance[:, :, 0])
        return point, np.tensordot(emission_weight, point, axes=(0, 0))

    pressure_grid = PressureGrid(
        edges=pressure_edges_bar,
        centers=np.sqrt(pressure_edges_bar[:-1] * pressure_edges_bar[1:]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array(wavelength_micron, unit="micron", role="opacity")
    layer_temperature = 0.5 * (case.temperature_level_k[:-1] + case.temperature_level_k[1:])
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=layer_temperature,
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-12)},
        mean_molecular_weight=2.3,
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="zero-opacity-grey-rt-comparison",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key=f"grey-rt-{case.name}",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure_grid.n_layers, spectral_grid.size, 1)),
        unit="m^2/molecule",
    )
    gas_tau = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)
    cloud = CloudOpticalProperties(
        name=case.name,
        extinction_tau=case.extinction_tau,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=case.single_scattering_albedo,
        asymmetry_factor=case.asymmetry_factor,
    )
    result = solve_emission(
        gas_tau,
        emission_angle_cosines=emission_mu,
        emission_angle_weights=emission_weight,
        bottom_boundary="blackbody",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="two_stream" if case.single_scattering_albedo > 0.0 else "none",
    )
    return np.asarray(result.point_radiance), np.asarray(result.radiance.values)


def _run_picaso(
    python: Path,
    case: GreyRTCase,
    wavelength_micron: np.ndarray,
    pressure_edges_bar: np.ndarray,
    emission_mu: np.ndarray,
    output_dir: Path,
    *,
    method: str = "toon",
) -> np.ndarray:
    input_path = output_dir / f"{case.name}_shared_input.npz"
    output_path = output_dir / f"{case.name}_picaso_{method}_output.npz"
    np.savez_compressed(
        input_path,
        wavelength_micron=wavelength_micron,
        pressure_edges_bar=pressure_edges_bar,
        temperature_level_k=case.temperature_level_k,
        extinction_tau=case.extinction_tau,
        single_scattering_albedo=np.full_like(case.extinction_tau, case.single_scattering_albedo),
        asymmetry_factor=np.full_like(case.extinction_tau, case.asymmetry_factor),
        emission_mu=emission_mu,
    )
    environment = os.environ.copy()
    environment.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "picaso-numba-cache"))
    completed = subprocess.run(
        [
            str(python),
            str(PICASO_RUNNER),
            str(input_path),
            str(output_path),
            "--method",
            method,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PICASO reference runner failed for {case.name}:\n{completed.stdout}\n{completed.stderr}"
        )
    with np.load(output_path, allow_pickle=False) as result:
        return np.asarray(result["point_radiance_w_m2_m_sr"], dtype=float)


def _comparison_report(
    case: GreyRTCase,
    robert_point: np.ndarray,
    picaso_point: np.ndarray,
    robert_disk: np.ndarray,
    picaso_disk: np.ndarray,
) -> dict[str, Any]:
    scale = np.maximum(np.abs(picaso_disk), np.finfo(float).tiny)
    relative = (robert_disk - picaso_disk) / scale
    point_scale = np.maximum(np.abs(picaso_point), np.finfo(float).tiny)
    point_relative = (robert_point - picaso_point) / point_scale
    max_relative = float(np.max(np.abs(relative)))
    passed = max_relative <= case.relative_tolerance
    return {
        "name": case.name,
        "single_scattering_albedo": case.single_scattering_albedo,
        "asymmetry_factor": case.asymmetry_factor,
        "column_extinction_tau": float(np.sum(case.extinction_tau[:, 0])),
        "relative_tolerance": case.relative_tolerance,
        "passed": passed,
        "median_relative_difference": float(np.median(relative)),
        "max_abs_relative_difference": max_relative,
        "max_abs_point_relative_difference": float(np.max(np.abs(point_relative))),
    }


def _disk_quadrature(n_mu: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    mu = 0.5 * (nodes + 1.0)
    disk_weights = 0.5 * weights * 2.0 * mu
    return mu, disk_weights / np.sum(disk_weights)


def _plot(
    output_path: Path,
    wavelength_micron: np.ndarray,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(len(spectra), 2, figsize=(12.5, 3.0 * len(spectra)), constrained_layout=True)
    for row, (name, (robert, picaso)) in enumerate(spectra.items()):
        ax_spectrum, ax_residual = axes[row]
        ax_spectrum.plot(wavelength_micron, picaso, color=REFERENCE_COLOR, label="PICASO Toon")
        ax_spectrum.plot(wavelength_micron, robert, color=ROBERT_COLOR, linestyle="--", label="ROBERT")
        ax_spectrum.set_xscale("log")
        ax_spectrum.set_yscale("log")
        ax_spectrum.set_ylabel("Radiance [W m$^{-2}$ m$^{-1}$ sr$^{-1}$]")
        ax_spectrum.set_title(name)
        ax_spectrum.grid(alpha=0.25, which="both")
        ax_spectrum.legend(frameon=False)
        relative = (robert - picaso) / np.maximum(np.abs(picaso), np.finfo(float).tiny)
        ax_residual.plot(wavelength_micron, 100.0 * relative, color=RESIDUAL_COLOR)
        ax_residual.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
        ax_residual.set_xscale("log")
        ax_residual.set_ylabel("(ROBERT - PICASO) / PICASO [%]")
        ax_residual.grid(alpha=0.25, which="both")
    axes[-1, 0].set_xlabel("Wavelength [micron]")
    axes[-1, 1].set_xlabel("Wavelength [micron]")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
