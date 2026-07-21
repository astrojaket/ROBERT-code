"""End-to-end MgSiO3 cloud parity between ROBERT and PICASO/Virga.

Both frameworks receive only the same physical contract: atmospheric state,
gas composition, measured refractive index, discretized particle population,
condensate profile, geometry, and body parameters. Each framework separately
computes gas optical depth, Mie cloud optics, layer cloud optical depth, and the
four-stream thermal spectrum. The analytic gas-opacity contract is a validation
fixture, not a science opacity model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    OpticalConstantsCatalog,
    SpectralGrid,
    mie_efficiencies,
    mie_phase_function_moments,
    planck_radiance_wavelength,
)
from robert_exoplanets.rt.sh4 import solve_thermal_sh4
from robert_exoplanets.diagnostics.benchmark_style import (
    PURPLE_DARK,
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "end_to_end_cloud_parity"
EXTERNAL_RUNNER = Path(__file__).with_name("run_picaso_virga_cloud_parity.py")
OPTICAL_CONSTANTS = Path(
    os.environ.get(
        "ROBERT_OPTICAL_CONSTANTS",
        ROOT / "data" / "optical_constants" / "exo_skryer",
    )
).expanduser()
GAS_SPECIES = ("H2O", "CO", "CO2", "CH4")


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-wavelength", type=int, default=96)
    parser.add_argument("--n-radius", type=int, default=36)
    parser.add_argument("--n-layer", type=int, default=72)
    args = parser.parse_args()
    report = run(
        args.picaso_python,
        args.output_dir,
        n_wavelength=args.n_wavelength,
        n_radius=args.n_radius,
        n_layer=args.n_layer,
    )
    print(json.dumps(report, indent=2))
    return report


def run(
    picaso_python: Path,
    output_dir: Path,
    *,
    n_wavelength: int = 96,
    n_radius: int = 36,
    n_layer: int = 72,
) -> dict[str, Any]:
    if n_wavelength < 16 or n_radius < 8 or n_layer < 8:
        raise ValueError("benchmark resolution is too small")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _make_contract(n_wavelength, n_radius, n_layer)
    contract_path = output_dir / "shared_physical_contract.npz"
    np.savez_compressed(contract_path, **contract)

    robert = _evaluate_robert(contract)
    external_path = output_dir / "picaso_virga_independent_output.npz"
    _run_external(picaso_python, contract_path, external_path)
    with np.load(external_path, allow_pickle=False) as archive:
        external = {name: np.array(archive[name], copy=True) for name in archive.files}

    metrics = _metrics(contract, robert, external)
    report = {
        "schema_version": 1,
        "benchmark": "independent_end_to_end_MgSiO3_cloud_emission_parity",
        "scope": {
            "shared": [
                "pressure_temperature_and_gas_mass_fraction_state",
                "analytic_validation_opacity_specification_not_optical_depth",
                "measured_MgSiO3_refractive_index_values",
                "discrete_lognormal_particle_population",
                "vertical_condensate_mass_fraction_profile",
                "gravity_geometry_and_body_parameters",
            ],
            "independent": [
                "gas_optical_depth_assembly",
                "Mie_efficiency_solution",
                "particle_population_integration",
                "cloud_layer_optical_depth",
                "SH4_thermal_radiative_transfer",
            ],
            "gas_opacity_warning": "analytic validation fixture; not retrieval or science opacity",
            "matched_phase_closure": "four_term_Henyey_Greenstein_without_delta_M",
        },
        "provenance": {
            "optical_constants": str(
                (OPTICAL_CONSTANTS / "MgSiO3.txt").relative_to(ROOT)
            ),
            "optical_constants_sha256": _sha256(OPTICAL_CONSTANTS / "MgSiO3.txt"),
            "external_metadata": json.loads(str(external["metadata_json"])),
            "robert_mie_solver": "robert_exoplanets.mie_efficiencies",
            "robert_rt_solver": "robert_exoplanets.rt.sh4.solve_thermal_sh4",
        },
        "resolution": {
            "n_layers": n_layer,
            "n_wavelength": n_wavelength,
            "n_particle_radius_bins": n_radius,
            "n_mie_subradii_per_bin": 6,
            "n_emission_angles": int(contract["emission_mu"].size),
        },
        "physical_contract": {
            "wavelength_micron": [
                float(contract["wavelength_micron"][0]),
                float(contract["wavelength_micron"][-1]),
            ],
            "pressure_bar": [
                float(contract["pressure_edges_bar"][0]),
                float(contract["pressure_edges_bar"][-1]),
            ],
            "temperature_k": [
                float(np.min(contract["temperature_level_k"])),
                float(np.max(contract["temperature_level_k"])),
            ],
            "gravity_m_s2": float(contract["gravity_m_s2"]),
            "particle_density_kg_m3": float(contract["particle_density_kg_m3"]),
            "particle_effective_radius_micron": float(
                np.sum(contract["radius_number_weights"] * contract["radius_cm"] ** 3)
                / np.sum(contract["radius_number_weights"] * contract["radius_cm"] ** 2)
                * 1.0e4
            ),
            "particle_geometric_stddev": 1.6,
            "peak_condensate_mass_fraction": float(
                np.max(contract["condensate_mass_fraction"])
            ),
            "gas_species": list(GAS_SPECIES),
        },
        "metrics": metrics,
    }
    report_path = output_dir / "end_to_end_cloud_parity.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "robert_independent_output.npz", **robert)
    _plot(
        output_dir / "end_to_end_cloud_parity.png", contract, robert, external, metrics
    )
    return report


def _make_contract(
    n_wavelength: int, n_radius: int, n_layer: int
) -> dict[str, np.ndarray]:
    wavelength = np.geomspace(1.0, 12.0, n_wavelength)
    spectral_grid = SpectralGrid.from_array(wavelength, unit="micron", role="benchmark")
    refractive = OpticalConstantsCatalog(OPTICAL_CONSTANTS).load("MgSiO3")
    index = refractive.evaluate(spectral_grid.values)

    radius_cm = np.geomspace(5.0e-7, 5.0e-4, n_radius)
    ratio = radius_cm[1] / radius_cm[0]
    radius_upper_cm = 2.0 * ratio / (ratio + 1.0) * radius_cm
    effective_radius_cm = 3.0e-5
    sigma = 1.6
    geometric_mean_cm = effective_radius_cm * np.exp(-2.5 * np.log(sigma) ** 2)
    number_weights = np.exp(
        -0.5 * (np.log(radius_cm / geometric_mean_cm) / np.log(sigma)) ** 2
    )
    number_weights /= np.sum(number_weights)

    pressure_edges = np.geomspace(1.0e-5, 10.0, n_layer + 1)
    log_fraction = np.linspace(0.0, 1.0, n_layer + 1)
    temperature_level = 780.0 + 820.0 * log_fraction**0.72
    pressure_layer = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    condensate = 1.2e-5 * np.exp(-0.5 * ((np.log10(pressure_layer) + 1.7) / 0.85) ** 2)
    base_mass_fractions = np.array([8.0e-4, 3.0e-4, 2.0e-5, 4.0e-6])
    gas_mass_fractions = np.repeat(base_mass_fractions[None, :], n_layer, axis=0)
    gas_mass_fractions[:, 0] *= (pressure_layer / 0.1) ** 0.025

    mu, weights = _disk_quadrature(6)
    return {
        "contract_schema_version": np.array(1),
        "wavelength_micron": wavelength,
        "refractive_index_n": np.real(index),
        "refractive_index_k": np.imag(index),
        "radius_cm": radius_cm,
        "radius_upper_cm": radius_upper_cm,
        "radius_number_weights": number_weights,
        "pressure_edges_bar": pressure_edges,
        "temperature_level_k": temperature_level,
        "condensate_mass_fraction": condensate,
        "gas_species": np.asarray(GAS_SPECIES),
        "gas_mass_fractions": gas_mass_fractions,
        "gravity_m_s2": np.array(8.42),
        "particle_density_kg_m3": np.array(3200.0),
        "emission_mu": mu,
        "emission_weights": weights,
        "planet_radius_m": np.array(7.1492e7 * 1.057),
        "star_radius_m": np.array(6.957e8 * 0.813),
        "star_temperature_k": np.array(4715.0),
    }


def _evaluate_robert(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    wavelength = contract["wavelength_micron"]
    n_wave = wavelength.size
    radius = contract["radius_cm"]
    upper = contract["radius_upper_cm"]
    qext = np.zeros((n_wave, radius.size))
    qsca = np.zeros_like(qext)
    g_qsca = np.zeros_like(qext)
    phase_numerator = np.zeros((5, n_wave))
    weights = contract["radius_number_weights"]
    radius_m = radius * 1.0e-2
    area_weight = weights * np.pi * radius_m**2
    for i_wave, (wave, real, imaginary) in enumerate(
        zip(
            wavelength,
            contract["refractive_index_n"],
            contract["refractive_index_k"],
            strict=True,
        )
    ):
        for i_radius in range(radius.size):
            start = radius[0] if i_radius == 0 else upper[i_radius - 1]
            subradii = np.linspace(start, upper[i_radius], 6)
            values = []
            moment_values = []
            for subradius in subradii:
                size = 2.0 * np.pi * (subradius * 1.0e4) / wave
                q_ext, q_sca, asymmetry = mie_efficiencies(
                    size, complex(real, imaginary)
                )
                values.append((q_ext, q_sca, asymmetry * q_sca))
                moment_values.append(
                    q_sca
                    * mie_phase_function_moments(
                        size, complex(real, imaginary), order=5
                    )
                )
            qext[i_wave, i_radius], qsca[i_wave, i_radius], g_qsca[i_wave, i_radius] = (
                np.mean(values, axis=0)
            )
            phase_numerator[:, i_wave] += area_weight[i_radius] * np.mean(
                moment_values, axis=0
            )

    population = _integrate_population(
        qext,
        qsca,
        g_qsca,
        radius,
        weights,
        float(contract["particle_density_kg_m3"]),
    )
    exact_moments = np.divide(
        phase_numerator,
        population["area_scattering_m2"][None, :],
        out=np.zeros_like(phase_numerator),
        where=population["area_scattering_m2"][None, :] > 0.0,
    )
    exact_moments[0] = 1.0

    gas_tau = _validation_gas_tau(contract)
    layer_mass = (
        np.diff(contract["pressure_edges_bar"])
        * 1.0e5
        / float(contract["gravity_m_s2"])
    )
    cloud_tau = (
        layer_mass[:, None]
        * contract["condensate_mass_fraction"][:, None]
        * population["mass_extinction_m2_kg"][None, :]
    )
    scattering_tau = cloud_tau * population["single_scattering_albedo"][None, :]
    total_tau = gas_tau + cloud_tau
    omega = np.divide(
        scattering_tau,
        total_tau,
        out=np.zeros_like(total_tau),
        where=total_tau > 0.0,
    )
    asymmetry = np.repeat(
        population["asymmetry_factor"][None, :], cloud_tau.shape[0], axis=0
    )
    level_planck = np.vstack(
        [
            planck_radiance_wavelength(wavelength, temperature)
            for temperature in contract["temperature_level_k"]
        ]
    )
    cloudy = solve_thermal_sh4(
        total_tau[:, :, None],
        omega[:, :, None],
        asymmetry[:, :, None],
        level_planck,
        contract["emission_mu"],
        bottom_planck_radiance=level_planck[-1],
        delta_m=False,
        source_quadrature_order=8,
    )
    clear = solve_thermal_sh4(
        gas_tau[:, :, None],
        np.zeros_like(gas_tau[:, :, None]),
        np.zeros_like(gas_tau[:, :, None]),
        level_planck,
        contract["emission_mu"],
        bottom_planck_radiance=level_planck[-1],
        delta_m=False,
        source_quadrature_order=8,
    )
    native_moments = np.repeat(
        exact_moments[:4, None, :, None], cloud_tau.shape[0], axis=1
    )
    native = solve_thermal_sh4(
        total_tau[:, :, None],
        omega[:, :, None],
        asymmetry[:, :, None],
        level_planck,
        contract["emission_mu"],
        bottom_planck_radiance=level_planck[-1],
        phase_function_moments=native_moments,
        delta_m_forward_fraction=np.repeat(
            (exact_moments[4] / 9.0)[None, :, None], cloud_tau.shape[0], axis=0
        ),
        delta_m=True,
        source_quadrature_order=8,
    )
    cloudy_point = np.asarray(cloudy.point_radiance[:, :, 0])
    cloud_free_point = np.asarray(clear.point_radiance[:, :, 0])
    native_point = np.asarray(native.point_radiance[:, :, 0])
    cloudy_disk = np.tensordot(contract["emission_weights"], cloudy_point, axes=(0, 0))
    cloud_free_disk = np.tensordot(contract["emission_weights"], cloud_free_point, axes=(0, 0))
    native_disk = np.tensordot(contract["emission_weights"], native_point, axes=(0, 0))
    stellar = planck_radiance_wavelength(
        wavelength, float(contract["star_temperature_k"])
    )
    area_ratio = (
        float(contract["planet_radius_m"]) / float(contract["star_radius_m"])
    ) ** 2
    return {
        "mie_qext": qext,
        "mie_qsca": qsca,
        "mie_g_qsca": g_qsca,
        "mass_extinction_m2_kg": population["mass_extinction_m2_kg"],
        "mass_scattering_m2_kg": population["mass_scattering_m2_kg"],
        "single_scattering_albedo": population["single_scattering_albedo"],
        "asymmetry_factor": population["asymmetry_factor"],
        "exact_phase_function_moments": exact_moments,
        "gas_tau": gas_tau,
        "cloud_tau": cloud_tau,
        "total_tau": total_tau,
        "cloudy_point_radiance_w_m2_m_sr": cloudy_point,
        "cloud_free_point_radiance_w_m2_m_sr": cloud_free_point,
        "native_point_radiance_w_m2_m_sr": native_point,
        "cloudy_disk_radiance_w_m2_m_sr": cloudy_disk,
        "cloud_free_disk_radiance_w_m2_m_sr": cloud_free_disk,
        "native_disk_radiance_w_m2_m_sr": native_disk,
        "cloudy_eclipse_depth": cloudy_disk / stellar * area_ratio,
        "cloud_free_eclipse_depth": cloud_free_disk / stellar * area_ratio,
        "native_eclipse_depth": native_disk / stellar * area_ratio,
    }


def _integrate_population(
    qext: np.ndarray,
    qsca: np.ndarray,
    g_qsca: np.ndarray,
    radius_cm: np.ndarray,
    weights: np.ndarray,
    density: float,
) -> dict[str, np.ndarray]:
    radius_m = radius_cm * 1.0e-2
    area = np.pi * radius_m**2
    mass = (4.0 / 3.0) * np.pi * density * radius_m**3
    mean_mass = np.sum(weights * mass)
    extinction = np.sum(qext * (weights * area)[None, :], axis=1)
    scattering = np.sum(qsca * (weights * area)[None, :], axis=1)
    scattering_g = np.sum(g_qsca * (weights * area)[None, :], axis=1)
    mass_extinction = extinction / mean_mass
    mass_scattering = scattering / mean_mass
    return {
        "area_scattering_m2": scattering,
        "mass_extinction_m2_kg": mass_extinction,
        "mass_scattering_m2_kg": mass_scattering,
        "single_scattering_albedo": np.divide(
            mass_scattering,
            mass_extinction,
            out=np.zeros_like(mass_extinction),
            where=mass_extinction > 0.0,
        ),
        "asymmetry_factor": np.divide(
            scattering_g,
            scattering,
            out=np.zeros_like(scattering_g),
            where=scattering > 0.0,
        ),
    }


def _validation_gas_tau(contract: dict[str, np.ndarray]) -> np.ndarray:
    """ROBERT-side implementation of analytic validation opacity contract v1."""

    wavelength = contract["wavelength_micron"]
    pressure_edges = contract["pressure_edges_bar"]
    pressure = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    temperature = 0.5 * (
        contract["temperature_level_k"][:-1] + contract["temperature_level_k"][1:]
    )
    bands = {
        "H2O": ((1.4, 0.13, 1.7), (1.9, 0.12, 1.5), (2.7, 0.15, 2.4), (6.3, 0.18, 3.2)),
        "CO": ((4.7, 0.09, 2.8),),
        "CO2": ((4.3, 0.08, 3.5), (9.4, 0.15, 0.7)),
        "CH4": ((3.3, 0.10, 2.5), (7.7, 0.14, 2.0)),
    }
    baseline = {"H2O": 0.12, "CO": 0.015, "CO2": 0.02, "CH4": 0.02}
    scale = np.clip(1.0 + 0.06 * np.log10(pressure / 0.1), 0.65, 1.35)
    scale *= (temperature / 1200.0) ** 0.3
    opacity = np.zeros((pressure.size, wavelength.size))
    for i_species, name in enumerate(GAS_SPECIES):
        spectrum = np.full(wavelength.size, baseline[name])
        for center, width, strength in bands[name]:
            spectrum += strength * np.exp(
                -0.5 * ((np.log(wavelength / center)) / width) ** 2
            )
        opacity += (
            contract["gas_mass_fractions"][:, i_species, None]
            * scale[:, None]
            * spectrum
        )
    opacity += (
        2.0e-4
        * pressure[:, None]
        * (1200.0 / temperature[:, None]) ** 0.5
        * (wavelength[None, :] / 2.0) ** 0.25
    )
    layer_mass = np.diff(pressure_edges) * 1.0e5 / float(contract["gravity_m_s2"])
    return layer_mass[:, None] * opacity


def _metrics(
    contract: dict[str, np.ndarray],
    robert: dict[str, np.ndarray],
    external: dict[str, np.ndarray],
) -> dict[str, Any]:
    metrics = {
        "mie_qext": _relative_metrics(robert["mie_qext"], external["mie_qext"]),
        "mie_qsca": _relative_metrics(robert["mie_qsca"], external["mie_qsca"]),
        "mie_g_qsca": _absolute_metrics(robert["mie_g_qsca"], external["mie_g_qsca"]),
        "mass_extinction": _relative_metrics(
            robert["mass_extinction_m2_kg"], external["mass_extinction_m2_kg"]
        ),
        "single_scattering_albedo": _absolute_metrics(
            robert["single_scattering_albedo"], external["single_scattering_albedo"]
        ),
        "asymmetry_factor": _absolute_metrics(
            robert["asymmetry_factor"], external["asymmetry_factor"]
        ),
        "gas_tau": _relative_metrics(robert["gas_tau"], external["gas_tau"]),
        "cloud_tau": _relative_metrics(robert["cloud_tau"], external["cloud_tau"]),
        "matched_hg_cloudy_disk_spectrum": _relative_metrics(
            robert["cloudy_disk_radiance_w_m2_m_sr"],
            external["cloudy_disk_radiance_w_m2_m_sr"],
        ),
        "matched_hg_cloud_free_disk_spectrum": _relative_metrics(
            robert["cloud_free_disk_radiance_w_m2_m_sr"],
            external["cloud_free_disk_radiance_w_m2_m_sr"],
        ),
        "matched_hg_cloudy_eclipse_depth_ppm": _absolute_metrics(
            robert["cloudy_eclipse_depth"] * 1.0e6,
            external["cloudy_eclipse_depth"] * 1.0e6,
        ),
        "native_exact_mie_delta_m_vs_picaso_hg": _relative_metrics(
            robert["native_disk_radiance_w_m2_m_sr"],
            external["cloudy_disk_radiance_w_m2_m_sr"],
        ),
    }
    cloud_effect_robert = robert["cloudy_eclipse_depth"] - robert["cloud_free_eclipse_depth"]
    cloud_effect_picaso = (
        external["cloudy_eclipse_depth"] - external["cloud_free_eclipse_depth"]
    )
    metrics["cloud_effect_eclipse_depth_ppm"] = {
        "robert_max_abs": float(np.max(np.abs(cloud_effect_robert)) * 1.0e6),
        "picaso_virga_max_abs": float(np.max(np.abs(cloud_effect_picaso)) * 1.0e6),
        **_absolute_metrics(cloud_effect_robert * 1.0e6, cloud_effect_picaso * 1.0e6),
    }
    metrics["acceptance"] = {
        "gas_tau_max_relative_lt_1e-12": bool(
            metrics["gas_tau"]["max_abs_relative_difference"] < 1.0e-12
        ),
        "mass_extinction_rms_relative_lt_0p02": bool(
            metrics["mass_extinction"]["rms_relative_difference"] < 0.02
        ),
        "cloudy_disk_rms_relative_lt_0p01": bool(
            metrics["matched_hg_cloudy_disk_spectrum"]["rms_relative_difference"] < 0.01
        ),
        "cloud_effect_max_abs_difference_lt_10ppm": bool(
            metrics["cloud_effect_eclipse_depth_ppm"]["max_abs_difference"] < 10.0
        ),
    }
    metrics["acceptance"]["all_pass"] = bool(all(metrics["acceptance"].values()))
    return metrics


def _relative_metrics(
    left: np.ndarray, right: np.ndarray, *, floor: float | None = None
) -> dict[str, float]:
    scale_floor = np.finfo(float).tiny if floor is None else floor
    relative = (np.asarray(left) - np.asarray(right)) / np.maximum(
        np.abs(right), scale_floor
    )
    return {
        "max_abs_relative_difference": float(np.max(np.abs(relative))),
        "rms_relative_difference": float(np.sqrt(np.mean(relative**2))),
        "median_relative_difference": float(np.median(relative)),
    }


def _absolute_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = np.asarray(left) - np.asarray(right)
    return {
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
        "median_difference": float(np.median(difference)),
    }


def _run_external(python: Path, contract: Path, output: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "picaso-matplotlib")
    )
    environment.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "picaso-numba-cache")
    )
    completed = subprocess.run(
        [str(python), str(EXTERNAL_RUNNER), str(contract), str(output)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "PICASO/Virga independent runner failed:\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _disk_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mu = 0.5 * (nodes + 1.0)
    disk_weights = weights * mu
    return mu, disk_weights / np.sum(disk_weights)


def _plot(
    path: Path,
    contract: dict[str, np.ndarray],
    robert: dict[str, np.ndarray],
    external: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    wavelength = contract["wavelength_micron"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    ax_mie, ax_tau, ax_spectrum, ax_residual = axes.flat

    ax_mie.plot(
        wavelength,
        robert["mass_extinction_m2_kg"],
        color=ROBERT_COLOR,
        label="ROBERT",
        lw=2,
    )
    ax_mie.plot(
        wavelength,
        external["mass_extinction_m2_kg"],
        color=REFERENCE_COLOR,
        label="Virga/PyMieScatt",
        lw=1.5,
        ls="--",
    )
    ax_mie.set(
        xscale="log",
        yscale="log",
        ylabel="Mass extinction [m² kg⁻¹]",
        title="Independent MgSiO3 Mie optics",
    )
    ax_mie.legend(frameon=False)

    ax_tau.plot(
        wavelength,
        np.sum(robert["cloud_tau"], axis=0),
        color=ROBERT_COLOR,
        label="ROBERT",
        lw=2,
    )
    ax_tau.plot(
        wavelength,
        np.sum(external["cloud_tau"], axis=0),
        color=REFERENCE_COLOR,
        label="Virga",
        lw=1.5,
        ls="--",
    )
    ax_tau.set(
        xscale="log",
        yscale="log",
        ylabel="Column cloud optical depth",
        title="Independently assembled vertical cloud",
    )

    ax_spectrum.plot(
        wavelength,
        robert["cloudy_eclipse_depth"] * 1.0e6,
        color=ROBERT_COLOR,
        label="ROBERT HG-SH4",
        lw=2,
    )
    ax_spectrum.plot(
        wavelength,
        external["cloudy_eclipse_depth"] * 1.0e6,
        color=REFERENCE_COLOR,
        label="PICASO/Virga HG-SH4",
        lw=1.5,
        ls="--",
    )
    ax_spectrum.plot(
        wavelength,
        robert["native_eclipse_depth"] * 1.0e6,
        color=PURPLE_DARK,
        label="ROBERT exact-Mie + delta-M",
        lw=1.2,
        alpha=0.8,
    )
    ax_spectrum.set(
        xscale="log", ylabel="Eclipse depth [ppm]", title="End-to-end cloudy emission"
    )
    ax_spectrum.legend(frameon=False)

    residual = (
        robert["cloudy_disk_radiance_w_m2_m_sr"]
        - external["cloudy_disk_radiance_w_m2_m_sr"]
    ) / external["cloudy_disk_radiance_w_m2_m_sr"]
    ax_residual.plot(wavelength, residual * 100.0, color=RESIDUAL_COLOR, lw=1.6)
    ax_residual.axhline(0.0, color=REFERENCE_COLOR, lw=0.8)
    rms = metrics["matched_hg_cloudy_disk_spectrum"]["rms_relative_difference"] * 100.0
    ax_residual.text(
        0.03, 0.94, f"RMS = {rms:.6f}%", transform=ax_residual.transAxes, va="top"
    )
    ax_residual.set(
        xscale="log",
        ylabel="(ROBERT - PICASO) / PICASO [%]",
        title="Matched-closure spectral residual",
    )

    for axis in axes.flat:
        axis.set_xlabel("Wavelength [micron]")
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
