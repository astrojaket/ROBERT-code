"""Benchmark ROBERT's clear-sky emission solver against HAT-P-32b output."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    CompositionMeanMolecularWeight,
    CorrelatedKOpacityProvider,
    PressureGrid,
    SpectralGrid,
    TabulatedTemperatureProfile,
    assemble_gas_optical_depth,
    cia_optical_depth,
    disk_average_quadrature,
    load_emission_benchmark_csv,
    rayleigh_scattering_optical_depth,
    read_nemesis_cia_table,
    solve_clear_sky_emission,
)

DEFAULT_BENCHMARK_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "HAT-P-32b"
    / "emission"
    / "emission_R1000.csv"
)
DEFAULT_PT_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "PTprofiles-Teq_1800-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_1.8-TiOVO_false-daysideavg-w_mu_area.csv"
)
DEFAULT_KTA_DIR = Path.home() / "Dropbox" / "PostDoc4" / "Emission_Example" / "HAT-P-32b" / "kta_temp"
DEFAULT_CIA_FILE = (
    Path.home()
    / "Dropbox"
    / "NemesisPy-Docker"
    / "nemesispy"
    / "nemesispy"
    / "data"
    / "cia"
    / "exocia_hitran12_200-3800K.tab"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_emission_rt_benchmark"

R_SUN_M = 6.957e8
R_JUP_M = 7.1492e7

DEFAULT_SPECIES = ("H2O",)
DEFAULT_ACTIVE_VMR = {
    "H2O": 1.0e-2,
    "CO": 1.0e-4,
    "CO2": 1.0e-5,
    "CH4": 1.0e-6,
    "NH3": 1.0e-7,
    "HCN": 1.0e-7,
}
DEFAULT_H2_FRACTION_OF_BACKGROUND = 0.8547
DEFAULT_GRAVITY_M_S2 = 4.3
DEFAULT_PLANET_RADIUS_M = 1.98 * R_JUP_M
DEFAULT_STAR_RADIUS_M = 1.32 * R_SUN_M
DEFAULT_STAR_TEMPERATURE_K = 6001.0
RUNTIME_NONFINITE_FILL_VALUE = 1.0e-300

MISSING_PHYSICS = (
    "FastChem equilibrium abundance profiles for active gases",
    "scattering source-function treatment",
    "cloud and aerosol opacity/scattering",
    "NEMESIS path-geometry/layering parity",
)


def main() -> dict[str, object]:
    """Run the local HAT-P-32b clear-sky emission benchmark."""

    benchmark_csv = _path_from_env("HAT_P_32B_EMISSION_CSV", DEFAULT_BENCHMARK_CSV)
    pt_csv = _path_from_env("HAT_P_32B_PT_CSV", DEFAULT_PT_CSV)
    kta_dir = _path_from_env("HAT_P_32B_KTA_DIR", DEFAULT_KTA_DIR)
    cia_file = _path_from_env("HAT_P_32B_CIA_FILE", DEFAULT_CIA_FILE)
    species = _species_from_env(kta_dir)
    species_vmr = _species_vmr(species)
    gas_combination = os.environ.get("ROBERT_HAT_P_32B_GAS_COMBINATION", "random_overlap")
    include_cia = _env_bool("ROBERT_HAT_P_32B_INCLUDE_CIA", True)
    include_rayleigh = _env_bool("ROBERT_HAT_P_32B_INCLUDE_RAYLEIGH", True)

    for path, label in (
        (benchmark_csv, "benchmark CSV"),
        (pt_csv, "P-T CSV"),
        (kta_dir, "k-table directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"HAT-P-32b {label} was not found: {path}")

    benchmark = load_emission_benchmark_csv(benchmark_csv, name="HAT-P-32b emission R1000")
    kta_paths = _species_paths(kta_dir, species)
    n_mu = int(os.environ.get("ROBERT_HAT_P_32B_RT_NMU", "4"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Benchmark: HAT-P-32b clear-sky emission RT")
    print(f"Benchmark CSV: {benchmark_csv}")
    print(f"P-T CSV: {pt_csv}")
    print(f"K-tables: {', '.join(str(path) for path in kta_paths.values())}")
    print(f"Species VMRs: {species_vmr}, gas_combination={gas_combination}, n_mu={n_mu}")
    print(f"CIA: {'on' if include_cia else 'off'} ({cia_file})")
    print(f"Rayleigh: {'on' if include_rayleigh else 'off'}")

    start = perf_counter()
    provider = CorrelatedKOpacityProvider.from_kta_paths(
        kta_paths,
        interpolation="log_pressure_temperature_log_k",
        nonfinite_policy="floor",
        nonfinite_fill_value=RUNTIME_NONFINITE_FILL_VALUE,
    )
    table = provider.tables[species[0]]
    pressure_grid = _pressure_grid_from_centers(table.pressure_bar)
    spectral_grid = SpectralGrid.from_array(
        benchmark.wavelength_micron,
        unit="micron",
        role="opacity",
        name="HAT-P-32b emission R1000",
    )
    profile = TabulatedTemperatureProfile.from_csv(
        pt_csv,
        name="HAT-P-32b external PT",
    )
    temperature = profile.evaluate({}, pressure_grid)
    composition = _composition(pressure_grid, species_vmr)
    mean_molecular_weight = CompositionMeanMolecularWeight().evaluate(
        composition,
        pressure_grid,
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        composition=composition,
        mean_molecular_weight=mean_molecular_weight,
        metadata={
            "source": "HAT-P-32b local benchmark",
            "chemistry": "fixed active gases plus H2/He background",
        },
    )
    prepared = provider.prepare(spectral_grid, pressure_grid, species=species)
    evaluated = provider.evaluate(atmosphere, prepared)
    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=DEFAULT_GRAVITY_M_S2,
        gas_combination=gas_combination,
    )
    additional_optical_depths = []
    if include_cia:
        if not cia_file.exists():
            raise FileNotFoundError(f"HAT-P-32b CIA file was not found: {cia_file}")
        cia_table = read_nemesis_cia_table(cia_file)
        additional_optical_depths.append(cia_optical_depth(gas_tau, cia_table))
    if include_rayleigh:
        additional_optical_depths.append(rayleigh_scattering_optical_depth(gas_tau))
    mu, weights = disk_average_quadrature(n_mu)
    result = solve_clear_sky_emission(
        gas_tau,
        emission_angle_cosines=mu,
        emission_angle_weights=weights,
        additional_optical_depths=additional_optical_depths,
        planet_radius_m=DEFAULT_PLANET_RADIUS_M,
        star_radius_m=DEFAULT_STAR_RADIUS_M,
        star_temperature_k=DEFAULT_STAR_TEMPERATURE_K,
    )
    runtime_s = perf_counter() - start
    if result.eclipse_depth is None:
        raise RuntimeError("clear-sky emission result did not include eclipse depth")

    comparison = _comparison_metrics(
        model=result.eclipse_depth.values,
        benchmark=benchmark.eclipse_depth,
    )
    plot_path = _plot_benchmark(benchmark, result, temperature, pressure_grid)
    summary_path = OUTPUT_DIR / "hat_p_32b_emission_rt_benchmark_summary.json"
    summary = {
        "benchmark": "HAT-P-32b clear-sky emission RT",
        "status": "diagnostic_not_strict",
        "runtime_s": runtime_s,
        "benchmark_csv": str(benchmark_csv),
        "pt_csv": str(pt_csv),
        "kta_paths": {item: str(path) for item, path in kta_paths.items()},
        "cia_file": str(cia_file) if include_cia else None,
        "species": list(species),
        "active_vmr": species_vmr,
        "gas_combination": gas_tau.metadata["gas_combination"],
        "additional_optical_depths": _optical_depth_summary(additional_optical_depths),
        "n_layers": atmosphere.n_layers,
        "n_wavelength": benchmark.n_points,
        "n_mu": n_mu,
        "planet_radius_m": DEFAULT_PLANET_RADIUS_M,
        "star_radius_m": DEFAULT_STAR_RADIUS_M,
        "star_temperature_k": DEFAULT_STAR_TEMPERATURE_K,
        "gravity_m_s2": DEFAULT_GRAVITY_M_S2,
        "solver": result.metadata,
        "missing_physics_relative_to_mature_nemesis": list(MISSING_PHYSICS),
        "comparison": comparison,
        "outputs": {
            "summary_json": str(summary_path),
            "plot_png": str(plot_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ROBERT median depth: {comparison['model_median_ppm']:.2f} ppm")
    print(f"Benchmark median depth: {comparison['benchmark_median_ppm']:.2f} ppm")
    print(f"Residual RMSE: {comparison['rmse_ppm']:.2f} ppm")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")
    return summary


def _path_from_env(name: str, default: Path) -> Path:
    configured = os.environ.get(name)
    if configured:
        return Path(configured).expanduser()
    return default


def _species_path(kta_dir: Path, species: str) -> Path:
    candidates = sorted(kta_dir.glob(f"{species}_*.kta"))
    if not candidates:
        raise FileNotFoundError(f"No {species} .kta file found in {kta_dir}")
    return candidates[0]


def _species_paths(kta_dir: Path, species: tuple[str, ...]) -> dict[str, Path]:
    return {item: _species_path(kta_dir, item) for item in species}


def _species_from_env(kta_dir: Path) -> tuple[str, ...]:
    configured = os.environ.get("ROBERT_HAT_P_32B_RT_SPECIES")
    if configured is None or not configured.strip():
        return DEFAULT_SPECIES
    normalized = configured.strip()
    if normalized.lower() == "all":
        species = tuple(sorted(path.name.split("_", maxsplit=1)[0] for path in kta_dir.glob("*.kta")))
    else:
        species = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not species:
        raise ValueError("ROBERT_HAT_P_32B_RT_SPECIES did not contain any species")
    return species


def _species_vmr(species: tuple[str, ...]) -> dict[str, float]:
    vmr = {}
    for item in species:
        default = DEFAULT_ACTIVE_VMR.get(item, 1.0e-12)
        value = float(os.environ.get(f"ROBERT_HAT_P_32B_{item}_VMR", str(default)))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{item} VMR must be finite and non-negative")
        vmr[item] = value
    total = sum(vmr.values())
    if total >= 1.0:
        raise ValueError("active gas VMRs must sum to less than one")
    return vmr


def _composition(
    pressure_grid: PressureGrid,
    species_vmr: dict[str, float],
) -> dict[str, np.ndarray]:
    background = 1.0 - sum(species_vmr.values())
    h2 = background * DEFAULT_H2_FRACTION_OF_BACKGROUND
    he = background * (1.0 - DEFAULT_H2_FRACTION_OF_BACKGROUND)
    composition = {
        "H2": np.full(pressure_grid.n_layers, h2),
        "He": np.full(pressure_grid.n_layers, he),
    }
    for species, vmr in species_vmr.items():
        composition[species] = np.full(pressure_grid.n_layers, vmr)
    return composition


def _env_bool(name: str, default: bool) -> bool:
    configured = os.environ.get(name)
    if configured is None:
        return default
    return configured.strip().lower() not in {"0", "false", "no", "off"}


def _optical_depth_summary(optical_depths) -> dict[str, dict[str, object]]:
    summary = {}
    for optical_depth in optical_depths:
        summary[optical_depth.name] = {
            "kind": optical_depth.kind,
            "max_tau": float(np.max(optical_depth.tau)),
            "median_tau": float(np.median(optical_depth.tau)),
            "metadata": dict(optical_depth.metadata),
        }
    return summary


def _comparison_metrics(
    *,
    model: np.ndarray,
    benchmark: np.ndarray,
) -> dict[str, float]:
    model_ppm = np.asarray(model, dtype=float) * 1.0e6
    benchmark_ppm = np.asarray(benchmark, dtype=float) * 1.0e6
    residual = model_ppm - benchmark_ppm
    denominator = np.maximum(np.abs(benchmark_ppm), 1.0e-12)
    relative = residual / denominator
    return {
        "model_median_ppm": float(np.median(model_ppm)),
        "benchmark_median_ppm": float(np.median(benchmark_ppm)),
        "residual_median_ppm": float(np.median(residual)),
        "residual_max_abs_ppm": float(np.max(np.abs(residual))),
        "residual_median_abs_ppm": float(np.median(np.abs(residual))),
        "rmse_ppm": float(np.sqrt(np.mean(residual**2))),
        "relative_median_abs": float(np.median(np.abs(relative))),
        "relative_p95_abs": float(np.percentile(np.abs(relative), 95.0)),
    }


def _plot_benchmark(
    benchmark,
    result,
    temperature: np.ndarray,
    pressure_grid: PressureGrid,
) -> Path:
    output_path = OUTPUT_DIR / "hat_p_32b_clear_sky_emission_rt.png"
    model_ppm = result.eclipse_depth.values * 1.0e6
    benchmark_ppm = benchmark.eclipse_depth * 1.0e6
    residual_ppm = model_ppm - benchmark_ppm
    contribution = result.normalized_layer_contribution()
    mean_contribution = np.mean(contribution, axis=1)
    if np.max(mean_contribution) > 0.0:
        mean_contribution = mean_contribution / np.max(mean_contribution)

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[2.0, 1.0])
    ax_spectrum = fig.add_subplot(grid[0, 0])
    ax_residual = fig.add_subplot(grid[1, 0], sharex=ax_spectrum)
    ax_profile = fig.add_subplot(grid[:, 1])

    ax_spectrum.plot(
        benchmark.wavelength_micron,
        benchmark_ppm,
        color="#111111",
        linewidth=1.6,
        label="NemesisPy benchmark",
    )
    ax_spectrum.plot(
        result.eclipse_depth.spectral_grid.values,
        model_ppm,
        color="#d62728",
        linewidth=1.4,
        label="ROBERT clear-sky + CIA/Rayleigh",
    )
    for label, values in benchmark.references.items():
        ax_spectrum.plot(
            benchmark.wavelength_micron,
            values * 1.0e6,
            linestyle="--",
            linewidth=1.0,
            label=label.replace("_", " "),
        )
    ax_spectrum.set_ylabel("Eclipse Depth [ppm]")
    ax_spectrum.set_title("HAT-P-32b Emission Benchmark")
    ax_spectrum.grid(alpha=0.25)
    ax_spectrum.legend(frameon=False, fontsize=8.5)

    ax_residual.axhline(0.0, color="#333333", linewidth=1.0)
    ax_residual.plot(
        benchmark.wavelength_micron,
        residual_ppm,
        color="#1f77b4",
        linewidth=1.1,
    )
    ax_residual.set_xlabel("Wavelength [micron]")
    ax_residual.set_ylabel("ROBERT - benchmark [ppm]")
    ax_residual.grid(alpha=0.25)

    pressure = pressure_grid.centers
    ax_profile.plot(
        temperature,
        pressure,
        color="#111111",
        linewidth=1.8,
        label="P-T profile",
    )
    ax_profile.set_yscale("log")
    ax_profile.set_ylim(float(np.max(pressure)), float(np.min(pressure)))
    ax_profile.set_xlabel("Temperature [K]")
    ax_profile.set_ylabel("Pressure [bar]")
    ax_profile.set_title("Mean Emission Contribution")
    ax_profile.grid(alpha=0.25, which="both")
    ax_weight = ax_profile.twiny()
    ax_weight.fill_betweenx(
        pressure,
        0.0,
        mean_contribution,
        color="#2a9d8f",
        alpha=0.25,
    )
    ax_weight.plot(
        mean_contribution,
        pressure,
        color="#2a9d8f",
        linewidth=1.7,
        label="Mean contribution",
    )
    ax_weight.set_xlim(0.0, 1.05)
    ax_weight.set_xlabel("Normalized Contribution")
    lines = ax_profile.get_lines() + ax_weight.get_lines()
    labels = [line.get_label() for line in lines]
    ax_profile.legend(lines, labels, frameon=False, loc="lower right", fontsize=8.5)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _pressure_grid_from_centers(centers: np.ndarray) -> PressureGrid:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 1:
        raise ValueError("centers must be a non-empty one-dimensional array")
    if centers.size == 1:
        edges = np.array([centers[0] / np.sqrt(10.0), centers[0] * np.sqrt(10.0)], dtype=float)
    else:
        log_centers = np.log(centers)
        inner_edges = 0.5 * (log_centers[:-1] + log_centers[1:])
        first_edge = log_centers[0] - (inner_edges[0] - log_centers[0])
        last_edge = log_centers[-1] + (log_centers[-1] - inner_edges[-1])
        edges = np.exp(np.concatenate(([first_edge], inner_edges, [last_edge])))
    return PressureGrid(edges=edges, centers=centers, unit="bar", name="HAT-P-32b emission RT benchmark")


if __name__ == "__main__":
    main()
