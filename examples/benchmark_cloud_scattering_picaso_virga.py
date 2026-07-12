"""Benchmark cloud-scattering RT hooks against PICASO/Virga-style inputs."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from robert_exoplanets import (
    AtmosphereState,
    CloudOpticalProperties,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    compare_cloud_optical_properties,
    gauss_legendre_disk_geometry,
    load_cloud_optical_properties_csv,
    load_cloud_optical_properties_npz,
    load_picaso_cloud_optical_properties,
    solve_clear_sky_emission,
    time_callable,
    write_cloud_optical_properties_npz,
)
from robert_exoplanets.opacity import pressure_values_in_unit, spectral_grid_values_in_unit

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cloud_scattering_benchmark"


def main() -> dict[str, Any]:
    """Run the cloud-scattering benchmark and write plots plus JSON metrics."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cloud_path = _cloud_property_path()
    cloud = _load_cloud_file(cloud_path)
    gas_tau = _zero_gas_tau(cloud)
    geometry = gauss_legendre_disk_geometry(4)

    clear = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="blackbody",
        thermal_integration_backend="auto",
    )
    extinction_numpy = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="blackbody",
        additional_optical_depths=[cloud],
        thermal_integration_backend="numpy",
    )
    extinction_auto = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="blackbody",
        additional_optical_depths=[cloud],
        thermal_integration_backend="auto",
    )
    two_stream_auto = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        bottom_boundary="blackbody",
        additional_optical_depths=[cloud],
        multiple_scattering_backend="two_stream",
        thermal_integration_backend="auto",
    )

    repeat = int(os.environ.get("ROBERT_CLOUD_BENCHMARK_REPEAT", "20"))
    timings = _timings(cloud_path, gas_tau, geometry, cloud, repeat=repeat)
    external_spectrum = _optional_external_spectrum()
    report = _report(
        cloud_path,
        cloud,
        clear,
        extinction_numpy,
        extinction_auto,
        two_stream_auto,
        timings,
        external_spectrum,
    )

    plot_path = OUTPUT_DIR / "cloud_scattering_picaso_virga_benchmark.png"
    _plot(plot_path, cloud, clear, extinction_auto, two_stream_auto, timings, external_spectrum)
    json_path = OUTPUT_DIR / "cloud_scattering_picaso_virga_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {plot_path}")
    print(f"Wrote {json_path}")
    return report


def _cloud_property_path() -> Path:
    configured = os.environ.get("ROBERT_CLOUD_PROPERTY_FILE")
    if configured:
        return Path(configured).expanduser()
    output_path = OUTPUT_DIR / "synthetic_virga_style_cloud_properties.npz"
    if not output_path.exists():
        write_cloud_optical_properties_npz(_synthetic_cloud(), output_path)
    return output_path


def _load_cloud_file(path: Path) -> CloudOpticalProperties:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_cloud_optical_properties_npz(path, name=path.stem)
    if suffix == ".csv":
        return load_cloud_optical_properties_csv(path, name=path.stem)
    if suffix == ".cld":
        return load_picaso_cloud_optical_properties(
            path,
            name=path.stem,
            pressure_path=_optional_env_path("ROBERT_PICASO_PRESSURE_FILE")
            or _discover_picaso_pressure_path(path),
            wave_grid_path=_optional_env_path("ROBERT_PICASO_WAVE_GRID_FILE")
            or _discover_picaso_wave_grid_path(path),
        )
    raise ValueError("ROBERT_CLOUD_PROPERTY_FILE must point to a .npz, .csv, or PICASO .cld file")


def _optional_env_path(name: str) -> Path | None:
    configured = os.environ.get(name)
    if not configured:
        return None
    return Path(configured).expanduser()


def _discover_picaso_pressure_path(path: Path) -> Path | None:
    known_pairs = {
        "jupiterf3": "jupiter.pt",
        "HJ": "HJ.pt",
        "t1270g200f1_m0.0_co1.0": "t1270g200f1_m0.0_co1.0.cmp",
    }
    candidates = [
        path.with_name(known_pairs[path.stem]),
    ] if path.stem in known_pairs else []
    candidates.extend([path.with_suffix(".pt"), path.with_suffix(".cmp")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_picaso_wave_grid_path(path: Path) -> Path | None:
    candidates = [path.with_name("wave_EGP.dat")]
    if len(path.parents) >= 2:
        candidates.append(path.parent.parent / "opacities" / "wave_EGP.dat")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _timings(
    cloud_path: Path,
    gas_tau,
    geometry,
    cloud: CloudOpticalProperties,
    *,
    repeat: int,
) -> list[dict[str, float | int | str]]:
    repeat = max(1, repeat)
    timing_results = [
        time_callable(
            lambda: _load_cloud_file(cloud_path),
            name="load_cloud_properties",
            repeat=repeat,
            warmup=1,
            metadata={"path": str(cloud_path)},
        ),
        time_callable(
            lambda: solve_clear_sky_emission(
                gas_tau,
                geometry=geometry,
                bottom_boundary="blackbody",
                additional_optical_depths=[cloud],
                thermal_integration_backend="numpy",
            ),
            name="rt_extinction_numpy",
            repeat=repeat,
            warmup=1,
        ),
        time_callable(
            lambda: solve_clear_sky_emission(
                gas_tau,
                geometry=geometry,
                bottom_boundary="blackbody",
                additional_optical_depths=[cloud],
                thermal_integration_backend="auto",
            ),
            name="rt_extinction_auto",
            repeat=repeat,
            warmup=1,
        ),
        time_callable(
            lambda: solve_clear_sky_emission(
                gas_tau,
                geometry=geometry,
                bottom_boundary="blackbody",
                additional_optical_depths=[cloud],
                multiple_scattering_backend="two_stream",
                thermal_integration_backend="auto",
            ),
            name="rt_two_stream_auto",
            repeat=repeat,
            warmup=1,
        ),
    ]
    return [result.as_dict() for result in timing_results]


def _report(
    cloud_path: Path,
    cloud: CloudOpticalProperties,
    clear,
    extinction_numpy,
    extinction_auto,
    two_stream_auto,
    timings: list[dict[str, float | int | str]],
    external_spectrum: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    comparison = compare_cloud_optical_properties(cloud, cloud, name="cloud self-comparison")
    report: dict[str, Any] = {
        "cloud_file": str(cloud_path),
        "cloud": {
            "name": cloud.name,
            "n_layers": cloud.pressure_grid.n_layers,
            "n_wavelength": cloud.spectral_grid.size,
            "extinction_tau_min": float(np.min(cloud.extinction_tau)),
            "extinction_tau_max": float(np.max(cloud.extinction_tau)),
            "single_scattering_albedo_min": float(np.min(cloud.single_scattering_albedo)),
            "single_scattering_albedo_max": float(np.max(cloud.single_scattering_albedo)),
            "asymmetry_factor_min": float(np.min(cloud.asymmetry_factor)),
            "asymmetry_factor_max": float(np.max(cloud.asymmetry_factor)),
            "metadata": dict(cloud.metadata),
        },
        "cloud_property_comparison": comparison.as_dict(),
        "rt": {
            "clear_backend": clear.metadata["thermal_integration_backend"],
            "extinction_numpy_backend": extinction_numpy.metadata["thermal_integration_backend"],
            "extinction_auto_backend": extinction_auto.metadata["thermal_integration_backend"],
            "two_stream_backend": two_stream_auto.metadata["thermal_integration_backend"],
            "max_abs_numpy_auto_radiance_difference": float(
                np.max(np.abs(extinction_numpy.radiance.values - extinction_auto.radiance.values))
            ),
            "median_cloudy_clear_ratio": float(np.median(extinction_auto.radiance.values / clear.radiance.values)),
            "median_two_stream_clear_ratio": float(np.median(two_stream_auto.radiance.values / clear.radiance.values)),
        },
        "timings": timings,
    }
    if external_spectrum is not None:
        wavelength, values = external_spectrum
        model_wavelength = spectral_grid_values_in_unit(two_stream_auto.radiance.spectral_grid, "micron")
        model_order = np.argsort(model_wavelength)
        model = np.interp(
            wavelength,
            model_wavelength[model_order],
            two_stream_auto.radiance.values[model_order],
        )
        residual = model - values
        report["external_spectrum_comparison"] = {
            "n_points": int(wavelength.size),
            "median_residual": float(np.median(residual)),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "max_abs_residual": float(np.max(np.abs(residual))),
        }
    return report


def _plot(
    output_path: Path,
    cloud: CloudOpticalProperties,
    clear,
    extinction,
    two_stream,
    timings: list[dict[str, float | int | str]],
    external_spectrum: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    wavelength = spectral_grid_values_in_unit(clear.radiance.spectral_grid, "micron")
    wavelength_order = np.argsort(wavelength)
    wavelength = wavelength[wavelength_order]
    clear_radiance = clear.radiance.values[wavelength_order]
    ratio_extinction = (extinction.radiance.values / clear.radiance.values)[wavelength_order]
    ratio_two_stream = (two_stream.radiance.values / clear.radiance.values)[wavelength_order]
    physical_tau = np.sum(extinction.extinction_optical_depth, axis=0)[:, 0][wavelength_order]
    effective_tau = np.sum(two_stream.total_optical_depth, axis=0)[:, 0][wavelength_order]

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.0), constrained_layout=True)
    ax_ratio, ax_tau, ax_layer_tau, ax_contribution, ax_optical, ax_timing = axes.ravel()

    ax_ratio.plot(wavelength, ratio_extinction, color="#4c78a8", linewidth=1.8, label="Extinction only")
    ax_ratio.plot(wavelength, ratio_two_stream, color="#f58518", linewidth=1.8, label="Two-stream")
    if external_spectrum is not None:
        external_wavelength, external_values = external_spectrum
        ax_ratio.plot(
            external_wavelength,
            external_values / np.interp(external_wavelength, wavelength, clear_radiance),
            color="#111111",
            linewidth=1.0,
            alpha=0.7,
            label="External cloudy/clear",
        )
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Wavelength [micron]")
    ax_ratio.set_ylabel("Cloudy / clear radiance")
    ax_ratio.set_title("Spectrum Benchmark")
    ax_ratio.grid(alpha=0.25, which="both")
    ax_ratio.legend(frameon=False)

    ax_tau.plot(
        wavelength,
        _positive_for_log(physical_tau),
        color="#4c78a8",
        linewidth=1.8,
        label="Physical extinction tau",
    )
    ax_tau.plot(
        wavelength,
        _positive_for_log(effective_tau),
        color="#f58518",
        linewidth=1.8,
        label="Two-stream effective tau",
    )
    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    ax_tau.set_xlabel("Wavelength [micron]")
    ax_tau.set_ylabel("Column optical depth")
    ax_tau.set_title("Tau Diagnostic")
    ax_tau.grid(alpha=0.25, which="both")
    ax_tau.legend(frameon=False)

    pressure = pressure_values_in_unit(
        cloud.pressure_grid.centers,
        cloud.pressure_grid.unit,
        "bar",
    )
    layer_tau = cloud.extinction_tau[:, wavelength_order]
    positive_tau = layer_tau[layer_tau > 0.0]
    tau_floor = max(float(np.min(positive_tau)) if positive_tau.size else 1.0e-12, 1.0e-12)
    tau_ceiling = max(float(np.max(layer_tau)), tau_floor * 10.0)
    tau_mesh = ax_layer_tau.pcolormesh(
        wavelength,
        pressure,
        np.maximum(layer_tau, tau_floor),
        shading="auto",
        norm=LogNorm(vmin=tau_floor, vmax=tau_ceiling),
        cmap="magma",
    )
    ax_layer_tau.set_xscale("log")
    ax_layer_tau.set_yscale("log")
    ax_layer_tau.invert_yaxis()
    ax_layer_tau.set_xlabel("Wavelength [micron]")
    ax_layer_tau.set_ylabel("Pressure [bar]")
    ax_layer_tau.set_title("Layer Cloud Extinction")
    fig.colorbar(tau_mesh, ax=ax_layer_tau, label="Layer optical depth")

    normalized_contribution = two_stream.normalized_layer_contribution()[:, wavelength_order]
    contribution_mesh = ax_contribution.pcolormesh(
        wavelength,
        pressure,
        normalized_contribution,
        shading="auto",
        vmin=0.0,
        vmax=max(float(np.max(normalized_contribution)), 1.0e-12),
        cmap="viridis",
    )
    ax_contribution.set_xscale("log")
    ax_contribution.set_yscale("log")
    ax_contribution.invert_yaxis()
    ax_contribution.set_xlabel("Wavelength [micron]")
    ax_contribution.set_ylabel("Pressure [bar]")
    ax_contribution.set_title("Two-Stream Thermal Contribution")
    fig.colorbar(contribution_mesh, ax=ax_contribution, label="Normalized contribution")

    mean_ssa = np.mean(cloud.single_scattering_albedo, axis=0)[wavelength_order]
    mean_g = np.mean(cloud.asymmetry_factor, axis=0)[wavelength_order]
    ax_optical.plot(wavelength, mean_ssa, color="#54a24b", linewidth=1.8, label="Mean omega0")
    ax_optical.plot(wavelength, mean_g, color="#b279a2", linewidth=1.8, label="Mean g")
    ax_optical.set_xscale("log")
    ax_optical.set_xlabel("Wavelength [micron]")
    ax_optical.set_ylabel("Layer mean")
    ax_optical.set_title("Cloud Optical Properties")
    ax_optical.set_ylim(-0.05, 1.05)
    ax_optical.grid(alpha=0.25, which="both")
    ax_optical.legend(frameon=False)

    names = [str(item["name"]) for item in timings]
    medians_ms = [float(item["median_s"]) * 1.0e3 for item in timings]
    ax_timing.barh(names, medians_ms, color=["#79706e", "#4c78a8", "#f58518", "#e45756"])
    ax_timing.set_xlabel("Median time [ms]")
    ax_timing.set_title("Timing Smoke Benchmark")
    ax_timing.grid(alpha=0.25, axis="x")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def _optional_external_spectrum() -> tuple[np.ndarray, np.ndarray] | None:
    configured = os.environ.get("ROBERT_CLOUD_BENCHMARK_SPECTRUM_CSV")
    if not configured:
        return None
    path = Path(configured).expanduser()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("external spectrum CSV is missing a header row")
        wavelength_column = "wavelength_micron" if "wavelength_micron" in reader.fieldnames else "wavelength_um"
        value_column = "radiance" if "radiance" in reader.fieldnames else "value"
        wavelength: list[float] = []
        values: list[float] = []
        for row in reader:
            wavelength.append(float(row[wavelength_column]))
            values.append(float(row[value_column]))
    return np.asarray(wavelength, dtype=float), np.asarray(values, dtype=float)


def _synthetic_cloud() -> CloudOpticalProperties:
    pressure_edges = np.logspace(-5.0, 1.0, 65)
    pressure = PressureGrid(
        edges=pressure_edges,
        centers=np.sqrt(pressure_edges[:-1] * pressure_edges[1:]),
        unit="bar",
        name="synthetic PICASO/Virga cloud pressure",
    )
    spectral = SpectralGrid.from_array(
        np.geomspace(0.6, 14.0, 900),
        unit="micron",
        role="cloud_optical_properties",
        name="synthetic PICASO/Virga wavelength",
    )
    log_pressure = np.log10(pressure.centers)
    vertical_shape = np.exp(-0.5 * ((log_pressure - np.log10(3.0e-2)) / 0.45) ** 2)
    vertical_shape = vertical_shape / np.sum(vertical_shape)
    wavelength = spectral.values
    haze = 0.35 * np.power(wavelength / 1.0, -4.0)
    deck = np.full_like(wavelength, 1.4)
    tau = vertical_shape[:, None] * (deck[None, :] + haze[None, :])
    ssa = 0.48 + 0.42 / (1.0 + np.power(wavelength / 2.0, 2.0))
    g = 0.18 + 0.35 / (1.0 + np.power(wavelength / 4.0, 1.5))
    return CloudOpticalProperties(
        name="synthetic PICASO/Virga cloud",
        extinction_tau=tau,
        spectral_grid=spectral,
        pressure_grid=pressure,
        single_scattering_albedo=ssa,
        asymmetry_factor=g,
        metadata={"benchmark": "synthetic_picaso_virga_style"},
    )


def _zero_gas_tau(cloud: CloudOpticalProperties):
    pressure_grid = cloud.pressure_grid
    spectral_grid = cloud.spectral_grid
    pressure_scale = (
        np.log10(pressure_grid.centers) - np.log10(np.min(pressure_grid.centers))
    ) / (
        np.log10(np.max(pressure_grid.centers)) - np.log10(np.min(pressure_grid.centers))
    )
    temperature = 850.0 + 900.0 * pressure_scale**0.75
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=np.interp(
            np.log(pressure_grid.edges),
            np.log(pressure_grid.centers),
            temperature,
        ),
        composition={
            "H2O": np.full(pressure_grid.n_layers, 1.0e-12),
            "H2": np.full(pressure_grid.n_layers, 0.84),
            "He": np.full(pressure_grid.n_layers, 0.16),
        },
        mean_molecular_weight=2.3,
    )
    g_samples = np.array([0.10, 0.35, 0.65, 0.90])
    g_weights = np.array([0.20, 0.30, 0.30, 0.20])
    prepared = PreparedCorrelatedKOpacity(
        provider_name="synthetic-zero-opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=g_samples,
        g_weights=g_weights,
        cache_key="cloud-scattering-picaso-virga-benchmark",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure_grid.n_layers, spectral_grid.size, g_samples.size)),
        unit="m^2/molecule",
    )
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


if __name__ == "__main__":
    main()
