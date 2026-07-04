"""Benchmark and plot local HAT-P-32b correlated-k opacity tables."""

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
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    NemesisKTableHeader,
    PressureGrid,
    SpectralGrid,
    compare_opacity_arrays,
    read_kta,
    read_kta_header,
    time_callable,
)


DEFAULT_KTA_DIR = Path.home() / "Dropbox" / "PostDoc4" / "Emission_Example" / "HAT-P-32b" / "kta_temp"
DEFAULT_SPECIES = ("H2O", "CO", "CO2")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_opacity_benchmark"
KTA_FLOAT_DTYPE = np.dtype("<f4")
KTA_KCOEFF_SCALE = 1.0e-20
RUNTIME_NONFINITE_FILL_VALUE = 1.0e-300

TARGET_PRESSURES_BAR = np.array([1.0e-5, 1.0e-3, 1.0e-1, 1.0e1], dtype=float)
TARGET_TEMPERATURES_K = np.array([1000.0, 1500.0, 2000.0, 2500.0], dtype=float)
SLICE_PRESSURE_BAR = 1.0e-3
SLICE_TEMPERATURE_K = 1500.0
HEATMAP_WAVELENGTH_MICRON = 4.5


def main() -> dict[str, object]:
    """Run the local HAT-P-32b opacity benchmark and return its summary."""

    kta_dir = _kta_dir()
    species = _requested_species()
    repeat = int(os.environ.get("ROBERT_OPACITY_BENCHMARK_REPEAT", "5"))
    warmup = int(os.environ.get("ROBERT_OPACITY_BENCHMARK_WARMUP", "1"))

    if not kta_dir.exists():
        raise FileNotFoundError(
            "HAT-P-32b k-table directory was not found. Set HAT_P_32B_KTA_DIR "
            f"or place files at {DEFAULT_KTA_DIR}."
        )
    paths = _species_paths(kta_dir, species)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Benchmark: HAT-P-32b correlated-k opacity")
    print(f"K-table directory: {kta_dir}")
    print(f"Species: {', '.join(paths)}")
    print(f"Timing repeats: {repeat}, warmup: {warmup}")

    tables: dict[str, CorrelatedKTable] = {}
    species_summaries: dict[str, object] = {}
    for species_name, path in paths.items():
        table, summary = _benchmark_species(species_name, path, repeat=repeat, warmup=warmup)
        tables[species_name] = table
        species_summaries[species_name] = summary
        timing = summary.get("evaluation_timing")
        timing_text = "" if timing is None else f", eval median={timing['median_s'] * 1.0e3:.2f} ms"
        strict_text = "strict" if summary["strict_load_passed"] else "runtime-floor"
        print(
            f"- {species_name}: {strict_text}, load={summary['load_s']:.2f} s{timing_text}, "
            f"comparisons_passed={summary['all_exact_comparisons_passed']}"
        )

    if not tables:
        raise RuntimeError("No requested HAT-P-32b k-tables could be loaded for plotting")

    wavelength_plot = _plot_wavelength_slices(tables, OUTPUT_DIR / "hat_p_32b_opacity_wavelength_slices.png")
    heatmap_plots = _plot_pt_heatmaps(tables, OUTPUT_DIR)
    nonfinite_plot = _plot_nonfinite_pt_summary(paths, OUTPUT_DIR / "hat_p_32b_opacity_nonfinite_pt_summary.png")
    summary_path = OUTPUT_DIR / "hat_p_32b_opacity_benchmark_summary.json"
    summary = {
        "kta_dir": str(kta_dir),
        "species": list(paths),
        "target_pressures_bar": TARGET_PRESSURES_BAR.tolist(),
        "target_temperatures_K": TARGET_TEMPERATURES_K.tolist(),
        "slice_pressure_bar": SLICE_PRESSURE_BAR,
        "slice_temperature_K": SLICE_TEMPERATURE_K,
        "heatmap_wavelength_micron": HEATMAP_WAVELENGTH_MICRON,
        "species_summaries": species_summaries,
        "outputs": {
            "summary_json": str(summary_path),
            "wavelength_slices_png": str(wavelength_plot),
            "pt_heatmaps_png": {species_name: str(path) for species_name, path in heatmap_plots.items()},
            "nonfinite_pt_summary_png": str(nonfinite_plot),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {wavelength_plot}")
    for path in heatmap_plots.values():
        print(f"Wrote {path}")
    print(f"Wrote {nonfinite_plot}")
    return summary


def _benchmark_species(
    species: str,
    path: Path,
    *,
    repeat: int,
    warmup: int,
) -> tuple[CorrelatedKTable, dict[str, object]]:
    header = read_kta_header(path)
    start = perf_counter()
    _raw_kcoeff, raw_stats = _native_kcoeff_and_stats(header)
    strict_load_passed = True
    strict_reader_error = None
    try:
        nemesis_table = read_kta(path)
    except Exception as exc:
        strict_load_passed = False
        strict_reader_error = f"{type(exc).__name__}: {exc}"
        nemesis_table = read_kta(
            path,
            nonfinite_policy="floor",
            nonfinite_fill_value=RUNTIME_NONFINITE_FILL_VALUE,
        )
    table = CorrelatedKTable.from_nemesis(species, nemesis_table)
    load_s = perf_counter() - start

    pressure_indices = _nearest_unique_indices(table.pressure_bar, TARGET_PRESSURES_BAR, log_space=True)
    temperature_indices = _nearest_unique_indices(table.temperature_K, TARGET_TEMPERATURES_K, log_space=False)
    pressure_grid = _pressure_grid_from_centers(table.pressure_bar[pressure_indices])
    spectral_grid = SpectralGrid.from_array(table.wavenumber_cm_inverse, unit="cm^-1", role="opacity")
    comparisons = []
    timing_summary = None
    provider = CorrelatedKOpacityProvider({species: table})
    prepared = provider.prepare(spectral_grid, pressure_grid, species=(species,))
    for temperature_index in temperature_indices:
        atmosphere = _atmosphere(
            pressure_grid,
            species,
            temperature=float(table.temperature_K[temperature_index]),
        )
        evaluated = provider.evaluate(atmosphere, prepared)
        reference = table.kcoeff[pressure_indices, temperature_index, :, :]
        result = compare_opacity_arrays(
            evaluated.kcoeff[0],
            reference,
            name=f"{species}-exact-runtime-evaluator",
            axis_names=("pressure", "wavelength", "g_ordinate"),
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            metadata={
                "species": species,
                "temperature_K": f"{table.temperature_K[temperature_index]:.6g}",
                "strict_source_load_passed": str(strict_load_passed),
                "comparison_reference": "runtime_floor_loaded_table",
            },
        )
        comparisons.append(result.to_mapping())

    timing_temperature_index = int(_nearest_unique_indices(table.temperature_K, [SLICE_TEMPERATURE_K])[0])
    timing_atmosphere = _atmosphere(
        pressure_grid,
        species,
        temperature=float(table.temperature_K[timing_temperature_index]),
    )
    timing = time_callable(
        lambda: provider.evaluate(timing_atmosphere, prepared),
        name=f"{species}-exact-opacity-evaluate",
        repeat=repeat,
        warmup=warmup,
    )
    timing_summary = timing.as_dict()

    return table, {
        "path": str(path),
        "file_size_mib": header.file_size_bytes / 1024**2,
        "native_shape": list(header.native_shape),
        "strict_load_passed": strict_load_passed,
        "strict_reader_error": strict_reader_error,
        "runtime_nonfinite_policy": "raise" if strict_load_passed else "floor",
        "runtime_nonfinite_fill_value": None if strict_load_passed else RUNTIME_NONFINITE_FILL_VALUE,
        "raw_kcoeff_stats": raw_stats,
        "runtime_metadata": dict(table.metadata),
        "load_s": load_s,
        "selected_pressures_bar": table.pressure_bar[pressure_indices].tolist(),
        "selected_temperatures_K": table.temperature_K[temperature_indices].tolist(),
        "spectral_range_micron": [
            float(np.min(table.wavelength_micron)),
            float(np.max(table.wavelength_micron)),
        ],
        "wavenumber_range_cm_inverse": [
            float(np.min(table.wavenumber_cm_inverse)),
            float(np.max(table.wavenumber_cm_inverse)),
        ],
        "evaluation_timing": timing_summary,
        "exact_comparisons": comparisons,
        "all_exact_comparisons_passed": bool(comparisons and all(item["passed"] for item in comparisons)),
    }


def _native_kcoeff_and_stats(header: NemesisKTableHeader) -> tuple[np.ndarray, dict[str, object]]:
    raw = np.fromfile(
        header.path,
        dtype=KTA_FLOAT_DTYPE,
        count=header.n_kcoefficients,
        offset=header.data_offset_bytes,
    )
    finite = np.isfinite(raw)
    finite_raw = raw[finite]
    stored_nonfinite = ~finite.reshape(header.stored_shape)
    native_nonfinite = stored_nonfinite[::-1].transpose(1, 2, 0, 3)
    native_kcoeff = raw.reshape(header.stored_shape)[::-1].transpose(1, 2, 0, 3).astype(float)
    native_kcoeff *= KTA_KCOEFF_SCALE
    stats: dict[str, object] = {
        "n_values": int(raw.size),
        "n_finite": int(np.sum(finite)),
        "n_nonfinite": int(raw.size - np.sum(finite)),
        "n_nan": int(np.sum(np.isnan(raw))),
        "n_posinf": int(np.sum(np.isposinf(raw))),
        "n_neginf": int(np.sum(np.isneginf(raw))),
        "nonfinite_fraction": float(np.sum(~finite) / raw.size),
        "top_pressure_bar": _top_axis_counts(
            native_nonfinite.sum(axis=(1, 2, 3)),
            header.pressure_bar,
            denominator=header.n_temperature * header.n_wavelength * header.n_g,
        ),
        "top_temperature_K": _top_axis_counts(
            native_nonfinite.sum(axis=(0, 2, 3)),
            header.temperature_K,
            denominator=header.n_pressure * header.n_wavelength * header.n_g,
        ),
        "top_wavelength_micron": _top_axis_counts(
            native_nonfinite.sum(axis=(0, 1, 3)),
            header.wavelength_micron,
            denominator=header.n_pressure * header.n_temperature * header.n_g,
        ),
        "top_g_ordinate": _top_axis_counts(
            native_nonfinite.sum(axis=(0, 1, 2)),
            np.arange(header.n_g, dtype=float),
            denominator=header.n_pressure * header.n_temperature * header.n_wavelength,
        ),
    }
    if finite_raw.size:
        stats.update(
            {
                "n_negative_finite": int(np.sum(finite_raw < 0.0)),
                "min_finite_scaled": float(np.min(finite_raw) * KTA_KCOEFF_SCALE),
                "max_finite_scaled": float(np.max(finite_raw) * KTA_KCOEFF_SCALE),
            }
        )
    return native_kcoeff, stats


def _top_axis_counts(
    counts: np.ndarray,
    coordinates: np.ndarray,
    *,
    denominator: int,
    n_items: int = 8,
) -> list[dict[str, float | int]]:
    counts = np.asarray(counts, dtype=np.int64)
    if not np.any(counts):
        return []
    indices = np.argsort(counts)[-n_items:][::-1]
    return [
        {
            "index": int(index),
            "coordinate": float(coordinates[index]),
            "count": int(counts[index]),
            "axis_fraction": float(counts[index] / denominator),
        }
        for index in indices
        if counts[index] > 0
    ]


def _plot_wavelength_slices(tables: dict[str, CorrelatedKTable], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    for species, table in tables.items():
        pressure_index = _nearest_index(table.pressure_bar, SLICE_PRESSURE_BAR, log_space=True)
        temperature_index = _nearest_index(table.temperature_K, SLICE_TEMPERATURE_K)
        mean_k = np.tensordot(
            table.kcoeff[pressure_index, temperature_index, :, :],
            table.g_weights,
            axes=([-1], [0]),
        )
        wavelength = np.asarray(table.wavelength_micron)
        order = np.argsort(wavelength)
        plotted_k = np.ma.masked_less_equal(mean_k[order], RUNTIME_NONFINITE_FILL_VALUE * 10.0)
        ax.plot(
            wavelength[order],
            plotted_k,
            linewidth=1.4,
            label=(
                f"{species}, P={table.pressure_bar[pressure_index]:.1e} bar, "
                f"T={table.temperature_K[temperature_index]:.0f} K"
            ),
        )

    ax.set_title("HAT-P-32b k-table weighted-mean opacity slices")
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel("g-weighted k coefficient [cm^2 molecule^-1]")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 6.0)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_pt_heatmaps(tables: dict[str, CorrelatedKTable], output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for species, table in tables.items():
        path = output_dir / f"hat_p_32b_opacity_pt_heatmap_{species}.png"
        paths[species] = _plot_pt_heatmap(table, path)
    return paths


def _plot_pt_heatmap(table: CorrelatedKTable, output_path: Path) -> Path:
    wavelength_index = _nearest_index(table.wavelength_micron, HEATMAP_WAVELENGTH_MICRON)
    mean_k = np.tensordot(table.kcoeff[:, :, wavelength_index, :], table.g_weights, axes=([-1], [0]))
    masked_mean_k = np.ma.masked_less_equal(mean_k, RUNTIME_NONFINITE_FILL_VALUE * 10.0)
    log_k = np.ma.log10(masked_mean_k)

    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        table.temperature_K,
        table.pressure_bar,
        log_k,
        shading="auto",
        cmap="magma",
    )
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("log10 g-weighted k [cm^2 molecule^-1]")
    ax.set_title(
        f"{table.species} opacity P-T map at "
        f"{table.wavelength_micron[wavelength_index]:.2f} micron"
    )
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim(float(np.max(table.pressure_bar)), float(np.min(table.pressure_bar)))
    ax.grid(alpha=0.20, which="both")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_nonfinite_pt_summary(paths: dict[str, Path], output_path: Path) -> Path:
    n_species = len(paths)
    n_columns = min(3, n_species)
    n_rows = int(np.ceil(n_species / n_columns))
    fractions: dict[str, tuple[NemesisKTableHeader, np.ndarray]] = {}
    global_vmax = 0.0
    for species, path in paths.items():
        header = read_kta_header(path)
        bad = _native_nonfinite_mask(header)
        fraction = bad.sum(axis=(2, 3)) / (header.n_wavelength * header.n_g)
        fractions[species] = (header, fraction)
        global_vmax = max(global_vmax, float(np.max(fraction)))
    global_vmax = max(global_vmax, 0.01)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    mesh = None
    for axis, (species, (header, fraction)) in zip(axes.flat, fractions.items()):
        masked_fraction = np.ma.masked_equal(fraction, 0.0)
        mesh = axis.pcolormesh(
            header.temperature_K,
            header.pressure_bar,
            masked_fraction,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=global_vmax,
        )
        axis.set_title(species)
        axis.set_xlabel("Temperature [K]")
        axis.set_ylabel("Pressure [bar]")
        axis.set_yscale("log")
        axis.set_ylim(float(np.max(header.pressure_bar)), float(np.min(header.pressure_bar)))
        axis.grid(alpha=0.15, which="both")
    for axis in axes.flat[n_species:]:
        axis.set_visible(False)
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist())
        colorbar.set_label("Fraction of wavelength x g entries that are non-finite")
    fig.suptitle("HAT-P-32b `.kta` non-finite k-coefficient locations")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _native_nonfinite_mask(header: NemesisKTableHeader) -> np.ndarray:
    raw = np.fromfile(
        header.path,
        dtype=KTA_FLOAT_DTYPE,
        count=header.n_kcoefficients,
        offset=header.data_offset_bytes,
    )
    return (~np.isfinite(raw)).reshape(header.stored_shape)[::-1].transpose(1, 2, 0, 3)


def _kta_dir() -> Path:
    configured = os.environ.get("HAT_P_32B_KTA_DIR")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_KTA_DIR


def _requested_species() -> tuple[str, ...]:
    configured = os.environ.get("ROBERT_OPACITY_BENCHMARK_SPECIES")
    if not configured:
        return DEFAULT_SPECIES
    species = tuple(item.strip() for item in configured.split(",") if item.strip())
    if species == ("all",):
        return ("all",)
    if not species:
        raise ValueError("ROBERT_OPACITY_BENCHMARK_SPECIES did not contain any species names")
    return species


def _species_paths(kta_dir: Path, requested: tuple[str, ...]) -> dict[str, Path]:
    available = {_species_from_kta_path(path): path for path in sorted(kta_dir.glob("*.kta"))}
    if requested == ("all",):
        return available
    missing = tuple(species for species in requested if species not in available)
    if missing:
        available_text = ", ".join(available) or "none"
        raise FileNotFoundError(
            f"Missing requested k-table species: {', '.join(missing)}. Available: {available_text}"
        )
    return {species: available[species] for species in requested}


def _species_from_kta_path(path: Path) -> str:
    return path.stem.split("_")[0]


def _atmosphere(pressure_grid: PressureGrid, species: str, *, temperature: float) -> AtmosphereState:
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.full(pressure_grid.n_layers, temperature),
        composition={species: np.full(pressure_grid.n_layers, 1.0e-4)},
        mean_molecular_weight=np.full(pressure_grid.n_layers, 2.3),
    )


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
    return PressureGrid(edges=edges, centers=centers, unit="bar", name="HAT-P-32b opacity benchmark")


def _nearest_unique_indices(
    values: np.ndarray,
    targets: np.ndarray | list[float],
    *,
    log_space: bool = False,
) -> np.ndarray:
    indices: list[int] = []
    for target in targets:
        index = _nearest_index(values, float(target), log_space=log_space)
        if index not in indices:
            indices.append(index)
    return np.asarray(sorted(indices), dtype=np.int64)


def _nearest_index(values: np.ndarray, target: float, *, log_space: bool = False) -> int:
    axis = np.asarray(values, dtype=float)
    if log_space:
        distances = np.abs(np.log10(axis) - np.log10(float(target)))
    else:
        distances = np.abs(axis - float(target))
    return int(np.argmin(distances))


if __name__ == "__main__":
    main()
