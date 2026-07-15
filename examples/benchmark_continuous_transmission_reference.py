"""Compare ROBERT and PICASO with a continuous high-order transmission reference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    HydrostaticPathGeometry,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    solve_absorption_transmission,
)

try:
    from examples.benchmark_shared_deck_haze_external_parity import ROOT, _rms
except ModuleNotFoundError:
    from benchmark_shared_deck_haze_external_parity import ROOT, _rms


BOLTZMANN_CONSTANT_J_K = 1.380649e-23
ATOMIC_MASS_KG = 1.66053906660e-27
RUNNER = Path(__file__).with_name("run_picaso_continuous_transmission_reference.py")
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "continuous_transmission_reference"
DEFAULT_LAYERS = (16, 32, 64, 80, 128, 256, 512)


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picaso-python",
        type=Path,
        default=Path("/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--layers",
        default=",".join(str(value) for value in DEFAULT_LAYERS),
    )
    parser.add_argument("--reference-order", type=int, default=1024)
    args = parser.parse_args()
    report = run(
        args.picaso_python,
        args.output_dir,
        layer_counts=tuple(
            dict.fromkeys(int(value) for value in args.layers.split(",") if value)
        ),
        reference_order=args.reference_order,
    )
    print(json.dumps(report, indent=2))
    return report


def run(
    picaso_python: Path,
    output_dir: Path,
    *,
    layer_counts: tuple[int, ...],
    reference_order: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _contract(layer_counts)
    started = perf_counter()
    reference = _continuous_transit_depth(contract, reference_order)
    reference_seconds = perf_counter() - started
    check_order = max(64, reference_order // 2)
    check = _continuous_transit_depth(contract, check_order)
    reference_error_ppm = (check - reference) * 1.0e6

    contract_path = output_dir / "continuous_transmission_contract.npz"
    picaso_path = output_dir / "picaso_continuous_transmission.npz"
    np.savez_compressed(contract_path, **contract)
    _run_picaso(picaso_python, contract_path, picaso_path)
    with np.load(picaso_path, allow_pickle=False) as archive:
        picaso = {name: np.array(archive[name], copy=True) for name in archive.files}

    spectra = {"continuous_reference": reference}
    metrics: dict[str, dict[str, dict[str, float]]] = {
        "robert": {},
        "picaso": {},
    }
    timings: dict[str, float] = {}
    for layers in layer_counts:
        started = perf_counter()
        robert = _evaluate_robert(contract, layers)
        timings[str(layers)] = perf_counter() - started
        picaso_depth = np.asarray(picaso[f"transit_depth_{layers}"], dtype=float)
        spectra[f"robert_{layers}"] = robert
        spectra[f"picaso_{layers}"] = picaso_depth
        metrics["robert"][str(layers)] = _error_metrics(robert, reference)
        metrics["picaso"][str(layers)] = _error_metrics(picaso_depth, reference)

    report = {
        "schema_version": 1,
        "benchmark": "continuous_inverse_square_transmission_reference",
        "interpretation": (
            "Both solvers receive the same analytic pressure-radius and absorption "
            "contract. The reference directly integrates the continuous density "
            "field and is independent of either finite-layer convention."
        ),
        "contract": {
            key: _json_scalar(value)
            for key, value in contract.items()
            if np.asarray(value).ndim == 0
        },
        "layer_counts": list(layer_counts),
        "continuous_reference": {
            "quadrature_order": reference_order,
            "check_order": check_order,
            "check_to_reference_rms_ppm": _rms(reference_error_ppm),
            "check_to_reference_max_abs_ppm": float(
                np.max(np.abs(reference_error_ppm))
            ),
            "seconds": reference_seconds,
        },
        "metrics": metrics,
        "convergence_order_fit": {
            code: _convergence_order(layer_counts, metrics[code])
            for code in ("robert", "picaso")
        },
        "robert_seconds_by_layers": timings,
        "picaso_metadata": json.loads(str(picaso["metadata_json"])),
    }
    (output_dir / "continuous_transmission_reference.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "continuous_transmission_reference_spectra.npz",
        wavelength_micron=contract["wavelength_micron"],
        cross_section_m2=contract["cross_section_m2"],
        **spectra,
    )
    _plot(
        output_dir / "continuous_transmission_reference.png",
        contract["wavelength_micron"],
        reference,
        spectra,
        metrics,
        layer_counts,
    )
    return report


def _contract(layer_counts):
    wavelength = np.geomspace(1.0, 12.0, 180)
    log_wavelength = np.log(wavelength)
    cross_section = np.full(wavelength.size, 2.0e-31)
    bands = (
        (1.4, 0.08, 3.0e-28),
        (1.9, 0.09, 9.0e-28),
        (2.7, 0.12, 2.0e-27),
        (4.3, 0.11, 1.2e-27),
        (6.3, 0.14, 3.0e-27),
        (9.5, 0.17, 8.0e-28),
    )
    for center, width, amplitude in bands:
        cross_section += amplitude * np.exp(
            -0.5 * ((log_wavelength - np.log(center)) / width) ** 2
        )
    return {
        "wavelength_micron": wavelength,
        "cross_section_m2": cross_section,
        "layer_counts": np.asarray(layer_counts, dtype=int),
        "pressure_bottom_bar": np.array(10.0),
        "pressure_top_bar": np.array(1.0e-9),
        "planet_radius_m": np.array(75_567_044.0),
        "star_radius_m": np.array(695_700_000.0),
        "gravity_m_s2": np.array(8.42),
        "temperature_k": np.array(1200.0),
        "mean_molecular_weight_amu": np.array(2.3),
    }


def _continuous_transit_depth(contract, order):
    radius_bottom = float(contract["planet_radius_m"])
    radius_top = _radius_at_pressure(
        float(contract["pressure_top_bar"]) * 1.0e5,
        contract,
    )
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half_width = 0.5 * (radius_top - radius_bottom)
    midpoint = 0.5 * (radius_top + radius_bottom)
    impact = midpoint + half_width * nodes
    half_chord = np.sqrt(radius_top**2 - impact**2)
    line_coordinate = 0.5 * half_chord[:, None] * (nodes[None, :] + 1.0)
    radius = np.sqrt(impact[:, None] ** 2 + line_coordinate**2)
    number_density = _pressure_at_radius(radius, contract) / (
        BOLTZMANN_CONSTANT_J_K * float(contract["temperature_k"])
    )
    slant_column = half_chord * np.sum(
        weights[None, :] * number_density,
        axis=1,
    )
    slant_tau = slant_column[:, None] * np.asarray(
        contract["cross_section_m2"],
        dtype=float,
    )[None, :]
    blocked_area = radius_bottom**2 + half_width * np.sum(
        weights[:, None]
        * 2.0
        * impact[:, None]
        * (1.0 - np.exp(-slant_tau)),
        axis=0,
    )
    return blocked_area / float(contract["star_radius_m"]) ** 2


def _evaluate_robert(contract, layers):
    pressure_edges = np.geomspace(
        float(contract["pressure_top_bar"]),
        float(contract["pressure_bottom_bar"]),
        layers + 1,
    )
    pressure_centers = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_centers,
        unit="bar",
        name="continuous inverse-square reference",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.full(layers, float(contract["temperature_k"])),
        composition={"absorber": np.ones(layers)},
        mean_molecular_weight=float(contract["mean_molecular_weight_amu"]),
    )
    wavelength = np.asarray(contract["wavelength_micron"], dtype=float)
    spectral_grid = SpectralGrid.from_array(
        wavelength,
        unit="micron",
        role="opacity",
        name="analytic continuous absorption law",
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="analytic continuous absorption law",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("absorber",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key=f"continuous-reference-{layers}",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, layers, wavelength.size, 1)),
        unit="m^2/molecule",
    )
    gas = assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=float(contract["gravity_m_s2"]),
    )
    geometry = _exact_geometry(contract, atmosphere)
    vertical_column = _exact_layer_vertical_columns(geometry.edge_radius_m, contract)
    vertical_tau = vertical_column[:, None] * np.asarray(
        contract["cross_section_m2"],
        dtype=float,
    )[None, :]
    gas = replace(
        gas,
        species_tau=None,
        total_tau=vertical_tau[:, :, None],
        metadata={
            **dict(gas.metadata),
            "source": "exact continuous layer-integrated vertical column",
        },
    )
    result = solve_absorption_transmission(
        gas,
        geometry,
        star_radius_m=float(contract["star_radius_m"]),
        impact_quadrature_order=8,
    )
    return np.asarray(result.transit_depth.values, dtype=float)


def _exact_geometry(contract, atmosphere):
    pressure_edges_pa = np.asarray(atmosphere.pressure_grid.edges) * 1.0e5
    pressure_centers_pa = np.asarray(atmosphere.pressure_grid.centers) * 1.0e5
    edge_radius = _radius_at_pressure(pressure_edges_pa, contract)
    center_radius = _radius_at_pressure(pressure_centers_pa, contract)
    scale_height = (edge_radius[:-1] - edge_radius[1:]) / np.log(
        pressure_edges_pa[1:] / pressure_edges_pa[:-1]
    )
    gravity = float(contract["gravity_m_s2"]) * (
        float(contract["planet_radius_m"]) / center_radius
    ) ** 2
    return HydrostaticPathGeometry(
        pressure_grid=atmosphere.pressure_grid,
        reference_radius_m=float(contract["planet_radius_m"]),
        reference_pressure_pa=float(contract["pressure_bottom_bar"]) * 1.0e5,
        gravity_m_s2=gravity,
        scale_height_m=scale_height,
        edge_radius_m=edge_radius,
        center_radius_m=center_radius,
        metadata={
            "path_model": "analytic_inverse_square_isothermal",
            "gravity_model": "exact_inverse_square",
        },
    )


def _exact_layer_vertical_columns(edge_radius, contract, order=32):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    outer = np.asarray(edge_radius[:-1], dtype=float)
    inner = np.asarray(edge_radius[1:], dtype=float)
    half_width = 0.5 * (outer - inner)
    midpoint = 0.5 * (outer + inner)
    radius = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    number_density = _pressure_at_radius(radius, contract) / (
        BOLTZMANN_CONSTANT_J_K * float(contract["temperature_k"])
    )
    return half_width * np.sum(weights[None, :] * number_density, axis=1)


def _hydrostatic_coefficient_m(contract):
    return (
        float(contract["mean_molecular_weight_amu"])
        * ATOMIC_MASS_KG
        * float(contract["gravity_m_s2"])
        * float(contract["planet_radius_m"]) ** 2
        / (BOLTZMANN_CONSTANT_J_K * float(contract["temperature_k"]))
    )


def _radius_at_pressure(pressure_pa, contract):
    pressure_bottom_pa = float(contract["pressure_bottom_bar"]) * 1.0e5
    radius_bottom = float(contract["planet_radius_m"])
    return 1.0 / (
        1.0 / radius_bottom
        + np.log(np.asarray(pressure_pa, dtype=float) / pressure_bottom_pa)
        / _hydrostatic_coefficient_m(contract)
    )


def _pressure_at_radius(radius_m, contract):
    radius_bottom = float(contract["planet_radius_m"])
    pressure_bottom_pa = float(contract["pressure_bottom_bar"]) * 1.0e5
    return pressure_bottom_pa * np.exp(
        _hydrostatic_coefficient_m(contract)
        * (1.0 / np.asarray(radius_m, dtype=float) - 1.0 / radius_bottom)
    )


def _run_picaso(python, contract_path, output_path):
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "picaso-mpl")
    environment["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "picaso-numba")
    subprocess.run(
        [str(python), str(RUNNER), str(contract_path), str(output_path)],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def _error_metrics(values, reference):
    residual = (np.asarray(values) - np.asarray(reference)) * 1.0e6
    return {
        "rms_ppm": _rms(residual),
        "median_ppm": float(np.median(residual)),
        "max_abs_ppm": float(np.max(np.abs(residual))),
    }


def _convergence_order(layer_counts, metrics):
    layers = np.asarray(layer_counts, dtype=float)
    errors = np.asarray([metrics[str(value)]["rms_ppm"] for value in layer_counts])
    selected = (layers >= 32) & (errors > 0.0)
    if np.count_nonzero(selected) < 2:
        selected = errors > 0.0
    slope = np.polyfit(np.log(layers[selected]), np.log(errors[selected]), 1)[0]
    return float(-slope)


def _json_scalar(value):
    scalar = np.asarray(value).item()
    return int(scalar) if isinstance(scalar, np.integer) else float(scalar)


def _plot(path, wavelength, reference, spectra, metrics, layer_counts):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    axes[0].plot(wavelength, reference * 1.0e6, color="black", lw=1.2)
    axes[0].set_title("Continuous reference spectrum")
    axes[0].set_ylabel("Transit depth (ppm)")
    comparison_layers = 80 if 80 in layer_counts else layer_counts[len(layer_counts) // 2]
    for code, color in (("robert", "tab:blue"), ("picaso", "tab:orange")):
        axes[1].plot(
            wavelength,
            (spectra[f"{code}_{comparison_layers}"] - reference) * 1.0e6,
            color=color,
            lw=1.0,
            label=code.upper(),
        )
        axes[2].plot(
            layer_counts,
            [metrics[code][str(value)]["rms_ppm"] for value in layer_counts],
            color=color,
            marker="o",
            label=code.upper(),
        )
    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[1].set_title(f"Residual at {comparison_layers} layers")
    axes[1].set_ylabel("Model - continuous reference (ppm)")
    axes[2].set_title("Finite-layer convergence")
    axes[2].set_xlabel("Pressure layers")
    axes[2].set_ylabel("RMS residual (ppm)")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    for axis in axes[:2]:
        axis.set_xscale("log")
        axis.set_xlabel("Wavelength (micron)")
    axes[1].legend(frameon=False)
    axes[2].legend(frameon=False)
    figure.suptitle("Continuous inverse-square transmission benchmark")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
