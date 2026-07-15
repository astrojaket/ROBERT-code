"""Official-PICASO benchmark for ROBERT's shared finite deck/haze model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    LayerOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    hydrostatic_path_geometry,
    solve_absorption_transmission,
)

try:
    from examples.benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _inverse_square_hydrostatic_profiles,
    )
    from examples.benchmark_shared_deck_haze_external_parity import (
        CASES,
        ROOT,
        _bin,
        _contract,
        _evaluate_robert,
        _rms,
    )
except ModuleNotFoundError:
    from benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _inverse_square_hydrostatic_profiles,
    )
    from benchmark_shared_deck_haze_external_parity import (
        CASES,
        ROOT,
        _bin,
        _contract,
        _evaluate_robert,
        _rms,
    )


RUNNER = Path(__file__).with_name("run_picaso_shared_deck_haze.py")
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "shared_deck_haze_picaso"


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picaso-python",
        type=Path,
        default=Path("/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python"),
    )
    parser.add_argument("--picaso-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--picaso-database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--convergence-layers", default="48,80,120,160")
    parser.add_argument("--opacity-stride", type=int, default=5)
    parser.add_argument("--output-bins", type=int, default=180)
    args = parser.parse_args()
    return run(
        args.picaso_python,
        args.picaso_reference,
        args.picaso_database,
        args.output_dir,
        layers=args.layers,
        convergence_layers=tuple(
            dict.fromkeys(int(value) for value in args.convergence_layers.split(",") if value)
        ),
        opacity_stride=args.opacity_stride,
        output_bins=args.output_bins,
    )


def run(
    picaso_python: Path,
    reference: Path,
    database: Path,
    output_dir: Path,
    *,
    layers: int,
    convergence_layers: tuple[int, ...],
    opacity_stride: int,
    output_bins: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    compact_by_layer = {}
    layer_counts = tuple(dict.fromkeys((*convergence_layers, layers)))
    for n_layer in layer_counts:
        contract = _contract(n_layer)
        started = perf_counter()
        robert = _evaluate_robert(contract, opacity_stride)
        robert_seconds = perf_counter() - started
        with tempfile.TemporaryDirectory(prefix="robert-picaso-deck-haze-") as scratch:
            contract_path = Path(scratch) / "contract.npz"
            external_path = Path(scratch) / "picaso.npz"
            np.savez_compressed(contract_path, **contract)
            _run_external(
                picaso_python,
                reference,
                database,
                contract_path,
                external_path,
                opacity_stride,
            )
            with np.load(external_path, allow_pickle=False) as archive:
                picaso = {name: np.array(archive[name], copy=True) for name in archive.files}
        compact = _compact(robert, picaso, output_bins)
        compact_by_layer[n_layer] = compact
        metrics = _metrics(compact)
        matched_optical_depth = _matched_picaso_tau_transmission(
            picaso, contract, output_bins
        )
        results[str(n_layer)] = {
            "metrics": metrics,
            "matched_picaso_layer_optical_depth_transmission": matched_optical_depth,
            "robert_seconds": robert_seconds,
            "picaso": json.loads(str(picaso["metadata_json"])),
        }
        if n_layer == layers:
            np.savez_compressed(
                output_dir / "shared_deck_haze_picaso_spectra.npz", **compact
            )
            _plot(output_dir / "shared_deck_haze_picaso.png", compact)
    primary = results[str(layers)]["metrics"]
    numerical_convergence = _convergence_metrics(compact_by_layer)
    report = {
        "schema_version": 1,
        "benchmark": "official_PICASO_shared_finite_deck_haze_emission_transmission",
        "interpretation": (
            "The pressure-temperature-composition and layer aerosol optical-depth contract "
            "is shared; molecular/CIA opacities and both RT calculations are independent."
        ),
        "cases": list(CASES),
        "primary_layers": layers,
        "convergence_layers": list(layer_counts),
        "opacity_stride": opacity_stride,
        "output_bins": output_bins,
        "external_comparison_targets_not_truth_criteria": {
            "emission_cloud_effect_rms_ppm": 10.0,
            "transmission_cloud_effect_rms_ppm": 20.0,
            "interpretation": (
                "These targets flag differences for investigation; they do not assert "
                "that PICASO is the correct answer."
            ),
        },
        "primary_metrics": primary,
        "primary_matched_picaso_layer_optical_depth_transmission": results[
            str(layers)
        ]["matched_picaso_layer_optical_depth_transmission"],
        "vertical_convergence": results,
        "numerical_convergence_to_finest_grid": numerical_convergence,
    }
    (output_dir / "shared_deck_haze_picaso.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def _run_external(python, reference, database, contract, output, stride):
    environment = dict(os.environ)
    environment["picaso_refdata"] = str(reference.resolve())
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "picaso-mpl")
    environment["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "picaso-numba")
    subprocess.run(
        [
            str(python),
            str(RUNNER),
            str(contract),
            str(output),
            "--opacity-db",
            str(database),
            "--resample",
            str(stride),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def _compact(robert, picaso, output_bins):
    edges = np.geomspace(1.0, 12.0, output_bins + 1)
    output = {"wavelength_micron": np.sqrt(edges[:-1] * edges[1:])}
    for case in CASES:
        for geometry in ("emission", "transmission"):
            suffix = "eclipse_depth" if geometry == "emission" else "transit_depth"
            output[f"{case}_robert_{geometry}"] = _bin(
                robert["wavelength_micron"], robert[f"{case}_{suffix}"], edges
            )
            output[f"{case}_picaso_{geometry}"] = _bin(
                picaso["wavelength_micron"], picaso[f"{case}_{suffix}"], edges
            )
    return output


def _metrics(compact):
    metrics = {}
    for case in CASES:
        case_metrics = {}
        for geometry in ("emission", "transmission"):
            robert = compact[f"{case}_robert_{geometry}"]
            picaso = compact[f"{case}_picaso_{geometry}"]
            effect_difference = (
                (robert - compact[f"clear_robert_{geometry}"])
                - (picaso - compact[f"clear_picaso_{geometry}"])
            ) * 1.0e6
            case_metrics[f"{geometry}_absolute_rms_difference_ppm"] = _rms(
                (robert - picaso) * 1.0e6
            )
            case_metrics[f"{geometry}_effect_rms_difference_ppm"] = _rms(
                effect_difference
            )
            case_metrics[f"{geometry}_effect_max_abs_difference_ppm"] = float(
                np.max(np.abs(effect_difference))
            )
        metrics[case] = case_metrics
    return metrics


def _convergence_metrics(compact_by_layer):
    finest_layers = max(compact_by_layer)
    finest = compact_by_layer[finest_layers]
    output = {"finest_layers": finest_layers, "by_layers": {}}
    for layers, compact in compact_by_layer.items():
        layer_metrics = {}
        for framework in ("robert", "picaso"):
            for geometry in ("emission", "transmission"):
                for case in CASES[1:]:
                    effect = (
                        compact[f"{case}_{framework}_{geometry}"]
                        - compact[f"clear_{framework}_{geometry}"]
                    )
                    finest_effect = (
                        finest[f"{case}_{framework}_{geometry}"]
                        - finest[f"clear_{framework}_{geometry}"]
                    )
                    layer_metrics[
                        f"{framework}_{geometry}_{case}_effect_rms_ppm"
                    ] = _rms((effect - finest_effect) * 1.0e6)
        output["by_layers"][str(layers)] = layer_metrics
    return output


def _matched_picaso_tau_transmission(picaso, contract, output_bins):
    """Run ROBERT spherical RT with PICASO's evaluated layer optical depths."""

    pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    pressure = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure,
        unit="bar",
        name="PICASO matched optical-depth diagnostic",
    )
    wavelength = np.asarray(picaso["wavelength_micron"], dtype=float)
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="PICASO evaluated opacity"
    )
    composition = {
        name: np.asarray(contract["gas_vmr"][:, index], dtype=float)
        for index, name in enumerate(SPECIES)
    }
    composition["H2"] = np.full(pressure.size, float(contract["h2_vmr"]))
    composition["He"] = np.asarray(contract["he_vmr"], dtype=float)
    mean_molecular_weight = sum(
        composition[name] * MOLECULAR_WEIGHTS[name] for name in composition
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=0.5
        * (contract["temperature_level_k"][:-1] + contract["temperature_level_k"][1:]),
        temperature_edges=contract["temperature_level_k"],
        composition=composition,
        mean_molecular_weight=mean_molecular_weight,
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="PICASO matched evaluated opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key=f"picaso-matched-{pressure.size}-{wavelength.size}",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure.size, wavelength.size, 1)),
        unit="m^2/molecule",
    )
    gas = assemble_gas_optical_depth(
        atmosphere, opacity, gravity_m_s2=float(contract["gravity_m_s2"])
    )
    gas = replace(
        gas,
        species_tau=None,
        total_tau=np.asarray(picaso["clear_gas_tau"], dtype=float)[:, :, None],
        metadata={**dict(gas.metadata), "source": "PICASO evaluated taugas"},
    )
    hydrostatic = _inverse_square_hydrostatic_profiles(contract)
    geometry_atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=contract["temperature_level_k"][1:],
        temperature_edges=contract["temperature_level_k"],
        composition=composition,
        mean_molecular_weight=hydrostatic["mean_molecular_weight_level"][1:],
    )
    geometry = hydrostatic_path_geometry(
        geometry_atmosphere,
        gravity_m_s2=hydrostatic["geometry_gravity_m_s2"],
        reference_radius_m=float(contract["planet_radius_m"]),
        reference_pressure=float(contract["reference_pressure_bar"]),
        reference_pressure_unit="bar",
    )
    edges = np.geomspace(1.0, 12.0, output_bins + 1)
    robert_depth = {}
    picaso_depth = {}
    for case in CASES:
        additional = ()
        if case != "clear":
            additional = (
                LayerOpticalDepth(
                    name=f"PICASO {case} cloud extinction",
                    tau=np.asarray(picaso[f"{case}_cloud_tau"], dtype=float),
                    spectral_grid=spectral_grid,
                    pressure_grid=pressure_grid,
                ),
            )
        result = solve_absorption_transmission(
            gas,
            geometry,
            star_radius_m=float(contract["star_radius_m"]),
            additional_optical_depths=additional,
            impact_quadrature_order=8,
        )
        robert_depth[case] = _bin(wavelength, result.transit_depth.values, edges)
        picaso_depth[case] = _bin(wavelength, picaso[f"{case}_transit_depth"], edges)
    output = {}
    for case in CASES:
        absolute = (robert_depth[case] - picaso_depth[case]) * 1.0e6
        effect = (
            (robert_depth[case] - robert_depth["clear"])
            - (picaso_depth[case] - picaso_depth["clear"])
        ) * 1.0e6
        output[case] = {
            "absolute_rms_difference_ppm": _rms(absolute),
            "absolute_median_difference_ppm": float(np.median(absolute)),
            "cloud_effect_rms_difference_ppm": _rms(effect),
            "cloud_effect_max_abs_difference_ppm": float(np.max(np.abs(effect))),
        }
    return output


def _plot(path, compact):
    wavelength = compact["wavelength_micron"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for column, geometry in enumerate(("emission", "transmission")):
        for case, alpha in zip(CASES, (0.45, 0.65, 0.8, 1.0), strict=True):
            robert = compact[f"{case}_robert_{geometry}"] * 1.0e6
            picaso = compact[f"{case}_picaso_{geometry}"] * 1.0e6
            axes[0, column].plot(wavelength, picaso, color="black", alpha=alpha, lw=1.0)
            axes[0, column].plot(wavelength, robert, color="#6f2dbd", alpha=alpha, lw=1.0)
            effect = (
                (compact[f"{case}_robert_{geometry}"] - compact[f"clear_robert_{geometry}"])
                - (compact[f"{case}_picaso_{geometry}"] - compact[f"clear_picaso_{geometry}"])
            ) * 1.0e6
            axes[1, column].plot(wavelength, effect, alpha=alpha, lw=1.0, label=case)
        axes[0, column].set_title(geometry.capitalize())
        axes[0, column].set_ylabel("Depth (ppm)")
        axes[1, column].set_ylabel("Cloud-effect difference (ppm)")
        axes[1, column].set_xlabel("Wavelength (micron)")
        axes[1, column].axhline(0.0, color="0.5", lw=0.7)
        axes[1, column].legend(frameon=False, fontsize=8, ncol=2)
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
    figure.suptitle("Shared deck/haze: ROBERT (purple) vs official PICASO (black)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
