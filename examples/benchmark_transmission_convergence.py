#!/usr/bin/env python3
"""Benchmark transmission convergence against native ExoMolOP H2O opacity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets.atmosphere import (
    BackgroundGasMixture,
    CompositionMeanMolecularWeight,
    FreeChemistry,
    IsothermalTemperatureProfile,
)
from robert_exoplanets.bodies import Planet, Star
from robert_exoplanets.core import PressureGrid, Spectrum
from robert_exoplanets.forward import (
    ParameterizedTransmissionFactoryConfig,
    ParameterizedTransmissionModelConfig,
    build_parameterized_transmission_model,
)
from robert_exoplanets.instruments import (
    Observation,
    TopHatObservationResponse,
    infer_wavelength_bin_edges,
)
from robert_exoplanets.opacity import (
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    OpacitySamplingProvider,
    file_sha256,
)
from robert_exoplanets.rt import CiaTable, load_nemesispy_cia_table

PARAMETERS = {"log_H2O": -3.3, "radius_scale": 1.003}
G_POINTS = (2, 4, 8, 16, 32, 64)
PRESSURE_LAYERS = (24, 48, 80, 120, 160, 240)
IMPACT_ORDERS = (2, 4, 6, 8, 12, 16, 24)
REFERENCE_LAYERS = 240
REFERENCE_IMPACT_ORDER = 24


def _observation() -> Observation:
    wavelength = np.geomspace(0.85, 5.0, 32)
    return Observation(
        wavelength=wavelength,
        wavelength_bin_edges=infer_wavelength_bin_edges(wavelength),
        flux=np.full(wavelength.shape, 0.018),
        uncertainty=np.full(wavelength.shape, 12.0e-6),
        wavelength_unit="micron",
        flux_unit="transit_depth",
        observable="transit_depth",
        instrument="Synthetic spectrograph",
    )


def _model(
    provider,
    spectral_grid,
    *,
    layers: int,
    impact_order: int,
    cia: CiaTable | None,
    include_rayleigh: bool,
):
    chemistry = FreeChemistry(
        active_species=("H2O",),
        background=BackgroundGasMixture({"H2": 0.8547, "He": 0.1453}),
        parameter_names={"H2O": "log_H2O"},
        parameter_mode="log10",
        fill_background=True,
    )
    factory = ParameterizedTransmissionFactoryConfig(
        planet=Planet(
            name="Synthetic hot Jupiter b",
            radius_m=8.0e7,
            mass_kg=1.1502935e27,
        ),
        star=Star(name="Synthetic K dwarf", radius_m=6.0e8),
        temperature_profile=IsothermalTemperatureProfile(temperature=1100.0),
        chemistry_model=chemistry,
        mean_molecular_weight_model=CompositionMeanMolecularWeight(
            normalization="require"
        ),
        pressure_grid=PressureGrid.from_log_centers(
            10.0,
            1.0e-7,
            n_layers=layers,
            unit="bar",
        ),
        cia_table=cia,
        opacity_source=provider,
        opacity_binning=None,
        model=ParameterizedTransmissionModelConfig(
            opacity_species=("H2O",),
            reference_pressure_bar=1.0,
            radius_scale_parameter="radius_scale",
            gravity_model="inverse_square",
            include_rayleigh=include_rayleigh,
            gas_combination="random_overlap",
            impact_quadrature_order=impact_order,
        ),
    )
    return build_parameterized_transmission_model(factory, spectral_grid=spectral_grid)


def _native_prediction(
    provider: OpacitySamplingProvider,
    native_grid,
    observation: Observation,
    *,
    layers: int,
    impact_order: int,
    cia: CiaTable | None,
    include_rayleigh: bool = True,
) -> tuple[Spectrum, float]:
    started = perf_counter()
    model = _model(
        provider,
        native_grid,
        layers=layers,
        impact_order=impact_order,
        cia=cia,
        include_rayleigh=include_rayleigh,
    )
    native = model(PARAMETERS)
    binned = TopHatObservationResponse().prepare(observation).observe(native)
    return binned, perf_counter() - started


def _correlated_k_prediction(
    source: Path,
    observation: Observation,
    *,
    g_points: int,
    layers: int,
    impact_order: int,
    cia: CiaTable | None,
    include_rayleigh: bool = True,
) -> tuple[Spectrum, float, float]:
    started = perf_counter()
    table = CorrelatedKTable.from_exomol_cross_section_hdf(
        source,
        species="H2O",
        spectral_grid=observation.spectral_grid,
        g_points=g_points,
        checksum=False,
    )
    preparation_seconds = perf_counter() - started
    provider = CorrelatedKOpacityProvider(
        {"H2O": table},
        name=f"ExoMolOP-POKAZATEL-target-bin-g{g_points}",
        interpolation="log_pressure_temperature_log_k_clip",
    )
    started = perf_counter()
    model = _model(
        provider,
        observation.spectral_grid,
        layers=layers,
        impact_order=impact_order,
        cia=cia,
        include_rayleigh=include_rayleigh,
    )
    spectrum = model(PARAMETERS)
    evaluation_seconds = perf_counter() - started
    return spectrum, preparation_seconds, evaluation_seconds


def _metrics(spectrum: Spectrum, reference: Spectrum) -> dict[str, float]:
    residual_ppm = (spectrum.values - reference.values) * 1.0e6
    return {
        "rms_ppm": float(np.sqrt(np.mean(np.square(residual_ppm)))),
        "max_abs_ppm": float(np.max(np.abs(residual_ppm))),
        "mean_bias_ppm": float(np.mean(residual_ppm)),
    }


def run_benchmark(source: Path, output: Path) -> dict[str, object]:
    observation = _observation()
    cia = load_nemesispy_cia_table()
    sampling = OpacitySamplingProvider.from_exomol_paths(
        {"H2O": source},
        name="ExoMolOP-H2O-POKAZATEL-R15000",
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=False,
    )
    edges = observation.wavelength_bin_edges
    assert edges is not None
    native_grid = sampling.native_spectral_grid(
        wavelength_bounds_micron=(float(edges[0]) * 0.999, float(edges[-1]) * 1.001),
        name="ExoMolOP H2O POKAZATEL R=15000",
    )
    cache: dict[tuple[int, int], tuple[Spectrum, float]] = {}

    def native(layers: int, impact_order: int) -> tuple[Spectrum, float]:
        key = (layers, impact_order)
        if key not in cache:
            cache[key] = _native_prediction(
                sampling,
                native_grid,
                observation,
                layers=layers,
                impact_order=impact_order,
                cia=cia,
            )
        return cache[key]

    reference, reference_seconds = native(
        REFERENCE_LAYERS,
        REFERENCE_IMPACT_ORDER,
    )
    molecular_reference, molecular_reference_seconds = _native_prediction(
        sampling,
        native_grid,
        observation,
        layers=REFERENCE_LAYERS,
        impact_order=REFERENCE_IMPACT_ORDER,
        cia=None,
        include_rayleigh=False,
    )
    layer_results = []
    for layers in PRESSURE_LAYERS:
        spectrum, elapsed = native(layers, REFERENCE_IMPACT_ORDER)
        layer_results.append(
            {"value": layers, "runtime_seconds": elapsed, **_metrics(spectrum, reference)}
        )
    impact_results = []
    for order in IMPACT_ORDERS:
        spectrum, elapsed = native(REFERENCE_LAYERS, order)
        impact_results.append(
            {"value": order, "runtime_seconds": elapsed, **_metrics(spectrum, reference)}
        )
    g_results = []
    molecular_g_results = []
    correlated: dict[int, Spectrum] = {}
    for points in G_POINTS:
        spectrum, preparation, evaluation = _correlated_k_prediction(
            source,
            observation,
            g_points=points,
            layers=REFERENCE_LAYERS,
            impact_order=REFERENCE_IMPACT_ORDER,
            cia=cia,
        )
        correlated[points] = spectrum
        g_results.append(
            {
                "value": points,
                "preparation_seconds": preparation,
                "runtime_seconds": evaluation,
                **_metrics(spectrum, reference),
            }
        )
        molecular_spectrum, molecular_preparation, molecular_evaluation = (
            _correlated_k_prediction(
                source,
                observation,
                g_points=points,
                layers=REFERENCE_LAYERS,
                impact_order=REFERENCE_IMPACT_ORDER,
                cia=None,
                include_rayleigh=False,
            )
        )
        molecular_g_results.append(
            {
                "value": points,
                "preparation_seconds": molecular_preparation,
                "runtime_seconds": molecular_evaluation,
                **_metrics(molecular_spectrum, molecular_reference),
            }
        )
    production, _, production_seconds = _correlated_k_prediction(
        source,
        observation,
        g_points=8,
        layers=48,
        impact_order=6,
        cia=cia,
    )
    production_metrics = {
        "g_points": 8,
        "pressure_layers": 48,
        "impact_quadrature_order": 6,
        "runtime_seconds": production_seconds,
        **_metrics(production, reference),
    }
    recommended, _, recommended_seconds = _correlated_k_prediction(
        source,
        observation,
        g_points=16,
        layers=80,
        impact_order=4,
        cia=cia,
    )
    recommended_metrics = {
        "g_points": 16,
        "pressure_layers": 80,
        "impact_quadrature_order": 4,
        "runtime_seconds": recommended_seconds,
        **_metrics(recommended, reference),
    }
    report = {
        "benchmark": "transmission_native_opacity_convergence",
        "source": {
            "path": str(source),
            "sha256": file_sha256(source),
            "database": "ExoMolOP",
            "line_list": "1H2-16O__POKAZATEL",
            "doi": "10.1093/mnras/sty1877",
            "native_resolving_power": 15000,
            "native_samples": native_grid.size,
        },
        "reference": {
            "opacity": "native opacity sampling",
            "pressure_layers": REFERENCE_LAYERS,
            "impact_quadrature_order": REFERENCE_IMPACT_ORDER,
            "runtime_seconds": reference_seconds,
            "molecular_only_runtime_seconds": molecular_reference_seconds,
        },
        "parameters": PARAMETERS,
        "g_point_sweep": molecular_g_results,
        "g_point_sweep_full_physics": g_results,
        "pressure_layer_sweep": layer_results,
        "impact_quadrature_sweep": impact_results,
        "production_candidate": production_metrics,
        "recommended_production_default": recommended_metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "transmission_convergence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(report, output / "transmission_convergence.csv")
    _plot_convergence(report, output / "transmission_convergence.png")
    _plot_spectra(
        observation,
        reference,
        correlated,
        production,
        output / "transmission_native_vs_correlated_k.png",
    )
    return report


def _write_csv(report: dict[str, object], path: Path) -> None:
    rows = []
    for key, label in (
        ("g_point_sweep", "g_points"),
        ("g_point_sweep_full_physics", "g_points_full_physics"),
        ("pressure_layer_sweep", "pressure_layers"),
        ("impact_quadrature_sweep", "impact_quadrature_order"),
    ):
        for item in report[key]:
            rows.append({"sweep": label, **item})
    fields = (
        "sweep",
        "value",
        "rms_ppm",
        "max_abs_ppm",
        "mean_bias_ppm",
        "runtime_seconds",
        "preparation_seconds",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_convergence(report: dict[str, object], path: Path) -> None:
    sweeps = (
        ("g_point_sweep", "Correlated-k g points"),
        ("pressure_layer_sweep", "Pressure layers"),
        ("impact_quadrature_sweep", "Impact quadrature order"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for axis, (key, label) in zip(axes, sweeps, strict=True):
        rows = report[key]
        values = np.array([item["value"] for item in rows])
        rms = np.maximum([item["rms_ppm"] for item in rows], 1.0e-4)
        maximum = np.maximum([item["max_abs_ppm"] for item in rows], 1.0e-4)
        axis.plot(values, rms, "o-", label="RMS")
        axis.plot(values, maximum, "s--", label="Maximum absolute")
        axis.axhline(10.0, color="0.4", linestyle=":", label="10 ppm")
        axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("Difference from native reference (ppm)")
        axis.grid(alpha=0.25)
    full_rows = report["g_point_sweep_full_physics"]
    axes[0].plot(
        [item["value"] for item in full_rows],
        np.maximum([item["rms_ppm"] for item in full_rows], 1.0e-4),
        color="tab:purple",
        marker="^",
        linestyle="-.",
        label="Full-physics RMS",
    )
    axes[0].legend()
    figure.suptitle("ROBERT transmission convergence: ExoMolOP H2O POKAZATEL")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_spectra(
    observation: Observation,
    reference: Spectrum,
    correlated: dict[int, Spectrum],
    production: Spectrum,
    path: Path,
) -> None:
    wavelength = observation.wavelength
    figure, (spectrum_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
    )
    spectrum_axis.plot(wavelength, reference.values * 1.0e6, "k-", label="Native R=15,000")
    for points, spectrum in correlated.items():
        spectrum_axis.plot(
            wavelength,
            spectrum.values * 1.0e6,
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            label=f"Correlated-k g={points}",
        )
        residual_axis.plot(
            wavelength,
            (spectrum.values - reference.values) * 1.0e6,
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            label=f"g={points}",
        )
    residual_axis.plot(
        wavelength,
        (production.values - reference.values) * 1.0e6,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Current 48/6/g8",
    )
    spectrum_axis.set_ylabel("Transit depth (ppm)")
    spectrum_axis.legend(ncol=2, fontsize=8)
    spectrum_axis.grid(alpha=0.2)
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.axhspan(-10.0, 10.0, color="0.8", alpha=0.35)
    residual_axis.set_xlabel("Wavelength (micron)")
    residual_axis.set_ylabel("Model - native (ppm)")
    residual_axis.legend(ncol=3, fontsize=8)
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("opacity_data/exomol_xsec/H2O.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/transmission_convergence"),
    )
    args = parser.parse_args()
    report = run_benchmark(args.source, args.output)
    print(json.dumps(report["recommended_production_default"], indent=2))


if __name__ == "__main__":
    main()
