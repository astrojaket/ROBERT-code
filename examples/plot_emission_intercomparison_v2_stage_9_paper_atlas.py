#!/usr/bin/env python3
"""Generate compact publication atlases for all completed Stage-9 retrievals.

The atlas is deliberately distinct from the per-run audit diagnostics. For
each atmospheric scenario it compresses all 18 directed retrievals into:

* a 3x3 normalized spectral-residual atlas;
* a 3x3 temperature-pressure recovery atlas;
* a 3x4 chemistry-posterior atlas; and
* for cloudy scenarios, a config-driven cloud-posterior atlas.

Only saved compact products are read. No opacity, forward-model, sampler, or
science workload is evaluated by this plotting script.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import string
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_intercomparison_v2_stage_9_native import (  # noqa: E402
    atmospheric_state,
    load_common_contract,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    FRAMEWORKS,
    NOISE_TIERS_PPM,
    SCENARIOS,
)


MODEL_COLORS = {
    "robert": "#9370DB",  # mediumpurple
    "petitradtrans": "#DDA0DD",  # plum
    "picaso": "#36454F",  # charcoal
}
DISPLAY_NAMES = {
    "robert": "ROBERT",
    "petitradtrans": "petitRADTRANS",
    "picaso": "PICASO",
}
SHORT_NAMES = {
    "robert": "ROBERT",
    "petitradtrans": "pRT",
    "picaso": "PICASO",
}
INJECTOR_LINESTYLES = {
    "robert": "-",
    "petitradtrans": "--",
    "picaso": ":",
}
BIAS_CLASS_COLORS = {
    "within_1sigma": "#5B8E7D",
    "within_2sigma": "#E0A458",
    "outside_2sigma": "#B84A62",
}
BIAS_CLASS_LABELS = {
    "within_1sigma": "Injected value inside central 68% interval",
    "within_2sigma": "Outside central 68%, inside central 95%",
    "outside_2sigma": "Injected value outside central 95% interval",
}
SCIENTIFIC_PARAMETER_LABELS = {
    "log10_vmr_H2O": r"$\log_{10}\,\mathrm{VMR}(\mathrm{H_2O})$",
    "log10_vmr_CO": r"$\log_{10}\,\mathrm{VMR}(\mathrm{CO})$",
    "log10_vmr_CO2": r"$\log_{10}\,\mathrm{VMR}(\mathrm{CO_2})$",
    "log10_vmr_CH4": r"$\log_{10}\,\mathrm{VMR}(\mathrm{CH_4})$",
    "log10_cloud_tau_5um": (
        r"$\log_{10}\tau_{\mathrm{cloud}}(5\,\mu\mathrm{m})$"
    ),
    "log10_cloud_top_pressure_bar": (
        r"$\log_{10}P_{\mathrm{cloud,top}}\ [\mathrm{bar}]$"
    ),
    "cloud_single_scattering_albedo": r"$\omega_0$",
}
SCENARIO_LABELS = {
    "clear_non_inverted": "Clear, non-inverted",
    "clear_inverted": "Clear, inverted",
    "grey_absorbing_non_inverted": "Grey absorbing cloud, non-inverted",
    "grey_scattering_non_inverted": "Grey scattering cloud, non-inverted",
}
SPECTRAL_KEYS = (
    "posterior_spectrum_q025_eclipse_depth",
    "posterior_spectrum_q16_eclipse_depth",
    "posterior_spectrum_q50_eclipse_depth",
    "posterior_spectrum_q84_eclipse_depth",
    "posterior_spectrum_q975_eclipse_depth",
)
TP_KEYS = (
    "posterior_temperature_q025_k",
    "posterior_temperature_q16_k",
    "posterior_temperature_q50_k",
    "posterior_temperature_q84_k",
    "posterior_temperature_q975_k",
)


def _parameter_label(item: Mapping[str, Any]) -> str:
    scientific = SCIENTIFIC_PARAMETER_LABELS.get(str(item["name"]))
    if scientific is not None:
        return scientific
    label = str(item.get("label") or item["name"])
    unit = item.get("unit")
    return f"{label} [{unit}]" if unit else label


def _reference_value(item: Mapping[str, Any]) -> float | None:
    value = item.get("reference_value", item.get("truth"))
    return None if value is None else float(value)


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, values[order]))


def _required_npz_keys(path: Path, keys: Sequence[str]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive.files]
    if missing:
        raise RuntimeError(
            f"paper atlas requires current posterior envelopes in {path}; "
            f"missing: {', '.join(missing)}"
        )


def _load_completed_runs(
    project: Path,
    scenario_filter: str | None,
) -> list[dict[str, Any]]:
    rows = json.loads((project / "run_index.json").read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        config_path = project / row["run_config"]
        run = json.loads(config_path.read_text(encoding="utf-8"))
        if scenario_filter is not None and run["scenario"] != scenario_filter:
            continue
        run_directory = Path(run["run_directory"])
        products = {
            "result": run_directory / "result.json",
            "arrays": run_directory / "result_arrays.npz",
            "spectra": run_directory / "diagnostic_spectra.npz",
            "tp": run_directory / "diagnostic_tp.npz",
            "chemistry": run_directory / "diagnostic_chemistry.npz",
            "summary": run_directory / "posterior_summary.json",
        }
        absent = [
            name
            for name, path in products.items()
            if not path.is_file() or path.stat().st_size == 0
        ]
        if absent:
            missing.append(f"{run['run_id']}: {', '.join(absent)}")
            continue
        _required_npz_keys(
            products["spectra"],
            (
                "wavelength_micron",
                "observed_eclipse_depth",
                "observational_uncertainty_eclipse_depth",
                "best_fit_eclipse_depth",
                *SPECTRAL_KEYS,
            ),
        )
        _required_npz_keys(
            products["tp"],
            ("pressure_bar", "best_fit_temperature_k", *TP_KEYS),
        )
        result = json.loads(products["result"].read_text(encoding="utf-8"))
        run.update(products)
        run["config_path"] = config_path
        run["parameter_names"] = tuple(str(name) for name in result["parameter_names"])
        runs.append(run)
    if missing:
        preview = "\n".join(missing[:8])
        remainder = len(missing) - min(len(missing), 8)
        suffix = f"\n... and {remainder} more" if remainder else ""
        raise RuntimeError(
            "paper atlas requires every selected retrieval to be complete:\n"
            + preview
            + suffix
        )

    scenarios = (
        [scenario_filter]
        if scenario_filter is not None
        else [item.name for item in SCENARIOS]
    )
    for scenario in scenarios:
        selected = [run for run in runs if run["scenario"] == scenario]
        if len(selected) != 18:
            raise RuntimeError(
                f"paper atlas requires 18 completed runs for {scenario}; "
                f"found {len(selected)}"
            )
        identities = {
            (int(run["noise_ppm"]), run["injector"], run["retriever"])
            for run in selected
        }
        expected = {
            (tier, injector, retriever)
            for tier in NOISE_TIERS_PPM
            for injector in FRAMEWORKS
            for retriever in FRAMEWORKS
            if retriever != injector
        }
        if identities != expected:
            raise RuntimeError(f"directed retrieval matrix is incomplete for {scenario}")
    return runs


def _select_cell(
    runs: Sequence[dict[str, Any]],
    *,
    scenario: str,
    tier: int,
    injector: str,
) -> list[dict[str, Any]]:
    return sorted(
        (
            run
            for run in runs
            if run["scenario"] == scenario
            and int(run["noise_ppm"]) == tier
            and run["injector"] == injector
        ),
        key=lambda run: FRAMEWORKS.index(run["retriever"]),
    )


def _panel_label(axis: plt.Axes, index: int) -> None:
    axis.text(
        0.015,
        0.98,
        f"({string.ascii_lowercase[index]})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def _spectral_metrics(run: Mapping[str, Any]) -> tuple[float, float]:
    summary = json.loads(Path(run["summary"]).read_text(encoding="utf-8"))
    metrics = summary.get("fit_metrics", {})
    reduced = metrics.get("best_fit_reduced_chi_square")
    rms = metrics.get("best_fit_residual_rms_ppm")
    if reduced is not None and rms is not None:
        return float(reduced), float(rms)
    with np.load(run["spectra"], allow_pickle=False) as archive:
        observed = np.asarray(archive["observed_eclipse_depth"], dtype=float)
        uncertainty = np.asarray(
            archive["observational_uncertainty_eclipse_depth"], dtype=float
        )
        best = np.asarray(archive["best_fit_eclipse_depth"], dtype=float)
    parameter_count = len(run["parameter_names"])
    degrees_of_freedom = max(observed.size - parameter_count, 1)
    residual = best - observed
    return (
        float(np.sum((residual / uncertainty) ** 2) / degrees_of_freedom),
        float(np.sqrt(np.mean(residual**2)) * 1.0e6),
    )


def _spectral_residual_atlas(
    runs: Sequence[dict[str, Any]],
    scenario: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(13.2, 11.2), constrained_layout=True)
    outer = fig.add_gridspec(3, 3)
    spectrum_axes: list[plt.Axes] = []
    residual_axes: list[plt.Axes] = []
    maximum = 3.0
    panel = 0
    for row, tier in enumerate(NOISE_TIERS_PPM):
        for column, injector in enumerate(FRAMEWORKS):
            inner = outer[row, column].subgridspec(
                2,
                1,
                height_ratios=(2.15, 1.0),
                hspace=0.04,
            )
            spectrum_axis = fig.add_subplot(inner[0])
            residual_axis = fig.add_subplot(inner[1], sharex=spectrum_axis)
            spectrum_axes.append(spectrum_axis)
            residual_axes.append(residual_axis)
            residual_axis.axhspan(
                -1.0,
                1.0,
                color="#808080",
                alpha=0.08,
                linewidth=0.0,
            )
            residual_axis.axhline(0.0, color="#202020", lw=0.65, alpha=0.7)
            observation_drawn = False
            for annotation_row, run in enumerate(
                _select_cell(
                    runs,
                    scenario=scenario,
                    tier=tier,
                    injector=injector,
                )
            ):
                retriever = str(run["retriever"])
                color = MODEL_COLORS[retriever]
                with np.load(run["spectra"], allow_pickle=False) as archive:
                    wavelength = np.asarray(
                        archive["wavelength_micron"], dtype=float
                    )
                    observed = np.asarray(
                        archive["observed_eclipse_depth"], dtype=float
                    )
                    uncertainty = np.asarray(
                        archive["observational_uncertainty_eclipse_depth"],
                        dtype=float,
                    )
                    best = np.asarray(
                        archive["best_fit_eclipse_depth"], dtype=float
                    )
                    q025 = np.asarray(archive[SPECTRAL_KEYS[0]], dtype=float)
                    q16 = np.asarray(archive[SPECTRAL_KEYS[1]], dtype=float)
                    q84 = np.asarray(archive[SPECTRAL_KEYS[3]], dtype=float)
                    q975 = np.asarray(archive[SPECTRAL_KEYS[4]], dtype=float)
                if not observation_drawn:
                    spectrum_axis.errorbar(
                        wavelength,
                        observed * 1.0e6,
                        yerr=uncertainty * 1.0e6,
                        fmt="o",
                        ms=1.5,
                        mew=0.0,
                        color="#202020",
                        ecolor="#707070",
                        elinewidth=0.45,
                        capsize=0.0,
                        alpha=0.85,
                        rasterized=True,
                        zorder=6,
                    )
                    observation_drawn = True
                spectrum_axis.fill_between(
                    wavelength,
                    q025 * 1.0e6,
                    q975 * 1.0e6,
                    color=color,
                    alpha=0.07,
                    linewidth=0.0,
                )
                spectrum_axis.fill_between(
                    wavelength,
                    q16 * 1.0e6,
                    q84 * 1.0e6,
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                )
                spectrum_axis.plot(
                    wavelength,
                    best * 1.0e6,
                    color=color,
                    lw=1.2,
                    zorder=5,
                )
                normalized = {
                    "best": (best - observed) / uncertainty,
                    "q025": (q025 - observed) / uncertainty,
                    "q16": (q16 - observed) / uncertainty,
                    "q84": (q84 - observed) / uncertainty,
                    "q975": (q975 - observed) / uncertainty,
                }
                maximum = max(
                    maximum,
                    float(np.max(np.abs(normalized["q025"]))),
                    float(np.max(np.abs(normalized["q975"]))),
                )
                residual_axis.fill_between(
                    wavelength,
                    normalized["q025"],
                    normalized["q975"],
                    color=color,
                    alpha=0.07,
                    linewidth=0.0,
                )
                residual_axis.fill_between(
                    wavelength,
                    normalized["q16"],
                    normalized["q84"],
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                )
                residual_axis.plot(
                    wavelength,
                    normalized["best"],
                    color=color,
                    lw=1.25,
                )
                reduced, rms = _spectral_metrics(run)
                residual_axis.text(
                    0.985,
                    0.96 - 0.22 * annotation_row,
                    rf"{SHORT_NAMES[retriever]}: $\chi^2_\nu$={reduced:.2f}, "
                    f"RMS={rms:.1f} ppm",
                    transform=residual_axis.transAxes,
                    color=color,
                    ha="right",
                    va="top",
                    fontsize=7,
                )
            if row == 0:
                spectrum_axis.set_title(
                    f"Data generated with {DISPLAY_NAMES[injector]}"
                )
            if column == 0:
                spectrum_axis.set_ylabel(f"{tier} ppm\nEclipse depth [ppm]")
                residual_axis.set_ylabel("Δ / σ")
            if row == 2:
                residual_axis.set_xlabel("Wavelength [µm]")
            else:
                residual_axis.tick_params(labelbottom=False)
            spectrum_axis.tick_params(labelbottom=False)
            spectrum_axis.grid(alpha=0.15)
            residual_axis.grid(alpha=0.15)
            _panel_label(spectrum_axis, panel)
            panel += 1
    limit = 1.05 * maximum
    for axis in residual_axes:
        axis.set_ylim(-limit, limit)
    handles = [
        Line2D(
            [0],
            [0],
            color="#202020",
            marker="o",
            lw=0.6,
            ms=3,
            label="Observed data ±1σ",
        )
    ]
    handles.extend(
        Line2D([0], [0], color=MODEL_COLORS[name], lw=1.5, label=DISPLAY_NAMES[name])
        for name in FRAMEWORKS
    )
    handles.extend(
        (
            Patch(facecolor="#777777", alpha=0.18, label="Central 68% posterior"),
            Patch(facecolor="#777777", alpha=0.07, label="Central 95% posterior"),
        )
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=6,
        fontsize=8,
        title=f"{SCENARIO_LABELS[scenario]} — spectra and normalized residuals",
        title_fontsize=11,
    )
    return fig


def _reference_parameters(run: Mapping[str, Any]) -> dict[str, float]:
    configured = run.get("parameters")
    if not isinstance(configured, list):
        raise RuntimeError(f"{run['run_id']} has no parameter metadata")
    values = {
        str(item["name"]): _reference_value(item)
        for item in configured
        if isinstance(item, dict) and "name" in item
    }
    if not values or any(value is None for value in values.values()):
        raise RuntimeError(
            f"Stage-9 paper atlas requires reference values in {run['config_path']}"
        )
    return {name: float(value) for name, value in values.items()}


def _tp_atlas(
    runs: Sequence[dict[str, Any]],
    common: Mapping[str, Any],
    scenario: str,
) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.2, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    first = next(run for run in runs if run["scenario"] == scenario)
    reference = atmospheric_state(
        common,
        scenario,
        _reference_parameters(first),
    ).temperature_cells_k
    panel = 0
    temperatures = [reference]
    for row, tier in enumerate(NOISE_TIERS_PPM):
        for column, injector in enumerate(FRAMEWORKS):
            axis = axes[row, column]
            for annotation_row, run in enumerate(
                _select_cell(
                    runs,
                    scenario=scenario,
                    tier=tier,
                    injector=injector,
                )
            ):
                retriever = str(run["retriever"])
                color = MODEL_COLORS[retriever]
                with np.load(run["tp"], allow_pickle=False) as archive:
                    pressure = np.asarray(archive["pressure_bar"], dtype=float)
                    best = np.asarray(
                        archive["best_fit_temperature_k"], dtype=float
                    )
                    q025 = np.asarray(archive[TP_KEYS[0]], dtype=float)
                    q16 = np.asarray(archive[TP_KEYS[1]], dtype=float)
                    q84 = np.asarray(archive[TP_KEYS[3]], dtype=float)
                    q975 = np.asarray(archive[TP_KEYS[4]], dtype=float)
                temperatures.extend((q025, q975))
                axis.fill_betweenx(
                    pressure,
                    q025,
                    q975,
                    color=color,
                    alpha=0.07,
                    linewidth=0.0,
                )
                axis.fill_betweenx(
                    pressure,
                    q16,
                    q84,
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                )
                axis.plot(best, pressure, color=color, lw=1.25)
                rms = float(np.sqrt(np.mean((best - reference) ** 2)))
                axis.text(
                    0.985,
                    0.97 - 0.105 * annotation_row,
                    f"{SHORT_NAMES[retriever]}: ΔT RMS={rms:.0f} K",
                    transform=axis.transAxes,
                    color=color,
                    ha="right",
                    va="top",
                    fontsize=7,
                )
            axis.plot(
                reference,
                pressure,
                color="#202020",
                lw=1.1,
                ls="--",
                zorder=5,
            )
            if row == 0:
                axis.set_title(f"Data generated with {DISPLAY_NAMES[injector]}")
            if column == 0:
                axis.set_ylabel(f"{tier} ppm\nPressure [bar]")
            if row == 2:
                axis.set_xlabel("Temperature [K]")
            axis.set_yscale("log")
            axis.invert_yaxis()
            axis.grid(alpha=0.15)
            _panel_label(axis, panel)
            panel += 1
    all_temperature = np.concatenate(temperatures)
    padding = 0.04 * (float(np.max(all_temperature)) - float(np.min(all_temperature)))
    for axis in axes.flat:
        axis.set_xlim(
            float(np.min(all_temperature)) - padding,
            float(np.max(all_temperature)) + padding,
        )
    handles = [
        Line2D([0], [0], color="#202020", lw=1.2, ls="--", label="Injected TP")
    ]
    handles.extend(
        Line2D([0], [0], color=MODEL_COLORS[name], lw=1.5, label=DISPLAY_NAMES[name])
        for name in FRAMEWORKS
    )
    handles.extend(
        (
            Patch(facecolor="#777777", alpha=0.18, label="Central 68% posterior"),
            Patch(facecolor="#777777", alpha=0.07, label="Central 95% posterior"),
        )
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=6,
        fontsize=8,
        title=f"{SCENARIO_LABELS[scenario]} — temperature-pressure recovery",
        title_fontsize=11,
    )
    return fig


def _parameter_groups(
    runs: Sequence[dict[str, Any]],
    scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first = next(run for run in runs if run["scenario"] == scenario)
    configured = first.get("parameters")
    if not isinstance(configured, list):
        raise RuntimeError(f"{first['run_id']} has no parameter metadata")
    chemistry = [
        item
        for item in configured
        if isinstance(item, dict)
        and str(item.get("name", "")).startswith("log10_vmr_")
    ]
    clouds = [
        item
        for item in configured
        if isinstance(item, dict)
        and (
            "cloud" in str(item.get("name", ""))
            or "scattering_albedo" in str(item.get("name", ""))
        )
    ]
    if not chemistry:
        raise RuntimeError(f"no chemistry parameters are configured for {scenario}")
    return chemistry, clouds


def _posterior_cache(
    runs: Sequence[dict[str, Any]],
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    cache = {}
    for run in runs:
        with np.load(run["arrays"], allow_pickle=False) as archive:
            samples = np.asarray(archive["samples"], dtype=float)
            weights = (
                np.asarray(archive["weights"], dtype=float)
                if "weights" in archive.files
                else np.ones(samples.shape[0], dtype=float)
            )
        weights /= np.sum(weights)
        cache[str(run["run_id"])] = (samples, weights)
    return cache


def _parameter_family(name: str) -> str:
    if name.startswith("log10_vmr_"):
        return "chemistry"
    if "cloud" in name or "scattering_albedo" in name:
        return "cloud"
    return "temperature_profile"


def _bias_coverage_records(
    runs: Sequence[dict[str, Any]],
    cache: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in runs:
        configured = run.get("parameters")
        if not isinstance(configured, list):
            raise RuntimeError(f"{run['run_id']} has no parameter metadata")
        metadata = {
            str(item["name"]): item
            for item in configured
            if isinstance(item, dict) and "name" in item
        }
        samples, weights = cache[str(run["run_id"])]
        for index, name in enumerate(run["parameter_names"]):
            if name not in metadata:
                raise RuntimeError(f"{run['run_id']} has no metadata for {name}")
            parameter = metadata[name]
            family = _parameter_family(name)
            if family == "temperature_profile":
                continue
            injected = _reference_value(parameter)
            if injected is None:
                continue
            values = samples[:, index]
            q025, q16, q50, q84, q975 = (
                _weighted_quantile(values, weights, quantile)
                for quantile in (0.025, 0.16, 0.50, 0.84, 0.975)
            )
            mean = float(np.sum(weights * values))
            standard_deviation = float(
                np.sqrt(np.sum(weights * (values - mean) ** 2))
            )
            median_bias = q50 - injected
            normalized_bias = (
                median_bias / standard_deviation
                if standard_deviation > 0.0
                else float("nan")
            )
            within_68 = bool(q16 <= injected <= q84)
            within_95 = bool(q025 <= injected <= q975)
            classification = (
                "within_1sigma"
                if within_68
                else "within_2sigma"
                if within_95
                else "outside_2sigma"
            )
            records.append(
                {
                    "scenario": run["scenario"],
                    "noise_ppm": int(run["noise_ppm"]),
                    "injector": run["injector"],
                    "retriever": run["retriever"],
                    "run_id": run["run_id"],
                    "parameter": name,
                    "parameter_family": family,
                    "injected_value": injected,
                    "posterior_q025": q025,
                    "posterior_q16": q16,
                    "posterior_median": q50,
                    "posterior_q84": q84,
                    "posterior_q975": q975,
                    "posterior_mean": mean,
                    "posterior_standard_deviation": standard_deviation,
                    "median_bias": median_bias,
                    "normalized_median_bias_sigma": normalized_bias,
                    "truth_within_central_68_percent": within_68,
                    "truth_within_central_95_percent": within_95,
                    "sigma_classification": classification,
                }
            )
    return records


def _aggregate_bias_coverage(
    records: Sequence[dict[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record[name] for name in group_keys)
        grouped.setdefault(key, []).append(record)
    summaries: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        normalized = np.asarray(
            [record["normalized_median_bias_sigma"] for record in selected],
            dtype=float,
        )
        within_68 = sum(
            bool(record["truth_within_central_68_percent"])
            for record in selected
        )
        within_95 = sum(
            bool(record["truth_within_central_95_percent"])
            for record in selected
        )
        summary = dict(zip(group_keys, key, strict=True))
        summary.update(
            {
                "retrieved_parameter_count": len(selected),
                "within_central_68_percent_count": within_68,
                "within_central_68_percent_fraction": within_68 / len(selected),
                "within_central_95_percent_count": within_95,
                "within_central_95_percent_fraction": within_95 / len(selected),
                "outside_central_95_percent_count": len(selected) - within_95,
                "outside_central_95_percent_fraction": (
                    len(selected) - within_95
                )
                / len(selected),
                "median_absolute_normalized_bias_sigma": float(
                    np.nanmedian(np.abs(normalized))
                ),
                "maximum_absolute_normalized_bias_sigma": float(
                    np.nanmax(np.abs(normalized))
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _write_bias_coverage_reports(
    destination: Path,
    records: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    csv_path = destination / "parameter_bias_coverage.csv"
    json_path = destination / "parameter_bias_coverage.json"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "schema_version": "1.0",
        "definitions": {
            "normalized_median_bias_sigma": (
                "(posterior median - injected value) / weighted posterior "
                "standard deviation"
            ),
            "sigma_classification": BIAS_CLASS_LABELS,
            "note": (
                "Stage 9 uses one unperturbed synthetic mean per injector; "
                "these are single-run truth-inclusion fractions, not "
                "frequentist coverage estimates."
            ),
            "scope": (
                "Scalar bias reporting includes retrieved molecular and cloud "
                "parameters only. Temperature-profile parameters are excluded; "
                "thermal recovery is assessed from the saved TP envelopes."
            ),
        },
        "record_count": len(records),
        "records_csv": str(csv_path),
        "aggregates": {
            "by_scenario_and_noise_tier": _aggregate_bias_coverage(
                records,
                ("scenario", "noise_ppm"),
            ),
            "by_directed_pair": _aggregate_bias_coverage(
                records,
                ("scenario", "noise_ppm", "injector", "retriever"),
            ),
            "by_parameter_family": _aggregate_bias_coverage(
                records,
                ("scenario", "noise_ppm", "parameter_family"),
            ),
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def _bias_coverage_matrix(
    records: Sequence[dict[str, Any]],
    runs: Sequence[dict[str, Any]],
    scenario: str,
) -> plt.Figure:
    first = next(run for run in runs if run["scenario"] == scenario)
    configured = first.get("parameters")
    if not isinstance(configured, list):
        raise RuntimeError(f"{first['run_id']} has no parameter metadata")
    parameters = [
        item
        for item in configured
        if _reference_value(item) is not None
        and _parameter_family(str(item["name"])) != "temperature_profile"
    ]
    parameter_names = [str(item["name"]) for item in parameters]
    directions = [
        (injector, retriever)
        for injector in FRAMEWORKS
        for retriever in FRAMEWORKS
        if injector != retriever
    ]
    lookup = {
        (
            int(record["noise_ppm"]),
            str(record["injector"]),
            str(record["retriever"]),
            str(record["parameter"]),
        ): record
        for record in records
        if record["scenario"] == scenario
    }
    categories = tuple(BIAS_CLASS_COLORS)
    cmap = ListedColormap([BIAS_CLASS_COLORS[name] for name in categories])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(12.0, 0.82 * len(parameters)), 9.0),
        sharex=True,
        constrained_layout=True,
    )
    for panel, (axis, tier) in enumerate(zip(axes, NOISE_TIERS_PPM, strict=True)):
        matrix = np.empty((len(directions), len(parameters)), dtype=float)
        pulls = np.empty_like(matrix)
        selected: list[dict[str, Any]] = []
        for row, (injector, retriever) in enumerate(directions):
            for column, name in enumerate(parameter_names):
                record = lookup[(tier, injector, retriever, name)]
                selected.append(record)
                matrix[row, column] = categories.index(
                    record["sigma_classification"]
                )
                pulls[row, column] = record["normalized_median_bias_sigma"]
        axis.imshow(
            matrix,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(categories) - 0.5,
            aspect="auto",
            interpolation="none",
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{pulls[row, column]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if matrix[row, column] == 2 else "#202020",
                )
        within_68 = np.mean(
            [record["truth_within_central_68_percent"] for record in selected]
        )
        within_95 = np.mean(
            [record["truth_within_central_95_percent"] for record in selected]
        )
        axis.set_title(
            f"{tier} ppm — truth inclusion: "
            f"68%={within_68:.0%}, 95%={within_95:.0%}",
            fontsize=9,
        )
        axis.set_yticks(range(len(directions)))
        axis.set_yticklabels(
            [
                f"{SHORT_NAMES[injector]} → {SHORT_NAMES[retriever]}"
                for injector, retriever in directions
            ],
            fontsize=7,
        )
        axis.set_xticks(range(len(parameters)))
        axis.tick_params(length=0)
        _panel_label(axis, panel)
    axes[-1].set_xticklabels(
        [_parameter_label(item) for item in parameters],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    handles = [
        Patch(facecolor=BIAS_CLASS_COLORS[name], label=BIAS_CLASS_LABELS[name])
        for name in categories
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3,
        fontsize=7.5,
        title=(
            f"{SCENARIO_LABELS[scenario]} — parameter bias and truth inclusion; "
            "cell text is signed median bias / posterior σ"
        ),
        title_fontsize=10.5,
    )
    return fig


def _posterior_atlas(
    runs: Sequence[dict[str, Any]],
    scenario: str,
    parameters: Sequence[dict[str, Any]],
    *,
    family: str,
    cache: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> plt.Figure:
    columns = len(parameters)
    fig, axes = plt.subplots(
        3,
        columns,
        figsize=(max(10.5, 3.05 * columns), 7.6),
        squeeze=False,
        constrained_layout=True,
    )
    panel = 0
    for row, tier in enumerate(NOISE_TIERS_PPM):
        tier_runs = [
            run
            for run in runs
            if run["scenario"] == scenario and int(run["noise_ppm"]) == tier
        ]
        for column, parameter in enumerate(parameters):
            axis = axes[row, column]
            name = str(parameter["name"])
            edges = np.linspace(
                float(parameter["lower"]),
                float(parameter["upper"]),
                80,
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            for run in sorted(
                tier_runs,
                key=lambda item: (
                    FRAMEWORKS.index(item["retriever"]),
                    FRAMEWORKS.index(item["injector"]),
                ),
            ):
                names = tuple(run["parameter_names"])
                if name not in names:
                    continue
                samples, weights = cache[str(run["run_id"])]
                values = samples[:, names.index(name)]
                density, _ = np.histogram(
                    values,
                    bins=edges,
                    weights=weights,
                    density=True,
                )
                axis.plot(
                    centers,
                    density,
                    color=MODEL_COLORS[str(run["retriever"])],
                    ls=INJECTOR_LINESTYLES[str(run["injector"])],
                    lw=1.15,
                    alpha=0.9,
                )
                median = _weighted_quantile(values, weights, 0.50)
                axis.plot(
                    median,
                    0.985 * max(float(np.max(density)), 1.0e-12),
                    marker="|",
                    ms=5,
                    color=MODEL_COLORS[str(run["retriever"])],
                    alpha=0.85,
                )
            reference = _reference_value(parameter)
            if reference is not None:
                axis.axvline(reference, color="#202020", lw=0.8, ls="--")
            if row == 0:
                axis.set_title(_parameter_label(parameter), fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{tier} ppm\nPosterior density")
            if row < 2:
                axis.set_xticklabels([])
            axis.set_yticks([])
            axis.set_xlim(float(parameter["lower"]), float(parameter["upper"]))
            axis.grid(axis="x", alpha=0.15)
            _panel_label(axis, panel)
            panel += 1
    handles = [
        Line2D([0], [0], color=MODEL_COLORS[name], lw=1.5, label=f"Retrieved with {DISPLAY_NAMES[name]}")
        for name in FRAMEWORKS
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color="#505050",
            lw=1.3,
            ls=INJECTOR_LINESTYLES[name],
            label=f"Injected by {DISPLAY_NAMES[name]}",
        )
        for name in FRAMEWORKS
    )
    handles.append(
        Line2D([0], [0], color="#202020", lw=0.9, ls="--", label="Injected value")
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        fontsize=7.5,
        title=f"{SCENARIO_LABELS[scenario]} — {family} posterior comparison",
        title_fontsize=11,
    )
    return fig


def _save_figure(
    figure: plt.Figure,
    output: Path,
    stem: str,
    multipage: PdfPages,
    dpi: int,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{stem}.pdf"
    png = output / f"{stem}.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=dpi, bbox_inches="tight")
    multipage.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    return [pdf, png]


def generate_paper_atlas(
    project_root: Path,
    *,
    output: Path | None = None,
    scenario_filter: str | None = None,
    dpi: int = 220,
) -> tuple[Path, Path, list[Path]]:
    if dpi < 100:
        raise ValueError("paper-atlas PNG resolution must be at least 100 dpi")
    project = project_root.expanduser().resolve()
    destination = (
        project / "diagnostics" / "paper_atlas"
        if output is None
        else output.expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    runs = _load_completed_runs(project, scenario_filter)
    common = load_common_contract(project / "contracts" / "common_contract.json")
    scenarios = (
        [scenario_filter]
        if scenario_filter is not None
        else [item.name for item in SCENARIOS]
    )
    all_pdf = destination / (
        f"{scenario_filter}_paper_atlas.pdf"
        if scenario_filter is not None
        else "stage9_paper_atlas.pdf"
    )
    created: list[Path] = []
    cache = _posterior_cache(runs)
    bias_records = _bias_coverage_records(runs, cache)
    bias_csv, bias_json = _write_bias_coverage_reports(
        destination,
        bias_records,
    )
    with PdfPages(all_pdf) as multipage:
        for scenario in scenarios:
            scenario_output = destination / scenario
            figures = (
                (
                    _spectral_residual_atlas(runs, scenario),
                    f"{scenario}_spectral_residual_atlas",
                ),
                (
                    _tp_atlas(runs, common, scenario),
                    f"{scenario}_tp_atlas",
                ),
            )
            for figure, stem in figures:
                created.extend(
                    _save_figure(figure, scenario_output, stem, multipage, dpi)
                )
            chemistry, clouds = _parameter_groups(runs, scenario)
            created.extend(
                _save_figure(
                    _posterior_atlas(
                        runs,
                        scenario,
                        chemistry,
                        family="chemistry",
                        cache=cache,
                    ),
                    scenario_output,
                    f"{scenario}_chemistry_posterior_atlas",
                    multipage,
                    dpi,
                )
            )
            if clouds:
                created.extend(
                    _save_figure(
                        _posterior_atlas(
                            runs,
                            scenario,
                            clouds,
                            family="cloud",
                            cache=cache,
                        ),
                        scenario_output,
                        f"{scenario}_cloud_posterior_atlas",
                        multipage,
                        dpi,
                    )
                )
            created.extend(
                _save_figure(
                    _bias_coverage_matrix(
                        bias_records,
                        runs,
                        scenario,
                    ),
                    scenario_output,
                    f"{scenario}_parameter_bias_coverage_matrix",
                    multipage,
                    dpi,
                )
            )
    manifest = destination / "paper_atlas_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_root": str(project),
                "scenario_filter": scenario_filter,
                "completed_retrievals_used": len(runs),
                "multipage_pdf": str(all_pdf),
                "figures": [str(path) for path in created],
                "bias_coverage_reports": {
                    "per_parameter_csv": str(bias_csv),
                    "aggregate_json": str(bias_json),
                },
                "encoding": {
                    "retrieval_framework_color": MODEL_COLORS,
                    "injection_framework_linestyle": INJECTOR_LINESTYLES,
                    "spectral_and_tp_intervals": {
                        "central_68_percent": [0.16, 0.84],
                        "central_95_percent": [0.025, 0.975],
                    },
                    "bias_classification_colors": BIAS_CLASS_COLORS,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return all_pdf, manifest, created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scenario",
        choices=tuple(item.name for item in SCENARIOS),
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    pdf, manifest, created = generate_paper_atlas(
        args.project_root,
        output=args.output,
        scenario_filter=args.scenario,
        dpi=args.dpi,
    )
    print(f"{pdf} ({len(created) // 2} figure pages)")
    print(manifest)


if __name__ == "__main__":
    main()
