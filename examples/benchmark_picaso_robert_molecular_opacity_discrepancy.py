#!/usr/bin/env python3
"""Close the PICASO--ROBERT molecular-opacity attribution diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    OpacitySamplingProvider,
    PressureGrid,
    assemble_gas_optical_depth,
)
from robert_exoplanets.io.configured_tasks import build_problem, load_observations
from robert_exoplanets.io.task_config import load_task_config

try:
    from examples.benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
        EXOMOL,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _bin_mean,
        _inverse_square_hydrostatic_profiles,
    )
    from examples.benchmark_shared_deck_haze_external_parity import _bin
    from examples.picaso_jwst_transmission_retrieval import (
        DEFAULT_PICASO_PYTHON,
        TRUTH,
        build_picaso_contract,
        jwst_grid,
    )
except ModuleNotFoundError:
    from benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
        EXOMOL,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _bin_mean,
        _inverse_square_hydrostatic_profiles,
    )
    from benchmark_shared_deck_haze_external_parity import _bin
    from picaso_jwst_transmission_retrieval import (
        DEFAULT_PICASO_PYTHON,
        TRUTH,
        build_picaso_contract,
        jwst_grid,
    )


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_picaso_jwst_transmission_injection.py")
DEFAULT_CONFIG = ROOT / "configurations" / "picaso_jwst_transmission_retrieval_multinest.yaml"
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "picaso_jwst_molecular_opacity_discrepancy"
BARSTOW_2020_DOI = "10.1093/mnras/staa548"


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--picaso-python", type=Path, default=DEFAULT_PICASO_PYTHON)
    parser.add_argument("--picaso-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--picaso-database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    return run(
        args.config,
        args.output_dir,
        args.picaso_python,
        args.picaso_reference,
        args.picaso_database,
    )


def run(
    config_path: Path,
    output_dir: Path,
    picaso_python: Path,
    picaso_reference: Path,
    picaso_database: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = build_picaso_contract()
    contract_path = output_dir / "molecular_opacity_contract.npz"
    full_path = output_dir / "picaso_full_native.npz"
    stride_path = output_dir / "picaso_stride_5.npz"
    np.savez_compressed(contract_path, **contract)
    _run_picaso(
        picaso_python,
        picaso_reference,
        picaso_database,
        contract_path,
        full_path,
        stride=1,
        include_components=True,
    )
    _run_picaso(
        picaso_python,
        picaso_reference,
        picaso_database,
        contract_path,
        stride_path,
        stride=5,
        include_components=False,
    )
    with np.load(full_path, allow_pickle=False) as archive:
        full = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(stride_path, allow_pickle=False) as archive:
        stride = {name: np.array(archive[name], copy=True) for name in archive.files}

    robert = _evaluate_robert_molecular_tau(contract)
    _, observation_edges, uncertainty_ppm = jwst_grid()
    diagnostic_edges = np.concatenate(
        (np.array([1.0]), observation_edges[observation_edges > 1.0])
    )
    wavelength = np.sqrt(diagnostic_edges[:-1] * diagnostic_edges[1:])
    compact: dict[str, np.ndarray] = {"wavelength_micron": wavelength}
    species_metrics = {}
    for index, species in enumerate(SPECIES):
        robert_column = np.sum(robert["molecular_tau_by_species"][index], axis=0)
        picaso_column = np.sum(full["molecular_tau_by_species"][index], axis=0)
        compact[f"robert_{species}_column_tau"] = _bin_mean(
            robert["wavelength_micron"], robert_column, diagnostic_edges
        )
        compact[f"picaso_{species}_column_tau"] = _bin_mean(
            full["wavelength_micron"], picaso_column, diagnostic_edges
        )
        species_metrics[species] = _species_metrics(
            wavelength,
            compact[f"robert_{species}_column_tau"],
            compact[f"picaso_{species}_column_tau"],
        )

    picaso_full_binned = _bin(
        full["wavelength_micron"], full["transit_depth"], observation_edges
    )
    picaso_stride_binned = _bin(
        stride["wavelength_micron"], stride["transit_depth"], observation_edges
    )
    config = load_task_config(config_path)
    observations = load_observations(config)
    problem = build_problem(config, observations)
    robert_truth = np.asarray(
        problem.model_spectra(TRUTH)[config.observations.datasets[0]].values
    )
    full_difference_ppm = (robert_truth - picaso_full_binned) * 1.0e6
    stride_difference_ppm = (robert_truth - picaso_stride_binned) * 1.0e6
    sampling_effect_ppm = (picaso_stride_binned - picaso_full_binned) * 1.0e6
    report = {
        "schema_version": 1,
        "benchmark": "final_PICASO_ROBERT_molecular_opacity_attribution",
        "scope": (
            "One closing diagnostic on the exact PICASO-JWST retrieval atmosphere; "
            "no new production physics is proposed."
        ),
        "sampling": {
            "full_native_picaso_samples": int(full["wavelength_micron"].size),
            "stride_5_picaso_samples": int(stride["wavelength_micron"].size),
            "robert_native_exomolop_samples": int(robert["wavelength_micron"].size),
            "picaso_stride_5_minus_full_native": _ppm_metrics(sampling_effect_ppm),
        },
        "truth_spectrum_difference": {
            "robert_minus_picaso_full_native": _ppm_metrics(full_difference_ppm),
            "robert_minus_picaso_stride_5": _ppm_metrics(stride_difference_ppm),
            "uncertainty_weighted_chi_square_full_native": float(
                np.sum((full_difference_ppm / uncertainty_ppm) ** 2)
            ),
            "uncertainty_weighted_chi_square_stride_5": float(
                np.sum((stride_difference_ppm / uncertainty_ppm) ** 2)
            ),
        },
        "species_column_optical_depth": species_metrics,
        "literature_context": {
            "reference": "Barstow et al. 2020, MNRAS 493, 4884",
            "doi": BARSTOW_2020_DOI,
            "reported_realistic_forward_model_agreement": "a few tens of ppm",
            "cross_retrieval_error_envelopes_ppm": [30, 60, 100],
        },
        "decision": {
            "diagnostic_campaign": "close after this attribution",
            "production_change_required": False,
            "interpretation": (
                "Residuals at this level are expected from independent line databases, "
                "opacity tabulation, and sampling choices; they are not evidence of an "
                "unvalidated transmission solver."
            ),
        },
        "picaso_full_metadata": json.loads(str(full["metadata_json"])),
    }
    report_path = output_dir / "molecular_opacity_discrepancy.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "molecular_opacity_discrepancy_compact.npz",
        **compact,
        picaso_full_binned=picaso_full_binned,
        picaso_stride_5_binned=picaso_stride_binned,
        robert_truth_binned=robert_truth,
    )
    _plot(output_dir / "molecular_opacity_discrepancy.png", compact)
    print(json.dumps(report, indent=2))
    return report


def _evaluate_robert_molecular_tau(
    contract: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    pressure_layer = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_layer,
        unit="bar",
        name="final molecular opacity attribution",
    )
    composition = {
        name: np.asarray(contract["gas_vmr"][:, index], dtype=float)
        for index, name in enumerate(SPECIES)
    }
    composition["H2"] = np.full(pressure_layer.size, float(contract["h2_vmr"]))
    composition["He"] = np.asarray(contract["he_vmr"], dtype=float)
    mean_molecular_weight = sum(
        composition[name] * MOLECULAR_WEIGHTS[name] for name in composition
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.full(pressure_layer.size, 1100.0),
        temperature_edges=np.full(pressure_edges.size, 1100.0),
        composition=composition,
        mean_molecular_weight=mean_molecular_weight,
    )
    provider = OpacitySamplingProvider.from_exomol_paths(
        {name: EXOMOL / f"{name}.h5" for name in SPECIES},
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=False,
    )
    spectral_grid = provider.native_spectral_grid(
        sampling=1,
        wavelength_bounds_micron=(1.0, 12.0),
        name="ExoMolOP native molecular attribution",
    )
    prepared = provider.prepare(spectral_grid, pressure_grid, SPECIES)
    evaluated = provider.evaluate(atmosphere, prepared)
    hydrostatic = _inverse_square_hydrostatic_profiles(contract)
    gas = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=hydrostatic["column_gravity_m_s2"],
        retain_species_tau=True,
    )
    return {
        "wavelength_micron": spectral_grid.values,
        "molecular_tau_by_species": np.asarray(gas.species_tau[:, :, :, 0]),
    }


def _species_metrics(
    wavelength: np.ndarray,
    robert_tau: np.ndarray,
    picaso_tau: np.ndarray,
) -> dict[str, Any]:
    floor = max(float(np.max(robert_tau)), float(np.max(picaso_tau))) * 1.0e-2
    active = np.maximum(robert_tau, picaso_tau) >= floor
    delta = np.log10(np.maximum(robert_tau, 1.0e-300)) - np.log10(
        np.maximum(picaso_tau, 1.0e-300)
    )
    active_indices = np.flatnonzero(active)
    ranked = active_indices[np.argsort(np.abs(delta[active]))[-5:][::-1]]
    return {
        "active_bin_definition": "either code at least 1 percent of species maximum",
        "active_bins": int(np.sum(active)),
        "active_rms_log10_tau_difference_dex": float(
            np.sqrt(np.mean(delta[active] ** 2))
        ),
        "active_median_log10_tau_ratio_dex": float(np.median(delta[active])),
        "active_max_abs_log10_tau_difference_dex": float(
            np.max(np.abs(delta[active]))
        ),
        "largest_active_differences": [
            {
                "wavelength_micron": float(wavelength[index]),
                "robert_minus_picaso_log10_tau_dex": float(delta[index]),
            }
            for index in ranked
        ],
    }


def _ppm_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "rms_ppm": float(np.sqrt(np.mean(values**2))),
        "median_ppm": float(np.median(values)),
        "max_abs_ppm": float(np.max(np.abs(values))),
    }


def _plot(path: Path, compact: dict[str, np.ndarray]) -> None:
    wavelength = compact["wavelength_micron"]
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 11.0), sharex=True)
    figure.patch.set_facecolor("white")
    for row, species in enumerate(SPECIES):
        robert = compact[f"robert_{species}_column_tau"]
        picaso = compact[f"picaso_{species}_column_tau"]
        axes[row, 0].plot(wavelength, robert, color="#6f4bb8", label="ROBERT")
        axes[row, 0].plot(wavelength, picaso, color="#17202a", label="PICASO")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylabel(f"{species} column tau")
        delta = np.log10(np.maximum(robert, 1.0e-300)) - np.log10(
            np.maximum(picaso, 1.0e-300)
        )
        active_floor = max(float(np.max(robert)), float(np.max(picaso))) * 1.0e-2
        active_delta = np.where(np.maximum(robert, picaso) >= active_floor, delta, np.nan)
        axes[row, 1].plot(wavelength, active_delta, color="#2471a3")
        axes[row, 1].axhline(0.0, color="0.5", linewidth=0.8)
        axes[row, 1].set_ylabel("log10 ratio (active bins)")
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("Wavelength (micron)")
    axes[-1, 1].set_xlabel("Wavelength (micron)")
    figure.suptitle("Final molecular-opacity attribution: full native grids")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _run_picaso(
    python: Path,
    reference: Path,
    database: Path,
    contract: Path,
    output: Path,
    *,
    stride: int,
    include_components: bool,
) -> None:
    environment = dict(os.environ)
    environment["picaso_refdata"] = str(reference.resolve())
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "picaso-mpl")
    environment["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "picaso-numba")
    command = [
        str(python),
        str(RUNNER),
        str(contract),
        str(output),
        "--opacity-db",
        str(database),
        "--resample",
        str(stride),
    ]
    if include_components:
        command.append("--include-opacity-components")
    subprocess.run(command, check=True, cwd=ROOT, env=environment)


if __name__ == "__main__":
    main()
