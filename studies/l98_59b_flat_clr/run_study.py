#!/usr/bin/env python3
"""Generate, run, and analyze the L 98-59 b flat-spectrum CLR ensembles."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

from robert_exoplanets.io import load_bello_arufe2025_l9859b
from robert_exoplanets.postprocessing import weighted_quantile
from robert_exoplanets.retrieval import save_observation_npz
from robert_exoplanets.validation import (
    MOLECULAR_MASS_AMU,
    abundance_constraint,
    closed_composition,
    composition_mean_molecular_weight,
    generate_flat_spectrum_ensemble,
)


ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
OUTPUTS = STUDY / "outputs"
DATA = OUTPUTS / "data"
RESOLVED_CONFIGS = OUTPUTS / "configs"
RUNS = OUTPUTS / "runs"
LOGS = OUTPUTS / "logs"
ANALYSIS = OUTPUTS / "analysis"
SOURCE_DATA = ROOT / "data" / "observations" / "l98_59b_bello_arufe2025"
RUN_RETRIEVAL = ROOT / "run_retrieval.py"
OPACITY_DIRECTORY_ENV = "ROBERT_K_TABLE_DIRECTORY"
ENSEMBLES = {
    "a": {
        "template": STUDY / "configs" / "ensemble_a.yaml",
        "species": ("H2O", "CO2", "CO", "H2S", "SO2"),
        "free_species": ("H2O", "CO2", "CO", "SO2"),
        "target": "SO2",
        "seed_base": 19_000,
    },
    "b": {
        "template": STUDY / "configs" / "ensemble_b.yaml",
        "species": ("H2O", "CO2", "CO", "H2S"),
        "free_species": ("H2O", "CO2", "CO"),
        "target": "CO2",
        "seed_base": 20_000,
    },
}
THRESHOLDS = (0.001, 0.01, 0.1, 0.5)


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def generate() -> None:
    source = load_bello_arufe2025_l9859b(SOURCE_DATA)
    spectra, manifest = generate_flat_spectrum_ensemble(
        source, n_realizations=100, seed=20260719
    )
    DATA.mkdir(parents=True, exist_ok=True)
    files = []
    for realization_id, observation in enumerate(spectra):
        path = DATA / f"realization_{realization_id:03d}.npz"
        save_observation_npz(observation, path, overwrite=True)
        files.append(
            {
                "realization_id": realization_id,
                "path": str(path.relative_to(STUDY)),
                "sha256": _sha256(path),
            }
        )
    manifest.update(
        {
            "study_target": "L 98-59 b",
            "naming_correction": (
                "The request said L98-89b; the repository, paper, and source "
                "data identify L 98-59 b."
            ),
            "source_file": str(
                (SOURCE_DATA / "L9859b_combined_spectrum_eureka.txt").relative_to(ROOT)
            ),
            "source_file_sha256_repository_bytes": _sha256(
                SOURCE_DATA / "L9859b_combined_spectrum_eureka.txt"
            ),
            "multinest_sampler_seed_ranges": {
                "ensemble_a": [19000, 19099],
                "ensemble_b": [20000, 20099],
            },
            "files": files,
        }
    )
    _json_write(DATA / "manifest.json", manifest)
    _write_resolved_configs()
    print(json.dumps({key: manifest[key] for key in (
        "median_transit_depth_ppm", "median_uncertainty_ppm", "n_realizations", "root_seed", "grid"
    )}, indent=2))


def _opacity_directory(*, required: bool) -> Path | None:
    configured = os.environ.get(OPACITY_DIRECTORY_ENV)
    if configured is None:
        if required:
            raise SystemExit(
                f"set {OPACITY_DIRECTORY_ENV} to the ExoMol k-table directory"
            )
        return None
    path = Path(configured).expanduser().resolve()
    if required and not path.is_dir():
        raise SystemExit(f"opacity directory does not exist: {path}")
    return path


def _write_resolved_configs(*, require_opacity: bool = False) -> None:
    opacity_directory = _opacity_directory(required=require_opacity)
    for ensemble, definition in ENSEMBLES.items():
        template = yaml.safe_load(Path(definition["template"]).read_text())
        for realization_id in range(100):
            config = json.loads(json.dumps(template))
            tag = f"{realization_id:03d}"
            config["run"]["name"] = f"l98-59b-flat-clr-{ensemble}-{tag}"
            config["observations"]["path"] = str(
                (DATA / f"realization_{tag}.npz").resolve()
            )
            config["opacity"]["cache_directory"] = str(
                (OUTPUTS / "opacity_cache").resolve()
            )
            if opacity_directory is not None:
                config["opacity"]["path"] = str(opacity_directory)
            config["sampler"]["seed"] = int(definition["seed_base"]) + realization_id
            config["outputs"]["directory"] = str(
                (RUNS / ensemble / tag).resolve()
            )
            config["runtime"]["scratch_directory"] = str(
                (OUTPUTS / "scratch" / ensemble / tag).resolve()
            )
            target = RESOLVED_CONFIGS / ensemble / f"{tag}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _require_generated(*, require_opacity: bool = False) -> None:
    if not (DATA / "manifest.json").is_file():
        raise SystemExit("generate the study inputs first")
    _write_resolved_configs(require_opacity=require_opacity)


def _environment() -> dict[str, str]:
    (OUTPUTS / "matplotlib").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "numba_cache").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["MPLCONFIGDIR"] = str(OUTPUTS / "matplotlib")
    environment["NUMBA_CACHE_DIR"] = str(OUTPUTS / "numba_cache")
    return environment


def prepare_opacity() -> None:
    _require_generated(require_opacity=True)
    command = [
        sys.executable,
        str(RUN_RETRIEVAL),
        "--config",
        str(RESOLVED_CONFIGS / "a" / "000.yaml"),
        "--prepare-opacity",
    ]
    subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def smoke() -> None:
    _require_generated(require_opacity=True)
    records = {}
    for ensemble in ENSEMBLES:
        command = [
            sys.executable,
            str(RUN_RETRIEVAL),
            "--config",
            str(RESOLVED_CONFIGS / ensemble / "000.yaml"),
            "--smoke-only",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(),
            check=True,
            capture_output=True,
            text=True,
        )
        records[ensemble] = {
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        print(completed.stdout, end="")
    _json_write(OUTPUTS / "smoke.json", records)


def run(ensemble: str, start: int, stop: int) -> None:
    _require_generated(require_opacity=True)
    selected = tuple(ENSEMBLES) if ensemble == "both" else (ensemble,)
    mpirun = Path(sys.executable).with_name("mpirun")
    if not mpirun.is_file():
        raise SystemExit(f"MultiNest run requires MPI launcher: {mpirun}")
    LOGS.mkdir(parents=True, exist_ok=True)
    timing_path = OUTPUTS / "run_timings.jsonl"
    for realization_id in range(start, stop):
        for name in selected:
            tag = f"{realization_id:03d}"
            result = RUNS / name / tag / "multinest" / "result.json"
            if result.is_file():
                print(f"skip complete {name}/{tag}", flush=True)
                continue
            command = [
                str(mpirun),
                "-n",
                "3",
                sys.executable,
                str(RUN_RETRIEVAL),
                "--config",
                str(RESOLVED_CONFIGS / name / f"{tag}.yaml"),
            ]
            log_path = LOGS / f"{name}_{tag}.log"
            print(f"start {name}/{tag}", flush=True)
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=_environment(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.perf_counter() - started
            record = {
                "ensemble": name,
                "realization_id": realization_id,
                "elapsed_seconds": elapsed,
                "returncode": completed.returncode,
                "log": str(log_path.relative_to(STUDY)),
            }
            with timing_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"finish {name}/{tag} status={completed.returncode} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if completed.returncode != 0:
                raise SystemExit(f"retrieval failed; inspect {log_path}")


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = float(np.sum(weights * x))
    mean_y = float(np.sum(weights * y))
    covariance = float(np.sum(weights * (x - mean_x) * (y - mean_y)))
    variance_x = float(np.sum(weights * (x - mean_x) ** 2))
    variance_y = float(np.sum(weights * (y - mean_y) ** 2))
    return covariance / np.sqrt(variance_x * variance_y)


def _summarize_run(ensemble: str, realization_id: int) -> dict[str, object] | None:
    directory = RUNS / ensemble / f"{realization_id:03d}" / "multinest"
    result_path = directory / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    with np.load(directory / result["arrays"], allow_pickle=False) as arrays:
        samples = np.asarray(arrays["samples"], dtype=float)
        weights = np.asarray(arrays["weights"], dtype=float)
    weights /= np.sum(weights)
    columns = {
        name: samples[:, index]
        for index, name in enumerate(result["parameter_names"])
    }
    definition = ENSEMBLES[ensemble]
    vmr = closed_composition(
        {
            species: columns[f"log_{species}"]
            for species in definition["free_species"]
        },
        closure_species="H2S",
    )
    mmw = composition_mean_molecular_weight(vmr)
    species_summary = {}
    medians = {}
    for species in definition["species"]:
        quantiles = weighted_quantile(
            vmr[species], weights, (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)
        )
        medians[species] = float(quantiles[3])
        species_summary[species] = {
            "vmr_quantiles_2.5_5_16_50_84_95_97.5": quantiles.tolist(),
            "log10_vmr_quantiles_2.5_16_50_84_97.5": weighted_quantile(
                np.log10(vmr[species]), weights, (0.025, 0.16, 0.5, 0.84, 0.975)
            ).tolist(),
            "near_lower_prior_boundary": bool(
                weighted_quantile(np.log10(vmr[species]), weights, (0.025,))[0]
                <= -11.5
            ),
            "near_upper_closure_boundary": bool(quantiles[-1] >= 0.95),
            "constraints": {
                str(threshold): abundance_constraint(
                    vmr[species], weights, threshold=threshold, credibility=0.95
                )
                for threshold in THRESHOLDS
            },
        }
    rank = sorted(medians, key=medians.get, reverse=True)
    mmw_quantiles = weighted_quantile(mmw, weights, (0.025, 0.16, 0.5, 0.84, 0.975))
    temperature = columns["temperature"]
    radius = columns["radius_scale"]
    with np.load(DATA / f"realization_{realization_id:03d}.npz", allow_pickle=False) as observation:
        uncertainty = np.asarray(observation["err"], dtype=float)
    gaussian_normalization = float(np.sum(np.log(2.0 * np.pi * uncertainty**2)))
    chi_squared = float(-2.0 * result["best_fit_log_likelihood"] - gaussian_normalization)
    degrees_of_freedom = int(uncertainty.size - len(result["parameter_names"]))
    return {
        "ensemble": ensemble,
        "realization_id": realization_id,
        "target_species": definition["target"],
        "log_evidence": result["log_evidence"],
        "log_evidence_error": result["log_evidence_error"],
        "best_fit_log_likelihood": result["best_fit_log_likelihood"],
        "best_fit_chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "best_fit_reduced_chi_squared": chi_squared / degrees_of_freedom,
        "inference_elapsed_seconds": float(result["metadata"]["inference_elapsed_seconds"]),
        "posterior_sample_count": int(samples.shape[0]),
        "abundance_rank_by_posterior_median": rank,
        "species": species_summary,
        "mmw_quantiles_2.5_16_50_84_97.5_amu": mmw_quantiles.tolist(),
        "temperature_quantiles_2.5_16_50_84_97.5_K": weighted_quantile(
            temperature, weights, (0.025, 0.16, 0.5, 0.84, 0.975)
        ).tolist(),
        "radius_scale_quantiles_2.5_16_50_84_97.5": weighted_quantile(
            radius, weights, (0.025, 0.16, 0.5, 0.84, 0.975)
        ).tolist(),
        "weighted_correlations": {
            "mmw_temperature": _weighted_correlation(mmw, temperature, weights),
            "mmw_radius_scale": _weighted_correlation(mmw, radius, weights),
            "temperature_radius_scale": _weighted_correlation(temperature, radius, weights),
        },
    }


def analyze() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = []
    for realization_id in range(100):
        for ensemble in ENSEMBLES:
            summary = _summarize_run(ensemble, realization_id)
            if summary is not None:
                rows.append(summary)
    if not rows:
        raise SystemExit("no completed retrieval results found")
    _json_write(ANALYSIS / "per_run_summary.json", rows)
    aggregate: dict[str, object] = {
        "preregistered_primary_definition": "one-sided 95% VMR lower bound > 0.01",
        "threshold_sensitivity": list(THRESHOLDS),
        "ensembles": {},
        "clr_prior_reference": {
            ensemble: _clr_prior_reference(ensemble) for ensemble in ENSEMBLES
        },
    }
    flat_rows = []
    for ensemble, definition in ENSEMBLES.items():
        selected = [row for row in rows if row["ensemble"] == ensemble]
        target = str(definition["target"])
        frequencies = {}
        for threshold in THRESHOLDS:
            constrained = sum(
                bool(row["species"][target]["constraints"][str(threshold)]["constrained"])
                for row in selected
            )
            frequencies[str(threshold)] = {
                "count": constrained,
                "denominator": len(selected),
                "fraction": constrained / len(selected),
            }
        top_rank = sum(
            row["abundance_rank_by_posterior_median"][0] == target for row in selected
        )
        rank_counts = {
            species: sum(
                row["abundance_rank_by_posterior_median"][0] == species
                for row in selected
            )
            for species in definition["species"]
        }
        target_medians = [
            row["species"][target]["vmr_quantiles_2.5_5_16_50_84_95_97.5"][3]
            for row in selected
        ]
        target_lower_bounds = [
            row["species"][target]["constraints"]["0.01"]["lower_bound_vmr"]
            for row in selected
        ]
        target_log_widths = [
            row["species"][target]["log10_vmr_quantiles_2.5_16_50_84_97.5"][4]
            - row["species"][target]["log10_vmr_quantiles_2.5_16_50_84_97.5"][0]
            for row in selected
        ]
        aggregate["ensembles"][ensemble] = {
            "completed": len(selected),
            "target_species": target,
            "constraint_frequencies": frequencies,
            "target_top_abundance_rank": {
                "count": top_rank,
                "fraction": top_rank / len(selected),
            },
            "top_abundance_rank_counts": rank_counts,
            "target_median_vmr_across_runs_16_50_84": np.quantile(
                target_medians, (0.16, 0.5, 0.84)
            ).tolist(),
            "target_95pct_lower_bound_across_runs_16_50_84": np.quantile(
                target_lower_bounds, (0.16, 0.5, 0.84)
            ).tolist(),
            "target_near_lower_prior_boundary_count": sum(
                row["species"][target]["near_lower_prior_boundary"]
                for row in selected
            ),
            "target_near_upper_closure_boundary_count": sum(
                row["species"][target]["near_upper_closure_boundary"]
                for row in selected
            ),
            "alternative_constraint_sensitivity": {
                "central_95pct_log_width_below_2dex_count": sum(
                    width < 2.0 for width in target_log_widths
                ),
                "central_95pct_log_width_below_4dex_count": sum(
                    width < 4.0 for width in target_log_widths
                ),
                "posterior_median_above_1pct_count": sum(
                    value > 0.01 for value in target_medians
                ),
                "log_width_across_runs_16_50_84_dex": np.quantile(
                    target_log_widths, (0.16, 0.5, 0.84)
                ).tolist(),
            },
            "median_inference_elapsed_seconds": float(
                np.median([row["inference_elapsed_seconds"] for row in selected])
            ),
            "median_log_evidence": float(np.median([row["log_evidence"] for row in selected])),
            "median_log_evidence_error": float(
                np.median([row["log_evidence_error"] for row in selected])
            ),
            "median_best_fit_reduced_chi_squared": float(
                np.median([row["best_fit_reduced_chi_squared"] for row in selected])
            ),
            "median_mmw_amu": float(
                np.median([row["mmw_quantiles_2.5_16_50_84_97.5_amu"][2] for row in selected])
            ),
            "median_correlations": {
                key: float(np.median([row["weighted_correlations"][key] for row in selected]))
                for key in ("mmw_temperature", "mmw_radius_scale", "temperature_radius_scale")
            },
        }
        for row in selected:
            flat_rows.append(
                {
                    "ensemble": ensemble,
                    "realization_id": row["realization_id"],
                    "target": target,
                    "target_constrained_primary": row["species"][target]["constraints"]["0.01"]["constrained"],
                    "target_median_vmr": row["species"][target]["vmr_quantiles_2.5_5_16_50_84_95_97.5"][3],
                    "target_rank": row["abundance_rank_by_posterior_median"].index(target) + 1,
                    "mmw_median_amu": row["mmw_quantiles_2.5_16_50_84_97.5_amu"][2],
                    "temperature_median_K": row["temperature_quantiles_2.5_16_50_84_97.5_K"][2],
                    "radius_scale_median": row["radius_scale_quantiles_2.5_16_50_84_97.5"][2],
                    "log_evidence": row["log_evidence"],
                    "best_fit_reduced_chi_squared": row["best_fit_reduced_chi_squared"],
                    "inference_elapsed_seconds": row["inference_elapsed_seconds"],
                }
            )
    paired_ids = sorted(
        set(row["realization_id"] for row in rows if row["ensemble"] == "a")
        & set(row["realization_id"] for row in rows if row["ensemble"] == "b")
    )
    by_key = {(row["ensemble"], row["realization_id"]): row for row in rows}
    aggregate["paired"] = {
        "count": len(paired_ids),
        "median_delta_log_evidence_a_minus_b": (
            float(np.median([
                by_key[("a", item)]["log_evidence"] - by_key[("b", item)]["log_evidence"]
                for item in paired_ids
            ])) if paired_ids else None
        ),
        "median_delta_mmw_a_minus_b_amu": (
            float(np.median([
                by_key[("a", item)]["mmw_quantiles_2.5_16_50_84_97.5_amu"][2]
                - by_key[("b", item)]["mmw_quantiles_2.5_16_50_84_97.5_amu"][2]
                for item in paired_ids
            ])) if paired_ids else None
        ),
        "fraction_delta_log_evidence_favors_a": (
            float(np.mean([
                by_key[("a", item)]["log_evidence"]
                > by_key[("b", item)]["log_evidence"]
                for item in paired_ids
            ])) if paired_ids else None
        ),
    }
    _json_write(ANALYSIS / "aggregate_summary.json", aggregate)
    with (ANALYSIS / "compact_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    _plot_summary(aggregate, rows, paired_ids, by_key)
    _write_science_report(aggregate)
    print(json.dumps(aggregate, indent=2))


def _clr_prior_reference(ensemble: str, n_samples: int = 50_000) -> dict[str, object]:
    """Vectorized Monte Carlo reference for the exact configured CLR prior."""

    definition = ENSEMBLES[ensemble]
    free_species = tuple(definition["free_species"])
    species = ("H2S",) + free_species
    n_free = len(free_species)
    n_total = n_free + 1
    lower = -12.0
    rng = np.random.default_rng(20260720 + (0 if ensemble == "a" else 1))
    accepted = []
    while sum(item.shape[0] for item in accepted) < n_samples:
        values = rng.random((max(10_000, n_samples // 2), n_free))
        prior_lower = (n_free / n_total) * (
            lower * np.log(10.0) + np.log(float(n_free))
        )
        prior_upper = -(n_free / n_total) * lower * np.log(10.0)
        clr_free = prior_lower + values * (prior_upper - prior_lower)
        clr = np.column_stack((-np.sum(clr_free, axis=1), clr_free))
        valid = (np.abs(np.sum(clr_free, axis=1)) <= prior_upper) & (
            np.ptp(clr, axis=1) <= -lower * np.log(10.0)
        )
        shifted = clr[valid] - np.max(clr[valid], axis=1, keepdims=True)
        mixing = np.exp(shifted)
        mixing /= np.sum(mixing, axis=1, keepdims=True)
        mixing = mixing[np.all(mixing >= 10.0**lower, axis=1)]
        accepted.append(mixing)
    samples = np.concatenate(accepted, axis=0)[:n_samples]
    index = {name: position for position, name in enumerate(species)}
    target = str(definition["target"])
    target_values = samples[:, index[target]]
    masses = np.array([MOLECULAR_MASS_AMU[name] for name in species])
    mmw = samples @ masses
    return {
        "samples": n_samples,
        "seed": 20260720 + (0 if ensemble == "a" else 1),
        "species_order_closure_first": list(species),
        "target_species": target,
        "target_top_rank_probability": float(
            np.mean(np.argmax(samples, axis=1) == index[target])
        ),
        "target_vmr_quantiles_5_16_50_84_95": np.quantile(
            target_values, (0.05, 0.16, 0.5, 0.84, 0.95)
        ).tolist(),
        "target_probability_above_threshold": {
            str(threshold): float(np.mean(target_values > threshold))
            for threshold in THRESHOLDS
        },
        "mmw_quantiles_5_16_50_84_95_amu": np.quantile(
            mmw, (0.05, 0.16, 0.5, 0.84, 0.95)
        ).tolist(),
    }


def _plot_summary(aggregate, rows, paired_ids, by_key) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    x = np.arange(len(THRESHOLDS))
    width = 0.36
    for offset, ensemble in ((-width / 2, "a"), (width / 2, "b")):
        frequencies = aggregate["ensembles"][ensemble]["constraint_frequencies"]
        axes[0, 0].bar(
            x + offset,
            [frequencies[str(value)]["fraction"] for value in THRESHOLDS],
            width,
            label=f"{ensemble.upper()}: {ENSEMBLES[ensemble]['target']}",
        )
    axes[0, 0].set_xticks(x, [f"{100 * value:g}%" for value in THRESHOLDS])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Fraction with 95% lower bound above threshold")
    axes[0, 0].legend(frameon=False)
    if paired_ids:
        mmw_a = [by_key[("a", item)]["mmw_quantiles_2.5_16_50_84_97.5_amu"][2] for item in paired_ids]
        mmw_b = [by_key[("b", item)]["mmw_quantiles_2.5_16_50_84_97.5_amu"][2] for item in paired_ids]
        axes[0, 1].scatter(mmw_b, mmw_a, s=20, alpha=0.7)
        limits = [min(mmw_a + mmw_b), max(mmw_a + mmw_b)]
        axes[0, 1].plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axes[0, 1].set_xlabel("B median MMW (amu)")
        axes[0, 1].set_ylabel("A median MMW (amu)")
        delta_z = [by_key[("a", item)]["log_evidence"] - by_key[("b", item)]["log_evidence"] for item in paired_ids]
        axes[1, 1].hist(delta_z, bins=min(20, max(5, len(delta_z))), color="#6a4c93", alpha=0.8)
        axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[1, 1].set_xlabel("Paired ln Z(A) - ln Z(B)")
    for ensemble, color in (("a", "#d95f02"), ("b", "#1b9e77")):
        selected = [row for row in rows if row["ensemble"] == ensemble]
        target = ENSEMBLES[ensemble]["target"]
        axes[1, 0].hist(
            [row["species"][target]["vmr_quantiles_2.5_5_16_50_84_95_97.5"][3] for row in selected],
            bins=15,
            alpha=0.55,
            label=f"{ensemble.upper()}: {target}",
            color=color,
        )
    axes[1, 0].set_xlabel("Target-gas posterior median VMR")
    axes[1, 0].legend(frameon=False)
    figure.suptitle("L 98-59 b flat-spectrum CLR ensemble: exploratory 50-live-point look")
    figure.tight_layout()
    figure.savefig(ANALYSIS / "science_look_summary.png", dpi=180)
    plt.close(figure)

    interval_figure, interval_axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    for axis, ensemble, color in zip(
        interval_axes, ("a", "b"), ("#d95f02", "#1b9e77"), strict=True
    ):
        selected = sorted(
            (row for row in rows if row["ensemble"] == ensemble),
            key=lambda row: row["realization_id"],
        )
        target = ENSEMBLES[ensemble]["target"]
        ids = np.array([row["realization_id"] for row in selected])
        quantiles = np.array(
            [
                row["species"][target][
                    "vmr_quantiles_2.5_5_16_50_84_95_97.5"
                ]
                for row in selected
            ]
        )
        axis.vlines(ids, quantiles[:, 1], quantiles[:, 5], color=color, alpha=0.35)
        axis.scatter(ids, quantiles[:, 3], color=color, s=12)
        axis.axhline(0.01, color="black", linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set_ylim(5.0e-13, 1.1)
        axis.set_ylabel(f"{target} VMR")
        axis.set_title(f"Ensemble {ensemble.upper()}: 5--95% intervals and medians")
    interval_axes[-1].set_xlabel("Noise realization ID")
    interval_figure.tight_layout()
    interval_figure.savefig(ANALYSIS / "target_abundance_intervals.png", dpi=180)
    plt.close(interval_figure)

    rank_figure, rank_axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for axis, ensemble, color in zip(
        rank_axes, ("a", "b"), ("#d95f02", "#1b9e77"), strict=True
    ):
        counts = aggregate["ensembles"][ensemble]["top_abundance_rank_counts"]
        names = list(counts)
        denominator = aggregate["ensembles"][ensemble]["completed"]
        axis.bar(
            names,
            [counts[name] / denominator for name in names],
            color=["#e76f51" if name == "H2S" else color for name in names],
        )
        axis.set_ylim(0, 1)
        axis.set_title(f"Ensemble {ensemble.upper()}")
        axis.set_ylabel("Fraction ranked most abundant")
    rank_figure.suptitle("Posterior-median abundance leaders (H2S includes CLR closure)")
    rank_figure.tight_layout()
    rank_figure.savefig(ANALYSIS / "abundance_rank_frequencies.png", dpi=180)
    plt.close(rank_figure)


def _write_science_report(aggregate: dict[str, object]) -> None:
    a = aggregate["ensembles"]["a"]
    b = aggregate["ensembles"]["b"]
    paired = aggregate["paired"]
    a_primary = a["constraint_frequencies"]["0.01"]
    b_primary = b["constraint_frequencies"]["0.01"]
    text = f"""# L 98-59 b flat-spectrum CLR science look

This report contains {a['completed']} completed ensemble-A runs,
{b['completed']} completed ensemble-B runs, and {paired['count']} paired
realizations. The requested target is L 98-59 b; “L98-89b” was corrected as a
naming error using the repository, cited paper, and Zenodo source record.

## Preregistered abundance result

- SO2 in A: {a_primary['count']}/{a_primary['denominator']} runs
  ({a_primary['fraction']:.1%}) have a one-sided 95% VMR lower bound above 1%.
- CO2 in B: {b_primary['count']}/{b_primary['denominator']} runs
  ({b_primary['fraction']:.1%}) have a one-sided 95% VMR lower bound above 1%.

Threshold sensitivity and the exact CLR prior reference are recorded in
`aggregate_summary.json`. A fitted, opacity-bearing H2S category supplies the
omitted mathematical CLR coordinate in both ensembles; it is not a physical
background assumption or phantom gas.

## Ensemble summaries

- A median posterior MMW across runs: {a['median_mmw_amu']:.3f} amu; median
  reduced chi-square: {a['median_best_fit_reduced_chi_squared']:.3f}; median
  ln Z: {a['median_log_evidence']:.3f} +/- {a['median_log_evidence_error']:.3f}.
- B median posterior MMW across runs: {b['median_mmw_amu']:.3f} amu; median
  reduced chi-square: {b['median_best_fit_reduced_chi_squared']:.3f}; median
  ln Z: {b['median_log_evidence']:.3f} +/- {b['median_log_evidence_error']:.3f}.
- Paired median ln Z(A)-ln Z(B):
  {paired['median_delta_log_evidence_a_minus_b']}; paired median MMW(A)-MMW(B):
  {paired['median_delta_mmw_a_minus_b_amu']} amu.

These are exploratory 50-live-point MultiNest results. They diagnose how the
configured CLR prior and flat-spectrum likelihood interact; they are not
molecular detections and are not suitable for final evidence claims. Higher
live-point cluster runs are required for any science-grade posterior or model
comparison statement.
"""
    (ANALYSIS / "science_look.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate")
    subparsers.add_parser("prepare-opacity")
    subparsers.add_parser("smoke")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--ensemble", choices=("a", "b", "both"), default="both")
    run_parser.add_argument("--start", type=int, default=0)
    run_parser.add_argument("--stop", type=int, default=100)
    subparsers.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "generate":
        generate()
    elif args.command == "prepare-opacity":
        prepare_opacity()
    elif args.command == "smoke":
        smoke()
    elif args.command == "run":
        if not 0 <= args.start < args.stop <= 100:
            raise SystemExit("run bounds must satisfy 0 <= start < stop <= 100")
        run(args.ensemble, args.start, args.stop)
    else:
        analyze()


if __name__ == "__main__":
    main()
