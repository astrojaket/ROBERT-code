#!/usr/bin/env python3
"""Make and evaluate a reproducible ROBERT/Virga self-injection experiment.

The generated spectrum is a self-injection: ROBERT supplies both the forward
truth and the retrieval model. It is not an independent validation of the
cloud or atmospheric physics. A grid NPZ is used only for its wavelength
centres and bin edges; its flux and uncertainty arrays are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

from robert_exoplanets import weighted_quantile
from robert_exoplanets.instruments import Observation, infer_wavelength_bin_edges
from robert_exoplanets.io.configured_tasks import (
    build_problem,
    load_observations,
    prepare_opacity,
    run_forward_task,
)
from robert_exoplanets.io.task_config import (
    TaskConfig,
    configured_regions,
    initialize_task_directories,
    load_task_config,
)
from robert_exoplanets.retrieval import save_observation_npz


DEFAULT_WAVELENGTH_MIN_MICRON = 0.6039311351970722
DEFAULT_WAVELENGTH_MAX_MICRON = 11.95087089337295
DEFAULT_N_WAVELENGTH = 301
DEFAULT_SIGMA_PPM = 100.0
_PG14_PARAMETER_ATTRIBUTES = (
    "kappa_ir_parameter",
    "gamma1_parameter",
    "gamma2_parameter",
    "irradiation_temperature_parameter",
    "alpha_parameter",
)


def default_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the default 301-point wavelength grid and inferred bin edges."""

    wavelength = np.geomspace(
        DEFAULT_WAVELENGTH_MIN_MICRON,
        DEFAULT_WAVELENGTH_MAX_MICRON,
        DEFAULT_N_WAVELENGTH,
    )
    return wavelength, infer_wavelength_bin_edges(wavelength)


def grid_from_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Read only wavelength centres and edges from a grid-template NPZ."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        required = {"wavelength", "wavelength_bin_edges"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"grid NPZ is missing arrays: {missing}")
        wavelength = np.asarray(archive["wavelength"], dtype=float)
        edges = np.asarray(archive["wavelength_bin_edges"], dtype=float)
    validate_grid(wavelength, edges)
    return wavelength, edges, sha256_file(source)


def validate_grid(wavelength: np.ndarray, edges: np.ndarray) -> None:
    """Check wavelength centres and bin edges before any model work starts."""

    wavelength = np.asarray(wavelength, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if wavelength.ndim != 1 or wavelength.size != DEFAULT_N_WAVELENGTH:
        raise ValueError(f"wavelength grid must contain {DEFAULT_N_WAVELENGTH} values")
    if edges.ndim != 1 or edges.size != wavelength.size + 1:
        raise ValueError("bin-edge array must contain n_wavelength + 1 values")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(edges)):
        raise ValueError("wavelength centres and edges must be finite")
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelengths must be positive")
    if not np.all(np.diff(wavelength) > 0.0) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("wavelength centres and edges must increase strictly")
    if not (edges[0] < wavelength[0] and edges[-1] > wavelength[-1]):
        raise ValueError("bin edges must enclose the first and last wavelength centres")


def synthetic_flux(
    model_values: np.ndarray,
    *,
    sigma_ppm: float = DEFAULT_SIGMA_PPM,
    add_noise: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return simulated data and its absolute uncertainty in eclipse depth."""

    model = np.asarray(model_values, dtype=float)
    if model.ndim != 1 or model.size != DEFAULT_N_WAVELENGTH:
        raise ValueError(f"forward spectrum must contain {DEFAULT_N_WAVELENGTH} values")
    if not np.all(np.isfinite(model)):
        raise ValueError("forward spectrum must contain finite values")
    if not np.isfinite(sigma_ppm) or sigma_ppm <= 0.0:
        raise ValueError("sigma_ppm must be finite and positive")
    if add_noise and seed is None:
        raise ValueError("--seed is required when Gaussian noise is added")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative")
    uncertainty = np.full(model.shape, sigma_ppm * 1.0e-6, dtype=float)
    if not add_noise:
        return model.copy(), uncertainty
    rng = np.random.default_rng(seed)
    return model + rng.normal(0.0, uncertainty), uncertainty


def validate_injection_config(config: TaskConfig) -> dict[str, object]:
    """Require R1000, Virga clouds, and all five retrieved PG14 parameters."""

    if config.opacity.resolution != "R1000":
        raise ValueError("Virga injection recovery requires opacity resolution R1000")
    if len(config.observations.datasets) != 1:
        raise ValueError("injection recovery requires exactly one observation dataset")
    regions = configured_regions(config)
    virga_models = sorted(
        {
            str(region.clouds.model)
            for region in regions
            if region.clouds.model == "virga_eddy_sed"
        }
    )
    if not virga_models:
        raise ValueError("at least one configured atmosphere region must use Virga")

    configured_parameters = {item.name for item in config.parameters}
    pg14_names: set[str] = set()
    region_models: dict[str, str] = {}
    for region in regions:
        temperature = region.atmosphere.temperature
        model = str(temperature.model)
        region_models[region.name] = model
        if model != "parmentier_guillot_2014":
            raise ValueError(
                f"region {region.name!r} must use the fitted PG14 temperature model"
            )
        pg14_names.update(
            str(getattr(temperature, attribute))
            for attribute in _PG14_PARAMETER_ATTRIBUTES
        )
    missing = sorted(pg14_names - configured_parameters)
    if missing:
        raise ValueError(f"PG14 parameters must be fitted in the retrieval: {missing}")

    return {
        "opacity_resolution": config.opacity.resolution,
        "cloud_models": sorted({str(region.clouds.model) for region in regions}),
        "temperature_models": region_models,
        "pg14_fitted_parameter_names": sorted(pg14_names),
    }


def weighted_parameter_recovery(
    names: tuple[str, ...],
    samples: np.ndarray,
    weights: np.ndarray,
    truth: dict[str, float],
) -> dict[str, dict[str, float | bool]]:
    """Summarize posterior medians and central 68/95 percent intervals."""

    values = np.asarray(samples, dtype=float)
    probability = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names) or values.shape[0] == 0:
        raise ValueError("posterior samples do not match the parameter names")
    if probability.shape != (values.shape[0],):
        raise ValueError("posterior weights do not match the sample count")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(probability)):
        raise ValueError("posterior samples and weights must be finite")
    if np.any(probability < 0.0) or float(np.sum(probability)) <= 0.0:
        raise ValueError("posterior weights must be non-negative with a positive sum")
    probability = probability / np.sum(probability)
    summaries: dict[str, dict[str, float | bool]] = {}
    for index, name in enumerate(names):
        if name not in truth:
            raise ValueError(f"truth does not contain parameter {name!r}")
        q = weighted_quantile(
            values[:, index], probability, (0.025, 0.16, 0.50, 0.84, 0.975)
        )
        mean = float(np.sum(values[:, index] * probability))
        variance = float(np.sum(np.square(values[:, index] - mean) * probability))
        standard_deviation = float(np.sqrt(max(variance, 0.0)))
        injected = float(truth[name])
        summaries[name] = {
            "truth": injected,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "q02_5": float(q[0]),
            "q16": float(q[1]),
            "weighted_median": float(q[2]),
            "q84": float(q[3]),
            "q97_5": float(q[4]),
            "truth_in_68_percent_interval": bool(q[1] <= injected <= q[3]),
            "truth_in_95_percent_interval": bool(q[0] <= injected <= q[4]),
        }
    return summaries


def generate_fixture(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    grid_npz: str | Path | None = None,
    sigma_ppm: float = DEFAULT_SIGMA_PPM,
    add_noise: bool = False,
    seed: int | None = None,
) -> dict[str, object]:
    """Write one forward truth, observation, retrieval config, and manifest."""

    source_config = Path(config_path).expanduser().resolve()
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    config = load_task_config(source_config)
    model_contract = validate_injection_config(config)
    if config.sampler.engine != "multinest":
        raise ValueError("weighted injection recovery requires a MultiNest config")
    if not np.isfinite(sigma_ppm) or sigma_ppm <= 0.0:
        raise ValueError("sigma_ppm must be finite and positive")
    if add_noise and seed is None:
        raise ValueError("a seed is required when adding Gaussian noise")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative")

    output = Path(output_directory).expanduser().resolve()
    if grid_npz is None:
        wavelength, edges = default_grid()
        grid_sha256 = None
        grid_source = "geometric grid from configured 301-point wavelength bounds"
    else:
        wavelength, edges, grid_sha256 = grid_from_npz(grid_npz)
        grid_source = str(Path(grid_npz).expanduser().resolve())

    output.mkdir(parents=True, exist_ok=False)

    observable, flux_unit = _observable(config.radiative_transfer.model)
    uncertainty = np.full(wavelength.shape, sigma_ppm * 1.0e-6)
    placeholder = Observation(
        wavelength=wavelength,
        wavelength_bin_edges=edges,
        flux=np.full(wavelength.shape, 0.01),
        uncertainty=uncertainty,
        wavelength_unit="micron",
        flux_unit=flux_unit,
        observable=observable,
        instrument="Synthetic R1000 self-injection",
        metadata={"purpose": "temporary forward-model grid placeholder"},
    )

    work = output / "forward"
    cache = output / "opacity_cache"
    scratch = output / "scratch"
    placeholder_path = output / ".forward_grid_placeholder.npz"
    save_observation_npz(placeholder, placeholder_path)
    forward_config = _copy_config_for_forward(
        config,
        placeholder_path=placeholder_path,
        output_directory=work,
        cache_directory=cache,
        scratch_directory=scratch,
    )
    initialize_task_directories(forward_config)
    observations = load_observations(forward_config)
    prepare_opacity(forward_config, observations)
    forward_path = run_forward_task(forward_config, source_config)

    dataset_name = config.observations.datasets[0]
    with np.load(forward_path, allow_pickle=False) as archive:
        model_wavelength = np.asarray(
            archive[f"{dataset_name}_wavelength_micron"], dtype=float
        )
        model_values = np.asarray(archive[f"{dataset_name}_model"], dtype=float)
        parameters = {
            key.removeprefix("parameter_"): float(archive[key])
            for key in archive.files
            if key.startswith("parameter_")
        }
    if not np.array_equal(model_wavelength, wavelength):
        raise ValueError("configured forward model did not preserve the requested grid")
    data, uncertainty = synthetic_flux(
        model_values,
        sigma_ppm=sigma_ppm,
        add_noise=add_noise,
        seed=seed,
    )
    observation = Observation(
        wavelength=wavelength,
        wavelength_bin_edges=edges,
        flux=data,
        uncertainty=uncertainty,
        wavelength_unit="micron",
        flux_unit=flux_unit,
        observable=observable,
        instrument="Synthetic R1000 self-injection",
        metadata={
            "validation_kind": "self_injection",
            "truth_source": "same configured ROBERT forward model used by retrieval",
            "noise": "independent Gaussian" if add_noise else "none added",
            "noise_seed": "none" if seed is None else str(seed),
            "uncertainty_ppm": f"{sigma_ppm:.12g}",
            "grid_source": grid_source,
            "grid_source_sha256": "none" if grid_sha256 is None else grid_sha256,
        },
    )
    observation_path = output / "synthetic_observation.npz"
    save_observation_npz(observation, observation_path)

    retrieval_config_path = output / "retrieval_config.yaml"
    _write_retrieval_config(config, retrieval_config_path, output, observation_path)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "validation_kind": "self_injection",
        "independent_validation": False,
        "case": config.run.name,
        "source_config": str(source_config),
        "source_config_sha256": sha256_file(source_config),
        "forward_model": str(forward_path),
        "forward_model_sha256": sha256_file(forward_path),
        "synthetic_observation": str(observation_path),
        "synthetic_observation_sha256": sha256_file(observation_path),
        "retrieval_config": str(retrieval_config_path),
        "retrieval_config_sha256": sha256_file(retrieval_config_path),
        "grid_source": grid_source,
        "grid_source_sha256": "none" if grid_sha256 is None else grid_sha256,
        "n_wavelength": int(wavelength.size),
        "wavelength_min_micron": float(wavelength[0]),
        "wavelength_max_micron": float(wavelength[-1]),
        "bin_edge_min_micron": float(edges[0]),
        "bin_edge_max_micron": float(edges[-1]),
        "uncertainty_ppm": float(sigma_ppm),
        "noise_added": bool(add_noise),
        "noise_seed": seed,
        "parameters": parameters,
        "parameter_initialization": {
            item.name: ("configured value" if item.value is not None else "prior midpoint")
            for item in config.parameters
        },
        "model_contract": model_contract,
        "scientific_scope": (
            "This is a same-model self-injection. It checks the configured pipeline; "
            "it does not establish independent cloud-physics or GCM recovery."
        ),
    }
    truth_path = output / "injection_truth.json"
    _write_json_exclusive(truth_path, payload)
    placeholder_path.unlink()
    return payload


def evaluate_recovery(
    config_path: str | Path,
    *,
    truth_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate an existing MultiNest result against its self-injected truth."""

    source_config = Path(config_path).expanduser().resolve()
    config = load_task_config(source_config)
    model_contract = validate_injection_config(config)
    if config.sampler.engine != "multinest":
        raise ValueError("weighted injection recovery requires a MultiNest result")
    truth_file = (
        Path(truth_path).expanduser().resolve()
        if truth_path is not None
        else source_config.parent / "injection_truth.json"
    )
    truth_payload = json.loads(truth_file.read_text(encoding="utf-8"))
    if truth_payload.get("validation_kind") != "self_injection":
        raise ValueError("truth manifest is not marked as a self-injection")
    truth = {str(k): float(v) for k, v in truth_payload["parameters"].items()}
    result_directory = config.outputs.directory / "multinest"
    result_path = result_directory / "result.json"
    arrays_path = result_directory / "result_arrays.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    names = tuple(str(name) for name in result["parameter_names"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        weights = (
            np.asarray(archive["weights"], dtype=float)
            if "weights" in archive.files
            else np.ones(samples.shape[0], dtype=float)
        )
    if set(names) != set(truth):
        raise ValueError("posterior parameters do not match the injected truth")
    summaries = weighted_parameter_recovery(names, samples, weights, truth)
    posterior_median = {
        name: float(summary["weighted_median"])
        for name, summary in summaries.items()
    }

    observations = load_observations(config)
    problem = build_problem(config, observations)
    dataset_name = config.observations.datasets[0]
    model = problem.model_spectra(posterior_median)[dataset_name]
    observation = observations.datasets[0].observation
    residual = (observation.flux - model.values) / observation.uncertainty
    chi_square = float(np.sum(np.square(residual)))
    dof = observation.n_points - problem.ndim
    report: dict[str, object] = {
        "schema_version": "1.0",
        "validation_kind": "self_injection",
        "independent_validation": False,
        "case": config.run.name,
        "inference_converged": bool(result["converged"]),
        "parameter_recoveries": summaries,
        "all_parameters_inside_95_percent_interval": all(
            bool(value["truth_in_95_percent_interval"])
            for value in summaries.values()
        ),
        "chi_square_at_weighted_posterior_median": chi_square,
        "reduced_chi_square_at_weighted_posterior_median": (
            chi_square / dof if dof > 0 else None
        ),
        "n_wavelength": observation.n_points,
        "n_parameters": problem.ndim,
        "effective_sample_size": float(
            1.0 / np.sum(np.square(weights / np.sum(weights)))
        ),
        "truth_manifest": str(truth_file),
        "truth_manifest_sha256": sha256_file(truth_file),
        "result": str(result_path),
        "model_contract": model_contract,
        "scientific_scope": (
            "Weighted recovery diagnostics for a same-model self-injection only."
        ),
    }
    target = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else truth_file.parent / "injection_recovery_report.json"
    )
    _write_json_exclusive(target, report)
    return report


def _copy_config_for_forward(
    config: TaskConfig,
    *,
    placeholder_path: Path,
    output_directory: Path,
    cache_directory: Path,
    scratch_directory: Path,
) -> TaskConfig:
    observations = config.observations.model_copy(
        update={"loader": "robert_npz", "path": placeholder_path}
    )
    outputs = config.outputs.model_copy(update={"directory": output_directory})
    opacity = config.opacity.model_copy(update={"cache_directory": cache_directory})
    runtime = config.runtime.model_copy(update={"scratch_directory": scratch_directory})
    paths = config.paths
    if paths is not None:
        paths = paths.model_copy(
            update={
                "project_directory": output_directory.parent,
                "observations_directory": placeholder_path,
                "opacity_cache_directory": cache_directory,
                "output_directory": output_directory,
                "scratch_directory": scratch_directory,
            }
        )
    return config.model_copy(
        update={
            "observations": observations,
            "outputs": outputs,
            "opacity": opacity,
            "runtime": runtime,
            "paths": paths,
        }
    )


def _write_retrieval_config(
    config: TaskConfig,
    target: Path,
    output: Path,
    observation_path: Path,
) -> None:
    raw = config.model_dump(mode="json")
    cache = output / "opacity_cache"
    retrieval_output = output / "retrieval"
    scratch = output / "scratch"
    raw["observations"]["loader"] = "robert_npz"
    raw["observations"]["path"] = str(observation_path)
    raw["opacity"]["cache_directory"] = str(cache)
    raw["outputs"]["directory"] = str(retrieval_output)
    raw["runtime"]["scratch_directory"] = str(scratch)
    paths = raw.get("paths") or {}
    raw["paths"] = paths
    paths.update(
        {
            "project_directory": str(output),
            "observations_directory": str(observation_path),
            "opacity_cache_directory": str(cache),
            "output_directory": str(retrieval_output),
            "scratch_directory": str(scratch),
        }
    )
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _observable(model: str) -> tuple[str, str]:
    if model == "emission":
        return "eclipse_depth", "eclipse_depth"
    if model == "transmission":
        return "transit_depth", "transit_depth"
    raise ValueError(f"unsupported radiative-transfer model: {model}")


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mpi_rank() -> int:
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except ImportError:
        for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"):
            if key in os.environ:
                return int(os.environ[key])
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--grid-npz", type=Path)
    parser.add_argument("--sigma-ppm", type=float, default=DEFAULT_SIGMA_PPM)
    parser.add_argument("--noise", action="store_true", help="add Gaussian noise")
    parser.add_argument("--seed", type=int, help="required when --noise is set")
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    # Nonzero ranks exit before touching files. The generator itself runs once
    # on rank zero and does not split the forward-model calculation across ranks.
    if _mpi_rank() != 0:
        return 0
    if args.generate:
        if args.output_dir is None:
            parser.error("--output-dir is required with --generate")
        payload = generate_fixture(
            args.config,
            args.output_dir,
            grid_npz=args.grid_npz,
            sigma_ppm=args.sigma_ppm,
            add_noise=args.noise,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    report = evaluate_recovery(
        args.config,
        truth_path=args.truth,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
