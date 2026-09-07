#!/usr/bin/env python3
"""Benchmark one real ExoMol opacity-sampling emission likelihood on CPU and JAX.

This is deliberately a *single-proposal* benchmark.  It selects JAX device
zero for the requested platform and never distributes work across visible
accelerators.  It does not generate opacity data: the caller supplies the
six ExoMolOP TauREx HDF5 files in ``--data-dir``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.atmosphere.temperature import _parmentier_guillot_eta
from robert_exoplanets.core import PressureGrid, RobertValidationError
from robert_exoplanets.metal import (
    MetalBinnedProjection,
    MetalResourcePolicy,
    clear_metal_caches,
    prepare_opacity_sampling_cloud_free_emission_from_prepared,
    require_accelerator_runtime,
)
from robert_exoplanets.opacity import OpacitySamplingProvider
from robert_exoplanets.rt import assemble_opacity_sampling_gas_optical_depth
from robert_exoplanets.rt.emission import solve_emission_spectrum
from robert_exoplanets.rt.geometry import gauss_legendre_disk_geometry


SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
PG14_PARAMETER_NAMES = ("kappa_IR", "gamma1", "gamma2", "T_irr", "alpha")
DEFAULT_PARAMETERS = np.array(
    [3.16e-3, 0.12, 1.7, 1510.0, 0.38, -3.0, -4.0, -4.5, -5.0, -5.0, -5.0],
    dtype=np.float64,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "metal", "cuda"), required=True)
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--sampling-stride", type=int, default=1)
    parser.add_argument("--likelihood-bins", type=int, default=128)
    parser.add_argument(
        "--maximum-working-set-gib",
        type=float,
        default=1.0,
        help="Explicit accelerator allocation ceiling; batch size remains one.",
    )
    parser.add_argument("--minimum-pressure-bar", type=float, default=1.0e-5)
    parser.add_argument("--maximum-pressure-bar", type=float, default=100.0)
    return parser.parse_args()


def _pg14(
    pressure_bar: np.ndarray, parameters: np.ndarray, gravity: float
) -> np.ndarray:
    """Exact CPU PG14 equation, evaluated at centres or layer edges."""

    kappa, gamma1, gamma2, irradiation, alpha = parameters[:5]
    tau = kappa * pressure_bar * 1.0e5 / gravity
    fourth = 0.75 * 100.0**4 * (2.0 / 3.0 + tau)
    fourth += (
        0.75
        * irradiation**4
        * (
            (1.0 - alpha) * _parmentier_guillot_eta(gamma1, tau)
            + alpha * _parmentier_guillot_eta(gamma2, tau)
        )
    )
    return np.power(fourth, 0.25)


def _mean_molecular_weight(parameters: np.ndarray) -> tuple[np.ndarray, float]:
    trace = np.power(10.0, parameters[5:])
    molecular_weights = np.array(
        [18.01528, 28.0101, 44.0095, 16.0425, 17.0305, 27.0253]
    )
    return trace, float((1.0 - trace.sum()) * 2.3 + trace @ molecular_weights)


def _cpu_spectrum_and_loglike(
    parameters: np.ndarray,
    *,
    provider: OpacitySamplingProvider,
    prepared: Any,
    pressure_grid: PressureGrid,
    geometry: Any,
    gravity: float,
    planet_radius: float,
    star_radius: float,
    star_temperature: float,
    projection: np.ndarray,
    observation: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, float]:
    temperature = _pg14(np.asarray(pressure_grid.centers), parameters, gravity)
    temperature_edges = _pg14(np.asarray(pressure_grid.edges), parameters, gravity)
    trace, mmw = _mean_molecular_weight(parameters)
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=temperature_edges,
        composition={
            name: np.full(pressure_grid.n_layers, value)
            for name, value in zip(SPECIES, trace, strict=True)
        },
        mean_molecular_weight=np.full(pressure_grid.n_layers, mmw),
    )
    tau = assemble_opacity_sampling_gas_optical_depth(
        atmosphere, provider, prepared, gravity_m_s2=gravity
    )
    spectrum = solve_emission_spectrum(
        tau,
        geometry=geometry,
        planet_radius_m=planet_radius,
        star_radius_m=star_radius,
        star_temperature_k=star_temperature,
        thermal_integration_backend="numba",
    ).values
    loglike = float(
        -0.5 * np.sum(((observation - projection @ spectrum) / uncertainty) ** 2)
    )
    return np.asarray(spectrum), loglike


def _timed_cpu(function, repeats: int) -> tuple[float, list[float]]:
    function()  # CPU JIT/setup is excluded from the reproducible warm statistic.
    samples: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return float(np.median(samples)), samples


def _nbytes(*arrays: Any) -> int:
    return int(sum(int(getattr(value, "nbytes", 0)) for value in arrays))


def _peak_host_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _git_state() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _bin_average_projection(n_native: int, n_bins: int) -> np.ndarray:
    """Return an explicit, reproducible native-to-observation averaging matrix."""

    if n_bins < 1 or n_bins > n_native:
        raise RobertValidationError(
            "likelihood-bins must lie in [1, native_wavelengths]"
        )
    edges = np.linspace(0, n_native, n_bins + 1, dtype=int)
    matrix = np.zeros((n_bins, n_native), dtype=np.float32)
    for row, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        matrix[row, start:stop] = 1.0 / (stop - start)
    return matrix


def main() -> dict[str, Any]:
    args = _parse_args()
    if (
        args.layers < 2
        or args.warm_repeats < 1
        or args.sampling_stride < 1
        or not np.isfinite(args.maximum_working_set_gib)
        or args.maximum_working_set_gib <= 0.0
    ):
        raise RobertValidationError(
            "layers, repeats, sampling stride, and working-set limit must be positive"
        )
    paths = {name: args.data_dir / f"{name}.h5" for name in SPECIES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RobertValidationError(
            "missing ExoMol opacity files: " + ", ".join(missing)
        )

    # File loading and table transfer are setup, never included in either timing.
    provider = OpacitySamplingProvider.from_exomol_paths(
        paths,
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=True,
    )
    pressure = PressureGrid.logspace(
        args.minimum_pressure_bar,
        args.maximum_pressure_bar,
        args.layers,
        unit="bar",
        name="opacity-sampling accelerator benchmark",
    )
    grid = provider.native_spectral_grid(sampling=args.sampling_stride)
    projection = _bin_average_projection(grid.size, args.likelihood_bins)
    projection_edges = np.linspace(
        0, grid.size, args.likelihood_bins + 1, dtype=np.int32
    )
    prepared = provider.prepare(grid, pressure, SPECIES)
    geometry = gauss_legendre_disk_geometry(4)
    gravity, planet_radius, star_radius, star_temperature = (
        10.0,
        7.1492e7,
        6.957e8,
        5000.0,
    )
    truth_spectrum, _ = _cpu_spectrum_and_loglike(
        DEFAULT_PARAMETERS,
        provider=provider,
        prepared=prepared,
        pressure_grid=pressure,
        geometry=geometry,
        gravity=gravity,
        planet_radius=planet_radius,
        star_radius=star_radius,
        star_temperature=star_temperature,
        projection=projection,
        observation=np.zeros(args.likelihood_bins),
        uncertainty=np.ones(args.likelihood_bins),
    )
    # A non-zero but deterministic residual makes likelihood agreement meaningful.
    truth_observation = projection @ truth_spectrum
    observation = truth_observation * (
        1.0 + 2.0e-3 * np.sin(np.arange(args.likelihood_bins))
    )
    uncertainty = np.maximum(np.abs(truth_observation) * 1.0e-2, 1.0e-12)

    def cpu_call() -> tuple[np.ndarray, float]:
        return _cpu_spectrum_and_loglike(
            DEFAULT_PARAMETERS,
            provider=provider,
            prepared=prepared,
            pressure_grid=pressure,
            geometry=geometry,
            gravity=gravity,
            planet_radius=planet_radius,
            star_radius=star_radius,
            star_temperature=star_temperature,
            projection=projection,
            observation=observation,
            uncertainty=uncertainty,
        )

    cpu_spectrum, cpu_loglike = cpu_call()
    cpu_warm_seconds, cpu_warm_samples = _timed_cpu(cpu_call, args.warm_repeats)

    runtime = require_accelerator_runtime(args.platform)
    device_projection = MetalBinnedProjection.from_edges(
        projection_edges, grid.size, runtime
    )
    resource_policy = MetalResourcePolicy(
        total_memory_bytes=max(1, int(args.maximum_working_set_gib * 1024**3)),
        maximum_fraction=1.0,
        absolute_maximum_bytes=max(1, int(args.maximum_working_set_gib * 1024**3)),
        maximum_batch_size=1,
    )
    model = prepare_opacity_sampling_cloud_free_emission_from_prepared(
        provider=provider,
        prepared=prepared,
        runtime=runtime,
        projection=device_projection,
        observation=observation,
        uncertainty=uncertainty,
        gravity_m_s2=gravity,
        planet_radius_m=planet_radius,
        star_radius_m=star_radius,
        star_temperature_k=star_temperature,
        resource_policy=resource_policy,
    )
    compiled = runtime.jax.jit(model.loglike)
    parameter_device = runtime.put(DEFAULT_PARAMETERS)
    cold_start = perf_counter()
    cold_value = compiled(parameter_device)
    cold_value.block_until_ready()
    cold_seconds = perf_counter() - cold_start
    warm_samples: list[float] = []
    for _ in range(args.warm_repeats):
        start = perf_counter()
        compiled(parameter_device).block_until_ready()
        warm_samples.append(perf_counter() - start)
    device_spectrum = np.asarray(
        model.native_eclipse_depth(parameter_device).block_until_ready()
    )
    device_loglike = float(np.asarray(compiled(parameter_device).block_until_ready()))
    repeated_loglikes = [
        float(np.asarray(compiled(parameter_device).block_until_ready()))
        for _ in range(3)
    ]

    sensitivity: dict[str, dict[str, float | bool]] = {}
    for index, name in enumerate(SPECIES):
        changed = DEFAULT_PARAMETERS.copy()
        changed[5 + index] += 0.5
        cpu_changed, cpu_changed_loglike = _cpu_spectrum_and_loglike(
            changed,
            provider=provider,
            prepared=prepared,
            pressure_grid=pressure,
            geometry=geometry,
            gravity=gravity,
            planet_radius=planet_radius,
            star_radius=star_radius,
            star_temperature=star_temperature,
            projection=projection,
            observation=observation,
            uncertainty=uncertainty,
        )
        device_changed = np.asarray(
            model.native_eclipse_depth(runtime.put(changed)).block_until_ready()
        )
        device_changed_loglike = float(
            np.asarray(model.loglike(runtime.put(changed)).block_until_ready())
        )
        cpu_delta = float(np.max(np.abs(cpu_changed - cpu_spectrum)))
        device_delta = float(np.max(np.abs(device_changed - device_spectrum)))
        cpu_response = cpu_changed - cpu_spectrum
        device_response = device_changed - device_spectrum
        response_scale = max(float(np.max(np.abs(cpu_response))), 1.0e-30)
        response_rms_relative_to_peak = float(
            np.sqrt(np.mean((device_response - cpu_response) ** 2)) / response_scale
        )
        if cpu_delta <= 0.0 or device_delta <= 0.0:
            raise RobertValidationError(
                f"opacity-sampling benchmark is insensitive to {name}"
            )
        sensitivity[name] = {
            "cpu_max_absolute_change": cpu_delta,
            "jax_max_absolute_change": device_delta,
            "both_nonzero": bool(cpu_delta > 0.0 and device_delta > 0.0),
            "perturbed_spectrum_rms_relative": float(
                np.sqrt(
                    np.mean(
                        (
                            (device_changed - cpu_changed)
                            / np.maximum(np.abs(cpu_changed), 1.0e-30)
                        )
                        ** 2
                    )
                )
            ),
            "response_rms_relative_to_cpu_peak": response_rms_relative_to_peak,
            "cpu_delta_loglike": float(cpu_changed_loglike - cpu_loglike),
            "jax_delta_loglike": float(device_changed_loglike - device_loglike),
            "delta_loglike_absolute_difference": float(
                abs(
                    (device_changed_loglike - device_loglike)
                    - (cpu_changed_loglike - cpu_loglike)
                )
            ),
        }

    device_arrays = (
        model.opacity.log_cross_sections,
        model.opacity.log_pressure_grid_bar,
        model.opacity.temperature_grid_k,
        model.pressure_center_bar,
        model.pressure_edge_bar,
        model.wavelength_micron,
        model.likelihood.projection.starts,
        model.likelihood.projection.stops,
        model.likelihood.projection.counts,
        model.likelihood.observation,
        model.likelihood.uncertainty,
    )
    conservative_peak_bytes = (
        resource_policy.estimate_opacity_sampling_bytes(
            species=len(SPECIES),
            pressure_grid=prepared.stacked_log_cross_sections.shape[1],
            temperature_grid=prepared.stacked_log_cross_sections.shape[2],
            layers=pressure.n_layers,
            spectral=grid.size,
            disc_points=geometry.n_points,
        )
        + device_projection.device_bytes * 3
    )
    metadata = runtime.metadata
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "single-proposal ExoMol opacity-sampling clear-emission likelihood",
        "timing_scope": "setup/table transfer excluded; JAX cold includes compile plus first execution; warm calls synchronize device",
        "accelerator": {
            **metadata,
            "requested_platform": args.platform,
            "visible_device_count": runtime.visible_device_count,
            "selected_device_index": 0,
        },
        "software": {
            "python": platform.python_version(),
            "robert_exoplanets": _package_version("robert-exoplanets"),
            "numpy": _package_version("numpy"),
            "numba": _package_version("numba"),
            "git": _git_state(),
        },
        "configuration": {
            "species": list(SPECIES),
            "pg14_parameter_names": list(PG14_PARAMETER_NAMES),
            "parameters": DEFAULT_PARAMETERS.tolist(),
            "layers": pressure.n_layers,
            "disc_points": geometry.n_points,
            "opacity_sampling_stride": args.sampling_stride,
            "likelihood_bins": args.likelihood_bins,
            "prepared_cache_key": prepared.cache_key,
            "opacity_interpolation": provider.interpolation,
            "pressure_bounds_bar": [
                float(np.min(pressure.centers)),
                float(np.max(pressure.centers)),
            ],
            "wavelength_bounds_micron": [
                float(np.min(grid.values)),
                float(np.max(grid.values)),
            ],
            "projection": "contiguous_equal-count-bin-average-prefix-sum",
            "projection_edges_sha256": sha256(projection_edges.tobytes()).hexdigest(),
        },
        "opacity_identifiers": dict(model.opacity_identifiers),
        "arrays": {
            "log_cross_sections_shape": list(prepared.stacked_log_cross_sections.shape),
            "native_wavelengths": grid.size,
            "visible_device_array_bytes": _nbytes(*device_arrays),
            "conservative_peak_guard_bytes": conservative_peak_bytes,
            "configured_working_set_limit_bytes": (
                resource_policy.working_set_limit_bytes
            ),
            "peak_host_rss_bytes": _peak_host_rss_bytes(),
        },
        "seconds": {
            "cpu_warm_median": cpu_warm_seconds,
            "cpu_warm_samples": cpu_warm_samples,
            "jax_cold_compile_and_first_call": cold_seconds,
            "jax_warm_median": float(np.median(warm_samples)),
            "jax_warm_samples": warm_samples,
        },
        "accuracy": {
            "spectrum_max_absolute": float(
                np.max(np.abs(device_spectrum - cpu_spectrum))
            ),
            "spectrum_rms_relative": float(
                np.sqrt(
                    np.mean(
                        (
                            (device_spectrum - cpu_spectrum)
                            / np.maximum(np.abs(cpu_spectrum), 1.0e-30)
                        )
                        ** 2
                    )
                )
            ),
            "loglike_cpu": cpu_loglike,
            "loglike_jax": device_loglike,
            "loglike_absolute_difference": float(abs(device_loglike - cpu_loglike)),
            "jax_repeat_loglike_spread": float(
                np.max(repeated_loglikes) - np.min(repeated_loglikes)
            ),
            "gas_sensitivity": sensitivity,
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    clear_metal_caches()
    return report


if __name__ == "__main__":
    main()
