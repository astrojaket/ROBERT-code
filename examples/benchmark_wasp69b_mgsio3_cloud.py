"""Benchmark six-gas cloud-free and MgSiO3/Mie/SH4 emission forward models.

The comparison reuses the WASP-69b full-band cloud workflow so that the cloud-free
and cloudy cases have identical atmosphere, opacity, CIA, Rayleigh, geometry,
pressure, spectral, and correlated-k settings.  The only additional cloudy
work is the MgSiO3 particle optics, cloud optical-depth construction, and SH4
multiple-scattering solution.
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
from typing import Callable, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)

import numba
import numpy as np
import scipy

from robert_exoplanets import ParameterizedEmissionForwardModel
from robert_exoplanets.core import Spectrum

from retrieve_wasp69b_mie_cloud import SPECIES, build_problem


PARAMETERS = {
    "metallicity": 1.0,
    "CtoO": 0.775,
    "kappa_IR": np.sqrt(1.0e-5),
    "gamma1": 1.0,
    "gamma2": 1.0,
    "T_irr": 1500.0,
    "alpha": 0.5,
    "log_cloud_mass_fraction": -7.0,
    "log_cloud_radius_micron": -1.0,
    "log_cloud_top_pressure_bar": -4.0,
    "log_cloud_base_pressure_bar": 0.5,
    "miri_offset": 0.0,
}


@dataclass(frozen=True)
class NamedModels:
    """Evaluate named spectral models without changing their call paths."""

    models: Mapping[str, Callable[[Mapping[str, float]], Spectrum]]

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, Spectrum]:
        return {name: model(parameters) for name, model in self.models.items()}


def _cloud_free_counterpart(cloudy) -> ParameterizedEmissionForwardModel:
    """Build a cloud-free model from the exact prepared inputs of a cloudy model."""

    return ParameterizedEmissionForwardModel(
        planet=cloudy.planet,
        star=cloudy.star,
        spectral_grid=cloudy.spectral_grid,
        atmosphere_builder=cloudy.atmosphere_builder,
        opacity_provider=cloudy.opacity_provider,
        config=cloudy.config,
        cia_table=cloudy.cia_table,
        geometry=cloudy.geometry,
    )


def _summary(elapsed: list[float]) -> dict[str, object]:
    values = np.asarray(elapsed, dtype=float)
    return {
        "call_seconds": values.tolist(),
        "median_seconds": float(np.median(values)),
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
        "calls_per_second": float(1.0 / np.median(values)),
    }


def _profile(call: Callable[[], object], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    call()
    profiler.disable()
    profiler.dump_stats(path)


def run(
    repeats: int,
    warmups: int,
    *,
    cloud_free_profile: Path | None = None,
    cloudy_profile: Path | None = None,
) -> dict[str, object]:
    """Build, warm, time, profile, and validate the matched forward models."""

    if len(SPECIES) != 6:
        raise RuntimeError(f"benchmark requires six gases, found {len(SPECIES)}")

    setup_started = perf_counter()
    problem = build_problem(
        cloud_mode="catalog",
        material="MgSiO3",
        particle_density_kg_m3=3200.0,
    )
    setup_seconds = perf_counter() - setup_started
    cloudy = problem.forward_model
    clear = NamedModels(
        {
            name: _cloud_free_counterpart(model)
            for name, model in problem.forward_model.models.items()
        }
    )

    def cloud_free_call():
        return clear(PARAMETERS)

    def cloudy_call():
        return cloudy(PARAMETERS)

    for _ in range(warmups):
        cloud_free_call()
        cloudy_call()

    # Alternate which case runs first to reduce slow drift as a source of bias.
    cloud_free_elapsed: list[float] = []
    cloudy_elapsed: list[float] = []
    for repeat in range(repeats):
        ordered = (
            ((cloud_free_call, cloud_free_elapsed), (cloudy_call, cloudy_elapsed))
            if repeat % 2 == 0
            else ((cloudy_call, cloudy_elapsed), (cloud_free_call, cloud_free_elapsed))
        )
        for call, elapsed in ordered:
            started = perf_counter()
            call()
            elapsed.append(perf_counter() - started)

    cloud_free_spectra = cloud_free_call()
    cloudy_spectra = cloudy_call()
    diagnostic_cloudy_spectra = {
        name: replace(
            model,
            config=replace(model.config, compute_diagnostics=True),
        )(PARAMETERS)
        for name, model in problem.forward_model.models.items()
    }
    if cloud_free_spectra.keys() != cloudy_spectra.keys():
        raise RuntimeError("cloud-free and cloudy benchmark dataset names differ")
    finite = all(
        np.all(np.isfinite(spectrum.values))
        for spectra in (cloud_free_spectra, cloudy_spectra)
        for spectrum in spectra.values()
    )
    same_grids = all(
        np.array_equal(
            cloud_free_spectra[name].spectral_grid.values,
            cloudy_spectra[name].spectral_grid.values,
        )
        for name in cloud_free_spectra
    )
    maximum_spectral_difference = max(
        float(np.max(np.abs(cloudy_spectra[name].values - cloud_free_spectra[name].values)))
        for name in cloud_free_spectra
    )
    fast_reference_max_abs_difference = max(
        float(
            np.max(
                np.abs(
                    cloudy_spectra[name].values - diagnostic_cloudy_spectra[name].values
                )
            )
        )
        for name in cloudy_spectra
    )
    fast_reference_max_relative_difference = max(
        float(
            np.max(
                np.abs(
                    cloudy_spectra[name].values - diagnostic_cloudy_spectra[name].values
                )
                / np.maximum(
                    np.abs(diagnostic_cloudy_spectra[name].values),
                    1.0e-300,
                )
            )
        )
        for name in cloudy_spectra
    )
    fast_matches_reference = all(
        np.allclose(
            cloudy_spectra[name].values,
            diagnostic_cloudy_spectra[name].values,
            rtol=2.0e-11,
            atol=2.0e-13,
        )
        for name in cloudy_spectra
    )
    if not finite or not same_grids or maximum_spectral_difference == 0.0:
        raise RuntimeError("cloud-free/cloudy benchmark output validation failed")
    if not fast_matches_reference:
        raise RuntimeError(
            "spectrum-only cloudy output differs from diagnostic reference"
        )

    _profile(cloud_free_call, cloud_free_profile)
    _profile(cloudy_call, cloudy_profile)

    cloud_free_result = _summary(cloud_free_elapsed)
    cloudy_result = _summary(cloudy_elapsed)
    cloud_free_median = float(cloud_free_result["median_seconds"])
    cloudy_median = float(cloudy_result["median_seconds"])
    models = problem.forward_model.models
    first_model = next(iter(models.values()))
    return {
        "benchmark": "WASP-69b six-gas cloud-free versus MgSiO3 Mie SH4 emission",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "omp_threads": os.environ["OMP_NUM_THREADS"],
            "numba_threads": os.environ["NUMBA_NUM_THREADS"],
        },
        "configuration": {
            "species": list(SPECIES),
            "n_species": len(SPECIES),
            "datasets": list(models),
            "n_observation_points": problem.observations.n_points,
            "n_layers": first_model.pressure_grid.n_layers,
            "n_g_points": int(first_model.prepared_opacity.g_samples.size),
            "material": "MgSiO3",
            "particle_density_kg_m3": 3200.0,
            "effective_radius_micron": 10.0 ** PARAMETERS["log_cloud_radius_micron"],
            "condensate_mass_fraction": 10.0 ** PARAMETERS["log_cloud_mass_fraction"],
            "cloud_top_pressure_bar": 10.0 ** PARAMETERS["log_cloud_top_pressure_bar"],
            "cloud_base_pressure_bar": 10.0
            ** PARAMETERS["log_cloud_base_pressure_bar"],
            "multiple_scattering_backend": "sh4",
            "phase_function": "exact Mie moments through l=4 with delta-M",
            "repeats": repeats,
            "warmups": warmups,
        },
        "cloudy_problem_setup_seconds": setup_seconds,
        "emission": cloud_free_result,
        "mgsio3_mie_sh4_emission": cloudy_result,
        "cloudy_over_cloud_free_median_time": cloudy_median / cloud_free_median,
        "cloud_free_over_cloudy_throughput": cloudy_median / cloud_free_median,
        "additional_cloudy_median_seconds": cloudy_median - cloud_free_median,
        "validation": {
            "all_outputs_finite": finite,
            "spectral_grids_equal": same_grids,
            "maximum_absolute_eclipse_depth_difference": (maximum_spectral_difference),
            "spectrum_only_matches_diagnostic_reference": fast_matches_reference,
            "spectrum_only_reference_max_abs_difference": (
                fast_reference_max_abs_difference
            ),
            "spectrum_only_reference_max_relative_difference": (
                fast_reference_max_relative_difference
            ),
        },
        "profiles": {
            "clear": None if cloud_free_profile is None else str(cloud_free_profile),
            "cloudy": None if cloudy_profile is None else str(cloudy_profile),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-cloud-free", type=Path)
    parser.add_argument("--profile-cloudy", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups non-negative")
    result = run(
        args.repeats,
        args.warmups,
        cloud_free_profile=args.profile_clear,
        cloudy_profile=args.profile_cloudy,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
