"""Time two multi-instrument forward-model workflows for WASP-69b.

The compared likelihood calls are:

1. one emission calculation on the native R~1000 opacity grid, followed by
   top-hat integration onto every observation grid; and
2. one emission calculation per instrument mode using opacity pre-compressed
   into that mode's published wavelength bins.

The per-mode path is ROBERT's default for retrieval. The native path is a
plotting and diagnostic option. Setup (HDF5/cache loading and model
construction) is reported separately from repeated likelihood-call timing.
This is a validation benchmark, not a production WASP-69b retrieval model.
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Callable, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)

import h5py
import numpy as np

from robert_exoplanets import (
    CompositionMeanMolecularWeight,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    FastChemEquilibriumChemistry,
    MultiDatasetForwardModel,
    MultiDatasetGaussianLikelihood,
    ParameterizedClearSkyEmissionFactoryConfig,
    ParameterizedClearSkyEmissionModelConfig,
    ParmentierGuillot2014TemperatureProfile,
    Planet,
    PressureGrid,
    Star,
    build_parameterized_clear_sky_emission_model,
    load_schlawin2024_wasp69b,
)
from robert_exoplanets.core import SpectralGrid, Spectrum

from retrieve_wasp69b_nircam_clear import (
    CACHE,
    DATA,
    FASTCHEM,
    PATTERNS,
    PRT_DATA,
    SPECIES,
    _cia_tables,
    _load_table,
)

RJUP_M = 7.1492e7
MJUP_KG = 1.89813e27
RSUN_M = 6.957e8


@dataclass(frozen=True)
class NamedModels:
    models: Mapping[str, Callable[[Mapping[str, float]], Spectrum]]

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, Spectrum]:
        return {name: model(parameters) for name, model in self.models.items()}


def _native_table(
    path: Path, species: str, lower: float, upper: float
) -> CorrelatedKTable:
    """Read only the native-opacity wavelength slab needed by the data."""

    with h5py.File(path, "r") as handle:
        wavelength = 10000.0 / np.asarray(handle["bin_centers"], dtype=float)
        selected = np.flatnonzero((wavelength >= lower) & (wavelength <= upper))
        # Include a native sample outside each boundary for top-hat interpolation.
        start = max(0, int(selected[0]) - 1)
        stop = min(wavelength.size, int(selected[-1]) + 2)
        unit = str(handle["kcoeff"].attrs.get("units", "")).strip()
        return CorrelatedKTable(
            species=species,
            pressure_bar=np.asarray(handle["p"], dtype=float),
            temperature_K=np.asarray(handle["t"], dtype=float),
            wavenumber_cm_inverse=np.asarray(
                handle["bin_centers"][start:stop], dtype=float
            ),
            wavelength_micron=wavelength[start:stop],
            g_samples=np.asarray(handle["samples"], dtype=float),
            g_weights=np.asarray(handle["weights"], dtype=float),
            kcoeff=np.asarray(handle["kcoeff"][:, :, start:stop, :], dtype=float),
            unit=unit,
            metadata={
                "source_path": str(path.resolve()),
                "spectral_preparation": "native_R1000",
            },
        )


def _canonicalize_g(
    tables: Mapping[str, CorrelatedKTable],
) -> dict[str, CorrelatedKTable]:
    canonical = tables[SPECIES[0]]
    output = {}
    for species, table in tables.items():
        if not np.allclose(table.g_samples, canonical.g_samples, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"{species} uses a different correlated-k g grid")
        if not np.allclose(table.g_weights, canonical.g_weights, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"{species} uses different correlated-k weights")
        output[species] = replace(
            table,
            g_samples=canonical.g_samples,
            g_weights=canonical.g_weights,
        )
    return output


def _config(
    provider: CorrelatedKOpacityProvider, pressure: PressureGrid, cia
) -> ParameterizedClearSkyEmissionFactoryConfig:
    gravity = 6.67430e-11 * (0.26 * MJUP_KG) / (1.06 * RJUP_M) ** 2
    return ParameterizedClearSkyEmissionFactoryConfig(
        planet=Planet(name="WASP-69b", radius_m=1.06 * RJUP_M, mass_kg=0.26 * MJUP_KG),
        star=Star(
            name="WASP-69", radius_m=0.813 * RSUN_M, effective_temperature_k=4750.0
        ),
        temperature_profile=ParmentierGuillot2014TemperatureProfile(
            gravity=gravity, internal_temperature=100.0
        ),
        chemistry_model=FastChemEquilibriumChemistry(
            fastchem_path=FASTCHEM, metadata={"element_abundances": "asplund_2009"}
        ),
        mean_molecular_weight_model=CompositionMeanMolecularWeight(
            normalization="raw_sum"
        ),
        pressure_grid=pressure,
        cia_table=cia,
        opacity_source=provider,
        opacity_binning=None,
        model=ParameterizedClearSkyEmissionModelConfig(
            opacity_species=SPECIES,
            include_rayleigh=True,
            gas_combination="random_overlap",
            thermal_integration_backend="auto",
            metadata={"target": "WASP-69b", "purpose": "multi_instrument_timing"},
        ),
    )


def _time(call: Callable[[], float], repeats: int) -> tuple[list[float], float]:
    elapsed = []
    value = float("nan")
    for _ in range(repeats):
        start = perf_counter()
        value = float(call())
        elapsed.append(perf_counter() - start)
    return elapsed, value


def run(
    repeats: int,
    warmups: int,
    *,
    profile: Path | None = None,
) -> dict[str, object]:
    observations = load_schlawin2024_wasp69b(DATA, miri_offset_parameter=None)
    if not CACHE.exists():
        raise FileNotFoundError(f"missing observation-bin opacity cache: {CACHE}")
    pressure = PressureGrid.from_log_centers(
        100.0, 1.0e-6, n_layers=80, unit="bar", name="WASP-69b emission pressure grid"
    )
    cia = _cia_tables()
    parameters = {
        "metallicity": 0.5,
        "CtoO": 0.55,
        "kappa_IR": 0.01,
        "gamma1": 0.4,
        "gamma2": 0.4,
        "T_irr": 1400.0,
        "alpha": 0.5,
    }
    likelihood = MultiDatasetGaussianLikelihood(include_normalization=True)

    setup_start = perf_counter()
    lower = min(
        float(d.observation.wavelength_bin_edges.min()) for d in observations.datasets
    )
    upper = max(
        float(d.observation.wavelength_bin_edges.max()) for d in observations.datasets
    )
    native_tables = _canonicalize_g(
        {
            species: _native_table(
                next(PRT_DATA.rglob(PATTERNS[species])), species, lower, upper
            )
            for species in SPECIES
        }
    )
    native_provider = CorrelatedKOpacityProvider(
        native_tables,
        name="WASP-69b-native-R1000",
        interpolation="log_pressure_temperature_log_k_clip",
    )
    native_grid = SpectralGrid(
        values=native_tables[SPECIES[0]].wavelength_micron,
        unit="micron",
        name="petitRADTRANS native R1000 WASP-69b slab",
        role="model",
    )
    native_model = build_parameterized_clear_sky_emission_model(
        _config(native_provider, pressure, cia), spectral_grid=native_grid
    )
    one_native = MultiDatasetForwardModel(
        native_model=native_model, observations=observations
    )
    native_setup_seconds = perf_counter() - setup_start

    setup_start = perf_counter()
    mode_models = {}
    for dataset in observations.datasets:
        tables = _canonicalize_g(
            {species: _load_table(dataset.name, species) for species in SPECIES}
        )
        provider = CorrelatedKOpacityProvider(
            tables,
            name=f"WASP-69b-{dataset.name}-observation-bins",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        mode_models[dataset.name] = build_parameterized_clear_sky_emission_model(
            _config(provider, pressure, cia),
            spectral_grid=dataset.observation.spectral_grid,
        )
    per_mode = NamedModels(mode_models)
    per_mode_setup_seconds = perf_counter() - setup_start

    def native_loglike() -> float:
        return likelihood.loglike(one_native(parameters), observations, parameters)

    def per_mode_loglike() -> float:
        return likelihood.loglike(per_mode(parameters), observations, parameters)

    for _ in range(warmups):
        native_loglike()
        per_mode_loglike()
    profiler = None if profile is None else cProfile.Profile()
    if profiler is not None:
        profiler.enable()
    native_times, native_ll = _time(native_loglike, repeats)
    mode_times, mode_ll = _time(per_mode_loglike, repeats)
    if profiler is not None:
        profiler.disable()
        profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile)
    native_median = float(np.median(native_times))
    mode_median = float(np.median(mode_times))
    return {
        "target": "WASP-69b",
        "modes": list(observations.names),
        "n_observation_points": observations.n_points,
        "n_native_wavelengths": int(native_grid.values.size),
        "n_layers": pressure.n_layers,
        "n_species": len(SPECIES),
        "repeats": repeats,
        "warmups": warmups,
        "one_native_then_bin": {
            "setup_seconds": native_setup_seconds,
            "call_seconds": native_times,
            "median_seconds": native_median,
            "log_likelihood": native_ll,
        },
        "bin_opacity_then_model_per_mode": {
            "setup_seconds": per_mode_setup_seconds,
            "call_seconds": mode_times,
            "median_seconds": mode_median,
            "log_likelihood": mode_ll,
        },
        "median_speed_ratio_native_over_per_mode": native_median / mode_median,
        "note": "Log likelihoods need not match: correlated-k recompression before RT is not equivalent to integrating a native-grid spectrum after RT.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        help="write a cProfile capture of warmed measured calls",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups must be non-negative")
    result = run(args.repeats, args.warmups, profile=args.profile)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
