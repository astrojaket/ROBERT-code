"""Benchmark stratified ExoMol sampling on the published WASP-69b bins."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
from time import perf_counter

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from robert_exoplanets import (
    CorrelatedKOpacityProvider,
    MultiDatasetEmissionForwardModel,
    OpacitySamplingProvider,
    PressureGrid,
    StratifiedSamplingObservationResponse,
    TopHatObservationResponse,
    build_parameterized_clear_sky_emission_model,
    load_schlawin2024_wasp69b,
)

from benchmark_wasp69b_multi_instrument import _canonicalize_g, _config
from retrieve_wasp69b_nircam_clear import (
    DATA,
    SPECIES,
    _cia_tables,
    _load_table,
)

ROOT = Path(__file__).resolve().parents[1]
XSEC = ROOT / "opacity_data" / "exomol_xsec"
OUTPUT = ROOT / "examples" / "outputs" / "opacity_sampling"
RETAINED_MODES = ("f322w2", "f444w", "lrs")
SAMPLE_COUNTS = (2, 4, 8, 16, 24)
PARAMETERS = {
    "metallicity": 0.5,
    "CtoO": 0.55,
    "kappa_IR": 0.01,
    "gamma1": 0.4,
    "gamma2": 0.4,
    "T_irr": 1400.0,
    "alpha": 0.5,
}


def main() -> dict[str, object]:
    published = load_schlawin2024_wasp69b(DATA, miri_offset_parameter=None)
    datasets = tuple(
        dataset for dataset in published.datasets if dataset.name in RETAINED_MODES
    )
    pressure = PressureGrid.from_log_centers(
        100.0,
        1.0e-6,
        n_layers=80,
        unit="bar",
        name="WASP-69b emission pressure grid",
    )
    cia = _cia_tables()
    correlated = _build_correlated_k(datasets, pressure, cia)
    correlated(PARAMETERS)  # compile and warm spectrum-only RT
    correlated_seconds = _median_time(lambda: correlated(PARAMETERS))
    correlated_fixed_atmosphere_seconds = _fixed_atmosphere_time(correlated)
    correlated_spectra = correlated(PARAMETERS)

    sampling_provider = OpacitySamplingProvider.from_exomol_paths(
        {species: XSEC / f"{species}.h5" for species in SPECIES},
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=False,
    )
    source_grid = sampling_provider.native_spectral_grid()
    lower = min(float(d.observation.wavelength_bin_edges[0]) for d in datasets)
    upper = max(float(d.observation.wavelength_bin_edges[-1]) for d in datasets)
    reference_grid = sampling_provider.native_spectral_grid(
        wavelength_bounds_micron=(lower - 0.01, upper + 0.01)
    )
    reference_model = build_parameterized_clear_sky_emission_model(
        _config(sampling_provider, pressure, cia),
        spectral_grid=reference_grid,
    )
    reference_model(PARAMETERS)
    reference_seconds = _median_time(
        lambda model=reference_model: model(PARAMETERS), repeats=3
    )
    reference_native = reference_model(PARAMETERS)
    reference_spectra = {
        dataset.name: TopHatObservationResponse()
        .prepare(dataset.observation)
        .observe(reference_native)
        for dataset in datasets
    }
    correlated_accuracy_vs_reference = _accuracy(
        correlated_spectra, reference_spectra, datasets
    )
    del reference_model, reference_native
    gc.collect()
    sampling_results = {}
    for samples_per_bin in SAMPLE_COUNTS:
        started = perf_counter()
        sampled, n_samples = _build_sampled(
            datasets,
            pressure,
            cia,
            sampling_provider,
            source_grid,
            samples_per_bin,
        )
        setup_seconds = perf_counter() - started
        sampled(PARAMETERS)  # compile/warm this shape
        call_seconds = _median_time(lambda model=sampled: model(PARAMETERS))
        fixed_atmosphere_seconds = _fixed_atmosphere_time(sampled)
        spectra = sampled(PARAMETERS)
        sampling_results[str(samples_per_bin)] = {
            "samples_per_bin": samples_per_bin,
            "total_opacity_samples": n_samples,
            "setup_seconds": setup_seconds,
            "median_call_seconds": call_seconds,
            "speedup_over_correlated_k": correlated_seconds / call_seconds,
            "fixed_atmosphere_seconds": fixed_atmosphere_seconds,
            "fixed_atmosphere_speedup_over_correlated_k": (
                correlated_fixed_atmosphere_seconds / fixed_atmosphere_seconds
            ),
            "accuracy_vs_correlated_k": _accuracy(
                spectra, correlated_spectra, datasets
            ),
            "accuracy_vs_full_exomol_sampling": _accuracy(
                spectra, reference_spectra, datasets
            ),
        }
        del sampled, spectra
        gc.collect()

    report = {
        "schema_version": 1,
        "target": "WASP-69b Schlawin et al. 2024 native bins",
        "modes": list(RETAINED_MODES),
        "n_observation_bins": int(sum(d.observation.n_points for d in datasets)),
        "n_layers": pressure.n_layers,
        "n_species": len(SPECIES),
        "diagnostics": "disabled for every timed retrieval call",
        "correlated_k": {
            "median_call_seconds": correlated_seconds,
            "g_ordinates": 16,
            "gas_combination": "random_overlap",
            "fixed_atmosphere_seconds": correlated_fixed_atmosphere_seconds,
            "accuracy_vs_full_exomol_sampling": correlated_accuracy_vs_reference,
        },
        "full_exomol_sampling_reference": {
            "native_opacity_samples": reference_grid.size,
            "median_call_seconds": reference_seconds,
        },
        "opacity_sampling": sampling_results,
        "comparison_note": (
            "correlated-k CO uses HITEMP while ExoMol sampling uses Li2015; "
            "the comparison therefore includes this line-list difference"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "wasp69b_stratified_sampling_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def _build_correlated_k(datasets, pressure, cia):
    models = {}
    shared_builder = None
    for dataset in datasets:
        tables = _canonicalize_g(
            {species: _load_table(dataset.name, species) for species in SPECIES}
        )
        provider = CorrelatedKOpacityProvider(
            tables,
            name=f"WASP-69b-{dataset.name}-observation-bins",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        model = build_parameterized_clear_sky_emission_model(
            _config(provider, pressure, cia),
            spectral_grid=dataset.observation.spectral_grid,
        )
        if shared_builder is None:
            shared_builder = model.atmosphere_builder
        else:
            model = replace(model, atmosphere_builder=shared_builder)
        models[dataset.name] = model
    return MultiDatasetEmissionForwardModel(models)


def _build_sampled(
    datasets,
    pressure,
    cia,
    provider,
    source_grid,
    samples_per_bin,
):
    models = {}
    responses = {}
    shared_builder = None
    for dataset in datasets:
        response = StratifiedSamplingObservationResponse(
            samples_per_bin=samples_per_bin
        ).prepare(dataset.observation, source_grid)
        model = build_parameterized_clear_sky_emission_model(
            _config(provider, pressure, cia),
            spectral_grid=response.spectral_grid,
        )
        if shared_builder is None:
            shared_builder = model.atmosphere_builder
        else:
            model = replace(model, atmosphere_builder=shared_builder)
        models[dataset.name] = model
        responses[dataset.name] = response
    return (
        MultiDatasetEmissionForwardModel(models, responses=responses),
        sum(response.spectral_grid.size for response in responses.values()),
    )


def _median_time(function, repeats: int = 7) -> float:
    durations = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        durations.append(perf_counter() - started)
    return float(np.median(durations))


def _fixed_atmosphere_time(model: MultiDatasetEmissionForwardModel) -> float:
    reference = next(iter(model.models.values()))
    values = reference.validated_parameters(PARAMETERS)
    atmosphere = model.atmosphere_builder.build(values)

    def evaluate():
        spectra = {
            name: item.evaluate_atmosphere(atmosphere, values)
            for name, item in model.models.items()
        }
        if model.responses:
            spectra = {
                name: model.responses[name].observe(spectrum)
                for name, spectrum in spectra.items()
            }
        return spectra

    evaluate()
    return _median_time(evaluate)


def _accuracy(sampled, correlated, datasets):
    sampled_values = np.concatenate([sampled[d.name].values for d in datasets])
    correlated_values = np.concatenate([correlated[d.name].values for d in datasets])
    uncertainty = np.concatenate([d.observation.uncertainty for d in datasets])
    difference = sampled_values - correlated_values
    relative = difference / correlated_values
    return {
        "rms_relative": float(np.sqrt(np.mean(relative**2))),
        "p95_abs_relative": float(np.quantile(np.abs(relative), 0.95)),
        "max_abs_relative": float(np.max(np.abs(relative))),
        "rms_eclipse_depth_ppm": float(1.0e6 * np.sqrt(np.mean(difference**2))),
        "max_abs_eclipse_depth_ppm": float(1.0e6 * np.max(np.abs(difference))),
        "rms_difference_in_observational_sigma": float(
            np.sqrt(np.mean((difference / uncertainty) ** 2))
        ),
    }


if __name__ == "__main__":
    main()
