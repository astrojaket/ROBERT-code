"""Benchmark six-species opacity sampling against correlated-k and pRT3."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    EvaluatedCorrelatedKOpacity,
    OpacitySamplingProvider,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    solve_clear_sky_emission,
)

ROOT = Path(__file__).resolve().parents[1]
XSEC_DATA = ROOT / "opacity_data" / "exomol_xsec"
K_DATA = ROOT / "opacity_data" / "petitRADTRANS" / "input_data"
OUTPUT = ROOT / "examples" / "outputs" / "opacity_sampling"
SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
K_PATTERNS = {
    "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
    "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
    "CO2": "*UCL-4000*.ktable.petitRADTRANS.h5",
    "CH4": "*YT34to10*.ktable.petitRADTRANS.h5",
    "NH3": "*CoYuTe*.ktable.petitRADTRANS.h5",
    "HCN": "*Harris*.ktable.petitRADTRANS.h5",
}
VOLUME_FRACTIONS = {
    "H2": 0.8491908757254121,
    "He": 0.15039417437416785,
    "H2O": 0.00012901275259325404,
    "CO": 0.00024893172764876213,
    "CO2": 1.5843403321134283e-05,
    "CH4": 1.4487808363169974e-05,
    "NH3": 4.09418067364625e-06,
    "HCN": 2.5800278200851027e-06,
}
MEAN_MOLAR_MASS = 2.3242008615381975


def main() -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atmosphere = _atmosphere(40)
    geometry = gauss_legendre_disk_geometry(8)

    start = perf_counter()
    sampling_provider = OpacitySamplingProvider.from_exomol_paths(
        {species: XSEC_DATA / f"{species}.h5" for species in SPECIES},
        checksum=False,
    )
    sampling_grid = sampling_provider.native_spectral_grid(
        sampling=15, wavelength_bounds_micron=(1.0, 12.0)
    )
    sampling_prepared = sampling_provider.prepare(
        sampling_grid, atmosphere.pressure_grid, SPECIES
    )
    sampling_opacity = sampling_provider.evaluate(atmosphere, sampling_prepared)
    sampling_prepare_s = perf_counter() - start

    sampling_result, sampling_timings = _time_solver(
        atmosphere, sampling_opacity, "random_overlap", geometry
    )
    sampling_flux = np.pi * np.asarray(sampling_result.radiance.values)

    start = perf_counter()
    sampling_grid_3 = sampling_provider.native_spectral_grid(
        sampling=3, wavelength_bounds_micron=(1.0, 12.0)
    )
    sampling_prepared_3 = sampling_provider.prepare(
        sampling_grid_3, atmosphere.pressure_grid, SPECIES
    )
    sampling_opacity_3 = sampling_provider.evaluate(atmosphere, sampling_prepared_3)
    sampling_prepare_3_s = perf_counter() - start
    sampling_result_3, sampling_timings_3 = _time_solver(
        atmosphere, sampling_opacity_3, "random_overlap", geometry
    )
    sampling_flux_3 = np.pi * np.asarray(sampling_result_3.radiance.values)

    start = perf_counter()
    k_grid, k_opacity = _evaluated_correlated_k(atmosphere)
    k_prepare_s = perf_counter() - start
    k_result, k_timings = _time_solver(
        atmosphere, k_opacity, "random_overlap", geometry
    )
    k_flux = np.pi * np.asarray(k_result.radiance.values)

    p_rt = {
        name: _load_prt(OUTPUT / filename)
        for name, filename in {
            "sampling_R15000": "petitradtrans3_sampling1.npz",
            "sampling_R1000": "petitradtrans3_sampling15.npz",
            "sampling_R3000": "petitradtrans3_sampling5.npz",
            "sampling_R5000": "petitradtrans3_sampling3.npz",
            "correlated_k_R1000": "petitradtrans3_correlated_k.npz",
        }.items()
    }
    centers, p_rt_reference = _bin_r100(*p_rt["sampling_R15000"][:2])
    _, p_rt_sampling = _bin_r100(*p_rt["sampling_R1000"][:2])
    _, p_rt_sampling_3000 = _bin_r100(*p_rt["sampling_R3000"][:2])
    _, p_rt_sampling_5000 = _bin_r100(*p_rt["sampling_R5000"][:2])
    _, p_rt_k = _bin_r100(*p_rt["correlated_k_R1000"][:2])
    _, robert_sampling = _bin_r100(sampling_grid.values, sampling_flux)
    _, robert_sampling_5000 = _bin_r100(
        sampling_grid_3.values, sampling_flux_3
    )
    _, robert_k = _bin_r100(k_grid.values, k_flux)

    report = {
        "schema_version": 1,
        "comparison": "six_species_opacity_sampling_vs_correlated_k_vs_pRT3",
        "domain": {
            "wavelength_micron": [1.0, 12.0],
            "layers": 40,
            "species": list(SPECIES),
            "output_accuracy_resolution": 100,
        },
        "methods": {
            "opacity_sampling": {
                "source": "ExoMolOP TauREx cross sections R=15000 sampled every 15th point",
                "native_samples": sampling_grid.size,
                "gas_combination": "direct optical-depth sum",
            },
            "correlated_k": {
                "source": "pRT/ExoMolOP R=1000 k-tables; CO uses HITEMP",
                "native_samples": k_grid.size,
                "g_ordinates": int(k_opacity.prepared.g_weights.size),
                "gas_combination": "random overlap with resort/rebin",
            },
        },
        "timings_seconds": {
            "robert": {
                "opacity_sampling_prepare_load_evaluate": sampling_prepare_s,
                "opacity_sampling_R5000_prepare_load_evaluate": sampling_prepare_3_s,
                "correlated_k_load_evaluate": k_prepare_s,
                "opacity_sampling": sampling_timings,
                "opacity_sampling_R5000": sampling_timings_3,
                "correlated_k": k_timings,
                "steady_speedup_sampling_over_correlated_k": (
                    k_timings["steady_median"] / sampling_timings["steady_median"]
                ),
                "steady_speedup_sampling_R5000_over_correlated_k": (
                    k_timings["steady_median"] / sampling_timings_3["steady_median"]
                ),
            },
            "petitradtrans3": {
                name: values[2]["timings"] for name, values in p_rt.items()
            },
        },
        "accuracy_at_R100": {
            "reference": "pRT3 R=15000 ExoMolOP opacity sampling",
            "petitradtrans3_sampling_R1000": _metrics(p_rt_sampling, p_rt_reference),
            "petitradtrans3_sampling_R3000": _metrics(
                p_rt_sampling_3000, p_rt_reference
            ),
            "petitradtrans3_sampling_R5000": _metrics(
                p_rt_sampling_5000, p_rt_reference
            ),
            "petitradtrans3_correlated_k_R1000": _metrics(p_rt_k, p_rt_reference),
            "robert_sampling_R1000_vs_pRT3_sampling_R15000": _metrics(
                robert_sampling, p_rt_reference
            ),
            "robert_sampling_R5000_vs_pRT3_sampling_R15000": _metrics(
                robert_sampling_5000, p_rt_reference
            ),
            "robert_correlated_k_R1000_vs_pRT3_sampling_R15000": _metrics(
                robert_k, p_rt_reference
            ),
            "robert_sampling_R1000_vs_pRT3_sampling_R1000": _metrics(
                robert_sampling, p_rt_sampling
            ),
        },
        "caveats": [
            "The correlated-k CO table is HITEMP whereas the requested ExoMol sampling table is Li2015.",
            "ROBERT and pRT3 use different vertical discretizations and thermal solvers; the pRT3-to-pRT3 metrics isolate opacity treatment most cleanly.",
            "R=15000 opacity sampling is a high-resolution numerical reference, not a claim of line-by-line truth.",
        ],
    }
    (OUTPUT / "opacity_sampling_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT / "opacity_sampling_benchmark_spectra.npz",
        wavelength_R100_micron=centers,
        petitradtrans_sampling_R15000_R100=p_rt_reference,
        petitradtrans_sampling_R1000_R100=p_rt_sampling,
        petitradtrans_sampling_R3000_R100=p_rt_sampling_3000,
        petitradtrans_sampling_R5000_R100=p_rt_sampling_5000,
        petitradtrans_correlated_k_R1000_R100=p_rt_k,
        robert_sampling_R1000_R100=robert_sampling,
        robert_sampling_R5000_R100=robert_sampling_5000,
        robert_correlated_k_R1000_R100=robert_k,
    )
    print(json.dumps(report, indent=2))
    return report


def _atmosphere(n_layers: int) -> AtmosphereState:
    pressure = np.geomspace(1.0e-5, 100.0, n_layers)
    grid = PressureGrid.from_log_centers(
        pressure[0], pressure[-1], n_layers, unit="bar"
    )
    temperature = 900.0 + 900.0 * (np.log10(pressure) + 5.0) / 7.0
    return AtmosphereState(
        pressure_grid=grid,
        temperature=temperature,
        composition={
            species: np.full(n_layers, abundance)
            for species, abundance in VOLUME_FRACTIONS.items()
        },
        mean_molecular_weight=np.full(n_layers, MEAN_MOLAR_MASS),
    )


def _evaluated_correlated_k(atmosphere):
    evaluated = None
    wavelength = None
    g_samples = None
    g_weights = None
    for species_index, species in enumerate(SPECIES):
        path = next(K_DATA.rglob(K_PATTERNS[species]))
        table = CorrelatedKTable.from_petitradtrans_hdf(path, species=species)
        if wavelength is None:
            selected = (table.wavelength_micron >= 1.0) & (table.wavelength_micron <= 12.0)
            wavelength = np.sort(np.asarray(table.wavelength_micron[selected]))
            g_samples = table.g_samples
            g_weights = table.g_weights
            evaluated = np.empty(
                (len(SPECIES), atmosphere.n_layers, wavelength.size, g_weights.size)
            )
        grid = SpectralGrid.from_array(wavelength, unit="micron", role="opacity")
        provider = CorrelatedKOpacityProvider(
            {species: table}, interpolation="log_pressure_temperature_log_k"
        )
        prepared = provider.prepare(grid, atmosphere.pressure_grid, (species,))
        evaluated[species_index] = provider.evaluate(atmosphere, prepared).kcoeff[0]
        del provider, prepared, table
        gc.collect()
    prepared = PreparedCorrelatedKOpacity(
        provider_name="six-species-correlated-k-benchmark",
        spectral_grid=grid,
        pressure_grid=atmosphere.pressure_grid,
        species=SPECIES,
        g_samples=g_samples,
        g_weights=g_weights,
        cache_key="six-species-correlated-k-benchmark",
        metadata={"opacity_mode": "correlated_k"},
    )
    return grid, EvaluatedCorrelatedKOpacity(prepared=prepared, kcoeff=evaluated)


def _time_solver(atmosphere, opacity, combination, geometry):
    def solve():
        gas_tau = assemble_gas_optical_depth(
            atmosphere, opacity, gravity_m_s2=15.0, gas_combination=combination
        )
        return solve_clear_sky_emission(
            gas_tau,
            geometry=geometry,
            bottom_boundary="blackbody",
            thermal_integration_backend="auto",
        )

    start = perf_counter()
    result = solve()
    first = perf_counter() - start
    durations = []
    for _ in range(5):
        start = perf_counter()
        solve()
        durations.append(perf_counter() - start)
    return result, {"first": first, "steady_median": float(np.median(durations))}


def _load_prt(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        wavelength = np.asarray(archive["wavelength_cm"]) * 1.0e4
        flux = np.asarray(archive["flux_cgs_per_cm"]) * 1.0e-1
    return wavelength, flux, metadata


def _bin_r100(wavelength, flux):
    n_bins = int(np.floor(100.0 * np.log(12.0)))
    edges = np.geomspace(1.0, 12.0, n_bins + 1)
    totals, _ = np.histogram(wavelength, bins=edges, weights=flux)
    counts, _ = np.histogram(wavelength, bins=edges)
    if np.any(counts == 0):
        raise RuntimeError("R=100 benchmark bin has no native samples")
    return np.sqrt(edges[:-1] * edges[1:]), totals / counts


def _metrics(candidate, reference):
    relative = np.asarray(candidate) / np.asarray(reference) - 1.0
    return {
        "median_relative": float(np.median(relative)),
        "rms_relative": float(np.sqrt(np.mean(relative**2))),
        "p95_abs_relative": float(np.quantile(np.abs(relative), 0.95)),
        "max_abs_relative": float(np.max(np.abs(relative))),
    }


if __name__ == "__main__":
    main()
