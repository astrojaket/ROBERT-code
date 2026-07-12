"""Benchmark ROBERT-native pRT HDF interpolation on cell-centred grids."""

from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    CiaTable,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    cia_optical_depth,
    gauss_legendre_disk_geometry,
    solve_clear_sky_emission,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "opacity_data" / "petitRADTRANS" / "input_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "petitradtrans3_stable"
SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
TABLE_PATTERNS = {
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
GRAVITY_M_S2 = 15.0
RESOLUTIONS = (40, 80, 160)


def main() -> dict[str, object]:
    raw_references = _load_references()
    h2o_path = next(DATA.rglob(TABLE_PATTERNS["H2O"]))
    h2o_table = CorrelatedKTable.from_petitradtrans_hdf(h2o_path, species="H2O")
    wavelength_mask = (h2o_table.wavelength_micron >= 0.3) & (
        h2o_table.wavelength_micron <= 12.0
    )
    wavelength = np.sort(
        np.asarray(h2o_table.wavelength_micron[wavelength_mask], dtype=float)
    )
    references = {
        n: (
            wavelength,
            np.interp(wavelength, raw_references[n][0], raw_references[n][1]),
        )
        for n in RESOLUTIONS
    }
    spectral_grid = SpectralGrid.from_array(
        wavelength,
        unit="micron",
        role="opacity",
        name="pRT_R1000_0.3-12um",
    )
    atmospheres = {n: _atmosphere(n) for n in RESOLUTIONS}
    evaluated_arrays: dict[int, np.ndarray] = {}
    g_samples = None
    g_weights = None
    opacity_start = perf_counter()
    for species_index, species in enumerate(SPECIES):
        if species == "H2O":
            table = h2o_table
        else:
            path = next(DATA.rglob(TABLE_PATTERNS[species]))
            table = CorrelatedKTable.from_petitradtrans_hdf(path, species=species)
        provider = CorrelatedKOpacityProvider(
            {species: table},
            name=f"native-pRT-HDF-{species}",
            interpolation="log_pressure_temperature_log_k",
        )
        if g_samples is None:
            g_samples = table.g_samples
            g_weights = table.g_weights
            for n in RESOLUTIONS:
                evaluated_arrays[n] = np.empty(
                    (len(SPECIES), n, wavelength.size, g_weights.size),
                    dtype=float,
                )
        for n in RESOLUTIONS:
            prepared = provider.prepare(
                spectral_grid,
                atmospheres[n].pressure_grid,
                species=(species,),
            )
            evaluated_arrays[n][species_index] = provider.evaluate(
                atmospheres[n], prepared
            ).kcoeff[0]
        del provider, table
        gc.collect()
    opacity_load_evaluate_s = perf_counter() - opacity_start
    if g_samples is None or g_weights is None:
        raise RuntimeError("no opacity tables were loaded")

    h2_h2_path = next(DATA.rglob("*H2--H2*.ciatable.petitRADTRANS.h5"))
    h2_he_path = next(DATA.rglob("*H2--He*.ciatable.petitRADTRANS.h5"))
    h2_h2 = CiaTable.from_petitradtrans_hdf(h2_h2_path, collision_pair="H2-H2")
    h2_he = CiaTable.from_petitradtrans_hdf(h2_he_path, collision_pair="H2-He")
    geometry = gauss_legendre_disk_geometry(n_mu=8)
    fluxes = {}
    timings = {}
    for n in RESOLUTIONS:
        prepared = PreparedCorrelatedKOpacity(
            provider_name="native-pRT-HDF-six-species",
            spectral_grid=spectral_grid,
            pressure_grid=atmospheres[n].pressure_grid,
            species=SPECIES,
            g_samples=g_samples,
            g_weights=g_weights,
            cache_key=f"native-pRT-HDF-six-species-{n}",
            metadata={"interpolation": "log_pressure_temperature_log_k"},
        )
        opacity = EvaluatedCorrelatedKOpacity(
            prepared=prepared,
            kcoeff=evaluated_arrays[n],
            unit="cm^2/molecule",
            metadata={"source": "ROBERT-native petitRADTRANS HDF5 loading"},
        )

        def solve(current_opacity=opacity, current_n=n):
            gas_tau = assemble_gas_optical_depth(
                atmospheres[current_n],
                current_opacity,
                gravity_m_s2=GRAVITY_M_S2,
                gas_combination="random_overlap",
            )
            cia = [
                cia_optical_depth(
                    gas_tau,
                    table,
                    coefficient_interpolation="log",
                    temperature_extrapolation="clip",
                    spectral_extrapolation="zero",
                )
                for table in (h2_h2, h2_he)
            ]
            return solve_clear_sky_emission(
                gas_tau,
                geometry=geometry,
                bottom_boundary="blackbody",
                additional_optical_depths=cia,
                thermal_integration_backend="auto",
            )

        start = perf_counter()
        result = solve()
        first_s = perf_counter() - start
        durations = []
        for _ in range(3):
            start = perf_counter()
            solve()
            durations.append(perf_counter() - start)
        fluxes[n] = np.pi * np.asarray(result.radiance.values)
        timings[n] = {
            "first_s": first_s,
            "steady_median_s": float(np.median(durations)),
        }
        del opacity, evaluated_arrays[n]
        gc.collect()

    paired_metrics = {
        str(n): _relative_metrics(fluxes[n] / references[n][1] - 1.0)
        for n in RESOLUTIONS
    }
    robert_convergence = {
        f"{coarse}_to_{fine}": _relative_metrics(fluxes[coarse] / fluxes[fine] - 1.0)
        for coarse, fine in ((40, 80), (80, 160))
    }
    p_rt_convergence = {
        f"{coarse + 1}_to_{fine + 1}_nodes": _relative_metrics(
            references[coarse][1] / references[fine][1] - 1.0
        )
        for coarse, fine in ((40, 80), (80, 160))
    }
    report = {
        "schema_version": 1,
        "comparison": "ROBERT_native_HDF_cell_centres_vs_pRT3_pressure_nodes",
        "wavelength_micron": [float(wavelength[0]), float(wavelength[-1])],
        "cell_resolutions": list(RESOLUTIONS),
        "petitradtrans_node_resolutions": [n + 1 for n in RESOLUTIONS],
        "line_species": list(SPECIES),
        "cia_pairs": ["H2-H2", "H2-He"],
        "interpolation": {
            "line": "bilinear log(k) in log(pressure) and temperature",
            "cia_temperature": "linear in log(alpha)",
            "cia_wavenumber": "linear in alpha",
        },
        "paired_metrics": paired_metrics,
        "robert_self_convergence": robert_convergence,
        "petitradtrans3_self_convergence": p_rt_convergence,
        "timings": {
            "native_HDF_load_and_evaluate_all_resolutions_s": opacity_load_evaluate_s,
            "robert_by_resolution": timings,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "native_hdf_emission_convergence.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT_DIR / "native_hdf_emission_convergence_spectra.npz",
        wavelength_micron=wavelength,
        **{f"robert_cells_{n}": fluxes[n] for n in RESOLUTIONS},
        **{f"petitradtrans_nodes_{n + 1}": references[n][1] for n in RESOLUTIONS},
    )
    _plot(wavelength, fluxes, references, report)
    print(json.dumps(report, indent=2))
    return report


def _atmosphere(n_layers: int) -> AtmosphereState:
    edges = np.geomspace(1.0e-5, 100.0, n_layers + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    temperature = _temperature(centers)
    return AtmosphereState(
        pressure_grid=PressureGrid(
            edges=edges,
            centers=centers,
            unit="bar",
            name=f"cell_centred_{n_layers}",
        ),
        temperature=temperature,
        temperature_edges=_temperature(edges),
        composition={
            species: np.full(n_layers, abundance)
            for species, abundance in VOLUME_FRACTIONS.items()
        },
        mean_molecular_weight=np.full(n_layers, MEAN_MOLAR_MASS),
    )


def _temperature(pressure_bar: np.ndarray) -> np.ndarray:
    return 900.0 + 900.0 * (np.log10(pressure_bar) + 5.0) / 7.0


def _load_references() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for n in RESOLUTIONS:
        path = OUTPUT_DIR / f"native_convergence_prt_nodes_{n + 1}.npz"
        with np.load(path, allow_pickle=False) as archive:
            wavelength = np.asarray(archive["wavelength_cm"], dtype=float) * 1.0e4
            flux = np.asarray(archive["flux_cgs_per_cm"], dtype=float) * 1.0e-1
        output[n] = (wavelength, flux)
    reference_wavelength = output[RESOLUTIONS[0]][0]
    if any(
        not np.allclose(reference_wavelength, output[n][0], rtol=1.0e-12, atol=0.0)
        for n in RESOLUTIONS[1:]
    ):
        raise RuntimeError("pRT wavelength grid changed with pressure resolution")
    return output


def _relative_metrics(values: np.ndarray) -> dict[str, float]:
    return {
        "median_relative_difference": float(np.median(values)),
        "rms_relative_difference": float(np.sqrt(np.mean(values**2))),
        "max_abs_relative_difference": float(np.max(np.abs(values))),
    }


def _plot(
    wavelength: np.ndarray,
    fluxes: dict[int, np.ndarray],
    references: dict[int, tuple[np.ndarray, np.ndarray]],
    report: dict[str, object],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    spectrum, residual, convergence, timing = axes.flat
    spectrum.loglog(wavelength, references[80][1], color="#222222", lw=1.2, label="pRT3, 81 nodes")
    spectrum.loglog(wavelength, fluxes[80], color="#e45756", lw=0.9, ls="--", label="ROBERT, 80 cells")
    spectrum.set(ylabel=r"Planet flux $F_\lambda$ [W m$^{-2}$ m$^{-1}$]", title="Native HDF emission")
    spectrum.legend(frameon=False)
    colors = {40: "#9ecae9", 80: "#4c78a8", 160: "#17365d"}
    for n in RESOLUTIONS:
        residual.semilogx(
            wavelength,
            (fluxes[n] / references[n][1] - 1.0) * 100.0,
            color=colors[n],
            lw=0.8,
            label=f"{n} cells / {n + 1} nodes",
        )
    residual.axhline(0.0, color="#222222", lw=0.7)
    residual.set(ylabel="ROBERT - pRT3 [%]", title="Matched-boundary residual")
    residual.legend(frameon=False)
    pairs = ("40_to_80", "80_to_160")
    x = np.arange(2)
    robert_values = [report["robert_self_convergence"][key]["rms_relative_difference"] * 100 for key in pairs]
    p_rt_keys = ("41_to_81_nodes", "81_to_161_nodes")
    p_rt_values = [report["petitradtrans3_self_convergence"][key]["rms_relative_difference"] * 100 for key in p_rt_keys]
    convergence.bar(x - 0.18, robert_values, 0.36, color="#e45756", label="ROBERT")
    convergence.bar(x + 0.18, p_rt_values, 0.36, color="#4c78a8", label="pRT3")
    convergence.set_xticks(x, ("40/80", "80/160"))
    convergence.set(ylabel="RMS spectral change [%]", title="Pressure-grid self-convergence")
    convergence.legend(frameon=False)
    timing.bar(
        np.arange(3),
        [report["timings"]["robert_by_resolution"][n]["steady_median_s"] for n in RESOLUTIONS],
        color=[colors[n] for n in RESOLUTIONS],
    )
    timing.set_xticks(np.arange(3), [str(n) for n in RESOLUTIONS])
    timing.set(ylabel="Steady wall time [s]", title="ROBERT native-HDF evaluation + RT")
    for axis in (spectrum, residual):
        axis.set_xlabel("Wavelength [micron]")
        axis.grid(alpha=0.25)
    convergence.grid(axis="y", alpha=0.25)
    timing.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "native_hdf_emission_convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
