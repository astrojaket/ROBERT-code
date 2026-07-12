"""Benchmark ROBERT multispecies + Rayleigh transmission against stable pRT3."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    LayerOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    hydrostatic_path_geometry,
    rayleigh_scattering_optical_depth,
    solve_absorption_transmission,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "petitradtrans3_stable"
REFERENCE = OUTPUT_DIR / "multispecies_transmission_reference.npz"
SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
STAR_RADIUS_M = 1.2 * 6.957e8
PLANET_RADIUS_M = 1.0e8
GRAVITY_M_S2 = 15.0
REFERENCE_PRESSURE_BAR = 0.01
ATOMIC_MASS_G = 1.66053906660e-24


def main() -> dict[str, object]:
    with np.load(REFERENCE, allow_pickle=False) as archive:
        reference = json.loads(str(archive["metadata_json"]))
        pressure = np.asarray(archive["pressure_bar"], dtype=float)
        temperature = np.asarray(archive["temperature_K"], dtype=float)
        wavelength = np.asarray(archive["wavelength_cm"], dtype=float) * 1.0e4
        p_rt_rayleigh = (
            np.asarray(archive["transit_radius_rayleigh_cm"], dtype=float)
            * 1.0e-2
            / STAR_RADIUS_M
        ) ** 2
        p_rt_clear = (
            np.asarray(archive["transit_radius_no_rayleigh_cm"], dtype=float)
            * 1.0e-2
            / STAR_RADIUS_M
        ) ** 2
        absorption_cm2_g = np.asarray(archive["absorption_opacities"], dtype=float)
        scattering_cm2_g = np.asarray(
            archive["continuum_scattering_opacities"], dtype=float
        )

    volume_fractions = {
        key: float(value) for key, value in reference["volume_fractions"].items()
    }
    mean_molar_mass = float(reference["mean_molar_mass_amu"])
    pressure_grid = PressureGrid.from_log_centers(
        pressure[0], pressure[-1], pressure.size, unit="bar", name="pRT3_pressure_nodes"
    )
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="pRT3_R1000_0.3-12um"
    )
    composition = {
        species: np.full(pressure.size, volume_fractions[species])
        for species in volume_fractions
    }
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        composition=composition,
        mean_molecular_weight=np.full(pressure.size, mean_molar_mass),
    )
    opacity_path = next(
        (ROOT / "opacity_data" / "petitRADTRANS" / "input_data").rglob(
            "*POKAZATEL*.ktable.petitRADTRANS.h5"
        )
    )
    with h5py.File(opacity_path, "r") as opacity_file:
        g_samples = np.asarray(opacity_file["samples"], dtype=float)
        g_weights = np.asarray(opacity_file["weights"], dtype=float)
    prepared = PreparedCorrelatedKOpacity(
        provider_name="petitradtrans3-multispecies-evaluated-opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=SPECIES,
        g_samples=g_samples,
        g_weights=g_weights,
        cache_key="petitradtrans3-multispecies-rayleigh-0.3-12um",
        metadata={"source": "petitRADTRANS 3.3.3 returned species opacities"},
    )
    opacity_lswg = absorption_cm2_g.transpose(2, 3, 1, 0)
    kcoeff = np.empty_like(opacity_lswg)
    for index, species in enumerate(SPECIES):
        kcoeff[index] = (
            opacity_lswg[index]
            * mean_molar_mass
            * ATOMIC_MASS_G
            / volume_fractions[species]
        )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit="cm^2/molecule",
        metadata={"source": "petitRADTRANS3_species_opacities_after_mass_weighting"},
    )
    path_geometry = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=GRAVITY_M_S2,
        reference_radius_m=PLANET_RADIUS_M,
        reference_pressure=REFERENCE_PRESSURE_BAR,
        reference_pressure_unit="bar",
    )

    def solve_cases():
        gas_tau = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=GRAVITY_M_S2,
            gas_combination="random_overlap",
        )
        strict_rayleigh_tau = (
            scattering_cm2_g.T
            * 0.1
            * gas_tau.layer_pressure_thickness_pa[:, None]
            / GRAVITY_M_S2
        )
        strict_rayleigh = LayerOpticalDepth(
            name="pRT3 H2/He Rayleigh extinction",
            tau=strict_rayleigh_tau,
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            kind="scattering_extinction",
            metadata={"source": "petitRADTRANS3 returned continuum scattering opacity"},
        )
        native_rayleigh = rayleigh_scattering_optical_depth(gas_tau)
        clear = solve_absorption_transmission(
            gas_tau,
            path_geometry,
            star_radius_m=STAR_RADIUS_M,
            impact_quadrature_order=8,
        )
        strict = solve_absorption_transmission(
            gas_tau,
            path_geometry,
            star_radius_m=STAR_RADIUS_M,
            additional_optical_depths=[strict_rayleigh],
            impact_quadrature_order=8,
        )
        native = solve_absorption_transmission(
            gas_tau,
            path_geometry,
            star_radius_m=STAR_RADIUS_M,
            additional_optical_depths=[native_rayleigh],
            impact_quadrature_order=8,
        )
        return clear, strict, native, strict_rayleigh, native_rayleigh

    def solve_native_case():
        gas_tau = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=GRAVITY_M_S2,
            gas_combination="random_overlap",
        )
        return solve_absorption_transmission(
            gas_tau,
            path_geometry,
            star_radius_m=STAR_RADIUS_M,
            additional_optical_depths=[rayleigh_scattering_optical_depth(gas_tau)],
            impact_quadrature_order=8,
        )

    start = perf_counter()
    clear_result, strict_result, native_result, strict_rayleigh, native_rayleigh = solve_cases()
    first_s = perf_counter() - start
    durations = []
    for _ in range(3):
        start = perf_counter()
        solve_cases()
        durations.append(perf_counter() - start)
    steady_s = float(np.median(durations))
    start = perf_counter()
    solve_native_case()
    native_first_s = perf_counter() - start
    native_durations = []
    for _ in range(3):
        start = perf_counter()
        solve_native_case()
        native_durations.append(perf_counter() - start)
    native_steady_s = float(np.median(native_durations))
    robert_clear = np.asarray(clear_result.transit_depth.values)
    robert_strict = np.asarray(strict_result.transit_depth.values)
    robert_native = np.asarray(native_result.transit_depth.values)
    residual_clear_ppm = (robert_clear - p_rt_clear) * 1.0e6
    residual_strict_ppm = (robert_strict - p_rt_rayleigh) * 1.0e6
    residual_native_ppm = (robert_native - p_rt_rayleigh) * 1.0e6
    p_rt_rayleigh_effect_ppm = (p_rt_rayleigh - p_rt_clear) * 1.0e6
    native_rayleigh_effect_ppm = (robert_native - robert_clear) * 1.0e6
    strict_rayleigh_effect_ppm = (robert_strict - robert_clear) * 1.0e6
    report = {
        "schema_version": 1,
        "comparison": "ROBERT_vs_stable_pRT3_multispecies_CIA_Rayleigh_transmission",
        "wavelength_micron": [float(wavelength[0]), float(wavelength[-1])],
        "n_layers": int(pressure.size),
        "n_wavelength": int(wavelength.size),
        "line_species": reference["line_species"],
        "cia_species": reference["cia_species"],
        "rayleigh_species": reference["rayleigh_species"],
        "composition": {
            "petitradtrans_mass_fractions": reference["mass_fractions"],
            "robert_volume_fractions": volume_fractions,
            "mean_molar_mass_amu": mean_molar_mass,
        },
        "metrics_by_band": _metrics_by_band(
            wavelength,
            residual_clear_ppm,
            residual_strict_ppm,
            residual_native_ppm,
            p_rt_rayleigh_effect_ppm,
        ),
        "rayleigh_vertical_tau_comparison": {
            "native_to_pRT_median_ratio_0.3-1um": float(
                np.median(
                    np.asarray(native_rayleigh.tau)[:, wavelength <= 1.0]
                    / np.asarray(strict_rayleigh.tau)[:, wavelength <= 1.0]
                )
            ),
        },
        "timings": {
            **reference["timings"],
            "robert_three_cases_first_s": first_s,
            "robert_three_cases_steady_median_s": steady_s,
            "robert_native_case_first_s": native_first_s,
            "robert_native_case_steady_median_s": native_steady_s,
        },
    }
    json_path = OUTPUT_DIR / "multispecies_transmission_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUTPUT_DIR / "multispecies_transmission_benchmark_spectra.npz",
        wavelength_micron=wavelength,
        petitradtrans_clear=p_rt_clear,
        petitradtrans_rayleigh=p_rt_rayleigh,
        robert_clear=robert_clear,
        robert_strict_rayleigh=robert_strict,
        robert_native_rayleigh=robert_native,
    )
    _plot(
        wavelength,
        p_rt_clear,
        p_rt_rayleigh,
        robert_strict,
        residual_clear_ppm,
        residual_strict_ppm,
        residual_native_ppm,
        p_rt_rayleigh_effect_ppm,
        strict_rayleigh_effect_ppm,
        native_rayleigh_effect_ppm,
    )
    print(json.dumps(report, indent=2))
    return report


def _metrics_by_band(
    wavelength: np.ndarray,
    clear: np.ndarray,
    strict: np.ndarray,
    native: np.ndarray,
    rayleigh_effect: np.ndarray,
) -> dict[str, dict[str, float]]:
    bands = {"full_0.3-12um": (0.3, 12.0), "optical_0.3-1um": (0.3, 1.0), "infrared_1-12um": (1.0, 12.0)}
    output = {}
    for name, (lower, upper) in bands.items():
        selected = (wavelength >= lower) & (wavelength <= upper)
        output[name] = {
            "clear_rms_difference_ppm": float(np.sqrt(np.mean(clear[selected] ** 2))),
            "clear_max_abs_difference_ppm": float(np.max(np.abs(clear[selected]))),
            "shared_rayleigh_rms_difference_ppm": float(np.sqrt(np.mean(strict[selected] ** 2))),
            "shared_rayleigh_max_abs_difference_ppm": float(np.max(np.abs(strict[selected]))),
            "native_rayleigh_rms_difference_ppm": float(np.sqrt(np.mean(native[selected] ** 2))),
            "native_rayleigh_max_abs_difference_ppm": float(np.max(np.abs(native[selected]))),
            "petitradtrans_rayleigh_max_effect_ppm": float(np.max(np.abs(rayleigh_effect[selected]))),
        }
    return output


def _plot(
    wavelength: np.ndarray,
    p_rt_clear: np.ndarray,
    p_rt_rayleigh: np.ndarray,
    robert_strict: np.ndarray,
    residual_clear: np.ndarray,
    residual_strict: np.ndarray,
    residual_native: np.ndarray,
    p_rt_effect: np.ndarray,
    strict_effect: np.ndarray,
    native_effect: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    spectrum, residual, effect, optical = axes.flat
    spectrum.semilogx(wavelength, p_rt_clear * 1.0e6, color="#999999", lw=1.0, label="pRT3 no Rayleigh")
    spectrum.semilogx(wavelength, p_rt_rayleigh * 1.0e6, color="#222222", lw=1.2, label="pRT3 + Rayleigh")
    spectrum.semilogx(wavelength, robert_strict * 1.0e6, color="#e45756", lw=1.0, ls="--", label="ROBERT shared Rayleigh")
    spectrum.set(ylabel="Transit depth [ppm]", title="Six molecules + CIA + H$_2$/He Rayleigh")
    spectrum.legend(frameon=False)
    residual.semilogx(wavelength, residual_clear, color="#777777", lw=0.8, label="No Rayleigh")
    residual.semilogx(wavelength, residual_strict, color="#4c78a8", lw=0.9, label="Shared pRT Rayleigh")
    residual.semilogx(wavelength, residual_native, color="#f58518", lw=0.8, alpha=0.8, label="ROBERT native Rayleigh")
    residual.axhline(0.0, color="#222222", lw=0.8)
    residual.set(ylabel="ROBERT - pRT3 [ppm]", title="Transmission residual")
    residual.legend(frameon=False)
    effect.semilogx(wavelength, p_rt_effect, color="#222222", lw=1.2, label="pRT3")
    effect.semilogx(wavelength, strict_effect, color="#4c78a8", lw=0.9, ls="--", label="ROBERT, pRT opacity")
    effect.semilogx(wavelength, native_effect, color="#54a24b", lw=0.9, label="ROBERT native formula")
    effect.set(ylabel="Rayleigh transit-depth effect [ppm]", title="Rayleigh extinction isolation")
    effect.legend(frameon=False)
    selected = wavelength <= 1.0
    optical.plot(wavelength[selected], p_rt_effect[selected], color="#222222", lw=1.2, label="pRT3")
    optical.plot(wavelength[selected], native_effect[selected], color="#54a24b", lw=1.0, label="ROBERT native")
    optical.set(ylabel="Rayleigh effect [ppm]", title="Optical Rayleigh slope")
    for axis in axes.flat:
        axis.set_xlabel("Wavelength [micron]")
        axis.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "multispecies_transmission_benchmark.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
