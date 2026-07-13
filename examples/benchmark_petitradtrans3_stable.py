"""Compare ROBERT with stable petitRADTRANS 3 over 0.3--12 micron."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import h5py
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    hydrostatic_path_geometry,
    solve_absorption_transmission,
    solve_emission,
)
from robert_exoplanets.diagnostics.benchmark_style import (
    PURPLE_LIGHT,
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "petitradtrans3_stable"
REFERENCE = OUTPUT_DIR / "reference_80.npz"
STAR_RADIUS_M = 1.2 * 6.957e8
PLANET_RADIUS_M = 1.0e8
REFERENCE_PRESSURE_BAR = 0.01
GRAVITY_M_S2 = 15.0
WATER_VMR = 0.00012853674809066052
ATOMIC_MASS_G = 1.66053906660e-24


def main() -> dict[str, object]:
    with np.load(REFERENCE, allow_pickle=False) as archive:
        reference_metadata = json.loads(str(archive["metadata_json"]))
        pressure = np.asarray(archive["pressure_bar"], dtype=float)
        temperature = np.asarray(archive["temperature_K"], dtype=float)
        mean_molar_mass = float(archive["mean_molar_mass_amu"])
        wavelength = np.asarray(archive["wavelength_cm"], dtype=float) * 1.0e4
        p_rt_flux = np.asarray(archive["flux_cgs_per_cm"], dtype=float) * 1.0e-1
        p_rt_transit_depth = (
            np.asarray(archive["transit_radius_cm"], dtype=float) * 1.0e-2 / STAR_RADIUS_M
        ) ** 2
        emission_opacity_cm2_g = np.asarray(archive["emission_opacities"], dtype=float)
        emission_cumulative_tau = np.asarray(
            archive["emission_optical_depths"], dtype=float
        )
        transmission_opacity_cm2_g = np.asarray(
            archive["transmission_opacities"], dtype=float
        )

    pressure_grid = PressureGrid.from_log_centers(
        pressure[0], pressure[-1], pressure.size, unit="bar", name="pRT3_pressure_nodes"
    )
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="pRT3_R1000_0.3-12um"
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=np.concatenate(([temperature[0]], temperature)),
        composition={"total_absorption": np.full(pressure.size, WATER_VMR)},
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
        provider_name="petitradtrans3-stable-evaluated-opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("total_absorption",),
        g_samples=g_samples,
        g_weights=g_weights,
        cache_key="petitradtrans3-stable-0.3-12um-80",
        metadata={"source": "petitRADTRANS 3.3.3 returned total opacity"},
    )
    emission_opacity = _effective_opacity(
        emission_opacity_cm2_g,
        prepared,
        mean_molar_mass,
    )
    transmission_opacity = _effective_opacity(
        transmission_opacity_cm2_g,
        prepared,
        mean_molar_mass,
    )
    cumulative_tau = emission_cumulative_tau[:, :, 0, :].transpose(2, 1, 0)
    layer_tau = np.maximum(
        np.diff(cumulative_tau, axis=0, prepend=np.zeros_like(cumulative_tau[:1])),
        0.0,
    )
    geometry = gauss_legendre_disk_geometry(n_mu=8)
    path_geometry = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=GRAVITY_M_S2,
        reference_radius_m=PLANET_RADIUS_M,
        reference_pressure=REFERENCE_PRESSURE_BAR,
        reference_pressure_unit="bar",
    )

    def robert_emission():
        optical_depth = assemble_gas_optical_depth(
            atmosphere, emission_opacity, gravity_m_s2=GRAVITY_M_S2
        )
        optical_depth = replace(
            optical_depth,
            species_tau=layer_tau[None, ...],
            total_tau=layer_tau,
            metadata={
                **dict(optical_depth.metadata),
                "vertical_tau_source": "petitRADTRANS3_cumulative_tau",
            },
        )
        return solve_emission(
            optical_depth,
            geometry=geometry,
            bottom_boundary="blackbody",
            thermal_integration_backend="auto",
        )

    def robert_transmission():
        optical_depth = assemble_gas_optical_depth(
            atmosphere, transmission_opacity, gravity_m_s2=GRAVITY_M_S2
        )
        return solve_absorption_transmission(
            optical_depth,
            path_geometry,
            star_radius_m=STAR_RADIUS_M,
            impact_quadrature_order=8,
        )

    emission_first, emission_result = _time_first(robert_emission)
    emission_steady = _time_steady(robert_emission, repeats=5)
    transmission_first, transmission_result = _time_first(robert_transmission)
    transmission_steady = _time_steady(robert_transmission, repeats=5)
    robert_flux = np.pi * np.asarray(emission_result.radiance.values)
    robert_transit_depth = np.asarray(transmission_result.transit_depth.values)
    emission_relative = robert_flux / p_rt_flux - 1.0
    transmission_difference_ppm = (robert_transit_depth - p_rt_transit_depth) * 1.0e6
    metrics = _band_metrics(wavelength, emission_relative, transmission_difference_ppm)
    convergence = _stable_convergence(wavelength, p_rt_flux)
    timings = {
        **{f"petitradtrans3_{key}": float(value) for key, value in reference_metadata["timings"].items()},
        "robert_emission_first_s": emission_first,
        "robert_emission_steady_median_s": emission_steady,
        "robert_transmission_first_s": transmission_first,
        "robert_transmission_steady_median_s": transmission_steady,
    }
    report = {
        "schema_version": 1,
        "comparison": "ROBERT_vs_stable_petitRADTRANS3_H2O_CIA_0.3-12um",
        "petitradtrans_version": reference_metadata["petitradtrans_version"],
        "n_layers": int(pressure.size),
        "n_wavelength": int(wavelength.size),
        "n_g_ordinates": 16,
        "wavelength_micron": [float(wavelength[0]), float(wavelength[-1])],
        "opacity": {
            "line": reference_metadata["line_species"],
            "cia": reference_metadata["cia_species"],
            "shared_evaluated_total_opacity": True,
        },
        "composition": {
            "petitradtrans_convention": "mass_mixing_ratio",
            "petitradtrans_mass_fractions": {"H2": 0.740, "He": 0.259, "H2O": 0.001},
            "robert_convention": "volume_mixing_ratio",
            "robert_H2O_volume_mixing_ratio": WATER_VMR,
            "conversion": "x_i = (X_i / M_i) / sum_j(X_j / M_j)",
            "mean_molar_mass_amu": mean_molar_mass,
        },
        "metrics_by_band": metrics,
        "petitradtrans3_emission_self_convergence": convergence,
        "timings": timings,
        "timing_note": "pRT3 calls include opacity evaluation and RT; ROBERT starts from pRT3-evaluated opacity/tau",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "petitradtrans3_stable_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT_DIR / "petitradtrans3_stable_benchmark_spectra.npz",
        wavelength_micron=wavelength,
        petitradtrans_flux_w_m2_m=p_rt_flux,
        robert_flux_w_m2_m=robert_flux,
        petitradtrans_transit_depth=p_rt_transit_depth,
        robert_transit_depth=robert_transit_depth,
    )
    _plot(
        wavelength,
        p_rt_flux,
        robert_flux,
        emission_relative,
        p_rt_transit_depth,
        robert_transit_depth,
        transmission_difference_ppm,
        timings,
    )
    print(json.dumps(report, indent=2))
    return report


def _effective_opacity(
    opacity_cm2_g: np.ndarray,
    prepared: PreparedCorrelatedKOpacity,
    mean_molar_mass: float,
) -> EvaluatedCorrelatedKOpacity:
    opacity_lwg = opacity_cm2_g[:, :, 0, :].transpose(2, 1, 0)
    kcoeff = opacity_lwg * mean_molar_mass * ATOMIC_MASS_G / WATER_VMR
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff[None, ...],
        unit="cm^2/molecule",
        metadata={"source": "petitRADTRANS3_evaluated_total_absorption_opacity"},
    )


def _band_metrics(
    wavelength: np.ndarray,
    emission_relative: np.ndarray,
    transmission_difference_ppm: np.ndarray,
) -> dict[str, dict[str, float]]:
    bands = {
        "full_0.3-12um": (0.3, 12.0),
        "optical_0.3-1um": (0.3, 1.0),
        "near_ir_1-5um": (1.0, 5.0),
        "mid_ir_5-12um": (5.0, 12.0),
    }
    output = {}
    for name, (lower, upper) in bands.items():
        selected = (wavelength >= lower) & (wavelength <= upper)
        emission = emission_relative[selected]
        transmission = transmission_difference_ppm[selected]
        output[name] = {
            "emission_median_relative_difference": float(np.median(emission)),
            "emission_rms_relative_difference": float(np.sqrt(np.mean(emission**2))),
            "emission_max_abs_relative_difference": float(np.max(np.abs(emission))),
            "transmission_median_difference_ppm": float(np.median(transmission)),
            "transmission_rms_difference_ppm": float(np.sqrt(np.mean(transmission**2))),
            "transmission_max_abs_difference_ppm": float(np.max(np.abs(transmission))),
        }
    return output


def _time_first(function):
    start = perf_counter()
    result = function()
    return perf_counter() - start, result


def _time_steady(function, repeats: int) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def _stable_convergence(
    wavelength: np.ndarray,
    flux_80: np.ndarray,
) -> dict[str, dict[str, float]]:
    spectra = {80: flux_80}
    for n_layers in (40, 160, 320, 640):
        with np.load(OUTPUT_DIR / f"convergence_{n_layers}.npz") as archive:
            candidate_wavelength = np.asarray(archive["wavelength_cm"], dtype=float) * 1.0e4
            if not np.allclose(candidate_wavelength, wavelength, rtol=1.0e-12, atol=0.0):
                raise RuntimeError("stable pRT wavelength grid changed during convergence test")
            spectra[n_layers] = np.asarray(archive["flux_cgs_per_cm"], dtype=float) * 1.0e-1
    output = {}
    for coarse, fine in ((40, 80), (80, 160), (160, 320), (320, 640)):
        relative = spectra[coarse] / spectra[fine] - 1.0
        output[f"{coarse}_vs_{fine}_layers"] = {
            "rms_relative_difference": float(np.sqrt(np.mean(relative**2))),
            "median_relative_difference": float(np.median(relative)),
            "max_abs_relative_difference": float(np.max(np.abs(relative))),
        }
    _plot_convergence(wavelength, spectra)
    return output


def _plot(
    wavelength: np.ndarray,
    p_rt_flux: np.ndarray,
    robert_flux: np.ndarray,
    emission_relative: np.ndarray,
    p_rt_transit: np.ndarray,
    robert_transit: np.ndarray,
    transmission_difference_ppm: np.ndarray,
    timings: dict[str, float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    emission, residual, transmission, timing = axes.flat
    emission.loglog(wavelength, p_rt_flux, color=REFERENCE_COLOR, lw=1.3, label="pRT 3.3.3")
    emission.loglog(wavelength, robert_flux, color=ROBERT_COLOR, lw=1.0, ls="--", label="ROBERT")
    emission.set(ylabel=r"Planet flux $F_\lambda$ [W m$^{-2}$ m$^{-1}$]", title="H$_2$O + H$_2$ CIA emission")
    residual.semilogx(wavelength, emission_relative * 100.0, color=RESIDUAL_COLOR, lw=1.0)
    residual.axhline(0.0, color=REFERENCE_COLOR, lw=0.8)
    residual.set(ylabel="(ROBERT - pRT3) / pRT3 [%]", title="Shared-opacity emission residual")
    transmission.semilogx(wavelength, p_rt_transit * 1.0e6, color=REFERENCE_COLOR, lw=1.2, label="pRT 3.3.3")
    transmission.semilogx(wavelength, robert_transit * 1.0e6, color=ROBERT_COLOR, lw=1.0, ls="--", label="ROBERT")
    twin = transmission.twinx()
    twin.semilogx(wavelength, transmission_difference_ppm, color=PURPLE_LIGHT, alpha=0.55, lw=0.8)
    transmission.set(ylabel="Transit depth [ppm]", title="Shared-opacity transmission")
    twin.set_ylabel("ROBERT - pRT3 [ppm]", color=PURPLE_LIGHT)
    labels = ["pRT3 emission", "ROBERT emission", "pRT3 transit", "ROBERT transit"]
    first = [timings["petitradtrans3_emission_first_s"], timings["robert_emission_first_s"], timings["petitradtrans3_transmission_first_s"], timings["robert_transmission_first_s"]]
    steady = [timings["petitradtrans3_emission_steady_median_s"], timings["robert_emission_steady_median_s"], timings["petitradtrans3_transmission_steady_median_s"], timings["robert_transmission_steady_median_s"]]
    x = np.arange(4)
    timing.bar(x - 0.18, first, 0.36, color="#b9b9b9", label="First")
    timing.bar(x + 0.18, steady, 0.36, color=ROBERT_COLOR, label="Steady")
    timing.set_yscale("log")
    timing.set_xticks(x, labels, rotation=18, ha="right")
    timing.set(ylabel="Wall time [s]", title="0.3--12 micron timings")
    for axis in (emission, residual, transmission):
        axis.set_xlabel("Wavelength [micron]")
        axis.grid(alpha=0.25)
    emission.legend(frameon=False)
    transmission.legend(frameon=False)
    timing.legend(frameon=False)
    timing.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "petitradtrans3_stable_benchmark.png", dpi=180)
    plt.close(fig)


def _plot_convergence(wavelength: np.ndarray, spectra: dict[int, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), constrained_layout=True)
    for n_layers in sorted(spectra):
        axes[0].loglog(wavelength, spectra[n_layers], lw=1.0, label=f"{n_layers} layers")
    reference = spectra[640]
    for n_layers in sorted(spectra)[:-1]:
        axes[1].semilogx(
            wavelength,
            (spectra[n_layers] / reference - 1.0) * 100.0,
            lw=0.9,
            label=f"{n_layers} / 640",
        )
    axes[0].set(
        ylabel=r"pRT3 planet flux $F_\lambda$ [W m$^{-2}$ m$^{-1}$]",
        title="Stable petitRADTRANS 3.3.3 emission self-convergence",
    )
    axes[1].set(xlabel="Wavelength [micron]", ylabel="Relative to 640 layers [%]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=2)
    fig.savefig(OUTPUT_DIR / "petitradtrans3_stable_emission_convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
