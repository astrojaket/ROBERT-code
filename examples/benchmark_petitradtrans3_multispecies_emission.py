"""Benchmark ROBERT six-molecule+CIA emission against stable pRT3."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
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
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    solve_emission,
)
from robert_exoplanets.rt.random_overlap import random_overlap_species_tau
from robert_exoplanets.diagnostics.benchmark_style import (
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "petitradtrans3_stable"
REFERENCE = OUTPUT_DIR / "multispecies_emission_reference.npz"
SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
GRAVITY_M_S2 = 15.0
ATOMIC_MASS_G = 1.66053906660e-24


def main() -> dict[str, object]:
    with np.load(REFERENCE, allow_pickle=False) as archive:
        reference = json.loads(str(archive["metadata_json"]))
        pressure = np.asarray(archive["pressure_bar"], dtype=float)
        temperature = np.asarray(archive["temperature_K"], dtype=float)
        wavelength = np.asarray(archive["wavelength_cm"], dtype=float) * 1.0e4
        p_rt_flux = np.asarray(archive["flux_cgs_per_cm"], dtype=float) * 1.0e-1
        p_rt_contribution = np.asarray(archive["contribution"], dtype=float)
        absorption_cm2_g = np.asarray(archive["absorption_opacities"], dtype=float)
        cumulative_tau = np.asarray(archive["optical_depths"], dtype=float)

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
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=np.concatenate(([temperature[0]], temperature)),
        composition={
            species: np.full(pressure.size, volume_fractions[species])
            for species in volume_fractions
        },
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
        cache_key="petitradtrans3-multispecies-emission-0.3-12um",
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
    p_rt_layer_tau = np.diff(
        cumulative_tau,
        axis=-1,
        prepend=np.zeros_like(cumulative_tau[..., :1]),
    ).transpose(2, 3, 1, 0)
    p_rt_layer_tau = np.maximum(p_rt_layer_tau, 0.0)
    geometry = gauss_legendre_disk_geometry(n_mu=8)

    def solve(gas_combination: str = "random_overlap", *, shared_layer_tau: bool = False):
        gas_tau = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=GRAVITY_M_S2,
            gas_combination=gas_combination,
        )
        if shared_layer_tau:
            gas_tau = replace(
                gas_tau,
                species_tau=p_rt_layer_tau,
                total_tau=random_overlap_species_tau(p_rt_layer_tau, g_weights),
                metadata={
                    **dict(gas_tau.metadata),
                    "vertical_tau_source": "petitRADTRANS3_species_cumulative_tau",
                },
            )
        return solve_emission(
            gas_tau,
            geometry=geometry,
            bottom_boundary="blackbody",
            thermal_integration_backend="auto",
        )

    start = perf_counter()
    result = solve()
    first_s = perf_counter() - start
    durations = []
    for _ in range(5):
        start = perf_counter()
        solve()
        durations.append(perf_counter() - start)
    steady_s = float(np.median(durations))

    robert_flux = np.pi * np.asarray(result.radiance.values)
    relative = robert_flux / p_rt_flux - 1.0
    strict_result = solve(shared_layer_tau=True)
    strict_flux = np.pi * np.asarray(strict_result.radiance.values)
    strict_relative = strict_flux / p_rt_flux - 1.0
    correlated_result = solve("sum_by_g")
    correlated_flux = np.pi * np.asarray(correlated_result.radiance.values)
    correlated_relative = correlated_flux / p_rt_flux - 1.0
    robert_contribution = np.array(result.layer_contribution_radiance, copy=True)
    robert_contribution[-1] += np.asarray(result.bottom_contribution_radiance)
    robert_contribution /= np.sum(robert_contribution, axis=0, keepdims=True)
    strict_contribution = np.array(strict_result.layer_contribution_radiance, copy=True)
    strict_contribution[-1] += np.asarray(strict_result.bottom_contribution_radiance)
    strict_contribution /= np.sum(strict_contribution, axis=0, keepdims=True)
    contribution_metrics = _contribution_metrics(
        pressure,
        wavelength,
        p_rt_contribution,
        strict_contribution,
    )
    report = {
        "schema_version": 1,
        "comparison": "ROBERT_vs_stable_pRT3_multispecies_CIA_emission",
        "wavelength_micron": [float(wavelength[0]), float(wavelength[-1])],
        "n_layers": int(pressure.size),
        "n_wavelength": int(wavelength.size),
        "n_g_ordinates": int(g_weights.size),
        "line_species": reference["line_species"],
        "cia_species": reference["cia_species"],
        "composition": {
            "petitradtrans_mass_fractions": reference["mass_fractions"],
            "robert_volume_fractions": volume_fractions,
            "mean_molar_mass_amu": mean_molar_mass,
        },
        "metrics_by_band": _band_metrics(wavelength, strict_relative),
        "evaluated_opacity_metrics_by_band": _band_metrics(wavelength, relative),
        "perfect_g_correlation_diagnostic_by_band": _band_metrics(
            wavelength, correlated_relative
        ),
        "contribution_function_metrics": contribution_metrics,
        "timings": {
            **reference["timings"],
            "robert_emission_first_s": first_s,
            "robert_emission_steady_median_s": steady_s,
        },
        "method": {
            "opacity_isolation": "pRT3 returned species opacity converted to molecular cross-sections",
            "gas_combination": "random_overlap",
            "robert_thermal_source": "linear Planck function in vertical optical depth",
            "petitradtrans_solver": "stable pRT3 Feautrier/correlated-k emission",
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "multispecies_emission_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT_DIR / "multispecies_emission_benchmark_spectra.npz",
        wavelength_micron=wavelength,
        pressure_bar=pressure,
        petitradtrans_flux_w_m2_m=p_rt_flux,
        robert_flux_w_m2_m=robert_flux,
        robert_shared_vertical_tau_flux_w_m2_m=strict_flux,
        robert_perfect_g_correlation_flux_w_m2_m=correlated_flux,
        petitradtrans_contribution=p_rt_contribution,
        robert_contribution=robert_contribution,
        robert_shared_vertical_tau_contribution=strict_contribution,
    )
    _plot(
        wavelength,
        pressure,
        p_rt_flux,
        robert_flux,
        strict_flux,
        relative,
        strict_relative,
        p_rt_contribution,
        strict_contribution,
    )
    print(json.dumps(report, indent=2))
    return report


def _band_metrics(wavelength: np.ndarray, relative: np.ndarray) -> dict[str, dict[str, float]]:
    bands = {
        "full_0.3-12um": (0.3, 12.0),
        "optical_0.3-1um": (0.3, 1.0),
        "near_ir_1-5um": (1.0, 5.0),
        "mid_ir_5-12um": (5.0, 12.0),
    }
    output = {}
    for name, (lower, upper) in bands.items():
        selected = (wavelength >= lower) & (wavelength <= upper)
        values = relative[selected]
        output[name] = {
            "median_relative_difference": float(np.median(values)),
            "rms_relative_difference": float(np.sqrt(np.mean(values**2))),
            "max_abs_relative_difference": float(np.max(np.abs(values))),
        }
    return output


def _contribution_metrics(
    pressure: np.ndarray,
    wavelength: np.ndarray,
    p_rt: np.ndarray,
    robert: np.ndarray,
) -> dict[str, float]:
    selected = wavelength >= 1.0
    log_pressure = np.log10(pressure)[:, None]
    p_rt_centroid = np.sum(p_rt[:, selected] * log_pressure, axis=0)
    robert_centroid = np.sum(robert[:, selected] * log_pressure, axis=0)
    p_rt_peak = pressure[np.argmax(p_rt[:, selected], axis=0)]
    robert_peak = pressure[np.argmax(robert[:, selected], axis=0)]
    centroid_difference = robert_centroid - p_rt_centroid
    peak_difference = np.log10(robert_peak / p_rt_peak)
    return {
        "domain": "1-12um",
        "centroid_pressure_rms_difference_dex": float(
            np.sqrt(np.mean(centroid_difference**2))
        ),
        "centroid_pressure_median_difference_dex": float(np.median(centroid_difference)),
        "peak_pressure_rms_difference_dex": float(np.sqrt(np.mean(peak_difference**2))),
        "peak_pressure_median_difference_dex": float(np.median(peak_difference)),
    }


def _plot(
    wavelength: np.ndarray,
    pressure: np.ndarray,
    p_rt_flux: np.ndarray,
    robert_flux: np.ndarray,
    strict_flux: np.ndarray,
    relative: np.ndarray,
    strict_relative: np.ndarray,
    p_rt_contribution: np.ndarray,
    robert_contribution: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    spectrum, residual, p_rt_map, robert_map = axes.flat
    spectrum.loglog(wavelength, p_rt_flux, color=REFERENCE_COLOR, lw=1.2, label="pRT 3.3.3")
    spectrum.loglog(
        wavelength, robert_flux, color=ROBERT_COLOR, lw=0.8, alpha=0.7, label="ROBERT evaluated opacity"
    )
    spectrum.loglog(
        wavelength,
        strict_flux,
        color="darkorchid",
        lw=0.9,
        ls="--",
        label="ROBERT shared vertical tau",
    )
    spectrum.set(
        ylabel=r"Planet flux $F_\lambda$ [W m$^{-2}$ m$^{-1}$]",
        title="Six molecules + H$_2$-H$_2$/H$_2$-He CIA",
    )
    spectrum.legend(frameon=False)
    residual.semilogx(
        wavelength, relative * 100.0, color=ROBERT_COLOR, lw=0.7, label="Evaluated opacity"
    )
    residual.semilogx(
        wavelength,
        strict_relative * 100.0,
        color=RESIDUAL_COLOR,
        lw=0.8,
        label="Shared vertical tau",
    )
    residual.axhline(0.0, color=REFERENCE_COLOR, lw=0.7)
    residual.set(ylabel="(ROBERT - pRT3) / pRT3 [%]", title="Emission residual")
    residual.legend(frameon=False)
    positive = wavelength >= 1.0
    vmax = max(
        float(np.percentile(p_rt_contribution[:, positive], 99.5)),
        float(np.percentile(robert_contribution[:, positive], 99.5)),
    )
    for axis, contribution, title in (
        (p_rt_map, p_rt_contribution, "pRT3 normalized contribution"),
        (robert_map, robert_contribution, "ROBERT normalized contribution"),
    ):
        mesh = axis.pcolormesh(
            wavelength,
            pressure,
            contribution,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set(xscale="log", yscale="log", ylabel="Pressure [bar]", title=title)
        axis.invert_yaxis()
        fig.colorbar(mesh, ax=axis, label="Normalized layer contribution")
    for axis in axes.flat:
        axis.set_xlabel("Wavelength [micron]")
    spectrum.grid(alpha=0.25)
    residual.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "multispecies_emission_benchmark.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
