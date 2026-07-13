"""Benchmark ROBERT against JAX petitRADTRANS 4 with one shared ExoMol table."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from petitRADTRANS import __version__ as petit_version  # noqa: E402
from petitRADTRANS.radtrans import Radtrans  # noqa: E402

from robert_exoplanets import (  # noqa: E402
    AtmosphereState,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    EvaluatedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    hydrostatic_path_geometry,
    solve_absorption_transmission,
    solve_emission,
)
from robert_exoplanets.diagnostics.benchmark_style import (  # noqa: E402
    PURPLE_LIGHT,
    REFERENCE_COLOR,
    RESIDUAL_COLOR,
    ROBERT_COLOR,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = ROOT / "opacity_data" / "petitRADTRANS" / "input_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "petitradtrans4"
LINE_SPECIES = "H2O__POKAZATEL"
ROBERT_SPECIES = "H2O"
WAVELENGTH_BOUNDARIES_MICRON = np.array([2.8, 5.2])
N_LAYERS = 80
GRAVITY_CGS = 1500.0
PLANET_RADIUS_CM = 1.0e10
STAR_RADIUS_M = 1.2 * 6.957e8
REFERENCE_PRESSURE_BAR = 0.01
MASS_FRACTIONS = {"H2": 0.740, "He": 0.259, "H2O": 0.001}
MOLAR_MASSES = {"H2": 2.01588, "He": 4.002602, "H2O": 18.01528}


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pressure = np.geomspace(1.0e-5, 100.0, N_LAYERS)
    temperature = 900.0 + 900.0 * (np.log10(pressure) + 5.0) / 7.0
    mean_molar_mass, water_vmr = _mass_to_volume_composition()
    p_rt_mass_fractions = {LINE_SPECIES: np.full(N_LAYERS, MASS_FRACTIONS["H2O"])}
    mean_molar_masses = np.full(N_LAYERS, mean_molar_mass)

    construct_start = perf_counter()
    p_rt = Radtrans(
        pressures=pressure,
        wavelength_boundaries=WAVELENGTH_BOUNDARIES_MICRON,
        line_species=[LINE_SPECIES],
        gas_continuum_contributors=[],
        scattering_in_emission=False,
        path_input_data=str(INPUT_DATA),
    )
    p_rt_construct_s = perf_counter() - construct_start

    def p_rt_emission() -> tuple[object, object, dict[str, object]]:
        return p_rt.calculate_flux(
            temperatures=temperature,
            mass_fractions=p_rt_mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=GRAVITY_CGS,
            frequencies_to_wavelengths=True,
            return_opacities=True,
        )

    def p_rt_transmission() -> tuple[object, object, dict[str, object]]:
        return p_rt.calculate_transit_radii(
            temperatures=temperature,
            mass_fractions=p_rt_mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=GRAVITY_CGS,
            reference_pressure=REFERENCE_PRESSURE_BAR,
            planet_radius=PLANET_RADIUS_CM,
            variable_gravity=False,
            frequencies_to_wavelengths=True,
        )

    p_rt_emission_first_s, p_rt_emission_result = _time_first(p_rt_emission)
    p_rt_emission_steady_s = _time_steady(p_rt_emission)
    p_rt_transmission_first_s, p_rt_transmission_result = _time_first(p_rt_transmission)
    p_rt_transmission_steady_s = _time_steady(p_rt_transmission)

    wavelength_cm = np.asarray(p_rt_emission_result[0], dtype=float)
    wavelength_micron = wavelength_cm * 1.0e4
    p_rt_flux_cgs_per_cm = np.asarray(p_rt_emission_result[1], dtype=float)
    # erg s^-1 cm^-2 cm^-1 -> W m^-2 m^-1 (10^-7 * 10^4 * 10^2).
    p_rt_flux_si = p_rt_flux_cgs_per_cm * 1.0e-1
    p_rt_transit_radius_m = np.asarray(p_rt_transmission_result[1], dtype=float) * 1.0e-2
    p_rt_transit_depth = (p_rt_transit_radius_m / STAR_RADIUS_M) ** 2

    robert_construct_start = perf_counter()
    robert_state = _build_robert_state(
        pressure,
        temperature,
        wavelength_micron,
        mean_molar_mass,
        water_vmr,
    )
    robert_construct_s = perf_counter() - robert_construct_start
    provider, spectral_grid, pressure_grid, atmosphere = robert_state
    prepared = provider.prepare(spectral_grid, pressure_grid, species=(ROBERT_SPECIES,))
    geometry = gauss_legendre_disk_geometry(n_mu=8)
    p_rt_opacity_cm2_g = np.asarray(
        p_rt_emission_result[2]["opacities"],
        dtype=float,
    )[:, :, 0, :].transpose(2, 1, 0)
    shared_kcoeff = (
        p_rt_opacity_cm2_g
        * mean_molar_mass
        * 1.66053906660e-24
        / water_vmr
    )
    shared_opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=shared_kcoeff[None, ...],
        unit="cm^2/molecule",
        metadata={"source": "petitRADTRANS4_evaluated_total_absorption_opacity"},
    )
    p_rt_cumulative_tau = np.asarray(
        p_rt_emission_result[2]["optical_depths"],
        dtype=float,
    )[:, :, 0, :].transpose(2, 1, 0)
    p_rt_layer_tau = np.diff(
        p_rt_cumulative_tau,
        axis=0,
        prepend=np.zeros_like(p_rt_cumulative_tau[:1]),
    )
    p_rt_layer_tau = np.maximum(p_rt_layer_tau, 0.0)

    def robert_emission():
        optical_depth = assemble_gas_optical_depth(
            atmosphere,
            shared_opacity,
            gravity_m_s2=GRAVITY_CGS / 100.0,
        )
        optical_depth = replace(
            optical_depth,
            species_tau=p_rt_layer_tau[None, ...],
            total_tau=p_rt_layer_tau,
            metadata={
                **dict(optical_depth.metadata),
                "vertical_tau_source": "petitRADTRANS4_cumulative_optical_depth_difference",
            },
        )
        return solve_emission(
            optical_depth,
            geometry=geometry,
            bottom_boundary="blackbody",
            thermal_integration_backend="auto",
        )

    path_geometry = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=GRAVITY_CGS / 100.0,
        reference_radius_m=PLANET_RADIUS_CM * 1.0e-2,
        reference_pressure=REFERENCE_PRESSURE_BAR,
        reference_pressure_unit="bar",
    )

    def robert_transmission():
        optical_depth = assemble_gas_optical_depth(
            atmosphere,
            shared_opacity,
            gravity_m_s2=GRAVITY_CGS / 100.0,
        )
        return solve_absorption_transmission(
            optical_depth,
            path_geometry,
            star_radius_m=STAR_RADIUS_M,
            impact_quadrature_order=8,
        )

    robert_emission_first_s, robert_emission_result = _time_first(robert_emission)
    robert_emission_steady_s = _time_steady(robert_emission)
    robert_transmission_first_s, robert_transmission_result = _time_first(robert_transmission)
    robert_transmission_steady_s = _time_steady(robert_transmission)

    robert_flux_si = np.pi * np.asarray(robert_emission_result.radiance.values)
    robert_transit_depth = np.asarray(robert_transmission_result.transit_depth.values)
    emission_relative = (robert_flux_si - p_rt_flux_si) / p_rt_flux_si
    transmission_difference_ppm = (robert_transit_depth - p_rt_transit_depth) * 1.0e6
    p_rt_convergence = _petitradtrans_emission_convergence(
        wavelength_micron,
        p_rt_flux_si,
    )

    timings = {
        "petitradtrans4_construct_and_load_s": p_rt_construct_s,
        "petitradtrans4_emission_first_s": p_rt_emission_first_s,
        "petitradtrans4_emission_steady_median_s": p_rt_emission_steady_s,
        "petitradtrans4_transmission_first_s": p_rt_transmission_first_s,
        "petitradtrans4_transmission_steady_median_s": p_rt_transmission_steady_s,
        "robert_load_and_construct_s": robert_construct_s,
        "robert_emission_first_s": robert_emission_first_s,
        "robert_emission_steady_median_s": robert_emission_steady_s,
        "robert_transmission_first_s": robert_transmission_first_s,
        "robert_transmission_steady_median_s": robert_transmission_steady_s,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "comparison": "ROBERT_vs_petitRADTRANS4_shared_ExoMol_POKAZATEL_H2O",
        "petitradtrans_version": petit_version,
        "petitradtrans_git_commit": "8f612343f9cdef0b232c4ec469f0a96b9224ccd8",
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "platform": platform.platform(),
        "n_layers": N_LAYERS,
        "n_wavelength": int(wavelength_micron.size),
        "n_g_ordinates": int(prepared.g_weights.size),
        "opacity": {
            "species": LINE_SPECIES,
            "unit": provider.tables[ROBERT_SPECIES].unit,
            "source_format": "petitradtrans_hdf5",
            "doi": provider.tables[ROBERT_SPECIES].metadata.get("doi", ""),
            "cia_included": False,
        },
        "atmosphere": {
            "pressure_bar": [float(pressure[0]), float(pressure[-1])],
            "temperature_K": [float(temperature[0]), float(temperature[-1])],
            "gravity_m_s2": GRAVITY_CGS / 100.0,
            "mean_molar_mass_amu": mean_molar_mass,
            "water_mass_fraction": MASS_FRACTIONS["H2O"],
            "water_volume_mixing_ratio": water_vmr,
            "variable_gravity": False,
        },
        "metrics": {
            "emission_median_relative_difference": float(np.median(emission_relative)),
            "emission_rms_relative_difference": float(np.sqrt(np.mean(emission_relative**2))),
            "emission_max_abs_relative_difference": float(np.max(np.abs(emission_relative))),
            "transmission_median_difference_ppm": float(np.median(transmission_difference_ppm)),
            "transmission_rms_difference_ppm": float(
                np.sqrt(np.mean(transmission_difference_ppm**2))
            ),
            "transmission_max_abs_difference_ppm": float(
                np.max(np.abs(transmission_difference_ppm))
            ),
        },
        "timings": timings,
        "petitradtrans4_emission_self_convergence": p_rt_convergence,
        "timing_definition": {
            "first": "first synchronized call; includes JAX compilation where applicable",
            "steady": "median of 7 synchronized calls after first call",
            "robert_scope": "RT using pRT4 cumulative optical depth differenced onto its pressure nodes",
        },
        "interpretation_limits": [
            "The pRT4 cumulative molecular optical depth is supplied exactly to ROBERT.",
            "CIA, clouds, scattering, refraction, and stellar limb darkening are excluded.",
            "Residuals include pressure-cell, lower-boundary, source interpolation, angular quadrature, and spherical-annulus discretization differences.",
        ],
    }
    json_path = OUTPUT_DIR / "petitradtrans4_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUTPUT_DIR / "petitradtrans4_benchmark_spectra.npz",
        wavelength_micron=wavelength_micron,
        petitradtrans_flux_w_m2_m=p_rt_flux_si,
        robert_flux_w_m2_m=robert_flux_si,
        petitradtrans_transit_depth=p_rt_transit_depth,
        robert_transit_depth=robert_transit_depth,
    )
    _plot(
        wavelength_micron,
        p_rt_flux_si,
        robert_flux_si,
        emission_relative,
        p_rt_transit_depth,
        robert_transit_depth,
        transmission_difference_ppm,
        timings,
    )
    print(json.dumps(report, indent=2))
    return report


def _build_robert_state(
    pressure: np.ndarray,
    temperature: np.ndarray,
    wavelength_micron: np.ndarray,
    mean_molar_mass: float,
    water_vmr: float,
) -> tuple[CorrelatedKOpacityProvider, SpectralGrid, PressureGrid, AtmosphereState]:
    opacity_path = next(INPUT_DATA.rglob("*POKAZATEL*.ktable.petitRADTRANS.h5"))
    full_table = CorrelatedKTable.from_petitradtrans_hdf(
        opacity_path,
        species=ROBERT_SPECIES,
    )
    selected = np.array(
        [
            int(np.argmin(np.abs(full_table.wavelength_micron - wavelength)))
            for wavelength in wavelength_micron
        ],
        dtype=int,
    )
    if not np.allclose(
        full_table.wavelength_micron[selected],
        wavelength_micron,
        rtol=2.0e-7,
        atol=0.0,
    ):
        raise RuntimeError("pRT runtime wavelength grid does not match its HDF5 opacity bins")
    table = CorrelatedKTable(
        species=ROBERT_SPECIES,
        pressure_bar=full_table.pressure_bar,
        temperature_K=full_table.temperature_K,
        wavenumber_cm_inverse=full_table.wavenumber_cm_inverse[selected],
        wavelength_micron=full_table.wavelength_micron[selected],
        g_samples=full_table.g_samples,
        g_weights=full_table.g_weights,
        kcoeff=full_table.kcoeff[:, :, selected, :],
        unit=full_table.unit,
        metadata=full_table.metadata,
    )
    provider = CorrelatedKOpacityProvider(
        {ROBERT_SPECIES: table},
        name="petitradtrans4-shared-pokazatel",
        interpolation="log_pressure_temperature_log_k_clip",
    )
    pressure_grid = PressureGrid.from_log_centers(
        float(pressure[0]),
        float(pressure[-1]),
        pressure.size,
        unit="bar",
        name="petitradtrans4_pressure_points",
    )
    temperature_edges = np.concatenate(([temperature[0]], temperature))
    spectral_grid = SpectralGrid.from_array(
        table.wavelength_micron,
        unit="micron",
        role="opacity",
        name="petitradtrans4_R1000",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=temperature_edges,
        composition={ROBERT_SPECIES: np.full(pressure.size, water_vmr)},
        mean_molecular_weight=np.full(pressure.size, mean_molar_mass),
        metadata={"composition_source": "mass-fraction conversion shared with pRT4"},
    )
    return provider, spectral_grid, pressure_grid, atmosphere


def _mass_to_volume_composition() -> tuple[float, float]:
    reciprocal_mean_mass = sum(
        MASS_FRACTIONS[species] / MOLAR_MASSES[species] for species in MASS_FRACTIONS
    )
    mean_molar_mass = 1.0 / reciprocal_mean_mass
    water_vmr = MASS_FRACTIONS["H2O"] * mean_molar_mass / MOLAR_MASSES["H2O"]
    return mean_molar_mass, water_vmr


def _time_first(function):
    start = perf_counter()
    result = function()
    _synchronize(result)
    return perf_counter() - start, result


def _time_steady(function, repeats: int = 7) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        _synchronize(result)
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def _petitradtrans_emission_convergence(
    wavelength_micron: np.ndarray,
    flux_80: np.ndarray,
) -> dict[str, dict[str, float]]:
    spectra = {80: np.asarray(flux_80, dtype=float)}
    for n_layers in (40, 160, 320, 640):
        pressure = np.geomspace(1.0e-5, 100.0, n_layers)
        temperature = 900.0 + 900.0 * (np.log10(pressure) + 5.0) / 7.0
        atmosphere = Radtrans(
            pressures=pressure,
            wavelength_boundaries=WAVELENGTH_BOUNDARIES_MICRON,
            line_species=[LINE_SPECIES],
            gas_continuum_contributors=[],
            scattering_in_emission=False,
            path_input_data=str(INPUT_DATA),
        )
        wavelengths, flux, _ = atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions={
                LINE_SPECIES: np.full(n_layers, MASS_FRACTIONS["H2O"])
            },
            mean_molar_masses=np.full(n_layers, _mass_to_volume_composition()[0]),
            reference_gravity=GRAVITY_CGS,
            frequencies_to_wavelengths=True,
        )
        _synchronize(flux)
        if not np.allclose(
            np.asarray(wavelengths) * 1.0e4,
            wavelength_micron,
            rtol=2.0e-7,
            atol=0.0,
        ):
            raise RuntimeError("pRT4 wavelength grid changed during convergence test")
        spectra[n_layers] = np.asarray(flux, dtype=float) * 1.0e-1

    metrics: dict[str, dict[str, float]] = {}
    for coarse, fine in ((40, 80), (80, 160), (160, 320), (320, 640)):
        relative = spectra[coarse] / spectra[fine] - 1.0
        metrics[f"{coarse}_vs_{fine}_layers"] = {
            "rms_relative_difference": float(np.sqrt(np.mean(relative**2))),
            "median_relative_difference": float(np.median(relative)),
            "max_abs_relative_difference": float(np.max(np.abs(relative))),
        }
    _plot_convergence(wavelength_micron, spectra)
    return metrics


def _synchronize(value) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


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
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    emission, emission_residual, transmission, timing = axes.flat
    emission.plot(wavelength, p_rt_flux, color=REFERENCE_COLOR, lw=1.5, label="petitRADTRANS 4")
    emission.plot(wavelength, robert_flux, color=ROBERT_COLOR, lw=1.1, ls="--", label="ROBERT")
    emission.set(ylabel=r"Planet flux $F_\lambda$ [W m$^{-2}$ m$^{-1}$]", title="Shared ExoMol POKAZATEL H$_2$O emission")
    emission.legend(frameon=False)
    emission_residual.plot(wavelength, emission_relative * 100.0, color=RESIDUAL_COLOR)
    emission_residual.axhline(0.0, color=REFERENCE_COLOR, lw=0.8)
    emission_residual.set(ylabel="(ROBERT - pRT4) / pRT4 [%]", title="Emission residual")
    transmission.plot(wavelength, p_rt_transit * 1.0e6, color=REFERENCE_COLOR, lw=1.5, label="petitRADTRANS 4")
    transmission.plot(wavelength, robert_transit * 1.0e6, color=ROBERT_COLOR, lw=1.1, ls="--", label="ROBERT")
    transmission_twin = transmission.twinx()
    transmission_twin.plot(wavelength, transmission_difference_ppm, color=PURPLE_LIGHT, alpha=0.55, lw=0.9)
    transmission.set(ylabel="Transit depth [ppm]", title="Shared-opacity transmission")
    transmission_twin.set_ylabel("ROBERT - pRT4 [ppm]", color=PURPLE_LIGHT)
    transmission.legend(frameon=False)

    labels = ["pRT4 emission", "ROBERT emission", "pRT4 transit", "ROBERT transit"]
    first = [
        timings["petitradtrans4_emission_first_s"],
        timings["robert_emission_first_s"],
        timings["petitradtrans4_transmission_first_s"],
        timings["robert_transmission_first_s"],
    ]
    steady = [
        timings["petitradtrans4_emission_steady_median_s"],
        timings["robert_emission_steady_median_s"],
        timings["petitradtrans4_transmission_steady_median_s"],
        timings["robert_transmission_steady_median_s"],
    ]
    x = np.arange(len(labels))
    timing.bar(x - 0.18, first, width=0.36, color="#b9b9b9", label="First call")
    timing.bar(x + 0.18, steady, width=0.36, color=ROBERT_COLOR, label="Steady median")
    timing.set_yscale("log")
    timing.set_xticks(x, labels, rotation=18, ha="right")
    timing.set(ylabel="Wall time [s]", title="Synchronized CPU timings")
    timing.legend(frameon=False)
    for axis in (emission, emission_residual, transmission):
        axis.set_xlabel("Wavelength [micron]")
        axis.grid(alpha=0.25)
    timing.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "petitradtrans4_benchmark.png", dpi=180)
    plt.close(fig)


def _plot_convergence(
    wavelength: np.ndarray,
    spectra: dict[int, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), constrained_layout=True)
    for n_layers in sorted(spectra):
        axes[0].plot(wavelength, spectra[n_layers], lw=1.0, label=f"{n_layers} layers")
    reference = spectra[max(spectra)]
    for n_layers in sorted(spectra)[:-1]:
        axes[1].plot(
            wavelength,
            (spectra[n_layers] / reference - 1.0) * 100.0,
            lw=1.0,
            label=f"{n_layers} / {max(spectra)}",
        )
    axes[0].set(
        ylabel=r"pRT4 planet flux $F_\lambda$ [W m$^{-2}$ m$^{-1}$]",
        title="petitRADTRANS 4 emission self-convergence",
    )
    axes[1].set(
        xlabel="Wavelength [micron]",
        ylabel="Relative to 640 layers [%]",
    )
    for axis in axes:
        axis.legend(frameon=False, ncol=2)
        axis.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "petitradtrans4_emission_convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
