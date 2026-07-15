"""Benchmark the shared Mie cloud protocol in emission and transmission.

The geometry-independent cloud object is compared with an explicit lower-level
Mie construction using the same MgSiO3 refractive index, lognormal population,
and condensate profile. Both optical properties and resulting spectra must be
identical. The report also links the independent official-PICASO Mie benchmark,
which tests the underlying solver and database differences separately.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    OpticalConstantsCatalog,
    ParameterizedMieCloudModel,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    gauss_legendre_disk_geometry,
    hydrostatic_path_geometry,
    lognormal_mie_optics,
    mie_cloud_from_mass_fraction,
    solve_absorption_transmission,
    solve_emission_spectrum,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "shared_mie_protocol"
INDEPENDENT_REPORT = (
    ROOT
    / "examples"
    / "outputs"
    / "official_picaso_molecular_cloud_parity"
    / "official_picaso_molecular_cloud_parity.json"
)


def main(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pressure_grid = PressureGrid.from_log_centers(
        1.0e-5, 10.0, 80, unit="bar", name="shared Mie benchmark"
    )
    spectral_grid = SpectralGrid.from_array(
        np.geomspace(1.0, 12.0, 240), unit="micron", role="opacity"
    )
    temperature_edges = 780.0 + 820.0 * np.linspace(
        0.0, 1.0, pressure_grid.n_layers + 1
    ) ** 0.72
    temperature = 0.5 * (temperature_edges[:-1] + temperature_edges[1:])
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=temperature_edges,
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-3)},
        mean_molecular_weight=2.3,
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="Mie protocol isolation",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="shared-mie-protocol",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure_grid.n_layers, spectral_grid.size, 1)),
        unit="m^2/molecule",
    )
    gravity = 8.42
    gas = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=gravity)
    index = OpticalConstantsCatalog(
        ROOT / "data" / "optical_constants" / "exo_skryer"
    ).load("MgSiO3")
    model = ParameterizedMieCloudModel(
        refractive_index_wavelength_micron=(),
        real_index_parameter_names=(),
        log10_imaginary_index_parameter_names=(),
        fixed_refractive_index=index,
        log10_condensate_mass_fraction_parameter="log_cloud_mass_fraction",
        log10_effective_radius_micron_parameter="log_cloud_radius_micron",
        particle_density_kg_m3=3200.0,
        geometric_stddev=1.6,
        log10_cloud_top_pressure_bar_parameter="log_cloud_top_pressure_bar",
        log10_cloud_base_pressure_bar_parameter="log_cloud_base_pressure_bar",
        quadrature_points=24,
        multiple_scattering_backend="sh4",
    )
    parameters = {
        "log_cloud_mass_fraction": np.log10(1.2e-5),
        "log_cloud_radius_micron": np.log10(0.3),
        "log_cloud_top_pressure_bar": -4.0,
        "log_cloud_base_pressure_bar": 0.0,
    }
    started = perf_counter()
    (shared_cloud,) = model.evaluate(gas, parameters)
    shared_seconds = perf_counter() - started
    started = perf_counter()
    particle_optics = lognormal_mie_optics(
        index,
        spectral_grid,
        effective_radius_micron=0.3,
        geometric_stddev=1.6,
        particle_density_kg_m3=3200.0,
        quadrature_points=24,
    )
    pressure_bar = pressure_grid.centers
    mass_fraction = np.where(
        (pressure_bar >= 1.0e-4) & (pressure_bar <= 1.0), 1.2e-5, 0.0
    )
    manual_cloud = mie_cloud_from_mass_fraction(
        gas, particle_optics, condensate_mass_fraction=mass_fraction
    )
    manual_seconds = perf_counter() - started

    geometry = hydrostatic_path_geometry(
        atmosphere,
        gravity_m_s2=gravity,
        reference_radius_m=7.1492e7 * 1.057,
        reference_pressure=10.0,
        reference_pressure_unit="bar",
    )
    planet_radius = 7.1492e7 * 1.057
    star_radius = 6.957e8 * 0.813
    common_emission = dict(
        geometry=gauss_legendre_disk_geometry(6),
        multiple_scattering_backend="sh4",
        planet_radius_m=planet_radius,
        star_radius_m=star_radius,
        star_temperature_k=4715.0,
    )
    clear_emission = solve_emission_spectrum(gas, **common_emission)
    shared_emission = solve_emission_spectrum(
        gas, additional_optical_depths=(shared_cloud,), **common_emission
    )
    manual_emission = solve_emission_spectrum(
        gas, additional_optical_depths=(manual_cloud,), **common_emission
    )
    clear_transmission = solve_absorption_transmission(
        gas, geometry, star_radius_m=star_radius, impact_quadrature_order=8
    )
    shared_transmission = solve_absorption_transmission(
        gas,
        geometry,
        star_radius_m=star_radius,
        additional_optical_depths=(shared_cloud,),
        impact_quadrature_order=8,
    )
    manual_transmission = solve_absorption_transmission(
        gas,
        geometry,
        star_radius_m=star_radius,
        additional_optical_depths=(manual_cloud,),
        impact_quadrature_order=8,
    )
    emission_difference_ppm = (
        shared_emission.values - manual_emission.values
    ) * 1.0e6
    transmission_difference_ppm = (
        shared_transmission.transit_depth.values
        - manual_transmission.transit_depth.values
    ) * 1.0e6
    independent = None
    if INDEPENDENT_REPORT.exists():
        independent_report = json.loads(INDEPENDENT_REPORT.read_text(encoding="utf-8"))
        independent = {
            "report": str(INDEPENDENT_REPORT.relative_to(ROOT)),
            "opacity_stride": independent_report["sampling"]["opacity_stride"],
            "cloud_mass_extinction": independent_report["metrics"]["cloud_mass_extinction"],
            "emission_cloud_effect_disagreement_ppm": independent_report["metrics"][
                "emission_cloud_effect_disagreement_ppm"
            ],
            "transmission_cloud_effect_disagreement_ppm": independent_report["metrics"][
                "transmission_cloud_effect_disagreement_ppm"
            ],
        }
    report = {
        "schema_version": 1,
        "benchmark": "shared_MgSiO3_Mie_cloud_protocol_emission_transmission",
        "contract": {
            "material": "MgSiO3",
            "effective_radius_micron": 0.3,
            "geometric_stddev": 1.6,
            "quadrature_points": 24,
            "particle_density_kg_m3": 3200.0,
            "condensate_mass_fraction": 1.2e-5,
            "cloud_pressure_bar": [1.0e-4, 1.0],
        },
        "protocol_identity": {
            "extinction_tau_max_abs": float(
                np.max(np.abs(shared_cloud.extinction_tau - manual_cloud.extinction_tau))
            ),
            "single_scattering_albedo_max_abs": float(
                np.max(
                    np.abs(
                        shared_cloud.single_scattering_albedo
                        - manual_cloud.single_scattering_albedo
                    )
                )
            ),
            "phase_moments_max_abs": float(
                np.max(
                    np.abs(
                        shared_cloud.phase_function_moments
                        - manual_cloud.phase_function_moments
                    )
                )
            ),
            "emission_spectrum_max_abs_difference_ppm": float(
                np.max(np.abs(emission_difference_ppm))
            ),
            "transmission_spectrum_max_abs_difference_ppm": float(
                np.max(np.abs(transmission_difference_ppm))
            ),
        },
        "timing_seconds": {
            "shared_protocol_cloud_evaluation": shared_seconds,
            "explicit_lower_level_cloud_evaluation": manual_seconds,
        },
        "independent_official_picaso_context": independent,
    }
    arrays = {
        "wavelength_micron": spectral_grid.values,
        "mass_extinction_m2_kg": particle_optics.mass_extinction_m2_kg,
        "single_scattering_albedo": shared_cloud.single_scattering_albedo[0],
        "asymmetry_factor": shared_cloud.asymmetry_factor[0],
        "clear_emission": clear_emission.values,
        "shared_emission": shared_emission.values,
        "clear_transmission": clear_transmission.transit_depth.values,
        "shared_transmission": shared_transmission.transit_depth.values,
    }
    (output_dir / "shared_mie_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "shared_mie_protocol_spectra.npz", **arrays)
    _plot(output_dir / "shared_mie_protocol.png", arrays)
    print(json.dumps(report, indent=2))
    return report


def _plot(path: Path, arrays) -> None:
    wavelength = arrays["wavelength_micron"]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col")
    axes[0, 0].loglog(wavelength, arrays["mass_extinction_m2_kg"], color="#6f2dbd")
    axes[0, 0].set_ylabel("Mass extinction (m2 kg-1)")
    axes[1, 0].semilogx(wavelength, arrays["single_scattering_albedo"], label="omega")
    axes[1, 0].semilogx(wavelength, arrays["asymmetry_factor"], label="g")
    axes[1, 0].set_ylabel("Particle scattering")
    axes[1, 0].set_xlabel("Wavelength (micron)")
    axes[1, 0].legend(frameon=False)
    axes[0, 1].semilogx(
        wavelength,
        (arrays["shared_emission"] - arrays["clear_emission"]) * 1.0e6,
        color="#6f2dbd",
    )
    axes[0, 1].set_ylabel("Emission cloud effect (ppm)")
    axes[1, 1].semilogx(
        wavelength,
        (arrays["shared_transmission"] - arrays["clear_transmission"]) * 1.0e6,
        color="#6f2dbd",
    )
    axes[1, 1].set_ylabel("Transmission cloud effect (ppm)")
    axes[1, 1].set_xlabel("Wavelength (micron)")
    figure.suptitle("Shared MgSiO3 Mie particle model")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
