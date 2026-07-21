"""Benchmark ROBERT against Taylor et al. (2021) Figures 1 and 2.

The reference spectra and profiles come from the paper's archived NEMESIS
forward-model folder. ROBERT uses the locally installed petitRADTRANS R=1000
HDF5 correlated-k tables for H2O and CO plus H2-H2/H2-He CIA. Differences in
line data, k-distribution construction, and RT closure are therefore expected
and are measured rather than hidden.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)

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
    grey_cloud_from_mass_extinction,
    solve_emission,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = Path(
    os.environ.get(
        "ROBERT_TAYLOR2021_DATA",
        ROOT / "data" / "taylor2021_cloud_paper",
    )
).expanduser()
PRT_DATA = ROOT / "opacity_data" / "petitRADTRANS" / "input_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "taylor2021_figures_1_2"
PAPER_DOI = "https://doi.org/10.1093/mnras/stab1854"

HDF_PATTERNS = {
    "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
    "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
}
VMR = {"H2O": 1.0e-3, "CO": 1.0e-3, "H2": 0.8483, "He": 0.1497}
MEAN_MOLECULAR_WEIGHT = 2.33
PLANET_RADIUS_M = 1.036 * 71_492_000.0
PLANET_MASS_KG = 2.034 * 1.89813e27
STAR_RADIUS_M = 0.667 * 6.957e8
STAR_TEMPERATURE_K = 4520.0
GRAVITATIONAL_CONSTANT = 6.67430e-11
GRAVITY_M_S2 = GRAVITATIONAL_CONSTANT * PLANET_MASS_KG / PLANET_RADIUS_M**2
FIGURE1_LOG_KAPPA = tuple(range(9))
FIGURE2_SSA = (0.0, 0.2, 0.5, 0.9, 0.99)
# NEMESIS kappa_cld is not ROBERT's bulk-atmosphere mass-extinction unit.
# A single offset aligns the relative cloud/gas-opacity transition; it is
# explicit here and reported as a benchmark convention, not a unit identity.
NEMESIS_TO_ROBERT_LOG_KAPPA_OFFSET = -4.0


@dataclass(frozen=True)
class SpectrumCase:
    """One named model spectrum on the shared wavelength grid."""

    name: str
    wavelength_micron: np.ndarray
    eclipse_depth: np.ndarray


@dataclass(frozen=True)
class PreparedInputs:
    """Opacity and CIA inputs shared by the model cases."""

    spectral_grid: SpectralGrid
    g_samples: np.ndarray
    g_weights: np.ndarray
    kcoeff_by_profile: Mapping[str, np.ndarray]
    cia_tables: tuple[CiaTable, CiaTable]
    opacity_paths: Mapping[str, str]


def load_nemesis_figure1(
    reference_dir: Path = REFERENCE_DIR,
) -> dict[int, SpectrumCase]:
    """Load the nine archived NEMESIS Figure 1 forward spectra."""

    cases: dict[int, SpectrumCase] = {}
    for log_kappa in FIGURE1_LOG_KAPPA:
        path = reference_dir / "figure1" / f"wasp121_{10**log_kappa}.mre"
        values = np.genfromtxt(path, skip_header=5, skip_footer=8)
        cases[log_kappa] = SpectrumCase(
            name=rf"NEMESIS log kappa={log_kappa}",
            wavelength_micron=np.asarray(values[:, 1], dtype=float),
            eclipse_depth=np.asarray(values[:, 5], dtype=float),
        )
    return cases


def load_taylor_temperature_profiles(
    reference_dir: Path = REFERENCE_DIR,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the exact 50-layer pressure-temperature profiles archived for Figure 2."""

    paths = {
        "Non-inverted": reference_dir / "figure2" / "wasp121_profile.txt",
        "Isothermal": reference_dir / "figure2" / "wasp121.txt",
        "Inverted": reference_dir / "figure2" / "wasp121_profile_inv.txt",
    }
    profiles: dict[str, np.ndarray] = {}
    pressure = np.geomspace(1.0e-5, 100.0, 50)
    for name, path in paths.items():
        rows = _atmosphere_rows(path)
        current_pressure = rows[:, 1][::-1]
        current_temperature = rows[:, 2][::-1]
        np.testing.assert_allclose(current_pressure, pressure, rtol=2.0e-3)
        profiles[name] = current_temperature
    return pressure, profiles


def _atmosphere_rows(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.replace("\t", " ").split()
        if len(fields) < 7:
            continue
        try:
            values = [float(item) for item in fields[:7]]
        except ValueError:
            continue
        if values[1] > 0.0 and values[2] > 0.0:
            rows.append(values)
    array = np.asarray(rows, dtype=float)
    if array.shape != (50, 7):
        raise ValueError(f"expected 50 atmospheric rows in {path}, found {array.shape}")
    return array


def pressure_grid_from_centers(pressure_bar: np.ndarray) -> PressureGrid:
    """Construct logarithmic edges around archived layer-center pressures."""

    centers = np.asarray(pressure_bar, dtype=float)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return PressureGrid(
        edges=edges, centers=centers, unit="bar", name="Taylor2021 archived"
    )


def temperature_edges_from_centers(temperature_k: np.ndarray) -> np.ndarray:
    """Linearly reconstruct level temperatures from archived layer centers."""

    centers = np.asarray(temperature_k, dtype=float)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = 2.0 * centers[0] - edges[1]
    edges[-1] = 2.0 * centers[-1] - edges[-2]
    return edges


def prepare_hdf_opacity(
    pressure_grid: PressureGrid,
    profiles: Mapping[str, np.ndarray],
    wavelength_micron: np.ndarray,
    *,
    prt_data: Path = PRT_DATA,
) -> PreparedInputs:
    """Bin and evaluate the existing pRT HDF5 H2O/CO k tables."""

    wavelength = np.asarray(wavelength_micron, dtype=float)
    spectral_edges = np.empty(wavelength.size + 1)
    spectral_edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    spectral_edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    spectral_edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    spectral_grid = SpectralGrid(
        values=wavelength,
        bin_edges=spectral_edges,
        unit="micron",
        role="opacity",
        name="Taylor2021 NEMESIS grid",
    )
    atmospheres = {
        name: AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=temperature,
            temperature_edges=temperature_edges_from_centers(temperature),
            composition={
                species: np.full(pressure_grid.n_layers, value)
                for species, value in VMR.items()
            },
            mean_molecular_weight=MEAN_MOLECULAR_WEIGHT,
        )
        for name, temperature in profiles.items()
    }
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in profiles}
    canonical_g = None
    canonical_weights = None
    paths: dict[str, str] = {}
    for species in ("H2O", "CO"):
        path = next(prt_data.rglob(HDF_PATTERNS[species]))
        paths[species] = str(path.resolve())
        source = CorrelatedKTable.from_petitradtrans_hdf(path, species=species)
        provider = CorrelatedKOpacityProvider(
            {species: source},
            name=f"Taylor2021-{species}-pRT-HDF5",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        table = provider.bin_to_spectral_grid(
            spectral_grid, num=300, use_rebin=False, remove_zeros=True
        ).tables[species]
        if canonical_g is None:
            canonical_g = np.asarray(table.g_samples)
            canonical_weights = np.asarray(table.g_weights)
        else:
            np.testing.assert_allclose(
                table.g_samples, canonical_g, atol=1.0e-8, rtol=0.0
            )
            np.testing.assert_allclose(
                table.g_weights, canonical_weights, atol=1.0e-8, rtol=0.0
            )
            table = replace(table, g_samples=canonical_g, g_weights=canonical_weights)
        binned_provider = CorrelatedKOpacityProvider(
            {species: table},
            name=f"Taylor2021-{species}-binned",
            interpolation="log_pressure_temperature_log_k_clip",
        )
        for name, atmosphere in atmospheres.items():
            prepared = binned_provider.prepare(
                spectral_grid, pressure_grid, species=(species,)
            )
            arrays[name].append(
                binned_provider.evaluate(atmosphere, prepared).kcoeff[0]
            )
    if canonical_g is None or canonical_weights is None:
        raise RuntimeError("no HDF5 opacity was loaded")
    return PreparedInputs(
        spectral_grid=spectral_grid,
        g_samples=canonical_g,
        g_weights=canonical_weights,
        kcoeff_by_profile={name: np.stack(values) for name, values in arrays.items()},
        cia_tables=(
            CiaTable.from_petitradtrans_hdf(
                next(prt_data.rglob("*H2--H2*.ciatable.petitRADTRANS.h5")),
                collision_pair="H2-H2",
            ),
            CiaTable.from_petitradtrans_hdf(
                next(prt_data.rglob("*H2--He*.ciatable.petitRADTRANS.h5")),
                collision_pair="H2-He",
            ),
        ),
        opacity_paths=paths,
    )


def run_profile_cases(
    pressure_grid: PressureGrid,
    temperature: np.ndarray,
    profile_name: str,
    prepared_inputs: PreparedInputs,
    *,
    log_kappa: int,
    ssa_values: tuple[float, ...],
    backend: str,
) -> dict[float, SpectrumCase]:
    """Run ROBERT cases for one archived TP profile and cloud opacity."""

    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        temperature_edges=temperature_edges_from_centers(temperature),
        composition={
            species: np.full(pressure_grid.n_layers, value)
            for species, value in VMR.items()
        },
        mean_molecular_weight=MEAN_MOLECULAR_WEIGHT,
    )
    opacity_prepared = PreparedCorrelatedKOpacity(
        provider_name="Taylor2021-pRT-HDF5-H2O-CO",
        spectral_grid=prepared_inputs.spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O", "CO"),
        g_samples=prepared_inputs.g_samples,
        g_weights=prepared_inputs.g_weights,
        cache_key=f"Taylor2021-{profile_name}",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=opacity_prepared,
        kcoeff=prepared_inputs.kcoeff_by_profile[profile_name],
        unit="cm^2/molecule",
        metadata={"source": "pRT R1000 HDF5 H2O POKAZATEL + CO HITEMP"},
    )
    gas_tau = assemble_gas_optical_depth(
        atmosphere, opacity, gravity_m_s2=GRAVITY_M_S2, gas_combination="random_overlap"
    )
    cia = [
        cia_optical_depth(
            gas_tau,
            table,
            coefficient_interpolation="log",
            temperature_extrapolation="clip",
            spectral_extrapolation="zero",
        )
        for table in prepared_inputs.cia_tables
    ]
    geometry = gauss_legendre_disk_geometry(n_mu=8)
    cases = {}
    for ssa in ssa_values:
        cloud = grey_cloud_from_mass_extinction(
            gas_tau,
            mass_extinction_cm2_g=10.0
            ** (log_kappa + NEMESIS_TO_ROBERT_LOG_KAPPA_OFFSET),
            name=f"Taylor2021 grey cloud log kappa={log_kappa}",
            single_scattering_albedo=ssa,
            asymmetry_factor=0.0,
        )
        result = solve_emission(
            gas_tau,
            geometry=geometry,
            bottom_boundary="blackbody",
            additional_optical_depths=[*cia, cloud],
            multiple_scattering_backend=backend,
            planet_radius_m=PLANET_RADIUS_M,
            star_radius_m=STAR_RADIUS_M,
            star_temperature_k=STAR_TEMPERATURE_K,
        )
        if result.eclipse_depth is None:
            raise RuntimeError("ROBERT did not return an eclipse-depth spectrum")
        cases[ssa] = SpectrumCase(
            name=f"ROBERT {profile_name} log kappa={log_kappa} SSA={ssa:g}",
            wavelength_micron=np.asarray(result.eclipse_depth.spectral_grid.values),
            eclipse_depth=np.asarray(result.eclipse_depth.values),
        )
    return cases


def run_benchmark(
    output_dir: Path = OUTPUT_DIR,
    *,
    reference_dir: Path = REFERENCE_DIR,
    prt_data: Path = PRT_DATA,
    backend: str = "sh4",
) -> dict[str, object]:
    """Run the data-backed ROBERT/NEMESIS Figures 1 and 2 benchmark."""

    output_dir.mkdir(parents=True, exist_ok=True)
    references = load_nemesis_figure1(reference_dir)
    wavelength = references[0].wavelength_micron
    pressure, profiles = load_taylor_temperature_profiles(reference_dir)
    pressure_grid = pressure_grid_from_centers(pressure)
    inputs = prepare_hdf_opacity(pressure_grid, profiles, wavelength, prt_data=prt_data)

    figure1 = {}
    for log_kappa in FIGURE1_LOG_KAPPA:
        figure1[log_kappa] = run_profile_cases(
            pressure_grid,
            profiles["Isothermal"],
            "Isothermal",
            inputs,
            log_kappa=log_kappa,
            ssa_values=(0.9,),
            backend=backend,
        )[0.9]
    figure2 = {
        name: run_profile_cases(
            pressure_grid,
            temperature,
            name,
            inputs,
            log_kappa=3,
            ssa_values=FIGURE2_SSA,
            backend=backend,
        )
        for name, temperature in profiles.items()
    }

    metrics = _figure1_metrics(references, figure1)
    _plot_figure1(
        output_dir / "taylor2021_figure1_robert_vs_nemesis.png", references, figure1
    )
    _plot_figure2(
        output_dir / "taylor2021_figure2_robert.png", pressure, profiles, figure2
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "paper": PAPER_DOI,
        "comparison": "ROBERT_pRT_HDF5_vs_archived_NEMESIS_forward_models",
        "rt_backend": backend,
        "opacity": {
            "line_tables": dict(inputs.opacity_paths),
            "species": ["H2O", "CO"],
            "vmr": {key: float(value) for key, value in VMR.items()},
            "cia": ["H2-H2", "H2-He"],
        },
        "system": {
            "gravity_m_s2": GRAVITY_M_S2,
            "planet_radius_m": PLANET_RADIUS_M,
            "star_radius_m": STAR_RADIUS_M,
            "star_temperature_k": STAR_TEMPERATURE_K,
        },
        "figure1": metrics,
        "figure2": {
            "nemesis_log10_kappa_cld": 3,
            "robert_log10_mass_extinction_cm2_g": 3
            + NEMESIS_TO_ROBERT_LOG_KAPPA_OFFSET,
            "single_scattering_albedo": list(FIGURE2_SSA),
            "temperature_profiles": list(profiles),
            "numeric_reference_status": "archived as PDFs; exact TP profiles used",
        },
    }
    (output_dir / "taylor2021_figures_1_2_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _figure1_metrics(reference, model) -> dict[str, object]:
    by_case = {}
    for log_kappa in FIGURE1_LOG_KAPPA:
        observed = reference[log_kappa].eclipse_depth
        predicted = np.interp(
            reference[log_kappa].wavelength_micron,
            model[log_kappa].wavelength_micron,
            model[log_kappa].eclipse_depth,
        )
        residual = predicted - observed
        by_case[str(log_kappa)] = {
            "median_offset_ppm": float(1.0e6 * np.median(residual)),
            "rms_ppm": float(1.0e6 * np.sqrt(np.mean(residual**2))),
            "median_fractional_difference": float(np.median(residual / observed)),
        }
    return {
        "single_scattering_albedo": 0.9,
        "nemesis_log10_kappa_cld": list(FIGURE1_LOG_KAPPA),
        "nemesis_to_robert_log_kappa_offset": NEMESIS_TO_ROBERT_LOG_KAPPA_OFFSET,
        "robert_log10_mass_extinction_cm2_g": [
            value + NEMESIS_TO_ROBERT_LOG_KAPPA_OFFSET for value in FIGURE1_LOG_KAPPA
        ],
        "reference_files": "data/taylor2021_cloud_paper/figure1/wasp121_*.mre",
        "metrics_by_log_kappa": by_case,
    }


def _plot_figure1(path: Path, reference, model) -> None:
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    colours = plt.cm.rainbow_r(np.linspace(0.05, 0.95, len(FIGURE1_LOG_KAPPA)))
    for log_kappa, colour in zip(FIGURE1_LOG_KAPPA, colours, strict=True):
        nemesis = reference[log_kappa]
        robert = model[log_kappa]
        ax.plot(nemesis.wavelength_micron, nemesis.eclipse_depth, color=colour, lw=1.5)
        ax.plot(
            robert.wavelength_micron,
            robert.eclipse_depth,
            color=colour,
            lw=1.15,
            ls="--",
        )
        interpolated = np.interp(
            nemesis.wavelength_micron, robert.wavelength_micron, robert.eclipse_depth
        )
        residual_ax.plot(
            nemesis.wavelength_micron,
            1.0e6 * (interpolated - nemesis.eclipse_depth),
            color=colour,
            lw=1.0,
        )
    ax.plot([], [], color="black", lw=1.7, label="Archived NEMESIS")
    ax.plot([], [], color="black", lw=1.4, ls="--", label="ROBERT (pRT HDF5 k tables)")
    ax.set(
        ylabel=r"$F_p/F_\star$",
        title="Taylor et al. (2021) Figure 1: ROBERT vs archived forward models",
    )
    ax.legend(frameon=False)
    colour_scale = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=8), cmap="rainbow_r"
    )
    colour_scale.set_array([])
    colour_bar = fig.colorbar(colour_scale, ax=ax, pad=0.015, ticks=FIGURE1_LOG_KAPPA)
    colour_bar.set_label(r"NEMESIS $\log_{10}(\kappa_{cld})$")
    residual_ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    residual_ax.set(
        xscale="log", xlabel="Wavelength [micron]", ylabel="ROBERT - NEMESIS [ppm]"
    )
    ax.set_xscale("log")
    ax.grid(alpha=0.2, which="both")
    residual_ax.grid(alpha=0.2, which="both")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _plot_figure2(path: Path, pressure, profiles, spectra) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    colours = ("#ff0000", "#ff9f43", "#70e8a5", "#00aeea", "#6a2cff")
    for ax, (name, cases) in zip(
        (axes[0, 0], axes[0, 1], axes[1, 0]), spectra.items(), strict=True
    ):
        for (ssa, case), colour in zip(cases.items(), colours, strict=True):
            ax.plot(
                case.wavelength_micron,
                case.eclipse_depth,
                color=colour,
                lw=1.35,
                label=f"SSA = {ssa:g}",
            )
        ax.set(
            xscale="log",
            xlabel="Wavelength [micron]",
            ylabel=r"$F_p/F_\star$",
            title=name,
        )
        ax.grid(alpha=0.2, which="both")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for (name, temperature), colour in zip(
        profiles.items(), ("#d62728", "#315efb", "#bd36c7"), strict=True
    ):
        axes[1, 1].plot(temperature, pressure, color=colour, lw=2.0, label=name)
    axes[1, 1].set(
        yscale="log",
        ylim=(pressure.max(), pressure.min()),
        xlabel="Temperature [K]",
        ylabel="Pressure [bar]",
        title="Archived TP profiles",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(alpha=0.2, which="both")
    fig.suptitle("Taylor et al. (2021) Figure 2: ROBERT with pRT HDF5 k tables")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--prt-data", type=Path, default=PRT_DATA)
    parser.add_argument("--backend", choices=("two_stream", "sh4"), default="sh4")
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(
                args.output_dir,
                reference_dir=args.reference_dir,
                prt_data=args.prt_data,
                backend=args.backend,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
