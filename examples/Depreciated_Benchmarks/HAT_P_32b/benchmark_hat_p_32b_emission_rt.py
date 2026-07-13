"""Benchmark ROBERT's cloud-free emission solver against HAT-P-32b output."""

from __future__ import annotations

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
    CompositionMeanMolecularWeight,
    CorrelatedKOpacityProvider,
    DirectStellarBeam,
    FastChemEquilibriumChemistry,
    PressureGrid,
    SpectralGrid,
    SingleScatteringSource,
    TabulatedTemperatureProfile,
    assemble_gas_optical_depth,
    cia_optical_depth,
    gauss_legendre_disk_geometry,
    hydrostatic_path_geometry,
    load_emission_benchmark_csv,
    lobatto_phase_geometry,
    rayleigh_scattering_optical_depth,
    read_cia_table,
    solve_clear_sky_emission,
)

DEFAULT_BENCHMARK_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "HAT-P-32b"
    / "emission"
    / "emission_R1000.csv"
)
DEFAULT_PT_CSV = (
    Path.home()
    / "Dropbox"
    / "PostDoc4"
    / "Emission_Example"
    / "PTprofiles-Teq_1800-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_1.8-TiOVO_false-daysideavg-w_mu_area.csv"
)
DEFAULT_KTA_DIR = Path.home() / "Dropbox" / "PostDoc4" / "Emission_Example" / "HAT-P-32b" / "kta_temp"
DEFAULT_CIA_FILE: Path | None = None
DEFAULT_FASTCHEM_PATH = Path.home() / "Dropbox" / "fastchem"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_emission_rt_benchmark"

R_SUN_M = 6.957e8
R_JUP_M = 7.1492e7
M_JUP_KG = 1.898e27
GRAVITATIONAL_CONSTANT = 6.67430e-11
AU_M = 1.495978707e11

DEFAULT_SPECIES = ("H2O",)
DEFAULT_FASTCHEM_ACTIVE_SPECIES = ("H2O", "CO2", "CO", "CH4", "NH3", "HCN")
DEFAULT_FASTCHEM_SPECIES = ("H2O1", "C1O2", "C1O1", "C1H4", "H3N1", "C1H1N1_1", "H2", "He")
DEFAULT_FASTCHEM_LABELS = ("H2O", "CO2", "CO", "CH4", "NH3", "HCN", "H2", "He")
DEFAULT_ACTIVE_VMR = {
    "H2O": 1.0e-2,
    "CO": 1.0e-4,
    "CO2": 1.0e-5,
    "CH4": 1.0e-6,
    "NH3": 1.0e-7,
    "HCN": 1.0e-7,
}
DEFAULT_H2_FRACTION_OF_BACKGROUND = 0.8547
DEFAULT_FASTCHEM_METALLICITY = 0.0
DEFAULT_FASTCHEM_CTOO = 0.55
DEFAULT_PLANET_RADIUS_M = 1.98 * R_JUP_M
DEFAULT_PLANET_MASS_KG = 0.68 * M_JUP_KG
DEFAULT_STAR_RADIUS_M = 1.32 * R_SUN_M
DEFAULT_STAR_TEMPERATURE_K = 6001.0
DEFAULT_SEMI_MAJOR_AXIS_M = 0.034 * AU_M
DEFAULT_REFERENCE_RADIUS_PRESSURE_BAR = 100.0
DEFAULT_REFERENCE_RADIUS_PRESSURE_PA = DEFAULT_REFERENCE_RADIUS_PRESSURE_BAR * 101325.0
DEFAULT_GRAVITY_M_S2 = GRAVITATIONAL_CONSTANT * DEFAULT_PLANET_MASS_KG / DEFAULT_PLANET_RADIUS_M**2
RUNTIME_NONFINITE_FILL_VALUE = 1.0e-300

MISSING_PHYSICS = (
    "scattering source-function treatment",
    "cloud and aerosol opacity/scattering",
    "benchmark hydrostatic path-geometry parity",
    "original 100-level pressure grid parity",
)


def main() -> dict[str, object]:
    """Run the local HAT-P-32b cloud-free emission benchmark."""

    benchmark_csv = _path_from_env("HAT_P_32B_EMISSION_CSV", DEFAULT_BENCHMARK_CSV)
    pt_csv = _path_from_env("HAT_P_32B_PT_CSV", DEFAULT_PT_CSV)
    kta_dir = _path_from_env("HAT_P_32B_KTA_DIR", DEFAULT_KTA_DIR)
    cia_file = _optional_path_from_env("HAT_P_32B_CIA_FILE", DEFAULT_CIA_FILE)
    chemistry_mode = os.environ.get("ROBERT_HAT_P_32B_CHEMISTRY", "fastchem").strip().lower()
    if chemistry_mode not in {"fastchem", "fixed"}:
        raise ValueError("ROBERT_HAT_P_32B_CHEMISTRY must be 'fastchem' or 'fixed'")
    species = _species_from_env(kta_dir, chemistry_mode=chemistry_mode)
    species_vmr = _species_vmr(species) if chemistry_mode == "fixed" else {}
    fastchem_path = _path_from_env("ROBERT_FASTCHEM_PATH", DEFAULT_FASTCHEM_PATH)
    fastchem_parameters = {
        "metallicity": float(
            os.environ.get("ROBERT_HAT_P_32B_FASTCHEM_METALLICITY", str(DEFAULT_FASTCHEM_METALLICITY))
        ),
        "CtoO": float(os.environ.get("ROBERT_HAT_P_32B_FASTCHEM_CTOO", str(DEFAULT_FASTCHEM_CTOO))),
    }
    gas_combination = os.environ.get("ROBERT_HAT_P_32B_GAS_COMBINATION", "random_overlap")
    include_cia = _env_bool("ROBERT_HAT_P_32B_INCLUDE_CIA", cia_file is not None)
    include_rayleigh = _env_bool("ROBERT_HAT_P_32B_INCLUDE_RAYLEIGH", True)
    include_scattering_source = _env_bool("ROBERT_HAT_P_32B_INCLUDE_SCATTERING_SOURCE", False)
    path_geometry_mode = os.environ.get("ROBERT_HAT_P_32B_PATH_GEOMETRY", "plane_parallel").strip().lower()
    if path_geometry_mode not in {"hydrostatic_spherical", "spherical", "none", "plane_parallel"}:
        raise ValueError(
            "ROBERT_HAT_P_32B_PATH_GEOMETRY must be 'hydrostatic_spherical' or 'plane_parallel'"
        )
    eclipse_radius_mode = os.environ.get("ROBERT_HAT_P_32B_ECLIPSE_RADIUS", "reference").strip().lower()
    if eclipse_radius_mode not in {"reference", "top", "bottom"}:
        raise ValueError("ROBERT_HAT_P_32B_ECLIPSE_RADIUS must be 'reference', 'top', or 'bottom'")

    for path, label in (
        (benchmark_csv, "benchmark CSV"),
        (pt_csv, "P-T CSV"),
        (kta_dir, "k-table directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"HAT-P-32b {label} was not found: {path}")

    benchmark = load_emission_benchmark_csv(benchmark_csv, name="HAT-P-32b emission R1000")
    kta_paths = _species_paths(kta_dir, species)
    n_mu = int(os.environ.get("ROBERT_HAT_P_32B_RT_NMU", "4"))
    geometry_mode = os.environ.get("ROBERT_HAT_P_32B_RT_GEOMETRY", "legendre_disk").strip().lower()
    default_phase = "180.0" if include_scattering_source else "0.0"
    phase_angle_deg = float(os.environ.get("ROBERT_HAT_P_32B_PHASE_DEG", default_phase))
    if include_scattering_source and geometry_mode in {
        "legendre",
        "legendre_disk",
        "gauss_legendre",
        "gauss_legendre_disk",
    }:
        geometry_mode = "lobatto_phase"
    geometry = _disc_geometry_from_env(
        geometry_mode=geometry_mode,
        n_mu=n_mu,
        phase_angle_deg=phase_angle_deg,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Benchmark: HAT-P-32b cloud-free emission RT")
    print(f"Benchmark CSV: {benchmark_csv}")
    print(f"P-T CSV: {pt_csv}")
    print(f"K-tables: {', '.join(str(path) for path in kta_paths.values())}")
    if chemistry_mode == "fastchem":
        print(
            "Chemistry: FastChem "
            f"(path={fastchem_path}, metallicity={fastchem_parameters['metallicity']}, "
            f"C/O={fastchem_parameters['CtoO']})"
        )
    else:
        print(f"Chemistry: fixed VMRs {species_vmr}")
    print(f"Opacity species: {species}, gas_combination={gas_combination}, n_mu={n_mu}")
    print(
        "Geometry: "
        f"{geometry.name} ({geometry.quadrature}, n_points={geometry.n_points}, "
        f"phase={geometry.phase_angle_deg})"
    )
    print(f"CIA: {'on' if include_cia else 'off'} ({cia_file if cia_file is not None else 'not configured'})")
    print(f"Rayleigh: {'on' if include_rayleigh else 'off'}")
    print(f"Single-scattering source: {'on' if include_scattering_source else 'off'}")
    print(f"Path geometry: {path_geometry_mode}, eclipse radius={eclipse_radius_mode}")

    start = perf_counter()
    provider = CorrelatedKOpacityProvider.from_kta_paths(
        kta_paths,
        interpolation="log_pressure_temperature_log_k",
        nonfinite_policy="floor",
        nonfinite_fill_value=RUNTIME_NONFINITE_FILL_VALUE,
    )
    table = provider.tables[species[0]]
    pressure_grid = _pressure_grid_from_centers(table.pressure_bar)
    spectral_grid = SpectralGrid.from_array(
        benchmark.wavelength_micron,
        unit="micron",
        role="opacity",
        name="HAT-P-32b emission R1000",
    )
    profile = TabulatedTemperatureProfile.from_csv(
        pt_csv,
        name="HAT-P-32b external PT",
    )
    temperature = profile.evaluate({}, pressure_grid)
    if chemistry_mode == "fastchem":
        chemistry = FastChemEquilibriumChemistry(
            fastchem_path=fastchem_path,
            fastchem_species=DEFAULT_FASTCHEM_SPECIES,
            labels=DEFAULT_FASTCHEM_LABELS,
            metadata={"source": "HAT-P-32b config_emission_clean.py"},
        )
        composition = chemistry.evaluate(fastchem_parameters, pressure_grid, temperature)
        mmw_normalization = "raw_sum"
    else:
        composition = _fixed_composition(pressure_grid, species_vmr)
        mmw_normalization = "require"
    mean_molecular_weight = CompositionMeanMolecularWeight(normalization=mmw_normalization).evaluate(
        composition,
        pressure_grid,
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature,
        composition=composition,
        mean_molecular_weight=mean_molecular_weight,
        metadata={
            "source": "HAT-P-32b local benchmark",
            "chemistry": chemistry_mode,
            "mmw_normalization": mmw_normalization,
        },
    )
    prepared = provider.prepare(spectral_grid, pressure_grid, species=species)
    evaluated = provider.evaluate(atmosphere, prepared)
    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=DEFAULT_GRAVITY_M_S2,
        gas_combination=gas_combination,
    )
    path_geometry = None
    if path_geometry_mode in {"hydrostatic_spherical", "spherical"}:
        path_geometry = hydrostatic_path_geometry(
            atmosphere,
            gravity_m_s2=DEFAULT_GRAVITY_M_S2,
            reference_radius_m=DEFAULT_PLANET_RADIUS_M,
            reference_pressure=DEFAULT_REFERENCE_RADIUS_PRESSURE_BAR,
            reference_pressure_unit="bar",
        )
        print(
            "Hydrostatic radii: "
            f"bottom={path_geometry.bottom_radius_m:.3e} m, "
            f"reference={path_geometry.reference_radius_m:.3e} m at "
            f"{DEFAULT_REFERENCE_RADIUS_PRESSURE_BAR:g} bar, "
            f"top={path_geometry.top_radius_m:.3e} m"
        )
    planet_radius_for_depth = _planet_radius_for_eclipse_depth(
        path_geometry=path_geometry,
        eclipse_radius_mode=eclipse_radius_mode,
    )
    additional_optical_depths = []
    if include_cia:
        if cia_file is None:
            raise FileNotFoundError("Set HAT_P_32B_CIA_FILE to enable the HAT-P-32b CIA benchmark term")
        if not cia_file.exists():
            raise FileNotFoundError(f"HAT-P-32b CIA file was not found: {cia_file}")
        cia_table = read_cia_table(cia_file)
        additional_optical_depths.append(cia_optical_depth(gas_tau, cia_table))
    if include_rayleigh:
        additional_optical_depths.append(rayleigh_scattering_optical_depth(gas_tau))
    scattering_source = None
    if include_scattering_source:
        scattering_source = SingleScatteringSource(
            stellar_beam=DirectStellarBeam.blackbody(
                spectral_grid,
                star_temperature_k=DEFAULT_STAR_TEMPERATURE_K,
                star_radius_m=DEFAULT_STAR_RADIUS_M,
                semi_major_axis_m=DEFAULT_SEMI_MAJOR_AXIS_M,
            ),
            phase_function=os.environ.get("ROBERT_HAT_P_32B_SCATTERING_PHASE_FUNCTION", "rayleigh"),
        )
    result = solve_clear_sky_emission(
        gas_tau,
        geometry=geometry,
        additional_optical_depths=additional_optical_depths,
        path_geometry=path_geometry,
        scattering_source=scattering_source,
        planet_radius_m=planet_radius_for_depth,
        star_radius_m=DEFAULT_STAR_RADIUS_M,
        star_temperature_k=DEFAULT_STAR_TEMPERATURE_K,
    )
    runtime_s = perf_counter() - start
    if result.eclipse_depth is None:
        raise RuntimeError("cloud-free emission result did not include eclipse depth")

    comparison = _comparison_metrics(
        model=result.eclipse_depth.values,
        benchmark=benchmark.eclipse_depth,
    )
    output_suffix = _output_suffix(
        geometry_mode=geometry_mode,
        path_geometry_mode=path_geometry_mode,
        eclipse_radius_mode=eclipse_radius_mode,
        include_scattering_source=include_scattering_source,
    )
    plot_path = _plot_benchmark(
        benchmark,
        result,
        temperature,
        pressure_grid,
        output_suffix=output_suffix,
    )
    geometry_plot_path = _plot_geometry(result, output_suffix=output_suffix)
    summary_path = OUTPUT_DIR / f"hat_p_32b_emission_rt_benchmark{output_suffix}_summary.json"
    summary = {
        "benchmark": "HAT-P-32b cloud-free emission RT",
        "status": "diagnostic_not_strict",
        "runtime_s": runtime_s,
        "benchmark_csv": str(benchmark_csv),
        "pt_csv": str(pt_csv),
        "kta_paths": {item: str(path) for item, path in kta_paths.items()},
        "cia_file": str(cia_file) if include_cia else None,
        "chemistry_mode": chemistry_mode,
        "fastchem_path": str(fastchem_path) if chemistry_mode == "fastchem" else None,
        "fastchem_parameters": fastchem_parameters if chemistry_mode == "fastchem" else None,
        "species": list(species),
        "active_vmr": species_vmr,
        "composition_summary": _composition_summary(composition),
        "mean_molecular_weight_summary": {
            "min_amu": float(np.min(mean_molecular_weight)),
            "median_amu": float(np.median(mean_molecular_weight)),
            "max_amu": float(np.max(mean_molecular_weight)),
            "normalization": mmw_normalization,
        },
        "gas_combination": gas_tau.metadata["gas_combination"],
        "additional_optical_depths": _optical_depth_summary(additional_optical_depths),
        "single_scattering_source": _scattering_source_summary(scattering_source),
        "n_layers": atmosphere.n_layers,
        "n_wavelength": benchmark.n_points,
        "n_mu": n_mu,
        "geometry": _geometry_summary(result.geometry),
        "path_geometry": _path_geometry_summary(path_geometry),
        "eclipse_radius": {
            "mode": eclipse_radius_mode,
            "radius_m": planet_radius_for_depth,
        },
        "planet_radius_m": DEFAULT_PLANET_RADIUS_M,
        "planet_mass_kg": DEFAULT_PLANET_MASS_KG,
        "semi_major_axis_m": DEFAULT_SEMI_MAJOR_AXIS_M,
        "star_radius_m": DEFAULT_STAR_RADIUS_M,
        "star_temperature_k": DEFAULT_STAR_TEMPERATURE_K,
        "gravity_m_s2": DEFAULT_GRAVITY_M_S2,
        "system_units": {
            "planet_radius": "1.98 R_JUP_E = 1.98 * 7.1492e7 m",
            "planet_mass": "0.68 M_JUP = 0.68 * 1.898e27 kg",
            "semi_major_axis": "0.034 AU",
            "star_radius": "1.32 R_SUN = 1.32 * 6.957e8 m",
            "gravity": "G * M_plt / R_plt^2",
        },
        "reference_radius_pressure": {
            "pref_bar_config": DEFAULT_REFERENCE_RADIUS_PRESSURE_BAR,
            "pref_pa_benchmark_resolution": DEFAULT_REFERENCE_RADIUS_PRESSURE_PA,
            "used_by_current_robert_emission_solver": path_geometry is not None,
            "note": "ROBERT anchors the hydrostatic radius grid at this pressure when hydrostatic spherical path geometry is enabled.",
        },
        "solver": dict(result.metadata),
        "remaining_benchmark_physics_gaps": _missing_physics(include_scattering_source),
        "comparison": comparison,
        "outputs": {
            "summary_json": str(summary_path),
            "plot_png": str(plot_path),
            "geometry_plot_png": str(geometry_plot_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ROBERT median depth: {comparison['model_median_ppm']:.2f} ppm")
    print(f"Benchmark median depth: {comparison['benchmark_median_ppm']:.2f} ppm")
    print(f"Residual RMSE: {comparison['rmse_ppm']:.2f} ppm")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {geometry_plot_path}")
    return summary


def _disc_geometry_from_env(
    *,
    geometry_mode: str,
    n_mu: int,
    phase_angle_deg: float,
):
    if geometry_mode in {"legendre", "legendre_disk", "gauss_legendre", "gauss_legendre_disk"}:
        return gauss_legendre_disk_geometry(n_mu)
    if geometry_mode in {"lobatto", "lobatto_phase", "phase"}:
        return lobatto_phase_geometry(phase_angle_deg=phase_angle_deg, n_mu=n_mu)
    raise ValueError(
        "ROBERT_HAT_P_32B_RT_GEOMETRY must be 'legendre_disk' or 'lobatto_phase'"
    )


def _output_suffix(
    *,
    geometry_mode: str,
    path_geometry_mode: str,
    eclipse_radius_mode: str,
    include_scattering_source: bool,
) -> str:
    parts = []
    if path_geometry_mode in {"hydrostatic_spherical", "spherical"}:
        parts.append("hydrostatic_spherical")
    if eclipse_radius_mode != "reference":
        parts.append(f"{eclipse_radius_mode}_radius")
    if geometry_mode not in {"legendre", "legendre_disk", "gauss_legendre", "gauss_legendre_disk"}:
        parts.append(geometry_mode)
    if include_scattering_source:
        parts.append("single_scattering")
    return "" if not parts else "_" + "_".join(parts)


def _planet_radius_for_eclipse_depth(
    *,
    path_geometry,
    eclipse_radius_mode: str,
) -> float:
    if path_geometry is None:
        return DEFAULT_PLANET_RADIUS_M
    if eclipse_radius_mode == "reference":
        return float(path_geometry.reference_radius_m)
    if eclipse_radius_mode == "top":
        return float(path_geometry.top_radius_m)
    if eclipse_radius_mode == "bottom":
        return float(path_geometry.bottom_radius_m)
    raise ValueError("unsupported eclipse radius mode")


def _geometry_summary(geometry) -> dict[str, object]:
    if geometry is None:
        return {}
    mu = geometry.emission_angle_cosines
    weights = geometry.emission_angle_weights
    stellar_mu = geometry.stellar_mu
    finite_stellar_mu = stellar_mu[np.isfinite(stellar_mu)]
    return {
        "name": geometry.name,
        "quadrature": geometry.quadrature,
        "phase_angle_deg": geometry.phase_angle_deg,
        "n_points": geometry.n_points,
        "mu_min": float(np.min(mu)),
        "mu_max": float(np.max(mu)),
        "weight_sum": float(np.sum(weights)),
        "weight_max": float(np.max(weights)),
        "stellar_mu_min": float(np.min(finite_stellar_mu)) if finite_stellar_mu.size else None,
        "stellar_mu_max": float(np.max(finite_stellar_mu)) if finite_stellar_mu.size else None,
        "metadata": dict(geometry.metadata),
    }


def _path_geometry_summary(path_geometry) -> dict[str, object] | None:
    if path_geometry is None:
        return None
    return {
        "path_model": path_geometry.metadata.get("path_model", "hydrostatic_spherical_shell"),
        "reference_radius_m": path_geometry.reference_radius_m,
        "reference_pressure_pa": path_geometry.reference_pressure_pa,
        "top_radius_m": path_geometry.top_radius_m,
        "bottom_radius_m": path_geometry.bottom_radius_m,
        "atmosphere_thickness_m": path_geometry.top_radius_m - path_geometry.bottom_radius_m,
        "scale_height_min_m": float(np.min(path_geometry.scale_height_m)),
        "scale_height_median_m": float(np.median(path_geometry.scale_height_m)),
        "scale_height_max_m": float(np.max(path_geometry.scale_height_m)),
        "metadata": dict(path_geometry.metadata),
    }


def _scattering_source_summary(scattering_source) -> dict[str, object] | None:
    if scattering_source is None:
        return None
    return {
        "enabled": True,
        "name": scattering_source.name,
        "phase_function": scattering_source.phase_function,
        "stellar_beam_unit": scattering_source.stellar_beam.unit,
        "stellar_beam_metadata": dict(scattering_source.stellar_beam.metadata),
        "metadata": dict(scattering_source.metadata),
    }


def _missing_physics(include_scattering_source: bool) -> list[str]:
    missing = list(MISSING_PHYSICS)
    if include_scattering_source:
        missing = [item for item in missing if item != "scattering source-function treatment"]
        missing.insert(0, "multiple-scattering and cloud/surface scattering source-function treatment")
    return missing


def _path_from_env(name: str, default: Path) -> Path:
    configured = os.environ.get(name)
    if configured:
        return Path(configured).expanduser()
    return default


def _optional_path_from_env(name: str, default: Path | None) -> Path | None:
    configured = os.environ.get(name)
    if configured:
        return Path(configured).expanduser()
    return default


def _species_path(kta_dir: Path, species: str) -> Path:
    candidates = sorted(kta_dir.glob(f"{species}_*.kta"))
    if not candidates:
        raise FileNotFoundError(f"No {species} .kta file found in {kta_dir}")
    return candidates[0]


def _species_paths(kta_dir: Path, species: tuple[str, ...]) -> dict[str, Path]:
    return {item: _species_path(kta_dir, item) for item in species}


def _species_from_env(kta_dir: Path, *, chemistry_mode: str) -> tuple[str, ...]:
    configured = os.environ.get("ROBERT_HAT_P_32B_RT_SPECIES")
    if configured is None or not configured.strip():
        if chemistry_mode == "fastchem":
            available = {path.name.split("_", maxsplit=1)[0] for path in kta_dir.glob("*.kta")}
            return tuple(item for item in DEFAULT_FASTCHEM_ACTIVE_SPECIES if item in available)
        return DEFAULT_SPECIES
    normalized = configured.strip()
    if normalized.lower() == "all":
        species = tuple(sorted(path.name.split("_", maxsplit=1)[0] for path in kta_dir.glob("*.kta")))
    else:
        species = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not species:
        raise ValueError("ROBERT_HAT_P_32B_RT_SPECIES did not contain any species")
    return species


def _species_vmr(species: tuple[str, ...]) -> dict[str, float]:
    vmr = {}
    for item in species:
        default = DEFAULT_ACTIVE_VMR.get(item, 1.0e-12)
        value = float(os.environ.get(f"ROBERT_HAT_P_32B_{item}_VMR", str(default)))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{item} VMR must be finite and non-negative")
        vmr[item] = value
    total = sum(vmr.values())
    if total >= 1.0:
        raise ValueError("active gas VMRs must sum to less than one")
    return vmr


def _fixed_composition(
    pressure_grid: PressureGrid,
    species_vmr: dict[str, float],
) -> dict[str, np.ndarray]:
    background = 1.0 - sum(species_vmr.values())
    h2 = background * DEFAULT_H2_FRACTION_OF_BACKGROUND
    he = background * (1.0 - DEFAULT_H2_FRACTION_OF_BACKGROUND)
    composition = {
        "H2": np.full(pressure_grid.n_layers, h2),
        "He": np.full(pressure_grid.n_layers, he),
    }
    for species, vmr in species_vmr.items():
        composition[species] = np.full(pressure_grid.n_layers, vmr)
    return composition


def _composition_summary(composition: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        species: {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
        for species, values in composition.items()
    }


def _env_bool(name: str, default: bool) -> bool:
    configured = os.environ.get(name)
    if configured is None:
        return default
    return configured.strip().lower() not in {"0", "false", "no", "off"}


def _optical_depth_summary(optical_depths) -> dict[str, dict[str, object]]:
    summary = {}
    for optical_depth in optical_depths:
        summary[optical_depth.name] = {
            "kind": optical_depth.kind,
            "max_tau": float(np.max(optical_depth.tau)),
            "median_tau": float(np.median(optical_depth.tau)),
            "metadata": dict(optical_depth.metadata),
        }
    return summary


def _comparison_metrics(
    *,
    model: np.ndarray,
    benchmark: np.ndarray,
) -> dict[str, float]:
    model_ppm = np.asarray(model, dtype=float) * 1.0e6
    benchmark_ppm = np.asarray(benchmark, dtype=float) * 1.0e6
    residual = model_ppm - benchmark_ppm
    denominator = np.maximum(np.abs(benchmark_ppm), 1.0e-12)
    relative = residual / denominator
    return {
        "model_median_ppm": float(np.median(model_ppm)),
        "benchmark_median_ppm": float(np.median(benchmark_ppm)),
        "residual_median_ppm": float(np.median(residual)),
        "residual_max_abs_ppm": float(np.max(np.abs(residual))),
        "residual_median_abs_ppm": float(np.median(np.abs(residual))),
        "rmse_ppm": float(np.sqrt(np.mean(residual**2))),
        "relative_median_abs": float(np.median(np.abs(relative))),
        "relative_p95_abs": float(np.percentile(np.abs(relative), 95.0)),
    }


def _plot_benchmark(
    benchmark,
    result,
    temperature: np.ndarray,
    pressure_grid: PressureGrid,
    *,
    output_suffix: str = "",
) -> Path:
    output_path = OUTPUT_DIR / f"hat_p_32b_clear_sky_emission_rt{output_suffix}.png"
    model_ppm = result.eclipse_depth.values * 1.0e6
    benchmark_ppm = benchmark.eclipse_depth * 1.0e6
    residual_ppm = model_ppm - benchmark_ppm
    contribution = result.normalized_layer_contribution()
    mean_contribution = np.mean(contribution, axis=1)
    if np.max(mean_contribution) > 0.0:
        mean_contribution = mean_contribution / np.max(mean_contribution)
    mean_scattering_contribution = None
    if result.scattering_layer_contribution_radiance is not None:
        mean_scattering_contribution = np.mean(result.scattering_layer_contribution_radiance, axis=1)
        if np.max(mean_scattering_contribution) > 0.0:
            mean_scattering_contribution = mean_scattering_contribution / np.max(mean_scattering_contribution)

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[2.0, 1.0])
    ax_spectrum = fig.add_subplot(grid[0, 0])
    ax_residual = fig.add_subplot(grid[1, 0], sharex=ax_spectrum)
    ax_profile = fig.add_subplot(grid[:, 1])

    ax_spectrum.plot(
        benchmark.wavelength_micron,
        benchmark_ppm,
        color="#111111",
        linewidth=1.6,
        label="External benchmark",
    )
    ax_spectrum.plot(
        result.eclipse_depth.spectral_grid.values,
        model_ppm,
        color="#d62728",
        linewidth=1.4,
        label="ROBERT cloud-free + CIA/Rayleigh",
    )
    for label, values in benchmark.references.items():
        ax_spectrum.plot(
            benchmark.wavelength_micron,
            values * 1.0e6,
            linestyle="--",
            linewidth=1.0,
            label=label.replace("_", " "),
        )
    ax_spectrum.set_ylabel("Eclipse Depth [ppm]")
    ax_spectrum.set_title("HAT-P-32b Emission Benchmark")
    ax_spectrum.grid(alpha=0.25)
    ax_spectrum.legend(frameon=False, fontsize=8.5)

    ax_residual.axhline(0.0, color="#333333", linewidth=1.0)
    ax_residual.plot(
        benchmark.wavelength_micron,
        residual_ppm,
        color="#1f77b4",
        linewidth=1.1,
    )
    ax_residual.set_xlabel("Wavelength [micron]")
    ax_residual.set_ylabel("ROBERT - benchmark [ppm]")
    ax_residual.grid(alpha=0.25)

    pressure = pressure_grid.centers
    ax_profile.plot(
        temperature,
        pressure,
        color="#111111",
        linewidth=1.8,
        label="P-T profile",
    )
    ax_profile.set_yscale("log")
    ax_profile.set_ylim(float(np.max(pressure)), float(np.min(pressure)))
    ax_profile.set_xlabel("Temperature [K]")
    ax_profile.set_ylabel("Pressure [bar]")
    ax_profile.set_title("Mean Emission Contribution")
    ax_profile.grid(alpha=0.25, which="both")
    ax_weight = ax_profile.twiny()
    ax_weight.fill_betweenx(
        pressure,
        0.0,
        mean_contribution,
        color="#2a9d8f",
        alpha=0.25,
    )
    ax_weight.plot(
        mean_contribution,
        pressure,
        color="#2a9d8f",
        linewidth=1.7,
        label="Mean contribution",
    )
    if mean_scattering_contribution is not None:
        ax_weight.plot(
            mean_scattering_contribution,
            pressure,
            color="#7b2cbf",
            linestyle=":",
            linewidth=1.7,
            label="Scattering contribution",
        )
    ax_weight.set_xlim(0.0, 1.05)
    ax_weight.set_xlabel("Normalized Contribution")
    lines = ax_profile.get_lines() + ax_weight.get_lines()
    labels = [line.get_label() for line in lines]
    ax_profile.legend(lines, labels, frameon=False, loc="lower right", fontsize=8.5)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_geometry(result, *, output_suffix: str = "") -> Path:
    output_path = OUTPUT_DIR / f"hat_p_32b_disc_geometry{output_suffix}.png"
    geometry = result.geometry
    if geometry is None:
        raise RuntimeError("emission result did not include disc geometry")

    mu = geometry.emission_angle_cosines
    weights = geometry.emission_angle_weights
    radii = geometry.projected_radius
    azimuth = geometry.projected_azimuth_deg
    stellar_mu = geometry.stellar_mu

    unique_mu = np.array(sorted({round(float(value), 12) for value in mu}), dtype=float)
    ring_weights = np.array([np.sum(weights[np.isclose(mu, value, rtol=0.0, atol=1.0e-12)]) for value in unique_mu])
    ring_radii = np.sqrt(np.maximum(0.0, 1.0 - unique_mu**2))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    ax_disc, ax_weights = axes

    outer = plt.Circle((0.0, 0.0), 1.0, fill=False, color="#222222", linewidth=1.4)
    ax_disc.add_patch(outer)
    finite_projected = np.isfinite(radii) & np.isfinite(azimuth)
    if np.count_nonzero(finite_projected) >= 2:
        x = radii[finite_projected] * np.cos(np.radians(azimuth[finite_projected]))
        y = radii[finite_projected] * np.sin(np.radians(azimuth[finite_projected]))
        color_values = stellar_mu[finite_projected]
        if not np.all(np.isfinite(color_values)):
            color_values = weights[finite_projected]
            color_label = "Point weight"
            cmap = "viridis"
        else:
            color_label = "Stellar mu"
            cmap = "coolwarm"
        sizes = 1200.0 * weights[finite_projected] / np.max(weights[finite_projected]) + 25.0
        scatter = ax_disc.scatter(
            x,
            y,
            s=sizes,
            c=color_values,
            cmap=cmap,
            edgecolor="#111111",
            linewidth=0.45,
            alpha=0.88,
        )
        fig.colorbar(scatter, ax=ax_disc, fraction=0.046, pad=0.04, label=color_label)
    else:
        for radius, weight in zip(ring_radii, ring_weights):
            ring = plt.Circle(
                (0.0, 0.0),
                float(radius),
                fill=False,
                color="#1f77b4",
                linewidth=0.8 + 5.0 * float(weight) / float(np.max(ring_weights)),
                alpha=0.55,
            )
            ax_disc.add_patch(ring)
        ax_disc.scatter([0.0], [0.0], s=36, color="#111111")

    ax_disc.set_xlim(-1.08, 1.08)
    ax_disc.set_ylim(-1.08, 1.08)
    ax_disc.set_aspect("equal", adjustable="box")
    ax_disc.set_xlabel("Projected x / R_p")
    ax_disc.set_ylabel("Projected y / R_p")
    ax_disc.set_title(f"{geometry.name}")
    ax_disc.grid(alpha=0.2)

    ax_weights.vlines(unique_mu, 0.0, ring_weights, color="#1f77b4", linewidth=1.7)
    ax_weights.scatter(unique_mu, ring_weights, color="#1f77b4", s=50, zorder=3)
    ax_weights.set_xlim(0.0, 1.04)
    ax_weights.set_ylim(0.0, max(1.0e-12, float(np.max(ring_weights))) * 1.18)
    ax_weights.set_xlabel("Emission mu")
    ax_weights.set_ylabel("Disc weight per mu ring")
    ax_weights.set_title(geometry.quadrature)
    ax_weights.grid(alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _pressure_grid_from_centers(centers: np.ndarray) -> PressureGrid:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 1:
        raise ValueError("centers must be a non-empty one-dimensional array")
    if centers.size == 1:
        edges = np.array([centers[0] / np.sqrt(10.0), centers[0] * np.sqrt(10.0)], dtype=float)
    else:
        log_centers = np.log(centers)
        inner_edges = 0.5 * (log_centers[:-1] + log_centers[1:])
        first_edge = log_centers[0] - (inner_edges[0] - log_centers[0])
        last_edge = log_centers[-1] + (log_centers[-1] - inner_edges[-1])
        edges = np.exp(np.concatenate(([first_edge], inner_edges, [last_edge])))
    return PressureGrid(edges=edges, centers=centers, unit="bar", name="HAT-P-32b emission RT benchmark")


if __name__ == "__main__":
    main()
