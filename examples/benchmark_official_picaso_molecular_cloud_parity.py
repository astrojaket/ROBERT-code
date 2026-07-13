"""Paper benchmark with independent ROBERT and official PICASO opacities.

The cloud contract is unchanged from the end-to-end MgSiO3 benchmark.  ROBERT
uses its ExoMolOP opacity-sampling tables while the external PICASO process
queries the official PICASO SQLite database.  No opacity or optical depth is
shared.  Emission and transmission are compared after independent evaluation
and integration onto common R~100 bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    OpacitySamplingProvider,
    PressureGrid,
    assemble_gas_optical_depth,
    cia_optical_depth,
    hydrostatic_path_geometry,
    load_nemesispy_cia_table,
    mie_efficiencies,
    planck_radiance_wavelength,
    rayleigh_scattering_optical_depth,
    solve_absorption_transmission,
)
from robert_exoplanets.rt.sh4 import solve_thermal_sh4_spectrum
from robert_exoplanets.diagnostics.benchmark_style import (
    PURPLE_PALETTE,
    REFERENCE_COLOR,
    ROBERT_COLOR,
)

try:
    from examples.benchmark_end_to_end_cloud_parity import (
        _integrate_population,
        _make_contract,
        _relative_metrics,
    )
except ModuleNotFoundError:  # Direct execution places examples/ on sys.path.
    from benchmark_end_to_end_cloud_parity import (
        _integrate_population,
        _make_contract,
        _relative_metrics,
    )


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_RUNNER = Path(__file__).with_name(
    "run_picaso_official_molecular_cloud_parity.py"
)
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "official_picaso_molecular_cloud_parity"
DEFAULT_DATABASE = (
    ROOT
    / "opacity_data"
    / "picaso_official"
    / "reference"
    / "opacities"
    / "opacities_0.3_15_R15000.db"
)
DEFAULT_REFERENCE = ROOT / "opacity_data" / "picaso_official" / "reference_v3_2"
EXOMOL = ROOT / "opacity_data" / "exomol_xsec"
SPECIES = ("H2O", "CO", "CO2", "CH4")
MOLECULAR_WEIGHTS = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.01528,
    "CO": 28.0101,
    "CO2": 44.0095,
    "CH4": 16.0425,
}
ZENODO_DOI = "10.5281/zenodo.14861730"
ARCHIVE_MD5 = "2f003f823d5f4b3f7a206d0ece9874b1"
VERIFIED_LINE_LIST_DOIS = {
    "12C-16O2__UCL-4000": "10.1093/mnras/staa1874",
}
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
ATOMIC_MASS_KG = 1.66053906660e-27


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picaso-python",
        type=Path,
        default=Path("/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python"),
    )
    parser.add_argument("--picaso-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--picaso-database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opacity-stride", type=int, default=1)
    parser.add_argument("--cloud-wavelengths", type=int, default=192)
    parser.add_argument("--output-bins", type=int, default=240)
    parser.add_argument("--layers", type=int, default=72)
    args = parser.parse_args()
    report = run(
        args.picaso_python,
        args.picaso_reference,
        args.picaso_database,
        args.output_dir,
        opacity_stride=args.opacity_stride,
        cloud_wavelengths=args.cloud_wavelengths,
        output_bins=args.output_bins,
        layers=args.layers,
    )
    print(json.dumps(report, indent=2))
    return report


def run(
    picaso_python: Path,
    picaso_reference: Path,
    picaso_database: Path,
    output_dir: Path,
    *,
    opacity_stride: int = 1,
    cloud_wavelengths: int = 192,
    output_bins: int = 240,
    layers: int = 72,
) -> dict[str, Any]:
    if opacity_stride < 1 or cloud_wavelengths < 32 or output_bins < 32 or layers < 16:
        raise ValueError("benchmark resolution is too small")
    for path in (picaso_python, picaso_reference / "config.json", picaso_database):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _make_science_contract(cloud_wavelengths, 36, layers)
    with tempfile.TemporaryDirectory(prefix="robert-picaso-contract-") as scratch:
        contract_path = Path(scratch) / "shared_science_physical_contract.npz"
        external_path = Path(scratch) / f"picaso_official_stride_{opacity_stride}.npz"
        np.savez_compressed(contract_path, **contract)

        started = perf_counter()
        robert = _evaluate_robert(contract, opacity_stride)
        robert_seconds = perf_counter() - started
        _run_external(
            picaso_python,
            picaso_reference,
            picaso_database,
            contract_path,
            external_path,
            opacity_stride,
        )
        with np.load(external_path, allow_pickle=False) as archive:
            picaso = {name: np.array(archive[name], copy=True) for name in archive.files}
    external_metadata = json.loads(str(picaso["metadata_json"]))

    edges = np.geomspace(1.0, 12.0, output_bins + 1)
    wavelength = np.sqrt(edges[:-1] * edges[1:])
    compact = _compact_outputs(edges, wavelength, robert, picaso, contract)
    metrics = _metrics(robert, picaso, compact, external_metadata)
    exomol_provenance = {
        name: {
            "path": str((EXOMOL / f"{name}.h5").relative_to(ROOT)),
            "sha256": _sha256(EXOMOL / f"{name}.h5"),
            "doi": VERIFIED_LINE_LIST_DOIS.get(
                str(robert["opacity_line_list"][index]),
                str(robert["opacity_doi"][index]),
            ),
            "embedded_doi": str(robert["opacity_doi"][index]),
            "line_list": robert["opacity_line_list"][index],
        }
        for index, name in enumerate(SPECIES)
    }
    report = {
        "schema_version": 1,
        "benchmark": "official_independent_PICASO_molecular_cloud_emission_transmission",
        "interpretation": (
            "This is a cross-database science comparison, not a same-opacity solver-parity test. "
            "Molecular spectral differences are expected and are reported rather than gated."
        ),
        "physical_contract": {
            "shared": [
                "pressure_temperature_and_exact_volume_mixing_ratios",
                "MgSiO3_refractive_index_and_particle_distribution",
                "vertical_condensate_profile_gravity_and_body_geometry",
            ],
            "not_shared": [
                "molecular_or_CIA_opacity",
                "gas_or_cloud_optical_depth",
                "Mie_efficiency",
                "emission_or_transmission_spectrum",
            ],
            "layers": layers,
            "cloud_wavelengths": cloud_wavelengths,
            "output_bins": output_bins,
            "gas_species": list(SPECIES),
            "h2_vmr": float(contract["h2_vmr"]),
            "he_vmr_range": [
                float(np.min(contract["he_vmr"])),
                float(np.max(contract["he_vmr"])),
            ],
            "radius_pressure_mapping": {
                "reference_pressure_bar": float(contract["reference_pressure_bar"]),
                "reference_radius_m": float(contract["planet_radius_m"]),
                "gravity_mode": "inverse_square_from_reference_radius",
                "robert_bottom_radius_m": float(robert["bottom_radius_m"]),
                "robert_top_radius_m": float(robert["top_radius_m"]),
                "picaso": external_metadata["radius_pressure_mapping"],
            },
        },
        "provenance": {
            "picaso_database": {
                "doi": ZENODO_DOI,
                "archive_md5": ARCHIVE_MD5,
                "extracted_size_bytes": picaso_database.stat().st_size,
                "extracted_sha256": external_metadata["opacity_database_sha256"],
                "database_metadata": {
                    "version": "default_3.3",
                    "resolution": 15000,
                    "wavelength_micron": [0.3, 15.0],
                },
                "picaso_version": external_metadata["picaso_version"],
                "virga_version": external_metadata["virga_version"],
            },
            "robert_exomolop": exomol_provenance,
            "robert_cia": "NemesisPy v1.0.1 HITRAN-2012 H2-H2/H2-He table",
        },
        "sampling": {
            "opacity_stride": opacity_stride,
            "robert_native_samples": int(robert["wavelength_micron"].size),
            "picaso_native_samples": int(picaso["wavelength_micron"].size),
            "effective_opacity_sampling_R": 15000.0 / opacity_stride,
            "comparison_output_R_approx": output_bins / np.log(12.0),
            "robert_seconds": robert_seconds,
            "picaso_timing_seconds": external_metadata["timing_seconds"],
        },
        "metrics": metrics,
    }
    (output_dir / "official_picaso_molecular_cloud_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "official_picaso_molecular_cloud_parity_compact.npz",
        wavelength_micron=wavelength,
        **compact,
    )
    _plot(
        output_dir / "official_picaso_molecular_cloud_parity.png",
        wavelength,
        compact,
    )
    return report


def _make_science_contract(n_wavelength: int, n_radius: int, n_layer: int):
    contract = _make_contract(n_wavelength, n_radius, n_layer)
    gas_vmr = np.array(contract["gas_mass_fractions"], copy=True)
    h2 = 0.84
    he = 1.0 - h2 - np.sum(gas_vmr, axis=1)
    if np.any(he <= 0.0):
        raise ValueError("gas volume mixing ratios do not leave a positive He background")
    contract["contract_schema_version"] = np.array(3)
    contract["gas_vmr"] = gas_vmr
    contract["h2_vmr"] = np.array(h2)
    contract["he_vmr"] = he
    contract["reference_pressure_bar"] = np.array(
        float(contract["pressure_edges_bar"][-1])
    )
    return contract


def _inverse_square_hydrostatic_profiles(contract):
    """Return PICASO-compatible radii and gravity from the shared anchor.

    PICASO anchors the supplied radius at ``p_reference`` and integrates
    upward with inverse-square gravity, using the temperature and mean
    molecular weight at the deeper bounding level of each layer.  Reproduce
    that physical convention independently so ROBERT's gas columns and
    spherical shells use the same pressure-radius mapping.
    """

    pressure = np.asarray(contract["pressure_edges_bar"], dtype=float)
    pressure_layer = np.sqrt(pressure[:-1] * pressure[1:])
    log_pressure = np.log(pressure)
    log_layer = np.log(pressure_layer)
    gas_vmr_level = np.stack(
        [
            np.interp(
                log_pressure,
                log_layer,
                contract["gas_vmr"][:, index],
                left=contract["gas_vmr"][0, index],
                right=contract["gas_vmr"][-1, index],
            )
            for index in range(len(SPECIES))
        ],
        axis=1,
    )
    h2_level = np.full(pressure.size, float(contract["h2_vmr"]))
    he_level = 1.0 - h2_level - np.sum(gas_vmr_level, axis=1)
    mean_molecular_weight_level = (
        h2_level * MOLECULAR_WEIGHTS["H2"]
        + he_level * MOLECULAR_WEIGHTS["He"]
        + sum(
            gas_vmr_level[:, index] * MOLECULAR_WEIGHTS[name]
            for index, name in enumerate(SPECIES)
        )
    )

    reference_pressure = float(contract["reference_pressure_bar"])
    if not np.isclose(reference_pressure, pressure[-1], rtol=0.0, atol=0.0):
        raise ValueError("the paper benchmark requires the reference pressure at the bottom edge")
    reference_radius = float(contract["planet_radius_m"])
    reference_gravity = float(contract["gravity_m_s2"])
    radius_level = np.empty(pressure.size, dtype=float)
    gravity_level = np.empty(pressure.size, dtype=float)
    radius_level[-1] = reference_radius
    gravity_level[-1] = reference_gravity
    for layer_index in range(pressure.size - 2, -1, -1):
        lower_level = layer_index + 1
        scale_height = (
            BOLTZMANN_CONSTANT_J_K
            * contract["temperature_level_k"][lower_level]
            / (
                mean_molecular_weight_level[lower_level]
                * ATOMIC_MASS_KG
                * gravity_level[lower_level]
            )
        )
        radius_level[layer_index] = radius_level[lower_level] + scale_height * np.log(
            pressure[lower_level] / pressure[layer_index]
        )
        gravity_level[layer_index] = reference_gravity * (
            reference_radius / radius_level[layer_index]
        ) ** 2

    return {
        "mean_molecular_weight_level": mean_molecular_weight_level,
        "radius_level_m": radius_level,
        "gravity_level_m_s2": gravity_level,
        "column_gravity_m_s2": 0.5 * (gravity_level[:-1] + gravity_level[1:]),
        "geometry_gravity_m_s2": gravity_level[1:],
    }


def _evaluate_robert(contract, opacity_stride: int):
    pressure_edges = contract["pressure_edges_bar"]
    pressure_layer = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_layer,
        unit="bar",
        name="official PICASO molecular parity",
    )
    composition = {
        name: contract["gas_vmr"][:, index] for index, name in enumerate(SPECIES)
    }
    composition["H2"] = np.full(pressure_layer.size, float(contract["h2_vmr"]))
    composition["He"] = contract["he_vmr"]
    mean_molecular_weight = sum(
        composition[name] * MOLECULAR_WEIGHTS[name] for name in composition
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=0.5 * (
            contract["temperature_level_k"][:-1]
            + contract["temperature_level_k"][1:]
        ),
        temperature_edges=contract["temperature_level_k"],
        composition=composition,
        mean_molecular_weight=mean_molecular_weight,
    )
    hydrostatic = _inverse_square_hydrostatic_profiles(contract)
    provider = OpacitySamplingProvider.from_exomol_paths(
        {name: EXOMOL / f"{name}.h5" for name in SPECIES},
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=False,
    )
    spectral_grid = provider.native_spectral_grid(
        sampling=opacity_stride,
        wavelength_bounds_micron=(1.0, 12.0),
        name=f"ExoMolOP stride {opacity_stride}",
    )
    prepared = provider.prepare(spectral_grid, pressure_grid, SPECIES)
    evaluated = provider.evaluate(atmosphere, prepared)
    gas = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=hydrostatic["column_gravity_m_s2"],
        retain_species_tau=True,
    )
    cia = cia_optical_depth(
        gas,
        load_nemesispy_cia_table(),
        temperature_extrapolation="clip",
        spectral_extrapolation="zero",
    )
    rayleigh = rayleigh_scattering_optical_depth(gas)
    mie = _robert_mie(contract)
    cloud_tau = _interpolate_cloud_tau(contract, mie, spectral_grid.values)
    cloud_scattering = cloud_tau * _interpolate_wavenumber(
        contract["wavelength_micron"],
        mie["single_scattering_albedo"],
        spectral_grid.values,
    )[None, :]
    cloud_asymmetry = _interpolate_wavenumber(
        contract["wavelength_micron"],
        mie["asymmetry_factor"],
        spectral_grid.values,
    )
    rayleigh_tau = np.asarray(rayleigh.tau)
    absorption_tau = gas.total_tau[:, :, 0] + np.asarray(cia.tau)
    cloud_free_tau = absorption_tau + rayleigh_tau
    cloud_free_omega = np.divide(
        rayleigh_tau, cloud_free_tau, out=np.zeros_like(cloud_free_tau), where=cloud_free_tau > 0.0
    )
    cloudy_tau = cloud_free_tau + cloud_tau
    scattering_tau = rayleigh_tau + cloud_scattering
    cloudy_omega = np.divide(
        scattering_tau,
        cloudy_tau,
        out=np.zeros_like(cloudy_tau),
        where=cloudy_tau > 0.0,
    )
    cloudy_g = np.divide(
        cloud_scattering * cloud_asymmetry[None, :],
        scattering_tau,
        out=np.zeros_like(scattering_tau),
        where=scattering_tau > 0.0,
    )
    level_planck = np.vstack(
        [
            planck_radiance_wavelength(spectral_grid.values, temperature)
            for temperature in contract["temperature_level_k"]
        ]
    )
    common = dict(
        emission_angle_cosines=contract["emission_mu"],
        emission_angle_weights=contract["emission_weights"],
        g_weights=np.array([1.0]),
        bottom_planck_radiance=level_planck[-1],
        delta_m=False,
        source_quadrature_order=8,
        backend="numba",
    )
    clear = solve_thermal_sh4_spectrum(
        cloud_free_tau[:, :, None],
        cloud_free_omega[:, :, None],
        np.zeros_like(cloud_free_tau[:, :, None]),
        level_planck,
        **common,
    )
    cloudy = solve_thermal_sh4_spectrum(
        cloudy_tau[:, :, None],
        cloudy_omega[:, :, None],
        cloudy_g[:, :, None],
        level_planck,
        **common,
    )
    stellar = planck_radiance_wavelength(
        spectral_grid.values, float(contract["star_temperature_k"])
    )
    area_ratio = (
        float(contract["planet_radius_m"]) / float(contract["star_radius_m"])
    ) ** 2
    geometry_atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=contract["temperature_level_k"][1:],
        temperature_edges=contract["temperature_level_k"],
        composition=composition,
        mean_molecular_weight=hydrostatic["mean_molecular_weight_level"][1:],
    )
    geometry = hydrostatic_path_geometry(
        geometry_atmosphere,
        gravity_m_s2=hydrostatic["geometry_gravity_m_s2"],
        reference_radius_m=float(contract["planet_radius_m"]),
        reference_pressure=float(contract["reference_pressure_bar"]),
        reference_pressure_unit="bar",
    )
    cloud_free_transmission = solve_absorption_transmission(
        gas,
        geometry,
        star_radius_m=float(contract["star_radius_m"]),
        additional_optical_depths=(cia, rayleigh),
        impact_quadrature_order=8,
    )
    cloudy_transmission = solve_absorption_transmission(
        gas,
        geometry,
        star_radius_m=float(contract["star_radius_m"]),
        additional_optical_depths=(cia, rayleigh, cloud_tau),
        impact_quadrature_order=8,
    )
    return {
        "wavelength_micron": spectral_grid.values,
        "mie_qext": mie["mie_qext"],
        "mie_qsca": mie["mie_qsca"],
        "mie_g_qsca": mie["mie_g_qsca"],
        "mass_extinction_m2_kg": mie["mass_extinction_m2_kg"],
        "single_scattering_albedo": mie["single_scattering_albedo"],
        "asymmetry_factor": mie["asymmetry_factor"],
        "molecular_tau_by_species": gas.species_tau[:, :, :, 0],
        "continuum_tau": np.asarray(cia.tau),
        "gas_tau": absorption_tau,
        "cloud_tau": cloud_tau,
        "cloud_free_eclipse_depth": clear.radiance / stellar * area_ratio,
        "cloudy_eclipse_depth": cloudy.radiance / stellar * area_ratio,
        "cloud_free_transit_depth": np.asarray(cloud_free_transmission.transit_depth.values),
        "cloudy_transit_depth": np.asarray(cloudy_transmission.transit_depth.values),
        "bottom_radius_m": np.array(geometry.bottom_radius_m),
        "top_radius_m": np.array(geometry.top_radius_m),
        "opacity_doi": np.array([provider.tables[name].metadata["doi"] for name in SPECIES]),
        "opacity_line_list": np.array(
            [provider.tables[name].metadata["line_list"] for name in SPECIES]
        ),
    }


def _robert_mie(contract):
    wavelength = contract["wavelength_micron"]
    radius = contract["radius_cm"]
    upper = contract["radius_upper_cm"]
    qext = np.zeros((wavelength.size, radius.size))
    qsca = np.zeros_like(qext)
    g_qsca = np.zeros_like(qext)
    for iw, (wave, real, imaginary) in enumerate(
        zip(
            wavelength,
            contract["refractive_index_n"],
            contract["refractive_index_k"],
            strict=True,
        )
    ):
        for ir in range(radius.size):
            lower = radius[0] if ir == 0 else upper[ir - 1]
            values = []
            for subradius in np.linspace(lower, upper[ir], 6):
                size = 2.0 * np.pi * subradius * 1.0e4 / wave
                ext, sca, asymmetry = mie_efficiencies(
                    size, complex(real, imaginary)
                )
                values.append((ext, sca, asymmetry * sca))
            qext[iw, ir], qsca[iw, ir], g_qsca[iw, ir] = np.mean(values, axis=0)
    population = _integrate_population(
        qext,
        qsca,
        g_qsca,
        radius,
        contract["radius_number_weights"],
        float(contract["particle_density_kg_m3"]),
    )
    return {"mie_qext": qext, "mie_qsca": qsca, "mie_g_qsca": g_qsca, **population}


def _interpolate_cloud_tau(contract, mie, wavelength):
    layer_mass = (
        np.diff(contract["pressure_edges_bar"])
        * 1.0e5
        / float(contract["gravity_m_s2"])
    )
    extinction = _interpolate_wavenumber(
        contract["wavelength_micron"],
        mie["mass_extinction_m2_kg"],
        wavelength,
    )
    return (
        layer_mass[:, None]
        * contract["condensate_mass_fraction"][:, None]
        * extinction[None, :]
    )


def _interpolate_wavenumber(source_wavelength, source_values, target_wavelength):
    source_wavenumber = 1.0e4 / np.asarray(source_wavelength)
    order = np.argsort(source_wavenumber)
    return np.interp(
        1.0e4 / np.asarray(target_wavelength),
        source_wavenumber[order],
        np.asarray(source_values)[order],
    )


def _run_external(python, reference, database, contract, output, stride):
    environment = dict(os.environ)
    environment["picaso_refdata"] = str(reference.resolve())
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "picaso-mpl")
    environment["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "picaso-numba")
    subprocess.run(
        [
            str(python),
            str(EXTERNAL_RUNNER),
            str(contract),
            str(output),
            "--opacity-db",
            str(database),
            "--resample",
            str(stride),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def _compact_outputs(edges, wavelength, robert, picaso, contract):
    compact = {}
    for framework, values in (("robert", robert), ("picaso", picaso)):
        native = values["wavelength_micron"]
        stellar_flux = planck_radiance_wavelength(
            native, float(contract["star_temperature_k"])
        )
        for name in ("cloud_free_eclipse_depth", "cloudy_eclipse_depth"):
            compact[f"{framework}_{name}"] = _bin_ratio(
                native, values[name] * stellar_flux, stellar_flux, edges
            )
        for name in ("cloud_free_transit_depth", "cloudy_transit_depth"):
            compact[f"{framework}_{name}"] = _bin_mean(native, values[name], edges)
        total_molecular = np.sum(values["molecular_tau_by_species"], axis=(0, 1))
        compact[f"{framework}_molecular_column_tau"] = _bin_mean(
            native, total_molecular, edges
        )
        for index, species in enumerate(SPECIES):
            column = np.sum(values["molecular_tau_by_species"][index], axis=0)
            compact[f"{framework}_{species}_column_tau"] = _bin_mean(
                native, column, edges
            )
        compact[f"{framework}_continuum_column_tau"] = _bin_mean(
            native, np.sum(values["continuum_tau"], axis=0), edges
        )
    return compact


def _bin_mean(x, y, edges):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    result = np.empty(edges.size - 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (x > lower) & (x < upper)
        sample_x = np.concatenate(([lower], x[selected], [upper]))
        sample_y = np.concatenate(
            ([np.interp(lower, x, y)], y[selected], [np.interp(upper, x, y)])
        )
        result[index] = np.trapezoid(sample_y, sample_x) / (upper - lower)
    return result


def _bin_ratio(x, numerator, denominator, edges):
    binned_denominator = _bin_mean(x, denominator, edges)
    return np.divide(
        _bin_mean(x, numerator, edges),
        binned_denominator,
        out=np.zeros_like(binned_denominator),
        where=binned_denominator > 0.0,
    )


def _metrics(robert, picaso, compact, metadata):
    cloud_free_eclipse_difference = (
        compact["robert_cloud_free_eclipse_depth"]
        - compact["picaso_cloud_free_eclipse_depth"]
    ) * 1.0e6
    eclipse_difference = (
        compact["robert_cloudy_eclipse_depth"]
        - compact["picaso_cloudy_eclipse_depth"]
    ) * 1.0e6
    transit_difference = (
        compact["robert_cloudy_transit_depth"]
        - compact["picaso_cloudy_transit_depth"]
    ) * 1.0e6
    cloud_free_transit_difference = (
        compact["robert_cloud_free_transit_depth"]
        - compact["picaso_cloud_free_transit_depth"]
    ) * 1.0e6
    robert_cloud_effect = (
        compact["robert_cloudy_eclipse_depth"]
        - compact["robert_cloud_free_eclipse_depth"]
    ) * 1.0e6
    picaso_cloud_effect = (
        compact["picaso_cloudy_eclipse_depth"]
        - compact["picaso_cloud_free_eclipse_depth"]
    ) * 1.0e6
    robert_transit_cloud = (
        compact["robert_cloudy_transit_depth"]
        - compact["robert_cloud_free_transit_depth"]
    ) * 1.0e6
    picaso_transit_cloud = (
        compact["picaso_cloudy_transit_depth"]
        - compact["picaso_cloud_free_transit_depth"]
    ) * 1.0e6
    species = {}
    for name in SPECIES:
        left = np.maximum(compact[f"robert_{name}_column_tau"], 1.0e-300)
        right = np.maximum(compact[f"picaso_{name}_column_tau"], 1.0e-300)
        delta = np.log10(left) - np.log10(right)
        species[name] = {
            "rms_log10_tau_difference_dex": float(np.sqrt(np.mean(delta**2))),
            "median_log10_tau_ratio_dex": float(np.median(delta)),
            "max_abs_log10_tau_difference_dex": float(np.max(np.abs(delta))),
        }
    all_finite = bool(
        all(np.all(np.isfinite(values)) for values in compact.values())
    )
    official_database = (
        metadata["opacity_database_size_bytes"] == 7_344_152_576
        and metadata["native_wavelength_count"] > 100
    )
    cloud_mie = _relative_metrics(
        robert["mass_extinction_m2_kg"], picaso["mass_extinction_m2_kg"]
    )
    return {
        "cloud_mass_extinction": cloud_mie,
        "molecular_column_tau_by_species": species,
        "cloud_free_emission_difference_ppm": _summary(
            cloud_free_eclipse_difference
        ),
        "cloudy_emission_difference_ppm": _summary(eclipse_difference),
        "cloud_free_transmission_difference_ppm": _summary(cloud_free_transit_difference),
        "cloudy_transmission_difference_ppm": _summary(transit_difference),
        "robert_emission_cloud_effect_ppm": _summary(robert_cloud_effect),
        "picaso_emission_cloud_effect_ppm": _summary(picaso_cloud_effect),
        "emission_cloud_effect_disagreement_ppm": _summary(
            robert_cloud_effect - picaso_cloud_effect
        ),
        "robert_transmission_cloud_effect_ppm": _summary(robert_transit_cloud),
        "picaso_transmission_cloud_effect_ppm": _summary(picaso_transit_cloud),
        "transmission_cloud_effect_disagreement_ppm": _summary(
            robert_transit_cloud - picaso_transit_cloud
        ),
        "acceptance": {
            "official_database_verified": official_database,
            "all_outputs_finite": all_finite,
            "independent_cloud_mass_extinction_rms_lt_2e-5": (
                cloud_mie["rms_relative_difference"] < 2.0e-5
            ),
            "spectral_agreement_is_not_an_acceptance_gate": True,
            "all_pass": bool(
                official_database
                and all_finite
                and cloud_mie["rms_relative_difference"] < 2.0e-5
            ),
        },
    }


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
    }


def _plot(path, wavelength, data):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(
        wavelength,
        data["robert_cloudy_eclipse_depth"] * 1e6,
        color=ROBERT_COLOR,
        label="ROBERT",
    )
    axes[0].plot(
        wavelength,
        data["picaso_cloudy_eclipse_depth"] * 1e6,
        color=REFERENCE_COLOR,
        linestyle="--",
        label="PICASO",
    )
    axes[0].set_ylabel("Eclipse depth (ppm)")
    axes[0].legend()
    axes[1].plot(
        wavelength,
        data["robert_cloudy_transit_depth"] * 1e6,
        color=ROBERT_COLOR,
        label="ROBERT",
    )
    axes[1].plot(
        wavelength,
        data["picaso_cloudy_transit_depth"] * 1e6,
        color=REFERENCE_COLOR,
        linestyle="--",
        label="PICASO",
    )
    axes[1].set_ylabel("Transit depth (ppm)")
    for index, name in enumerate(SPECIES):
        axes[2].plot(
            wavelength,
            np.log10(np.maximum(data[f"robert_{name}_column_tau"], 1e-300)),
            label=f"ROBERT {name}",
            color=PURPLE_PALETTE[index],
        )
        axes[2].plot(
            wavelength,
            np.log10(np.maximum(data[f"picaso_{name}_column_tau"], 1e-300)),
            color=PURPLE_PALETTE[index],
            linestyle="--",
            label=f"PICASO {name}",
        )
    axes[2].set_ylabel("log10 vertical molecular tau")
    axes[2].set_xlabel("Wavelength (micron)")
    axes[2].legend(ncol=4, fontsize=8)
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
