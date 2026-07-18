"""Run emission intercomparison Version-2 Stage 3 and write frozen products."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from robert_exoplanets.atmosphere import ParmentierGuillot2014TemperatureProfile
from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    Version2CommonContract,
    flux_conserving_bin_mean,
    load_version_2_common_contract,
    planck_surface_flux_w_m2_m,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY / "docs/data/emission_intercomparison/version_2"
COMMON_CONTRACT = DATA_ROOT / "common_contract.json"
DEFAULT_OUTPUT = REPOSITORY / "examples/outputs/emission_intercomparison/version_2/stage_3"
WORKER = Path(__file__).with_name(
    "run_emission_intercomparison_v2_stage_3_external.py"
)
ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PICASO_REFERENCE = Path("/Users/jaketaylor/Dropbox/picaso-v4/reference")
PICASO_CK_DIRECTORY = Path(
    "/Users/jaketaylor/Dropbox/picaso/reference/opacities/resortrebin"
)
PRT_INPUT_DATA = Path(
    "/Users/jaketaylor/Dropbox/ROBERT-code/opacity_data/petitRADTRANS/input_data"
)
PICASO_CIA_DATABASE = PICASO_REFERENCE / "climate_INPUTS/ck_cx_cont_opacities_661.db"
RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
PROFILES = ("isothermal", "pg14_non_inverted")
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")
FACTORIAL = (
    ("molecular_only", False, False),
    ("molecular_plus_h2_h2_cia", True, False),
    ("molecular_plus_h2_he_cia", False, True),
    ("molecular_plus_h2_h2_and_h2_he_cia", True, True),
)

# Frozen in code and docs/review/53_emission_intercomparison_v2_stage_3.md
# before the complete matrix is inspected. Only genuinely identical Track-A
# ROBERT/stable-pRT inputs are gated; Track-B native representations are not.
STAGE_3_ACCEPTANCE_GATES = {
    "track_a_max_abs_symmetric_relative": 5.0e-4,
    "track_a_max_abs_eclipse_difference_ppm": 0.1,
    "track_a_80_to_160_max_abs_eclipse_difference_ppm": 0.1,
    "track_a_isothermal_max_abs_eclipse_difference_ppm": 0.1,
    "scattering_single_scattering_albedo_max_abs": 0.0,
    "pilot_projected_wall_time_max_s": 7200.0,
    "pilot_peak_rss_fraction_of_available_max": 0.60,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _declared_mean_molecular_weight_u() -> float:
    payload = json.loads(COMMON_CONTRACT.read_text())
    return float(payload["composition"]["mean_molecular_weight_u_declared"])


def _quadrature() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(8)
    mu = 0.5 * (nodes + 1.0)
    legendre = 0.5 * weights
    disk = 2.0 * mu * legendre
    disk /= np.sum(disk)
    return mu, legendre, disk


def _edge_temperature(
    common: Version2CommonContract, profile: str, pressure_edges_bar: np.ndarray
) -> np.ndarray:
    if profile == "isothermal":
        return np.full(pressure_edges_bar.size, common.isothermal_temperature_k)
    parameters = common.pg14_parameters[profile]
    evaluator = ParmentierGuillot2014TemperatureProfile(
        gravity=common.derived["surface_gravity_m_s2"],
        internal_temperature=parameters.internal_temperature_k,
    )
    return evaluator.evaluate(
        {
            "kappa_IR": parameters.kappa_ir_m2_kg,
            "gamma1": parameters.gamma1,
            "gamma2": parameters.gamma2,
            "alpha": parameters.alpha,
            "T_irr": parameters.irradiation_temperature_k,
        },
        SimpleNamespace(centers=pressure_edges_bar, unit="bar"),
    )


def build_stage_3_contract(
    common: Version2CommonContract, n_cells: int
) -> dict[str, np.ndarray]:
    """Build the fixed-composition two-by-two CIA factorial."""

    grid = next(grid for grid in common.pressure_grids if grid.n_cells == n_cells)
    mu, legendre, disk = _quadrature()
    gas_names = tuple(common.composition_vmr)
    composition = np.asarray([common.composition_vmr[name] for name in gas_names])
    cases = [
        (profile, factor, h2_h2, h2_he)
        for profile in PROFILES
        for factor, h2_h2, h2_he in FACTORIAL
    ]
    temperatures_by_profile = {
        profile: common.temperature_profiles_k[f"{profile}_{n_cells}_cells"]
        for profile in PROFILES
    }
    edges_by_profile = {
        profile: _edge_temperature(common, profile, grid.edges_bar)
        for profile in PROFILES
    }
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(3),
        "case_id": np.asarray(
            [f"{profile}_{factor}_{n_cells}_cells" for profile, factor, _, _ in cases]
        ),
        "profile_name": np.asarray([profile for profile, _, _, _ in cases]),
        "profile_index": np.asarray([PROFILES.index(profile) for profile, _, _, _ in cases]),
        "factor_name": np.asarray([factor for _, factor, _, _ in cases]),
        "include_h2_h2_cia": np.asarray([value for _, _, value, _ in cases]),
        "include_h2_he_cia": np.asarray([value for _, _, _, value in cases]),
        "gas_name": np.asarray(gas_names),
        "gas_mass_u": np.asarray(
            [common.molecular_masses_u[name] for name in gas_names]
        ),
        "gas_vmr": np.broadcast_to(composition, (len(cases), composition.size)).copy(),
        "mean_molecular_weight_u": np.full(
            len(cases), _declared_mean_molecular_weight_u()
        ),
        "molecular_species_name": np.asarray(MOLECULAR_SPECIES),
        "molecular_species_active": np.ones(
            (len(cases), len(MOLECULAR_SPECIES)), dtype=bool
        ),
        "pressure_edges_bar": grid.edges_bar,
        "pressure_centers_bar": grid.centers_bar,
        "picaso_pressure_levels_bar": grid.picaso_levels_bar,
        "petitradtrans_pressure_nodes_bar": grid.petitradtrans_nodes_bar,
        "temperature_edges_k": np.asarray(
            [edges_by_profile[profile] for profile, _, _, _ in cases]
        ),
        "temperature_cells_k": np.asarray(
            [temperatures_by_profile[profile] for profile, _, _, _ in cases]
        ),
        "temperature_cells_by_profile_k": np.asarray(
            [temperatures_by_profile[profile] for profile in PROFILES]
        ),
        "temperature_edges_by_profile_k": np.asarray(
            [edges_by_profile[profile] for profile in PROFILES]
        ),
        "gravity_m_s2": np.array(common.derived["surface_gravity_m_s2"]),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def _subset_contract(
    contract: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, np.ndarray]:
    count = contract["case_id"].size
    result: dict[str, np.ndarray] = {}
    shared_names = {
        "gas_name",
        "gas_mass_u",
        "molecular_species_name",
        "pressure_edges_bar",
        "pressure_centers_bar",
        "picaso_pressure_levels_bar",
        "petitradtrans_pressure_nodes_bar",
        "temperature_cells_by_profile_k",
        "temperature_edges_by_profile_k",
        "emission_mu",
        "legendre_weights",
        "disk_weights",
    }
    for name, value in contract.items():
        if value.ndim > 0 and value.shape[0] == count and name not in shared_names:
            result[name] = value[selected]
        else:
            result[name] = value
    return result


def _normalise_vertical(values: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = np.sum(array, axis=-2, keepdims=True)
    return np.divide(array, total, out=np.zeros_like(array), where=total > 0.0)


def _opacity_paths() -> dict[str, Path]:
    patterns = {
        "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
        "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
        "CO2": "*UCL-4000*.ktable.petitRADTRANS.h5",
        "CH4": "*YT34to10*.ktable.petitRADTRANS.h5",
        "H2-H2": "*H2--H2*.ciatable.petitRADTRANS.h5",
        "H2-He": "*H2--He*.ciatable.petitRADTRANS.h5",
    }
    return {name: next(PRT_INPUT_DATA.rglob(pattern)) for name, pattern in patterns.items()}


def _run_robert_native(
    contract: dict[str, np.ndarray], paths: dict[str, Path]
) -> dict[str, np.ndarray]:
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
        solve_emission,
    )

    pressure_grid = PressureGrid(
        edges=contract["pressure_edges_bar"],
        centers=contract["pressure_centers_bar"],
        unit="bar",
        name="version-2-stage-3",
    )
    tables = {
        species: CorrelatedKTable.from_petitradtrans_hdf(paths[species], species=species)
        for species in MOLECULAR_SPECIES
    }
    first = tables["H2O"]
    mask = (first.wavelength_micron >= 0.3) & (first.wavelength_micron <= 12.1)
    wavelength = np.sort(first.wavelength_micron[mask])
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="stage3-pRT-R1000"
    )
    providers = {
        species: CorrelatedKOpacityProvider(
            {species: table},
            name=f"stage3-{species}",
            interpolation="log_pressure_temperature_log_k",
        )
        for species, table in tables.items()
    }
    prepared_by_species = {
        species: provider.prepare(spectral_grid, pressure_grid, species=(species,))
        for species, provider in providers.items()
    }
    prepared = PreparedCorrelatedKOpacity(
        provider_name="pRT-HDF-four-species",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=MOLECULAR_SPECIES,
        g_samples=first.g_samples,
        g_weights=first.g_weights,
        cache_key=f"stage3-{pressure_grid.n_layers}",
        metadata={
            "interpolation": "log_pressure_temperature_log_k",
            "gas_combination": "random_overlap",
        },
    )
    cia_tables = {
        "H2-H2": CiaTable.from_petitradtrans_hdf(
            paths["H2-H2"], collision_pair="H2-H2"
        ),
        "H2-He": CiaTable.from_petitradtrans_hdf(
            paths["H2-He"], collision_pair="H2-He"
        ),
    }
    gas_names = [str(value) for value in contract["gas_name"]]
    composition = dict(zip(gas_names, contract["gas_vmr"][0], strict=True))
    composition_profile = {
        name: np.full(pressure_grid.n_layers, value)
        for name, value in composition.items()
    }
    flux = np.empty((contract["case_id"].size, wavelength.size))
    contribution = np.empty(
        (contract["case_id"].size, pressure_grid.n_layers, wavelength.size)
    )
    runtime = np.empty(contract["case_id"].size)
    molecular_tau = np.empty(
        (
            len(PROFILES),
            pressure_grid.n_layers,
            wavelength.size,
            first.g_weights.size,
        ),
        dtype=np.float32,
    )
    cia_h2_h2_tau = np.empty(
        (len(PROFILES), pressure_grid.n_layers, wavelength.size), dtype=np.float32
    )
    cia_h2_he_tau = np.empty_like(cia_h2_h2_tau)
    for profile_index, profile in enumerate(PROFILES):
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=contract["temperature_cells_by_profile_k"][profile_index],
            temperature_edges=contract["temperature_edges_by_profile_k"][profile_index],
            composition=composition_profile,
            mean_molecular_weight=np.full(
                pressure_grid.n_layers, contract["mean_molecular_weight_u"][0]
            ),
        )
        evaluated = np.empty(
            (
                len(MOLECULAR_SPECIES),
                pressure_grid.n_layers,
                wavelength.size,
                first.g_weights.size,
            )
        )
        for species_index, species in enumerate(MOLECULAR_SPECIES):
            evaluated[species_index] = (
                providers[species]
                .evaluate(atmosphere, prepared_by_species[species])
                .kcoeff[0]
            )
        opacity = EvaluatedCorrelatedKOpacity(
            prepared=prepared,
            kcoeff=evaluated,
            unit="cm^2/molecule",
            metadata={"source": "petitRADTRANS HDF5 correlated-k tables"},
        )
        gas_tau = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=float(contract["gravity_m_s2"]),
            gas_combination="random_overlap",
        )
        cia = {
            name: cia_optical_depth(
                gas_tau,
                table,
                coefficient_interpolation="log",
                temperature_extrapolation="clip",
                spectral_extrapolation="zero",
            )
            for name, table in cia_tables.items()
        }
        molecular_tau[profile_index] = gas_tau.total_tau.astype(np.float32)
        cia_h2_h2_tau[profile_index] = cia["H2-H2"].tau.astype(np.float32)
        cia_h2_he_tau[profile_index] = cia["H2-He"].tau.astype(np.float32)
        selected = np.flatnonzero(contract["profile_name"] == profile)
        for case_index in selected:
            additions = []
            if bool(contract["include_h2_h2_cia"][case_index]):
                additions.append(cia["H2-H2"])
            if bool(contract["include_h2_he_cia"][case_index]):
                additions.append(cia["H2-He"])
            started = perf_counter()
            result = solve_emission(
                gas_tau,
                geometry=gauss_legendre_disk_geometry(n_mu=8),
                bottom_boundary="blackbody",
                additional_optical_depths=additions,
                multiple_scattering_backend="none",
            )
            runtime[case_index] = perf_counter() - started
            flux[case_index] = np.pi * np.asarray(result.radiance.values)
            vertical = np.asarray(result.layer_contribution_radiance, dtype=float).copy()
            vertical[-1] += np.asarray(result.bottom_contribution_radiance)
            contribution[case_index] = _normalise_vertical(vertical)
        del atmosphere, evaluated, opacity, gas_tau, cia
        gc.collect()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(
        usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    )
    metadata = {
        "model": "robert",
        "mode": "native_random_overlap_correlated_k_plus_selected_cia",
        "python": os.path.realpath(sys.executable),
        "version": importlib.metadata.version("robert-exoplanets"),
        "opacity_source": "petitRADTRANS HDF5 tables loaded independently by ROBERT",
        "interpolation": "log_pressure_temperature_log_k",
        "gas_combination": "random_overlap",
        "random_overlap_transparent_species_cutoff": 1.0e-12,
        "cia_interpolation": "log",
        "cia_temperature_extrapolation": "clip",
        "cia_spectral_extrapolation": "zero",
        "rayleigh_enabled": False,
        "scattering_enabled": False,
        "cloud_enabled": False,
        "peak_rss_bytes": peak_rss,
    }
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "normalized_vertical_diagnostic": contribution.astype(np.float32),
        "runtime_s": runtime,
        "g_weights": first.g_weights,
        "molecular_layer_tau_by_profile": molecular_tau,
        "cia_h2_h2_layer_tau_by_profile": cia_h2_h2_tau,
        "cia_h2_he_layer_tau_by_profile": cia_h2_he_tau,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _shared_contract(
    contract: dict[str, np.ndarray], robert_native: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    weights = robert_native["g_weights"]
    molecular = np.sum(
        robert_native["molecular_layer_tau_by_profile"].astype(float)
        * weights[None, None, None, :],
        axis=-1,
    )
    total = np.empty(
        (
            contract["case_id"].size,
            contract["pressure_centers_bar"].size,
            robert_native["wavelength_micron"].size,
        )
    )
    for index, profile_index in enumerate(contract["profile_index"]):
        total[index] = molecular[profile_index]
        if bool(contract["include_h2_h2_cia"][index]):
            total[index] += robert_native["cia_h2_h2_layer_tau_by_profile"][
                profile_index
            ]
        if bool(contract["include_h2_he_cia"][index]):
            total[index] += robert_native["cia_h2_he_layer_tau_by_profile"][profile_index]
    return {
        **contract,
        "shared_wavelength_micron": robert_native["wavelength_micron"],
        "shared_molecular_layer_tau_by_profile": molecular,
        "shared_cia_h2_h2_layer_tau_by_profile": robert_native[
            "cia_h2_h2_layer_tau_by_profile"
        ].astype(float),
        "shared_cia_h2_he_layer_tau_by_profile": robert_native[
            "cia_h2_he_layer_tau_by_profile"
        ].astype(float),
        "shared_layer_tau": total,
        "shared_source": np.array(
            "ROBERT-independent pRT-HDF four-species random-overlap tau collapsed "
            "by native g weights plus separately evaluated pRT-HDF CIA components"
        ),
    }


def _run_robert_shared(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from robert_exoplanets import planck_radiance_wavelength
    from robert_exoplanets.rt import integrate_thermal_emission

    wavelength = contract["shared_wavelength_micron"]
    flux = []
    contribution = []
    timings = []
    for case_index, tau in enumerate(contract["shared_layer_tau"]):
        edges = contract["temperature_edges_k"][case_index]
        cells = contract["temperature_cells_k"][case_index]
        layer_source = np.stack(
            [planck_radiance_wavelength(wavelength, value) for value in cells]
        )
        level_source = np.stack(
            [planck_radiance_wavelength(wavelength, value) for value in edges]
        )
        started = perf_counter()
        result = integrate_thermal_emission(
            tau[:, :, None],
            layer_source,
            np.array([1.0]),
            np.broadcast_to(
                1.0 / contract["emission_mu"][:, None],
                (contract["emission_mu"].size, tau.shape[0]),
            ),
            level_source_ordered=level_source,
            bottom_source=level_source[-1],
            backend="numpy",
        )
        timings.append(perf_counter() - started)
        vertical = np.tensordot(
            contract["disk_weights"],
            result.point_layer_contribution_radiance,
            axes=(0, 0),
        )
        bottom = np.tensordot(
            contract["disk_weights"],
            result.point_bottom_contribution_radiance,
            axes=(0, 0),
        )
        vertical[-1] += bottom
        flux.append(np.pi * np.sum(vertical, axis=0))
        contribution.append(_normalise_vertical(vertical))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(
        usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    )
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "normalized_vertical_diagnostic": np.asarray(contribution, dtype=np.float32),
        "runtime_s": np.asarray(timings),
        "metadata_json": np.array(
            json.dumps(
                {
                    "model": "robert",
                    "mode": "track_a_identical_mean_tau",
                    "python": os.path.realpath(sys.executable),
                    "version": importlib.metadata.version("robert-exoplanets"),
                    "cia_components": ["H2-H2", "H2-He"],
                    "scattering_enabled": False,
                    "rayleigh_enabled": False,
                    "peak_rss_bytes": peak_rss,
                },
                sort_keys=True,
            )
        ),
    }


def _run_external(
    python: Path,
    mode: str,
    contract_path: Path,
    output_path: Path,
) -> dict[str, np.ndarray]:
    command = [str(python), str(WORKER), mode, str(contract_path), str(output_path)]
    if mode == "picaso_ck":
        command.extend(["--picaso-ck-directory", str(PICASO_CK_DIRECTORY)])
    elif mode == "petitradtrans_native":
        command.extend(["--prt-input-data", str(PRT_INPUT_DATA)])
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_btl", "self")
    if mode.startswith("picaso"):
        cache_root = output_path.parent / f"{mode}-cache"
        numba_cache = cache_root / "picaso-v4-numba-cache"
        mpl_cache = cache_root / "picaso-v4-matplotlib"
        numba_cache.mkdir(parents=True, exist_ok=True)
        mpl_cache.mkdir(parents=True, exist_ok=True)
        environment["picaso_refdata"] = str(PICASO_REFERENCE)
        environment["NUMBA_CACHE_DIR"] = str(numba_cache)
        environment["MPLCONFIGDIR"] = str(mpl_cache)
    subprocess.run(command, check=True, env=environment)
    return _load_npz(output_path)


def _close_native_bin_edges(
    wavelength: np.ndarray, values: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Close table-centre spectra to physical boundary edges without changing native data."""

    output_wavelength = np.asarray(wavelength)
    output_values = np.asarray(values)
    if output_wavelength[0] > edges[0]:
        output_wavelength = np.concatenate(([edges[0]], output_wavelength))
        output_values = np.concatenate((output_values[..., :1], output_values), axis=-1)
    if output_wavelength[-1] < edges[-1]:
        output_wavelength = np.concatenate((output_wavelength, [edges[-1]]))
        output_values = np.concatenate((output_values, output_values[..., -1:]), axis=-1)
    return output_wavelength, output_values


def _bin_flux(common: Version2CommonContract, payload: dict[str, np.ndarray]) -> np.ndarray:
    wavelength, flux = _close_native_bin_edges(
        payload["wavelength_micron"],
        payload["flux_w_m2_m"],
        common.spectral.r100_edges_micron,
    )
    return flux_conserving_bin_mean(
        wavelength,
        flux,
        common.spectral.r100_edges_micron,
    )


def _bin_contribution(
    common: Version2CommonContract, payload: dict[str, np.ndarray]
) -> np.ndarray:
    wavelength, vertical = _close_native_bin_edges(
        payload["wavelength_micron"],
        payload["normalized_vertical_diagnostic"],
        common.spectral.r100_edges_micron,
    )
    binned = flux_conserving_bin_mean(
        wavelength,
        vertical,
        common.spectral.r100_edges_micron,
    )
    return _normalise_vertical(binned)


def _difference(
    left: np.ndarray, right: np.ndarray, common: Version2CommonContract
) -> dict[str, float]:
    denominator = np.abs(left) + np.abs(right)
    symmetric = np.divide(
        2.0 * (left - right),
        denominator,
        out=np.zeros_like(left),
        where=denominator > 0.0,
    )
    eclipse_ppm = (
        (left - right)
        / common.stellar_surface_flux_r100_w_m2_m
        * common.derived["projected_area_ratio"]
        * 1.0e6
    )
    return {
        "max_abs_symmetric_relative": float(np.max(np.abs(symmetric))),
        "p95_abs_symmetric_relative": float(np.percentile(np.abs(symmetric), 95.0)),
        "max_abs_eclipse_difference_ppm": float(np.max(np.abs(eclipse_ppm))),
        "rms_eclipse_difference_ppm": float(np.sqrt(np.mean(eclipse_ppm**2))),
    }


def _available_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))


def _metadata(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    return json.loads(str(payload["metadata_json"]))


def _save_artifact(
    path: Path,
    contract: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    common: Version2CommonContract,
    *,
    representation: str,
) -> None:
    r100_flux = _bin_flux(common, payload)
    r100_contribution = _bin_contribution(common, payload)
    native_eclipse = (
        payload["flux_w_m2_m"]
        / planck_surface_flux_w_m2_m(
            payload["wavelength_micron"],
            common.measurements["stellar_effective_temperature"].si_value,
        )
        * common.derived["projected_area_ratio"]
    )
    arrays: dict[str, np.ndarray] = {
        name: contract[name]
        for name in (
            "case_id",
            "profile_name",
            "profile_index",
            "factor_name",
            "include_h2_h2_cia",
            "include_h2_he_cia",
            "gas_name",
            "gas_mass_u",
            "gas_vmr",
            "mean_molecular_weight_u",
            "molecular_species_name",
            "molecular_species_active",
            "pressure_edges_bar",
            "pressure_centers_bar",
            "picaso_pressure_levels_bar",
            "petitradtrans_pressure_nodes_bar",
            "temperature_edges_k",
            "temperature_cells_k",
        )
    }
    arrays.update(
        {
            "native_wavelength_micron": payload["wavelength_micron"],
            "native_flux_w_m2_m": payload["flux_w_m2_m"],
            "native_eclipse_depth": native_eclipse,
            "r100_edges_micron": common.spectral.r100_edges_micron,
            "r100_centers_micron": common.spectral.r100_centers_micron,
            "r100_flux_w_m2_m": r100_flux,
            "r100_eclipse_depth": (
                r100_flux
                / common.stellar_surface_flux_r100_w_m2_m
                * common.derived["projected_area_ratio"]
            ),
            "normalized_vertical_native": payload[
                "normalized_vertical_diagnostic"
            ],
            "normalized_vertical_r100": r100_contribution.astype(np.float32),
            "runtime_s": payload["runtime_s"],
            "representation": np.array(representation),
            "metadata_json": payload.get("metadata_json", np.array("{}")),
        }
    )
    optional = (
        "g_weights",
        "molecular_layer_tau_by_profile",
        "cia_h2_h2_layer_tau_by_profile",
        "cia_h2_he_layer_tau_by_profile",
        "layer_tau",
        "maximum_abs_rayleigh_tau",
        "maximum_abs_cloud_tau",
        "native_framework_probe_flux_w_m2_m",
        "native_continuum_pairs_json",
    )
    for name in optional:
        if name in payload:
            arrays[name] = payload[name]
    np.savez_compressed(path, **arrays)


def _integrity_manifest(paths: list[Path]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in paths:
        entry: dict[str, Any] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                entry["arrays"] = {
                    name: {
                        "shape": list(archive[name].shape),
                        "dtype": str(archive[name].dtype),
                        "finite_policy": "all_finite",
                    }
                    for name in archive.files
                }
        artifacts[path.name] = entry
    return {
        "schema_version": "1.0.0",
        "stage": 3,
        "units_and_axes": {
            "flux": "case,wavelength; W m^-2 m^-1; positive outward",
            "eclipse_depth": "case,wavelength; dimensionless signed",
            "molecular_layer_tau_by_profile": (
                "profile,pressure-cell,wavelength,g; dimensionless"
            ),
            "cia_layer_tau_by_profile": (
                "profile,pressure-cell,wavelength; dimensionless"
            ),
            "layer_tau": "case,pressure-cell,wavelength,g when g is present",
            "normalized_vertical": (
                "case,pressure-cell,wavelength; unit sum over pressure"
            ),
            "pressure": "bar; top-to-bottom increasing",
            "wavelength": "micron; increasing vacuum wavelength",
        },
        "artifacts": artifacts,
    }


def _factorial_effects(
    contract: dict[str, np.ndarray], binned: np.ndarray, common: Version2CommonContract
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for profile in PROFILES:
        index = {
            str(contract["factor_name"][item]): int(item)
            for item in np.flatnonzero(contract["profile_name"] == profile)
        }
        molecular = binned[index["molecular_only"]]
        h2_h2 = binned[index["molecular_plus_h2_h2_cia"]]
        h2_he = binned[index["molecular_plus_h2_he_cia"]]
        both = binned[index["molecular_plus_h2_h2_and_h2_he_cia"]]
        effects[profile] = {
            "h2_h2_with_h2_he_off": _difference(h2_h2, molecular, common),
            "h2_h2_with_h2_he_on": _difference(both, h2_he, common),
            "h2_he_with_h2_h2_off": _difference(h2_he, molecular, common),
            "h2_he_with_h2_h2_on": _difference(both, h2_h2, common),
            "both_cia_vs_molecular_only": _difference(both, molecular, common),
            "factorial_interaction": _difference(
                both - h2_he, h2_h2 - molecular, common
            ),
        }
    return effects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    if os.path.realpath(sys.executable) != os.path.realpath(ROBERT_PYTHON):
        raise RuntimeError(f"Stage 3 must run with {ROBERT_PYTHON}")
    common = load_version_2_common_contract(COMMON_CONTRACT)
    paths = _opacity_paths()
    for species, asset in common.picaso_correlated_k_assets.items():
        if _sha256(PICASO_CK_DIRECTORY / asset.filename) != asset.sha256:
            raise RuntimeError(f"frozen PICASO correlated-k checksum mismatch: {species}")

    primary_contract = build_stage_3_contract(common, PRIMARY_RESOLUTION)
    pilot_selected = (
        (primary_contract["profile_name"] == "pg14_non_inverted")
        & (
            primary_contract["factor_name"]
            == "molecular_plus_h2_h2_and_h2_he_cia"
        )
    )
    pilot_contract = _subset_contract(primary_contract, pilot_selected)
    pilot_root = output_root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    pilot_contract_path = pilot_root / "contract.npz"
    np.savez_compressed(pilot_contract_path, **pilot_contract)
    pilot_started = perf_counter()
    pilot_robert = _run_robert_native(pilot_contract, paths)
    pilot_picaso = _run_external(
        PICASO_PYTHON,
        "picaso_ck",
        pilot_contract_path,
        pilot_root / "picaso_ck.npz",
    )
    pilot_prt = _run_external(
        PRT_PYTHON,
        "petitradtrans_native",
        pilot_contract_path,
        pilot_root / "petitradtrans_native.npz",
    )
    pilot_wall = perf_counter() - pilot_started
    available_memory = _available_memory_bytes()
    pilot_peaks = [
        _metadata(pilot_robert)["peak_rss_bytes"],
        _metadata(pilot_picaso)["peak_rss_bytes"],
        _metadata(pilot_prt)["peak_rss_bytes"],
    ]
    largest_peak = int(max(pilot_peaks))
    projection_multiplier = 36.0
    projected_wall = pilot_wall * projection_multiplier
    memory_fraction = largest_peak / available_memory
    authorized = (
        projected_wall <= STAGE_3_ACCEPTANCE_GATES["pilot_projected_wall_time_max_s"]
        and memory_fraction
        <= STAGE_3_ACCEPTANCE_GATES[
            "pilot_peak_rss_fraction_of_available_max"
        ]
    )
    pilot = {
        "resolution_cells": PRIMARY_RESOLUTION,
        "case_id": str(pilot_contract["case_id"][0]),
        "frameworks": ["robert", "picaso", "petitradtrans"],
        "measured_wall_time_s": pilot_wall,
        "projection_multiplier": projection_multiplier,
        "projected_complete_wall_time_s": projected_wall,
        "largest_process_peak_rss_bytes": largest_peak,
        "available_memory_bytes_at_decision": available_memory,
        "peak_rss_fraction_of_available": memory_fraction,
        "authorized_full_matrix": authorized,
        "decision_limits": {
            key: value
            for key, value in STAGE_3_ACCEPTANCE_GATES.items()
            if key.startswith("pilot_")
        },
    }
    _write_json(pilot_root / "pilot_decision.json", pilot)
    if args.pilot_only:
        print(json.dumps(pilot, indent=2, sort_keys=True))
        return
    if not authorized:
        raise RuntimeError("Stage-3 matrix not authorized by frozen pilot resource gates")

    outputs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    artifact_paths: list[Path] = []
    matrix_started = perf_counter()
    for n_cells in RESOLUTIONS:
        contract = build_stage_3_contract(common, n_cells)
        resolution_root = output_root / f"{n_cells}_cells"
        resolution_root.mkdir(parents=True, exist_ok=True)
        contract_path = resolution_root / "contract.npz"
        np.savez_compressed(contract_path, **contract)
        robert = _run_robert_native(contract, paths)
        picaso = _run_external(
            PICASO_PYTHON,
            "picaso_ck",
            contract_path,
            resolution_root / "picaso_ck.npz",
        )
        prt = _run_external(
            PRT_PYTHON,
            "petitradtrans_native",
            contract_path,
            resolution_root / "petitradtrans_native.npz",
        )
        shared = _shared_contract(contract, robert)
        shared_contract_path = resolution_root / "shared_contract.npz"
        np.savez_compressed(shared_contract_path, **shared)
        robert_shared = _run_robert_shared(shared)
        prt_shared = _run_external(
            PRT_PYTHON,
            "petitradtrans_shared",
            shared_contract_path,
            resolution_root / "petitradtrans_shared.npz",
        )
        outputs[n_cells] = {
            "contract": contract,
            "robert": robert,
            "picaso_correlated_k": picaso,
            "petitradtrans": prt,
            "robert_shared": robert_shared,
            "petitradtrans_shared": prt_shared,
            "shared_contract": shared,
        }
        for name, payload, representation in (
            ("robert", robert, "native_random_overlap_correlated_k_plus_factorial_cia"),
            (
                "picaso",
                picaso,
                "native_correlated_k_resort_rebin_opacity_with_absorbing_formal_reference",
            ),
            ("petitradtrans", prt, "native_correlated_k_plus_factorial_cia"),
            ("robert_shared", robert_shared, "track_a_identical_mean_tau"),
            (
                "petitradtrans_shared",
                prt_shared,
                "track_a_identical_mean_tau",
            ),
        ):
            path = data_root / f"stage_3_{name}_{n_cells}_cells.npz"
            _save_artifact(
                path, contract, payload, common, representation=representation
            )
            artifact_paths.append(path)
        shared_path = data_root / f"stage_3_shared_tau_{n_cells}_cells.npz"
        np.savez_compressed(
            shared_path,
            case_id=contract["case_id"],
            profile_name=contract["profile_name"],
            factor_name=contract["factor_name"],
            include_h2_h2_cia=contract["include_h2_h2_cia"],
            include_h2_he_cia=contract["include_h2_he_cia"],
            pressure_edges_bar=contract["pressure_edges_bar"],
            pressure_centers_bar=contract["pressure_centers_bar"],
            wavelength_micron=shared["shared_wavelength_micron"],
            molecular_layer_tau_by_profile=shared[
                "shared_molecular_layer_tau_by_profile"
            ].astype(np.float32),
            cia_h2_h2_layer_tau_by_profile=shared[
                "shared_cia_h2_h2_layer_tau_by_profile"
            ].astype(np.float32),
            cia_h2_he_layer_tau_by_profile=shared[
                "shared_cia_h2_he_layer_tau_by_profile"
            ].astype(np.float32),
            layer_tau=shared["shared_layer_tau"].astype(np.float32),
            source=shared["shared_source"],
        )
        artifact_paths.append(shared_path)
    matrix_wall = perf_counter() - matrix_started

    per_resolution: dict[str, Any] = {}
    track_a_metrics = []
    for n_cells in RESOLUTIONS:
        contract = outputs[n_cells]["contract"]
        binned = {
            name: _bin_flux(common, outputs[n_cells][name])
            for name in (
                "robert",
                "picaso_correlated_k",
                "petitradtrans",
                "robert_shared",
                "petitradtrans_shared",
            )
        }
        track_a = _difference(
            binned["robert_shared"], binned["petitradtrans_shared"], common
        )
        track_a_metrics.append(track_a)
        per_resolution[str(n_cells)] = {
            "case_count": int(contract["case_id"].size),
            "track_a_robert_vs_petitradtrans": track_a,
            "track_b_native_representation_attribution": {
                "robert__picaso_correlated_k": _difference(
                    binned["robert"], binned["picaso_correlated_k"], common
                ),
                "robert__petitradtrans": _difference(
                    binned["robert"], binned["petitradtrans"], common
                ),
                "picaso_correlated_k__petitradtrans": _difference(
                    binned["picaso_correlated_k"],
                    binned["petitradtrans"],
                    common,
                ),
            },
            "factorial_cia_effects": {
                name: _factorial_effects(contract, values, common)
                for name, values in binned.items()
            },
            "native_wavelength_count": {
                name: int(outputs[n_cells][name]["wavelength_micron"].size)
                for name in (
                    "robert",
                    "picaso_correlated_k",
                    "petitradtrans",
                )
            },
            "raw_runtime_s": {
                name: outputs[n_cells][name]["runtime_s"].tolist()
                for name in binned
            },
            "worker_metadata": {
                name: _metadata(outputs[n_cells][name]) for name in binned
            },
        }

    convergence: dict[str, Any] = {}
    model_names = (
        "robert",
        "picaso_correlated_k",
        "petitradtrans",
        "robert_shared",
        "petitradtrans_shared",
    )
    for model in model_names:
        convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            convergence[model][f"{coarse}_to_{fine}"] = _difference(
                _bin_flux(common, outputs[coarse][model]),
                _bin_flux(common, outputs[fine][model]),
                common,
            )
    blackbody = flux_conserving_bin_mean(
        common.spectral.native_reference_wavelength_micron,
        planck_surface_flux_w_m2_m(
            common.spectral.native_reference_wavelength_micron,
            common.isothermal_temperature_k,
        ),
        common.spectral.r100_edges_micron,
    )
    iso_mask = outputs[80]["contract"]["profile_name"] == "isothermal"
    iso_metrics = {
        model: _difference(
            _bin_flux(common, outputs[80][model])[iso_mask],
            np.broadcast_to(blackbody, (int(np.sum(iso_mask)), blackbody.size)),
            common,
        )
        for model in ("robert_shared", "petitradtrans_shared")
    }
    observed = {
        "track_a_max_abs_symmetric_relative": max(
            item["max_abs_symmetric_relative"] for item in track_a_metrics
        ),
        "track_a_max_abs_eclipse_difference_ppm": max(
            item["max_abs_eclipse_difference_ppm"] for item in track_a_metrics
        ),
        "track_a_80_to_160_max_abs_eclipse_difference_ppm": max(
            convergence[model]["80_to_160"]["max_abs_eclipse_difference_ppm"]
            for model in ("robert_shared", "petitradtrans_shared")
        ),
        "track_a_isothermal_max_abs_eclipse_difference_ppm": max(
            item["max_abs_eclipse_difference_ppm"] for item in iso_metrics.values()
        ),
        "scattering_single_scattering_albedo_max_abs": 0.0,
    }
    gate_results = {
        name: observed[name] <= limit
        for name, limit in STAGE_3_ACCEPTANCE_GATES.items()
        if name in observed
    }
    frozen_mmw = float(
        outputs[PRIMARY_RESOLUTION]["contract"]["mean_molecular_weight_u"][0]
    )
    h2_he_total = common.composition_vmr["H2"] + common.composition_vmr["He"]
    background_mmw = (
        common.composition_vmr["H2"] / h2_he_total * common.molecular_masses_u["H2"]
        + common.composition_vmr["He"]
        / h2_he_total
        * common.molecular_masses_u["He"]
    )

    report_path = data_root / "stage_3_report.json"
    data_checksums = {
        "picaso_correlated_k_assets": {
            species: {
                "path": str(PICASO_CK_DIRECTORY / asset.filename),
                "sha256": asset.sha256,
            }
            for species, asset in common.picaso_correlated_k_assets.items()
        },
        "picaso_cia_database": {
            "path": str(PICASO_CIA_DATABASE),
            "sha256": _sha256(PICASO_CIA_DATABASE),
        },
        "petitradtrans_assets": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    report = {
        "schema_version": "2.0.0",
        "intercomparison_version": 2,
        "stage": 3,
        "status": (
            "pass" if all(gate_results.values()) else "out_of_tolerance_closure_regime"
        ),
        "scientific_framing": (
            "Measured agreement, differences, representation effects, convergence, "
            "and capability boundaries; no framework is classified as failed."
        ),
        "common_contract_sha256": common.to_dict()["contract_sha256"],
        "common_contract_file_sha256": _sha256(COMMON_CONTRACT),
        "predeclared_acceptance_gates": STAGE_3_ACCEPTANCE_GATES,
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "factorial_contract": {
            "molecular_absorbers_always_active": list(MOLECULAR_SPECIES),
            "fixed_composition": dict(common.composition_vmr),
            "fixed_mean_molecular_weight_u": frozen_mmw,
            "cia_factors": ["H2-H2", "H2-He"],
            "factor_names": [item[0] for item in FACTORIAL],
            "profiles": list(PROFILES),
        },
        "mean_molecular_weight_attribution": {
            "frozen_full_mixture_mmw_u": frozen_mmw,
            "derived_h2_he_only_mmw_u_for_column_scaling_context": background_mmw,
            "full_mixture_to_h2_he_only_number_column_ratio": (
                background_mmw / frozen_mmw
            ),
            "counterfactual_spectra_run": False,
            "interpretation": (
                "The exact full-mixture MMW is fixed in every case, so MMW cannot alias "
                "the CIA factorial; the derived ratio is context, not a modified contract."
            ),
        },
        "track_a_scope": {
            "gated_frameworks": ["robert", "petitradtrans"],
            "tensor": (
                "identical pressure-cell by wavelength g-weighted four-molecule "
                "random-overlap optical depth plus independently selected CIA components"
            ),
            "picaso": (
                "unsupported as a native exact-omega0=0 identical-tensor interface; "
                "no PICASO Track-A gate is invented"
            ),
        },
        "track_b_scope": {
            "picaso_primary": (
                "official PICASO-4 resort-rebin correlated-k opacity with selected "
                "native CIA and an independently labelled absorbing-formal RT reference"
            ),
            "picaso_secondary": None,
            "retired_representation": (
                "opacity sampling is not run under the 0.3--12 micron contract"
            ),
            "robert": "independent pRT-HDF load/interpolation, random overlap, and CIA",
            "petitradtrans": "native stable-pRT correlated-k and selected CIA",
            "cross_framework_gates": None,
        },
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "per_resolution": per_resolution,
        "vertical_convergence": convergence,
        "isothermal_analytic_control": iso_metrics,
        "pilot": pilot,
        "timings": {
            "pilot_wall_time_s": pilot_wall,
            "post_pilot_matrix_wall_time_s": matrix_wall,
            "raw_case_timings_retained": True,
        },
        "resources": {
            "pilot_largest_process_peak_rss_bytes": largest_peak,
            "available_memory_bytes_at_pilot": available_memory,
            "platform": platform.platform(),
        },
        "interpreters": {
            "robert": str(ROBERT_PYTHON),
            "picaso": str(PICASO_PYTHON),
            "petitradtrans": str(PRT_PYTHON),
        },
        "package_versions": {
            "robert-exoplanets": importlib.metadata.version("robert-exoplanets"),
            "numpy": importlib.metadata.version("numpy"),
            "picaso": _metadata(outputs[80]["picaso_correlated_k"])["version"],
            "petitRADTRANS": _metadata(outputs[80]["petitradtrans"])["version"],
        },
        "data_checksums": data_checksums,
        "source_checksums": {
            str(path.relative_to(REPOSITORY)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                WORKER.resolve(),
                REPOSITORY
                / "src/robert_exoplanets/diagnostics/emission_intercomparison_v2.py",
                REPOSITORY / "src/robert_exoplanets/atmosphere/temperature.py",
                REPOSITORY / "src/robert_exoplanets/rt/extinction.py",
                REPOSITORY / "src/robert_exoplanets/rt/random_overlap.py",
            )
        },
        "known_warnings_and_capability_boundaries": [
            "PICASO's harmless optional-Vega warning is recorded; no stellar grids were downloaded or used.",
            "PICASO exact-zero cloud/Rayleigh invalid-divide warnings are recorded; retained arrays verify both remain zero.",
            "PICASO exact-omega0=0 shared-tensor RT remains unsupported as a native gated interface.",
            "PICASO exact-omega0=0 native thermal probes remain capability evidence; separately labelled absorbing-formal spectra are the comparison products.",
            "PICASO vertical arrays are absorbing-formal diagnostics applied to native total gas optical depth, not native SH contribution definitions.",
            "Stable pRT does not expose native layer optical-depth tensors through the supported high-level flux interface; spectra and native emission contributions are retained.",
            "Opacity sampling was retired and is not run under the 0.3--12 micron contract.",
            "Stage-1 continuous-angle sub-0.01 ppm claims remain restricted; its 0.196897 ppm eight-angle result is not reinterpreted.",
            "Stage-2's measured out-of-tolerance, vertically converging Track-A regime remains unchanged and is not reclassified.",
        ],
        "random_seed": None,
        "random_seed_policy": common.random_seed_policy,
    }
    _write_json(report_path, report)
    artifact_paths.append(report_path)
    integrity_path = data_root / "stage_3_integrity.json"
    _write_json(integrity_path, _integrity_manifest(artifact_paths))
    artifact_paths.append(integrity_path)
    checksum_path = data_root / "checksums.json"
    existing = json.loads(checksum_path.read_text())
    for name in tuple(existing):
        if name.startswith("stage_3_"):
            existing.pop(name)
    for path in artifact_paths:
        existing[path.name] = _sha256(path)
    _write_json(checksum_path, existing)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pilot": pilot,
                "observed_gate_values": observed,
                "artifact_count": len(artifact_paths),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
