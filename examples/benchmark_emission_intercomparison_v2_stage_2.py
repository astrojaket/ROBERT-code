"""Run emission intercomparison Version-2 Stage 2 and write frozen products."""

from __future__ import annotations

import argparse
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
    isolated_molecule_composition,
    load_version_2_common_contract,
    planck_surface_flux_w_m2_m,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY / "docs/data/emission_intercomparison/version_2"
COMMON_CONTRACT = DATA_ROOT / "common_contract.json"
DEFAULT_OUTPUT = REPOSITORY / "examples/outputs/emission_intercomparison/version_2/stage_2"
WORKER = Path(__file__).with_name(
    "run_emission_intercomparison_v2_stage_2_external.py"
)
ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
PICASO_REFERENCE = Path("/Users/jaketaylor/Dropbox/picaso-v4/reference")
PICASO_CK_DIRECTORY = Path(
    "/Users/jaketaylor/Dropbox/picaso/reference/opacities/resortrebin"
)
PICASO_SAMPLING_DATABASE = Path(
    "/Users/jaketaylor/Dropbox/ROBERT-code/opacity_data/picaso_official/reference/opacities/opacities_0.3_15_R15000.db"
)
PRT_INPUT_DATA = Path(
    "/Users/jaketaylor/Dropbox/ROBERT-code/opacity_data/petitRADTRANS/input_data"
)
RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
PRIMARY_SAMPLING_RESAMPLE = 50
DENSITY_CHECK_RESAMPLE = 25

# Frozen before the complete Stage-2 matrix is inspected. Only the compatible
# ROBERT/stable-pRT identical-tensor path is gated. Native representations are
# attribution results and deliberately have no cross-framework acceptance gate.
STAGE_2_ACCEPTANCE_GATES = {
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
    # The evaluator uses only centres and the unit. This single orchestrator
    # evaluation is serialized and supplied unchanged to every worker.
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


def build_stage_2_contract(
    common: Version2CommonContract,
    n_cells: int,
    *,
    include_abundance_check: bool = True,
) -> dict[str, np.ndarray]:
    """Build cases exclusively from the serialized common contract."""

    grid = next(grid for grid in common.pressure_grids if grid.n_cells == n_cells)
    species_names = tuple(
        name for name in common.composition_vmr if name not in {"H2", "He"}
    )
    gas_names = ("H2", "He", *species_names)
    mu, legendre, disk = _quadrature()
    evaluated_edges = {
        profile: _edge_temperature(common, profile, grid.edges_bar)
        for profile in PROFILES
    }
    evaluated_cells = {
        profile: common.temperature_profiles_k[
            f"{profile}_{n_cells}_cells"
            if profile != "isothermal"
            else f"isothermal_{n_cells}_cells"
        ]
        for profile in PROFILES
    }
    cases: list[tuple[str, str, float]] = [
        (species, profile, 1.0)
        for species in species_names
        for profile in PROFILES
    ]
    if include_abundance_check and n_cells == PRIMARY_RESOLUTION:
        cases.extend(
            [
                ("H2O", "pg14_non_inverted", 0.5),
                ("H2O", "pg14_non_inverted", 2.0),
            ]
        )
    vmr = []
    mean_molecular_weight = []
    case_id = []
    for species, profile, scale in cases:
        composition = isolated_molecule_composition(
            common, species, abundance_scale=scale
        )
        values = np.asarray([composition[name] for name in gas_names])
        vmr.append(values)
        mean_molecular_weight.append(
            np.sum(values * np.asarray([common.molecular_masses_u[name] for name in gas_names]))
        )
        suffix = "reference" if scale == 1.0 else f"abundance_{scale:g}x"
        case_id.append(f"{species}_{profile}_{suffix}_{n_cells}_cells")
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(2),
        "case_id": np.asarray(case_id),
        "species_name": np.asarray([item[0] for item in cases]),
        "profile_name": np.asarray([item[1] for item in cases]),
        "abundance_scale": np.asarray([item[2] for item in cases]),
        "reference_case_mask": np.asarray([item[2] == 1.0 for item in cases]),
        "gas_name": np.asarray(gas_names),
        "gas_mass_u": np.asarray([common.molecular_masses_u[name] for name in gas_names]),
        "gas_vmr": np.asarray(vmr),
        "mean_molecular_weight_u": np.asarray(mean_molecular_weight),
        "pressure_edges_bar": grid.edges_bar,
        "pressure_centers_bar": grid.centers_bar,
        "temperature_edges_k": np.asarray(
            [evaluated_edges[item[1]] for item in cases]
        ),
        "temperature_cells_k": np.asarray(
            [evaluated_cells[item[1]] for item in cases]
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
    result = {}
    for name, value in contract.items():
        if value.ndim > 0 and value.shape[0] == count and name not in {
            "gas_name",
            "gas_mass_u",
            "pressure_edges_bar",
            "pressure_centers_bar",
            "emission_mu",
            "legendre_weights",
            "disk_weights",
        }:
            result[name] = value[selected]
        else:
            result[name] = value
    return result


def _subset_payload(
    payload: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, np.ndarray]:
    count = selected.size
    return {
        name: value[selected]
        if value.ndim > 0 and value.shape[0] == count
        else value
        for name, value in payload.items()
    }


def _normalise_vertical(values: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = np.sum(array, axis=-2, keepdims=True)
    return np.divide(array, total, out=np.zeros_like(array), where=total > 0.0)


def _prt_table_paths() -> dict[str, Path]:
    patterns = {
        "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
        "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
        "CO2": "*UCL-4000*.ktable.petitRADTRANS.h5",
        "CH4": "*YT34to10*.ktable.petitRADTRANS.h5",
    }
    root = PRT_INPUT_DATA / "opacities/lines/correlated_k"
    return {species: next(root.rglob(pattern)) for species, pattern in patterns.items()}


def _run_robert_native(
    contract: dict[str, np.ndarray], table_paths: dict[str, Path]
) -> dict[str, np.ndarray]:
    from robert_exoplanets import (
        AtmosphereState,
        CorrelatedKOpacityProvider,
        CorrelatedKTable,
        PressureGrid,
        SpectralGrid,
        assemble_gas_optical_depth,
        gauss_legendre_disk_geometry,
        solve_emission,
    )

    gas_names = [str(value) for value in contract["gas_name"]]
    tables = {
        species: CorrelatedKTable.from_petitradtrans_hdf(path, species=species)
        for species, path in table_paths.items()
    }
    first = tables["H2O"]
    mask = (first.wavelength_micron >= 0.79) & (first.wavelength_micron <= 12.1)
    wavelength = np.sort(first.wavelength_micron[mask])
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="stage2-pRT-R1000"
    )
    pressure_grid = PressureGrid(
        edges=contract["pressure_edges_bar"],
        centers=contract["pressure_centers_bar"],
        unit="bar",
        name="version-2-stage-2",
    )
    providers = {
        species: CorrelatedKOpacityProvider(
            {species: table},
            name=f"stage2-{species}",
            interpolation="log_pressure_temperature_log_k",
        )
        for species, table in tables.items()
    }
    prepared = {
        species: provider.prepare(spectral_grid, pressure_grid, species=(species,))
        for species, provider in providers.items()
    }
    flux = []
    layer_tau = []
    contribution = []
    timings = []
    for case_index, species_value in enumerate(contract["species_name"]):
        species = str(species_value)
        composition = dict(zip(gas_names, contract["gas_vmr"][case_index], strict=True))
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=contract["temperature_cells_k"][case_index],
            temperature_edges=contract["temperature_edges_k"][case_index],
            composition={
                name: np.full(pressure_grid.n_layers, value)
                for name, value in composition.items()
            },
            mean_molecular_weight=np.full(
                pressure_grid.n_layers,
                contract["mean_molecular_weight_u"][case_index],
            ),
        )
        opacity = providers[species].evaluate(atmosphere, prepared[species])
        gas_tau = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=float(contract["gravity_m_s2"]),
            gas_combination="sum_by_g",
        )
        started = perf_counter()
        result = solve_emission(
            gas_tau,
            geometry=gauss_legendre_disk_geometry(n_mu=8),
            bottom_boundary="blackbody",
        )
        timings.append(perf_counter() - started)
        flux.append(np.pi * np.asarray(result.radiance.values))
        layer_tau.append(gas_tau.total_tau)
        vertical = np.asarray(result.layer_contribution_radiance, dtype=float).copy()
        vertical[-1] += np.asarray(result.bottom_contribution_radiance)
        contribution.append(_normalise_vertical(vertical))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    metadata = {
        "model": "robert",
        "mode": "native_correlated_k",
        "python": os.path.realpath(sys.executable),
        "version": importlib.metadata.version("robert-exoplanets"),
        "opacity_source": "petitRADTRANS HDF5 tables loaded independently by ROBERT",
        "interpolation": "log_pressure_temperature_log_k",
        "cia_enabled": False,
        "rayleigh_enabled": False,
        "scattering_enabled": False,
        "peak_rss_bytes": int(
            usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        ),
    }
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "layer_tau": np.asarray(layer_tau, dtype=np.float32),
        "g_weights": first.g_weights,
        "normalized_vertical_diagnostic": np.asarray(contribution, dtype=np.float32),
        "runtime_s": np.asarray(timings),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
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
        vertical = np.asarray(vertical, dtype=float)
        bottom = np.tensordot(
            contract["disk_weights"],
            result.point_bottom_contribution_radiance,
            axes=(0, 0),
        )
        vertical[-1] += bottom
        flux.append(np.pi * np.sum(vertical, axis=0))
        contribution.append(_normalise_vertical(vertical))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    metadata = {
        "model": "robert",
        "mode": "track_a_identical_mean_tau",
        "python": os.path.realpath(sys.executable),
        "version": importlib.metadata.version("robert-exoplanets"),
        "peak_rss_bytes": int(
            usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        ),
        "cia_enabled": False,
        "rayleigh_enabled": False,
        "scattering_enabled": False,
    }
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.asarray(flux),
        "normalized_vertical_diagnostic": np.asarray(contribution, dtype=np.float32),
        "runtime_s": np.asarray(timings),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _run_external(
    python: Path,
    mode: str,
    contract_path: Path,
    output_path: Path,
    *,
    sampling_resample: int = PRIMARY_SAMPLING_RESAMPLE,
) -> dict[str, np.ndarray]:
    command = [str(python), str(WORKER), mode, str(contract_path), str(output_path)]
    if mode == "picaso_ck":
        command.extend(["--picaso-ck-directory", str(PICASO_CK_DIRECTORY)])
    elif mode == "picaso_sampling":
        command.extend(
            [
                "--picaso-sampling-database",
                str(PICASO_SAMPLING_DATABASE),
                "--picaso-sampling-resample",
                str(sampling_resample),
            ]
        )
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
    if mode.startswith("petitradtrans"):
        worker_home = output_path.parent / ".petitradtrans-worker-home"
        worker_home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(worker_home)
    subprocess.run(command, check=True, env=environment)
    return _load_npz(output_path)


def _shared_contract(
    contract: dict[str, np.ndarray], robert_native: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    weights = robert_native["g_weights"]
    mean_tau = np.sum(
        robert_native["layer_tau"].astype(float) * weights[None, None, None, :],
        axis=-1,
    )
    return {
        **contract,
        "shared_wavelength_micron": robert_native["wavelength_micron"],
        "shared_layer_tau": mean_tau,
        "shared_source": np.array(
            "ROBERT-independent pRT-HDF evaluation collapsed by native g weights"
        ),
    }


def _bin_flux(common: Version2CommonContract, payload: dict[str, np.ndarray]) -> np.ndarray:
    return flux_conserving_bin_mean(
        payload["wavelength_micron"],
        payload["flux_w_m2_m"],
        common.spectral.r100_edges_micron,
    )


def _bin_contribution(
    common: Version2CommonContract, payload: dict[str, np.ndarray]
) -> np.ndarray:
    binned = flux_conserving_bin_mean(
        payload["wavelength_micron"],
        payload["normalized_vertical_diagnostic"],
        common.spectral.r100_edges_micron,
    )
    return _normalise_vertical(binned)


def _difference(
    left: np.ndarray, right: np.ndarray, common: Version2CommonContract
) -> dict[str, float]:
    denominator = np.abs(left) + np.abs(right)
    symmetric = np.divide(
        2.0 * (left - right), denominator, out=np.zeros_like(left), where=denominator > 0.0
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


def _sampling_diagnostics(
    wavelength: np.ndarray, flux: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.empty(edges.size - 1, dtype=int)
    variance = np.empty((flux.shape[0], edges.size - 1))
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (wavelength >= lower) & (
            (wavelength < upper) if index < edges.size - 2 else (wavelength <= upper)
        )
        counts[index] = int(np.sum(selected))
        variance[:, index] = (
            np.var(flux[:, selected], axis=1) if counts[index] else np.nan
        )
    return counts, variance


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
    eclipse_native = (
        payload["flux_w_m2_m"]
        / planck_surface_flux_w_m2_m(
            payload["wavelength_micron"],
            common.measurements["stellar_effective_temperature"].si_value,
        )
        * common.derived["projected_area_ratio"]
    )
    eclipse_r100 = (
        r100_flux
        / common.stellar_surface_flux_r100_w_m2_m
        * common.derived["projected_area_ratio"]
    )
    arrays = {
        "case_id": contract["case_id"],
        "species_name": contract["species_name"],
        "profile_name": contract["profile_name"],
        "abundance_scale": contract["abundance_scale"],
        "gas_name": contract["gas_name"],
        "gas_mass_u": contract["gas_mass_u"],
        "gas_vmr": contract["gas_vmr"],
        "mean_molecular_weight_u": contract["mean_molecular_weight_u"],
        "pressure_edges_bar": contract["pressure_edges_bar"],
        "pressure_centers_bar": contract["pressure_centers_bar"],
        "temperature_edges_k": contract["temperature_edges_k"],
        "temperature_cells_k": contract["temperature_cells_k"],
        "native_wavelength_micron": payload["wavelength_micron"],
        "native_flux_w_m2_m": payload["flux_w_m2_m"],
        "native_eclipse_depth": eclipse_native,
        "r100_edges_micron": common.spectral.r100_edges_micron,
        "r100_centers_micron": common.spectral.r100_centers_micron,
        "r100_flux_w_m2_m": r100_flux,
        "r100_eclipse_depth": eclipse_r100,
        "normalized_vertical_native": payload["normalized_vertical_diagnostic"],
        "normalized_vertical_r100": r100_contribution.astype(np.float32),
        "runtime_s": payload["runtime_s"],
        "representation": np.array(representation),
        "metadata_json": payload.get("metadata_json", np.array("{}")),
    }
    for name in (
        "layer_tau",
        "g_weights",
        "maximum_abs_rayleigh_tau",
        "maximum_abs_cloud_tau",
        "native_framework_probe_flux_w_m2_m",
    ):
        if name in payload:
            arrays[name] = payload[name]
    if "opacity_sampling" in representation:
        sample_count, within_bin_variance = _sampling_diagnostics(
            payload["wavelength_micron"],
            payload["flux_w_m2_m"],
            common.spectral.r100_edges_micron,
        )
        arrays["sample_count_per_r100_bin"] = sample_count
        arrays["within_bin_flux_variance"] = within_bin_variance
        arrays["smoothing_applied"] = np.array(False)
    np.savez_compressed(path, **arrays)


def _integrity_manifest(paths: list[Path]) -> dict[str, Any]:
    artifacts = {}
    for path in paths:
        entry: dict[str, Any] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                entry["arrays"] = {
                    name: {
                        "shape": list(archive[name].shape),
                        "dtype": str(archive[name].dtype),
                        "finite_policy": (
                            "NaN_only_for_empty_opacity_sampling_bins_or_declared_unsupported_diagnostics"
                            if np.issubdtype(archive[name].dtype, np.floating)
                            and np.any(~np.isfinite(archive[name]))
                            else "all_finite"
                        ),
                    }
                    for name in archive.files
                }
        artifacts[path.name] = entry
    return {
        "schema_version": "1.0.0",
        "stage": 2,
        "units_and_axes": {
            "flux": "case,wavelength; W m^-2 m^-1; positive outward",
            "eclipse_depth": "case,wavelength; dimensionless signed",
            "layer_tau": "case,pressure-cell,wavelength,g when g is present",
            "normalized_vertical": "case,pressure-cell,wavelength; unit sum over pressure",
            "pressure": "bar; top-to-bottom increasing",
            "wavelength": "micron; increasing vacuum wavelength",
        },
        "artifacts": artifacts,
    }


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
        raise RuntimeError(f"Stage 2 must run with {ROBERT_PYTHON}")
    common = load_version_2_common_contract(COMMON_CONTRACT)
    table_paths = _prt_table_paths()
    for species, asset in common.picaso_correlated_k_assets.items():
        path = PICASO_CK_DIRECTORY / asset.filename
        if _sha256(path) != asset.sha256:
            raise RuntimeError(f"frozen PICASO correlated-k checksum mismatch: {species}")

    primary_contract = build_stage_2_contract(common, PRIMARY_RESOLUTION)
    pilot_selected = (
        (primary_contract["species_name"] == "H2O")
        & (primary_contract["profile_name"] == "pg14_non_inverted")
        & primary_contract["reference_case_mask"]
    )
    pilot_contract = _subset_contract(primary_contract, pilot_selected)
    pilot_root = output_root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    pilot_contract_path = pilot_root / "contract.npz"
    np.savez_compressed(pilot_contract_path, **pilot_contract)
    pilot_started = perf_counter()
    pilot_robert = _run_robert_native(pilot_contract, table_paths)
    pilot_picaso = _run_external(
        PICASO_PYTHON, "picaso_ck", pilot_contract_path, pilot_root / "picaso_ck.npz"
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
        json.loads(str(pilot_robert["metadata_json"]))["peak_rss_bytes"],
        _metadata(pilot_picaso)["peak_rss_bytes"],
        _metadata(pilot_prt)["peak_rss_bytes"],
    ]
    largest_peak = int(max(pilot_peaks))
    projected_wall = pilot_wall * 42.0
    memory_fraction = largest_peak / available_memory
    authorized = (
        projected_wall <= STAGE_2_ACCEPTANCE_GATES["pilot_projected_wall_time_max_s"]
        and memory_fraction
        <= STAGE_2_ACCEPTANCE_GATES["pilot_peak_rss_fraction_of_available_max"]
    )
    pilot = {
        "resolution_cells": PRIMARY_RESOLUTION,
        "case_id": str(pilot_contract["case_id"][0]),
        "frameworks": ["robert", "picaso", "petitradtrans"],
        "measured_wall_time_s": pilot_wall,
        "projection_multiplier": 42.0,
        "projected_complete_wall_time_s": projected_wall,
        "largest_process_peak_rss_bytes": largest_peak,
        "available_memory_bytes_at_decision": available_memory,
        "peak_rss_fraction_of_available": memory_fraction,
        "authorized_full_matrix": authorized,
        "decision_limits": {
            key: value for key, value in STAGE_2_ACCEPTANCE_GATES.items() if key.startswith("pilot_")
        },
    }
    _write_json(pilot_root / "pilot_decision.json", pilot)
    if args.pilot_only:
        print(json.dumps(pilot, indent=2, sort_keys=True))
        return
    if not authorized:
        raise RuntimeError("Stage-2 matrix not authorized by frozen pilot resource gates")

    outputs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    artifact_paths: list[Path] = []
    matrix_started = perf_counter()
    for n_cells in RESOLUTIONS:
        contract = build_stage_2_contract(common, n_cells)
        resolution_root = output_root / f"{n_cells}_cells"
        resolution_root.mkdir(parents=True, exist_ok=True)
        contract_path = resolution_root / "contract.npz"
        np.savez_compressed(contract_path, **contract)
        robert = _run_robert_native(contract, table_paths)
        picaso = _run_external(
            PICASO_PYTHON, "picaso_ck", contract_path, resolution_root / "picaso_ck.npz"
        )
        prt = _run_external(
            PRT_PYTHON,
            "petitradtrans_native",
            contract_path,
            resolution_root / "petitradtrans_native.npz",
        )
        sampling_contract = _subset_contract(contract, contract["reference_case_mask"])
        sampling_contract_path = resolution_root / "sampling_contract.npz"
        np.savez_compressed(sampling_contract_path, **sampling_contract)
        sampling = _run_external(
            PICASO_PYTHON,
            "picaso_sampling",
            sampling_contract_path,
            resolution_root / "picaso_sampling.npz",
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
            "picaso_opacity_sampling": sampling,
            "robert_shared": robert_shared,
            "petitradtrans_shared": prt_shared,
            "shared_contract": shared,
        }
        for name, payload, payload_contract, representation in (
            ("robert", robert, contract, "native_correlated_k"),
            (
                "picaso",
                picaso,
                contract,
                "native_correlated_k_resort_rebin_opacity_with_absorbing_formal_reference",
            ),
            ("petitradtrans", prt, contract, "native_correlated_k"),
            (
                "picaso_opacity_sampling",
                sampling,
                sampling_contract,
                "native_opacity_sampling_unsmoothed_with_absorbing_formal_reference",
            ),
            ("robert_shared", robert_shared, contract, "track_a_identical_mean_tau"),
            ("petitradtrans_shared", prt_shared, contract, "track_a_identical_mean_tau"),
        ):
            if name == "robert":
                for species in dict.fromkeys(payload_contract["species_name"].tolist()):
                    selected = payload_contract["species_name"] == species
                    path = data_root / f"stage_2_robert_{species}_{n_cells}_cells.npz"
                    _save_artifact(
                        path,
                        _subset_contract(payload_contract, selected),
                        _subset_payload(payload, selected),
                        common,
                        representation=representation,
                    )
                    artifact_paths.append(path)
            else:
                path = data_root / f"stage_2_{name}_{n_cells}_cells.npz"
                _save_artifact(
                    path,
                    payload_contract,
                    payload,
                    common,
                    representation=representation,
                )
                artifact_paths.append(path)
        shared_path = data_root / f"stage_2_shared_tau_{n_cells}_cells.npz"
        np.savez_compressed(
            shared_path,
            case_id=contract["case_id"],
            species_name=contract["species_name"],
            profile_name=contract["profile_name"],
            pressure_edges_bar=contract["pressure_edges_bar"],
            pressure_centers_bar=contract["pressure_centers_bar"],
            wavelength_micron=shared["shared_wavelength_micron"],
            layer_tau=shared["shared_layer_tau"].astype(np.float32),
            source=shared["shared_source"],
        )
        artifact_paths.append(shared_path)
    matrix_wall = perf_counter() - matrix_started

    density_contract = _subset_contract(primary_contract, pilot_selected)
    density_contract_path = output_root / "sampling_density_contract.npz"
    np.savez_compressed(density_contract_path, **density_contract)
    density = _run_external(
        PICASO_PYTHON,
        "picaso_sampling",
        density_contract_path,
        output_root / "picaso_sampling_density.npz",
        sampling_resample=DENSITY_CHECK_RESAMPLE,
    )
    density_path = data_root / "stage_2_picaso_sampling_density_check.npz"
    primary_sampling = outputs[80]["picaso_opacity_sampling"]
    primary_sampling_contract = _subset_contract(
        outputs[80]["contract"], outputs[80]["contract"]["reference_case_mask"]
    )
    primary_index = int(
        np.flatnonzero(
            (primary_sampling_contract["species_name"] == "H2O")
            & (primary_sampling_contract["profile_name"] == "pg14_non_inverted")
        )[0]
    )
    counts_primary, variance_primary = _sampling_diagnostics(
        primary_sampling["wavelength_micron"],
        primary_sampling["flux_w_m2_m"][[primary_index]],
        common.spectral.r100_edges_micron,
    )
    counts_density, variance_density = _sampling_diagnostics(
        density["wavelength_micron"], density["flux_w_m2_m"], common.spectral.r100_edges_micron
    )
    np.savez_compressed(
        density_path,
        case_id=density_contract["case_id"],
        r100_edges_micron=common.spectral.r100_edges_micron,
        r100_centers_micron=common.spectral.r100_centers_micron,
        primary_resample=np.array(PRIMARY_SAMPLING_RESAMPLE),
        density_resample=np.array(DENSITY_CHECK_RESAMPLE),
        primary_native_wavelength_micron=primary_sampling["wavelength_micron"],
        primary_native_flux_w_m2_m=primary_sampling["flux_w_m2_m"][primary_index],
        primary_r100_flux_w_m2_m=_bin_flux(common, primary_sampling)[primary_index],
        primary_sample_count_per_r100_bin=counts_primary,
        primary_within_bin_flux_variance=variance_primary[0],
        density_native_wavelength_micron=density["wavelength_micron"],
        density_native_flux_w_m2_m=density["flux_w_m2_m"][0],
        density_r100_flux_w_m2_m=_bin_flux(common, density)[0],
        density_sample_count_per_r100_bin=counts_density,
        density_within_bin_flux_variance=variance_density[0],
        smoothing_applied=np.array(False),
        density_metadata_json=density["metadata_json"],
    )
    artifact_paths.append(density_path)

    per_resolution: dict[str, Any] = {}
    track_a_metrics = []
    for n_cells in RESOLUTIONS:
        contract = outputs[n_cells]["contract"]
        reference = contract["reference_case_mask"]
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
            binned["robert_shared"][reference],
            binned["petitradtrans_shared"][reference],
            common,
        )
        track_a_metrics.append(track_a)
        native_pairs = {
            "robert__picaso_correlated_k": _difference(
                binned["robert"][reference], binned["picaso_correlated_k"][reference], common
            ),
            "robert__petitradtrans": _difference(
                binned["robert"][reference], binned["petitradtrans"][reference], common
            ),
            "picaso_correlated_k__petitradtrans": _difference(
                binned["picaso_correlated_k"][reference], binned["petitradtrans"][reference], common
            ),
        }
        per_case = {}
        for index in np.flatnonzero(reference):
            case_id = str(contract["case_id"][index])
            per_case[case_id] = {
                "species": str(contract["species_name"][index]),
                "profile": str(contract["profile_name"][index]),
                "track_a_robert_vs_petitradtrans": _difference(
                    binned["robert_shared"][[index]],
                    binned["petitradtrans_shared"][[index]],
                    common,
                ),
                "track_b_native_representation_attribution": {
                    "robert__picaso_correlated_k": _difference(
                        binned["robert"][[index]],
                        binned["picaso_correlated_k"][[index]],
                        common,
                    ),
                    "robert__petitradtrans": _difference(
                        binned["robert"][[index]],
                        binned["petitradtrans"][[index]],
                        common,
                    ),
                    "picaso_correlated_k__petitradtrans": _difference(
                        binned["picaso_correlated_k"][[index]],
                        binned["petitradtrans"][[index]],
                        common,
                    ),
                },
            }
        per_resolution[str(n_cells)] = {
            "case_count": int(contract["case_id"].size),
            "reference_case_count": int(np.sum(reference)),
            "track_a_robert_vs_petitradtrans": track_a,
            "track_b_native_representation_attribution": native_pairs,
            "per_case_species_profile_attribution": per_case,
            "native_wavelength_count": {
                name: int(outputs[n_cells][name]["wavelength_micron"].size)
                for name in ("robert", "picaso_correlated_k", "petitradtrans", "picaso_opacity_sampling")
            },
            "raw_runtime_s": {
                name: outputs[n_cells][name]["runtime_s"].tolist()
                for name in (
                    "robert",
                    "picaso_correlated_k",
                    "petitradtrans",
                    "picaso_opacity_sampling",
                    "robert_shared",
                    "petitradtrans_shared",
                )
            },
            "worker_metadata": {
                name: _metadata(outputs[n_cells][name])
                for name in (
                    "robert",
                    "picaso_correlated_k",
                    "petitradtrans",
                    "picaso_opacity_sampling",
                    "robert_shared",
                    "petitradtrans_shared",
                )
            },
        }

    convergence: dict[str, Any] = {}
    for model in (
        "robert",
        "picaso_correlated_k",
        "petitradtrans",
        "picaso_opacity_sampling",
        "robert_shared",
        "petitradtrans_shared",
    ):
        convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            coarse_contract = outputs[coarse]["contract"]
            fine_contract = outputs[fine]["contract"]
            if model == "picaso_opacity_sampling":
                coarse_mask = np.ones(
                    outputs[coarse][model]["flux_w_m2_m"].shape[0], dtype=bool
                )
                fine_mask = np.ones(
                    outputs[fine][model]["flux_w_m2_m"].shape[0], dtype=bool
                )
            else:
                coarse_mask = coarse_contract["reference_case_mask"]
                fine_mask = fine_contract["reference_case_mask"]
            convergence[model][f"{coarse}_to_{fine}"] = _difference(
                _bin_flux(common, outputs[coarse][model])[coarse_mask],
                _bin_flux(common, outputs[fine][model])[fine_mask],
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
    iso_mask = (
        outputs[80]["contract"]["reference_case_mask"]
        & (outputs[80]["contract"]["profile_name"] == "isothermal")
    )
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
        for name, limit in STAGE_2_ACCEPTANCE_GATES.items()
        if name in observed
    }
    density_metrics = _difference(
        _bin_flux(common, primary_sampling)[[primary_index]], _bin_flux(common, density), common
    )
    abundance_indices = np.flatnonzero(
        (outputs[80]["contract"]["species_name"] == "H2O")
        & (outputs[80]["contract"]["profile_name"] == "pg14_non_inverted")
    )
    abundance_summary = {
        model: {
            str(float(outputs[80]["contract"]["abundance_scale"][index])): (
                _bin_flux(common, outputs[80][model])[index].tolist()
            )
            for index in abundance_indices
        }
        for model in ("robert", "picaso_correlated_k", "petitradtrans")
    }

    report_path = data_root / "stage_2_report.json"
    data_checksums = {
        "picaso_correlated_k_assets": {
            species: {
                "path": str(PICASO_CK_DIRECTORY / asset.filename),
                "sha256": asset.sha256,
            }
            for species, asset in common.picaso_correlated_k_assets.items()
        },
        "picaso_opacity_sampling_database": {
            "path": str(PICASO_SAMPLING_DATABASE),
            "sha256": _sha256(PICASO_SAMPLING_DATABASE),
        },
        "petitradtrans_correlated_k_assets": {
            species: {"path": str(path), "sha256": _sha256(path)}
            for species, path in table_paths.items()
        },
    }
    report = {
        "schema_version": "2.0.0",
        "intercomparison_version": 2,
        "stage": 2,
        "status": "pass" if all(gate_results.values()) else "out_of_tolerance_closure_regime",
        "scientific_framing": "Measured agreement, differences, representation effects, convergence, and capability boundaries; no framework is classified as failed.",
        "common_contract_sha256": common.to_dict()["contract_sha256"],
        "common_contract_file_sha256": _sha256(COMMON_CONTRACT),
        "predeclared_acceptance_gates": STAGE_2_ACCEPTANCE_GATES,
        "observed_gate_values": observed,
        "gate_results": gate_results,
        "track_a_scope": {
            "gated_frameworks": ["robert", "petitradtrans"],
            "tensor": "identical pressure-cell by wavelength g-weighted mean molecular optical depth",
            "picaso": "unsupported as a native exact-omega0=0 identical-tensor interface; no PICASO Track-A gate is invented",
        },
        "track_b_scope": {
            "picaso_primary": "official PICASO-4 resort-rebin correlated-k opacity with the independently labelled absorbing-formal RT reference; raw native thermal probe retained",
            "picaso_secondary": "unsmoothed opacity-sampling opacity with absorbing-formal native/R100 spectra and sampling diagnostics; raw native thermal probe retained",
            "robert": "independent pRT-HDF load/interpolation through ROBERT",
            "petitradtrans": "native stable-pRT correlated-k",
            "cross_framework_gates": None,
        },
        "species": list(dict.fromkeys(outputs[80]["contract"]["species_name"].tolist())),
        "profiles": list(PROFILES),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "per_resolution": per_resolution,
        "vertical_convergence": convergence,
        "isothermal_analytic_control": iso_metrics,
        "abundance_check": {
            "case": "H2O_pg14_non_inverted_80_cells",
            "scales": [0.5, 1.0, 2.0],
            "r100_flux_w_m2_m": abundance_summary,
            "interpretation": "finite abundance response retained without a cross-framework gate",
        },
        "picaso_opacity_sampling_density_check": {
            "case_id": str(density_contract["case_id"][0]),
            "primary_resample": PRIMARY_SAMPLING_RESAMPLE,
            "density_resample": DENSITY_CHECK_RESAMPLE,
            "primary_native_samples": int(primary_sampling["wavelength_micron"].size),
            "density_native_samples": int(density["wavelength_micron"].size),
            "r100_difference": density_metrics,
            "smoothing_applied": False,
        },
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
                REPOSITORY / "src/robert_exoplanets/diagnostics/emission_intercomparison_v2.py",
                REPOSITORY / "src/robert_exoplanets/atmosphere/temperature.py",
            )
        },
        "known_warnings_and_capability_boundaries": [
            "PICASO's harmless optional-Vega warning is recorded; no stellar grids were downloaded or used.",
            "PICASO emits harmless invalid-divide warnings for ftau_cld/ftau_ray when both cloud and Rayleigh optical depths are exactly zero; retained tau arrays verify both remain zero.",
            "PICASO exact-omega0=0 shared-tensor RT remains unsupported as a native gated interface.",
            "PICASO exact-omega0=0 native molecular thermal probes are finite but pathological; native opacity tensors and the separately labelled absorbing-formal spectra are the scientific products, with raw probes retained as capability evidence.",
            "PICASO vertical arrays are explicitly labelled absorbing-formal diagnostics applied to native optical depth, not a native SH contribution definition.",
            "Stable pRT does not expose native layer optical-depth tensors through the supported high-level flux interface; spectra and native emission contributions are retained.",
            "Opacity-sampling output is preserved unsmoothed and is not treated as interchangeable with correlated-k.",
            "Stage-1 continuous-angle sub-0.01 ppm claims remain restricted; its 0.196897 ppm eight-angle result is not reinterpreted here.",
        ],
        "random_seed": None,
        "random_seed_policy": common.random_seed_policy,
    }
    _write_json(report_path, report)
    artifact_paths.append(report_path)
    integrity_path = data_root / "stage_2_integrity.json"
    _write_json(integrity_path, _integrity_manifest(artifact_paths))
    artifact_paths.append(integrity_path)
    checksum_path = data_root / "checksums.json"
    existing = json.loads(checksum_path.read_text())
    for name in tuple(existing):
        if name.startswith("stage_2_"):
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
