"""Run one isolated Version-2 Stage-8 Track-B timing pilot.

The worker executes exactly one frozen 80-cell native case.  It is not a
production-matrix runner and intentionally has no Track-A interface.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Any, Callable
import warnings

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROBERT_PYTHON = Path("/opt/miniconda3/envs/robert-exoplanets/bin/python")
PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso-v4/bin/python")
PRT_PYTHON = Path("/opt/miniconda3/envs/petitradtrans-stable/bin/python")
EXPECTED = {"robert": ROBERT_PYTHON, "picaso": PICASO_PYTHON, "petitradtrans": PRT_PYTHON}
PICASO_CK_DIRECTORY = Path(
    "/Users/jaketaylor/Dropbox/picaso/reference/opacities/resortrebin"
)
PRT_INPUT_DATA = Path(
    "/Users/jaketaylor/Dropbox/ROBERT-code/opacity_data/petitRADTRANS/input_data"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _spectral_values(wavelength: np.ndarray, base: float, *, varying: bool) -> np.ndarray:
    if not varying:
        return np.full(wavelength.size, base)
    phase = 2.0 * np.pi * np.log(wavelength / 0.3) / np.log(12.0 / 0.3)
    if base <= 0.0:
        return np.zeros(wavelength.size)
    return np.clip(base * (0.82 + 0.18 * np.cos(phase)), 0.0, 0.999)


def _cloud_tau(contract: dict[str, np.ndarray], wavelength: np.ndarray) -> np.ndarray:
    source_x = contract["cloud_input_wavelength_micron"]
    source = contract["cloud_input_extinction_tau"][int(contract["case_cloud_index"][0])]
    return np.stack([np.interp(wavelength, source_x, row) for row in source])


def _gas_contract(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    profile_name = str(contract["profile_name"][0])
    return {
        "schema_version": np.array("2.0.0"),
        "stage": np.array(4),
        "case_id": np.array([f"{profile_name}_molecular_plus_cia"]),
        "profile_name": np.array([profile_name]),
        "profile_index": np.array([0]),
        "factor_name": np.array(["molecular_plus_h2_h2_and_h2_he_cia"]),
        "include_h2_h2_cia": np.ones(1, dtype=bool),
        "include_h2_he_cia": np.ones(1, dtype=bool),
        "gas_name": contract["gas_name"],
        "gas_mass_u": contract["gas_mass_u"],
        "gas_vmr": contract["gas_vmr"],
        "mean_molecular_weight_u": contract["mean_molecular_weight_u"],
        "molecular_species_name": contract["molecular_species_name"],
        "molecular_species_active": np.ones((1, 4), dtype=bool),
        "pressure_edges_bar": contract["pressure_edges_bar"],
        "pressure_centers_bar": contract["pressure_centers_bar"],
        "picaso_pressure_levels_bar": contract["picaso_pressure_levels_bar"],
        "petitradtrans_pressure_nodes_bar": contract["petitradtrans_pressure_nodes_bar"],
        "temperature_edges_k": contract["temperature_edges_k"],
        "temperature_cells_k": contract["temperature_cells_k"],
        "temperature_edges_by_profile_k": contract["temperature_edges_k"],
        "temperature_cells_by_profile_k": contract["temperature_cells_k"],
        "gravity_m_s2": contract["gravity_m_s2"],
        "emission_mu": contract["emission_mu"],
        "legendre_weights": contract["legendre_weights"],
        "disk_weights": contract["disk_weights"],
    }


def _robert_mgsio3_mie(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate only the shared MgSiO3 population optics and exact moments."""

    from robert_exoplanets import mie_efficiencies, mie_phase_function_moments

    wavelength = contract["wavelength_micron"]
    radius = contract["radius_cm"]
    upper = contract["radius_upper_cm"]
    weights = contract["radius_number_weights"]
    qext = np.zeros((wavelength.size, radius.size))
    qsca = np.zeros_like(qext)
    g_qsca = np.zeros_like(qext)
    phase_numerator = np.zeros((5, wavelength.size))
    area_weight = weights * np.pi * (radius * 1.0e-2) ** 2
    for wave_index, (wave, real, imaginary) in enumerate(
        zip(
            wavelength,
            contract["refractive_index_n"],
            contract["refractive_index_k"],
            strict=True,
        )
    ):
        for radius_index in range(radius.size):
            lower = radius[0] if radius_index == 0 else upper[radius_index - 1]
            values = []
            moment_values = []
            for subradius in np.linspace(lower, upper[radius_index], 6):
                size = 2.0 * np.pi * subradius * 1.0e4 / wave
                ext, sca, asymmetry = mie_efficiencies(
                    size, complex(real, imaginary)
                )
                values.append((ext, sca, asymmetry * sca))
                moment_values.append(
                    sca
                    * mie_phase_function_moments(
                        size, complex(real, imaginary), order=5
                    )
                )
            qext[wave_index, radius_index], qsca[wave_index, radius_index], g_qsca[
                wave_index, radius_index
            ] = np.mean(values, axis=0)
            phase_numerator[:, wave_index] += area_weight[radius_index] * np.mean(
                moment_values, axis=0
            )
    radius_m = radius * 1.0e-2
    area = np.pi * radius_m**2
    mass = (
        (4.0 / 3.0)
        * np.pi
        * float(contract["particle_density_kg_m3"])
        * radius_m**3
    )
    mean_mass = np.sum(weights * mass)
    area_extinction = np.sum(qext * (weights * area)[None, :], axis=1)
    area_scattering = np.sum(qsca * (weights * area)[None, :], axis=1)
    scattering_g = np.sum(g_qsca * (weights * area)[None, :], axis=1)
    omega = np.divide(
        area_scattering,
        area_extinction,
        out=np.zeros_like(area_scattering),
        where=area_extinction > 0.0,
    )
    asymmetry = np.divide(
        scattering_g,
        area_scattering,
        out=np.zeros_like(scattering_g),
        where=area_scattering > 0.0,
    )
    exact_moments = np.divide(
        phase_numerator,
        area_scattering[None, :],
        out=np.zeros_like(phase_numerator),
        where=area_scattering[None, :] > 0.0,
    )
    exact_moments[0] = 1.0
    return {
        "single_scattering_albedo": omega,
        "asymmetry_factor": asymmetry,
        "exact_phase_function_moments": exact_moments,
        "mass_extinction_m2_kg": area_extinction / mean_mass,
    }


def _robert(
    contract: dict[str, np.ndarray], path: str, omega0: float, tau: float, g: float, delta_m: bool | None
) -> tuple[dict[str, np.ndarray], float, float, int, int]:
    sys.path.insert(0, str(ROOT / "src"))
    stage3 = _load_module("stage3_for_stage8_worker", HERE / "benchmark_emission_intercomparison_v2_stage_3.py")
    from robert_exoplanets import integrate_thermal_emission, planck_radiance_wavelength
    from robert_exoplanets.rt import solve_thermal_sh4_spectrum, solve_thermal_two_stream

    original_profiles = stage3.PROFILES
    stage3.PROFILES = (str(contract["profile_name"][0]),)
    setup_started = perf_counter()
    try:
        gas = stage3._run_robert_native(_gas_contract(contract), stage3._opacity_paths())
    finally:
        stage3.PROFILES = original_profiles
    wavelength = np.asarray(gas["wavelength_micron"], dtype=float)
    gas_tau = (
        gas["molecular_layer_tau_by_profile"][0].astype(float)
        + gas["cia_h2_h2_layer_tau_by_profile"][0, :, :, None].astype(float)
        + gas["cia_h2_he_layer_tau_by_profile"][0, :, :, None].astype(float)
    )
    native_cloud = _cloud_tau(contract, wavelength) * tau
    varying = path.startswith("spectral_")
    native_omega = _spectral_values(wavelength, omega0, varying=varying)
    native_g = _spectral_values(wavelength, g, varying=varying)
    physical_seconds = 0.0
    phase_moments = None
    if path == "physical_mie_exact_moments_sh4":
        with np.load(ROOT / "data/validation/end_to_end_cloud_parity/shared_physical_contract.npz", allow_pickle=False) as archive:
            mie_contract = {name: np.array(archive[name], copy=True) for name in archive.files}
        mie_started = perf_counter()
        mie = _robert_mgsio3_mie(mie_contract)
        physical_seconds = perf_counter() - mie_started
        native_omega = np.interp(wavelength, mie_contract["wavelength_micron"], mie["single_scattering_albedo"])
        native_g = np.interp(wavelength, mie_contract["wavelength_micron"], mie["asymmetry_factor"])
        moments = np.stack([
            np.interp(wavelength, mie_contract["wavelength_micron"], row)
            for row in mie["exact_phase_function_moments"][:4]
        ])
        phase_moments = np.broadcast_to(moments[:, None, :, None], (4, gas_tau.shape[0], wavelength.size, gas_tau.shape[2])).copy()
    cloud_extinction = native_cloud[:, :, None]
    scattering = cloud_extinction * native_omega[None, :, None]
    extinction = gas_tau + cloud_extinction
    total_omega = np.divide(scattering, extinction, out=np.zeros_like(extinction), where=extinction > 0.0)
    total_g = np.broadcast_to(native_g[None, :, None], extinction.shape).copy()
    levels = np.stack([planck_radiance_wavelength(wavelength, value) for value in contract["temperature_edges_k"][0]])
    setup_s = perf_counter() - setup_started - physical_seconds
    case_started = perf_counter()
    solver_error = ""
    try:
        if path == "native_absorption":
            result = integrate_thermal_emission(
                extinction,
                np.stack(
                    [
                        planck_radiance_wavelength(wavelength, value)
                        for value in contract["temperature_cells_k"][0]
                    ]
                ),
                gas["g_weights"],
                np.broadcast_to(
                    1.0 / contract["emission_mu"][:, None],
                    (contract["emission_mu"].size, extinction.shape[0]),
                ),
                level_source_ordered=levels,
                bottom_source=levels[-1],
                backend="numpy",
            )
            spectrum = np.einsum(
                "a,as->s",
                contract["disk_weights"],
                np.sum(result.point_layer_contribution_radiance, axis=1)
                + result.point_bottom_contribution_radiance,
            )
        elif path.startswith("toon"):
            result = solve_thermal_two_stream(extinction, total_omega, total_g, levels, contract["emission_mu"], bottom_planck_radiance=levels[-1])
            spectrum = np.einsum("a,asg,g->s", contract["disk_weights"], result.point_radiance, gas["g_weights"])
        else:
            result = solve_thermal_sh4_spectrum(
                extinction, total_omega, total_g, levels, contract["emission_mu"], contract["disk_weights"], gas["g_weights"],
                bottom_planck_radiance=levels[-1], phase_function_moments=phase_moments,
                delta_m=bool(delta_m), backend="numpy",
            )
            spectrum = result.radiance
    except Exception as exc:  # Native failure is a required pilot outcome.
        solver_error = f"{type(exc).__name__}: {exc}"
        spectrum = np.full(wavelength.size, np.nan)
    case_s = perf_counter() - case_started + physical_seconds
    retained = int(extinction.nbytes + total_omega.nbytes + total_g.nbytes + spectrum.nbytes)
    flux = np.pi * spectrum
    return {"wavelength_micron": wavelength, "spectrum": spectrum, "flux_w_m2_m": flux, "extinction_tau": extinction, "omega0": total_omega, "g": total_g, "solver_error": np.array(solver_error)}, setup_s, case_s, gas["g_weights"].size, retained


def _cloud_frame(pd: Any, contract: dict[str, np.ndarray], omega0: float, g: float, varying: bool) -> Any:
    pressure = contract["pressure_centers_bar"]
    wavelength = contract["cloud_input_wavelength_micron"]
    extinction = contract["cloud_input_extinction_tau"][int(contract["case_cloud_index"][0])]
    omega = _spectral_values(wavelength, omega0, varying=varying)
    asymmetry = _spectral_values(wavelength, g, varying=varying)
    rows = [(pressure[layer], 1.0e4 / wavelength[wave], extinction[layer, wave], omega[wave], asymmetry[wave]) for layer in range(pressure.size) for wave in range(wavelength.size)]
    return pd.DataFrame(rows, columns=("pressure", "wavenumber", "opd", "w0", "g0"))


def _picaso(
    contract: dict[str, np.ndarray], path: str, omega0: float, tau: float, g: float, delta_m: bool | None
) -> tuple[dict[str, np.ndarray], float, float, int, int]:
    stage3w = _load_module("stage3w_for_stage8", HERE / "run_emission_intercomparison_v2_stage_3_external.py")
    stage3w._validate_picaso_environment()
    import astropy.units as u
    import pandas as pd
    from picaso import justdoit as jdi

    setup_started = perf_counter()
    opacity = jdi.opannection(method="resortrebin", ck_db=str(PICASO_CK_DIRECTORY), preload_gases=list(stage3w.MOLECULAR_SPECIES), wave_range=[0.3, 12.0], verbose=False)
    stage3w._restore_resort_rebin_absolute_vmr(opacity)
    observed: set[str] = set()
    stage3w._filter_picaso_continuum(opacity, ("H2-H2", "H2-He"), observed)
    profile: dict[str, Any] = {"pressure": contract["pressure_edges_bar"], "temperature": contract["temperature_edges_k"][0]}
    for index, name in enumerate(contract["gas_name"]):
        profile[str(name)] = np.full(contract["pressure_edges_bar"].size, contract["gas_vmr"][0, index])
    case = jdi.inputs(calculation="browndwarf")
    case.gravity(gravity=float(contract["gravity_m_s2"]), gravity_unit=u.m / u.s**2)
    case.atmosphere(df=pd.DataFrame(profile), verbose=False)
    rt_method = "toon" if path.startswith("toon") else "SH"
    stream = 2 if rt_method == "toon" else 4
    case.approx(rt_method=rt_method, stream=stream, delta_eddington=bool(delta_m), raman="none", w_single_rayleigh="off", w_multi_rayleigh="off", psingle_rayleigh="off")
    physical_seconds = 0.0
    varying = path.startswith("spectral_")
    if path == "physical_mie_virga_hg_sh4":
        from virga.calc_mie import calc_new_mieff
        virga_runner = _load_module("virga_for_stage8", HERE / "run_picaso_virga_cloud_parity.py")
        with np.load(ROOT / "data/validation/end_to_end_cloud_parity/shared_physical_contract.npz", allow_pickle=False) as archive:
            shared = {name: np.array(archive[name], copy=True) for name in archive.files}
        mie_started = perf_counter()
        radius_lower = np.concatenate(
            (shared["radius_cm"][:1], shared["radius_upper_cm"][:-1])
        )
        qext, qsca, gqsca = calc_new_mieff(
            shared["wavelength_micron"],
            shared["refractive_index_n"],
            shared["refractive_index_k"],
            shared["radius_cm"],
            radius_lower,
            shared["radius_upper_cm"],
            fort_calc_mie=False,
        )
        mie = virga_runner._integrate_discrete_population(qext, qsca, gqsca, shared["radius_cm"], shared["radius_number_weights"], float(shared["particle_density_kg_m3"]))
        physical_seconds = perf_counter() - mie_started
        wx = contract["cloud_input_wavelength_micron"]
        omega_arr = np.interp(wx, shared["wavelength_micron"], mie["single_scattering_albedo"])
        g_arr = np.interp(wx, shared["wavelength_micron"], mie["asymmetry_factor"])
        frame = _cloud_frame(pd, contract, 0.0, 0.0, False)
        frame["w0"] = np.tile(omega_arr, contract["pressure_centers_bar"].size)
        frame["g0"] = np.tile(g_arr, contract["pressure_centers_bar"].size)
    else:
        frame = _cloud_frame(pd, contract, omega0, g, varying)
    frame["opd"] *= tau
    case.clouds(df=frame)
    setup_s = perf_counter() - setup_started - physical_seconds
    case_started = perf_counter()
    result = case.spectrum(opacity, calculation="thermal", full_output=True)
    case_s = perf_counter() - case_started + physical_seconds
    wavelength = 1.0e4 / np.asarray(result["wavenumber"], dtype=float)
    order = np.argsort(wavelength)
    spectrum = np.asarray(result["thermal"], dtype=float)[order] * 0.1
    full = result["full_output"]
    tensors = [np.asarray(full[name]) for name in ("taugas", "taucld", "w0", "cosb") if name in full]
    retained = int(sum(item.nbytes for item in tensors) + spectrum.nbytes)
    return {"wavelength_micron": wavelength[order], "spectrum": spectrum, "flux_w_m2_m": spectrum, "native_tensor": tensors[0]}, setup_s, case_s, int(np.asarray(opacity.gauss_wts).size), retained


def _interp_callback(contract: dict[str, np.ndarray], tau: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    input_wavelength = contract["cloud_input_wavelength_micron"]
    input_pressure = contract["pressure_centers_bar"]
    dp = np.diff(contract["pressure_edges_bar"]) * 1.0e6
    opacity = contract["cloud_input_extinction_tau"][int(contract["case_cloud_index"][0])] * tau * float(contract["gravity_m_s2"]) * 100.0 / dp[:, None]
    def callback(wavelength: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        spectral = np.stack([np.interp(wavelength, input_wavelength, row) for row in opacity])
        if np.array_equal(pressure, input_pressure):
            return spectral.T
        return np.stack([np.interp(pressure, input_pressure, spectral[:, index]) for index in range(wavelength.size)])
    return callback


def _petitradtrans(
    contract: dict[str, np.ndarray], path: str, omega0: float, tau: float, g: float, delta_m: bool | None
) -> tuple[dict[str, np.ndarray], float, float, int, int]:
    del g, delta_m
    stage3w = _load_module("stage3w_for_stage8_prt", HERE / "run_emission_intercomparison_v2_stage_3_external.py")
    from petitRADTRANS.radtrans import Radtrans
    angle_count = 32 if path.endswith("32_angles") else 16 if path.endswith("16_angles") else contract["emission_mu"].size
    nodes, weights = np.polynomial.legendre.leggauss(angle_count)
    angle_grid = np.vstack((0.5 * (nodes + 1.0), 0.5 * weights))
    scattering = omega0 > 0.0 and path != "native_ck_absorption"
    setup_started = perf_counter()
    atmosphere = Radtrans(
        pressures=contract["pressure_centers_bar"], wavelength_boundaries=np.array([0.3, 12.1]),
        line_species=list(stage3w.PRT_LINE_SPECIES.values()), gas_continuum_contributors=list(stage3w.PRT_CIA_SPECIES.values()),
        rayleigh_species=[], cloud_species=[], scattering_in_emission=scattering,
        anisotropic_cloud_scattering=False, emission_angle_grid=angle_grid,
        path_input_data=str(PRT_INPUT_DATA),
    )
    names = [str(value) for value in contract["gas_name"]]
    n_cells = int(contract["pressure_centers_bar"].size)
    vmr = contract["gas_vmr"][0]
    mmw = float(contract["mean_molecular_weight_u"][0])
    fractions = vmr * contract["gas_mass_u"] / mmw
    mass_fractions = {line: np.full(n_cells, fractions[names.index(molecule)]) for molecule, line in stage3w.PRT_LINE_SPECIES.items()}
    mass_fractions["H2"] = np.full(n_cells, fractions[names.index("H2")])
    mass_fractions["He"] = np.full(n_cells, fractions[names.index("He")])
    total_callback = _interp_callback(contract, tau)
    def absorption(wavelength: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        values = total_callback(wavelength, pressure)
        spectral_omega = _spectral_values(np.asarray(wavelength), omega0, varying=path.startswith("spectral_"))
        return values * (1.0 - spectral_omega[:, None])
    def scattering_callback(wavelength: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        values = total_callback(wavelength, pressure)
        spectral_omega = _spectral_values(np.asarray(wavelength), omega0, varying=path.startswith("spectral_"))
        return values * spectral_omega[:, None]
    setup_s = perf_counter() - setup_started
    case_started = perf_counter()
    result = atmosphere.calculate_flux(
        temperatures=contract["temperature_cells_k"][0], mass_fractions=mass_fractions,
        mean_molar_masses=np.full(n_cells, mmw), reference_gravity=float(contract["gravity_m_s2"]) * 100.0,
        additional_absorption_opacities_function=absorption,
        additional_scattering_opacities_function=scattering_callback if scattering else None,
        frequencies_to_wavelengths=True, return_contribution=True,
    )
    case_s = perf_counter() - case_started
    wavelength = np.asarray(result[0]) * 1.0e4
    order = np.argsort(wavelength)
    spectrum = np.asarray(result[1])[order] * 0.1
    contribution = np.asarray(result[2]["emission_contribution"])
    retained = int(spectrum.nbytes + contribution.nbytes)
    return {"wavelength_micron": wavelength[order], "spectrum": spectrum, "flux_w_m2_m": spectrum, "native_tensor": contribution}, setup_s, case_s, angle_count, retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=tuple(EXPECTED))
    parser.add_argument("path")
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--omega0", type=float, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--g", type=float, required=True)
    parser.add_argument("--delta-m", choices=("none", "off", "on"), required=True)
    parser.add_argument("--study-status", default="future_study")
    parser.add_argument("--cloud-state", default="representative_pilot")
    args = parser.parse_args()
    if os.path.realpath(sys.executable) != os.path.realpath(EXPECTED[args.framework]):
        raise RuntimeError(f"{args.framework} must run with {EXPECTED[args.framework]}")
    if args.framework == "picaso":
        for name in ("picaso_refdata", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
            if not os.environ.get(name):
                raise RuntimeError(f"PICASO requires {name} before import")
        if os.path.realpath(sys.executable) == "/opt/miniconda3/envs/picaso/bin/python":
            raise RuntimeError("historical PICASO interpreter is forbidden")
    contract = _load(args.contract)
    if int(contract["pressure_centers_bar"].size) not in {40, 80, 160} or contract["case_id"].size != 1:
        raise ValueError("Stage-8 worker accepts exactly one 40/80/160-cell case")
    delta_m = None if args.delta_m == "none" else args.delta_m == "on"
    runner = {"robert": _robert, "picaso": _picaso, "petitradtrans": _petitradtrans}[args.framework]
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arrays, setup_s, case_s, native_order, retained = runner(contract, args.path, args.omega0, args.tau, args.g, delta_m)
    metadata = {
        "schema_version": "2.0.0", "stage": 8, "track": "track_b_native_scattering",
        "study_status": args.study_status, "cloud_state": args.cloud_state,
        "profile": str(contract["profile_name"][0]),
        "pressure_cell_count": int(contract["pressure_centers_bar"].size),
        "framework": args.framework, "path": args.path, "python": os.path.realpath(sys.executable),
        "package_version": importlib.metadata.version({"robert": "robert-exoplanets", "picaso": "picaso", "petitradtrans": "petitRADTRANS"}[args.framework]),
        "numpy_version": importlib.metadata.version("numpy"), "platform": platform.platform(),
        "wall_time_s": perf_counter() - started, "setup_time_s": setup_s, "case_time_s": case_s,
        "peak_rss_bytes": _peak_rss_bytes(), "native_wavelength_count": int(arrays["wavelength_micron"].size),
        "native_bin_count": int(arrays["wavelength_micron"].size), "native_g_or_angle_or_stream_count": native_order,
        "retained_tensor_bytes": retained, "finite_spectrum": bool(np.all(np.isfinite(arrays["spectrum"]))),
        "omega0": args.omega0, "tau": args.tau, "g": args.g, "delta_m": delta_m,
        "warnings": [str(item.message) for item in caught],
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
