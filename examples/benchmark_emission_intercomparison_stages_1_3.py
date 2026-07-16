"""Run emission-intercomparison Stages 1--3 in isolated environments.

ROBERT orchestrates the benchmark from its own environment.  PICASO and
petitRADTRANS are always launched as external workers with explicit Python
interpreters and exchange only NPZ/JSON artifacts with the orchestrator.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

from emission_intercomparison_common import (
    SPECIES,
    bin_mean,
    difference_metrics,
    pairwise_metrics,
    planck_flux_w_m2_m,
    r100_edges,
    stage_1_contract,
    stage_2_contract,
    stage_3_contract,
    write_checksums,
    write_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("run_emission_intercomparison_external.py")
DEFAULT_OUTPUT = Path(__file__).parent / "outputs/emission_intercomparison"
DEFAULT_REPORTS = REPOSITORY / "docs/data/emission_intercomparison"
DEFAULT_PICASO_PYTHON = Path("/opt/miniconda3/envs/picaso/bin/python")
DEFAULT_PRT_PYTHON = Path(
    "/opt/miniconda3/envs/petitradtrans-stable/bin/python"
)
DEFAULT_PICASO_REFERENCE = (
    REPOSITORY / "opacity_data/picaso_official/reference_v3_2"
)
DEFAULT_PICASO_DATABASE = (
    REPOSITORY
    / "opacity_data/picaso_official/reference/opacities"
    / "opacities_0.3_15_R15000.db"
)
DEFAULT_PRT_INPUT = REPOSITORY / "opacity_data/petitRADTRANS/input_data"
RESOLUTIONS = (40, 80, 160)


def _robert_metadata() -> dict[str, str]:
    return {
        "model": "robert",
        "version": importlib.metadata.version("robert-exoplanets"),
        "python": os.sys.executable,
    }


def _save_contract(path: Path, contract: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **contract)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _run_worker(
    python: Path,
    model: str,
    mode: str,
    contract: Path,
    output: Path,
    *,
    picaso_reference: Path,
    picaso_database: Path,
    prt_input: Path,
    picaso_resample: int,
) -> None:
    if not python.is_file():
        raise FileNotFoundError(f"external interpreter not found: {python}")
    command = [
        str(python),
        str(WORKER),
        model,
        mode,
        str(contract),
        str(output),
    ]
    if model == "picaso" and mode == "native":
        command.extend(
            [
                "--picaso-reference",
                str(picaso_reference),
                "--picaso-database",
                str(picaso_database),
                "--picaso-resample",
                str(picaso_resample),
            ]
        )
    elif model == "petitradtrans" and mode == "native":
        command.extend(["--input-data", str(prt_input)])
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_btl", "self")
    if model == "picaso" and mode == "native":
        environment["picaso_refdata"] = str(picaso_reference.resolve())
    if model == "petitradtrans":
        worker_home = contract.parent / ".petitradtrans-worker-home"
        worker_home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(worker_home)
    subprocess.run(command, check=True, env=environment)


def _shared_robert(contract: dict[str, np.ndarray]) -> np.ndarray:
    from robert_exoplanets import planck_radiance_wavelength
    from robert_exoplanets.rt import integrate_thermal_emission_spectrum

    wavelength = contract["wavelength_micron"]
    mu = contract["emission_mu"]
    point_weights = contract["disk_weights"]
    total_tau = np.sum(contract["component_tau"], axis=1)
    flux = np.empty((total_tau.shape[0], wavelength.size))
    for index, tau in enumerate(total_tau):
        level_temperature = contract["temperature_edges_k"][index]
        layer_temperature = 0.5 * (
            level_temperature[:-1] + level_temperature[1:]
        )
        layer_source = np.stack(
            [
                planck_radiance_wavelength(wavelength, temperature)
                for temperature in layer_temperature
            ]
        )
        level_source = np.stack(
            [
                planck_radiance_wavelength(wavelength, temperature)
                for temperature in level_temperature
            ]
        )
        result = integrate_thermal_emission_spectrum(
            tau[:, :, None],
            layer_source,
            np.array([1.0]),
            np.broadcast_to(1.0 / mu[:, None], (mu.size, tau.shape[0])),
            point_weights,
            level_source_ordered=level_source,
            bottom_source=level_source[-1],
            backend="numpy",
        )
        flux[index] = np.pi * result.radiance
    return flux


def _shared_stage(
    stage: int,
    resolutions: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage_dir = args.output_root / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    maximum_pairwise_relative = 0.0
    maximum_pairwise_ppm = 0.0
    maximum_analytic_relative = 0.0
    maximum_pairwise_p95 = {n_layers: 0.0 for n_layers in resolutions}
    per_resolution: dict[str, Any] = {}
    flux_by_model: dict[str, dict[int, np.ndarray]] = {
        "robert": {},
        "picaso": {},
        "petitradtrans": {},
    }
    for n_layers in resolutions:
        contract = (
            stage_1_contract(n_layers)
            if stage == 1
            else stage_2_contract(n_layers)
        )
        contract_path = stage_dir / f"contract_L{n_layers}.npz"
        _save_contract(contract_path, contract)
        worker_outputs = {}
        for model, python in (
            ("picaso", args.picaso_python),
            ("petitradtrans", args.prt_python),
        ):
            output = stage_dir / f"{model}_L{n_layers}.npz"
            _run_worker(
                python,
                model,
                "shared",
                contract_path,
                output,
                picaso_reference=args.picaso_reference,
                picaso_database=args.picaso_database,
                prt_input=args.prt_input,
                picaso_resample=args.picaso_resample,
            )
            worker_outputs[model] = _load(output)
        spectra = {
            "robert": _shared_robert(contract),
            "picaso": worker_outputs["picaso"]["flux_w_m2_m"],
            "petitradtrans": worker_outputs["petitradtrans"]["flux_w_m2_m"],
        }
        for model, values in spectra.items():
            flux_by_model[model][n_layers] = values
        case_metrics = []
        for case_index, case_id in enumerate(contract["case_id"]):
            current = {name: values[case_index] for name, values in spectra.items()}
            metrics = pairwise_metrics(current, contract["wavelength_micron"])
            maximum_pairwise_relative = max(
                maximum_pairwise_relative,
                *(item["max_abs_symmetric_relative"] for item in metrics.values()),
            )
            maximum_pairwise_ppm = max(
                maximum_pairwise_ppm,
                *(item["max_abs_eclipse_difference_ppm"] for item in metrics.values()),
            )
            maximum_pairwise_p95[n_layers] = max(
                maximum_pairwise_p95[n_layers],
                *(item["p95_abs_symmetric_relative"] for item in metrics.values()),
            )
            entry: dict[str, Any] = {
                "case_id": str(case_id),
                "pairwise": metrics,
            }
            if stage == 1:
                temperature = float(contract["temperature_edges_k"][case_index, 0])
                analytic = planck_flux_w_m2_m(
                    contract["wavelength_micron"], temperature
                )
                analytic_metrics = {
                    name: difference_metrics(
                        values[case_index], analytic, contract["wavelength_micron"]
                    )
                    for name, values in spectra.items()
                }
                maximum_analytic_relative = max(
                    maximum_analytic_relative,
                    *(
                        item["max_abs_symmetric_relative"]
                        for item in analytic_metrics.values()
                    ),
                )
                entry["versus_analytic"] = analytic_metrics
            case_metrics.append(entry)
        per_resolution[str(n_layers)] = {
            "case_count": len(case_metrics),
            "cases": case_metrics,
            "external_metadata": {
                name: json.loads(str(payload["metadata_json"]))
                for name, payload in worker_outputs.items()
            },
        }
        np.savez_compressed(
            stage_dir / f"combined_L{n_layers}.npz",
            case_id=contract["case_id"],
            wavelength_micron=contract["wavelength_micron"],
            **{f"flux_{name}_w_m2_m": values for name, values in spectra.items()},
        )
    self_convergence: dict[str, Any] = {}
    for model, by_resolution in flux_by_model.items():
        model_metrics = {}
        for coarse, fine in zip(resolutions[:-1], resolutions[1:], strict=True):
            pair = []
            for coarse_flux, fine_flux in zip(
                by_resolution[coarse], by_resolution[fine], strict=True
            ):
                pair.append(
                    difference_metrics(
                        coarse_flux,
                        fine_flux,
                        (
                            stage_1_contract(coarse)
                            if stage == 1
                            else stage_2_contract(coarse)
                        )["wavelength_micron"],
                    )
                )
            model_metrics[f"{coarse}_to_{fine}"] = {
                "max_abs_symmetric_relative": max(
                    item["max_abs_symmetric_relative"] for item in pair
                ),
                "max_abs_eclipse_difference_ppm": max(
                    item["max_abs_eclipse_difference_ppm"] for item in pair
                ),
            }
        self_convergence[model] = model_metrics
    gates = {"max_pairwise_eclipse_difference_ppm": 3.0}
    if stage == 1:
        gates["max_pairwise_symmetric_relative"] = 1.0e-4
        gates["max_analytic_symmetric_relative"] = 5.0e-5
        passed = (
            maximum_pairwise_relative <= gates["max_pairwise_symmetric_relative"]
            and maximum_pairwise_ppm
            <= gates["max_pairwise_eclipse_difference_ppm"]
            and maximum_analytic_relative
            <= gates["max_analytic_symmetric_relative"]
        )
    else:
        gates["primary_resolution"] = resolutions[-1]
        gates["max_primary_p95_symmetric_relative"] = 5.0e-3
        gates["max_finest_convergence_eclipse_difference_ppm"] = 3.0
        finest_pair = f"{resolutions[-2]}_to_{resolutions[-1]}"
        finest_convergence_ppm = max(
            values[finest_pair]["max_abs_eclipse_difference_ppm"]
            for values in self_convergence.values()
        )
        passed = (
            maximum_pairwise_p95[resolutions[-1]]
            <= gates["max_primary_p95_symmetric_relative"]
            and maximum_pairwise_ppm
            <= gates["max_pairwise_eclipse_difference_ppm"]
            and finest_convergence_ppm
            <= gates["max_finest_convergence_eclipse_difference_ppm"]
        )
    return {
        "schema_version": 1,
        "stage": stage,
        "track": "A_shared_optical_depth",
        "status": "pass" if passed else "fail",
        "orchestrator": _robert_metadata(),
        "resolutions": list(resolutions),
        "maximum_pairwise_symmetric_relative": maximum_pairwise_relative,
        "maximum_pairwise_p95_symmetric_relative_by_resolution": {
            str(key): value for key, value in maximum_pairwise_p95.items()
        },
        "maximum_pairwise_eclipse_difference_ppm": maximum_pairwise_ppm,
        "maximum_analytic_symmetric_relative": (
            maximum_analytic_relative if stage == 1 else None
        ),
        "gates": gates,
        "self_convergence": self_convergence,
        "per_resolution": per_resolution,
    }


def _native_robert(
    contract: dict[str, np.ndarray], input_data: Path
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

    patterns = {
        "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
        "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
        "CO2": "*UCL-4000*.ktable.petitRADTRANS.h5",
        "CH4": "*YT34to10*.ktable.petitRADTRANS.h5",
    }
    pressure_edges = contract["pressure_edges_bar"]
    pressure_centers = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    temperature_edges = contract["temperature_edges_k"][0]
    temperature_centers = 0.5 * (temperature_edges[:-1] + temperature_edges[1:])
    vmr = contract["gas_vmr"][0]
    composition = dict(zip(("H2", "He", *SPECIES), vmr, strict=True))
    mean_molecular_weight = sum(
        composition[name] * mass
        for name, mass in zip(
            ("H2", "He", *SPECIES),
            (2.01588, 4.002602, 18.01528, 28.0101, 44.0095, 16.04246),
            strict=True,
        )
    )
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_centers,
        unit="bar",
        name="emission_intercomparison_cells",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature_centers,
        temperature_edges=temperature_edges,
        composition={
            name: np.full(pressure_centers.size, value)
            for name, value in composition.items()
        },
        mean_molecular_weight=np.full(pressure_centers.size, mean_molecular_weight),
    )
    first_path = next(input_data.rglob(patterns["H2O"]))
    first_table = CorrelatedKTable.from_petitradtrans_hdf(first_path, species="H2O")
    mask = (first_table.wavelength_micron >= 0.5) & (
        first_table.wavelength_micron <= 12.0
    )
    wavelength = np.sort(first_table.wavelength_micron[mask])
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="pRT-R1000"
    )
    evaluated = np.empty(
        (
            len(SPECIES),
            pressure_centers.size,
            wavelength.size,
            first_table.g_weights.size,
        )
    )
    tables = {"H2O": first_table}
    for species_index, species in enumerate(SPECIES):
        table = tables.pop(species, None)
        if table is None:
            table = CorrelatedKTable.from_petitradtrans_hdf(
                next(input_data.rglob(patterns[species])), species=species
            )
        provider = CorrelatedKOpacityProvider(
            {species: table},
            name=f"emission-intercomparison-{species}",
            interpolation="log_pressure_temperature_log_k",
        )
        prepared_one = provider.prepare(
            spectral_grid, pressure_grid, species=(species,)
        )
        evaluated[species_index] = provider.evaluate(
            atmosphere, prepared_one
        ).kcoeff[0]
        del provider, prepared_one, table
        gc.collect()
    prepared = PreparedCorrelatedKOpacity(
        provider_name="pRT-HDF-four-species",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=SPECIES,
        g_samples=first_table.g_samples,
        g_weights=first_table.g_weights,
        cache_key=f"stage3-{pressure_centers.size}",
        metadata={"interpolation": "log_pressure_temperature_log_k"},
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=evaluated,
        unit="cm^2/molecule",
        metadata={"source": "petitRADTRANS HDF5 correlated-k tables"},
    )
    gas_tau = assemble_gas_optical_depth(
        atmosphere, opacity, gravity_m_s2=15.0, gas_combination="random_overlap"
    )
    cia_tables = (
        CiaTable.from_petitradtrans_hdf(
            next(input_data.rglob("*H2--H2*.ciatable.petitRADTRANS.h5")),
            collision_pair="H2-H2",
        ),
        CiaTable.from_petitradtrans_hdf(
            next(input_data.rglob("*H2--He*.ciatable.petitRADTRANS.h5")),
            collision_pair="H2-He",
        ),
    )
    cia = [
        cia_optical_depth(
            gas_tau,
            table,
            coefficient_interpolation="log",
            temperature_extrapolation="clip",
            spectral_extrapolation="zero",
        )
        for table in cia_tables
    ]
    started = perf_counter()
    result = solve_emission(
        gas_tau,
        geometry=gauss_legendre_disk_geometry(n_mu=8),
        bottom_boundary="blackbody",
        additional_optical_depths=cia,
    )
    runtime = perf_counter() - started
    species_tau = np.sum(
        gas_tau.species_tau * gas_tau.g_weights[None, None, None, :], axis=-1
    )
    molecular_tau = gas_tau.g_weighted_layer_tau()
    cia_tau = np.sum(np.stack([item.tau for item in cia]), axis=0)
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": np.pi * np.asarray(result.radiance.values),
        "molecular_tau_by_species": species_tau,
        "molecular_tau": molecular_tau,
        "cia_tau": cia_tau,
        "total_mean_tau": molecular_tau + cia_tau,
        "runtime_s": np.array(runtime),
    }


def _stage_3(args: argparse.Namespace) -> dict[str, Any]:
    stage_dir = args.output_root / "stage_3"
    stage_dir.mkdir(parents=True, exist_ok=True)
    spectra: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        "robert": {},
        "picaso": {},
        "petitradtrans": {},
    }
    per_resolution: dict[str, Any] = {}
    for n_layers in RESOLUTIONS:
        contract = stage_3_contract(n_layers)
        contract_path = stage_dir / f"contract_L{n_layers}.npz"
        _save_contract(contract_path, contract)
        external = {}
        for model, python in (
            ("picaso", args.picaso_python),
            ("petitradtrans", args.prt_python),
        ):
            output = stage_dir / f"{model}_L{n_layers}.npz"
            _run_worker(
                python,
                model,
                "native",
                contract_path,
                output,
                picaso_reference=args.picaso_reference,
                picaso_database=args.picaso_database,
                prt_input=args.prt_input,
                picaso_resample=args.picaso_resample,
            )
            external[model] = _load(output)
            spectra[model][n_layers] = (
                external[model]["wavelength_micron"],
                external[model]["flux_w_m2_m"][0],
            )
        robert = _native_robert(contract, args.prt_input)
        spectra["robert"][n_layers] = (
            robert["wavelength_micron"],
            robert["flux_w_m2_m"],
        )
        np.savez_compressed(
            stage_dir / f"robert_L{n_layers}.npz",
            case_id=contract["case_id"],
            **robert,
            metadata_json=np.array(
                json.dumps(
                    {
                        "model": "robert",
                        "version": importlib.metadata.version("robert-exoplanets"),
                        "python": os.sys.executable,
                        "native_include_cia": True,
                    },
                    sort_keys=True,
                )
            ),
        )
        edges = r100_edges()
        centers = np.sqrt(edges[:-1] * edges[1:])
        binned = {
            model: bin_mean(wavelength, flux, edges)
            for model, (wavelength, flux) in (
                (name, spectra[name][n_layers]) for name in spectra
            )
        }
        per_resolution[str(n_layers)] = {
            "pairwise_r100": pairwise_metrics(binned, centers),
            "native_wavelength_count": {
                model: int(spectra[model][n_layers][0].size) for model in spectra
            },
            "external_metadata": {
                name: json.loads(str(payload["metadata_json"]))
                for name, payload in external.items()
            },
            "robert_runtime_s": float(robert["runtime_s"]),
        }
        del robert, external
        gc.collect()
    convergence: dict[str, Any] = {}
    edges = r100_edges()
    centers = np.sqrt(edges[:-1] * edges[1:])
    for model, by_resolution in spectra.items():
        convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            coarse_binned = bin_mean(*by_resolution[coarse], edges)
            fine_binned = bin_mean(*by_resolution[fine], edges)
            convergence[model][f"{coarse}_to_{fine}"] = difference_metrics(
                coarse_binned, fine_binned, centers
            )
    return {
        "schema_version": 1,
        "stage": 3,
        "track": "B_native_opacity",
        "status": "characterized_no_cross_code_gate",
        "orchestrator": _robert_metadata(),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": 80,
        "molecular_species": list(SPECIES),
        "cia_pairs": ["H2-H2", "H2-He"],
        "per_resolution": per_resolution,
        "self_convergence": convergence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--picaso-python", type=Path, default=DEFAULT_PICASO_PYTHON)
    parser.add_argument("--prt-python", type=Path, default=DEFAULT_PRT_PYTHON)
    parser.add_argument(
        "--picaso-reference", type=Path, default=DEFAULT_PICASO_REFERENCE
    )
    parser.add_argument(
        "--picaso-database", type=Path, default=DEFAULT_PICASO_DATABASE
    )
    parser.add_argument("--prt-input", type=Path, default=DEFAULT_PRT_INPUT)
    parser.add_argument("--picaso-resample", type=int, default=50)
    args = parser.parse_args()
    invalid = set(args.stages) - {1, 2, 3}
    if invalid:
        parser.error(f"unsupported stages: {sorted(invalid)}")
    args.output_root = args.output_root.resolve()
    args.report_root = args.report_root.resolve()
    args.report_root.mkdir(parents=True, exist_ok=True)

    reports: dict[int, dict[str, Any]] = {}
    for stage in args.stages:
        started = perf_counter()
        if stage == 1:
            report = _shared_stage(1, (16, 32, 64, 128), args)
        elif stage == 2:
            report = _shared_stage(2, (20, 40, 80), args)
        else:
            report = _stage_3(args)
        report["wall_time_s"] = perf_counter() - started
        path = args.report_root / f"stage_{stage}_report.json"
        write_json(path, report)
        reports[stage] = report
        print(json.dumps({"stage": stage, "status": report["status"]}, indent=2))
    write_checksums(args.report_root)
    failed = [stage for stage, report in reports.items() if report["status"] == "fail"]
    if failed:
        raise RuntimeError(f"Track-A acceptance gates failed for stages {failed}")


if __name__ == "__main__":
    main()
