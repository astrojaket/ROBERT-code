"""Run Stage 4 of the isolated ROBERT/PICASO/pRT emission comparison.

Stage 4 compares native-opacity spectra and pressure-resolved thermal
contribution functions for four thermal structures.  ROBERT orchestration runs
in ``robert-exoplanets``; PICASO and stable petitRADTRANS run only through the
external worker in their dedicated environments.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from emission_intercomparison_common import (
    SPECIES,
    STAGE_4_PROFILE_NAMES,
    bin_mean,
    contribution_metrics,
    difference_metrics,
    normalize_contribution,
    pairwise_contribution_metrics,
    pairwise_metrics,
    r100_edges,
    stage_4_contract,
    write_checksums,
    write_json,
)
from benchmark_emission_intercomparison_stages_1_3 import (
    DEFAULT_OUTPUT,
    DEFAULT_PICASO_DATABASE,
    DEFAULT_PICASO_PYTHON,
    DEFAULT_PICASO_REFERENCE,
    DEFAULT_PRT_INPUT,
    DEFAULT_PRT_PYTHON,
    DEFAULT_REPORTS,
    _load,
    _robert_metadata,
    _run_worker,
    _save_contract,
)


RESOLUTIONS = (40, 80, 160)


def _native_robert_stage_4(
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
    pressure_centers = contract["pressure_centers_bar"]
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_centers,
        unit="bar",
        name="emission_intercomparison_stage_4_cells",
    )
    first_path = next(input_data.rglob(patterns["H2O"]))
    first_table = CorrelatedKTable.from_petitradtrans_hdf(
        first_path, species="H2O"
    )
    mask = (first_table.wavelength_micron >= 0.5) & (
        first_table.wavelength_micron <= 12.0
    )
    wavelength = np.sort(first_table.wavelength_micron[mask])
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="pRT-R1000"
    )
    g_samples = first_table.g_samples
    g_weights = first_table.g_weights
    output_flux = []
    output_contribution = []
    output_runtime = []
    for case_index, vmr in enumerate(contract["gas_vmr"]):
        composition = dict(zip(("H2", "He", *SPECIES), vmr, strict=True))
        mean_molecular_weight = sum(
            composition[name] * mass
            for name, mass in zip(
                ("H2", "He", *SPECIES),
                (2.01588, 4.002602, 18.01528, 28.0101, 44.0095, 16.04246),
                strict=True,
            )
        )
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=contract["temperature_cells_k"][case_index],
            temperature_edges=contract["temperature_edges_k"][case_index],
            composition={
                name: np.full(pressure_centers.size, value)
                for name, value in composition.items()
            },
            mean_molecular_weight=np.full(
                pressure_centers.size, mean_molecular_weight
            ),
        )
        evaluated = np.empty(
            (
                len(SPECIES),
                pressure_centers.size,
                wavelength.size,
                g_weights.size,
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
                name=f"emission-intercomparison-stage4-{species}",
                interpolation="log_pressure_temperature_log_k",
            )
            prepared_one = provider.prepare(
                spectral_grid, pressure_grid, species=(species,)
            )
            evaluated[species_index] = provider.evaluate(
                atmosphere, prepared_one
            ).kcoeff[0]
            if species != "H2O":
                del table
            del provider, prepared_one
            gc.collect()
        prepared = PreparedCorrelatedKOpacity(
            provider_name="pRT-HDF-four-species",
            spectral_grid=spectral_grid,
            pressure_grid=pressure_grid,
            species=SPECIES,
            g_samples=g_samples,
            g_weights=g_weights,
            cache_key=f"stage4-{pressure_centers.size}-{case_index}",
            metadata={"interpolation": "log_pressure_temperature_log_k"},
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
            gravity_m_s2=15.0,
            gas_combination="random_overlap",
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
        output_runtime.append(perf_counter() - started)
        output_flux.append(np.pi * np.asarray(result.radiance.values))
        contribution = np.array(result.layer_contribution_radiance, copy=True)
        contribution[-1] += np.asarray(result.bottom_contribution_radiance)
        output_contribution.append(normalize_contribution(contribution))
        del evaluated, prepared, opacity, gas_tau, cia_tables, cia, result
        gc.collect()
    return {
        "wavelength_micron": wavelength,
        "pressure_bar": pressure_centers,
        "flux_w_m2_m": np.asarray(output_flux),
        "normalized_contribution": np.asarray(output_contribution),
        "runtime_s": np.asarray(output_runtime),
    }


def _binned_model_output(
    payload: dict[str, np.ndarray], edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    binned_flux = bin_mean(
        payload["wavelength_micron"], payload["flux_w_m2_m"], edges
    )
    binned_contribution = np.stack(
        [
            normalize_contribution(
                bin_mean(
                    payload["wavelength_micron"], contribution, edges
                )
            )
            for contribution in payload["normalized_contribution"]
        ]
    )
    return binned_flux, binned_contribution


def _coarsen_vertical(
    contribution: np.ndarray, target_cells: int
) -> np.ndarray:
    if contribution.shape[0] % target_cells != 0:
        raise ValueError("fine contribution grid is not divisible by target grid")
    ratio = contribution.shape[0] // target_cells
    return contribution.reshape(target_cells, ratio, contribution.shape[1]).sum(
        axis=1
    )


def _stage_4(args: argparse.Namespace) -> dict[str, Any]:
    stage_dir = args.output_root / "stage_4"
    stage_dir.mkdir(parents=True, exist_ok=True)
    edges = r100_edges()
    wavelength_r100 = np.sqrt(edges[:-1] * edges[1:])
    outputs: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "robert": {},
        "picaso": {},
        "petitradtrans": {},
    }
    binned: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        "robert": {},
        "picaso": {},
        "petitradtrans": {},
    }
    per_resolution: dict[str, Any] = {}
    for n_cells in RESOLUTIONS:
        contract = stage_4_contract(n_cells)
        contract_path = stage_dir / f"contract_L{n_cells}.npz"
        _save_contract(contract_path, contract)
        external = {}
        for model, python in (
            ("picaso", args.picaso_python),
            ("petitradtrans", args.prt_python),
        ):
            output_path = stage_dir / f"{model}_L{n_cells}.npz"
            _run_worker(
                python,
                model,
                "native",
                contract_path,
                output_path,
                picaso_reference=args.picaso_reference,
                picaso_database=args.picaso_database,
                prt_input=args.prt_input,
                picaso_resample=args.picaso_resample,
            )
            external[model] = _load(output_path)
            outputs[model][n_cells] = external[model]
        robert = _native_robert_stage_4(contract, args.prt_input)
        outputs["robert"][n_cells] = robert
        np.savez_compressed(
            stage_dir / f"robert_L{n_cells}.npz",
            case_id=contract["case_id"],
            **robert,
            metadata_json=np.array(
                json.dumps(
                    {
                        "model": "robert",
                        "version": importlib.metadata.version(
                            "robert-exoplanets"
                        ),
                        "python": os.sys.executable,
                        "native_include_cia": True,
                        "contribution_definition": (
                            "disk-integrated layer source contribution with "
                            "blackbody bottom folded into deepest cell"
                        ),
                    },
                    sort_keys=True,
                )
            ),
        )
        for model in outputs:
            binned[model][n_cells] = _binned_model_output(
                outputs[model][n_cells], edges
            )
        profile_reports: dict[str, Any] = {}
        for case_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
            spectra = {
                model: binned[model][n_cells][0][case_index]
                for model in outputs
            }
            contributions = {
                model: binned[model][n_cells][1][case_index]
                for model in outputs
            }
            profile_reports[profile_name] = {
                "pairwise_spectrum_r100": pairwise_metrics(
                    spectra, wavelength_r100
                ),
                "pairwise_contribution_r100": pairwise_contribution_metrics(
                    contributions, contract["pressure_centers_bar"]
                ),
            }
        per_resolution[str(n_cells)] = {
            "profiles": profile_reports,
            "native_wavelength_count": {
                model: int(outputs[model][n_cells]["wavelength_micron"].size)
                for model in outputs
            },
            "runtime_s": {
                model: [
                    float(value)
                    for value in outputs[model][n_cells]["runtime_s"]
                ]
                for model in outputs
            },
            "external_metadata": {
                name: json.loads(str(payload["metadata_json"]))
                for name, payload in external.items()
            },
        }
        np.savez_compressed(
            stage_dir / f"combined_r100_L{n_cells}.npz",
            case_id=contract["case_id"],
            profile_name=contract["profile_name"],
            wavelength_micron=wavelength_r100,
            pressure_bar=contract["pressure_centers_bar"],
            temperature_edges_k=contract["temperature_edges_k"],
            temperature_cells_k=contract["temperature_cells_k"],
            **{
                f"flux_{model}_w_m2_m": binned[model][n_cells][0]
                for model in outputs
            },
            **{
                f"contribution_{model}": binned[model][n_cells][1]
                for model in outputs
            },
        )
        del robert, external
        gc.collect()

    self_convergence: dict[str, Any] = {}
    for model in outputs:
        self_convergence[model] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            pair_name = f"{coarse}_to_{fine}"
            profile_convergence: dict[str, Any] = {}
            coarse_contract = stage_4_contract(coarse)
            for case_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
                coarse_flux = binned[model][coarse][0][case_index]
                fine_flux = binned[model][fine][0][case_index]
                coarse_contribution = binned[model][coarse][1][case_index]
                fine_contribution = _coarsen_vertical(
                    binned[model][fine][1][case_index], coarse
                )
                profile_convergence[profile_name] = {
                    "spectrum_r100": difference_metrics(
                        coarse_flux, fine_flux, wavelength_r100
                    ),
                    "contribution_r100": contribution_metrics(
                        coarse_contribution,
                        fine_contribution,
                        coarse_contract["pressure_centers_bar"],
                    ),
                }
            self_convergence[model][pair_name] = profile_convergence
    return {
        "schema_version": 1,
        "stage": 4,
        "track": "B_native_opacity_thermal_structure",
        "status": "characterized_no_cross_code_gate",
        "orchestrator": _robert_metadata(),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": 80,
        "profiles": list(STAGE_4_PROFILE_NAMES),
        "molecular_species": list(SPECIES),
        "cia_pairs": ["H2-H2", "H2-He"],
        "vertical_grid_contract": {
            "robert": "pressure cells bounded by pressure_edges_bar",
            "picaso": "pressure_edges_bar supplied as atmospheric levels",
            "petitradtrans": (
                "ROBERT geometric cell centres supplied as pressure nodes"
            ),
            "contribution_coordinate": "ROBERT geometric cell centres",
        },
        "contribution_definitions": {
            "robert": (
                "native disk-integrated source decomposition; bottom boundary "
                "folded into deepest cell"
            ),
            "petitradtrans": "native normalized emission_contribution",
            "picaso": (
                "independent pure-absorption formal solution applied to PICASO "
                "native total optical depth; Stage-1 RT limit"
            ),
        },
        "per_resolution": per_resolution,
        "self_convergence": self_convergence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args.output_root = args.output_root.resolve()
    args.report_root = args.report_root.resolve()
    args.report_root.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    report = _stage_4(args)
    report["wall_time_s"] = perf_counter() - started
    write_json(args.report_root / "stage_4_report.json", report)
    write_checksums(args.report_root)
    print(json.dumps({"stage": 4, "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
