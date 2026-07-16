"""Run Stage 5 of the isolated ROBERT/PICASO/pRT emission comparison.

Stage 5 measures localized thermal response functions and temperature
Jacobians.  Track A freezes a shared optical-depth field to isolate the source
function and radiative-transfer response.  Track B asks every framework to
recompute its native temperature-dependent opacity for every perturbation.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

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
from emission_intercomparison_common import (
    SPECIES,
    STAGE_4_PROFILE_NAMES,
    STAGE_5_LOCALIZATION_SIGMA_DEX,
    STAGE_5_PERTURBATION_AMPLITUDE_K,
    STAGE_5_PERTURBATION_CENTERS_BAR,
    bin_mean,
    contribution_metrics,
    eclipse_jacobian_ppm_per_k,
    normalize_contribution,
    normalize_temperature_response,
    pairwise_contribution_metrics,
    pairwise_temperature_jacobian_metrics,
    r100_edges,
    sha256,
    stage_5_contract,
    temperature_jacobian_metrics,
    temperature_localization,
    write_checksums,
    write_json,
)


RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
MODELS = ("robert", "picaso", "petitradtrans")
TRACKS = ("track_a_shared_tau", "track_b_native_opacity")
TRACK_A_GATES = {
    "primary_p95_abs_difference_over_pair_peak": 0.05,
    "primary_rms_eclipse_jacobian_difference_ppm_per_k": 0.02,
    "primary_centroid_pressure_rms_difference_dex": 0.15,
    "primary_profile_total_variation_p95": 0.08,
    "80_to_160_p95_abs_difference_over_pair_peak": 0.05,
    "80_to_160_rms_eclipse_jacobian_difference_ppm_per_k": 0.02,
    "80_to_160_centroid_pressure_rms_difference_dex": 0.15,
    "80_to_160_profile_total_variation_p95": 0.08,
}


def _opacity_paths(input_data: Path) -> dict[str, Path]:
    patterns = {
        "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
        "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
        "CO2": "*UCL-4000*.ktable.petitRADTRANS.h5",
        "CH4": "*YT34to10*.ktable.petitRADTRANS.h5",
        "H2-H2": "*H2--H2*.ciatable.petitRADTRANS.h5",
        "H2-He": "*H2--He*.ciatable.petitRADTRANS.h5",
    }
    return {name: next(input_data.rglob(pattern)) for name, pattern in patterns.items()}


def _native_robert_stage_5(
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

    paths = _opacity_paths(input_data)
    pressure_edges = contract["pressure_edges_bar"]
    pressure_centers = contract["pressure_centers_bar"]
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_centers,
        unit="bar",
        name="emission_intercomparison_stage_5_cells",
    )
    # Load the four tables once.  Re-loading 1.2 GB of source HDF data for every
    # finite-difference case would dominate the benchmark without changing the
    # opacity calculation being tested.
    tables = {
        species: CorrelatedKTable.from_petitradtrans_hdf(
            paths[species], species=species
        )
        for species in SPECIES
    }
    first_table = tables["H2O"]
    mask = (first_table.wavelength_micron >= 0.5) & (
        first_table.wavelength_micron <= 12.0
    )
    wavelength = np.sort(first_table.wavelength_micron[mask])
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="pRT-R1000"
    )
    providers = {
        species: CorrelatedKOpacityProvider(
            {species: tables[species]},
            name=f"emission-intercomparison-stage5-{species}",
            interpolation="log_pressure_temperature_log_k",
        )
        for species in SPECIES
    }
    prepared_by_species = {
        species: providers[species].prepare(
            spectral_grid, pressure_grid, species=(species,)
        )
        for species in SPECIES
    }
    g_samples = first_table.g_samples
    g_weights = first_table.g_weights
    prepared = PreparedCorrelatedKOpacity(
        provider_name="pRT-HDF-four-species",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=SPECIES,
        g_samples=g_samples,
        g_weights=g_weights,
        cache_key=f"stage5-{pressure_centers.size}",
        metadata={"interpolation": "log_pressure_temperature_log_k"},
    )
    cia_tables = (
        CiaTable.from_petitradtrans_hdf(paths["H2-H2"], collision_pair="H2-H2"),
        CiaTable.from_petitradtrans_hdf(paths["H2-He"], collision_pair="H2-He"),
    )
    output_flux = []
    output_contribution = []
    contribution_case_index = []
    output_runtime = []
    shared_total_tau = np.empty(
        (len(STAGE_4_PROFILE_NAMES), pressure_centers.size, wavelength.size)
    )
    contribution_mask = np.asarray(
        contract["native_contribution_case_mask"], dtype=bool
    )
    for case_index, vmr in enumerate(contract["gas_vmr"]):
        started = perf_counter()
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
            mean_molecular_weight=np.full(pressure_centers.size, mean_molecular_weight),
        )
        evaluated = np.empty(
            (
                len(SPECIES),
                pressure_centers.size,
                wavelength.size,
                g_weights.size,
            )
        )
        for species_index, species in enumerate(SPECIES):
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
            gravity_m_s2=15.0,
            gas_combination="random_overlap",
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
        result = solve_emission(
            gas_tau,
            geometry=gauss_legendre_disk_geometry(n_mu=8),
            bottom_boundary="blackbody",
            additional_optical_depths=cia,
        )
        output_runtime.append(perf_counter() - started)
        output_flux.append(np.pi * np.asarray(result.radiance.values))
        if contribution_mask[case_index]:
            contribution = np.array(result.layer_contribution_radiance, copy=True)
            contribution[-1] += np.asarray(result.bottom_contribution_radiance)
            output_contribution.append(normalize_contribution(contribution))
            contribution_case_index.append(case_index)
            profile_index = int(contract["profile_index"][case_index])
            shared_total_tau[profile_index] = np.sum(
                np.asarray(result.total_optical_depth) * g_weights[None, None, :],
                axis=-1,
            )
        del atmosphere, evaluated, opacity, gas_tau, cia, result
        gc.collect()
    return {
        "wavelength_micron": wavelength,
        "pressure_bar": pressure_centers,
        "flux_w_m2_m": np.asarray(output_flux),
        "normalized_contribution": np.asarray(output_contribution),
        "contribution_case_index": np.asarray(contribution_case_index, dtype=int),
        "shared_total_tau": shared_total_tau,
        "runtime_s": np.asarray(output_runtime),
    }


def _shared_robert_stage_5(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from robert_exoplanets import planck_radiance_wavelength
    from robert_exoplanets.rt import integrate_thermal_emission_spectrum

    wavelength = contract["wavelength_micron"]
    mu = contract["emission_mu"]
    point_weights = contract["disk_weights"]
    shared_tau = contract["shared_total_tau"]
    flux = np.empty((contract["case_id"].size, wavelength.size))
    runtime = np.empty(contract["case_id"].size)
    for case_index, profile_index in enumerate(contract["profile_index"]):
        tau = shared_tau[int(profile_index)]
        level_temperature = contract["temperature_edges_k"][case_index]
        layer_temperature = 0.5 * (level_temperature[:-1] + level_temperature[1:])
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
        started = perf_counter()
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
        runtime[case_index] = perf_counter() - started
        flux[case_index] = np.pi * result.radiance
    return {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "runtime_s": runtime,
    }


def _finite_difference_jacobians(
    binned_flux: np.ndarray, contract: dict[str, np.ndarray]
) -> np.ndarray:
    amplitude = float(contract["perturbation_amplitude_k"])
    output = np.empty(
        (
            len(STAGE_4_PROFILE_NAMES),
            len(STAGE_5_PERTURBATION_CENTERS_BAR),
            binned_flux.shape[1],
        )
    )
    for profile_index in range(len(STAGE_4_PROFILE_NAMES)):
        for center_index in range(len(STAGE_5_PERTURBATION_CENTERS_BAR)):
            common = (contract["profile_index"] == profile_index) & (
                contract["perturbation_center_index"] == center_index
            )
            minus = np.flatnonzero(common & (contract["perturbation_sign"] == -1))
            plus = np.flatnonzero(common & (contract["perturbation_sign"] == 1))
            if minus.size != 1 or plus.size != 1:
                raise ValueError("finite-difference contract is not one-to-one")
            output[profile_index, center_index] = (
                binned_flux[plus[0]] - binned_flux[minus[0]]
            ) / (2.0 * amplitude)
    return output


def _binned_baseline_contribution(
    payload: dict[str, np.ndarray], edges: np.ndarray
) -> np.ndarray:
    contributions = payload["normalized_contribution"]
    case_indices = payload["contribution_case_index"]
    output = np.empty(
        (len(STAGE_4_PROFILE_NAMES), contributions.shape[1], edges.size - 1)
    )
    for contribution, case_index in zip(contributions, case_indices, strict=True):
        profile_index = int(case_index)
        if profile_index >= len(STAGE_4_PROFILE_NAMES):
            raise ValueError("baseline contributions must be the first four cases")
        output[profile_index] = normalize_contribution(
            bin_mean(payload["wavelength_micron"], contribution, edges)
        )
    return output


def _response_summary(
    response: np.ndarray, pressure_bar: np.ndarray
) -> dict[str, float]:
    normalized = normalize_temperature_response(response)
    log_pressure = np.log10(pressure_bar)[:, None]
    centroid = 10.0 ** np.sum(normalized * log_pressure, axis=0)
    peak = pressure_bar[np.argmax(normalized, axis=0)]
    return {
        "centroid_pressure_median_bar": float(np.median(centroid)),
        "centroid_pressure_p05_bar": float(np.percentile(centroid, 5.0)),
        "centroid_pressure_p95_bar": float(np.percentile(centroid, 95.0)),
        "peak_pressure_median_bar": float(np.median(peak)),
        "peak_pressure_p05_bar": float(np.percentile(peak, 5.0)),
        "peak_pressure_p95_bar": float(np.percentile(peak, 95.0)),
    }


def _timing_summary(values: np.ndarray) -> dict[str, float | int]:
    timing = np.asarray(values, dtype=float)
    return {
        "case_count": int(timing.size),
        "total_s": float(np.sum(timing)),
        "median_s": float(np.median(timing)),
        "minimum_s": float(np.min(timing)),
        "maximum_s": float(np.max(timing)),
    }


def _metadata(model: str, mode: str) -> dict[str, str]:
    return {
        **_robert_metadata(),
        "model": model,
        "mode": mode,
    }


def _project_contribution_to_centers(
    contribution: np.ndarray, pressure_bar: np.ndarray
) -> np.ndarray:
    localization = np.stack(
        [
            temperature_localization(
                pressure_bar,
                center,
                sigma_dex=STAGE_5_LOCALIZATION_SIGMA_DEX,
            )
            for center in STAGE_5_PERTURBATION_CENTERS_BAR
        ]
    )
    return normalize_temperature_response(localization @ contribution)


def _write_robert_output(
    path: Path,
    contract: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    mode: str,
) -> None:
    output = {
        "case_id": contract["case_id"],
        **payload,
        "metadata_json": np.array(
            json.dumps(
                {
                    **_metadata("robert", mode),
                    "native_include_cia": (
                        bool(contract["native_include_cia"])
                        if "native_include_cia" in contract
                        else None
                    ),
                },
                sort_keys=True,
            )
        ),
    }
    np.savez_compressed(path, **output)


def _run_stage_5(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stage_dir = args.output_root / "stage_5"
    stage_dir.mkdir(parents=True, exist_ok=True)
    edges = r100_edges()
    wavelength_r100 = np.sqrt(edges[:-1] * edges[1:])
    outputs: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]] = {
        track: {resolution: {} for resolution in RESOLUTIONS} for track in TRACKS
    }
    jacobians: dict[str, dict[int, dict[str, np.ndarray]]] = {
        track: {resolution: {} for resolution in RESOLUTIONS} for track in TRACKS
    }
    responses: dict[str, dict[int, dict[str, np.ndarray]]] = {
        track: {resolution: {} for resolution in RESOLUTIONS} for track in TRACKS
    }
    baseline_contributions: dict[int, dict[str, np.ndarray]] = {
        resolution: {} for resolution in RESOLUTIONS
    }
    projected_contributions: dict[int, dict[str, np.ndarray]] = {
        resolution: {} for resolution in RESOLUTIONS
    }
    per_resolution: dict[str, Any] = {}
    artifact_checksums: dict[str, str] = {}
    for n_cells in RESOLUTIONS:
        contract = stage_5_contract(n_cells)
        native_contract_path = stage_dir / f"contract_track_b_L{n_cells}.npz"
        native_contract = {**contract, "track": np.array("track_b_native_opacity")}
        _save_contract(native_contract_path, native_contract)

        external_metadata: dict[str, dict[str, Any]] = {}
        for model, python in (
            ("picaso", args.picaso_python),
            ("petitradtrans", args.prt_python),
        ):
            output_path = stage_dir / f"track_b_{model}_L{n_cells}.npz"
            _run_worker(
                python,
                model,
                "native",
                native_contract_path,
                output_path,
                picaso_reference=args.picaso_reference,
                picaso_database=args.picaso_database,
                prt_input=args.prt_input,
                picaso_resample=args.picaso_resample,
            )
            outputs["track_b_native_opacity"][n_cells][model] = _load(output_path)
            external_metadata[f"track_b_{model}"] = json.loads(
                str(outputs["track_b_native_opacity"][n_cells][model]["metadata_json"])
            )
        robert_native = _native_robert_stage_5(contract, args.prt_input)
        outputs["track_b_native_opacity"][n_cells]["robert"] = robert_native
        robert_native_path = stage_dir / f"track_b_robert_L{n_cells}.npz"
        _write_robert_output(robert_native_path, contract, robert_native, "native")

        shared_contract = {
            **contract,
            "track": np.array("track_a_shared_tau"),
            "wavelength_micron": robert_native["wavelength_micron"],
            "shared_total_tau": robert_native["shared_total_tau"],
            "native_return_contribution": np.array(False),
        }
        shared_contract_path = stage_dir / f"contract_track_a_L{n_cells}.npz"
        _save_contract(shared_contract_path, shared_contract)
        for model, python in (
            ("picaso", args.picaso_python),
            ("petitradtrans", args.prt_python),
        ):
            output_path = stage_dir / f"track_a_{model}_L{n_cells}.npz"
            _run_worker(
                python,
                model,
                "shared",
                shared_contract_path,
                output_path,
                picaso_reference=args.picaso_reference,
                picaso_database=args.picaso_database,
                prt_input=args.prt_input,
                picaso_resample=args.picaso_resample,
            )
            outputs["track_a_shared_tau"][n_cells][model] = _load(output_path)
            external_metadata[f"track_a_{model}"] = json.loads(
                str(outputs["track_a_shared_tau"][n_cells][model]["metadata_json"])
            )
        robert_shared = _shared_robert_stage_5(shared_contract)
        outputs["track_a_shared_tau"][n_cells]["robert"] = robert_shared
        robert_shared_path = stage_dir / f"track_a_robert_L{n_cells}.npz"
        _write_robert_output(
            robert_shared_path, shared_contract, robert_shared, "shared"
        )

        for track in TRACKS:
            for model in MODELS:
                payload = outputs[track][n_cells][model]
                binned_flux = bin_mean(
                    payload["wavelength_micron"], payload["flux_w_m2_m"], edges
                )
                jacobian = _finite_difference_jacobians(binned_flux, contract)
                jacobians[track][n_cells][model] = jacobian
                responses[track][n_cells][model] = np.stack(
                    [normalize_temperature_response(values) for values in jacobian]
                )
        for model in MODELS:
            payload = outputs["track_b_native_opacity"][n_cells][model]
            baseline_contributions[n_cells][model] = _binned_baseline_contribution(
                payload, edges
            )
            projected_contributions[n_cells][model] = np.stack(
                [
                    _project_contribution_to_centers(
                        contribution, contract["pressure_centers_bar"]
                    )
                    for contribution in baseline_contributions[n_cells][model]
                ]
            )

        track_reports: dict[str, Any] = {}
        for track in TRACKS:
            profile_reports: dict[str, Any] = {}
            for profile_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
                profile_jacobians = {
                    model: jacobians[track][n_cells][model][profile_index]
                    for model in MODELS
                }
                profile_responses = {
                    model: responses[track][n_cells][model][profile_index]
                    for model in MODELS
                }
                profile_reports[profile_name] = {
                    "pairwise_spectral_temperature_jacobian_r100": (
                        pairwise_temperature_jacobian_metrics(
                            profile_jacobians, wavelength_r100
                        )
                    ),
                    "pairwise_vertical_response_r100": (
                        pairwise_contribution_metrics(
                            profile_responses,
                            np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
                        )
                    ),
                    "framework_response_summary": {
                        model: {
                            **_response_summary(
                                profile_responses[model],
                                np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
                            ),
                            "eclipse_jacobian_rms_ppm_per_k": float(
                                np.sqrt(
                                    np.mean(
                                        eclipse_jacobian_ppm_per_k(
                                            profile_jacobians[model], wavelength_r100
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            "eclipse_jacobian_max_abs_ppm_per_k": float(
                                np.max(
                                    np.abs(
                                        eclipse_jacobian_ppm_per_k(
                                            profile_jacobians[model], wavelength_r100
                                        )
                                    )
                                )
                            ),
                        }
                        for model in MODELS
                    },
                    "stage_4_contribution_relation": {
                        model: contribution_metrics(
                            profile_responses[model],
                            projected_contributions[n_cells][model][profile_index],
                            np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
                        )
                        for model in MODELS
                    },
                }
            track_reports[track] = {
                "profiles": profile_reports,
                "timings": {
                    model: _timing_summary(outputs[track][n_cells][model]["runtime_s"])
                    for model in MODELS
                },
            }
        paths = (
            native_contract_path,
            shared_contract_path,
            robert_native_path,
            robert_shared_path,
            stage_dir / f"track_a_picaso_L{n_cells}.npz",
            stage_dir / f"track_a_petitradtrans_L{n_cells}.npz",
            stage_dir / f"track_b_picaso_L{n_cells}.npz",
            stage_dir / f"track_b_petitradtrans_L{n_cells}.npz",
        )
        for path in paths:
            artifact_checksums[str(path.relative_to(args.output_root))] = sha256(path)
        per_resolution[str(n_cells)] = {
            "tracks": track_reports,
            "external_metadata": external_metadata,
            "robert_metadata": {
                "track_a": _metadata("robert", "shared"),
                "track_b": _metadata("robert", "native"),
            },
            "native_wavelength_count": {
                model: int(
                    outputs["track_b_native_opacity"][n_cells][model][
                        "wavelength_micron"
                    ].size
                )
                for model in MODELS
            },
        }
        del robert_native, robert_shared
        gc.collect()

    self_convergence: dict[str, Any] = {}
    centers = np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR)
    for track in TRACKS:
        self_convergence[track] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            pair = f"{coarse}_to_{fine}"
            self_convergence[track][pair] = {}
            for model in MODELS:
                self_convergence[track][pair][model] = {}
                for profile_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
                    coarse_jacobian = jacobians[track][coarse][model][profile_index]
                    fine_jacobian = jacobians[track][fine][model][profile_index]
                    self_convergence[track][pair][model][profile_name] = {
                        "spectral_temperature_jacobian_r100": (
                            temperature_jacobian_metrics(
                                coarse_jacobian, fine_jacobian, wavelength_r100
                            )
                        ),
                        "vertical_response_r100": contribution_metrics(
                            responses[track][coarse][model][profile_index],
                            responses[track][fine][model][profile_index],
                            centers,
                        ),
                    }

    primary_profiles = per_resolution[str(PRIMARY_RESOLUTION)]["tracks"][
        "track_a_shared_tau"
    ]["profiles"]
    primary_jacobian_metrics = [
        metrics
        for profile in primary_profiles.values()
        for metrics in profile["pairwise_spectral_temperature_jacobian_r100"].values()
    ]
    primary_response_metrics = [
        metrics
        for profile in primary_profiles.values()
        for metrics in profile["pairwise_vertical_response_r100"].values()
    ]
    convergence_metrics = [
        metrics
        for model in self_convergence["track_a_shared_tau"]["80_to_160"].values()
        for metrics in model.values()
    ]
    observed = {
        "primary_p95_abs_difference_over_pair_peak": max(
            value["p95_abs_difference_over_pair_peak"]
            for value in primary_jacobian_metrics
        ),
        "primary_rms_eclipse_jacobian_difference_ppm_per_k": max(
            value["rms_eclipse_jacobian_difference_ppm_per_k"]
            for value in primary_jacobian_metrics
        ),
        "primary_centroid_pressure_rms_difference_dex": max(
            value["centroid_pressure_rms_difference_dex"]
            for value in primary_response_metrics
        ),
        "primary_profile_total_variation_p95": max(
            value["profile_total_variation_p95"] for value in primary_response_metrics
        ),
        "80_to_160_p95_abs_difference_over_pair_peak": max(
            value["spectral_temperature_jacobian_r100"][
                "p95_abs_difference_over_pair_peak"
            ]
            for value in convergence_metrics
        ),
        "80_to_160_rms_eclipse_jacobian_difference_ppm_per_k": max(
            value["spectral_temperature_jacobian_r100"][
                "rms_eclipse_jacobian_difference_ppm_per_k"
            ]
            for value in convergence_metrics
        ),
        "80_to_160_centroid_pressure_rms_difference_dex": max(
            value["vertical_response_r100"]["centroid_pressure_rms_difference_dex"]
            for value in convergence_metrics
        ),
        "80_to_160_profile_total_variation_p95": max(
            value["vertical_response_r100"]["profile_total_variation_p95"]
            for value in convergence_metrics
        ),
    }
    gate_results = {
        name: {
            "threshold": threshold,
            "observed": observed[name],
            "passed": bool(observed[name] <= threshold),
        }
        for name, threshold in TRACK_A_GATES.items()
    }
    acceptance_passed = all(result["passed"] for result in gate_results.values())

    response_artifact: dict[str, np.ndarray] = {
        "schema_version": np.array(1),
        "wavelength_r100_micron": wavelength_r100,
        "profile_name": np.asarray(STAGE_4_PROFILE_NAMES),
        "perturbation_centers_bar": centers,
        "perturbation_amplitude_k": np.array(STAGE_5_PERTURBATION_AMPLITUDE_K),
        "localization_sigma_dex": np.array(STAGE_5_LOCALIZATION_SIGMA_DEX),
    }
    for track in TRACKS:
        for resolution in RESOLUTIONS:
            for model in MODELS:
                prefix = f"{track}_L{resolution}_{model}"
                response_artifact[f"flux_jacobian_{prefix}_w_m2_m_k"] = jacobians[
                    track
                ][resolution][model]
                response_artifact[f"eclipse_jacobian_{prefix}_ppm_k"] = (
                    eclipse_jacobian_ppm_per_k(
                        jacobians[track][resolution][model], wavelength_r100
                    )
                )
                response_artifact[f"normalized_vertical_response_{prefix}"] = responses[
                    track
                ][resolution][model]
    for resolution in RESOLUTIONS:
        for model in MODELS:
            prefix = f"L{resolution}_{model}"
            response_artifact[f"stage4_contribution_{prefix}"] = baseline_contributions[
                resolution
            ][model]
            response_artifact[f"stage4_projected_contribution_{prefix}"] = (
                projected_contributions[resolution][model]
            )

    input_paths = _opacity_paths(args.prt_input)
    input_paths["PICASO-database"] = args.picaso_database
    input_paths["PICASO-reference-config"] = args.picaso_reference / "config.json"
    input_paths["PICASO-reference-version"] = args.picaso_reference / "version.md"
    source_paths = {
        "stage_5_launcher": Path(__file__),
        "shared_contracts_and_metrics": Path(__file__).with_name(
            "emission_intercomparison_common.py"
        ),
        "external_worker": Path(__file__).with_name(
            "run_emission_intercomparison_external.py"
        ),
    }
    report = {
        "schema_version": 1,
        "stage": 5,
        "status": "passed" if acceptance_passed else "failed_track_a_gates",
        "orchestrator": _robert_metadata(),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "profiles": list(STAGE_4_PROFILE_NAMES),
        "picaso_native_resample": int(args.picaso_resample),
        "environment_notes": {
            "picaso_reference_configuration": "3.2.1",
            "picaso_reference_warning": (
                "PICASO code 3.2.2 reports reference configuration 3.2.1; "
                "the known minor-version warning is retained"
            ),
            "petitradtrans_home": (
                "private writable HOME beneath the ignored Stage-5 output directory"
            ),
        },
        "perturbation_centers_bar": list(STAGE_5_PERTURBATION_CENTERS_BAR),
        "vertical_grid_contract": {
            "robert": "40/80/160 pressure cells bounded by pressure_edges_bar",
            "picaso": "matching 41/81/161 pressure edges supplied as levels",
            "petitradtrans": (
                "ROBERT geometric cell centres supplied as 40/80/160 pressure nodes"
            ),
            "response_coordinate": "six fixed perturbation centres in bar",
        },
        "method_definitions": {
            "finite_difference": ("J(P0,lambda)=[F(T+A L)-F(T-A L)]/(2 A), A=10 K"),
            "localization": (
                "L(P;P0)=exp[-0.5*(log10(P/P0)/0.35 dex)^2], unit peak at P0"
            ),
            "vertical_response": (
                "absolute R=100 Jacobian normalized over all six perturbation centres "
                "independently at each wavelength"
            ),
            "track_a": (
                "ROBERT baseline native gas+CIA total optical depth, g-weighted per "
                "layer and wavelength, frozen under +/- perturbations and supplied "
                "identically to all three pure-absorption RT paths"
            ),
            "track_b": (
                "each framework recomputes its native temperature-dependent line and "
                "CIA opacity for every +/- perturbation"
            ),
            "stage_4_relation": (
                "Stage-4 baseline contribution functions are projected through the "
                "same localization functions and normalized across centres before "
                "comparison with numerical temperature responses"
            ),
            "interpretation_caveat": (
                "contribution functions decompose emergent intensity, whereas full "
                "temperature derivatives also include dB/dT and, in Track B, opacity "
                "derivatives; they are therefore not generally identical"
            ),
        },
        "tracks": {
            "track_a_shared_tau": {
                "status": "passed" if acceptance_passed else "failed",
                "cross_code_acceptance_gates": gate_results,
            },
            "track_b_native_opacity": {
                "status": "characterized_no_cross_code_gate",
                "picaso_attribution": (
                    "PICASO uses its independent official opacity database; its "
                    "cross-code differences are attribution results, not acceptance gates"
                ),
            },
        },
        "per_resolution": per_resolution,
        "self_convergence": self_convergence,
        "artifact_checksums": artifact_checksums,
        "benchmark_source_checksums": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "input_data_checksums": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
    }
    return report, response_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--picaso-python", type=Path, default=DEFAULT_PICASO_PYTHON)
    parser.add_argument("--prt-python", type=Path, default=DEFAULT_PRT_PYTHON)
    parser.add_argument(
        "--picaso-reference", type=Path, default=DEFAULT_PICASO_REFERENCE
    )
    parser.add_argument("--picaso-database", type=Path, default=DEFAULT_PICASO_DATABASE)
    parser.add_argument("--prt-input", type=Path, default=DEFAULT_PRT_INPUT)
    parser.add_argument("--picaso-resample", type=int, default=50)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.report_root = args.report_root.resolve()
    args.report_root.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    report, response_artifact = _run_stage_5(args)
    response_path = args.report_root / "stage_5_response_profiles.npz"
    np.savez_compressed(response_path, **response_artifact)
    report["response_artifact"] = {
        "path": response_path.name,
        "sha256": sha256(response_path),
        "contents": (
            "complete R=100 flux and eclipse Jacobians, normalized vertical "
            "responses, and Stage-4 contribution projections"
        ),
    }
    report["wall_time_s"] = perf_counter() - started
    write_json(args.report_root / "stage_5_report.json", report)
    write_checksums(args.report_root)
    print(json.dumps({"stage": 5, "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
