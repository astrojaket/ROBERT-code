"""Run Stage 6 composition-Jacobian emission intercomparison.

Track A supplies identical case-specific optical depths, derived from the
shared source HDF opacity basis, to all three radiative-transfer paths.
Track B independently recomputes abundance-dependent opacity, correlated-k
mixing, mean molecular weight, and CIA in each framework.
"""

from __future__ import annotations

import argparse
import gc
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from benchmark_emission_intercomparison_stage_5 import (
    _metadata,
    _opacity_paths,
    _response_summary,
    _timing_summary,
    _write_robert_output,
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
from emission_intercomparison_common import (
    SPECIES,
    STAGE_4_PROFILE_NAMES,
    STAGE_5_LOCALIZATION_SIGMA_DEX,
    STAGE_5_PERTURBATION_CENTERS_BAR,
    STAGE_6_LINEARITY_AMPLITUDES_DEX,
    STAGE_6_PERTURBATION_AMPLITUDE_DEX,
    apply_jacobian_roundoff_floor,
    amplitude_linearity_metrics,
    bin_mean,
    contribution_metrics,
    cross_species_fraction_metrics,
    cross_species_sensitivity_fractions,
    eclipse_jacobian_ppm_per_dex,
    normalize_composition_response,
    normalize_contribution,
    r100_edges,
    sha256,
    signed_jacobian_metrics,
    stage_6_contract,
    write_checksums,
    write_json,
)


RESOLUTIONS = (40, 80, 160)
PRIMARY_RESOLUTION = 80
MODELS = ("robert", "picaso", "petitradtrans")
TRACKS = ("track_a_shared_tau", "track_b_native_opacity")
AUDIT_PROFILES = ("monotonic", "retrieved_like")
STAGE_5_RESPONSE_PATH = DEFAULT_REPORTS / "stage_5_response_profiles.npz"

# These gates are duplicated verbatim in review/49 and were fixed before the
# full Stage-6 matrix was run.  Track B is attribution/characterization only.
TRACK_A_GATES = {
    "primary_p95_abs_difference_over_pair_peak": 0.05,
    "primary_rms_eclipse_jacobian_difference_ppm_per_dex": 0.50,
    "primary_centroid_pressure_rms_difference_dex": 0.15,
    "primary_profile_total_variation_p95": 0.08,
    "primary_cross_species_total_variation_p95": 0.08,
    "80_to_160_p95_abs_difference_over_pair_peak": 0.05,
    "80_to_160_rms_eclipse_jacobian_difference_ppm_per_dex": 0.50,
    "80_to_160_centroid_pressure_rms_difference_dex": 0.15,
    "80_to_160_profile_total_variation_p95": 0.08,
    "80_to_160_cross_species_total_variation_p95": 0.08,
}


def _native_robert_stage_6(
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
        name="emission_intercomparison_stage_6_cells",
    )
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
            name=f"emission-intercomparison-stage6-{species}",
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
    g_weights = first_table.g_weights
    prepared = PreparedCorrelatedKOpacity(
        provider_name="pRT-HDF-four-species",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=SPECIES,
        g_samples=first_table.g_samples,
        g_weights=g_weights,
        cache_key=f"stage6-{pressure_centers.size}",
        metadata={"interpolation": "log_pressure_temperature_log_k"},
    )
    cia_tables = (
        CiaTable.from_petitradtrans_hdf(paths["H2-H2"], collision_pair="H2-H2"),
        CiaTable.from_petitradtrans_hdf(paths["H2-He"], collision_pair="H2-He"),
    )

    case_count = contract["case_id"].size
    flux = np.empty((case_count, wavelength.size))
    runtime = np.empty(case_count)
    shared_total_tau = np.empty((case_count, pressure_centers.size, wavelength.size))
    output_contribution: list[np.ndarray] = []
    contribution_case_index: list[int] = []
    contribution_mask = np.asarray(
        contract["native_contribution_case_mask"], dtype=bool
    )
    names = ("H2", "He", *SPECIES)
    for case_index, vmr in enumerate(contract["gas_vmr_cells"]):
        started = perf_counter()
        composition = {
            name: np.asarray(vmr[:, index], dtype=float)
            for index, name in enumerate(names)
        }
        atmosphere = AtmosphereState(
            pressure_grid=pressure_grid,
            temperature=contract["temperature_cells_k"][case_index],
            temperature_edges=contract["temperature_edges_k"][case_index],
            composition=composition,
            mean_molecular_weight=contract["mean_molecular_weight_cells"][case_index],
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
        runtime[case_index] = perf_counter() - started
        flux[case_index] = np.pi * np.asarray(result.radiance.values)
        shared_total_tau[case_index] = np.sum(
            np.asarray(result.total_optical_depth) * g_weights[None, None, :],
            axis=-1,
        )
        if contribution_mask[case_index]:
            contribution = np.array(result.layer_contribution_radiance, copy=True)
            contribution[-1] += np.asarray(result.bottom_contribution_radiance)
            output_contribution.append(normalize_contribution(contribution))
            contribution_case_index.append(case_index)
        del atmosphere, evaluated, opacity, gas_tau, cia, result
        gc.collect()
    return {
        "wavelength_micron": wavelength,
        "pressure_bar": pressure_centers,
        "flux_w_m2_m": flux,
        "normalized_contribution": np.asarray(output_contribution),
        "contribution_case_index": np.asarray(contribution_case_index, dtype=int),
        "shared_total_tau": shared_total_tau,
        "runtime_s": runtime,
    }


def _shared_robert_stage_6(
    contract: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from robert_exoplanets import planck_radiance_wavelength
    from robert_exoplanets.rt import integrate_thermal_emission_spectrum

    wavelength = contract["wavelength_micron"]
    mu = contract["emission_mu"]
    point_weights = contract["disk_weights"]
    shared_tau = contract["shared_total_tau"]
    flux = np.empty((contract["case_id"].size, wavelength.size))
    runtime = np.empty(contract["case_id"].size)
    for case_index, tau in enumerate(shared_tau):
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


def _composition_jacobians(
    binned_flux: np.ndarray, contract: dict[str, np.ndarray]
) -> np.ndarray:
    amplitude = float(contract["perturbation_amplitude_dex"])
    profile_count = contract["profile_name"].size
    output = np.empty(
        (
            profile_count,
            len(SPECIES),
            len(STAGE_5_PERTURBATION_CENTERS_BAR),
            binned_flux.shape[1],
        )
    )
    for profile_index in range(profile_count):
        for species_index in range(len(SPECIES)):
            for center_index in range(len(STAGE_5_PERTURBATION_CENTERS_BAR)):
                selected = (
                    (contract["profile_index"] == profile_index)
                    & (contract["target_species_index"] == species_index)
                    & (contract["perturbation_center_index"] == center_index)
                )
                minus = np.flatnonzero(selected & (contract["perturbation_sign"] == -1))
                plus = np.flatnonzero(selected & (contract["perturbation_sign"] == 1))
                if minus.size != 1 or plus.size != 1:
                    raise ValueError("finite-difference contract is not one-to-one")
                output[profile_index, species_index, center_index] = (
                    binned_flux[plus[0]] - binned_flux[minus[0]]
                ) / (2.0 * amplitude)
    flux_scale = np.max(np.abs(binned_flux), axis=0)
    return apply_jacobian_roundoff_floor(output, flux_scale, amplitude)


def _pairwise_signed_metrics(
    values: dict[str, np.ndarray], wavelength: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        f"{left}__{right}": signed_jacobian_metrics(
            values[left], values[right], wavelength
        )
        for left, right in combinations(values, 2)
    }


def _pairwise_pressure_metrics(
    values: dict[str, np.ndarray], pressure: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        f"{left}__{right}": contribution_metrics(values[left], values[right], pressure)
        for left, right in combinations(values, 2)
    }


def _pairwise_fraction_metrics(
    values: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {
        f"{left}__{right}": cross_species_fraction_metrics(values[left], values[right])
        for left, right in combinations(values, 2)
    }


def _run_contract(
    args: argparse.Namespace,
    contract: dict[str, np.ndarray],
    label: str,
    stage_dir: Path,
    edges: np.ndarray,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, Any]],
    list[Path],
]:
    outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {
        track: {} for track in TRACKS
    }
    metadata: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []

    if bool(getattr(args, "reuse_existing", False)):
        paths = [
            stage_dir / f"contract_track_b_{label}.npz",
            stage_dir / f"contract_track_a_{label}.npz",
            *(stage_dir / f"track_b_{model}_{label}.npz" for model in MODELS),
            *(stage_dir / f"track_a_{model}_{label}.npz" for model in MODELS),
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing reusable Stage-6 artifacts: {missing}")
        for track, short in (
            ("track_b_native_opacity", "track_b"),
            ("track_a_shared_tau", "track_a"),
        ):
            for model in MODELS:
                outputs[track][model] = _load(
                    stage_dir / f"{short}_{model}_{label}.npz"
                )
                if model != "robert":
                    metadata[f"{short}_{model}"] = json.loads(
                        str(outputs[track][model]["metadata_json"])
                    )
        for track in TRACKS:
            for model in MODELS:
                payload = outputs[track][model]
                binned_flux = bin_mean(
                    payload["wavelength_micron"], payload["flux_w_m2_m"], edges
                )
                jacobian = _composition_jacobians(binned_flux, contract)
                payload["flux_jacobian_r100"] = jacobian
                payload["normalized_response_r100"] = np.stack(
                    [normalize_composition_response(profile) for profile in jacobian]
                )
                payload["cross_species_fraction_r100"] = np.stack(
                    [
                        cross_species_sensitivity_fractions(profile)
                        for profile in jacobian
                    ]
                )
        return outputs, metadata, paths

    native_contract = {**contract, "track": np.array("track_b_native_opacity")}
    native_contract_path = stage_dir / f"contract_track_b_{label}.npz"
    _save_contract(native_contract_path, native_contract)
    paths.append(native_contract_path)
    for model, python in (
        ("picaso", args.picaso_python),
        ("petitradtrans", args.prt_python),
    ):
        output_path = stage_dir / f"track_b_{model}_{label}.npz"
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
        outputs["track_b_native_opacity"][model] = _load(output_path)
        metadata[f"track_b_{model}"] = json.loads(
            str(outputs["track_b_native_opacity"][model]["metadata_json"])
        )
        paths.append(output_path)

    robert_native = _native_robert_stage_6(contract, args.prt_input)
    outputs["track_b_native_opacity"]["robert"] = robert_native
    robert_native_path = stage_dir / f"track_b_robert_{label}.npz"
    _write_robert_output(
        robert_native_path, contract, robert_native, "native-composition"
    )
    paths.append(robert_native_path)

    shared_contract = {
        **contract,
        "track": np.array("track_a_shared_tau"),
        "wavelength_micron": robert_native["wavelength_micron"],
        "shared_total_tau": robert_native["shared_total_tau"],
        "native_return_contribution": np.array(False),
    }
    shared_contract_path = stage_dir / f"contract_track_a_{label}.npz"
    _save_contract(shared_contract_path, shared_contract)
    paths.append(shared_contract_path)
    for model, python in (
        ("picaso", args.picaso_python),
        ("petitradtrans", args.prt_python),
    ):
        output_path = stage_dir / f"track_a_{model}_{label}.npz"
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
        outputs["track_a_shared_tau"][model] = _load(output_path)
        metadata[f"track_a_{model}"] = json.loads(
            str(outputs["track_a_shared_tau"][model]["metadata_json"])
        )
        paths.append(output_path)

    robert_shared = _shared_robert_stage_6(shared_contract)
    outputs["track_a_shared_tau"]["robert"] = robert_shared
    robert_shared_path = stage_dir / f"track_a_robert_{label}.npz"
    _write_robert_output(
        robert_shared_path, shared_contract, robert_shared, "shared-composition"
    )
    paths.append(robert_shared_path)

    for track in TRACKS:
        for model in MODELS:
            payload = outputs[track][model]
            binned_flux = bin_mean(
                payload["wavelength_micron"], payload["flux_w_m2_m"], edges
            )
            jacobian = _composition_jacobians(binned_flux, contract)
            payload["flux_jacobian_r100"] = jacobian
            payload["normalized_response_r100"] = np.stack(
                [normalize_composition_response(profile) for profile in jacobian]
            )
            payload["cross_species_fraction_r100"] = np.stack(
                [cross_species_sensitivity_fractions(profile) for profile in jacobian]
            )
    return outputs, metadata, paths


def _relation_metrics(
    stage_5: dict[str, np.ndarray],
    track: str,
    resolution: int,
    model: str,
    profile_index: int,
    response: np.ndarray,
    centers: np.ndarray,
) -> dict[str, dict[str, float]]:
    stage_4 = stage_5[f"stage4_projected_contribution_L{resolution}_{model}"][
        profile_index
    ]
    temperature = stage_5[
        f"normalized_vertical_response_{track}_L{resolution}_{model}"
    ][profile_index]
    return {
        "stage_4_projected_contribution": contribution_metrics(
            response, stage_4, centers
        ),
        "stage_5_temperature_response": contribution_metrics(
            response, temperature, centers
        ),
    }


def _resolution_report(
    outputs: dict[str, dict[str, dict[str, np.ndarray]]],
    metadata: dict[str, dict[str, Any]],
    contract: dict[str, np.ndarray],
    resolution: int,
    wavelength: np.ndarray,
    stage_5: dict[str, np.ndarray],
) -> dict[str, Any]:
    centers = np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR)
    tracks: dict[str, Any] = {}
    for track in TRACKS:
        profiles: dict[str, Any] = {}
        for profile_index, profile_name in enumerate(contract["profile_name"]):
            species_report: dict[str, Any] = {}
            for species_index, species in enumerate(SPECIES):
                jacobians = {
                    model: outputs[track][model]["flux_jacobian_r100"][
                        profile_index, species_index
                    ]
                    for model in MODELS
                }
                responses = {
                    model: outputs[track][model]["normalized_response_r100"][
                        profile_index, species_index
                    ]
                    for model in MODELS
                }
                species_report[species] = {
                    "pairwise_signed_spectral_jacobian_r100": (
                        _pairwise_signed_metrics(jacobians, wavelength)
                    ),
                    "pairwise_vertical_response_r100": (
                        _pairwise_pressure_metrics(responses, centers)
                    ),
                    "framework_summary": {
                        model: {
                            **_response_summary(responses[model], centers),
                            "eclipse_jacobian_rms_ppm_per_dex": float(
                                np.sqrt(
                                    np.mean(
                                        eclipse_jacobian_ppm_per_dex(
                                            jacobians[model], wavelength
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            "eclipse_jacobian_max_abs_ppm_per_dex": float(
                                np.max(
                                    np.abs(
                                        eclipse_jacobian_ppm_per_dex(
                                            jacobians[model], wavelength
                                        )
                                    )
                                )
                            ),
                            "relation_to_stage_4_and_5": _relation_metrics(
                                stage_5,
                                track,
                                resolution,
                                model,
                                profile_index,
                                responses[model],
                                centers,
                            ),
                        }
                        for model in MODELS
                    },
                }
            fractions = {
                model: outputs[track][model]["cross_species_fraction_r100"][
                    profile_index
                ]
                for model in MODELS
            }
            profiles[str(profile_name)] = {
                "species": species_report,
                "pairwise_cross_species_sensitivity_fraction_r100": (
                    _pairwise_fraction_metrics(fractions)
                ),
            }
        tracks[track] = {
            "profiles": profiles,
            "timings": {
                model: {
                    "summary": _timing_summary(outputs[track][model]["runtime_s"]),
                    "raw_case_timings_s": outputs[track][model]["runtime_s"].tolist(),
                }
                for model in MODELS
            },
        }
    return {
        "tracks": tracks,
        "external_metadata": metadata,
        "robert_metadata": {
            "track_a": _metadata("robert", "shared-composition"),
            "track_b": _metadata("robert", "native-composition"),
        },
        "native_wavelength_count": {
            model: int(
                outputs["track_b_native_opacity"][model]["wavelength_micron"].size
            )
            for model in MODELS
        },
    }


def _self_convergence(
    main_outputs: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]],
    wavelength: np.ndarray,
) -> dict[str, Any]:
    centers = np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR)
    result: dict[str, Any] = {}
    for track in TRACKS:
        result[track] = {}
        for coarse, fine in ((40, 80), (80, 160)):
            pair = f"{coarse}_to_{fine}"
            result[track][pair] = {}
            for model in MODELS:
                result[track][pair][model] = {}
                for profile_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
                    species_metrics: dict[str, Any] = {}
                    for species_index, species in enumerate(SPECIES):
                        coarse_j = main_outputs[coarse][track][model][
                            "flux_jacobian_r100"
                        ][profile_index, species_index]
                        fine_j = main_outputs[fine][track][model]["flux_jacobian_r100"][
                            profile_index, species_index
                        ]
                        coarse_r = main_outputs[coarse][track][model][
                            "normalized_response_r100"
                        ][profile_index, species_index]
                        fine_r = main_outputs[fine][track][model][
                            "normalized_response_r100"
                        ][profile_index, species_index]
                        species_metrics[species] = {
                            "signed_spectral_jacobian_r100": (
                                signed_jacobian_metrics(coarse_j, fine_j, wavelength)
                            ),
                            "vertical_response_r100": contribution_metrics(
                                coarse_r, fine_r, centers
                            ),
                        }
                    coarse_f = main_outputs[coarse][track][model][
                        "cross_species_fraction_r100"
                    ][profile_index]
                    fine_f = main_outputs[fine][track][model][
                        "cross_species_fraction_r100"
                    ][profile_index]
                    result[track][pair][model][profile_name] = {
                        "species": species_metrics,
                        "cross_species_sensitivity_fraction_r100": (
                            cross_species_fraction_metrics(coarse_f, fine_f)
                        ),
                    }
    return result


def _linearity_report(
    main_outputs: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]],
    audit_outputs: dict[float, dict[str, dict[str, dict[str, np.ndarray]]]],
    wavelength: np.ndarray,
) -> dict[str, Any]:
    profile_lookup = {
        name: STAGE_4_PROFILE_NAMES.index(name) for name in AUDIT_PROFILES
    }
    result: dict[str, Any] = {}
    for track in TRACKS:
        result[track] = {}
        for model in MODELS:
            result[track][model] = {}
            for audit_profile_index, profile_name in enumerate(AUDIT_PROFILES):
                reference = main_outputs[PRIMARY_RESOLUTION][track][model][
                    "flux_jacobian_r100"
                ][profile_lookup[profile_name]]
                result[track][model][profile_name] = {}
                for species_index, species in enumerate(SPECIES):
                    result[track][model][profile_name][species] = {}
                    for amplitude in STAGE_6_LINEARITY_AMPLITUDES_DEX:
                        if amplitude == STAGE_6_PERTURBATION_AMPLITUDE_DEX:
                            candidate = reference
                        else:
                            candidate = audit_outputs[amplitude][track][model][
                                "flux_jacobian_r100"
                            ][audit_profile_index]
                        result[track][model][profile_name][species][
                            f"{amplitude:.2f}_vs_0.10_dex"
                        ] = amplitude_linearity_metrics(
                            reference[species_index],
                            candidate[species_index],
                            wavelength,
                        )
    return result


def _track_a_gate_results(
    main_outputs: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]],
    convergence: dict[str, Any],
    wavelength: np.ndarray,
) -> tuple[dict[str, dict[str, float | bool]], bool]:
    centers = np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR)
    primary_jacobian: list[dict[str, float]] = []
    primary_response: list[dict[str, float]] = []
    primary_fraction: list[dict[str, float]] = []
    primary = main_outputs[PRIMARY_RESOLUTION]["track_a_shared_tau"]
    for left, right in combinations(MODELS, 2):
        for profile_index in range(len(STAGE_4_PROFILE_NAMES)):
            for species_index in range(len(SPECIES)):
                primary_jacobian.append(
                    signed_jacobian_metrics(
                        primary[left]["flux_jacobian_r100"][
                            profile_index, species_index
                        ],
                        primary[right]["flux_jacobian_r100"][
                            profile_index, species_index
                        ],
                        wavelength,
                    )
                )
                primary_response.append(
                    contribution_metrics(
                        primary[left]["normalized_response_r100"][
                            profile_index, species_index
                        ],
                        primary[right]["normalized_response_r100"][
                            profile_index, species_index
                        ],
                        centers,
                    )
                )
            primary_fraction.append(
                cross_species_fraction_metrics(
                    primary[left]["cross_species_fraction_r100"][profile_index],
                    primary[right]["cross_species_fraction_r100"][profile_index],
                )
            )

    convergence_values = convergence["track_a_shared_tau"]["80_to_160"]
    convergence_jacobian = [
        species["signed_spectral_jacobian_r100"]
        for model in convergence_values.values()
        for profile in model.values()
        for species in profile["species"].values()
    ]
    convergence_response = [
        species["vertical_response_r100"]
        for model in convergence_values.values()
        for profile in model.values()
        for species in profile["species"].values()
    ]
    convergence_fraction = [
        profile["cross_species_sensitivity_fraction_r100"]
        for model in convergence_values.values()
        for profile in model.values()
    ]
    observed = {
        "primary_p95_abs_difference_over_pair_peak": max(
            value["p95_abs_difference_over_pair_peak"] for value in primary_jacobian
        ),
        "primary_rms_eclipse_jacobian_difference_ppm_per_dex": max(
            value["rms_eclipse_jacobian_difference_ppm_per_dex"]
            for value in primary_jacobian
        ),
        "primary_centroid_pressure_rms_difference_dex": max(
            value["centroid_pressure_rms_difference_dex"] for value in primary_response
        ),
        "primary_profile_total_variation_p95": max(
            value["profile_total_variation_p95"] for value in primary_response
        ),
        "primary_cross_species_total_variation_p95": max(
            value["species_total_variation_p95"] for value in primary_fraction
        ),
        "80_to_160_p95_abs_difference_over_pair_peak": max(
            value["p95_abs_difference_over_pair_peak"] for value in convergence_jacobian
        ),
        "80_to_160_rms_eclipse_jacobian_difference_ppm_per_dex": max(
            value["rms_eclipse_jacobian_difference_ppm_per_dex"]
            for value in convergence_jacobian
        ),
        "80_to_160_centroid_pressure_rms_difference_dex": max(
            value["centroid_pressure_rms_difference_dex"]
            for value in convergence_response
        ),
        "80_to_160_profile_total_variation_p95": max(
            value["profile_total_variation_p95"] for value in convergence_response
        ),
        "80_to_160_cross_species_total_variation_p95": max(
            value["species_total_variation_p95"] for value in convergence_fraction
        ),
    }
    results = {
        name: {
            "threshold": threshold,
            "observed": observed[name],
            "passed": bool(observed[name] <= threshold),
        }
        for name, threshold in TRACK_A_GATES.items()
    }
    return results, all(bool(value["passed"]) for value in results.values())


def _robert_prt_summary(
    main_outputs: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]],
    wavelength: np.ndarray,
) -> dict[str, float]:
    centers = np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR)
    track = main_outputs[PRIMARY_RESOLUTION]["track_b_native_opacity"]
    jacobian_metrics = []
    response_metrics = []
    fraction_metrics = []
    for profile_index in range(len(STAGE_4_PROFILE_NAMES)):
        for species_index in range(len(SPECIES)):
            jacobian_metrics.append(
                signed_jacobian_metrics(
                    track["robert"]["flux_jacobian_r100"][profile_index, species_index],
                    track["petitradtrans"]["flux_jacobian_r100"][
                        profile_index, species_index
                    ],
                    wavelength,
                )
            )
            response_metrics.append(
                contribution_metrics(
                    track["robert"]["normalized_response_r100"][
                        profile_index, species_index
                    ],
                    track["petitradtrans"]["normalized_response_r100"][
                        profile_index, species_index
                    ],
                    centers,
                )
            )
        fraction_metrics.append(
            cross_species_fraction_metrics(
                track["robert"]["cross_species_fraction_r100"][profile_index],
                track["petitradtrans"]["cross_species_fraction_r100"][profile_index],
            )
        )
    return {
        "worst_p95_abs_difference_over_pair_peak": max(
            value["p95_abs_difference_over_pair_peak"] for value in jacobian_metrics
        ),
        "worst_rms_eclipse_jacobian_difference_ppm_per_dex": max(
            value["rms_eclipse_jacobian_difference_ppm_per_dex"]
            for value in jacobian_metrics
        ),
        "worst_centroid_pressure_rms_difference_dex": max(
            value["centroid_pressure_rms_difference_dex"] for value in response_metrics
        ),
        "worst_profile_total_variation_p95": max(
            value["profile_total_variation_p95"] for value in response_metrics
        ),
        "worst_cross_species_total_variation_p95": max(
            value["species_total_variation_p95"] for value in fraction_metrics
        ),
    }


def _response_artifact(
    main_outputs: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]],
    audit_outputs: dict[float, dict[str, dict[str, dict[str, np.ndarray]]]],
    wavelength: np.ndarray,
    stage_5: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    artifact: dict[str, np.ndarray] = {
        "schema_version": np.array(1),
        "wavelength_r100_micron": wavelength,
        "profile_name": np.asarray(STAGE_4_PROFILE_NAMES),
        "species_name": np.asarray(SPECIES),
        "perturbation_centers_bar": np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
        "perturbation_amplitude_dex": np.array(STAGE_6_PERTURBATION_AMPLITUDE_DEX),
        "localization_sigma_dex": np.array(STAGE_5_LOCALIZATION_SIGMA_DEX),
        "linearity_amplitudes_dex": np.asarray(STAGE_6_LINEARITY_AMPLITUDES_DEX),
    }
    for resolution in RESOLUTIONS:
        for track in TRACKS:
            for model in MODELS:
                payload = main_outputs[resolution][track][model]
                prefix = f"{track}_L{resolution}_{model}"
                artifact[f"flux_jacobian_{prefix}_w_m2_m_dex"] = payload[
                    "flux_jacobian_r100"
                ]
                artifact[f"eclipse_jacobian_{prefix}_ppm_dex"] = (
                    eclipse_jacobian_ppm_per_dex(
                        payload["flux_jacobian_r100"], wavelength
                    )
                )
                artifact[f"normalized_vertical_response_{prefix}"] = payload[
                    "normalized_response_r100"
                ]
                artifact[f"cross_species_sensitivity_fraction_{prefix}"] = payload[
                    "cross_species_fraction_r100"
                ]
                artifact[f"stage5_temperature_response_{prefix}"] = stage_5[
                    f"normalized_vertical_response_{track}_L{resolution}_{model}"
                ]
                artifact[f"stage4_projected_contribution_L{resolution}_{model}"] = (
                    stage_5[f"stage4_projected_contribution_L{resolution}_{model}"]
                )
    for amplitude, outputs in audit_outputs.items():
        for track in TRACKS:
            for model in MODELS:
                prefix = f"audit_A{amplitude:.2f}_{track}_{model}"
                artifact[f"flux_jacobian_{prefix}_w_m2_m_dex"] = outputs[track][model][
                    "flux_jacobian_r100"
                ]
                artifact[f"eclipse_jacobian_{prefix}_ppm_dex"] = (
                    eclipse_jacobian_ppm_per_dex(
                        outputs[track][model]["flux_jacobian_r100"], wavelength
                    )
                )
                artifact[f"normalized_vertical_response_{prefix}"] = outputs[track][
                    model
                ]["normalized_response_r100"]
                artifact[f"cross_species_sensitivity_fraction_{prefix}"] = outputs[
                    track
                ][model]["cross_species_fraction_r100"]
    return artifact


def _run_stage_6(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stage_dir = args.output_root / "stage_6"
    stage_dir.mkdir(parents=True, exist_ok=True)
    edges = r100_edges()
    wavelength = np.sqrt(edges[:-1] * edges[1:])
    stage_5 = _load(STAGE_5_RESPONSE_PATH)
    main_outputs: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    per_resolution: dict[str, Any] = {}
    artifact_checksums: dict[str, str] = {}

    for resolution in RESOLUTIONS:
        contract = stage_6_contract(resolution)
        outputs, metadata, paths = _run_contract(
            args, contract, f"main_L{resolution}", stage_dir, edges
        )
        main_outputs[resolution] = outputs
        per_resolution[str(resolution)] = _resolution_report(
            outputs,
            metadata,
            contract,
            resolution,
            wavelength,
            stage_5,
        )
        for path in paths:
            artifact_checksums[str(path.relative_to(args.output_root))] = sha256(path)
        gc.collect()

    audit_outputs: dict[float, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    audit_execution: dict[str, Any] = {}
    for amplitude in STAGE_6_LINEARITY_AMPLITUDES_DEX:
        if amplitude == STAGE_6_PERTURBATION_AMPLITUDE_DEX:
            continue
        contract = stage_6_contract(
            PRIMARY_RESOLUTION,
            amplitude_dex=amplitude,
            profile_names=AUDIT_PROFILES,
        )
        outputs, metadata, paths = _run_contract(
            args,
            contract,
            f"audit_A{amplitude:.2f}_L{PRIMARY_RESOLUTION}",
            stage_dir,
            edges,
        )
        audit_outputs[amplitude] = outputs
        audit_execution[f"{amplitude:.2f}_dex"] = {
            "external_metadata": metadata,
            "timings": {
                track: {
                    model: {
                        "summary": _timing_summary(outputs[track][model]["runtime_s"]),
                        "raw_case_timings_s": outputs[track][model][
                            "runtime_s"
                        ].tolist(),
                    }
                    for model in MODELS
                }
                for track in TRACKS
            },
        }
        for path in paths:
            artifact_checksums[str(path.relative_to(args.output_root))] = sha256(path)
        gc.collect()

    convergence = _self_convergence(main_outputs, wavelength)
    linearity = _linearity_report(main_outputs, audit_outputs, wavelength)
    gate_results, acceptance_passed = _track_a_gate_results(
        main_outputs, convergence, wavelength
    )
    response_artifact = _response_artifact(
        main_outputs, audit_outputs, wavelength, stage_5
    )

    input_paths = _opacity_paths(args.prt_input)
    input_paths["PICASO-database"] = args.picaso_database
    input_paths["PICASO-reference-config"] = args.picaso_reference / "config.json"
    input_paths["PICASO-reference-version"] = args.picaso_reference / "version.md"
    input_paths["Stage-5-response-artifact"] = STAGE_5_RESPONSE_PATH
    input_paths["Stage-5-report"] = DEFAULT_REPORTS / "stage_5_report.json"
    source_paths = {
        "stage_6_launcher": Path(__file__),
        "shared_contracts_and_metrics": Path(__file__).with_name(
            "emission_intercomparison_common.py"
        ),
        "external_worker": Path(__file__).with_name(
            "run_emission_intercomparison_external.py"
        ),
        "predeclared_review": (
            Path(__file__).resolve().parents[1]
            / "docs/review/49_emission_intercomparison_stage_6.md"
        ),
    }
    report = {
        "schema_version": 1,
        "stage": 6,
        "status": "passed" if acceptance_passed else "failed_track_a_gates",
        "orchestrator": _robert_metadata(),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": PRIMARY_RESOLUTION,
        "profiles": list(STAGE_4_PROFILE_NAMES),
        "species": list(SPECIES),
        "picaso_native_resample": int(args.picaso_resample),
        "environment_notes": {
            "picaso_reference_configuration": "3.2.1",
            "picaso_reference_warning": (
                "PICASO code 3.2.2 reports reference configuration 3.2.1; "
                "the known minor-version warning is retained"
            ),
            "petitradtrans_home": (
                "private writable HOME beneath the ignored Stage-6 output directory"
            ),
        },
        "vertical_grid_contract": {
            "robert": "40/80/160 pressure cells bounded by pressure_edges_bar",
            "picaso": "matching 41/81/161 pressure edges supplied as levels",
            "petitradtrans": (
                "ROBERT geometric cell centres supplied as 40/80/160 pressure nodes"
            ),
            "response_coordinate": "six fixed perturbation centres in bar",
        },
        "method_definitions": {
            "finite_difference": (
                "J_s(P0,lambda)=[F(q_s 10^(+Delta L))-"
                "F(q_s 10^(-Delta L))]/(2 Delta), Delta=0.10 dex"
            ),
            "localization": (
                "L(P;P0)=exp[-0.5*(log10(P/P0)/0.35 dex)^2] at six centres "
                "from 1e-4 to 10 bar"
            ),
            "composition": (
                "target VMR is perturbed; other molecular VMRs stay fixed; the "
                "remaining atmosphere is 85:15 H2/He and sums explicitly to unity"
            ),
            "vertical_response": (
                "absolute signed R=100 species Jacobian normalized over six pressure "
                "centres independently for each species and wavelength"
            ),
            "roundoff_floor": (
                "before normalization, central-difference values no larger than "
                "32*machine_epsilon*max_abs_case_flux/Delta are set to exact zero; "
                "this prevents analytically zero isothermal derivatives from "
                "normalizing floating-point cancellation noise"
            ),
            "cross_species_fraction": (
                "sum of absolute Jacobian over pressure centres, normalized over "
                "H2O, CO, CO2, and CH4 independently at each wavelength"
            ),
            "track_a": (
                "case-specific plus/minus ROBERT source-HDF gas+CIA total optical "
                "depths, g-weighted per layer and wavelength and supplied identically "
                "with the same temperatures to all three pure-absorption RT paths"
            ),
            "track_b": (
                "each framework recomputes abundance-dependent molecular opacity, "
                "correlated-k mixing, mean molecular weight, and H2-H2/H2-He CIA"
            ),
            "linearity_audit": (
                "central-difference Jacobians at 0.05, 0.10, and 0.20 dex for all "
                "species in monotonic and retrieved-like profiles; reported metrics "
                "are truncation/nonlinearity differences relative to 0.10 dex"
            ),
            "diagnostic_distinction": (
                "a Stage-4 contribution function decomposes emergent intensity; a "
                "Stage-5 temperature derivative differentiates thermal structure; "
                "a Stage-6 composition derivative may change sign and includes opacity "
                "redistribution, gas overlap, CIA, mean-molecular-weight, and "
                "background-gas effects"
            ),
        },
        "tracks": {
            "track_a_shared_tau": {
                "status": "passed" if acceptance_passed else "failed",
                "cross_code_acceptance_gates": gate_results,
            },
            "track_b_native_opacity": {
                "status": "characterized_no_picaso_cross_code_gate",
                "robert_petitradtrans_primary_summary": _robert_prt_summary(
                    main_outputs, wavelength
                ),
                "picaso_attribution": (
                    "PICASO uses its independent official SQLite opacity database; "
                    "all numerical results are retained as attribution results, not "
                    "cross-code acceptance gates"
                ),
            },
        },
        "per_resolution": per_resolution,
        "self_convergence": convergence,
        "linearity_audit": {
            "profiles": list(AUDIT_PROFILES),
            "amplitudes_dex": list(STAGE_6_LINEARITY_AMPLITUDES_DEX),
            "metrics": linearity,
            "execution": audit_execution,
        },
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
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reanalyze complete Stage-6 worker artifacts without rerunning solvers",
    )
    parser.add_argument(
        "--full-matrix-wall-time-s",
        type=float,
        help="preserve measured full-run wall time during --reuse-existing analysis",
    )
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.report_root = args.report_root.resolve()
    args.report_root.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    report, response_artifact = _run_stage_6(args)
    response_path = args.report_root / "stage_6_response_tensors.npz"
    np.savez_compressed(response_path, **response_artifact)
    report["response_artifact"] = {
        "path": response_path.name,
        "sha256": sha256(response_path),
        "contents": (
            "complete signed R=100 species-by-pressure-by-wavelength flux and "
            "eclipse Jacobians, normalized vertical responses, cross-species "
            "fractions, linearity tensors, and Stage-4/5 reference diagnostics"
        ),
    }
    elapsed = perf_counter() - started
    if args.reuse_existing:
        report["analysis_wall_time_s"] = elapsed
        report["wall_time_s"] = (
            float(args.full_matrix_wall_time_s)
            if args.full_matrix_wall_time_s is not None
            else elapsed
        )
    else:
        report["wall_time_s"] = elapsed
    write_json(args.report_root / "stage_6_report.json", report)
    write_checksums(args.report_root)
    print(json.dumps({"stage": 6, "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
