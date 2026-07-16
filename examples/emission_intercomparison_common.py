"""Shared contracts and metrics for the emission intercomparison benchmarks."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

SPECIES = ("H2O", "CO", "CO2", "CH4")
MOLAR_MASS = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.01528,
    "CO": 28.0101,
    "CO2": 44.0095,
    "CH4": 16.04246,
}
STAR_TEMPERATURE_K = 5800.0
PLANET_RADIUS_M = 7.1492e7
STAR_RADIUS_M = 6.957e8
STAGE_1_TEMPERATURES_K = (500.0, 1000.0, 1500.0, 2000.0)
STAGE_1_TOTAL_TAU = (1.0e-4, 1.0e-2, 1.0, 100.0)
STAGE_2_VMR = tuple(float(value) for value in np.geomspace(1.0e-6, 1.0e-1, 6))
STAGE_3_MOLECULAR_VMR = {
    "H2O": 1.0e-3,
    "CO": 3.0e-4,
    "CO2": 1.0e-4,
    "CH4": 1.0e-5,
}
STAGE_4_PROFILE_NAMES = (
    "isothermal",
    "monotonic",
    "inverted",
    "retrieved_like",
)
STAGE_5_PERTURBATION_CENTERS_BAR = tuple(
    float(value) for value in np.geomspace(1.0e-4, 10.0, 6)
)
STAGE_5_PERTURBATION_AMPLITUDE_K = 10.0
STAGE_5_LOCALIZATION_SIGMA_DEX = 0.35
STAGE_6_PERTURBATION_AMPLITUDE_DEX = 0.10
STAGE_6_LINEARITY_AMPLITUDES_DEX = (0.05, 0.10, 0.20)
STAGE_7_CLOUD_OPTICAL_DEPTHS = (0.1, 1.0, 10.0, 100.0)
STAGE_7_CLOUD_TOP_PRESSURES_BAR = (1.0e-3, 1.0e-2, 1.0e-1)
STAGE_7_EXTINCTION_SLOPES = (-4.0, -2.0, 0.0, 2.0)
STAGE_7_REFERENCE_WAVELENGTH_MICRON = 5.0


def disk_quadrature(n_mu: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mu, Gauss-Legendre weights, and normalized disk weights."""

    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    mu = 0.5 * (nodes + 1.0)
    legendre = 0.5 * weights
    disk = 2.0 * mu * legendre
    disk /= disk.sum()
    return mu, legendre, disk


def pressure_grid(n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.geomspace(1.0e-5, 100.0, n_layers + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def temperature_profile(pressure_bar: np.ndarray, anchor_k: float) -> np.ndarray:
    pressure = np.asarray(pressure_bar, dtype=float)
    coordinate = 2.0 * (np.log10(pressure) + 5.0) / 7.0 - 1.0
    return float(anchor_k) + 150.0 * coordinate


def stage_4_temperature_profile(
    pressure_bar: np.ndarray, profile_name: str
) -> np.ndarray:
    """Return one deterministic Stage-4 thermal profile on arbitrary pressures.

    The profiles are defined in log pressure so ROBERT cell edges, ROBERT cell
    centres, PICASO levels, and pRT pressure nodes all sample the same
    continuous thermal structure.
    """

    pressure = np.asarray(pressure_bar, dtype=float)
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("pressure_bar must contain finite positive values")
    coordinate = (np.log10(pressure) + 5.0) / 7.0
    if profile_name == "isothermal":
        return np.full_like(pressure, 1250.0)
    if profile_name == "monotonic":
        return 800.0 + 900.0 * coordinate
    if profile_name == "inverted":
        return np.interp(
            coordinate,
            [0.0, 0.25, 0.45, 0.65, 1.0],
            [850.0, 1050.0, 1500.0, 1300.0, 1750.0],
        )
    if profile_name == "retrieved_like":
        return np.interp(
            coordinate,
            [0.0, 0.12, 0.28, 0.45, 0.62, 0.78, 1.0],
            [760.0, 780.0, 900.0, 1240.0, 1480.0, 1580.0, 1660.0],
        )
    raise ValueError(f"unsupported Stage-4 profile: {profile_name}")


def background_composition(
    molecular_vmr: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Complete a molecular VMR mapping with an 85:15 H2/He remainder."""

    molecular_total = float(sum(molecular_vmr.values()))
    if molecular_total < 0.0 or molecular_total >= 1.0:
        raise ValueError("molecular VMR sum must lie in [0, 1)")
    remainder = 1.0 - molecular_total
    composition = {
        "H2": 0.85 * remainder,
        "He": 0.15 * remainder,
        **{name: float(molecular_vmr.get(name, 0.0)) for name in SPECIES},
    }
    mean_molar_mass = sum(composition[name] * MOLAR_MASS[name] for name in composition)
    return composition, float(mean_molar_mass)


def complete_composition_profile(molecular_vmr: np.ndarray) -> np.ndarray:
    """Add an 85:15 H2/He remainder to pressure-dependent molecular VMRs."""

    molecular = np.asarray(molecular_vmr, dtype=float)
    if molecular.shape[-1:] != (len(SPECIES),):
        raise ValueError("molecular_vmr must end with the four-species axis")
    if np.any(~np.isfinite(molecular)) or np.any(molecular < 0.0):
        raise ValueError("molecular_vmr must be finite and non-negative")
    molecular_total = np.sum(molecular, axis=-1)
    if np.any(molecular_total >= 1.0):
        raise ValueError("molecular VMR sum must be less than one everywhere")
    remainder = 1.0 - molecular_total
    return np.concatenate(
        (
            (0.85 * remainder)[..., None],
            (0.15 * remainder)[..., None],
            molecular,
        ),
        axis=-1,
    )


def mean_molecular_weight_profile(gas_vmr: np.ndarray) -> np.ndarray:
    """Return mean molecular weight for H2, He, H2O, CO, CO2, CH4 VMRs."""

    vmr = np.asarray(gas_vmr, dtype=float)
    names = ("H2", "He", *SPECIES)
    if vmr.shape[-1:] != (len(names),):
        raise ValueError("gas_vmr must end with H2, He, and four species")
    return np.sum(vmr * np.asarray([MOLAR_MASS[name] for name in names]), axis=-1)


def localized_log_vmr_composition(
    pressure_bar: np.ndarray,
    target_species: str,
    center_bar: float,
    sign: int,
    amplitude_dex: float,
    *,
    sigma_dex: float = STAGE_5_LOCALIZATION_SIGMA_DEX,
) -> np.ndarray:
    """Return a renormalized localized +/- log10-VMR perturbation profile."""

    if target_species not in SPECIES:
        raise ValueError(f"unsupported target species: {target_species}")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    amplitude = float(amplitude_dex)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude_dex must be finite and positive")
    pressure = np.asarray(pressure_bar, dtype=float)
    molecular = np.broadcast_to(
        np.asarray([STAGE_3_MOLECULAR_VMR[name] for name in SPECIES]),
        (pressure.size, len(SPECIES)),
    ).copy()
    species_index = SPECIES.index(target_species)
    localization = temperature_localization(pressure, center_bar, sigma_dex=sigma_dex)
    molecular[:, species_index] *= 10.0 ** (sign * amplitude * localization)
    return complete_composition_profile(molecular)


def r100_edges(lower: float = 0.51, upper: float = 11.9) -> np.ndarray:
    count = int(np.floor(100.0 * np.log(upper / lower))) + 1
    return np.geomspace(lower, upper, count + 1)


def stage_1_contract(n_layers: int) -> dict[str, np.ndarray]:
    """Build the shared-grey/isothermal Track-A contract."""

    pressure_edges, _ = pressure_grid(n_layers)
    wavelength = np.geomspace(0.5, 12.0, 320)
    mu, legendre, disk = disk_quadrature()
    case_ids: list[str] = []
    temperatures: list[np.ndarray] = []
    component_tau: list[np.ndarray] = []
    pressure_fraction = np.diff(pressure_edges) / np.ptp(pressure_edges)
    for temperature in STAGE_1_TEMPERATURES_K:
        for total_tau in STAGE_1_TOTAL_TAU:
            case_ids.append(f"T{int(temperature)}_tau{total_tau:g}_L{n_layers}")
            temperatures.append(np.full(n_layers + 1, temperature))
            component_tau.append(
                total_tau * pressure_fraction[:, None] * np.ones((1, wavelength.size))
            )
    return {
        "schema_version": np.array(1),
        "stage": np.array(1),
        "case_id": np.asarray(case_ids),
        "wavelength_micron": wavelength,
        "pressure_edges_bar": pressure_edges,
        "temperature_edges_k": np.asarray(temperatures),
        "component_tau": np.asarray(component_tau)[:, None, :, :],
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def stage_2_contract(n_layers: int) -> dict[str, np.ndarray]:
    """Build the single-molecule/shared-tau Track-A contract."""

    pressure_edges, _ = pressure_grid(n_layers)
    wavelength = np.geomspace(0.5, 12.0, 320)
    mu, legendre, disk = disk_quadrature()
    pressure_fraction = np.diff(pressure_edges) / np.ptp(pressure_edges)
    case_ids: list[str] = []
    temperatures: list[np.ndarray] = []
    optical_depths: list[np.ndarray] = []
    gas_vmr: list[np.ndarray] = []
    for species_index, species in enumerate(SPECIES):
        template = molecular_band_template(wavelength, species)
        for anchor in STAGE_1_TEMPERATURES_K:
            for vmr in STAGE_2_VMR:
                case_ids.append(f"{species}_T{int(anchor)}_vmr{vmr:.0e}_L{n_layers}")
                temperatures.append(temperature_profile(pressure_edges, anchor))
                components = np.zeros((len(SPECIES), n_layers, wavelength.size))
                temperature_scale = (anchor / 1000.0) ** 0.25
                components[species_index] = (
                    pressure_fraction[:, None]
                    * template[None, :]
                    * (vmr / 1.0e-3)
                    * temperature_scale
                )
                optical_depths.append(components)
                composition, _ = background_composition({species: vmr})
                gas_vmr.append(
                    np.array(
                        [
                            composition["H2"],
                            composition["He"],
                            *(composition[name] for name in SPECIES),
                        ]
                    )
                )
    return {
        "schema_version": np.array(1),
        "stage": np.array(2),
        "case_id": np.asarray(case_ids),
        "wavelength_micron": wavelength,
        "pressure_edges_bar": pressure_edges,
        "temperature_edges_k": np.asarray(temperatures),
        "component_tau": np.asarray(optical_depths),
        "gas_vmr": np.asarray(gas_vmr),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def stage_3_contract(
    n_layers: int, *, include_cia: bool = True
) -> dict[str, np.ndarray]:
    """Build the four-molecule native-opacity Track-B contract."""

    pressure_edges, _ = pressure_grid(n_layers)
    composition, _ = background_composition(STAGE_3_MOLECULAR_VMR)
    mu, legendre, disk = disk_quadrature()
    vmr = np.array(
        [
            composition["H2"],
            composition["He"],
            *(composition[name] for name in SPECIES),
        ]
    )
    return {
        "schema_version": np.array(1),
        "stage": np.array(3),
        "case_id": np.asarray([f"full_L{n_layers}"]),
        "pressure_edges_bar": pressure_edges,
        "temperature_edges_k": temperature_profile(pressure_edges, 1250.0)[None, :],
        "gas_vmr": vmr[None, :],
        "native_include_cia": np.array(include_cia),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def stage_4_contract(
    n_cells: int, *, include_cia: bool = True
) -> dict[str, np.ndarray]:
    """Build the native-opacity thermal-structure Track-B contract.

    ROBERT and PICASO use ``pressure_edges_bar`` to describe ``n_cells``
    atmospheric cells.  pRT defines its atmosphere on pressure nodes, so it
    receives the geometric cell centres in ``prt_pressure_bar``.  This gives
    every code exactly ``n_cells`` contribution coordinates.
    """

    pressure_edges, pressure_centers = pressure_grid(n_cells)
    composition, _ = background_composition(STAGE_3_MOLECULAR_VMR)
    mu, legendre, disk = disk_quadrature()
    vmr = np.array(
        [
            composition["H2"],
            composition["He"],
            *(composition[name] for name in SPECIES),
        ]
    )
    edge_temperatures = np.stack(
        [
            stage_4_temperature_profile(pressure_edges, name)
            for name in STAGE_4_PROFILE_NAMES
        ]
    )
    cell_temperatures = np.stack(
        [
            stage_4_temperature_profile(pressure_centers, name)
            for name in STAGE_4_PROFILE_NAMES
        ]
    )
    return {
        "schema_version": np.array(1),
        "stage": np.array(4),
        "case_id": np.asarray([f"{name}_L{n_cells}" for name in STAGE_4_PROFILE_NAMES]),
        "profile_name": np.asarray(STAGE_4_PROFILE_NAMES),
        "pressure_edges_bar": pressure_edges,
        "pressure_centers_bar": pressure_centers,
        "prt_pressure_bar": pressure_centers,
        "temperature_edges_k": edge_temperatures,
        "temperature_cells_k": cell_temperatures,
        "gas_vmr": np.broadcast_to(vmr, (len(STAGE_4_PROFILE_NAMES), vmr.size)),
        "native_include_cia": np.array(include_cia),
        "native_return_contribution": np.array(True),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def temperature_localization(
    pressure_bar: np.ndarray,
    center_bar: float,
    *,
    sigma_dex: float = STAGE_5_LOCALIZATION_SIGMA_DEX,
) -> np.ndarray:
    """Return a unit-peak Gaussian temperature perturbation in log pressure."""

    pressure = np.asarray(pressure_bar, dtype=float)
    center = float(center_bar)
    sigma = float(sigma_dex)
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("pressure_bar must contain finite positive values")
    if not np.isfinite(center) or center <= 0.0:
        raise ValueError("center_bar must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_dex must be finite and positive")
    offset = np.log10(pressure / center) / sigma
    return np.exp(-0.5 * offset**2)


def stage_5_contract(
    n_cells: int,
    *,
    include_cia: bool = True,
    amplitude_k: float = STAGE_5_PERTURBATION_AMPLITUDE_K,
    sigma_dex: float = STAGE_5_LOCALIZATION_SIGMA_DEX,
) -> dict[str, np.ndarray]:
    """Build the Stage-5 localized-temperature finite-difference contract.

    The first four cases are unperturbed baselines, one for each Stage-4
    profile.  They are followed by symmetric minus/plus cases for every
    profile and perturbation centre.  PICASO samples the continuous profile on
    cell edges; ROBERT and pRT sample it at ROBERT geometric cell centres.
    """

    amplitude = float(amplitude_k)
    sigma = float(sigma_dex)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude_k must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_dex must be finite and positive")
    pressure_edges, pressure_centers = pressure_grid(n_cells)
    composition, _ = background_composition(STAGE_3_MOLECULAR_VMR)
    mu, legendre, disk = disk_quadrature()
    vmr = np.array(
        [
            composition["H2"],
            composition["He"],
            *(composition[name] for name in SPECIES),
        ]
    )
    case_ids: list[str] = []
    profile_indices: list[int] = []
    center_indices: list[int] = []
    signs: list[int] = []
    edge_temperatures: list[np.ndarray] = []
    cell_temperatures: list[np.ndarray] = []
    for profile_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
        case_ids.append(f"{profile_name}_baseline_L{n_cells}")
        profile_indices.append(profile_index)
        center_indices.append(-1)
        signs.append(0)
        edge_temperatures.append(
            stage_4_temperature_profile(pressure_edges, profile_name)
        )
        cell_temperatures.append(
            stage_4_temperature_profile(pressure_centers, profile_name)
        )
    for profile_index, profile_name in enumerate(STAGE_4_PROFILE_NAMES):
        baseline_edges = stage_4_temperature_profile(pressure_edges, profile_name)
        baseline_cells = stage_4_temperature_profile(pressure_centers, profile_name)
        for center_index, center in enumerate(STAGE_5_PERTURBATION_CENTERS_BAR):
            edge_shape = temperature_localization(
                pressure_edges, center, sigma_dex=sigma
            )
            cell_shape = temperature_localization(
                pressure_centers, center, sigma_dex=sigma
            )
            for sign, label in ((-1, "minus"), (1, "plus")):
                case_ids.append(f"{profile_name}_P{center:.0e}_{label}_L{n_cells}")
                profile_indices.append(profile_index)
                center_indices.append(center_index)
                signs.append(sign)
                edge_temperatures.append(baseline_edges + sign * amplitude * edge_shape)
                cell_temperatures.append(baseline_cells + sign * amplitude * cell_shape)
    case_count = len(case_ids)
    contribution_mask = np.asarray(signs) == 0
    return {
        "schema_version": np.array(1),
        "stage": np.array(5),
        "case_id": np.asarray(case_ids),
        "profile_name": np.asarray(STAGE_4_PROFILE_NAMES),
        "profile_index": np.asarray(profile_indices, dtype=int),
        "perturbation_center_index": np.asarray(center_indices, dtype=int),
        "perturbation_sign": np.asarray(signs, dtype=int),
        "perturbation_centers_bar": np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
        "perturbation_amplitude_k": np.array(amplitude),
        "localization_sigma_dex": np.array(sigma),
        "pressure_edges_bar": pressure_edges,
        "pressure_centers_bar": pressure_centers,
        "prt_pressure_bar": pressure_centers,
        "temperature_edges_k": np.asarray(edge_temperatures),
        "temperature_cells_k": np.asarray(cell_temperatures),
        "gas_vmr": np.broadcast_to(vmr, (case_count, vmr.size)),
        "native_include_cia": np.array(include_cia),
        "native_return_contribution": np.array(True),
        "native_contribution_case_mask": contribution_mask,
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def stage_6_contract(
    n_cells: int,
    *,
    include_cia: bool = True,
    amplitude_dex: float = STAGE_6_PERTURBATION_AMPLITUDE_DEX,
    sigma_dex: float = STAGE_5_LOCALIZATION_SIGMA_DEX,
    profile_names: tuple[str, ...] = STAGE_4_PROFILE_NAMES,
) -> dict[str, np.ndarray]:
    """Build localized composition finite-difference cases for Stage 6.

    PICASO compositions and temperatures are sampled at pressure edges.
    ROBERT cell quantities and pRT nodes use the geometric cell centres.
    Baseline cases precede symmetric minus/plus cases for each target species
    and perturbation centre.
    """

    amplitude = float(amplitude_dex)
    sigma = float(sigma_dex)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude_dex must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_dex must be finite and positive")
    profiles = tuple(profile_names)
    if not profiles or any(name not in STAGE_4_PROFILE_NAMES for name in profiles):
        raise ValueError("profile_names must contain supported Stage-4 profiles")

    pressure_edges, pressure_centers = pressure_grid(n_cells)
    mu, legendre, disk = disk_quadrature()
    baseline_molecular_edges = np.broadcast_to(
        np.asarray([STAGE_3_MOLECULAR_VMR[name] for name in SPECIES]),
        (pressure_edges.size, len(SPECIES)),
    )
    baseline_molecular_cells = np.broadcast_to(
        np.asarray([STAGE_3_MOLECULAR_VMR[name] for name in SPECIES]),
        (pressure_centers.size, len(SPECIES)),
    )
    baseline_edges = complete_composition_profile(baseline_molecular_edges)
    baseline_cells = complete_composition_profile(baseline_molecular_cells)

    case_ids: list[str] = []
    profile_indices: list[int] = []
    species_indices: list[int] = []
    center_indices: list[int] = []
    signs: list[int] = []
    edge_temperatures: list[np.ndarray] = []
    cell_temperatures: list[np.ndarray] = []
    edge_compositions: list[np.ndarray] = []
    cell_compositions: list[np.ndarray] = []

    for profile_index, profile_name in enumerate(profiles):
        case_ids.append(f"{profile_name}_baseline_L{n_cells}")
        profile_indices.append(profile_index)
        species_indices.append(-1)
        center_indices.append(-1)
        signs.append(0)
        edge_temperatures.append(
            stage_4_temperature_profile(pressure_edges, profile_name)
        )
        cell_temperatures.append(
            stage_4_temperature_profile(pressure_centers, profile_name)
        )
        edge_compositions.append(baseline_edges)
        cell_compositions.append(baseline_cells)

    for profile_index, profile_name in enumerate(profiles):
        profile_edges = stage_4_temperature_profile(pressure_edges, profile_name)
        profile_cells = stage_4_temperature_profile(pressure_centers, profile_name)
        for species_index, species in enumerate(SPECIES):
            for center_index, center in enumerate(STAGE_5_PERTURBATION_CENTERS_BAR):
                for sign, label in ((-1, "minus"), (1, "plus")):
                    case_ids.append(
                        f"{profile_name}_{species}_P{center:.0e}_{label}_"
                        f"A{amplitude:.2f}_L{n_cells}"
                    )
                    profile_indices.append(profile_index)
                    species_indices.append(species_index)
                    center_indices.append(center_index)
                    signs.append(sign)
                    edge_temperatures.append(profile_edges)
                    cell_temperatures.append(profile_cells)
                    edge_compositions.append(
                        localized_log_vmr_composition(
                            pressure_edges,
                            species,
                            center,
                            sign,
                            amplitude,
                            sigma_dex=sigma,
                        )
                    )
                    cell_compositions.append(
                        localized_log_vmr_composition(
                            pressure_centers,
                            species,
                            center,
                            sign,
                            amplitude,
                            sigma_dex=sigma,
                        )
                    )

    gas_vmr_edges = np.asarray(edge_compositions)
    gas_vmr_cells = np.asarray(cell_compositions)
    contribution_mask = np.asarray(signs) == 0
    return {
        "schema_version": np.array(1),
        "stage": np.array(6),
        "case_id": np.asarray(case_ids),
        "profile_name": np.asarray(profiles),
        "species_name": np.asarray(SPECIES),
        "profile_index": np.asarray(profile_indices, dtype=int),
        "target_species_index": np.asarray(species_indices, dtype=int),
        "perturbation_center_index": np.asarray(center_indices, dtype=int),
        "perturbation_sign": np.asarray(signs, dtype=int),
        "perturbation_centers_bar": np.asarray(STAGE_5_PERTURBATION_CENTERS_BAR),
        "perturbation_amplitude_dex": np.array(amplitude),
        "localization_sigma_dex": np.array(sigma),
        "pressure_edges_bar": pressure_edges,
        "pressure_centers_bar": pressure_centers,
        "prt_pressure_bar": pressure_centers,
        "temperature_edges_k": np.asarray(edge_temperatures),
        "temperature_cells_k": np.asarray(cell_temperatures),
        "gas_vmr_edges": gas_vmr_edges,
        "gas_vmr_cells": gas_vmr_cells,
        "gas_vmr": gas_vmr_cells,
        "mean_molecular_weight_edges": mean_molecular_weight_profile(gas_vmr_edges),
        "mean_molecular_weight_cells": mean_molecular_weight_profile(gas_vmr_cells),
        "native_include_cia": np.array(include_cia),
        "native_return_contribution": np.array(True),
        "native_contribution_case_mask": contribution_mask,
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def absorbing_cloud_tau(
    pressure_edges_bar: np.ndarray,
    wavelength_micron: np.ndarray,
    *,
    optical_depth_at_reference: float,
    cloud_top_pressure_bar: float,
    slope: float,
    reference_wavelength_micron: float = STAGE_7_REFERENCE_WAVELENGTH_MICRON,
) -> np.ndarray:
    """Return the frozen Stage-7 absorbing-deck extinction contract.

    The reference optical depth is distributed uniformly in log pressure below
    the cloud top, including fractional overlap of the intersected layer.  Its
    spectral law is ``tau(lambda)=tau_ref*(lambda/lambda_ref)**slope``.
    """

    edges = np.asarray(pressure_edges_bar, dtype=float)
    wavelength = np.asarray(wavelength_micron, dtype=float)
    tau_reference = float(optical_depth_at_reference)
    top = float(cloud_top_pressure_bar)
    spectral_slope = float(slope)
    reference = float(reference_wavelength_micron)
    if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)):
        raise ValueError("pressure_edges_bar must be a finite one-dimensional grid")
    if np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("pressure_edges_bar must be positive and increasing")
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)):
        raise ValueError("wavelength_micron must be a finite one-dimensional grid")
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_micron must be positive")
    if not np.isfinite(tau_reference) or tau_reference < 0.0:
        raise ValueError("optical_depth_at_reference must be finite and non-negative")
    if not np.isfinite(top) or top <= 0.0:
        raise ValueError("cloud_top_pressure_bar must be finite and positive")
    if not np.isfinite(spectral_slope):
        raise ValueError("slope must be finite")
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_wavelength_micron must be finite and positive")
    overlap_low = np.maximum(edges[:-1], top)
    overlap = np.where(edges[1:] > overlap_low, np.log(edges[1:] / overlap_low), 0.0)
    if not np.any(overlap > 0.0):
        raise ValueError("cloud top leaves no active pressure interval")
    vertical_fraction = overlap / np.sum(overlap)
    spectral_tau = tau_reference * (wavelength / reference) ** spectral_slope
    return vertical_fraction[:, None] * spectral_tau[None, :]


def regrid_tabulated_cloud_tau(
    source_pressure_edges_bar: np.ndarray,
    source_wavelength_micron: np.ndarray,
    source_extinction_tau: np.ndarray,
    target_pressure_edges_bar: np.ndarray,
    target_wavelength_micron: np.ndarray,
) -> np.ndarray:
    """Conservatively regrid archived layer extinction for Stage 7.

    Pressure remapping conserves layer optical depth assuming uniform
    ``d tau / d log(P)`` inside each archived layer.  Spectral interpolation is
    log-linear in positive extinction and uses constant endpoint extension.
    """

    source_edges = np.asarray(source_pressure_edges_bar, dtype=float)
    source_wavelength = np.asarray(source_wavelength_micron, dtype=float)
    source_tau = np.asarray(source_extinction_tau, dtype=float)
    target_edges = np.asarray(target_pressure_edges_bar, dtype=float)
    target_wavelength = np.asarray(target_wavelength_micron, dtype=float)
    if source_tau.shape != (source_edges.size - 1, source_wavelength.size):
        raise ValueError("source extinction must be layer by wavelength")
    for name, values in (
        ("source pressure", source_edges),
        ("target pressure", target_edges),
        ("source wavelength", source_wavelength),
        ("target wavelength", target_wavelength),
    ):
        if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} grid must be finite, positive, and one-dimensional")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{name} grid must be increasing")
    if np.any(~np.isfinite(source_tau)) or np.any(source_tau < 0.0):
        raise ValueError("source extinction must be finite and non-negative")

    log_source = np.log(source_edges)
    log_target = np.log(target_edges)
    remapped = np.zeros((target_edges.size - 1, source_wavelength.size))
    for source_index in range(source_edges.size - 1):
        width = log_source[source_index + 1] - log_source[source_index]
        overlap_low = np.maximum(log_target[:-1], log_source[source_index])
        overlap_high = np.minimum(log_target[1:], log_source[source_index + 1])
        fraction = np.clip(overlap_high - overlap_low, 0.0, None) / width
        remapped += fraction[:, None] * source_tau[source_index][None, :]

    tiny = np.finfo(float).tiny
    output = np.empty((remapped.shape[0], target_wavelength.size))
    log_source_wavelength = np.log(source_wavelength)
    log_target_wavelength = np.log(target_wavelength)
    for layer_index, layer_tau in enumerate(remapped):
        if not np.any(layer_tau > 0.0):
            output[layer_index] = 0.0
            continue
        output[layer_index] = np.exp(
            np.interp(
                log_target_wavelength,
                log_source_wavelength,
                np.log(np.maximum(layer_tau, tiny)),
                left=np.log(max(layer_tau[0], tiny)),
                right=np.log(max(layer_tau[-1], tiny)),
            )
        )
    return output


def stage_7_contract(
    n_cells: int,
    wavelength_micron: np.ndarray,
    *,
    archived_pressure_edges_bar: np.ndarray,
    archived_wavelength_micron: np.ndarray,
    archived_extinction_tau: np.ndarray,
    cloud_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the frozen Stage-7 absorbing-cloud case contract."""

    pressure_edges, pressure_centers = pressure_grid(n_cells)
    wavelength = np.asarray(wavelength_micron, dtype=float)
    cloud_kind = ["clear"]
    cloud_labels = ["clear"]
    optical_depth = [0.0]
    top_pressure = [np.nan]
    slope = [0.0]
    for tau_ref in STAGE_7_CLOUD_OPTICAL_DEPTHS:
        for top_bar in STAGE_7_CLOUD_TOP_PRESSURES_BAR:
            for spectral_slope in STAGE_7_EXTINCTION_SLOPES:
                cloud_kind.append("power_law_deck")
                cloud_labels.append(
                    f"deck_tau{tau_ref:g}_top{top_bar * 1e3:g}mbar_slope{spectral_slope:+g}"
                )
                optical_depth.append(tau_ref)
                top_pressure.append(top_bar)
                slope.append(spectral_slope)
    cloud_kind.append("archived_tabulated")
    cloud_labels.append("archived_virga_mie_extinction")
    optical_depth.append(np.nan)
    top_pressure.append(np.nan)
    slope.append(np.nan)

    all_cloud_indices = np.arange(len(cloud_labels), dtype=int)
    selected_cloud_indices = (
        all_cloud_indices
        if cloud_indices is None
        else np.asarray(cloud_indices, dtype=int)
    )
    if selected_cloud_indices.ndim != 1 or selected_cloud_indices.size == 0:
        raise ValueError("cloud_indices must select at least one cloud definition")
    if np.any(selected_cloud_indices < 0) or np.any(selected_cloud_indices >= len(cloud_labels)):
        raise ValueError("cloud_indices contains an invalid cloud definition")

    cloud_tau = np.zeros((len(cloud_labels), n_cells, wavelength.size))
    for cloud_index in range(1, len(cloud_labels) - 1):
        cloud_tau[cloud_index] = absorbing_cloud_tau(
            pressure_edges,
            wavelength,
            optical_depth_at_reference=optical_depth[cloud_index],
            cloud_top_pressure_bar=top_pressure[cloud_index],
            slope=slope[cloud_index],
        )
    cloud_tau[-1] = regrid_tabulated_cloud_tau(
        archived_pressure_edges_bar,
        archived_wavelength_micron,
        archived_extinction_tau,
        pressure_edges,
        wavelength,
    )

    composition, _ = background_composition(STAGE_3_MOLECULAR_VMR)
    vmr = np.asarray(
        [composition["H2"], composition["He"], *(composition[name] for name in SPECIES)]
    )
    case_profile_index = np.repeat(
        np.arange(len(STAGE_4_PROFILE_NAMES), dtype=int), selected_cloud_indices.size
    )
    case_cloud_index = np.tile(selected_cloud_indices, len(STAGE_4_PROFILE_NAMES))
    case_ids = np.asarray(
        [
            f"{STAGE_4_PROFILE_NAMES[profile_index]}__{cloud_labels[cloud_index]}__L{n_cells}"
            for profile_index, cloud_index in zip(
                case_profile_index, case_cloud_index, strict=True
            )
        ]
    )
    edge_temperature_by_profile = np.stack(
        [
            stage_4_temperature_profile(pressure_edges, profile_name)
            for profile_name in STAGE_4_PROFILE_NAMES
        ]
    )
    cell_temperature_by_profile = np.stack(
        [
            stage_4_temperature_profile(pressure_centers, profile_name)
            for profile_name in STAGE_4_PROFILE_NAMES
        ]
    )
    mu, legendre, disk = disk_quadrature()
    return {
        "schema_version": np.array(1),
        "stage": np.array(7),
        "case_id": case_ids,
        "profile_name": np.asarray(STAGE_4_PROFILE_NAMES),
        "profile_index": case_profile_index,
        "case_cloud_index": case_cloud_index,
        "cloud_label": np.asarray(cloud_labels),
        "cloud_kind": np.asarray(cloud_kind),
        "cloud_optical_depth_at_reference": np.asarray(optical_depth),
        "cloud_top_pressure_bar": np.asarray(top_pressure),
        "cloud_extinction_slope": np.asarray(slope),
        "cloud_reference_wavelength_micron": np.array(
            STAGE_7_REFERENCE_WAVELENGTH_MICRON
        ),
        "cloud_single_scattering_albedo": np.zeros(len(cloud_labels)),
        "cloud_asymmetry_factor": np.zeros(len(cloud_labels)),
        "cloud_extinction_tau": cloud_tau,
        "selected_cloud_index": selected_cloud_indices,
        "archived_pressure_edges_bar": np.asarray(archived_pressure_edges_bar),
        "archived_wavelength_micron": np.asarray(archived_wavelength_micron),
        "archived_extinction_tau": np.asarray(archived_extinction_tau),
        "wavelength_micron": wavelength,
        "pressure_edges_bar": pressure_edges,
        "pressure_centers_bar": pressure_centers,
        "prt_pressure_bar": pressure_centers,
        "temperature_edges_by_profile_k": edge_temperature_by_profile,
        "temperature_cells_by_profile_k": cell_temperature_by_profile,
        "temperature_edges_k": edge_temperature_by_profile[case_profile_index],
        "temperature_cells_k": cell_temperature_by_profile[case_profile_index],
        "gas_vmr": np.broadcast_to(vmr, (case_ids.size, vmr.size)),
        "native_include_cia": np.array(True),
        "native_return_contribution": np.array(True),
        "emission_mu": mu,
        "legendre_weights": legendre,
        "disk_weights": disk,
    }


def molecular_band_template(wavelength_micron: np.ndarray, species: str) -> np.ndarray:
    """Return a deterministic structured optical-depth template for Track A."""

    bands = {
        "H2O": (1.15, 1.4, 1.9, 2.7, 6.3),
        "CO": (2.3, 4.7),
        "CO2": (2.0, 2.7, 4.3),
        "CH4": (1.7, 2.3, 3.3, 7.7),
    }
    if species not in bands:
        raise ValueError(f"unsupported molecular template: {species}")
    wavelength = np.asarray(wavelength_micron, dtype=float)
    template = np.full_like(wavelength, 0.02)
    for index, center in enumerate(bands[species]):
        width = 0.055 + 0.012 * (index % 3)
        strength = 0.6 + 0.25 * (index % 4)
        template += strength * np.exp(-0.5 * (np.log(wavelength / center) / width) ** 2)
    return template


def bin_mean(
    wavelength_micron: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Conservatively bin one or more spectra to wavelength intervals."""

    wavelength = np.asarray(wavelength_micron, dtype=float)
    spectra = np.atleast_2d(np.asarray(values, dtype=float))
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    spectra = spectra[:, order]
    if edges[0] < wavelength[0] or edges[-1] > wavelength[-1]:
        raise ValueError("input spectrum does not cover the requested bins")
    result = np.empty((spectra.shape[0], edges.size - 1), dtype=float)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (wavelength > lower) & (wavelength < upper)
        x = np.r_[lower, wavelength[selected], upper]
        lower_index = int(np.searchsorted(wavelength, lower, side="right") - 1)
        upper_index = int(np.searchsorted(wavelength, upper, side="right") - 1)
        lower_index = min(max(lower_index, 0), wavelength.size - 2)
        upper_index = min(max(upper_index, 0), wavelength.size - 2)
        lower_fraction = (
            (lower - wavelength[lower_index])
            / (wavelength[lower_index + 1] - wavelength[lower_index])
        )
        upper_fraction = (
            (upper - wavelength[upper_index])
            / (wavelength[upper_index + 1] - wavelength[upper_index])
        )
        lower_values = spectra[:, lower_index] + lower_fraction * (
            spectra[:, lower_index + 1] - spectra[:, lower_index]
        )
        upper_values = spectra[:, upper_index] + upper_fraction * (
            spectra[:, upper_index + 1] - spectra[:, upper_index]
        )
        y = np.concatenate(
            (lower_values[:, None], spectra[:, selected], upper_values[:, None]), axis=1
        )
        result[:, index] = np.trapezoid(y, x, axis=1) / (upper - lower)
    return result[0] if np.asarray(values).ndim == 1 else result


def planck_flux_w_m2_m(
    wavelength_micron: np.ndarray, temperature_k: float
) -> np.ndarray:
    from scipy.constants import c, h, k

    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    radiance = 2.0 * h * c**2 / wavelength_m**5
    radiance /= np.expm1(h * c / (wavelength_m * k * float(temperature_k)))
    return np.pi * radiance


def eclipse_depth(flux: np.ndarray, wavelength_micron: np.ndarray) -> np.ndarray:
    stellar = planck_flux_w_m2_m(wavelength_micron, STAR_TEMPERATURE_K)
    return (
        np.asarray(flux, dtype=float) / stellar * (PLANET_RADIUS_M / STAR_RADIUS_M) ** 2
    )


def difference_metrics(
    left: np.ndarray,
    right: np.ndarray,
    wavelength_micron: np.ndarray,
) -> dict[str, float]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    denominator = np.maximum(np.abs(left_values) + np.abs(right_values), 1.0e-300)
    symmetric = 2.0 * (left_values - right_values) / denominator
    eclipse_ppm = (
        eclipse_depth(left_values, wavelength_micron)
        - eclipse_depth(right_values, wavelength_micron)
    ) * 1.0e6
    return {
        "median_symmetric_relative": float(np.median(symmetric)),
        "p95_abs_symmetric_relative": float(np.percentile(np.abs(symmetric), 95.0)),
        "max_abs_symmetric_relative": float(np.max(np.abs(symmetric))),
        "rms_eclipse_difference_ppm": float(np.sqrt(np.mean(eclipse_ppm**2))),
        "max_abs_eclipse_difference_ppm": float(np.max(np.abs(eclipse_ppm))),
    }


def pairwise_metrics(
    spectra: dict[str, np.ndarray], wavelength_micron: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        f"{left}__{right}": difference_metrics(
            spectra[left], spectra[right], wavelength_micron
        )
        for left, right in combinations(spectra, 2)
    }


def normalize_contribution(values: np.ndarray) -> np.ndarray:
    """Normalize a pressure-by-wavelength contribution array by wavelength."""

    contribution = np.asarray(values, dtype=float)
    if contribution.ndim != 2:
        raise ValueError("contribution values must be pressure by wavelength")
    if np.any(~np.isfinite(contribution)) or np.any(contribution < 0.0):
        raise ValueError("contribution values must be finite and non-negative")
    total = np.sum(contribution, axis=0, keepdims=True)
    return np.divide(
        contribution,
        total,
        out=np.zeros_like(contribution),
        where=total > 0.0,
    )


def contribution_metrics(
    left: np.ndarray,
    right: np.ndarray,
    pressure_bar: np.ndarray,
) -> dict[str, float]:
    """Compare normalized contribution functions on a shared pressure grid."""

    left_values = normalize_contribution(left)
    right_values = normalize_contribution(right)
    pressure = np.asarray(pressure_bar, dtype=float)
    if left_values.shape != right_values.shape:
        raise ValueError("contribution arrays must have matching shapes")
    if pressure.shape != (left_values.shape[0],) or np.any(pressure <= 0.0):
        raise ValueError("pressure_bar must match the contribution pressure axis")
    log_pressure = np.log10(pressure)[:, None]
    left_centroid = np.sum(left_values * log_pressure, axis=0)
    right_centroid = np.sum(right_values * log_pressure, axis=0)
    centroid_difference = left_centroid - right_centroid
    left_peak = pressure[np.argmax(left_values, axis=0)]
    right_peak = pressure[np.argmax(right_values, axis=0)]
    peak_difference = np.log10(left_peak / right_peak)
    total_variation = 0.5 * np.sum(np.abs(left_values - right_values), axis=0)
    return {
        "centroid_pressure_rms_difference_dex": float(
            np.sqrt(np.mean(centroid_difference**2))
        ),
        "centroid_pressure_median_difference_dex": float(
            np.median(centroid_difference)
        ),
        "centroid_pressure_p95_abs_difference_dex": float(
            np.percentile(np.abs(centroid_difference), 95.0)
        ),
        "peak_pressure_rms_difference_dex": float(np.sqrt(np.mean(peak_difference**2))),
        "peak_pressure_median_difference_dex": float(np.median(peak_difference)),
        "profile_total_variation_median": float(np.median(total_variation)),
        "profile_total_variation_p95": float(np.percentile(total_variation, 95.0)),
    }


def pairwise_contribution_metrics(
    contributions: dict[str, np.ndarray], pressure_bar: np.ndarray
) -> dict[str, dict[str, float]]:
    """Return contribution metrics for every pair of named models."""

    return {
        f"{left}__{right}": contribution_metrics(
            contributions[left], contributions[right], pressure_bar
        )
        for left, right in combinations(contributions, 2)
    }


def normalize_temperature_response(values: np.ndarray) -> np.ndarray:
    """Normalize absolute temperature Jacobians over perturbation centres."""

    response = np.abs(np.asarray(values, dtype=float))
    if response.ndim != 2:
        raise ValueError("temperature response must be centre by wavelength")
    if np.any(~np.isfinite(response)):
        raise ValueError("temperature response must be finite")
    total = np.sum(response, axis=0, keepdims=True)
    return np.divide(
        response,
        total,
        out=np.zeros_like(response),
        where=total > 0.0,
    )


def normalize_composition_response(values: np.ndarray) -> np.ndarray:
    """Normalize absolute species-by-pressure-by-wavelength Jacobians in pressure."""

    response = np.abs(np.asarray(values, dtype=float))
    if response.ndim != 3:
        raise ValueError(
            "composition response must be species by pressure by wavelength"
        )
    if np.any(~np.isfinite(response)):
        raise ValueError("composition response must be finite")
    total = np.sum(response, axis=1, keepdims=True)
    return np.divide(response, total, out=np.zeros_like(response), where=total > 0.0)


def cross_species_sensitivity_fractions(values: np.ndarray) -> np.ndarray:
    """Normalize pressure-integrated absolute sensitivity over four species."""

    jacobian = np.asarray(values, dtype=float)
    if jacobian.ndim != 3 or jacobian.shape[0] != len(SPECIES):
        raise ValueError("values must be four species by pressure by wavelength")
    if np.any(~np.isfinite(jacobian)):
        raise ValueError("values must be finite")
    sensitivity = np.sum(np.abs(jacobian), axis=1)
    total = np.sum(sensitivity, axis=0, keepdims=True)
    return np.divide(
        sensitivity, total, out=np.zeros_like(sensitivity), where=total > 0.0
    )


def apply_jacobian_roundoff_floor(
    values: np.ndarray,
    flux_scale: np.ndarray,
    amplitude: float,
    *,
    epsilon_factor: float = 32.0,
) -> np.ndarray:
    """Set finite-difference values indistinguishable from roundoff to zero.

    The floor is ``epsilon_factor * eps * flux_scale / amplitude``.  This is
    especially important for analytically zero composition derivatives of an
    isothermal atmosphere with a blackbody lower boundary: normalizing raw
    cancellation noise would otherwise create arbitrary vertical responses.
    """

    jacobian = np.asarray(values, dtype=float)
    scale = np.asarray(flux_scale, dtype=float)
    step = float(amplitude)
    factor = float(epsilon_factor)
    if scale.shape != (jacobian.shape[-1],):
        raise ValueError("flux_scale must match the final wavelength axis")
    if np.any(~np.isfinite(jacobian)) or np.any(~np.isfinite(scale)):
        raise ValueError("Jacobian and flux scale must be finite")
    if np.any(scale < 0.0) or not np.isfinite(step) or step <= 0.0:
        raise ValueError("flux scale must be non-negative and amplitude positive")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("epsilon_factor must be finite and positive")
    floor = factor * np.finfo(float).eps * scale / step
    return np.where(np.abs(jacobian) <= floor, 0.0, jacobian)


def eclipse_jacobian_ppm_per_k(
    flux_jacobian: np.ndarray, wavelength_micron: np.ndarray
) -> np.ndarray:
    """Convert a planet-flux temperature Jacobian to eclipse ppm/K."""

    return eclipse_depth(flux_jacobian, wavelength_micron) * 1.0e6


def eclipse_jacobian_ppm_per_dex(
    flux_jacobian: np.ndarray, wavelength_micron: np.ndarray
) -> np.ndarray:
    """Convert a planet-flux log10-VMR Jacobian to eclipse ppm/dex."""

    return eclipse_depth(flux_jacobian, wavelength_micron) * 1.0e6


def signed_jacobian_metrics(
    left: np.ndarray,
    right: np.ndarray,
    wavelength_micron: np.ndarray,
) -> dict[str, float]:
    """Compare signed composition Jacobians without singular pointwise ratios."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape or left_values.ndim < 2:
        raise ValueError("Jacobian tensors must have matching shapes")
    wavelength = np.asarray(wavelength_micron, dtype=float)
    if wavelength.shape != (left_values.shape[-1],):
        raise ValueError("wavelength_micron must match the final Jacobian axis")
    if np.any(~np.isfinite(left_values)) or np.any(~np.isfinite(right_values)):
        raise ValueError("Jacobian tensors must be finite")
    flattened_left = left_values.reshape(-1, left_values.shape[-1])
    flattened_right = right_values.reshape(-1, right_values.shape[-1])
    peak = np.maximum(
        np.max(np.abs(flattened_left), axis=1),
        np.max(np.abs(flattened_right), axis=1),
    )[:, None]
    scaled = np.divide(
        np.abs(flattened_left - flattened_right),
        peak,
        out=np.zeros_like(flattened_left),
        where=peak > 0.0,
    )
    difference = left_values - right_values
    eclipse_difference = eclipse_jacobian_ppm_per_dex(difference, wavelength)
    pair_rms = max(
        float(np.sqrt(np.mean(left_values**2))),
        float(np.sqrt(np.mean(right_values**2))),
        np.finfo(float).tiny,
    )
    return {
        "median_abs_difference_over_pair_peak": float(np.median(scaled)),
        "p95_abs_difference_over_pair_peak": float(np.percentile(scaled, 95.0)),
        "max_abs_difference_over_pair_peak": float(np.max(scaled)),
        "relative_rms_difference": float(np.sqrt(np.mean(difference**2)) / pair_rms),
        "rms_eclipse_jacobian_difference_ppm_per_dex": float(
            np.sqrt(np.mean(eclipse_difference**2))
        ),
        "max_abs_eclipse_jacobian_difference_ppm_per_dex": float(
            np.max(np.abs(eclipse_difference))
        ),
    }


def amplitude_linearity_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    wavelength_micron: np.ndarray,
) -> dict[str, float]:
    """Report central-difference truncation/nonlinearity against a reference step."""

    metrics = signed_jacobian_metrics(reference, candidate, wavelength_micron)
    difference = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    metrics["rms_truncation_difference_flux_per_dex"] = float(
        np.sqrt(np.mean(difference**2))
    )
    metrics["max_abs_truncation_difference_flux_per_dex"] = float(
        np.max(np.abs(difference))
    )
    return metrics


def cross_species_fraction_metrics(
    left: np.ndarray, right: np.ndarray
) -> dict[str, float]:
    """Compare species-by-wavelength sensitivity fractions."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape or left_values.ndim != 2:
        raise ValueError("fraction arrays must be matching species by wavelength")
    if np.any(~np.isfinite(left_values)) or np.any(~np.isfinite(right_values)):
        raise ValueError("fraction arrays must be finite")
    difference = left_values - right_values
    total_variation = 0.5 * np.sum(np.abs(difference), axis=0)
    return {
        "fraction_rms_difference": float(np.sqrt(np.mean(difference**2))),
        "fraction_max_abs_difference": float(np.max(np.abs(difference))),
        "species_total_variation_median": float(np.median(total_variation)),
        "species_total_variation_p95": float(np.percentile(total_variation, 95.0)),
    }


def temperature_jacobian_metrics(
    left: np.ndarray,
    right: np.ndarray,
    wavelength_micron: np.ndarray,
) -> dict[str, float]:
    """Compare centre-by-wavelength temperature Jacobians.

    Absolute spectral differences are scaled separately for each perturbation
    centre by the larger peak absolute Jacobian of the pair.  This avoids
    singular relative errors at physical sign changes while retaining a
    dimensionless, pressure-local comparison.
    """

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape or left_values.ndim != 2:
        raise ValueError(
            "Jacobian arrays must have matching centre-by-wavelength shapes"
        )
    wavelength = np.asarray(wavelength_micron, dtype=float)
    if wavelength.shape != (left_values.shape[1],):
        raise ValueError("wavelength_micron must match the Jacobian spectral axis")
    if np.any(~np.isfinite(left_values)) or np.any(~np.isfinite(right_values)):
        raise ValueError("Jacobian arrays must be finite")
    peak = np.maximum(
        np.max(np.abs(left_values), axis=1),
        np.max(np.abs(right_values), axis=1),
    )[:, None]
    scaled = np.divide(
        np.abs(left_values - right_values),
        peak,
        out=np.zeros_like(left_values),
        where=peak > 0.0,
    )
    eclipse_difference = eclipse_jacobian_ppm_per_k(
        left_values - right_values, wavelength
    )
    left_rms = float(np.sqrt(np.mean(left_values**2)))
    right_rms = float(np.sqrt(np.mean(right_values**2)))
    pair_rms = max(left_rms, right_rms, np.finfo(float).tiny)
    return {
        "median_abs_difference_over_pair_peak": float(np.median(scaled)),
        "p95_abs_difference_over_pair_peak": float(np.percentile(scaled, 95.0)),
        "max_abs_difference_over_pair_peak": float(np.max(scaled)),
        "relative_rms_difference": float(
            np.sqrt(np.mean((left_values - right_values) ** 2)) / pair_rms
        ),
        "rms_eclipse_jacobian_difference_ppm_per_k": float(
            np.sqrt(np.mean(eclipse_difference**2))
        ),
        "max_abs_eclipse_jacobian_difference_ppm_per_k": float(
            np.max(np.abs(eclipse_difference))
        ),
    }


def pairwise_temperature_jacobian_metrics(
    jacobians: dict[str, np.ndarray], wavelength_micron: np.ndarray
) -> dict[str, dict[str, float]]:
    """Return temperature-Jacobian metrics for every pair of named models."""

    return {
        f"{left}__{right}": temperature_jacobian_metrics(
            jacobians[left], jacobians[right], wavelength_micron
        )
        for left, right in combinations(jacobians, 2)
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_checksums(directory: Path) -> None:
    checksums = {
        path.name: sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    write_json(directory / "checksums.json", checksums)
