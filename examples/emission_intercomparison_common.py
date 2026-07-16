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
    mean_molar_mass = sum(
        composition[name] * MOLAR_MASS[name] for name in composition
    )
    return composition, float(mean_molar_mass)


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
                total_tau
                * pressure_fraction[:, None]
                * np.ones((1, wavelength.size))
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
                case_ids.append(
                    f"{species}_T{int(anchor)}_vmr{vmr:.0e}_L{n_layers}"
                )
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


def stage_3_contract(n_layers: int, *, include_cia: bool = True) -> dict[str, np.ndarray]:
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


def molecular_band_template(
    wavelength_micron: np.ndarray, species: str
) -> np.ndarray:
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
        template += strength * np.exp(
            -0.5 * (np.log(wavelength / center) / width) ** 2
        )
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
        for row, spectrum in enumerate(spectra):
            y = np.r_[
                np.interp(lower, wavelength, spectrum),
                spectrum[selected],
                np.interp(upper, wavelength, spectrum),
            ]
            result[row, index] = np.trapezoid(y, x) / (upper - lower)
    return result[0] if np.asarray(values).ndim == 1 else result


def planck_flux_w_m2_m(wavelength_micron: np.ndarray, temperature_k: float) -> np.ndarray:
    from scipy.constants import c, h, k

    wavelength_m = np.asarray(wavelength_micron, dtype=float) * 1.0e-6
    radiance = 2.0 * h * c**2 / wavelength_m**5
    radiance /= np.expm1(h * c / (wavelength_m * k * float(temperature_k)))
    return np.pi * radiance


def eclipse_depth(flux: np.ndarray, wavelength_micron: np.ndarray) -> np.ndarray:
    stellar = planck_flux_w_m2_m(wavelength_micron, STAR_TEMPERATURE_K)
    return (
        np.asarray(flux, dtype=float)
        / stellar
        * (PLANET_RADIUS_M / STAR_RADIUS_M) ** 2
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
