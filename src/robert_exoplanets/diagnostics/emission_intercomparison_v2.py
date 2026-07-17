"""Frozen common contract for emission intercomparison Version 2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.atmosphere import ParmentierGuillot2014TemperatureProfile
from robert_exoplanets.core import PressureGrid, RobertValidationError


CONTRACT_SCHEMA_VERSION = "2.0.0"
CONTRACT_NAME = "wasp17b_emission_intercomparison_v2"
PRESSURE_CELLS = (40, 80, 160)
PRIMARY_PRESSURE_CELLS = 80
PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
MICRON_TO_METER = 1.0e-6
BAR_TO_PA = 1.0e5
DAY_TO_S = 86_400.0


def _readonly(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise RobertValidationError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _frozen_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    converted = {str(key): float(value) for key, value in values.items()}
    if any(not np.isfinite(value) for value in converted.values()):
        raise RobertValidationError("contract mappings must contain finite values")
    return MappingProxyType(converted)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 for a JSON-compatible payload."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceMeasurement:
    """One adopted source value and its frozen SI conversion."""

    value: float
    unit: str
    si_value: float
    si_unit: str
    uncertainty_plus: float
    uncertainty_minus: float
    provenance: str

    def __post_init__(self) -> None:
        numbers = (
            self.value,
            self.si_value,
            self.uncertainty_plus,
            self.uncertainty_minus,
        )
        if any(not np.isfinite(float(value)) for value in numbers):
            raise RobertValidationError("source measurements must be finite")
        if self.uncertainty_plus < 0.0 or self.uncertainty_minus < 0.0:
            raise RobertValidationError("source uncertainties must be non-negative")
        if not self.unit or not self.si_unit or not self.provenance:
            raise RobertValidationError("source measurement labels must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "si_value": self.si_value,
            "si_unit": self.si_unit,
            "uncertainty_plus": self.uncertainty_plus,
            "uncertainty_minus": self.uncertainty_minus,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceMeasurement":
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PressureGridContract:
    """One explicit cell/level/node mapping for the shared atmosphere."""

    n_cells: int
    edges_bar: NDArray[np.float64]
    centers_bar: NDArray[np.float64]
    orientation: str = "top_to_bottom_increasing_pressure"
    robert_mapping: str = "cells_with_geometric_centers_and_explicit_edges"
    picaso_mapping: str = "matching_cell_edges_as_pressure_levels"
    petitradtrans_mapping: str = "robert_geometric_cell_centers_as_pressure_nodes"

    def __post_init__(self) -> None:
        edges = _readonly(self.edges_bar, "pressure edges")
        centers = _readonly(self.centers_bar, "pressure centers")
        if self.n_cells not in PRESSURE_CELLS:
            raise RobertValidationError(f"n_cells must be one of {PRESSURE_CELLS}")
        if edges.size != self.n_cells + 1 or centers.size != self.n_cells:
            raise RobertValidationError("pressure grid shapes do not match n_cells")
        if edges[0] != 1.0e-5 or edges[-1] != 100.0:
            raise RobertValidationError("pressure edges must span exactly 1e-5--100 bar")
        if not np.all(np.diff(edges) > 0.0):
            raise RobertValidationError("pressure must increase from top to bottom")
        np.testing.assert_allclose(centers, np.sqrt(edges[:-1] * edges[1:]), rtol=1e-14)
        object.__setattr__(self, "edges_bar", edges)
        object.__setattr__(self, "centers_bar", centers)

    @property
    def picaso_levels_bar(self) -> NDArray[np.float64]:
        return self.edges_bar

    @property
    def petitradtrans_nodes_bar(self) -> NDArray[np.float64]:
        return self.centers_bar

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "edges_bar": self.edges_bar.tolist(),
            "centers_bar": self.centers_bar.tolist(),
            "orientation": self.orientation,
            "edge_semantics": "cell_boundaries",
            "center_semantics": "geometric_cell_centers",
            "robert_mapping": self.robert_mapping,
            "picaso_mapping": self.picaso_mapping,
            "petitradtrans_mapping": self.petitradtrans_mapping,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PressureGridContract":
        return cls(
            n_cells=int(payload["n_cells"]),
            edges_bar=payload["edges_bar"],
            centers_bar=payload["centers_bar"],
            orientation=str(payload["orientation"]),
            robert_mapping=str(payload["robert_mapping"]),
            picaso_mapping=str(payload["picaso_mapping"]),
            petitradtrans_mapping=str(payload["petitradtrans_mapping"]),
        )


@dataclass(frozen=True)
class PG14Parameters:
    """Frozen parameters for one canonical PG14 profile."""

    internal_temperature_k: float
    kappa_ir_m2_kg: float
    gamma1: float
    gamma2: float
    alpha: float
    irradiation_temperature_k: float

    def to_dict(self) -> dict[str, float]:
        return {
            "internal_temperature_k": self.internal_temperature_k,
            "kappa_ir_m2_kg": self.kappa_ir_m2_kg,
            "gamma1": self.gamma1,
            "gamma2": self.gamma2,
            "alpha": self.alpha,
            "irradiation_temperature_k": self.irradiation_temperature_k,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PG14Parameters":
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class OpacityAsset:
    """Versioned external opacity asset required by a later native path."""

    filename: str
    species: str
    sha256: str
    source_doi: str

    def __post_init__(self) -> None:
        if not self.filename or not self.species or not self.source_doi:
            raise RobertValidationError("opacity asset labels must not be empty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise RobertValidationError("opacity asset SHA-256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "species": self.species,
            "sha256": self.sha256,
            "source_doi": self.source_doi,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OpacityAsset":
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SpectralContract:
    """Shared comparison grid and retention/representation requirements."""

    r100_edges_micron: NDArray[np.float64]
    r100_centers_micron: NDArray[np.float64]
    native_reference_wavelength_micron: NDArray[np.float64]
    resolving_power: float = 100.0
    wavelength_min_micron: float = 0.8
    wavelength_max_micron: float = 12.0

    def __post_init__(self) -> None:
        edges = _readonly(self.r100_edges_micron, "R=100 edges")
        centers = _readonly(self.r100_centers_micron, "R=100 centers")
        native = _readonly(
            self.native_reference_wavelength_micron, "native reference wavelength"
        )
        if edges.size != centers.size + 1:
            raise RobertValidationError("R=100 edges and centers have inconsistent shapes")
        if edges[0] != self.wavelength_min_micron or edges[-1] != self.wavelength_max_micron:
            raise RobertValidationError("R=100 edges must cover exactly 0.8--12 micron")
        if not np.all(np.diff(edges) > 0.0) or not np.all(np.diff(native) > 0.0):
            raise RobertValidationError("wavelength arrays must be strictly increasing")
        np.testing.assert_allclose(centers, np.sqrt(edges[:-1] * edges[1:]), rtol=1e-14)
        if native[0] != edges[0] or native[-1] != edges[-1]:
            raise RobertValidationError("native reference must cover the complete domain")
        object.__setattr__(self, "r100_edges_micron", edges)
        object.__setattr__(self, "r100_centers_micron", centers)
        object.__setattr__(self, "native_reference_wavelength_micron", native)

    def to_dict(self) -> dict[str, Any]:
        log_width = float(np.log(self.r100_edges_micron[1] / self.r100_edges_micron[0]))
        return {
            "wavelength_min_micron": self.wavelength_min_micron,
            "wavelength_max_micron": self.wavelength_max_micron,
            "wavelength_convention": "vacuum_wavelength_increasing",
            "resolving_power": self.resolving_power,
            "grid_definition": "ceil(R*ln(lambda_max/lambda_min)) equal-log bins with exact endpoints",
            "effective_log_resolving_power": 1.0 / log_width,
            "r100_edges_micron": self.r100_edges_micron.tolist(),
            "r100_centers_micron": self.r100_centers_micron.tolist(),
            "native_reference_wavelength_micron": self.native_reference_wavelength_micron.tolist(),
            "binning": "piecewise-linear flux-conserving wavelength integration",
            "native_spectrum_retention_required": True,
            "native_vertical_array_retention_required": True,
            "picaso_representations": {
                "opacity_sampling": "picaso_opacity_sampling_native_and_flux_conserving_r100",
                "correlated_k": "picaso_correlated_k_native_and_flux_conserving_r100",
                "interchangeable": False,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpectralContract":
        return cls(
            r100_edges_micron=payload["r100_edges_micron"],
            r100_centers_micron=payload["r100_centers_micron"],
            native_reference_wavelength_micron=payload[
                "native_reference_wavelength_micron"
            ],
            resolving_power=float(payload["resolving_power"]),
            wavelength_min_micron=float(payload["wavelength_min_micron"]),
            wavelength_max_micron=float(payload["wavelength_max_micron"]),
        )


@dataclass(frozen=True)
class Version2CommonContract:
    """Typed, immutable, serializable Version-2 scientific contract."""

    measurements: Mapping[str, SourceMeasurement]
    constants: Mapping[str, float]
    derived: Mapping[str, float]
    pressure_grids: tuple[PressureGridContract, ...]
    pg14_parameters: Mapping[str, PG14Parameters]
    temperature_profiles_k: Mapping[str, NDArray[np.float64]]
    composition_vmr: Mapping[str, float]
    molecular_masses_u: Mapping[str, float]
    picaso_correlated_k_assets: Mapping[str, OpacityAsset]
    spectral: SpectralContract
    stellar_surface_flux_native_w_m2_m: NDArray[np.float64]
    stellar_surface_flux_r100_w_m2_m: NDArray[np.float64]
    pg14_implementation_sha256: str
    pg14_method_sha256: str
    schema_version: str = CONTRACT_SCHEMA_VERSION
    contract_name: str = CONTRACT_NAME
    composition_provenance: str = (
        "frozen FastChem/Asplund-2009 [M/H]=0, C/O=0.55, T=1500 K, P=0.1 bar"
    )
    background_fill_rule: str = "active molecules fixed; remainder H2:He=0.8547:0.1453"
    isothermal_temperature_k: float = 1755.0
    random_seed_policy: str = "deterministic_no_random_numbers"

    def __post_init__(self) -> None:
        measurements = MappingProxyType(dict(self.measurements))
        constants = _frozen_mapping(self.constants)
        derived = _frozen_mapping(self.derived)
        parameters = MappingProxyType(dict(self.pg14_parameters))
        profiles: dict[str, NDArray[np.float64]] = {}
        for name, values in self.temperature_profiles_k.items():
            profiles[str(name)] = _readonly(values, f"temperature profile {name}")
        composition = _frozen_mapping(self.composition_vmr)
        masses = _frozen_mapping(self.molecular_masses_u)
        opacity_assets = MappingProxyType(dict(self.picaso_correlated_k_assets))
        stellar_native = _readonly(
            self.stellar_surface_flux_native_w_m2_m, "native stellar surface flux"
        )
        stellar_r100 = _readonly(
            self.stellar_surface_flux_r100_w_m2_m, "R=100 stellar surface flux"
        )
        if tuple(grid.n_cells for grid in self.pressure_grids) != PRESSURE_CELLS:
            raise RobertValidationError("pressure grid ladder must be exactly 40/80/160")
        if not np.isclose(sum(composition.values()), 1.0, rtol=0.0, atol=2e-16):
            raise RobertValidationError("frozen composition must sum to one")
        if stellar_native.shape != self.spectral.native_reference_wavelength_micron.shape:
            raise RobertValidationError("native stellar spectrum shape mismatch")
        if stellar_r100.shape != self.spectral.r100_centers_micron.shape:
            raise RobertValidationError("R=100 stellar spectrum shape mismatch")
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "constants", constants)
        object.__setattr__(self, "derived", derived)
        object.__setattr__(self, "pg14_parameters", parameters)
        object.__setattr__(self, "temperature_profiles_k", MappingProxyType(profiles))
        object.__setattr__(self, "composition_vmr", composition)
        object.__setattr__(self, "molecular_masses_u", masses)
        object.__setattr__(self, "picaso_correlated_k_assets", opacity_assets)
        object.__setattr__(self, "stellar_surface_flux_native_w_m2_m", stellar_native)
        object.__setattr__(self, "stellar_surface_flux_r100_w_m2_m", stellar_r100)

    @property
    def mean_molecular_weight_u(self) -> float:
        return float(
            sum(
                self.composition_vmr[name] * self.molecular_masses_u[name]
                for name in self.composition_vmr
            )
        )

    @property
    def primary_pressure_grid(self) -> PressureGridContract:
        return next(
            grid for grid in self.pressure_grids if grid.n_cells == PRIMARY_PRESSURE_CELLS
        )

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "contract_name": self.contract_name,
            "measurements": {
                name: measurement.to_dict()
                for name, measurement in self.measurements.items()
            },
            "physical_constants": dict(self.constants),
            "derived_quantities": dict(self.derived),
            "stellar_blackbody": {
                "temperature_k": self.measurements["stellar_effective_temperature"].si_value,
                "law": "Planck B_lambda with numpy.expm1; F_lambda=pi*B_lambda",
                "radiance_unit": "W m^-3 sr^-1",
                "surface_flux_unit": "W m^-3",
                "native_surface_flux_w_m2_m": self.stellar_surface_flux_native_w_m2_m.tolist(),
                "r100_surface_flux_w_m2_m": self.stellar_surface_flux_r100_w_m2_m.tolist(),
            },
            "pressure_grids": [grid.to_dict() for grid in self.pressure_grids],
            "temperature_contract": {
                "isothermal_temperature_k": self.isothermal_temperature_k,
                "pg14_parameters": {
                    name: parameters.to_dict()
                    for name, parameters in self.pg14_parameters.items()
                },
                "evaluated_profiles_k": {
                    name: profile.tolist()
                    for name, profile in self.temperature_profiles_k.items()
                },
                "profile_key_format": "{profile_name}_{n_cells}_cells",
                "canonical_contract_grid_cells": PRIMARY_PRESSURE_CELLS,
                "optical_depth_coordinate": "tau=kappa_IR*pressure(Pa)/gravity(m s^-2)",
                "exponential_integral_method": "fixed 128-point Gauss-Legendre E2 quadrature on [0,1]",
                "implementation_sha256": self.pg14_implementation_sha256,
                "method_sha256": self.pg14_method_sha256,
            },
            "composition": {
                "vmr": dict(self.composition_vmr),
                "sum": float(sum(self.composition_vmr.values())),
                "background_fill_rule": self.background_fill_rule,
                "provenance_state": "frozen_constants_not_runtime_chemistry",
                "provenance": self.composition_provenance,
                "molecular_masses_u": dict(self.molecular_masses_u),
                "mean_molecular_weight_u_computed": self.mean_molecular_weight_u,
                "mean_molecular_weight_u_declared": 2.321438174776293,
            },
            "spectral_contract": self.spectral.to_dict(),
            "opacity_contract": {
                "picaso_primary_molecular_representation_from_stage_2": "correlated_k_resort_rebin",
                "picaso_opacity_sampling_role": "secondary_representation_diagnostic_only",
                "picaso_correlated_k_assets": {
                    name: asset.to_dict()
                    for name, asset in self.picaso_correlated_k_assets.items()
                },
                "asset_family": {
                    "format": "official PICASO 4.0 resort-rebin per-molecule HDF5",
                    "zenodo_doi": "10.5281/zenodo.18644980",
                    "zenodo_version": "v2",
                    "local_data_location": "/Users/jaketaylor/Dropbox/picaso/reference/opacities/resortrebin",
                    "spectral_bins": 661,
                    "double_gauss_points": 8,
                    "k_array_shape_pt": [20, 73],
                    "wavelength_range_micron": [0.267868, 267.559],
                    "pressure_range_bar": [1.0e-6, 3000.0],
                    "temperature_range_k": [75.0, 4000.0],
                    "covers_version_2_domain": True,
                    "verification_state": "parent_task_reported_SHA256_and_official_MD5_verified_exactly",
                },
                "version_2_picaso_environment": {
                    "interpreter": "/opt/miniconda3/envs/picaso-v4/bin/python",
                    "reference_path": "/Users/jaketaylor/Dropbox/picaso-v4/reference",
                    "required_worker_environment": {
                        "picaso_refdata": "/Users/jaketaylor/Dropbox/picaso-v4/reference",
                        "NUMBA_CACHE_DIR": "<writable task temp>/picaso-v4-numba-cache",
                        "MPLCONFIGDIR": "<writable task temp>/picaso-v4-matplotlib",
                        "set_before_import": True,
                    },
                    "versions": {
                        "python": "3.11.15",
                        "picaso": "4.0",
                        "numpy": "2.4.6",
                        "scipy": "1.17.1",
                        "h5py": "3.16.0",
                        "numba": "0.66.0",
                        "virga-exo": "2.0.1",
                        "astropy": "8.0.1",
                        "pandas": "3.0.3",
                    },
                    "reference": {
                        "version": "4.0",
                        "official_git_commit": "0369089372f748609dd0233e6de9361af31a38cf",
                        "sha256": {
                            "version.md": "a54cbe0a23a0bb5e4ea4b24ae830855f150736963aa087084d1f1a6a1dd123c4",
                            "config.json": "2cede6701e46267c8ebbfb58c02a344c9f8dfb08ed9639e9cf9276ead74cbde4",
                            "ck_cx_cont_opacities.db": "756f6d370aaef8242ec7a9988c4570730098d5cceb5b6912cc6b2abbdbdd0386",
                        },
                    },
                    "correlated_k_smoke": {
                        "status": "pass",
                        "species_loaded_exactly": ["H2O", "CO", "CO2", "CH4"],
                        "spectral_bins": 661,
                        "k_points": 8,
                        "layers": 40,
                        "taugas_shape": [40, 661, 8],
                        "finite_positive_bins_in_0p8_to_12_micron": 583,
                        "scattering": False,
                        "rayleigh": False,
                        "delta_eddington": False,
                    },
                    "known_harmless_warning": "optional Vega spectrum absent; Version 2 uses an explicit blackbody star",
                },
                "historical_version_1_picaso_environment": {
                    "interpreter": "/opt/miniconda3/envs/picaso/bin/python",
                    "python": "3.10.20",
                    "picaso": "3.2.2",
                    "version_2_molecular_use_forbidden": True,
                },
            },
            "eclipse_convention": {
                "definition": "D_lambda=(F_planet_lambda/F_star_lambda)*(R_planet/R_star)^2",
                "signed_surface_flux": "positive_outward",
                "distance_cancels": True,
            },
            "random_seed_policy": self.random_seed_policy,
        }
        if include_checksum:
            payload["contract_sha256"] = payload_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Version2CommonContract":
        supplied_checksum = payload.get("contract_sha256")
        without_checksum = dict(payload)
        without_checksum.pop("contract_sha256", None)
        if supplied_checksum is not None and supplied_checksum != payload_sha256(without_checksum):
            raise RobertValidationError("common-contract checksum mismatch")
        temperature = payload["temperature_contract"]
        composition = payload["composition"]
        stellar = payload["stellar_blackbody"]
        return cls(
            measurements={
                name: SourceMeasurement.from_dict(value)
                for name, value in payload["measurements"].items()
            },
            constants=payload["physical_constants"],
            derived=payload["derived_quantities"],
            pressure_grids=tuple(
                PressureGridContract.from_dict(item) for item in payload["pressure_grids"]
            ),
            pg14_parameters={
                name: PG14Parameters.from_dict(value)
                for name, value in temperature["pg14_parameters"].items()
            },
            temperature_profiles_k=temperature["evaluated_profiles_k"],
            composition_vmr=composition["vmr"],
            molecular_masses_u=composition["molecular_masses_u"],
            picaso_correlated_k_assets={
                name: OpacityAsset.from_dict(value)
                for name, value in payload["opacity_contract"][
                    "picaso_correlated_k_assets"
                ].items()
            },
            spectral=SpectralContract.from_dict(payload["spectral_contract"]),
            stellar_surface_flux_native_w_m2_m=stellar[
                "native_surface_flux_w_m2_m"
            ],
            stellar_surface_flux_r100_w_m2_m=stellar["r100_surface_flux_w_m2_m"],
            pg14_implementation_sha256=temperature["implementation_sha256"],
            pg14_method_sha256=temperature["method_sha256"],
            schema_version=str(payload["schema_version"]),
            contract_name=str(payload["contract_name"]),
            composition_provenance=str(composition["provenance"]),
            background_fill_rule=str(composition["background_fill_rule"]),
            isothermal_temperature_k=float(temperature["isothermal_temperature_k"]),
            random_seed_policy=str(payload["random_seed_policy"]),
        )


def planck_surface_flux_w_m2_m(
    wavelength_micron: ArrayLike, temperature_k: float
) -> NDArray[np.float64]:
    """Return exact blackbody hemispheric surface flux density per metre."""

    wavelength = _readonly(wavelength_micron, "wavelength_micron")
    temperature = float(temperature_k)
    if temperature <= 0.0 or not np.isfinite(temperature) or np.any(wavelength <= 0.0):
        raise RobertValidationError("wavelength and temperature must be positive")
    wavelength_m = wavelength * MICRON_TO_METER
    exponent = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / (
        wavelength_m * BOLTZMANN_CONSTANT_J_K * temperature
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        denominator = np.expm1(exponent)
        flux = (
            np.pi
            * 2.0
            * PLANCK_CONSTANT_J_S
            * SPEED_OF_LIGHT_M_S**2
            / (wavelength_m**5 * denominator)
        )
    if np.any(np.isnan(flux)) or np.any(flux < 0.0):
        raise RobertValidationError("Planck evaluation produced invalid values")
    flux.setflags(write=False)
    return flux


def flux_conserving_bin_mean(
    wavelength_micron: ArrayLike, values: ArrayLike, edges_micron: ArrayLike
) -> NDArray[np.float64]:
    """Bin spectral density with piecewise-linear, flux-conserving integration."""

    wavelength = np.asarray(wavelength_micron, dtype=float)
    spectra = np.asarray(values, dtype=float)
    edges = np.asarray(edges_micron, dtype=float)
    if wavelength.ndim != 1 or edges.ndim != 1 or spectra.shape[-1] != wavelength.size:
        raise RobertValidationError("spectral binning axes are inconsistent")
    if np.any(np.diff(wavelength) <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise RobertValidationError("spectral binning coordinates must increase")
    if edges[0] < wavelength[0] or edges[-1] > wavelength[-1]:
        raise RobertValidationError("input spectrum does not cover requested edges")
    flat = spectra.reshape((-1, wavelength.size))
    output = np.empty((flat.shape[0], edges.size - 1), dtype=float)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (wavelength > lower) & (wavelength < upper)
        x = np.concatenate(([lower], wavelength[selected], [upper]))
        y = np.vstack([np.interp(x, wavelength, row) for row in flat])
        output[:, index] = np.trapezoid(y, x, axis=1) / (upper - lower)
    result = output.reshape((*spectra.shape[:-1], output.shape[-1]))
    result.setflags(write=False)
    return result


def _r100_grid() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_bins = int(np.ceil(100.0 * np.log(12.0 / 0.8)))
    edges = np.geomspace(0.8, 12.0, n_bins + 1)
    edges[0] = 0.8
    edges[-1] = 12.0
    return edges, np.sqrt(edges[:-1] * edges[1:])


def _pg14_source_checksum() -> str:
    source = Path(__file__).resolve().parents[1] / "atmosphere" / "temperature.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _pg14_method_checksum() -> str:
    descriptor = {
        "method": "ROBERT ParmentierGuillot2014TemperatureProfile",
        "tau": "kappa_IR_m2_kg*pressure_Pa/gravity_m_s2",
        "e2": "128-point Gauss-Legendre quadrature integral_0^1 exp(-x/t) dt",
        "profile_version": "v2-pg14-contract-1",
    }
    return payload_sha256(descriptor)


def build_version_2_common_contract() -> Version2CommonContract:
    """Build and validate the single frozen Version-2 common contract."""

    constants = {
        "G_m3_kg_s2": 6.67430e-11,
        "M_J_kg": 1.898e27,
        "R_J_m": 7.1492e7,
        "M_sun_kg": 1.98847e30,
        "R_sun_m": 6.957e8,
        "AU_m": 1.495978707e11,
        "pc_m": 3.085677581491367e16,
        "h_J_s": PLANCK_CONSTANT_J_S,
        "c_m_s": SPEED_OF_LIGHT_M_S,
        "k_B_J_K": BOLTZMANN_CONSTANT_J_K,
        "bar_Pa": BAR_TO_PA,
        "day_s": DAY_TO_S,
    }
    measurements = {
        "planet_mass": SourceMeasurement(0.477, "M_J", 9.05346e26, "kg", 0.033, 0.033, "TEPCat WASP-17 Southworth solution"),
        "planet_radius": SourceMeasurement(1.932, "R_J", 1.38122544e8, "m", 0.053, 0.053, "TEPCat WASP-17 Southworth solution"),
        "stellar_mass": SourceMeasurement(1.286, "M_sun", 2.55717242e30, "kg", 0.079, 0.079, "TEPCat WASP-17 Southworth solution"),
        "stellar_radius": SourceMeasurement(1.583, "R_sun", 1.10129310e9, "m", 0.041, 0.041, "TEPCat WASP-17 Southworth solution"),
        "stellar_effective_temperature": SourceMeasurement(6550.0, "K", 6550.0, "K", 100.0, 100.0, "TEPCat WASP-17 Southworth solution"),
        "stellar_metallicity": SourceMeasurement(-0.25, "dex", -0.25, "dex", 0.09, 0.09, "TEPCat WASP-17 Southworth solution; provenance only"),
        "semimajor_axis": SourceMeasurement(0.05135, "AU", 7.681850660445e9, "m", 0.00103, 0.00103, "TEPCat WASP-17 Southworth solution"),
        "orbital_eccentricity": SourceMeasurement(0.0, "dimensionless", 0.0, "dimensionless", 0.0, 0.0, "Version-2 frozen circular orbit"),
        "orbital_period": SourceMeasurement(3.73548546, "day", 3.73548546 * DAY_TO_S, "s", 2.7e-7, 2.7e-7, "TEPCat current ephemeris"),
        "distance": SourceMeasurement(405.908, "pc", 1.252501215748e19, "m", 8.779, 8.421, "NASA Exoplanet Archive Gaia distance"),
    }
    planet_mass = measurements["planet_mass"].si_value
    planet_radius = measurements["planet_radius"].si_value
    stellar_radius = measurements["stellar_radius"].si_value
    semimajor_axis = measurements["semimajor_axis"].si_value
    stellar_temperature = measurements["stellar_effective_temperature"].si_value
    gravity = constants["G_m3_kg_s2"] * planet_mass / planet_radius**2
    radius_ratio = planet_radius / stellar_radius
    derived = {
        "surface_gravity_m_s2": gravity,
        "radius_ratio": radius_ratio,
        "projected_area_ratio": radius_ratio**2,
        "equilibrium_temperature_full_redistribution_zero_albedo_k": stellar_temperature
        * np.sqrt(stellar_radius / (2.0 * semimajor_axis)),
        "substellar_irradiation_temperature_k": stellar_temperature
        * np.sqrt(stellar_radius / semimajor_axis),
    }
    grids = tuple(
        PressureGridContract(
            n_cells=n_cells,
            edges_bar=np.geomspace(1.0e-5, 100.0, n_cells + 1),
            centers_bar=np.geomspace(1.0e-5, 100.0, n_cells + 1)[:-1]
            ** 0.5
            * np.geomspace(1.0e-5, 100.0, n_cells + 1)[1:] ** 0.5,
        )
        for n_cells in PRESSURE_CELLS
    )
    pg14_parameters = {
        "pg14_non_inverted": PG14Parameters(100.0, 0.001, 0.1, 0.5, 0.5, 1500.0),
        "pg14_inverted": PG14Parameters(100.0, 0.001, 0.1, 10.0, 0.5, 1500.0),
    }
    profiles: dict[str, NDArray[np.float64]] = {}
    for grid in grids:
        profiles[f"isothermal_{grid.n_cells}_cells"] = np.full(
            grid.n_cells, 1755.0
        )
        pressure_grid = PressureGrid(
            edges=grid.edges_bar,
            centers=grid.centers_bar,
            unit="bar",
            name=f"version_2_{grid.n_cells}_cells",
        )
        for name, parameters in pg14_parameters.items():
            evaluator = ParmentierGuillot2014TemperatureProfile(
                gravity=gravity,
                internal_temperature=parameters.internal_temperature_k,
            )
            profiles[f"{name}_{grid.n_cells}_cells"] = evaluator.evaluate(
                {
                    "kappa_IR": parameters.kappa_ir_m2_kg,
                    "gamma1": parameters.gamma1,
                    "gamma2": parameters.gamma2,
                    "alpha": parameters.alpha,
                    "T_irr": parameters.irradiation_temperature_k,
                },
                pressure_grid,
            )
    composition = {
        "H2O": 3.222572565623962e-4,
        "CO": 4.598708447732890e-4,
        "CO2": 6.734289181697181e-8,
        "CH4": 3.861237147673902e-8,
        "H2": 0.8540314245518249,
        "He": 0.14518634139157618,
    }
    masses = {
        "H2": 2.01588,
        "He": 4.002602,
        "H2O": 18.01528,
        "CO": 28.0101,
        "CO2": 44.0095,
        "CH4": 16.04246,
    }
    picaso_assets = {
        "CH4": OpacityAsset(
            "CH4_1460.hdf5",
            "CH4",
            "1474ca8c5236c9f7571aabb08b9d983ac8d244a55561a3348d078e9c7b31758f",
            "10.5281/zenodo.18644980",
        ),
        "CO2": OpacityAsset(
            "CO2_1460.hdf5",
            "CO2",
            "83feeddf1f3de9f385c6dc21650636af8a95ae5619aadcfaa43e51e9ae2d1510",
            "10.5281/zenodo.18644980",
        ),
        "CO": OpacityAsset(
            "CO_1460.hdf5",
            "CO",
            "96be68a9b1dce6e645c1fea28026fdf8b5f00dec20e6abda2f99693421f65cec",
            "10.5281/zenodo.18644980",
        ),
        "H2O": OpacityAsset(
            "H2O_1460.hdf5",
            "H2O",
            "9be15e41e59dc8689fb2f4d0992c8cef07dcc4eb3bd86a0b537d981f906b6672",
            "10.5281/zenodo.18644980",
        ),
    }
    r100_edges, r100_centers = _r100_grid()
    native = np.unique(
        np.concatenate((np.geomspace(0.8, 12.0, 4097), r100_edges))
    )
    spectral = SpectralContract(r100_edges, r100_centers, native)
    stellar_native = planck_surface_flux_w_m2_m(native, stellar_temperature)
    stellar_r100 = flux_conserving_bin_mean(native, stellar_native, r100_edges)
    return Version2CommonContract(
        measurements=measurements,
        constants=constants,
        derived=derived,
        pressure_grids=grids,
        pg14_parameters=pg14_parameters,
        temperature_profiles_k=profiles,
        composition_vmr=composition,
        molecular_masses_u=masses,
        picaso_correlated_k_assets=picaso_assets,
        spectral=spectral,
        stellar_surface_flux_native_w_m2_m=stellar_native,
        stellar_surface_flux_r100_w_m2_m=stellar_r100,
        pg14_implementation_sha256=_pg14_source_checksum(),
        pg14_method_sha256=_pg14_method_checksum(),
    )


def write_version_2_common_contract(
    contract: Version2CommonContract, json_path: Path, profiles_path: Path
) -> None:
    """Serialize the common contract and its reusable numerical arrays."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n")
    arrays: dict[str, NDArray[np.float64]] = {
        "r100_edges_micron": contract.spectral.r100_edges_micron,
        "r100_centers_micron": contract.spectral.r100_centers_micron,
        "native_reference_wavelength_micron": contract.spectral.native_reference_wavelength_micron,
        "stellar_surface_flux_native_w_m2_m": contract.stellar_surface_flux_native_w_m2_m,
        "stellar_surface_flux_r100_w_m2_m": contract.stellar_surface_flux_r100_w_m2_m,
    }
    for grid in contract.pressure_grids:
        arrays[f"pressure_edges_{grid.n_cells}_bar"] = grid.edges_bar
        arrays[f"pressure_centers_{grid.n_cells}_bar"] = grid.centers_bar
        arrays[f"picaso_levels_{grid.n_cells}_bar"] = grid.picaso_levels_bar
        arrays[f"petitradtrans_nodes_{grid.n_cells}_bar"] = grid.petitradtrans_nodes_bar
    arrays.update(contract.temperature_profiles_k)
    np.savez_compressed(profiles_path, **arrays)
