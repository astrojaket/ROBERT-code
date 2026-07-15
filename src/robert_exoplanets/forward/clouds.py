"""Shared parameterized cloud models for emission and transmission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.rt import (
    CloudOpticalProperties,
    GasOpticalDepth,
    grey_cloud_deck,
    power_law_haze_from_mass_extinction,
)


@runtime_checkable
class ParameterizedCloudModel(Protocol):
    """Geometry-independent cloud optical-property parameterization."""

    @property
    def required_parameters(self) -> tuple[str, ...]:
        """Names of retrieval parameters consumed by the cloud model."""

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        """Serializable description of the cloud parameterization."""

    @property
    def multiple_scattering_backend(self) -> str:
        """Emission multiple-scattering backend requested by the cloud."""

    def evaluate(
        self,
        gas_optical_depth: GasOpticalDepth,
        parameters: Mapping[str, float],
    ) -> tuple[CloudOpticalProperties, ...]:
        """Evaluate cloud properties on an atmosphere and spectral grid."""


@dataclass(frozen=True)
class ParameterizedDeckHazeCloudModel:
    """Shared finite grey-deck plus well-mixed power-law haze model.

    The grey deck is specified by its top pressure and integrated vertical
    extinction optical depth. The haze is specified by a bulk-atmosphere mass
    extinction coefficient at a reference wavelength and a wavelength power
    law. Both components return the same optical-property objects to emission
    and transmission; only the downstream radiative-transfer geometry differs.
    """

    log10_cloud_top_pressure_bar_parameter: str = "log_cloud_top_pressure_bar"
    log10_cloud_optical_depth_parameter: str = "log_cloud_optical_depth"
    log10_haze_mass_extinction_parameter: str = "log_haze_mass_extinction"
    haze_slope_parameter: str = "haze_slope"
    haze_reference_wavelength_micron: float = 1.0
    deck_single_scattering_albedo: float = 0.0
    deck_asymmetry_factor: float = 0.0
    haze_single_scattering_albedo: float = 1.0
    haze_asymmetry_factor: float = 0.0
    multiple_scattering_backend: str = "sh4"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameter_fields = (
            "log10_cloud_top_pressure_bar_parameter",
            "log10_cloud_optical_depth_parameter",
            "log10_haze_mass_extinction_parameter",
            "haze_slope_parameter",
        )
        parameters = tuple(str(getattr(self, name)).strip() for name in parameter_fields)
        if any(not parameter for parameter in parameters):
            raise RobertValidationError("cloud parameter names must not be empty")
        if len(set(parameters)) != len(parameters):
            raise RobertValidationError("cloud parameter names must be unique")

        reference_wavelength = float(self.haze_reference_wavelength_micron)
        if not np.isfinite(reference_wavelength) or reference_wavelength <= 0.0:
            raise RobertValidationError(
                "haze_reference_wavelength_micron must be finite and positive"
            )
        for field_name in (
            "deck_single_scattering_albedo",
            "haze_single_scattering_albedo",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RobertValidationError(f"{field_name} must lie in [0, 1]")
            object.__setattr__(self, field_name, value)
        for field_name in ("deck_asymmetry_factor", "haze_asymmetry_factor"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise RobertValidationError(f"{field_name} must lie in [-1, 1]")
            object.__setattr__(self, field_name, value)

        backend = self.multiple_scattering_backend.strip().lower()
        if backend not in {"none", "two_stream", "toon_hemispheric_mean", "sh4", "p3"}:
            raise RobertValidationError(
                "multiple_scattering_backend must be 'none', 'two_stream', or 'sh4'"
            )
        for field_name, value in zip(parameter_fields, parameters, strict=True):
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "haze_reference_wavelength_micron", reference_wavelength)
        object.__setattr__(self, "multiple_scattering_backend", backend)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (
            self.log10_cloud_top_pressure_bar_parameter,
            self.log10_cloud_optical_depth_parameter,
            self.log10_haze_mass_extinction_parameter,
            self.haze_slope_parameter,
        )

    @property
    def manifest_metadata(self) -> Mapping[str, str]:
        return immutable_mapping(
            {
                "cloud_model": "grey_deck_power_law_haze",
                "cloud_geometry_independent": "true",
                "cloud_log10_top_pressure_parameter": (
                    self.log10_cloud_top_pressure_bar_parameter
                ),
                "cloud_log10_optical_depth_parameter": (
                    self.log10_cloud_optical_depth_parameter
                ),
                "haze_log10_mass_extinction_parameter": (
                    self.log10_haze_mass_extinction_parameter
                ),
                "haze_slope_parameter": self.haze_slope_parameter,
                "haze_reference_wavelength_micron": (
                    f"{self.haze_reference_wavelength_micron:.17g}"
                ),
                "deck_single_scattering_albedo": (
                    f"{self.deck_single_scattering_albedo:.17g}"
                ),
                "deck_asymmetry_factor": f"{self.deck_asymmetry_factor:.17g}",
                "haze_single_scattering_albedo": (
                    f"{self.haze_single_scattering_albedo:.17g}"
                ),
                "haze_asymmetry_factor": f"{self.haze_asymmetry_factor:.17g}",
                "cloud_multiple_scattering_backend": self.multiple_scattering_backend,
                **dict(self.metadata),
            }
        )

    def evaluate(
        self,
        gas_optical_depth: GasOpticalDepth,
        parameters: Mapping[str, float],
    ) -> tuple[CloudOpticalProperties, CloudOpticalProperties]:
        """Evaluate deck and haze optical properties on the prepared RT grid."""

        values = {}
        for name in self.required_parameters:
            if name not in parameters:
                raise RobertValidationError(f"cloud parameter is missing: {name}")
            value = float(parameters[name])
            if not np.isfinite(value):
                raise RobertValidationError(f"cloud parameter {name!r} must be finite")
            values[name] = value

        cloud_top_pressure_bar = 10.0 ** values[
            self.log10_cloud_top_pressure_bar_parameter
        ]
        cloud_optical_depth = 10.0 ** values[
            self.log10_cloud_optical_depth_parameter
        ]
        haze_mass_extinction = 10.0 ** values[
            self.log10_haze_mass_extinction_parameter
        ]
        haze_slope = values[self.haze_slope_parameter]
        if not all(
            np.isfinite(value)
            for value in (
                cloud_top_pressure_bar,
                cloud_optical_depth,
                haze_mass_extinction,
            )
        ):
            raise RobertValidationError("log10 cloud parameters overflowed")

        deck = grey_cloud_deck(
            gas_optical_depth.pressure_grid,
            gas_optical_depth.spectral_grid,
            cloud_top_pressure=cloud_top_pressure_bar,
            cloud_top_pressure_unit="bar",
            optical_depth=cloud_optical_depth,
            single_scattering_albedo=self.deck_single_scattering_albedo,
            asymmetry_factor=self.deck_asymmetry_factor,
        )
        haze = power_law_haze_from_mass_extinction(
            gas_optical_depth,
            mass_extinction_at_reference_cm2_g=haze_mass_extinction,
            reference_wavelength_micron=self.haze_reference_wavelength_micron,
            slope=haze_slope,
            single_scattering_albedo=self.haze_single_scattering_albedo,
            asymmetry_factor=self.haze_asymmetry_factor,
        )
        return deck, haze


__all__ = [
    "ParameterizedCloudModel",
    "ParameterizedDeckHazeCloudModel",
]
