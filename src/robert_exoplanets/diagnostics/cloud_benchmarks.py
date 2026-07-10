"""Cloud optical-property benchmark diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import pressure_values_in_unit, spectral_grid_values_in_unit
from robert_exoplanets.rt import CloudOpticalProperties


@dataclass(frozen=True)
class CloudOpticalPropertyComparison:
    """Summary comparison between two cloud optical-property products."""

    name: str
    max_abs_extinction_tau: float
    max_abs_single_scattering_albedo: float
    max_abs_asymmetry_factor: float
    rms_extinction_tau: float
    rms_single_scattering_albedo: float
    rms_asymmetry_factor: float
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("cloud comparison name must not be empty")
        for field_name in (
            "max_abs_extinction_tau",
            "max_abs_single_scattering_albedo",
            "max_abs_asymmetry_factor",
            "rms_extinction_tau",
            "rms_single_scattering_albedo",
            "rms_asymmetry_factor",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise RobertValidationError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def as_dict(self) -> dict[str, float | str]:
        """Return a serializable comparison summary."""

        return {
            "name": self.name,
            "max_abs_extinction_tau": self.max_abs_extinction_tau,
            "max_abs_single_scattering_albedo": self.max_abs_single_scattering_albedo,
            "max_abs_asymmetry_factor": self.max_abs_asymmetry_factor,
            "rms_extinction_tau": self.rms_extinction_tau,
            "rms_single_scattering_albedo": self.rms_single_scattering_albedo,
            "rms_asymmetry_factor": self.rms_asymmetry_factor,
            **dict(self.metadata),
        }


def compare_cloud_optical_properties(
    reference: CloudOpticalProperties,
    candidate: CloudOpticalProperties,
    *,
    name: str = "cloud optical-property comparison",
) -> CloudOpticalPropertyComparison:
    """Compare two cloud optical-property products on matching grids."""

    _validate_cloud_grids_match(reference, candidate)
    tau_delta = candidate.extinction_tau - reference.extinction_tau
    ssa_delta = candidate.single_scattering_albedo - reference.single_scattering_albedo
    asymmetry_delta = candidate.asymmetry_factor - reference.asymmetry_factor
    return CloudOpticalPropertyComparison(
        name=name,
        max_abs_extinction_tau=float(np.max(np.abs(tau_delta))),
        max_abs_single_scattering_albedo=float(np.max(np.abs(ssa_delta))),
        max_abs_asymmetry_factor=float(np.max(np.abs(asymmetry_delta))),
        rms_extinction_tau=_rms(tau_delta),
        rms_single_scattering_albedo=_rms(ssa_delta),
        rms_asymmetry_factor=_rms(asymmetry_delta),
        metadata={
            "reference": reference.name,
            "candidate": candidate.name,
            "n_layers": str(reference.pressure_grid.n_layers),
            "n_wavelength": str(reference.spectral_grid.size),
        },
    )


def _validate_cloud_grids_match(
    reference: CloudOpticalProperties,
    candidate: CloudOpticalProperties,
) -> None:
    reference_pressure = pressure_values_in_unit(
        reference.pressure_grid.centers,
        reference.pressure_grid.unit,
        "pa",
    )
    candidate_pressure = pressure_values_in_unit(
        candidate.pressure_grid.centers,
        candidate.pressure_grid.unit,
        "pa",
    )
    if reference_pressure.shape != candidate_pressure.shape or not np.allclose(
        reference_pressure,
        candidate_pressure,
        rtol=1.0e-10,
        atol=0.0,
    ):
        raise RobertValidationError("cloud pressure grids must match for comparison")
    reference_wavelength = spectral_grid_values_in_unit(reference.spectral_grid, "micron")
    candidate_wavelength = spectral_grid_values_in_unit(candidate.spectral_grid, "micron")
    if reference_wavelength.shape != candidate_wavelength.shape or not np.allclose(
        reference_wavelength,
        candidate_wavelength,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise RobertValidationError("cloud spectral grids must match for comparison")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))
