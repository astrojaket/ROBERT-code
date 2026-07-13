"""Cloud and aerosol optical properties for RT calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import PressureGrid, RobertValidationError, SpectralGrid
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.opacity import pressure_values_in_unit, spectral_grid_values_in_unit

from .extinction import LayerOpticalDepth
from .optical_depth import GasOpticalDepth


@dataclass(frozen=True)
class CloudOpticalProperties:
    """Layer cloud or aerosol optical properties on the RT spectral grid.

    The object stores extinction optical depth together with single-scattering
    albedo and asymmetry factor. It can be passed directly to
    ``solve_emission`` as an additional optical-depth contributor, or
    split into absorption and scattering ``LayerOpticalDepth`` objects for
    explicit diagnostics.
    """

    name: str
    extinction_tau: ArrayLike
    spectral_grid: SpectralGrid
    pressure_grid: PressureGrid
    single_scattering_albedo: ArrayLike | float = 0.0
    asymmetry_factor: ArrayLike | float = 0.0
    phase_function_moments: ArrayLike | None = None
    unit: str = "dimensionless"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("cloud optical-property name must not be empty")
        if not self.unit:
            raise RobertValidationError("cloud optical-property unit must not be empty")
        target_shape = (self.pressure_grid.n_layers, self.spectral_grid.size)
        extinction_tau = _readonly_array(self.extinction_tau, "extinction_tau", target_shape)
        if np.any(extinction_tau < 0.0):
            raise RobertValidationError("cloud extinction optical depth must be non-negative")
        single_scattering_albedo = _readonly_broadcast_property(
            self.single_scattering_albedo,
            "single_scattering_albedo",
            target_shape,
        )
        if np.any(single_scattering_albedo < 0.0) or np.any(single_scattering_albedo > 1.0):
            raise RobertValidationError("single_scattering_albedo must be in [0, 1]")
        asymmetry_factor = _readonly_broadcast_property(
            self.asymmetry_factor,
            "asymmetry_factor",
            target_shape,
        )
        if np.any(asymmetry_factor < -1.0) or np.any(asymmetry_factor > 1.0):
            raise RobertValidationError("asymmetry_factor must be in [-1, 1]")
        phase_function_moments = None
        if self.phase_function_moments is not None:
            phase_function_moments = _readonly_phase_function_moments(
                self.phase_function_moments,
                target_shape,
            )
            if not np.allclose(
                phase_function_moments[1] / 3.0,
                asymmetry_factor,
                rtol=2.0e-8,
                atol=2.0e-10,
            ):
                raise RobertValidationError(
                    "phase_function_moments[1] / 3 must match asymmetry_factor"
                )

        object.__setattr__(self, "extinction_tau", extinction_tau)
        object.__setattr__(self, "single_scattering_albedo", single_scattering_albedo)
        object.__setattr__(self, "asymmetry_factor", asymmetry_factor)
        object.__setattr__(self, "phase_function_moments", phase_function_moments)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def absorption_tau(self) -> NDArray[np.float64]:
        """Absorbing part of the cloud optical depth."""

        tau = self.extinction_tau * (1.0 - self.single_scattering_albedo)
        tau.setflags(write=False)
        return tau

    @property
    def scattering_tau(self) -> NDArray[np.float64]:
        """Scattering part of the cloud optical depth."""

        tau = self.extinction_tau * self.single_scattering_albedo
        tau.setflags(write=False)
        return tau

    @property
    def transport_scattering_tau(self) -> NDArray[np.float64]:
        """Scattering optical depth reduced by the anisotropy transport factor."""

        tau = self.scattering_tau * (1.0 - self.asymmetry_factor)
        tau.setflags(write=False)
        return tau

    def absorption_optical_depth(self, *, name: str | None = None) -> LayerOpticalDepth:
        """Return the absorbing component as a layer optical-depth object."""

        return LayerOpticalDepth(
            name=name or f"{self.name} absorption",
            tau=self.absorption_tau,
            spectral_grid=self.spectral_grid,
            pressure_grid=self.pressure_grid,
            kind="cloud_absorption",
            unit=self.unit,
            metadata=self._component_metadata("absorption"),
        )

    def scattering_optical_depth(self, *, name: str | None = None) -> LayerOpticalDepth:
        """Return the scattering-extinction component as a layer optical depth."""

        return LayerOpticalDepth(
            name=name or f"{self.name} scattering",
            tau=self.scattering_tau,
            spectral_grid=self.spectral_grid,
            pressure_grid=self.pressure_grid,
            kind="cloud_scattering_extinction",
            unit=self.unit,
            phase_function_moments=self.phase_function_moments,
            metadata=self._component_metadata("scattering"),
        )

    def as_layer_optical_depths(self) -> tuple[LayerOpticalDepth, LayerOpticalDepth]:
        """Return absorption and scattering optical-depth components."""

        return self.absorption_optical_depth(), self.scattering_optical_depth()

    def _component_metadata(self, component: str) -> dict[str, str]:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "cloud_name": self.name,
                "cloud_component": component,
                "single_scattering_albedo_min": f"{float(np.min(self.single_scattering_albedo)):.12g}",
                "single_scattering_albedo_max": f"{float(np.max(self.single_scattering_albedo)):.12g}",
                "asymmetry_factor_min": f"{float(np.min(self.asymmetry_factor)):.12g}",
                "asymmetry_factor_max": f"{float(np.max(self.asymmetry_factor)):.12g}",
            }
        )
        return metadata


def grey_cloud_deck(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    *,
    cloud_top_pressure: float,
    optical_depth: float,
    cloud_top_pressure_unit: str | None = None,
    name: str = "grey cloud deck",
    single_scattering_albedo: ArrayLike | float = 0.0,
    asymmetry_factor: ArrayLike | float = 0.0,
) -> CloudOpticalProperties:
    """Create a spectrally grey cloud deck below a pressure level.

    ``optical_depth`` is the vertical extinction optical depth integrated over
    all layer centers deeper than ``cloud_top_pressure``.
    """

    tau_total = _non_negative_scalar(optical_depth, "optical_depth")
    top_pressure = _positive_scalar(cloud_top_pressure, "cloud_top_pressure")
    top_unit = cloud_top_pressure_unit or pressure_grid.unit
    pressure_pa = pressure_values_in_unit(pressure_grid.centers, pressure_grid.unit, "pa")
    top_pressure_pa = pressure_values_in_unit(np.array([top_pressure]), top_unit, "pa")[0]
    active = pressure_pa >= top_pressure_pa
    if not np.any(active):
        raise RobertValidationError("grey cloud deck has no active layers below cloud_top_pressure")
    tau = np.zeros((pressure_grid.n_layers, spectral_grid.size), dtype=float)
    tau[active] = tau_total / int(np.sum(active))
    return CloudOpticalProperties(
        name=name,
        extinction_tau=tau,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=single_scattering_albedo,
        asymmetry_factor=asymmetry_factor,
        metadata={
            "vertical_model": "grey_cloud_deck_uniform_tau_below_top",
            "cloud_top_pressure_pa": f"{float(top_pressure_pa):.12g}",
            "input_optical_depth": f"{tau_total:.12g}",
        },
    )


def grey_cloud_from_mass_extinction(
    gas_optical_depth: GasOpticalDepth,
    *,
    mass_extinction_cm2_g: float,
    name: str = "vertically uniform grey cloud",
    single_scattering_albedo: ArrayLike | float = 1.0,
    asymmetry_factor: ArrayLike | float = 0.0,
) -> CloudOpticalProperties:
    """Convert a constant cloud mass-extinction coefficient into layer tau.

    The coefficient is per gram of bulk atmosphere. Hydrostatic mass column
    gives ``tau_layer = kappa * delta_pressure / gravity`` after converting
    ``cm2/g`` to ``m2/kg``. This matches the vertically uniform gray-opacity
    parameterization used by the Schlawin et al. (2024) two-region model.
    """

    opacity = _non_negative_scalar(mass_extinction_cm2_g, "mass_extinction_cm2_g")
    layer_tau = (
        opacity
        * 0.1
        * gas_optical_depth.layer_pressure_thickness_pa
        / gas_optical_depth.gravity_m_s2
    )
    tau = np.repeat(layer_tau[:, None], gas_optical_depth.spectral_grid.size, axis=1)
    return CloudOpticalProperties(
        name=name,
        extinction_tau=tau,
        spectral_grid=gas_optical_depth.spectral_grid,
        pressure_grid=gas_optical_depth.pressure_grid,
        single_scattering_albedo=single_scattering_albedo,
        asymmetry_factor=asymmetry_factor,
        metadata={
            "vertical_model": "constant_mass_extinction_per_bulk_atmosphere_mass",
            "mass_extinction_cm2_g": f"{opacity:.17g}",
            "hydrostatic_conversion": "tau=kappa*delta_pressure/gravity",
        },
    )


def power_law_haze(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    *,
    optical_depth_at_reference: float,
    reference_wavelength_micron: float = 1.0,
    slope: float = -4.0,
    name: str = "power-law haze",
    single_scattering_albedo: ArrayLike | float = 1.0,
    asymmetry_factor: ArrayLike | float = 0.0,
) -> CloudOpticalProperties:
    """Create a vertically uniform aerosol haze with a wavelength power law."""

    tau_reference = _non_negative_scalar(optical_depth_at_reference, "optical_depth_at_reference")
    reference_wavelength = _positive_scalar(reference_wavelength_micron, "reference_wavelength_micron")
    spectral_slope = float(slope)
    if not np.isfinite(spectral_slope):
        raise RobertValidationError("slope must be finite")
    wavelength = spectral_grid_values_in_unit(spectral_grid, "micron")
    tau_spectral = tau_reference * np.power(wavelength / reference_wavelength, spectral_slope)
    tau = np.repeat((tau_spectral / pressure_grid.n_layers)[None, :], pressure_grid.n_layers, axis=0)
    return CloudOpticalProperties(
        name=name,
        extinction_tau=tau,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        single_scattering_albedo=single_scattering_albedo,
        asymmetry_factor=asymmetry_factor,
        metadata={
            "vertical_model": "uniform_tau_per_layer",
            "spectral_model": "power_law",
            "reference_wavelength_micron": f"{reference_wavelength:.12g}",
            "optical_depth_at_reference": f"{tau_reference:.12g}",
            "slope": f"{spectral_slope:.12g}",
        },
    )


def _readonly_array(
    values: ArrayLike,
    name: str,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != shape:
        raise RobertValidationError(f"{name} has incorrect shape")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_broadcast_property(
    values: ArrayLike | float,
    name: str,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim == 0:
        broadcast = np.full(shape, float(array), dtype=float)
    elif array.shape == shape:
        broadcast = array
    elif array.shape == (shape[0],):
        broadcast = np.repeat(array[:, None], shape[1], axis=1)
    elif array.shape == (shape[1],):
        broadcast = np.repeat(array[None, :], shape[0], axis=0)
    else:
        raise RobertValidationError(f"{name} has incorrect shape")
    if not np.all(np.isfinite(broadcast)):
        raise RobertValidationError(f"{name} must contain only finite values")
    broadcast = np.array(broadcast, dtype=float, copy=True)
    broadcast.setflags(write=False)
    return broadcast


def _readonly_phase_function_moments(
    values: ArrayLike,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape == (5, shape[1]):
        array = np.repeat(array[:, None, :], shape[0], axis=1)
    elif array.shape != (5, shape[0], shape[1]):
        raise RobertValidationError(
            "phase_function_moments must have shape (5, spectral) or "
            "(5, layer, spectral)"
        )
    if not np.all(np.isfinite(array)):
        raise RobertValidationError("phase_function_moments must be finite")
    limits = (2.0 * np.arange(5) + 1.0)[:, None, None]
    if np.any(np.abs(array) > limits * (1.0 + 1.0e-10)):
        raise RobertValidationError("phase_function_moments exceed physical bounds")
    if not np.allclose(array[0], 1.0, rtol=0.0, atol=1.0e-12):
        raise RobertValidationError("phase_function_moments[0] must equal one")
    array.setflags(write=False)
    return array


def _positive_scalar(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise RobertValidationError(f"{name} must be finite and positive")
    return number


def _non_negative_scalar(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise RobertValidationError(f"{name} must be finite and non-negative")
    return number
