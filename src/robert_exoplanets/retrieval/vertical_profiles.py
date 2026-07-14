"""Pressure-resolved state-vector helpers for atmospheric sounding.

The covariance convention follows the continuous-profile parameterisation
used by NEMESIS: state elements separated by ``dlnp`` are correlated as
``exp(-abs(dlnp) / correlation_length)``.  Temperature is normally carried
linearly, while positive quantities such as VMR and aerosol abundance are
normally carried in natural-log space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError

from .priors import RetrievalParameter, RetrievalParameterSet, UniformPrior


ProfileTransform = Literal["linear", "log"]
ProfileKind = Literal["temperature", "vmr", "aerosol", "cloud_fraction", "generic"]


@dataclass(frozen=True)
class VerticalProfileParameterization:
    """One pressure-resolved block in an OE state vector.

    ``prior_state`` and ``prior_standard_deviation`` are expressed in the
    retrieval coordinates.  For ``transform="log"`` they are therefore the
    natural logarithm of the physical profile and its standard deviation in
    log space, respectively.  The named constructors provide less ambiguous
    interfaces for common atmospheric quantities.
    """

    name: str
    pressure: ArrayLike
    prior_state: ArrayLike
    prior_standard_deviation: ArrayLike | float
    correlation_length: float
    transform: ProfileTransform = "linear"
    kind: ProfileKind = "generic"
    pressure_unit: str = "bar"
    physical_unit: str | None = None
    bound_sigma: float = 8.0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise RobertValidationError("vertical-profile name must not be empty")
        if self.transform not in {"linear", "log"}:
            raise RobertValidationError("vertical-profile transform must be 'linear' or 'log'")
        if self.kind not in {
            "temperature",
            "vmr",
            "aerosol",
            "cloud_fraction",
            "generic",
        }:
            raise RobertValidationError("unsupported vertical-profile kind")
        if not str(self.pressure_unit).strip():
            raise RobertValidationError("vertical-profile pressure_unit must not be empty")

        pressure = _finite_vector(self.pressure, "pressure")
        if pressure.size < 2 or np.any(pressure <= 0.0):
            raise RobertValidationError("vertical-profile pressure must contain at least two positive levels")
        if np.unique(pressure).size != pressure.size:
            raise RobertValidationError("vertical-profile pressure levels must be unique")
        state = _finite_vector(self.prior_state, "prior_state")
        if state.shape != pressure.shape:
            raise RobertValidationError("prior_state must match vertical-profile pressure")
        sigma = np.asarray(self.prior_standard_deviation, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(pressure.size, float(sigma), dtype=float)
        else:
            sigma = np.array(sigma, dtype=float, copy=True)
        if sigma.shape != pressure.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise RobertValidationError("prior_standard_deviation must be positive and match vertical-profile pressure")
        correlation_length = float(self.correlation_length)
        bound_sigma = float(self.bound_sigma)
        if not np.isfinite(correlation_length) or correlation_length <= 0.0:
            raise RobertValidationError("correlation_length must be finite and positive")
        if not np.isfinite(bound_sigma) or bound_sigma <= 0.0:
            raise RobertValidationError("bound_sigma must be finite and positive")

        pressure.setflags(write=False)
        state.setflags(write=False)
        sigma.setflags(write=False)
        object.__setattr__(self, "pressure", pressure)
        object.__setattr__(self, "prior_state", state)
        object.__setattr__(self, "prior_standard_deviation", sigma)
        object.__setattr__(self, "correlation_length", correlation_length)
        object.__setattr__(self, "bound_sigma", bound_sigma)

    @classmethod
    def temperature(
        cls,
        *,
        pressure: ArrayLike,
        prior_temperature: ArrayLike,
        prior_sigma_K: ArrayLike | float,
        correlation_length: float,
        name: str = "temperature",
        pressure_unit: str = "bar",
        bound_sigma: float = 8.0,
    ) -> "VerticalProfileParameterization":
        """Construct a linear temperature-profile state block."""

        return cls(
            name=name,
            pressure=pressure,
            prior_state=prior_temperature,
            prior_standard_deviation=prior_sigma_K,
            correlation_length=correlation_length,
            transform="linear",
            kind="temperature",
            pressure_unit=pressure_unit,
            physical_unit="K",
            bound_sigma=bound_sigma,
        )

    @classmethod
    def positive_profile(
        cls,
        *,
        name: str,
        pressure: ArrayLike,
        prior_profile: ArrayLike,
        prior_fractional_uncertainty: ArrayLike | float,
        correlation_length: float,
        kind: Literal["vmr", "aerosol", "cloud_fraction", "generic"] = "generic",
        pressure_unit: str = "bar",
        physical_unit: str | None = None,
        bound_sigma: float = 8.0,
    ) -> "VerticalProfileParameterization":
        """Construct a positive profile carried in natural-log space.

        NEMESIS converts an absolute a-priori error to log-space using the
        fractional error ``sigma / value``.  This constructor accepts that
        fractional uncertainty directly.
        """

        profile = _finite_vector(prior_profile, "prior_profile")
        if np.any(profile <= 0.0):
            raise RobertValidationError("positive prior profiles must be strictly positive")
        fractional = np.asarray(prior_fractional_uncertainty, dtype=float)
        if fractional.ndim == 0:
            fractional = np.full(profile.size, float(fractional), dtype=float)
        if fractional.shape != profile.shape:
            raise RobertValidationError("prior_fractional_uncertainty must match prior_profile")
        return cls(
            name=name,
            pressure=pressure,
            prior_state=np.log(profile),
            prior_standard_deviation=fractional,
            correlation_length=correlation_length,
            transform="log",
            kind=kind,
            pressure_unit=pressure_unit,
            physical_unit=physical_unit,
            bound_sigma=bound_sigma,
        )

    @property
    def n_levels(self) -> int:
        """Number of retrieved pressure levels."""

        return int(self.pressure.size)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Stable state-vector names in pressure-array order."""

        return tuple(f"{self.name}[{index:03d}]" for index in range(self.n_levels))

    @property
    def prior_covariance(self) -> NDArray[np.float64]:
        """Return the NEMESIS-style pressure-correlated a-priori covariance."""

        covariance = pressure_correlated_covariance(
            self.pressure,
            self.prior_standard_deviation,
            correlation_length=self.correlation_length,
        )
        covariance.setflags(write=False)
        return covariance

    def retrieval_parameters(self) -> tuple[RetrievalParameter, ...]:
        """Return bounded scalar parameters for ROBERT's flat retrieval API."""

        parameters: list[RetrievalParameter] = []
        for index, (name, center, sigma, pressure) in enumerate(
            zip(
                self.parameter_names,
                self.prior_state,
                self.prior_standard_deviation,
                self.pressure,
                strict=True,
            )
        ):
            lower = float(center - self.bound_sigma * sigma)
            upper = float(center + self.bound_sigma * sigma)
            if self.kind == "temperature":
                lower = max(lower, np.finfo(float).tiny)
            parameters.append(
                RetrievalParameter(
                    name,
                    UniformPrior(lower, upper),
                    unit=self.physical_unit if self.transform == "linear" else "ln(physical)",
                    metadata={
                        "profile": self.name,
                        "profile_kind": self.kind,
                        "profile_transform": self.transform,
                        "pressure": f"{float(pressure):.17g}",
                        "pressure_unit": self.pressure_unit,
                        "level_index": str(index),
                    },
                )
            )
        return tuple(parameters)

    def state_from_mapping(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        """Extract this block from a flat parameter mapping."""

        try:
            state = np.array([parameters[name] for name in self.parameter_names], dtype=float)
        except KeyError as exc:
            raise RobertValidationError(f"missing vertical-profile parameter: {exc.args[0]}") from exc
        if not np.all(np.isfinite(state)):
            raise RobertValidationError("vertical-profile state must be finite")
        state.setflags(write=False)
        return state

    def physical_profile(self, parameters: Mapping[str, float] | ArrayLike) -> NDArray[np.float64]:
        """Decode state coordinates into a physical layer profile."""

        if isinstance(parameters, Mapping):
            state = self.state_from_mapping(parameters)
        else:
            state = _finite_vector(parameters, "vertical-profile state")
            if state.shape != (self.n_levels,):
                raise RobertValidationError("vertical-profile state has the wrong number of levels")
        profile = np.exp(state) if self.transform == "log" else np.array(state, copy=True)
        if not np.all(np.isfinite(profile)) or np.any(profile <= 0.0):
            raise RobertValidationError("decoded vertical profile must be finite and positive")
        if self.kind in {"vmr", "cloud_fraction"} and np.any(profile > 1.0):
            raise RobertValidationError(f"decoded {self.kind} profile must not exceed one")
        profile.setflags(write=False)
        return profile


@dataclass(frozen=True)
class LayerByLayerStateVector:
    """Ordered collection of independent vertical-profile state blocks."""

    profiles: tuple[VerticalProfileParameterization, ...]

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles:
            raise RobertValidationError("layer-by-layer state vector requires at least one profile")
        names = [profile.name for profile in profiles]
        if len(set(names)) != len(names):
            raise RobertValidationError("vertical-profile names must be unique")
        object.__setattr__(self, "profiles", profiles)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for profile in self.profiles for name in profile.parameter_names)

    @property
    def prior_state(self) -> NDArray[np.float64]:
        state = np.concatenate([profile.prior_state for profile in self.profiles])
        state.setflags(write=False)
        return state

    @property
    def prior_covariance(self) -> NDArray[np.float64]:
        sizes = [profile.n_levels for profile in self.profiles]
        covariance = np.zeros((sum(sizes), sum(sizes)), dtype=float)
        start = 0
        for profile, size in zip(self.profiles, sizes, strict=True):
            covariance[start : start + size, start : start + size] = profile.prior_covariance
            start += size
        covariance.setflags(write=False)
        return covariance

    @property
    def retrieval_parameters(self) -> RetrievalParameterSet:
        return RetrievalParameterSet(
            tuple(parameter for profile in self.profiles for parameter in profile.retrieval_parameters())
        )

    def physical_profiles(self, parameters: Mapping[str, float]) -> dict[str, NDArray[np.float64]]:
        """Decode every profile block from one flat parameter mapping."""

        return {profile.name: profile.physical_profile(parameters) for profile in self.profiles}


def pressure_correlated_covariance(
    pressure: ArrayLike,
    standard_deviation: ArrayLike | float,
    *,
    correlation_length: float,
) -> NDArray[np.float64]:
    """Build an exponential covariance in natural-log pressure separation.

    ``correlation_length`` is measured in pressure scale heights because one
    scale height corresponds to a unit change in natural-log pressure under
    hydrostatic conditions.
    """

    pressure_array = _finite_vector(pressure, "pressure")
    if np.any(pressure_array <= 0.0):
        raise RobertValidationError("pressure must be strictly positive")
    sigma = np.asarray(standard_deviation, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(pressure_array.size, float(sigma), dtype=float)
    if sigma.shape != pressure_array.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise RobertValidationError("standard_deviation must be positive and match pressure")
    length = float(correlation_length)
    if not np.isfinite(length) or length <= 0.0:
        raise RobertValidationError("correlation_length must be finite and positive")
    separation = np.abs(np.log(pressure_array[:, None] / pressure_array[None, :]))
    covariance = sigma[:, None] * sigma[None, :] * np.exp(-separation / length)
    covariance = 0.5 * (covariance + covariance.T)
    covariance.setflags(write=False)
    return covariance


def _finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must be a finite one-dimensional array")
    return array


__all__ = [
    "LayerByLayerStateVector",
    "ProfileKind",
    "ProfileTransform",
    "VerticalProfileParameterization",
    "pressure_correlated_covariance",
]
