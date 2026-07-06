"""Retrieval parameter priors and unit-cube transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError


class Prior(Protocol):
    """Protocol for scalar retrieval priors."""

    lower: float
    upper: float

    def transform(self, unit_value: float) -> float:
        """Map a unit-cube value into physical parameter space."""

    def log_probability(self, value: float) -> float:
        """Return the normalized log prior probability density."""


@dataclass(frozen=True)
class UniformPrior:
    """Uniform prior between finite lower and upper bounds."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise RobertValidationError("uniform prior bounds must be finite")
        if not upper > lower:
            raise RobertValidationError("uniform prior upper bound must exceed lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def transform(self, unit_value: float) -> float:
        cube_value = _unit_interval_value(unit_value)
        return float(self.lower + cube_value * (self.upper - self.lower))

    def log_probability(self, value: float) -> float:
        number = float(value)
        if self.lower <= number <= self.upper:
            return float(-np.log(self.upper - self.lower))
        return float("-inf")


@dataclass(frozen=True)
class LogUniformPrior:
    """Log-uniform prior between positive finite lower and upper bounds."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise RobertValidationError("log-uniform prior bounds must be finite")
        if lower <= 0.0:
            raise RobertValidationError("log-uniform prior lower bound must be positive")
        if not upper > lower:
            raise RobertValidationError("log-uniform prior upper bound must exceed lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def transform(self, unit_value: float) -> float:
        cube_value = _unit_interval_value(unit_value)
        log_lower = np.log(self.lower)
        log_upper = np.log(self.upper)
        return float(np.exp(log_lower + cube_value * (log_upper - log_lower)))

    def log_probability(self, value: float) -> float:
        number = float(value)
        if self.lower <= number <= self.upper:
            return float(-np.log(number) - np.log(np.log(self.upper / self.lower)))
        return float("-inf")


@dataclass(frozen=True)
class RetrievalParameter:
    """One named retrieval parameter with a scalar prior."""

    name: str
    prior: Prior
    label: str | None = None
    unit: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("retrieval parameter name must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def midpoint(self) -> float:
        """Finite midpoint in physical parameter space."""

        return float(0.5 * (self.prior.lower + self.prior.upper))

    @property
    def bounds(self) -> tuple[float, float]:
        """Finite lower and upper bounds."""

        return float(self.prior.lower), float(self.prior.upper)


@dataclass(frozen=True)
class RetrievalParameterSet:
    """Ordered collection of retrieval parameters."""

    parameters: tuple[RetrievalParameter, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise RobertValidationError("retrieval parameter set must contain at least one parameter")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise RobertValidationError("retrieval parameter names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names in vector order."""

        return tuple(parameter.name for parameter in self.parameters)

    @property
    def ndim(self) -> int:
        """Number of retrieval dimensions."""

        return len(self.parameters)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Finite bounds in vector order."""

        return tuple(parameter.bounds for parameter in self.parameters)

    def midpoint_vector(self) -> NDArray[np.float64]:
        """Return the prior-midpoint vector."""

        values = np.array([parameter.midpoint for parameter in self.parameters], dtype=float)
        values.setflags(write=False)
        return values

    def transform(self, cube: ArrayLike) -> NDArray[np.float64]:
        """Map a unit-cube vector into parameter space."""

        values = _vector(cube, self.ndim, "unit-cube vector")
        transformed = np.array(
            [parameter.prior.transform(float(value)) for parameter, value in zip(self.parameters, values, strict=True)],
            dtype=float,
        )
        transformed.setflags(write=False)
        return transformed

    def vector_to_mapping(self, vector: ArrayLike) -> dict[str, float]:
        """Convert a vector into a parameter-name mapping."""

        values = _vector(vector, self.ndim, "parameter vector")
        return {name: float(value) for name, value in zip(self.names, values, strict=True)}

    def log_prior_from_vector(self, vector: ArrayLike) -> float:
        """Return summed log prior probability for a vector."""

        values = _vector(vector, self.ndim, "parameter vector")
        logp = 0.0
        for parameter, value in zip(self.parameters, values, strict=True):
            current = parameter.prior.log_probability(float(value))
            if not np.isfinite(current):
                return float("-inf")
            logp += current
        return float(logp)


def _unit_interval_value(value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise RobertValidationError("unit-cube values must be finite and in [0, 1]")
    return number


def _vector(values: ArrayLike, expected_size: int, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != (expected_size,):
        raise RobertValidationError(f"{name} must have shape ({expected_size},)")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
