"""Retrieval parameter priors and unit-cube transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping


CLR_INVALID_LOG10_VMR = -50.0


class Prior(Protocol):
    """Protocol for scalar retrieval priors."""

    lower: float
    upper: float

    def transform(self, unit_value: float) -> float:
        """Map a unit-cube value into physical parameter space."""

    def log_probability(self, value: float) -> float:
        """Return the normalized log prior probability density."""

    def gaussian_approximation(self) -> tuple[float, float]:
        """Return a centre and scale for diagnostic optimal estimation."""


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

    def gaussian_approximation(self) -> tuple[float, float]:
        """Approximate the bounded prior by a broad Gaussian."""

        return float(0.5 * (self.lower + self.upper)), float(0.5 * (self.upper - self.lower))


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

    def gaussian_approximation(self) -> tuple[float, float]:
        """Approximate the log prior around its geometric median."""

        center = self.transform(0.5)
        scale = 0.5 * (self.transform(0.84) - self.transform(0.16))
        return float(center), float(scale)


@dataclass(frozen=True)
class CenteredLogRatioPrior:
    """Marker for one member of a joint centred-log-ratio prior.

    A CLR composition with ``N + 1`` categories has only ``N`` independent
    coordinates.  ROBERT therefore transforms all parameters sharing a
    ``group`` together in :class:`RetrievalParameterSet`; the remaining
    composition category is supplied by the chemistry model's background
    fill.  ``lower`` is the minimum allowed log10 mixing ratio and ``upper``
    must be zero, matching the Benneke--Seager / POSEIDON convention.
    """

    lower: float = -12.0
    upper: float = 0.0
    group: str = "composition"

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        group = str(self.group).strip()
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise RobertValidationError("CLR prior bounds must be finite")
        if lower >= 0.0 or upper != 0.0:
            raise RobertValidationError(
                "CLR prior requires a negative lower log10 VMR bound and upper=0"
            )
        if not group:
            raise RobertValidationError("CLR prior group must not be empty")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "group", group)

    def transform(self, unit_value: float) -> float:
        raise RobertValidationError(
            "CLR priors are joint transforms; use RetrievalParameterSet.transform"
        )

    def log_probability(self, value: float) -> float:
        number = float(value)
        return 0.0 if self.lower <= number <= self.upper else float("-inf")

    def gaussian_approximation(self) -> tuple[float, float]:
        """Return a bounded diagnostic approximation.

        Optimal estimation is rejected for configured CLR retrievals because
        this scalar approximation does not encode the joint simplex geometry.
        """

        return float(0.5 * (self.lower + self.upper)), float(
            0.5 * (self.upper - self.lower)
        )


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
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def midpoint(self) -> float:
        """Prior median obtained from the unit-cube midpoint."""

        center, _ = self.prior.gaussian_approximation()
        return center

    @property
    def approximate_standard_deviation(self) -> float:
        """Robust Gaussian scale inferred from the central 68% prior interval."""

        _, scale = self.prior.gaussian_approximation()
        return scale

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
        for group, indices in self._clr_groups.items():
            priors = [self.parameters[index].prior for index in indices]
            reference = priors[0]
            if any(
                prior.lower != reference.lower or prior.upper != reference.upper
                for prior in priors[1:]
            ):
                raise RobertValidationError(
                    f"CLR prior group '{group}' must use identical bounds"
                )

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

    @property
    def _clr_groups(self) -> dict[str, tuple[int, ...]]:
        groups: dict[str, list[int]] = {}
        for index, parameter in enumerate(self.parameters):
            if isinstance(parameter.prior, CenteredLogRatioPrior):
                groups.setdefault(parameter.prior.group, []).append(index)
        return {name: tuple(indices) for name, indices in groups.items()}

    def midpoint_vector(self) -> NDArray[np.float64]:
        """Return the prior-midpoint vector."""

        return self.transform(np.full(self.ndim, 0.5, dtype=float))

    def transform(self, cube: ArrayLike) -> NDArray[np.float64]:
        """Map a unit-cube vector into parameter space."""

        values = _vector(cube, self.ndim, "unit-cube vector")
        transformed = np.empty(self.ndim, dtype=float)
        clr_indices = {
            index for indices in self._clr_groups.values() for index in indices
        }
        for index, (parameter, value) in enumerate(
            zip(self.parameters, values, strict=True)
        ):
            if index not in clr_indices:
                transformed[index] = parameter.prior.transform(float(value))
        for indices in self._clr_groups.values():
            prior = self.parameters[indices[0]].prior
            transformed[list(indices)] = centered_log_ratio_prior_transform(
                values[list(indices)],
                lower_log10_vmr=prior.lower,
            )
        transformed.setflags(write=False)
        return transformed

    def vector_to_mapping(self, vector: ArrayLike) -> dict[str, float]:
        """Convert a vector into a parameter-name mapping."""

        values = _vector(vector, self.ndim, "parameter vector")
        for group, indices in self._clr_groups.items():
            group_values = values[list(indices)]
            prior = self.parameters[indices[0]].prior
            if not _valid_centered_log_ratio_vmr(group_values, prior.lower):
                raise RobertValidationError(
                    f"CLR prior group '{group}' lies outside the composition simplex"
                )
        return {name: float(value) for name, value in zip(self.names, values, strict=True)}

    def log_prior_from_vector(self, vector: ArrayLike) -> float:
        """Return summed log prior probability for a vector."""

        values = _vector(vector, self.ndim, "parameter vector")
        self.vector_to_mapping(values)
        logp = 0.0
        for parameter, value in zip(self.parameters, values, strict=True):
            current = parameter.prior.log_probability(float(value))
            if not np.isfinite(current):
                return float("-inf")
            logp += current
        return float(logp)


def centered_log_ratio_prior_transform(
    unit_values: ArrayLike,
    *,
    lower_log10_vmr: float = -12.0,
) -> NDArray[np.float64]:
    """Map ``N`` unit coordinates to ``N`` log10 VMRs for ``N+1`` gases.

    The omitted first gas is recovered by the chemistry model from closure.
    Invalid samples use POSEIDON's ``-50`` sentinel; parameter mapping then
    rejects them before the forward model is evaluated.  The equations follow
    POSEIDON's ``CLR_Prior`` implementation while allowing any negative lower
    log10 mixing-ratio limit.
    """

    values = np.asarray(unit_values, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise RobertValidationError("CLR prior requires at least one unit coordinate")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise RobertValidationError("CLR unit-cube values must be finite and in [0, 1]")
    limit = float(lower_log10_vmr)
    if not np.isfinite(limit) or limit >= 0.0:
        raise RobertValidationError("CLR lower log10 VMR limit must be finite and negative")

    n_free = values.size
    n_total = n_free + 1
    log_n_free = np.log(float(n_free))
    prior_lower_clr = (n_free / n_total) * (limit * np.log(10.0) + log_n_free)
    prior_upper_clr = -(n_free / n_total) * limit * np.log(10.0)
    clr_free = prior_lower_clr + values * (prior_upper_clr - prior_lower_clr)
    invalid = np.full(n_free, CLR_INVALID_LOG10_VMR, dtype=float)
    if abs(float(np.sum(clr_free))) > prior_upper_clr:
        invalid.setflags(write=False)
        return invalid

    clr = np.concatenate(([-float(np.sum(clr_free))], clr_free))
    if float(np.ptp(clr)) > -limit * np.log(10.0):
        invalid.setflags(write=False)
        return invalid
    shifted = clr - np.max(clr)
    mixing_ratios = np.exp(shifted)
    mixing_ratios /= np.sum(mixing_ratios)
    if np.any(mixing_ratios < 10.0**limit):
        invalid.setflags(write=False)
        return invalid
    result = np.log10(mixing_ratios[1:])
    result.setflags(write=False)
    return result


def _valid_centered_log_ratio_vmr(
    free_log10_vmr: NDArray[np.float64],
    lower_log10_vmr: float,
) -> bool:
    if np.any(free_log10_vmr < lower_log10_vmr) or np.any(free_log10_vmr > 0.0):
        return False
    free_total = float(np.sum(10.0**free_log10_vmr))
    minimum = 10.0**lower_log10_vmr
    return minimum <= 1.0 - free_total <= 1.0


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
