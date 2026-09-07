"""Stable structural contracts used by retrieval backends.

The retrieval implementations in ROBERT use different forward-model and
likelihood representations.  Sampler adapters must not depend on those
representations.  This module owns the small structural contract that a
sampler needs and the additional contract used by optimal estimation.

The protocols are deliberately independent of concrete retrieval problem
classes.  A CPU problem, a named multi-dataset problem, a heterogeneous HRS
problem, and a device-compiled problem can therefore use the same adapters.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class RetrievalPriorProtocol(Protocol):
    """Read-only prior information needed for manifests and OE setup."""

    @property
    def lower(self) -> float:
        """Lower support boundary."""

    @property
    def upper(self) -> float:
        """Upper support boundary."""

    def gaussian_approximation(self) -> tuple[float, float]:
        """Return the diagnostic centre and scale of the prior."""


class RetrievalParameterProtocol(Protocol):
    """One named parameter in a retrieval parameter set."""

    @property
    def name(self) -> str:
        """Stable parameter name."""

    @property
    def prior(self) -> RetrievalPriorProtocol:
        """Prior assigned to the parameter."""

    @property
    def label(self) -> str | None:
        """Optional display label."""

    @property
    def unit(self) -> str | None:
        """Optional physical unit."""

    @property
    def metadata(self) -> Mapping[str, str]:
        """Immutable parameter metadata."""

    @property
    def approximate_standard_deviation(self) -> float:
        """Return the scale used by diagnostic optimal estimation."""


class RetrievalParameterSetProtocol(Protocol):
    """Parameter-set operations shared by retrieval problem implementations."""

    @property
    def parameters(self) -> Sequence[RetrievalParameterProtocol]:
        """Parameters in sampler vector order."""

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names in sampler vector order."""

    @property
    def ndim(self) -> int:
        """Number of parameters."""

    @property
    def bounds(self) -> Sequence[tuple[float, float]]:
        """Finite bounds in sampler vector order."""

    def transform(self, cube: ArrayLike) -> FloatArray:
        """Map a unit-cube vector into physical parameter space."""

    def vector_to_mapping(self, vector: ArrayLike) -> Mapping[str, float]:
        """Convert a vector into named physical parameters."""

    def midpoint_vector(self) -> FloatArray:
        """Return the prior-midpoint parameter vector."""


@runtime_checkable
class SamplerRetrievalProblem(Protocol):
    """Minimal contract consumed by nested-sampler adapters.

    ``prior_transform`` receives a unit-cube vector and returns the physical
    parameter vector.  ``log_likelihood_from_vector`` receives that physical
    vector and returns one finite scalar, or the problem's configured invalid
    value.  The adapter does not need to know which forward model or likelihood
    produced the value.
    """

    @property
    def name(self) -> str:
        """Stable retrieval name."""

    @property
    def parameters(self) -> RetrievalParameterSetProtocol:
        """Read-only retrieval parameter set."""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names in sampler vector order."""

    @property
    def ndim(self) -> int:
        """Number of retrieval dimensions."""

    @property
    def likelihood(self) -> object:
        """Likelihood object owned by the retrieval problem."""

    @property
    def metadata(self) -> Mapping[str, str]:
        """Immutable problem metadata."""

    @property
    def opacity_identifiers(self) -> Mapping[str, str]:
        """Opacity identities recorded in the run manifest."""

    def prior_transform(self, cube: ArrayLike) -> FloatArray:
        """Map a unit-cube vector into physical parameter space."""

    def parameter_mapping(self, vector: ArrayLike) -> Mapping[str, float]:
        """Convert a physical parameter vector into named values."""

    def log_likelihood_from_vector(self, vector: ArrayLike) -> float:
        """Evaluate the scalar likelihood for a physical parameter vector."""


@runtime_checkable
class OptimalEstimationProblem(SamplerRetrievalProblem, Protocol):
    """Additional capability required by the NumPy OE solver.

    The returned arrays are masked model values, data values, and effective
    independent uncertainties.  A likelihood must opt into this contract
    explicitly; a covariance or profiled likelihood cannot satisfy it by
    merely exposing a diagonal uncertainty array.
    """

    @property
    def likelihood(self) -> object:
        """Likelihood that explicitly opts into the OE contract."""

    def gaussian_inputs_from_vector(
        self,
        vector: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return model, data, and uncertainty arrays for OE."""


__all__ = [
    "OptimalEstimationProblem",
    "RetrievalParameterProtocol",
    "RetrievalParameterSetProtocol",
    "RetrievalPriorProtocol",
    "SamplerRetrievalProblem",
]
