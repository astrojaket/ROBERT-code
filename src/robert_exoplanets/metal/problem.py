"""Sampler-facing retrieval problem for a separately compiled Metal graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.retrieval.priors import RetrievalParameterSet
from .forward import compile_metal_likelihood


@dataclass(frozen=True)
class MetalRetrievalProblem:
    """Explicit Metal alternative satisfying ROBERT's sampler-facing contract.

    Only the parameter vector and final scalar cross the host boundary.  The
    supplied callable must implement the full scientific forward model and
    likelihood with JAX arrays; it is compiled once by :meth:`from_device_graph`.
    """

    name: str
    parameters: RetrievalParameterSet
    compiled_loglike: Callable[[Any], Any]
    runtime: Any
    likelihood: Any = None
    invalid_loglike: float = float("-inf")
    metadata: Mapping[str, str] = field(default_factory=dict)
    opacity_identifiers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError(
                "Metal retrieval problem name must not be empty"
            )
        metadata = {
            **self.runtime.metadata,
            "numerical_backend": f"{self.runtime.metadata['backend']}-float32",
            "cpu_fallback": "forbidden",
            "supported_retrieval_methods": "multinest",
            **dict(self.metadata),
        }
        object.__setattr__(self, "metadata", immutable_mapping(metadata))
        object.__setattr__(
            self,
            "opacity_identifiers",
            immutable_mapping(self.opacity_identifiers),
        )

    @classmethod
    def from_device_graph(
        cls,
        *,
        name: str,
        parameters: RetrievalParameterSet,
        parameter_to_loglike: Callable[[Any], Any],
        runtime: Any,
        likelihood: Any = None,
        invalid_loglike: float = float("-inf"),
        metadata: Mapping[str, str] | None = None,
        opacity_identifiers: Mapping[str, str] | None = None,
    ) -> "MetalRetrievalProblem":
        """Compile and bind one complete device likelihood graph."""

        return cls(
            name=name,
            parameters=parameters,
            compiled_loglike=compile_metal_likelihood(parameter_to_loglike, runtime),
            runtime=runtime,
            likelihood=likelihood,
            invalid_loglike=invalid_loglike,
            metadata={} if metadata is None else metadata,
            opacity_identifiers=(
                {} if opacity_identifiers is None else opacity_identifiers
            ),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.parameters.names

    @property
    def ndim(self) -> int:
        return self.parameters.ndim

    def prior_transform(self, cube: ArrayLike) -> NDArray[np.float64]:
        return self.parameters.transform(cube)

    def parameter_mapping(self, vector: ArrayLike) -> dict[str, float]:
        return self.parameters.vector_to_mapping(vector)

    def log_likelihood_from_vector(
        self,
        vector: ArrayLike,
    ) -> float:
        """Synchronize only the final device scalar for a Python sampler."""

        values = np.asarray(vector, dtype=np.float32)
        if values.shape != (self.ndim,) or not np.all(np.isfinite(values)):
            return float(self.invalid_loglike)
        result = self.compiled_loglike(self.runtime.put(values))
        result.block_until_ready()
        device_value = float(result)
        return device_value if np.isfinite(device_value) else float(self.invalid_loglike)

    def log_prior_from_vector(self, vector: ArrayLike) -> float:
        try:
            return self.parameters.log_prior_from_vector(vector)
        except RobertValidationError:
            return float("-inf")

    def gaussian_inputs_from_vector(self, vector: ArrayLike):
        """Reject CPU finite-difference OE through an explicit API boundary."""

        del vector
        raise RobertConfigError(
            "JAX accelerator retrieval problems currently support MultiNest only; "
            "optimal estimation requires an explicit device model "
            "output/Jacobian interface"
        )

    def log_posterior_from_vector(self, vector: ArrayLike) -> float:
        prior = self.log_prior_from_vector(vector)
        if not np.isfinite(prior):
            return float("-inf")
        likelihood = self.log_likelihood_from_vector(vector)
        if not np.isfinite(likelihood):
            return float("-inf")
        return float(prior + likelihood)


__all__ = ["MetalRetrievalProblem"]
