"""Common sampler result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping


@dataclass(frozen=True)
class NestedSamplerResult:
    """Sampler-independent nested-sampling result summary."""

    method: str
    parameter_names: tuple[str, ...]
    samples: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None
    log_evidence: float | None = None
    log_evidence_error: float | None = None
    best_fit_parameters: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    converged: bool = False
    message: str = "nested sampling did not report convergence"

    def __post_init__(self) -> None:
        samples = _readonly_array(self.samples, "samples", 2)
        log_likelihood = _readonly_array(self.log_likelihood, "log_likelihood", 1)
        if samples.shape[0] != log_likelihood.size:
            raise RobertValidationError("samples and log_likelihood must have matching rows")
        weights = None
        if self.weights is not None:
            weights = _readonly_array(self.weights, "weights", 1)
            if weights.size != samples.shape[0]:
                raise RobertValidationError("weights and samples must have matching rows")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "best_fit_parameters",
            immutable_mapping({str(key): float(value) for key, value in self.best_fit_parameters.items()}),
        )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


def validate_mpi_world_size(expected: int | None) -> None:
    """Reject launcher failures before a sampler opens shared output files."""

    if expected is None:
        return
    if isinstance(expected, bool) or int(expected) < 1:
        raise RobertConfigError("mpi_nprocs must be a positive integer when provided")
    requested = int(expected)
    try:
        from mpi4py import MPI

        actual = int(MPI.COMM_WORLD.Get_size())
    except ImportError:
        actual = 1
    if actual != requested:
        raise RobertConfigError(
            "MPI communicator size mismatch: "
            f"requested mpi_nprocs={requested}, but MPI.COMM_WORLD has size {actual}. "
            "The processes are not in one MPI communicator. Launch the Conda MPI "
            "environment with `mpirun -np ${SLURM_NTASKS} python ...`; do not "
            "continue or reuse a checkpoint created by independent writers."
        )


def _readonly_array(values: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != ndim:
        raise RobertValidationError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise RobertValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
