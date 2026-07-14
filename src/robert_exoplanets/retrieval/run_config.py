"""Typed Python configuration for complete retrieval runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from robert_exoplanets.core import RobertConfigError
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import Observation
from robert_exoplanets.likelihoods import GaussianLikelihood

from .priors import RetrievalParameterSet
from .problem import ForwardEvaluator, RetrievalProblem
from .results import RetrievalResult
from .runner import run_retrieval


@dataclass(frozen=True)
class OptimalEstimationRunConfig:
    """Settings for a diagnostic optimal-estimation run."""

    max_iterations: int = 8
    convergence_tolerance: float = 1.0e-4
    finite_difference_fraction: float = 1.0e-4
    damping: float = 0.0
    initial_state: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or int(self.max_iterations) < 1:
            raise RobertConfigError("max_iterations must be a positive integer")
        for name in ("convergence_tolerance", "finite_difference_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise RobertConfigError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        damping = float(self.damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise RobertConfigError("damping must be finite and non-negative")
        initial_state = None
        if self.initial_state is not None:
            initial_state = tuple(float(value) for value in self.initial_state)
            if not initial_state or not all(np.isfinite(initial_state)):
                raise RobertConfigError("initial_state must contain finite values")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "initial_state", initial_state)

    @property
    def method(self) -> str:
        return "optimal_estimation"

    def kwargs(self) -> dict[str, object]:
        settings: dict[str, object] = {
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "finite_difference_fraction": self.finite_difference_fraction,
            "damping": self.damping,
        }
        if self.initial_state is not None:
            settings["initial_state"] = self.initial_state
        return settings


@dataclass(frozen=True)
class UltraNestRunConfig:
    """Settings for a reproducible UltraNest run."""

    min_num_live_points: int = 400
    max_ncalls: int | None = None
    dlogz: float = 0.5
    resume: str = "resume"
    show_status: bool = True
    mpi_nprocs: int | None = None
    seed: int | None = None
    invalid_loglike_floor: float = -1.0e100
    extra_run_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        live_points = int(self.min_num_live_points)
        if isinstance(self.min_num_live_points, bool) or live_points < 1:
            raise RobertConfigError("min_num_live_points must be a positive integer")
        max_ncalls = None if self.max_ncalls is None else int(self.max_ncalls)
        if max_ncalls is not None and max_ncalls < 1:
            raise RobertConfigError("max_ncalls must be positive when provided")
        dlogz = float(self.dlogz)
        if not np.isfinite(dlogz) or dlogz <= 0.0:
            raise RobertConfigError("dlogz must be finite and positive")
        resume = str(self.resume).strip().lower()
        allowed_resume_modes = {"resume", "resume-similar", "overwrite", "subfolder"}
        if resume not in allowed_resume_modes:
            raise RobertConfigError(
                "resume must be one of: " + ", ".join(sorted(allowed_resume_modes))
            )
        mpi_nprocs = None if self.mpi_nprocs is None else int(self.mpi_nprocs)
        if mpi_nprocs is not None and mpi_nprocs < 1:
            raise RobertConfigError("mpi_nprocs must be positive when provided")
        seed = None if self.seed is None else int(self.seed)
        if seed is not None and seed < 0:
            raise RobertConfigError("seed must be non-negative when provided")
        invalid_floor = float(self.invalid_loglike_floor)
        if not np.isfinite(invalid_floor) or invalid_floor >= 0.0:
            raise RobertConfigError("invalid_loglike_floor must be finite and negative")
        reserved = {
            "min_num_live_points",
            "max_ncalls",
            "dlogz",
            "resume",
            "show_status",
            "mpi_nprocs",
            "seed",
            "invalid_loglike_floor",
            "output_dir",
        }
        overlap = reserved.intersection(self.extra_run_kwargs)
        if overlap:
            raise RobertConfigError(
                "extra_run_kwargs contains reserved settings: " + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "min_num_live_points", live_points)
        object.__setattr__(self, "max_ncalls", max_ncalls)
        object.__setattr__(self, "dlogz", dlogz)
        object.__setattr__(self, "resume", resume)
        object.__setattr__(self, "mpi_nprocs", mpi_nprocs)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "invalid_loglike_floor", invalid_floor)
        object.__setattr__(self, "extra_run_kwargs", immutable_mapping(self.extra_run_kwargs))

    @property
    def method(self) -> str:
        return "ultranest"

    def kwargs(self) -> dict[str, object]:
        return {
            "min_num_live_points": self.min_num_live_points,
            "max_ncalls": self.max_ncalls,
            "dlogz": self.dlogz,
            "resume": self.resume,
            "show_status": self.show_status,
            "mpi_nprocs": self.mpi_nprocs,
            "invalid_loglike_floor": self.invalid_loglike_floor,
            **dict(self.extra_run_kwargs),
        }


@dataclass(frozen=True)
class MultiNestRunConfig:
    """Settings for a reproducible conda-provided PyMultiNest run."""

    n_live_points: int = 400
    max_iter: int = 0
    evidence_tolerance: float = 0.5
    sampling_efficiency: float = 0.8
    resume: bool = True
    verbose: bool = True
    mpi_nprocs: int | None = None
    seed: int | None = None
    invalid_loglike_floor: float = -1.0e100
    importance_nested_sampling: bool = True
    multimodal: bool = True
    n_iter_before_update: int = 100
    extra_run_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        live_points = int(self.n_live_points)
        if isinstance(self.n_live_points, bool) or live_points < 1:
            raise RobertConfigError("n_live_points must be a positive integer")
        max_iter = int(self.max_iter)
        if isinstance(self.max_iter, bool) or max_iter < 0:
            raise RobertConfigError("max_iter must be a non-negative integer")
        evidence_tolerance = float(self.evidence_tolerance)
        if not np.isfinite(evidence_tolerance) or evidence_tolerance <= 0.0:
            raise RobertConfigError("evidence_tolerance must be finite and positive")
        sampling_efficiency = float(self.sampling_efficiency)
        if (
            not np.isfinite(sampling_efficiency)
            or sampling_efficiency <= 0.0
            or sampling_efficiency > 1.0
        ):
            raise RobertConfigError("sampling_efficiency must be in (0, 1]")
        mpi_nprocs = None if self.mpi_nprocs is None else int(self.mpi_nprocs)
        if mpi_nprocs is not None and mpi_nprocs < 1:
            raise RobertConfigError("mpi_nprocs must be positive when provided")
        seed = None if self.seed is None else int(self.seed)
        if seed is not None and seed < 0:
            raise RobertConfigError("seed must be non-negative when provided")
        invalid_floor = float(self.invalid_loglike_floor)
        if not np.isfinite(invalid_floor) or invalid_floor >= 0.0:
            raise RobertConfigError("invalid_loglike_floor must be finite and negative")
        update_interval = int(self.n_iter_before_update)
        if isinstance(self.n_iter_before_update, bool) or update_interval < 1:
            raise RobertConfigError("n_iter_before_update must be a positive integer")
        reserved = {
            "n_live_points",
            "max_iter",
            "evidence_tolerance",
            "sampling_efficiency",
            "resume",
            "verbose",
            "mpi_nprocs",
            "seed",
            "invalid_loglike_floor",
            "importance_nested_sampling",
            "multimodal",
            "n_iter_before_update",
            "output_dir",
        }
        overlap = reserved.intersection(self.extra_run_kwargs)
        if overlap:
            raise RobertConfigError(
                "extra_run_kwargs contains reserved settings: " + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "n_live_points", live_points)
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "evidence_tolerance", evidence_tolerance)
        object.__setattr__(self, "sampling_efficiency", sampling_efficiency)
        object.__setattr__(self, "mpi_nprocs", mpi_nprocs)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "invalid_loglike_floor", invalid_floor)
        object.__setattr__(self, "n_iter_before_update", update_interval)
        object.__setattr__(self, "extra_run_kwargs", immutable_mapping(self.extra_run_kwargs))

    @property
    def method(self) -> str:
        return "multinest"

    def kwargs(self) -> dict[str, object]:
        return {
            "n_live_points": self.n_live_points,
            "max_iter": self.max_iter,
            "evidence_tolerance": self.evidence_tolerance,
            "sampling_efficiency": self.sampling_efficiency,
            "resume": self.resume,
            "verbose": self.verbose,
            "mpi_nprocs": self.mpi_nprocs,
            "invalid_loglike_floor": self.invalid_loglike_floor,
            "importance_nested_sampling": self.importance_nested_sampling,
            "multimodal": self.multimodal,
            "n_iter_before_update": self.n_iter_before_update,
            **dict(self.extra_run_kwargs),
        }


InferenceRunConfig = OptimalEstimationRunConfig | UltraNestRunConfig | MultiNestRunConfig


@dataclass(frozen=True)
class RetrievalRunConfig:
    """Complete importable Python configuration for one retrieval run."""

    name: str
    observation: Observation
    parameters: RetrievalParameterSet
    forward_model: ForwardEvaluator
    inference: InferenceRunConfig
    output_dir: str | Path
    likelihood: GaussianLikelihood = field(default_factory=GaussianLikelihood)
    metadata: Mapping[str, str] = field(default_factory=dict)
    opacity_identifiers: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise RobertConfigError("retrieval run name must not be empty")
        output_dir = Path(self.output_dir).expanduser()
        if not str(output_dir):
            raise RobertConfigError("retrieval output_dir must not be empty")
        required_parameters = tuple(getattr(self.forward_model, "required_parameters", ()))
        missing = tuple(name for name in required_parameters if name not in self.parameters.names)
        if missing:
            raise RobertConfigError(
                "retrieval parameter set is missing forward-model parameters: "
                + ", ".join(missing)
            )
        opacity_identifiers = self.opacity_identifiers
        if opacity_identifiers is None:
            opacity_identifiers = getattr(self.forward_model, "opacity_identifiers", {})
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(self, "opacity_identifiers", immutable_mapping(opacity_identifiers))


def build_retrieval_problem(config: RetrievalRunConfig) -> RetrievalProblem:
    """Build a sampler-independent problem from a Python run configuration."""

    model_metadata = getattr(config.forward_model, "manifest_metadata", {})
    return RetrievalProblem(
        name=config.name,
        observation=config.observation,
        parameters=config.parameters,
        forward_model=config.forward_model,
        likelihood=config.likelihood,
        metadata={**dict(model_metadata), **dict(config.metadata)},
        opacity_identifiers=config.opacity_identifiers or {},
    )


def run_configured_retrieval(config: RetrievalRunConfig) -> RetrievalResult:
    """Build, run, and serialize a retrieval entirely from Python config."""

    problem = build_retrieval_problem(config)
    seed = (
        config.inference.seed
        if isinstance(config.inference, (UltraNestRunConfig, MultiNestRunConfig))
        else None
    )
    return run_retrieval(
        problem,
        method=config.inference.method,
        output_dir=config.output_dir,
        seed=seed,
        **config.inference.kwargs(),
    )


__all__ = [
    "InferenceRunConfig",
    "MultiNestRunConfig",
    "OptimalEstimationRunConfig",
    "RetrievalRunConfig",
    "UltraNestRunConfig",
    "build_retrieval_problem",
    "run_configured_retrieval",
]
