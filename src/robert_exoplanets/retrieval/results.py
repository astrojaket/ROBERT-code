"""Sampler-independent retrieval result serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from robert_exoplanets.core import RobertDataError, RobertValidationError
from robert_exoplanets.core._immutability import immutable_mapping

from .manifest import RunManifest
from .optimal_estimation import OptimalEstimationResult
from .samplers import NestedSamplerResult

RETRIEVAL_RESULT_SCHEMA_VERSION = "1.0"
RETRIEVAL_RESULT_FILENAME = "result.json"
RETRIEVAL_ARRAYS_FILENAME = "result_arrays.npz"

InferenceResult = OptimalEstimationResult | NestedSamplerResult


@dataclass(frozen=True)
class RetrievalResult:
    """Stable high-level result returned by :func:`run_retrieval`."""

    method: str
    parameter_names: tuple[str, ...]
    best_fit_parameters: Mapping[str, float]
    best_fit_log_likelihood: float
    converged: bool
    message: str
    manifest: RunManifest
    output_dir: Path
    inference_result: InferenceResult = field(repr=False)
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = RETRIEVAL_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.method or not self.parameter_names or not self.schema_version:
            raise RobertValidationError("retrieval result method, parameters, and schema version are required")
        if not np.isfinite(self.best_fit_log_likelihood):
            raise RobertValidationError("best_fit_log_likelihood must be finite")
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(
            self,
            "best_fit_parameters",
            immutable_mapping({str(key): float(value) for key, value in self.best_fit_parameters.items()}),
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def log_likelihood(self):
        """Compatibility view of method-specific log-likelihood output."""

        return self.inference_result.log_likelihood

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def result_path(self) -> Path:
        return self.output_dir / RETRIEVAL_RESULT_FILENAME

    @property
    def log_evidence(self) -> float | None:
        if isinstance(self.inference_result, NestedSamplerResult):
            return self.inference_result.log_evidence
        return None

    @property
    def log_evidence_error(self) -> float | None:
        if isinstance(self.inference_result, NestedSamplerResult):
            return self.inference_result.log_evidence_error
        return None

    def to_mapping(self) -> dict[str, object]:
        """Return the portable JSON summary for this result."""

        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "method": self.method,
            "problem_name": self.manifest.problem_name,
            "config_hash": self.manifest.config_hash,
            "robert_version": self.manifest.robert_version,
            "git_commit": self.manifest.git_commit,
            "opacity_identifiers": dict(self.manifest.opacity_identifiers),
            "parameter_names": list(self.parameter_names),
            "best_fit_parameters": dict(self.best_fit_parameters),
            "best_fit_log_likelihood": self.best_fit_log_likelihood,
            "converged": self.converged,
            "message": self.message,
            "manifest": self.manifest_path.name,
            "arrays": RETRIEVAL_ARRAYS_FILENAME,
            "metadata": dict(self.metadata),
        }
        if isinstance(self.inference_result, OptimalEstimationResult):
            result.update(
                {
                    "cost": self.inference_result.cost,
                    "n_iterations": self.inference_result.n_iterations,
                }
            )
        else:
            result.update(
                {
                    "log_evidence": self.inference_result.log_evidence,
                    "log_evidence_error": self.inference_result.log_evidence_error,
                }
            )
        return result


def build_retrieval_result(
    inference_result: InferenceResult,
    *,
    manifest: RunManifest,
    output_dir: str | Path,
) -> RetrievalResult:
    """Wrap a method-specific result in the stable result schema."""

    result_metadata = {
        "problem": manifest.problem_name,
        "config_hash": manifest.config_hash,
    }
    if isinstance(inference_result, OptimalEstimationResult):
        method = "optimal_estimation"
        best_fit = inference_result.best_fit_parameters
        best_loglike = inference_result.log_likelihood
        converged = inference_result.converged
        message = inference_result.message
    else:
        method = inference_result.method
        best_fit = inference_result.best_fit_parameters
        if inference_result.log_likelihood.size == 0:
            raise RobertValidationError("nested-sampling result contains no likelihood samples")
        best_loglike = float(np.max(inference_result.log_likelihood))
        converged = inference_result.converged
        message = inference_result.message
        result_metadata.update(inference_result.metadata)
    return RetrievalResult(
        method=method,
        parameter_names=inference_result.parameter_names,
        best_fit_parameters=best_fit,
        best_fit_log_likelihood=float(best_loglike),
        converged=converged,
        message=message,
        manifest=manifest,
        output_dir=Path(output_dir),
        inference_result=inference_result,
        metadata=result_metadata,
    )


def write_retrieval_result(result: RetrievalResult) -> tuple[Path, Path]:
    """Write JSON summary and method-specific numerical arrays."""

    result.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = result.result_path
    arrays_path = result.output_dir / RETRIEVAL_ARRAYS_FILENAME
    if isinstance(result.inference_result, OptimalEstimationResult):
        arrays = {
            "state_vector": result.inference_result.state_vector,
            "covariance": result.inference_result.covariance,
            "averaging_kernel": result.inference_result.averaging_kernel,
        }
        for name in (
            "jacobian",
            "gain_matrix",
            "measurement_error_covariance",
            "smoothing_error_covariance",
        ):
            value = getattr(result.inference_result, name)
            if value is not None:
                arrays[name] = value
    else:
        arrays = {
            "samples": result.inference_result.samples,
            "log_likelihood": result.inference_result.log_likelihood,
        }
        if result.inference_result.weights is not None:
            arrays["weights"] = result.inference_result.weights
    try:
        np.savez(arrays_path, **arrays)
        result_path.write_text(
            json.dumps(result.to_mapping(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RobertDataError(f"failed to write retrieval result under {result.output_dir}") from exc
    return result_path, arrays_path


__all__ = [
    "RETRIEVAL_ARRAYS_FILENAME",
    "RETRIEVAL_RESULT_FILENAME",
    "RETRIEVAL_RESULT_SCHEMA_VERSION",
    "RetrievalResult",
    "build_retrieval_result",
    "write_retrieval_result",
]
