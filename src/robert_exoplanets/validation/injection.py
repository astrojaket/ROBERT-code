"""Deterministic injection-recovery validation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from robert_exoplanets.core import RobertDataError, RobertValidationError, Spectrum
from robert_exoplanets.core._immutability import immutable_mapping
from robert_exoplanets.instruments import Observation
from robert_exoplanets.likelihoods import GaussianLikelihood

INJECTION_RECOVERY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ParameterRecovery:
    """Recovery diagnostic for one injected parameter."""

    name: str
    truth: float
    estimate: float
    absolute_error: float
    absolute_tolerance: float
    posterior_standard_deviation: float | None
    standardized_error: float | None
    passed: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("recovery parameter name must not be empty")
        finite_values = (
            self.truth,
            self.estimate,
            self.absolute_error,
            self.absolute_tolerance,
        )
        if not all(np.isfinite(value) for value in finite_values):
            raise RobertValidationError("parameter recovery values must be finite")
        if self.absolute_error < 0.0 or self.absolute_tolerance <= 0.0:
            raise RobertValidationError("recovery errors must be non-negative and tolerance positive")
        if self.posterior_standard_deviation is not None and (
            not np.isfinite(self.posterior_standard_deviation)
            or self.posterior_standard_deviation <= 0.0
        ):
            raise RobertValidationError("posterior standard deviation must be finite and positive")
        if self.standardized_error is not None and (
            not np.isfinite(self.standardized_error) or self.standardized_error < 0.0
        ):
            raise RobertValidationError("standardized recovery error must be finite and non-negative")

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable parameter diagnostic."""

        return {
            "name": self.name,
            "truth": self.truth,
            "estimate": self.estimate,
            "absolute_error": self.absolute_error,
            "absolute_tolerance": self.absolute_tolerance,
            "posterior_standard_deviation": self.posterior_standard_deviation,
            "standardized_error": self.standardized_error,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class InjectionRecoveryReport:
    """Versioned pass/fail report for an injection-recovery run."""

    case_name: str
    seed: int
    parameter_recoveries: tuple[ParameterRecovery, ...]
    chi_square: float
    reduced_chi_square: float
    reduced_chi_square_bounds: tuple[float, float]
    n_active_points: int
    n_parameters: int
    inference_converged: bool
    fit_passed: bool
    parameters_passed: bool
    passed: bool
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = INJECTION_RECOVERY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.case_name or not self.schema_version:
            raise RobertValidationError("recovery case name and schema version are required")
        if int(self.seed) < 0:
            raise RobertValidationError("recovery seed must be non-negative")
        recoveries = tuple(self.parameter_recoveries)
        if not recoveries:
            raise RobertValidationError("recovery report must contain parameter diagnostics")
        names = tuple(item.name for item in recoveries)
        if len(set(names)) != len(names):
            raise RobertValidationError("recovery parameter names must be unique")
        if not np.isfinite(self.chi_square) or self.chi_square < 0.0:
            raise RobertValidationError("recovery chi-square must be finite and non-negative")
        if not np.isfinite(self.reduced_chi_square) or self.reduced_chi_square < 0.0:
            raise RobertValidationError("reduced chi-square must be finite and non-negative")
        lower, upper = (float(value) for value in self.reduced_chi_square_bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0.0 or upper <= lower:
            raise RobertValidationError("reduced chi-square bounds must be finite and increasing")
        if self.n_active_points <= self.n_parameters or self.n_parameters < 1:
            raise RobertValidationError("recovery report requires positive residual degrees of freedom")
        expected_fit_passed = lower <= self.reduced_chi_square <= upper
        expected_parameters_passed = all(item.passed for item in recoveries)
        expected_passed = bool(self.inference_converged) and expected_fit_passed and expected_parameters_passed
        if self.fit_passed != expected_fit_passed:
            raise RobertValidationError("fit_passed is inconsistent with reduced chi-square bounds")
        if self.parameters_passed != expected_parameters_passed:
            raise RobertValidationError("parameters_passed is inconsistent with parameter diagnostics")
        if self.passed != expected_passed:
            raise RobertValidationError("recovery passed flag is inconsistent with component criteria")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "parameter_recoveries", recoveries)
        object.__setattr__(self, "reduced_chi_square_bounds", (lower, upper))
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def to_mapping(self) -> dict[str, object]:
        """Return a portable report mapping."""

        return {
            "schema_version": self.schema_version,
            "case_name": self.case_name,
            "seed": self.seed,
            "passed": self.passed,
            "parameters_passed": self.parameters_passed,
            "fit_passed": self.fit_passed,
            "inference_converged": self.inference_converged,
            "chi_square": self.chi_square,
            "reduced_chi_square": self.reduced_chi_square,
            "reduced_chi_square_bounds": list(self.reduced_chi_square_bounds),
            "n_active_points": self.n_active_points,
            "n_parameters": self.n_parameters,
            "parameter_recoveries": [item.to_mapping() for item in self.parameter_recoveries],
            "metadata": dict(self.metadata),
        }


def inject_spectrum(
    spectrum: Spectrum,
    uncertainty: ArrayLike,
    *,
    seed: int,
    noise_scale: float = 1.0,
    instrument: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Observation:
    """Create a reproducible Gaussian-noise observation from a model spectrum."""

    normalized_seed = int(seed)
    if normalized_seed < 0:
        raise RobertValidationError("injection seed must be non-negative")
    normalized_scale = float(noise_scale)
    if not np.isfinite(normalized_scale) or normalized_scale < 0.0:
        raise RobertValidationError("noise_scale must be finite and non-negative")
    uncertainty_array = np.asarray(uncertainty, dtype=float)
    if uncertainty_array.ndim == 0:
        uncertainty_array = np.full(spectrum.values.shape, float(uncertainty_array))
    if uncertainty_array.shape != spectrum.values.shape:
        raise RobertValidationError("injection uncertainty must be scalar or match the spectrum")
    if not np.all(np.isfinite(uncertainty_array)) or np.any(uncertainty_array <= 0.0):
        raise RobertValidationError("injection uncertainty must be finite and positive")
    noise = np.random.default_rng(normalized_seed).normal(
        loc=0.0,
        scale=uncertainty_array * normalized_scale,
    )
    observation_metadata = {
        "source": "synthetic_injection",
        "injection_seed": str(normalized_seed),
        "noise_scale": f"{normalized_scale:.12g}",
        **({} if metadata is None else dict(metadata)),
    }
    return Observation(
        wavelength=spectrum.spectral_grid.values,
        wavelength_bin_edges=spectrum.spectral_grid.bin_edges,
        flux=spectrum.values + noise,
        uncertainty=uncertainty_array,
        wavelength_unit=spectrum.spectral_grid.unit,
        flux_unit=spectrum.unit,
        observable=spectrum.observable,
        instrument=instrument,
        metadata=observation_metadata,
    )


def evaluate_injection_recovery(
    *,
    case_name: str,
    truth: Mapping[str, float],
    estimates: Mapping[str, float],
    absolute_tolerances: Mapping[str, float],
    observation: Observation,
    best_fit_spectrum: Spectrum,
    seed: int,
    parameter_order: Sequence[str] | None = None,
    posterior_covariance: ArrayLike | None = None,
    inference_converged: bool = True,
    reduced_chi_square_bounds: tuple[float, float] = (0.5, 1.5),
    metadata: Mapping[str, str] | None = None,
) -> InjectionRecoveryReport:
    """Evaluate explicit parameter and fit-quality recovery criteria."""

    names = tuple(truth) if parameter_order is None else tuple(str(name) for name in parameter_order)
    if not names or len(set(names)) != len(names):
        raise RobertValidationError("parameter_order must contain unique parameter names")
    expected_names = set(names)
    for label, values in (
        ("truth", truth),
        ("estimates", estimates),
        ("absolute_tolerances", absolute_tolerances),
    ):
        if set(values) != expected_names:
            raise RobertValidationError(f"{label} keys must match parameter_order")

    standard_deviations: np.ndarray | None = None
    if posterior_covariance is not None:
        covariance = np.asarray(posterior_covariance, dtype=float)
        if covariance.shape != (len(names), len(names)):
            raise RobertValidationError("posterior covariance must match parameter_order")
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance,
            covariance.T,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise RobertValidationError("posterior covariance must be finite and symmetric")
        diagonal = np.diag(covariance)
        if np.any(diagonal <= 0.0):
            raise RobertValidationError("posterior covariance diagonal must be positive")
        standard_deviations = np.sqrt(diagonal)

    recoveries: list[ParameterRecovery] = []
    for index, name in enumerate(names):
        truth_value = float(truth[name])
        estimate = float(estimates[name])
        tolerance = float(absolute_tolerances[name])
        if not np.isfinite(truth_value) or not np.isfinite(estimate):
            raise RobertValidationError("truth and estimate values must be finite")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise RobertValidationError("absolute recovery tolerances must be finite and positive")
        error = abs(estimate - truth_value)
        standard_deviation = None if standard_deviations is None else float(standard_deviations[index])
        standardized_error = None if standard_deviation is None else error / standard_deviation
        recoveries.append(
            ParameterRecovery(
                name=name,
                truth=truth_value,
                estimate=estimate,
                absolute_error=error,
                absolute_tolerance=tolerance,
                posterior_standard_deviation=standard_deviation,
                standardized_error=standardized_error,
                passed=error <= tolerance,
            )
        )

    model, data, uncertainty = GaussianLikelihood().effective_inputs(best_fit_spectrum, observation)
    chi_square = float(np.sum(np.square((data - model) / uncertainty)))
    n_active = int(data.size)
    degrees_of_freedom = n_active - len(names)
    if degrees_of_freedom <= 0:
        raise RobertValidationError("injection recovery requires positive residual degrees of freedom")
    reduced_chi_square = chi_square / degrees_of_freedom
    lower, upper = (float(value) for value in reduced_chi_square_bounds)
    fit_passed = lower <= reduced_chi_square <= upper
    parameters_passed = all(item.passed for item in recoveries)
    return InjectionRecoveryReport(
        case_name=case_name,
        seed=seed,
        parameter_recoveries=tuple(recoveries),
        chi_square=chi_square,
        reduced_chi_square=reduced_chi_square,
        reduced_chi_square_bounds=(lower, upper),
        n_active_points=n_active,
        n_parameters=len(names),
        inference_converged=bool(inference_converged),
        fit_passed=fit_passed,
        parameters_passed=parameters_passed,
        passed=bool(inference_converged) and fit_passed and parameters_passed,
        metadata={} if metadata is None else metadata,
    )


def write_injection_recovery_report(
    report: InjectionRecoveryReport,
    path: str | Path,
) -> Path:
    """Atomically write a validation report as JSON."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report.to_mapping(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        raise RobertDataError(f"failed to write injection-recovery report: {output_path}") from exc
    return output_path


__all__ = [
    "INJECTION_RECOVERY_SCHEMA_VERSION",
    "InjectionRecoveryReport",
    "ParameterRecovery",
    "evaluate_injection_recovery",
    "inject_spectrum",
    "write_injection_recovery_report",
]
