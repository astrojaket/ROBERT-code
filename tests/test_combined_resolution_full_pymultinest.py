"""Focused tests for the full combined-resolution validation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import examples.combined_resolution_full_pymultinest as full_pymultinest
from examples.combined_resolution_full_pymultinest import (
    _compare_nested_stride_posteriors,
    _fit_statistics,
    _evidence_agreement,
    _posterior_medians,
    _posterior_intervals,
    _production_stride_decision,
    _stride_metrics,
    _weighted_quantile,
    _workflow_status,
)


def test_weighted_quantile_is_deterministic_and_handles_unit_weights() -> None:
    values = np.array([2.0, 0.0, 1.0, 3.0])
    weights = np.ones(4)
    result = _weighted_quantile(values, weights, (0.0, 0.5, 1.0))
    np.testing.assert_allclose(result, (0.0, 1.5, 3.0))


def test_weighted_quantile_uses_posterior_weights() -> None:
    result = _weighted_quantile((0.0, 1.0, 2.0), (1.0, 8.0, 1.0), (0.5,))
    np.testing.assert_allclose(result, (1.0,))


@pytest.mark.parametrize(
    "values, weights, quantiles",
    [
        ((1.0,), (-1.0,), (0.5,)),
        ((1.0, 2.0), (1.0,), (0.5,)),
        ((1.0,), (0.0,), (0.5,)),
        ((1.0,), (1.0,), (1.1,)),
    ],
)
def test_weighted_quantile_rejects_invalid_inputs(
    values: tuple[float, ...],
    weights: tuple[float, ...],
    quantiles: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        _weighted_quantile(values, weights, quantiles)


def test_posterior_intervals_have_explicit_parameter_names() -> None:
    samples = np.array(
        [
            [0.0, -1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ]
    )
    intervals = _posterior_intervals(samples, None, ("temperature_K", "abundance"))
    assert set(intervals) == {"temperature_K", "abundance"}
    assert intervals["temperature_K"][0] == 0.0
    assert intervals["temperature_K"][1] == 3.0


def test_posterior_medians_use_the_same_weighted_definition() -> None:
    samples = np.array([[0.0], [1.0], [2.0]])
    medians = _posterior_medians(samples, np.array([1.0, 8.0, 1.0]), ("x",))
    assert medians["x"] == pytest.approx(1.0)


def test_evidence_gate_uses_absolute_and_reported_error_limits() -> None:
    passed = _evidence_agreement(
        (10.0, 10.3),
        (0.1, 0.1),
        absolute_tolerance=0.5,
    )
    assert passed["passed"] is True
    assert passed["difference"] == pytest.approx(0.3)
    assert passed["allowed_difference"] == pytest.approx(0.5)

    failed = _evidence_agreement(
        (10.0, 11.0),
        (0.1, 0.1),
        absolute_tolerance=0.5,
    )
    assert failed["passed"] is False


def test_stride_metrics_are_relative_to_the_stride_one_reference() -> None:
    result = _stride_metrics(
        np.ones(3),
        np.array([1.0, 1.001, 0.99]),
    )
    assert result["rms_relative_error"] == pytest.approx(
        np.sqrt((0.001**2 + 0.01**2) / 3.0)
    )
    assert result["max_absolute_relative_error"] == pytest.approx(0.01)


def test_stride_metrics_reject_grid_mismatch() -> None:
    with pytest.raises(ValueError, match="share the observation grid"):
        _stride_metrics(np.ones(2), np.ones(3))


def test_fit_statistics_use_projected_inputs_and_projection_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a regression to raw Gaussian residuals for the HRS data set."""

    class ForbiddenGaussianLikelihood:
        def __init__(self) -> None:
            raise AssertionError("fit statistics must use the prepared likelihood")

    monkeypatch.setattr(
        full_pymultinest,
        "GaussianLikelihood",
        ForbiddenGaussianLikelihood,
    )

    low_dataset = SimpleNamespace(
        name="low_resolution",
        observation=SimpleNamespace(n_points=5),
    )
    high_dataset = SimpleNamespace(
        name="high_resolution",
        observation=SimpleNamespace(n_points=5),
    )
    observations = SimpleNamespace(datasets=(low_dataset, high_dataset))
    spectra = {"low_resolution": object(), "high_resolution": object()}
    parameters = {"temperature_K": 1500.0}

    class PreparedMixedLikelihood:
        likelihoods = {
            "low_resolution": SimpleNamespace(projection=None),
            "high_resolution": SimpleNamespace(
                projection=SimpleNamespace(rank=2),
            ),
        }

        def effective_inputs_by_dataset(
            self,
            prediction: object,
            selected_observations: object,
            selected_parameters: object,
        ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
            assert prediction is spectra
            assert selected_observations is observations
            assert selected_parameters is parameters
            return {
                "low_resolution": (
                    np.zeros(5),
                    np.zeros(5),
                    np.ones(5),
                ),
                # These are already projected, whitened HRS vectors.
                "high_resolution": (
                    np.zeros(5),
                    np.ones(5),
                    np.ones(5),
                ),
            }

    problem = SimpleNamespace(ndim=4, likelihood=PreparedMixedLikelihood())
    statistics = _fit_statistics(problem, observations, spectra, parameters)

    assert statistics["chi_square"] == pytest.approx(5.0)
    assert statistics["reduced_chi_square"] == pytest.approx(1.25)
    assert statistics["raw_point_count"] == 10
    assert statistics["effective_point_count"] == 10
    assert statistics["effective_point_count_by_dataset"] == {
        "low_resolution": 5,
        "high_resolution": 5,
    }
    assert statistics["removed_basis_rank"] == 2
    assert statistics["removed_basis_rank_by_dataset"] == {
        "low_resolution": 0,
        "high_resolution": 2,
    }
    assert statistics["degrees_of_freedom"] == 4


def test_fit_statistics_prefer_exact_prepared_chi_square_and_retained_rank() -> None:
    low_dataset = SimpleNamespace(
        name="low_resolution",
        observation=SimpleNamespace(n_points=5),
    )
    high_dataset = SimpleNamespace(
        name="high_resolution",
        observation=SimpleNamespace(n_points=5),
    )
    observations = SimpleNamespace(datasets=(low_dataset, high_dataset))
    spectra = {"low_resolution": object(), "high_resolution": object()}
    parameters = {"temperature_K": 1500.0}

    class ExactPreparedLikelihood:
        likelihoods = {
            "low_resolution": SimpleNamespace(projection=None),
            "high_resolution": SimpleNamespace(projection=SimpleNamespace(rank=2)),
        }

        def effective_inputs_by_dataset(self, prediction, selected_observations, selected_parameters):
            del prediction, selected_observations, selected_parameters
            return {
                "low_resolution": (np.zeros(5), np.zeros(5), np.ones(5)),
                # Deliberately different from the exact prepared statistic.
                "high_resolution": (np.zeros(5), np.full(5, 100.0), np.ones(5)),
            }

        def chi_square_by_dataset(self, prediction, selected_observations, selected_parameters):
            del prediction, selected_observations, selected_parameters
            return {"low_resolution": 1.0, "high_resolution": 3.0}

        def effective_residual_rank_by_dataset(self, prediction, selected_observations, selected_parameters):
            del prediction, selected_observations, selected_parameters
            return {"low_resolution": 5, "high_resolution": 3}

    problem = SimpleNamespace(ndim=1, likelihood=ExactPreparedLikelihood())
    statistics = _fit_statistics(problem, observations, spectra, parameters)

    assert statistics["chi_square"] == pytest.approx(4.0)
    assert statistics["chi_square_by_dataset"] == {
        "low_resolution": 1.0,
        "high_resolution": 3.0,
    }
    assert statistics["retained_residual_rank_by_dataset"] == {
        "low_resolution": 5,
        "high_resolution": 3,
    }
    assert statistics["removed_basis_rank_by_dataset"] == {
        "low_resolution": 0,
        "high_resolution": 2,
    }
    assert statistics["degrees_of_freedom"] == 7


def test_nested_stride_comparison_reports_paired_centres_intervals_and_evidence() -> None:
    def nested(median_shift: float, evidence: float) -> dict[str, object]:
        medians = {
            "temperature_K": 1500.0 + median_shift,
            "log10_h2o_vmr": -3.3,
            "log10_co_vmr": -3.0,
            "radial_velocity_km_s": 5.0 + median_shift,
        }
        intervals = {
            "temperature_K": [1450.0, 1550.0],
            "log10_h2o_vmr": [-3.5, -3.1],
            "log10_co_vmr": [-3.2, -2.8],
            "radial_velocity_km_s": [4.0, 6.0],
        }
        return {
            "gate": {"natural_convergence": True, "chi_square": True},
            "runs": [
                {
                    "seed": 4,
                    "status": "completed",
                    "posterior_medians": medians,
                    "posterior_intervals": intervals,
                    "log_evidence": evidence,
                    "log_evidence_error": 0.1,
                    "likelihood_evaluations": 100,
                    "elapsed_seconds": 1.0,
                    "peak_rss_bytes": 100,
                },
                {
                    "seed": 5,
                    "status": "completed",
                    "posterior_medians": medians,
                    "posterior_intervals": intervals,
                    "log_evidence": evidence,
                    "log_evidence_error": 0.1,
                    "likelihood_evaluations": 100,
                    "elapsed_seconds": 1.0,
                    "peak_rss_bytes": 100,
                },
            ],
        }

    result = _compare_nested_stride_posteriors(
        {
            **nested(0.0, -10.0),
            "gate": {"natural_convergence": True},
        },
        nested(0.2, -10.2),
    )
    assert result["gate"]["passed"] is True
    assert result["paired_runs"][0]["posterior_median_shift_stride_2_minus_stride_1"][
        "radial_velocity_km_s"
    ] == pytest.approx(0.2)
    assert result["paired_runs"][0]["evidence_gate_passed"] is True
    assert result["paired_runs"][0]["log_evidence_stride_2_minus_stride_1"] == pytest.approx(
        -0.2
    )
    assert result["paired_runs"][0][
        "absolute_log_evidence_difference_stride_2_minus_stride_1"
    ] == pytest.approx(0.2)
    assert (
        "evidence_difference_stride_2_minus_stride_1"
        not in result["paired_runs"][0]
    )


def _stride_record_for_decision(*, evidence_passed: bool) -> dict[str, object]:
    comparison_gate = {
        "paired_seed_runs": True,
        "stride_2_completed_runs": True,
        "stride_1_natural_convergence": True,
        "stride_2_natural_convergence": True,
        "posterior_centres_and_intervals": True,
        "evidence_agreement_with_reported_errors": evidence_passed,
        "stride_2_chi_square": True,
        "passed": evidence_passed,
    }
    paired_runs = [
        {
            "seed": seed,
            "stride_1": {"status": "completed"},
            "stride_2": {"status": "completed"},
        }
        for seed in (24680, 24681)
    ]
    return {
        "gate": {
            "spectral_accuracy": True,
            "retrieval_bias_vs_stride_1": True,
            "stride_2_truth_recovery": True,
            "stride_2_chi_square": True,
            "stride_2_oe_converged": True,
            "nested_stride_2_comparison": evidence_passed,
        },
        "nested_posterior_comparison": {
            "gate": comparison_gate,
            "paired_runs": paired_runs,
        },
        "stride_2_nested": {
            "status": "pass",
            "runs": [
                {"status": "completed", "converged": True},
                {"status": "completed", "converged": True},
            ],
        },
    }


def test_production_stride_decision_approves_complete_candidate() -> None:
    record = _stride_record_for_decision(evidence_passed=True)
    decision = _production_stride_decision(record)

    assert decision == {
        "status": "approved",
        "decision": "approve",
        "assessment_complete": True,
        "reference_stride": 1,
        "candidate_stride": 2,
        "production_stride": 2,
        "reference_stride_decision": "approve",
        "candidate_stride_decision": "approve",
        "reason": "complete paired assessment passed all stride-2 gates",
        "failed_checks": [],
    }
    status = _workflow_status(
        run_pymultinest=True,
        run_native_stride_comparison=True,
        recovery_passed=True,
        memory_passed=True,
        cpu_passed=True,
        pymultinest_passed=True,
        production_stride_decision=decision,
    )
    assert status == "pass_native_stride_1_stride_2_approved"
    assert status.startswith("pass")


def test_production_stride_decision_rejects_complete_evidence_biased_candidate() -> None:
    record = _stride_record_for_decision(evidence_passed=False)
    decision = _production_stride_decision(record)

    assert record["gate"]["nested_stride_2_comparison"] is False
    assert record["nested_posterior_comparison"]["gate"]["passed"] is False
    assert decision["status"] == "rejected"
    assert decision["decision"] == "reject"
    assert decision["assessment_complete"] is True
    assert decision["production_stride"] == 1
    assert decision["reference_stride_decision"] == "approve"
    assert decision["candidate_stride_decision"] == "reject"
    assert decision["evidence_bias"] is True
    assert "evidence agreement failed" in decision["reason"]
    assert "evidence_agreement_with_reported_errors" in decision["failed_checks"]
    status = _workflow_status(
        run_pymultinest=True,
        run_native_stride_comparison=True,
        recovery_passed=True,
        memory_passed=True,
        cpu_passed=True,
        pymultinest_passed=True,
        production_stride_decision=decision,
    )
    assert status == "pass_native_stride_1_stride_2_rejected"
    assert status.startswith("pass")


def test_production_stride_decision_rejects_incomplete_sampler_assessment() -> None:
    record = _stride_record_for_decision(evidence_passed=False)
    record["stride_2_nested"]["runs"][1]["status"] = "failed"

    decision = _production_stride_decision(record)

    assert decision["status"] == "incomplete"
    assert decision["decision"] == "incomplete"
    assert decision["assessment_complete"] is False
    assert decision["production_stride"] is None
    assert decision["candidate_stride_decision"] == "not_assessed"
    assert "stride_2_sampler_records" in decision["failed_checks"]
    assert (
        _workflow_status(
            run_pymultinest=True,
            run_native_stride_comparison=True,
            recovery_passed=True,
            memory_passed=True,
            cpu_passed=True,
            pymultinest_passed=True,
            production_stride_decision=decision,
        )
        == "fail"
    )
