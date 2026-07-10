"""Tests for deterministic injection-recovery validation helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robert_exoplanets import (
    SpectralGrid,
    Spectrum,
    evaluate_injection_recovery,
    inject_spectrum,
    write_injection_recovery_report,
)
from robert_exoplanets.core import RobertValidationError


def _model_spectrum() -> Spectrum:
    return Spectrum(
        spectral_grid=SpectralGrid(
            values=[1.0, 2.0, 3.0, 4.0],
            bin_edges=[0.5, 1.5, 2.5, 3.5, 4.5],
            unit="micron",
            role="observed",
        ),
        values=[1.0, 1.2, 1.1, 0.9],
        unit="eclipse_depth",
        observable="eclipse_depth",
    )


def test_inject_spectrum_is_seeded_and_preserves_explicit_bins() -> None:
    spectrum = _model_spectrum()

    first = inject_spectrum(spectrum, 0.1, seed=42, instrument="synthetic")
    second = inject_spectrum(spectrum, 0.1, seed=42, instrument="synthetic")
    different = inject_spectrum(spectrum, 0.1, seed=43, instrument="synthetic")

    np.testing.assert_array_equal(first.flux, second.flux)
    assert not np.array_equal(first.flux, different.flux)
    np.testing.assert_allclose(first.wavelength_bin_edges, spectrum.spectral_grid.bin_edges)
    np.testing.assert_allclose(first.uncertainty, np.full(4, 0.1))
    assert first.metadata["injection_seed"] == "42"


def test_inject_spectrum_rejects_invalid_uncertainty_and_seed() -> None:
    spectrum = _model_spectrum()

    with pytest.raises(RobertValidationError, match="seed"):
        inject_spectrum(spectrum, 0.1, seed=-1)
    with pytest.raises(RobertValidationError, match="positive"):
        inject_spectrum(spectrum, [0.1, 0.0, 0.1, 0.1], seed=1)


def test_evaluate_injection_recovery_reports_parameter_and_fit_passes(tmp_path) -> None:
    spectrum = _model_spectrum()
    observation = inject_spectrum(spectrum, 0.1, seed=7, noise_scale=0.0)

    report = evaluate_injection_recovery(
        case_name="linear-injection",
        truth={"offset": 1.0},
        estimates={"offset": 1.02},
        absolute_tolerances={"offset": 0.05},
        observation=observation,
        best_fit_spectrum=spectrum,
        seed=7,
        parameter_order=("offset",),
        posterior_covariance=[[0.01]],
        reduced_chi_square_bounds=(0.0, 1.0),
        metadata={"fixture": "unit-test"},
    )

    assert report.passed
    assert report.parameters_passed
    assert report.fit_passed
    assert report.parameter_recoveries[0].standardized_error == pytest.approx(0.2)
    assert report.reduced_chi_square == 0.0

    path = write_injection_recovery_report(report, tmp_path / "recovery.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["passed"] is True
    assert payload["metadata"] == {"fixture": "unit-test"}


def test_evaluate_injection_recovery_fails_outside_parameter_tolerance() -> None:
    spectrum = _model_spectrum()
    observation = inject_spectrum(spectrum, 0.1, seed=9, noise_scale=0.0)

    report = evaluate_injection_recovery(
        case_name="failed-injection",
        truth={"offset": 1.0},
        estimates={"offset": 1.2},
        absolute_tolerances={"offset": 0.05},
        observation=observation,
        best_fit_spectrum=spectrum,
        seed=9,
        reduced_chi_square_bounds=(0.0, 1.0),
    )

    assert not report.passed
    assert not report.parameters_passed
    assert report.fit_passed


def test_evaluate_injection_recovery_requires_exact_parameter_keys() -> None:
    spectrum = _model_spectrum()
    observation = inject_spectrum(spectrum, 0.1, seed=1, noise_scale=0.0)

    with pytest.raises(RobertValidationError, match="keys"):
        evaluate_injection_recovery(
            case_name="bad-keys",
            truth={"offset": 1.0},
            estimates={"other": 1.0},
            absolute_tolerances={"offset": 0.1},
            observation=observation,
            best_fit_spectrum=spectrum,
            seed=1,
            reduced_chi_square_bounds=(0.0, 1.0),
        )


def test_evaluate_injection_recovery_requires_inference_convergence() -> None:
    spectrum = _model_spectrum()
    observation = inject_spectrum(spectrum, 0.1, seed=3, noise_scale=0.0)

    report = evaluate_injection_recovery(
        case_name="nonconverged-injection",
        truth={"offset": 1.0},
        estimates={"offset": 1.0},
        absolute_tolerances={"offset": 0.1},
        observation=observation,
        best_fit_spectrum=spectrum,
        seed=3,
        inference_converged=False,
        reduced_chi_square_bounds=(0.0, 1.0),
    )

    assert report.parameters_passed
    assert report.fit_passed
    assert not report.inference_converged
    assert not report.passed
