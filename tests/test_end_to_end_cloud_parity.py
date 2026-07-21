"""Tests for the independently evaluated PICASO/Virga cloud benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from examples.benchmark_end_to_end_cloud_parity import (
    _disk_quadrature,
    _make_contract,
    _relative_metrics,
    _validation_gas_tau,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_parity_contract_is_physical_and_normalized() -> None:
    contract = _make_contract(16, 36, 8)

    assert contract["gas_mass_fractions"].shape == (8, 4)
    assert contract["condensate_mass_fraction"].shape == (8,)
    assert np.all(np.diff(contract["pressure_edges_bar"]) > 0.0)
    assert np.all(np.diff(contract["wavelength_micron"]) > 0.0)
    np.testing.assert_allclose(np.sum(contract["radius_number_weights"]), 1.0)
    np.testing.assert_allclose(np.sum(contract["emission_weights"]), 1.0)

    radius = contract["radius_cm"]
    weights = contract["radius_number_weights"]
    effective_radius_micron = (
        np.sum(weights * radius**3) / np.sum(weights * radius**2) * 1.0e4
    )
    np.testing.assert_allclose(effective_radius_micron, 0.3, rtol=2.0e-3)


def test_validation_gas_opacity_uses_the_shared_state() -> None:
    contract = _make_contract(16, 8, 8)
    reference = _validation_gas_tau(contract)
    warmer = dict(contract)
    warmer["temperature_level_k"] = contract["temperature_level_k"] * 1.05
    warmer_tau = _validation_gas_tau(warmer)

    assert reference.shape == (8, 16)
    assert np.all(np.isfinite(reference))
    assert np.all(reference > 0.0)
    assert not np.array_equal(reference, warmer_tau)


def test_cloud_parity_metrics_report_signed_and_rms_residuals() -> None:
    metrics = _relative_metrics(np.array([1.0, 2.2]), np.array([1.0, 2.0]))

    np.testing.assert_allclose(metrics["max_abs_relative_difference"], 0.1)
    np.testing.assert_allclose(metrics["rms_relative_difference"], np.sqrt(0.005))
    np.testing.assert_allclose(metrics["median_relative_difference"], 0.05)


def test_cloud_parity_disk_quadrature_normalizes_intensity() -> None:
    mu, weights = _disk_quadrature(6)

    assert np.all((mu > 0.0) & (mu < 1.0))
    np.testing.assert_allclose(np.sum(weights), 1.0)


def test_versioned_cloud_parity_reference_passes_acceptance_gates() -> None:
    path = (
        ROOT
        / "data"
        / "validation"
        / "end_to_end_cloud_parity"
        / "end_to_end_cloud_parity.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["schema_version"] == 1
    assert report["metrics"]["acceptance"]["all_pass"] is True
    assert (
        report["metrics"]["matched_hg_cloudy_disk_spectrum"]["rms_relative_difference"]
        < 2.0e-6
    )
