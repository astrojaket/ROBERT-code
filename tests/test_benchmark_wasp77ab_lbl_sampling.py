"""Contract and light-math tests for the real WASP-77Ab LBL benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples import benchmark_wasp77ab_lbl_sampling as benchmark


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "data" / "wasp77ab_lbl_sampling_20260831.json"


def test_benchmark_declares_laptop_safe_resources_and_sampling_contract() -> None:
    assert benchmark.MAX_THREADS == 3
    assert benchmark.RUN_CPU_COUNT == 1
    assert len(benchmark.THREAD_VARIABLES) == 7
    assert benchmark.OPACITY_MEMORY_LIMIT_BYTES < 1024**3
    assert benchmark.PROCESS_MEMORY_LIMIT_BYTES < 1.9 * 1024**3
    assert benchmark.HRS_REFERENCE_STRIDE == 1
    assert benchmark.HRS_CANDIDATE_STRIDES == (2, 4, 6, 8)
    assert benchmark.LRS_REFERENCE_STRIDE == 10
    assert benchmark.LRS_CANDIDATE_STRIDES == (25, 50, 100, 250)
    assert benchmark.HRS_RESPONSE_STAGES == (
        "rotational-broadening",
        "gaussian-high-resolution",
        "pixel-bin-integration",
    )
    assert benchmark.LRS_BOUNDS_MICRON == (2.808, 5.168)
    assert benchmark.HRS_K_BAND_RANGE_MICRON == (1.90, 2.45)


def test_real_order_selection_is_wholly_inside_k_band() -> None:
    orders = np.array(
        [
            [1.90, 2.00],
            [2.00, 2.45],
            [1.89, 2.30],
            [2.10, 2.46],
        ]
    )
    assert benchmark._select_k_band_orders(orders) == (0, 1)


def test_fixed_state_is_vmr_only_and_hminus_is_included() -> None:
    state = benchmark._vmr_state()
    assert set(state) == {"H2O", "CO", "H-", "H", "e-", "H2", "He"}
    assert np.isclose(sum(state.values()), 1.0, rtol=0.0, atol=2.0e-15)
    assert state["H2O"] == pytest.approx(10.0 ** -4.02)
    assert state["CO"] == pytest.approx(10.0 ** -3.91)
    assert benchmark.HMINUS_CONFIG.species == ("H-", "H", "e-")
    assert benchmark.HMINUS_CONFIG.temperature_extrapolation == "raise"
    assert benchmark.HMINUS_CONFIG.spectral_extrapolation == "raise"
    atmosphere = benchmark._atmosphere()
    assert atmosphere.composition_convention == "volume_mixing_ratio"
    assert atmosphere.metadata["mass_fraction_parameters"] == "none"
    assert np.all(np.diff(atmosphere.temperature) > 0.0)


def test_normalized_hrs_metrics_ignore_absolute_scale() -> None:
    reference = np.array([1.0, 1.1, 0.9, 1.05])
    scaled = 7.0 * reference
    assert benchmark._line_shape_metrics(reference, scaled)["rms_fractional"] == pytest.approx(0.0)
    perturbed = scaled.copy()
    perturbed[2] *= 1.01
    metrics = benchmark._line_shape_metrics(reference, perturbed)
    assert metrics["rms_fractional"] > 0.0
    assert metrics["max_absolute_fractional"] > metrics["rms_fractional"]


def test_lrs_metrics_use_published_uncertainty_units() -> None:
    class Observation:
        instrument = "NRS1"
        uncertainty = np.array([0.1, 0.2])

        @property
        def n_points(self) -> int:
            return 2

    observations = (Observation(),)
    reference = {"NRS1": np.array([1.0, 2.0])}
    candidate = {"NRS1": np.array([1.1, 1.8])}
    aggregate, per_detector = benchmark._lrs_sigma_metrics(
        reference,
        candidate,
        observations,
    )
    np.testing.assert_allclose(aggregate["rms_sigma"], np.sqrt((1.0**2 + 1.0**2) / 2.0))
    assert aggregate["max_absolute_sigma"] == pytest.approx(1.0)
    assert per_detector["NRS1"]["n_points"] == 2


def test_report_is_compact_and_contains_no_spectra() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["record_type"] == "wasp77ab_real_grid_lbl_sampling_convergence"
    assert report["atmosphere"]["composition_convention"] == "volume_mixing_ratio"
    assert report["atmosphere"]["hminus_continuum"][
        "included_in_all_stride_evaluations"
    ] is True
    assert report["hrs"]["reference_stride"] == 1
    assert report["lrs"]["reference_stride"] == 10
    assert report["lrs"]["broad_stride_1_attempted"] is False
    assert report["resources"]["max_threads"] <= 3
    assert report["resources"]["process_rss_limit_bytes"] < 1.9 * 1024**3
    serialised = json.dumps(report)
    assert "spectrum" not in serialised.lower()
    assert "wavelength_values" not in serialised.lower()
