"""Tests for the real WASP-77Ab NIRSpec validation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples import validate_wasp77ab_smith2024_lrs as validation


def test_top_hat_bin_average_integrates_each_detector_bin() -> None:
    wavelength = np.linspace(1.0, 4.0, 3001)
    values = 2.0 * wavelength + 1.0
    edges = np.array([1.2, 1.7, 2.9, 3.8])

    result = validation.top_hat_bin_average(wavelength, values, edges)

    np.testing.assert_allclose(result, edges[:-1] + edges[1:] + 1.0, rtol=1e-13)


@pytest.mark.parametrize(
    ("wavelength", "values", "edges", "message"),
    (
        ([1.0, 2.0], [1.0], [1.0, 2.0], "matching"),
        ([2.0, 1.0], [1.0, 2.0], [1.0, 2.0], "increase"),
        ([1.0, 2.0], [1.0, 2.0], [0.9, 1.5], "coverage"),
    ),
)
def test_top_hat_bin_average_rejects_invalid_grids(
    wavelength: list[float],
    values: list[float],
    edges: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validation.top_hat_bin_average(wavelength, values, edges)


def test_workflow_contract_uses_public_smith_and_august_inputs() -> None:
    assert validation.TEMPLATE_BYTES == 72_672_825
    assert validation.TEMPLATE_MD5 == "93841125cab0a2c74ae4730ebf36b7dc"
    assert validation.TEMPLATE_SHA256 == (
        "fe1fb7cd37b6dec52173a19ee1fd7ea1529c421946c76d728d0bdcf4285ad34d"
    )
    assert validation.EXPECTED_DETECTOR_NAMES == (
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
    )
    assert validation.EXPECTED_POINT_COUNT == 150
    assert validation.MAX_MEMORY_BYTES == 2 * 1024**3
    assert validation.DEFAULT_MAX_MEMORY_BYTES < validation.MAX_MEMORY_BYTES
    assert len(validation.THREAD_VARIABLES) == 7


def test_real_public_template_reproduces_smith_nirspec_fit(tmp_path: Path) -> None:
    if not validation.DEFAULT_TEMPLATE.is_file():
        pytest.skip("external Smith et al. best-fit template is not installed")
    report_path = tmp_path / "report.json"

    report = validation.run_validation(
        report_path=report_path,
        plot_path=None,
    )

    assert report["status"] == "pass"
    metrics = report["metrics"]
    assert metrics["n_points"] == 150
    assert metrics["chi_square_symmetric"] == pytest.approx(
        108.71317554336035, rel=2e-10
    )
    assert metrics["chi_square_per_point_symmetric"] == pytest.approx(
        0.7247545036224023, rel=2e-10
    )
    assert metrics["smith_consistency_pass"] is True
    detector_metrics = metrics["detector_metrics"]
    assert [item["n_points"] for item in detector_metrics] == [70, 80]
    assert detector_metrics[0]["chi_square_symmetric"] == pytest.approx(
        51.98169446460743, rel=2e-10
    )
    assert detector_metrics[1]["chi_square_symmetric"] == pytest.approx(
        56.73148107875292, rel=2e-10
    )
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["data"]["count_interpretation"].startswith("the public 150-bin")
    assert saved["binning"]["detector_gap_integrated"] is False
    assert saved["resources"]["peak_rss_bytes"] < 1900 * 1024**2
    assert all(
        1 <= value <= 3 for value in saved["resources"]["thread_values"].values()
    )
