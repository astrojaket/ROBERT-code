"""Local artifact checks for the controlled Version-2 Stage-8 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "examples/outputs/emission_intercomparison/version_2/stage_8/controlled_study"
REPORT = PRODUCTS / "controlled_study_report.json"
ARRAYS = PRODUCTS / "controlled_study_arrays.npz"
SUMMARY = ROOT / "docs/data/emission_intercomparison/version_2/stage_8_controlled_study_summary.json"


def _require_products() -> None:
    if not REPORT.exists() or not ARRAYS.exists():
        pytest.skip("controlled Stage-8 numerical products have not been generated")


def test_controlled_report_covers_exact_complete_matrix_and_parameters() -> None:
    _require_products()
    report = json.loads(REPORT.read_text())
    assert report["status"] == "controlled_isotropic_grey_aerosol_study"
    assert report["track"] == "track_b_native_scattering"
    assert report["production_case_count"] == 36
    assert report["metrics"]["all_finite"]
    assert report["future_study_matrix_started"] is False
    assert len(report["execution"]) == 36
    observed = {
        (row["framework"], row["profile"], row["cells"], row["state"])
        for row in report["execution"]
    }
    assert len(observed) == 36
    expected_state = {
        "clear": (0.0, 0.0, 0.0),
        "absorbing_cloud": (1.0, 0.0, 0.0),
        "scattering_cloud": (1.0, 0.9, 0.0),
    }
    for row in report["execution"]:
        assert (row["tau"], row["omega0"], row["g"]) == expected_state[row["state"]]
        assert row["metadata"]["finite_spectrum"]
        assert row["metadata"]["study_status"] == report["status"]


def test_controlled_arrays_retain_complete_r100_states_and_exact_differences() -> None:
    _require_products()
    with np.load(ARRAYS, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    assert arrays["r100_centers_micron"].size == 369
    assert arrays["r100_flux_w_m2_m"].shape == (3, 4, 3, 369)
    assert arrays["r100_eclipse_depth"].shape == (3, 4, 3, 369)
    assert arrays["r100_cloud_effect_eclipse_ppm"].shape == (3, 4, 3, 369)
    assert arrays["r100_scattering_increment_eclipse_ppm"].shape == (3, 4, 369)
    assert np.all(np.isfinite(arrays["r100_flux_w_m2_m"]))
    states = arrays["state"].tolist()
    clear = states.index("clear")
    absorbing = states.index("absorbing_cloud")
    scattering = states.index("scattering_cloud")
    np.testing.assert_array_equal(
        arrays["r100_cloud_effect_eclipse_ppm"][:, :, clear],
        np.zeros((3, 4, 369)),
    )
    np.testing.assert_allclose(
        arrays["r100_scattering_increment_eclipse_ppm"],
        (
            arrays["r100_eclipse_depth"][:, :, scattering]
            - arrays["r100_eclipse_depth"][:, :, absorbing]
        )
        * 1.0e6,
        rtol=0.0,
        atol=0.0,
    )


def test_compact_summary_matches_full_report_resources_and_metrics() -> None:
    _require_products()
    report = json.loads(REPORT.read_text())
    summary = json.loads(SUMMARY.read_text())
    assert summary["production_case_count"] == report["production_case_count"]
    assert summary["all_spectra_finite"] == report["metrics"]["all_finite"]
    assert summary["resources"]["actual_launcher_wall_time_s"] == report["resources"][
        "launcher_wall_time_s"
    ]
    assert summary["resources"]["peak_process_tree_rss_bytes"] == report["resources"][
        "peak_process_tree_rss_bytes"
    ]
    for framework, values in report["metrics"]["frameworks"].items():
        compact = summary["primary_80_cell_pg14_non_inverted"][framework]
        assert compact["scattering_increment_rms_ppm"] == values[
            "primary_scattering_increment_rms_ppm"
        ]
        assert compact["scattering_increment_max_abs_ppm"] == values[
            "primary_scattering_increment_max_abs_ppm"
        ]


def test_controlled_integrity_index_and_plots_are_complete() -> None:
    _require_products()
    index = json.loads((PRODUCTS / "integrity.json").read_text())
    assert index["status"] == "controlled_isotropic_grey_aerosol_study"
    for item in index["files"]:
        path = PRODUCTS / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert (PRODUCTS / "controlled_primary.png").stat().st_size > 0
    assert (PRODUCTS / "controlled_convergence.png").stat().st_size > 0
