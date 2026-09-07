"""Focused contract tests for the expanded K-band LBL benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from examples import benchmark_lbl_kband_science_grid as benchmark


THREAD_VARIABLES = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
}


def test_process_and_opacity_guards_are_strictly_below_two_gib() -> None:
    two_gib = 2 * 1024**3
    one_gib = 1024**3

    assert 0 < benchmark.MAX_MEMORY_BYTES < two_gib
    assert 0 < benchmark.OPACITY_MEMORY_BYTES < one_gib


def test_report_records_reproducible_command_and_all_thread_values() -> None:
    report_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "data"
        / "lbl_kband_science_grid_20260830.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "examples/benchmark_lbl_kband_science_grid.py" in report["command"]
    assert set(report["thread_values"]) == THREAD_VARIABLES
    assert all(1 <= value <= 3 for value in report["thread_values"].values())
    assert report["resources"]["thread_values"] == report["thread_values"]
    assert report["resources"]["memory_limit_bytes"] < 2 * 1024**3
    assert (
        report["wide_band"]["resources"]["end_to_end_estimate_bytes"]
        < report["resources"]["memory_limit_bytes"]
    )


def test_thread_values_match_the_process_settings() -> None:
    assert benchmark._thread_values() == {
        name: int(benchmark.os.environ[name]) for name in THREAD_VARIABLES
    }
