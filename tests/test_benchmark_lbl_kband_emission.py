"""Focused contract tests for the base narrow LBL benchmark."""

from __future__ import annotations

import shlex
import sys

import pytest

from examples import benchmark_lbl_kband_emission as benchmark


EXPECTED_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)


def test_process_memory_ceiling_is_strictly_below_two_gib() -> None:
    two_gib = 2 * 1024**3

    assert benchmark.MAX_MEMORY_GIB == pytest.approx(1.9)
    assert 0 < benchmark.MAX_MEMORY_BYTES < two_gib


def test_report_thread_values_include_all_seven_variables() -> None:
    values = benchmark._thread_values()

    assert tuple(values) == EXPECTED_THREAD_VARIABLES
    assert values == {name: 3 for name in EXPECTED_THREAD_VARIABLES}


def test_reproducible_command_is_top_level_ready_and_preserves_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = [
        "benchmark_lbl_kband_emission.py",
        "--label",
        "K band window",
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    assert benchmark._reproducible_command() == shlex.join(
        [
            "conda",
            "run",
            "-n",
            "robert-exoplanets",
            "python",
            "examples/benchmark_lbl_kband_emission.py",
            "--label",
            "K band window",
        ]
    )
