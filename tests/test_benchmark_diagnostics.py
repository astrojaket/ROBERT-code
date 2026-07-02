"""Tests for external benchmark loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import EmissionBenchmark, load_emission_benchmark_csv
from robert_exoplanets.core import RobertDataError, RobertValidationError


def test_load_emission_benchmark_csv_reads_references(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.csv"
    benchmark_path.write_text(
        "wavelength_um,value,bb_1200K,bb_1500K\n"
        "1.0,0.001,0.0005,0.0008\n"
        "2.0,0.002,0.0010,0.0015\n",
        encoding="utf-8",
    )

    benchmark = load_emission_benchmark_csv(benchmark_path, name="tiny")

    assert benchmark.name == "tiny"
    assert benchmark.n_points == 2
    assert benchmark.source_path == benchmark_path
    np.testing.assert_allclose(benchmark.wavelength_micron, np.array([1.0, 2.0]))
    np.testing.assert_allclose(benchmark.eclipse_depth, np.array([0.001, 0.002]))
    assert tuple(benchmark.references) == ("bb_1200K", "bb_1500K")
    assert benchmark.eclipse_depth.flags.writeable is False


def test_emission_benchmark_converts_to_spectrum() -> None:
    benchmark = EmissionBenchmark(
        name="tiny",
        wavelength_micron=np.array([1.0, 2.0]),
        eclipse_depth=np.array([0.001, 0.002]),
    )

    spectrum = benchmark.to_spectrum()

    assert spectrum.spectral_grid.role == "benchmark"
    assert spectrum.unit == "eclipse_depth"
    assert spectrum.metadata["benchmark"] == "tiny"


def test_load_emission_benchmark_csv_requires_columns(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "bad.csv"
    benchmark_path.write_text("wavelength_um,not_value\n1.0,0.001\n", encoding="utf-8")

    with pytest.raises(RobertDataError, match="missing required columns"):
        load_emission_benchmark_csv(benchmark_path)


def test_emission_benchmark_requires_increasing_wavelengths() -> None:
    with pytest.raises(RobertValidationError, match="strictly increasing"):
        EmissionBenchmark(
            name="bad",
            wavelength_micron=np.array([2.0, 1.0]),
            eclipse_depth=np.array([0.001, 0.002]),
        )
