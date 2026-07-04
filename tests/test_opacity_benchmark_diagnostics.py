"""Tests for opacity benchmark comparison helpers."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import OpacityComparisonResult, compare_opacity_arrays
from robert_exoplanets.core import RobertValidationError


def test_compare_opacity_arrays_reports_passed_benchmark() -> None:
    reference = np.ones((2, 3, 4, 5))
    candidate = reference * (1.0 + 1.0e-8)

    result = compare_opacity_arrays(
        candidate,
        reference,
        name="tiny-k-table",
        axis_names=("pressure", "temperature", "wavelength", "g_ordinate"),
        relative_tolerance=1.0e-6,
        metadata={"species": "H2O"},
    )

    assert isinstance(result, OpacityComparisonResult)
    assert result.passed
    assert result.shape == (2, 3, 4, 5)
    assert result.axis_names == ("pressure", "temperature", "wavelength", "g_ordinate")
    assert result.max_relative_difference < 1.0e-6
    assert result.metadata["species"] == "H2O"


def test_compare_opacity_arrays_reports_failed_benchmark_location() -> None:
    reference = np.ones((2, 2, 2))
    candidate = reference.copy()
    candidate[1, 0, 1] = 2.0

    result = compare_opacity_arrays(
        candidate,
        reference,
        axis_names=("pressure", "temperature", "wavelength"),
        relative_tolerance=1.0e-3,
    )

    assert not result.passed
    assert result.max_absolute_difference == 1.0
    assert result.max_relative_difference == 1.0
    assert result.worst_index == (1, 0, 1)
    assert result.to_mapping()["passed"] is False


def test_compare_opacity_arrays_validates_shape_and_values() -> None:
    with pytest.raises(RobertValidationError, match="same shape"):
        compare_opacity_arrays(np.ones((2,)), np.ones((2, 1)))

    with pytest.raises(RobertValidationError, match="non-negative"):
        compare_opacity_arrays(np.array([-1.0]), np.array([1.0]))

    with pytest.raises(RobertValidationError, match="axis_names"):
        compare_opacity_arrays(np.ones((2, 2)), np.ones((2, 2)), axis_names=("pressure",))
