"""Tests for lightweight timing diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import TimingResult, time_callable
from robert_exoplanets.core import RobertValidationError


def test_timing_result_summarizes_durations() -> None:
    result = TimingResult(
        name="tiny",
        durations_s=np.array([0.3, 0.1, 0.2]),
        warmups=2,
    )

    assert result.repeats == 3
    assert result.min_s == pytest.approx(0.1)
    assert result.median_s == pytest.approx(0.2)
    assert result.mean_s == pytest.approx(0.2)
    assert result.calls_per_second == pytest.approx(5.0)
    assert result.durations_s.flags.writeable is False
    assert result.as_dict()["name"] == "tiny"


def test_time_callable_runs_warmups_and_repeats() -> None:
    calls = {"function": 0, "timer": 0}

    def function() -> None:
        calls["function"] += 1

    def timer() -> float:
        calls["timer"] += 1
        return float(calls["timer"]) * 0.5

    result = time_callable(function, name="tiny", repeat=3, warmup=2, timer=timer)

    assert calls["function"] == 5
    assert calls["timer"] == 6
    np.testing.assert_allclose(result.durations_s, np.full(3, 0.5))
    assert result.warmups == 2


def test_time_callable_rejects_invalid_repeat() -> None:
    with pytest.raises(RobertValidationError, match="repeat"):
        time_callable(lambda: None, repeat=0)


def test_time_callable_rejects_decreasing_timer() -> None:
    values = iter([1.0, 0.5])

    with pytest.raises(RobertValidationError, match="decreasing"):
        time_callable(lambda: None, repeat=1, warmup=0, timer=lambda: next(values))
