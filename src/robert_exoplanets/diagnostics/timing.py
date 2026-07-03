"""Lightweight timing diagnostics for ROBERT call paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets.core import RobertValidationError


@dataclass(frozen=True)
class TimingResult:
    """Summary of repeated wall-clock timings for one callable."""

    name: str
    durations_s: NDArray[np.float64]
    warmups: int = 0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RobertValidationError("timing result name must not be empty")
        durations = np.array(self.durations_s, dtype=float, copy=True)
        if durations.ndim != 1:
            raise RobertValidationError("timing durations must be one-dimensional")
        if durations.size == 0:
            raise RobertValidationError("timing durations must contain at least one value")
        if not np.all(np.isfinite(durations)) or np.any(durations < 0.0):
            raise RobertValidationError("timing durations must be finite and non-negative")
        warmups = int(self.warmups)
        if warmups < 0:
            raise RobertValidationError("warmups must be non-negative")

        durations.setflags(write=False)
        object.__setattr__(self, "durations_s", durations)
        object.__setattr__(self, "warmups", warmups)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def repeats(self) -> int:
        """Number of measured calls."""

        return int(self.durations_s.size)

    @property
    def min_s(self) -> float:
        """Fastest measured call in seconds."""

        return float(np.min(self.durations_s))

    @property
    def median_s(self) -> float:
        """Median measured call duration in seconds."""

        return float(np.median(self.durations_s))

    @property
    def mean_s(self) -> float:
        """Mean measured call duration in seconds."""

        return float(np.mean(self.durations_s))

    @property
    def std_s(self) -> float:
        """Population standard deviation of measured call durations."""

        return float(np.std(self.durations_s))

    @property
    def calls_per_second(self) -> float:
        """Median-rate estimate in calls per second."""

        if self.median_s == 0.0:
            return float("inf")
        return float(1.0 / self.median_s)

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a compact serializable timing summary."""

        return {
            "name": self.name,
            "repeats": self.repeats,
            "warmups": self.warmups,
            "min_s": self.min_s,
            "median_s": self.median_s,
            "mean_s": self.mean_s,
            "std_s": self.std_s,
            "calls_per_second": self.calls_per_second,
        }


def time_callable(
    function: Callable[[], object],
    *,
    name: str = "callable",
    repeat: int = 10,
    warmup: int = 1,
    timer: Callable[[], float] = perf_counter,
    metadata: Mapping[str, str] | None = None,
) -> TimingResult:
    """Time repeated calls to a zero-argument callable.

    This helper is intentionally small: it provides stable ROBERT-shaped timing
    summaries without taking a dependency on an external benchmarking package.
    For deep optimization work, use this for smoke benchmarks and then profile
    the hot kernel directly with a dedicated profiler.
    """

    if repeat < 1:
        raise RobertValidationError("repeat must be at least one")
    if warmup < 0:
        raise RobertValidationError("warmup must be non-negative")

    for _ in range(warmup):
        function()

    durations: list[float] = []
    for _ in range(repeat):
        start = float(timer())
        function()
        stop = float(timer())
        duration = stop - start
        if duration < 0.0:
            raise RobertValidationError("timer returned decreasing values")
        durations.append(duration)

    return TimingResult(
        name=name,
        durations_s=np.asarray(durations, dtype=float),
        warmups=warmup,
        metadata={} if metadata is None else metadata,
    )
