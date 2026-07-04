"""Diagnostics and reference calculations."""

from .benchmarks import EmissionBenchmark, load_emission_benchmark_csv
from .blackbody import (
    blackbody_eclipse_depth,
    blackbody_eclipse_depth_spectrum,
    planck_radiance_wavelength,
)
from .opacity_benchmarks import OpacityComparisonResult, compare_opacity_arrays
from .timing import TimingResult, time_callable

__all__ = [
    "EmissionBenchmark",
    "OpacityComparisonResult",
    "TimingResult",
    "blackbody_eclipse_depth",
    "blackbody_eclipse_depth_spectrum",
    "compare_opacity_arrays",
    "load_emission_benchmark_csv",
    "planck_radiance_wavelength",
    "time_callable",
]
