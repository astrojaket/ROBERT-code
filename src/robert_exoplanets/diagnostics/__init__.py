"""Diagnostics and reference calculations."""

from .benchmarks import EmissionBenchmark, load_emission_benchmark_csv
from .blackbody import (
    blackbody_eclipse_depth,
    blackbody_eclipse_depth_spectrum,
    planck_radiance_wavelength,
)
from .cloud_benchmarks import CloudOpticalPropertyComparison, compare_cloud_optical_properties
from .leave_one_out import (
    LEAVE_ONE_OUT_ARRAYS_FILENAME,
    LEAVE_ONE_OUT_SCHEMA_VERSION,
    LEAVE_ONE_OUT_SUMMARY_FILENAME,
    LeaveOneOutComparison,
    LeaveOneOutResult,
    compare_psis_leave_one_out,
    plot_leave_one_out_result,
    psis_leave_one_out,
    run_psis_leave_one_out,
    write_leave_one_out_result,
)
from .opacity_benchmarks import OpacityComparisonResult, compare_opacity_arrays
from .timing import TimingResult, time_callable

__all__ = [
    "CloudOpticalPropertyComparison",
    "EmissionBenchmark",
    "LEAVE_ONE_OUT_ARRAYS_FILENAME",
    "LEAVE_ONE_OUT_SCHEMA_VERSION",
    "LEAVE_ONE_OUT_SUMMARY_FILENAME",
    "LeaveOneOutComparison",
    "LeaveOneOutResult",
    "OpacityComparisonResult",
    "TimingResult",
    "blackbody_eclipse_depth",
    "blackbody_eclipse_depth_spectrum",
    "compare_cloud_optical_properties",
    "compare_opacity_arrays",
    "compare_psis_leave_one_out",
    "load_emission_benchmark_csv",
    "planck_radiance_wavelength",
    "plot_leave_one_out_result",
    "psis_leave_one_out",
    "run_psis_leave_one_out",
    "time_callable",
    "write_leave_one_out_result",
]
