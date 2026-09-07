"""Run a bounded synthetic contract benchmark for multi-order HRS data.

This benchmark exercises the Step-4 observation path only.  It uses two named
synthetic spectral orders with different grids, detector bins, masks, fixed
velocity terms, Gaussian LSFs, and fixed linear data/model filters.  The
native response is prepared once and reused for truth and trial spectra.  The
filter is then applied to detector-frame data and to every response-mapped
trial model with the same matrix.

The result is an operator contract, not a real-target science validation.  It
does not use opacity data, a sampler, or cross-order covariance.  Separate
per-order likelihoods assume independent order noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Mapping

# Cap numerical libraries before NumPy is imported.  The benchmark is small,
# but this keeps a local run within the project CPU contract.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
for _thread_variable in _THREAD_VARIABLES:
    _requested_threads = os.environ.get(_thread_variable, "1")
    try:
        _requested_threads = str(min(3, max(1, int(_requested_threads))))
    except ValueError:
        _requested_threads = "1"
    os.environ[_thread_variable] = _requested_threads

import numpy as np  # noqa: E402

from robert_exoplanets import (  # noqa: E402
    CorrelatedGaussianLikelihood,
    DenseCovariance,
    HighResolutionObservation,
    LinearDataModelFilter,
    Observation,
    PositiveSemidefiniteCovariance,
    SpectralGrid,
    Spectrum,
    __version__,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "data" / "high_resolution_observation_pipeline_20260831.json"
MAX_RSS_BYTES = 1900 * 1024**2
RUNTIME_RADIAL_VELOCITY_KM_S = 2.5
FLUX_UNIT = "eclipse_depth"
OBSERVABLE = "eclipse_depth"
WAVELENGTH_UNIT = "micron"


@dataclass(frozen=True)
class _OrderSpec:
    """Immutable configuration for one synthetic spectral order."""

    name: str
    order_number: int
    wavelengths: tuple[float, ...]
    pixel_edges: tuple[float, ...]
    observation_mask: tuple[bool, ...]
    order_mask: tuple[bool, ...]
    native_min: float
    native_max: float
    native_points: int
    barycentric_velocity_km_s: float
    systemic_velocity_km_s: float
    lsf_resolving_power: float
    filter_matrix: tuple[tuple[float, ...], ...]
    filter_output_wavelengths: tuple[float, ...]
    input_sigma: float
    input_correlation: float
    raw_residual: tuple[float, ...]
    line_centres: tuple[float, ...]
    line_widths: tuple[float, ...]
    line_amplitudes: tuple[float, ...]
    trial_amplitude_offsets: tuple[float, ...]


ORDER_SPECS: tuple[_OrderSpec, ...] = (
    _OrderSpec(
        name="order_blue",
        order_number=1,
        wavelengths=(1.49982, 1.49998, 1.50017, 1.50038, 1.50062, 1.50089),
        pixel_edges=(1.49973, 1.49990, 1.50008, 1.50027, 1.50050, 1.50075, 1.50104),
        observation_mask=(True, True, False, True, True, True),
        order_mask=(True, False, True, True, True, True),
        native_min=1.495,
        native_max=1.505,
        native_points=2001,
        barycentric_velocity_km_s=12.0,
        systemic_velocity_km_s=-3.0,
        lsf_resolving_power=80_000.0,
        filter_matrix=(
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0, -1.0),
            (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        ),
        filter_output_wavelengths=(1.49984, 1.50031, 1.50065, 1.50091),
        input_sigma=0.0025,
        input_correlation=0.30,
        raw_residual=(0.0006, -0.0002, 0.0004, -0.0005, 0.0003, -0.0001),
        line_centres=(1.50022, 1.50073),
        line_widths=(0.000025, 0.000018),
        line_amplitudes=(0.018, -0.011),
        trial_amplitude_offsets=(-0.0015, 0.0010),
    ),
    _OrderSpec(
        name="order_red",
        order_number=2,
        wavelengths=(1.79970, 1.79992, 1.80015, 1.80041, 1.80070, 1.80102, 1.80137),
        pixel_edges=(1.79956, 1.79982, 1.80004, 1.80029, 1.80057, 1.80085, 1.80120, 1.80155),
        observation_mask=(True, False, True, True, True, True, True),
        order_mask=(True, True, False, True, True, True, True),
        native_min=1.794,
        native_max=1.807,
        native_points=2601,
        barycentric_velocity_km_s=-7.0,
        systemic_velocity_km_s=4.0,
        lsf_resolving_power=120_000.0,
        filter_matrix=(
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0),
            (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        filter_output_wavelengths=(1.79972, 1.80012, 1.80043, 1.80073, 1.80105, 1.80135),
        input_sigma=0.0030,
        input_correlation=0.22,
        raw_residual=(0.0004, -0.0007, 0.0002, -0.0003, 0.0005, -0.0002, 0.0001),
        line_centres=(1.80020, 1.80091),
        line_widths=(0.000030, 0.000022),
        line_amplitudes=(0.014, -0.009),
        trial_amplitude_offsets=(0.0012, -0.0008),
    ),
)


def _peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _thread_values() -> dict[str, int]:
    """Return the numerical thread settings used by this process."""

    values: dict[str, int] = {}
    for name in _THREAD_VARIABLES:
        try:
            values[name] = int(os.environ.get(name, "0"))
        except ValueError:
            values[name] = 0
    return values


def _array_checksum(values: object) -> str:
    """Hash array dtype, shape, and bytes for compact input provenance."""

    array = np.asarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _native_grid(spec: _OrderSpec) -> SpectralGrid:
    """Build one small synthetic native grid."""

    values = np.linspace(spec.native_min, spec.native_max, spec.native_points)
    return SpectralGrid(
        values=values,
        unit=WAVELENGTH_UNIT,
        name=f"{spec.name}-native",
        role="native",
    )


def _observation(spec: _OrderSpec, flux: np.ndarray) -> Observation:
    """Build one detector-frame observation with explicit bins and masks."""

    return Observation.from_arrays(
        wavelength=spec.wavelengths,
        flux=flux,
        uncertainty=np.full(len(spec.wavelengths), spec.input_sigma),
        wavelength_unit=WAVELENGTH_UNIT,
        flux_unit=FLUX_UNIT,
        observable=OBSERVABLE,
        instrument=f"synthetic-{spec.name}",
        mask=spec.observation_mask,
        wavelength_bin_edges=spec.pixel_edges,
    )


def _collection(observations: Mapping[str, Observation]) -> HighResolutionObservation:
    """Build the named-order collection with order-specific response settings."""

    return HighResolutionObservation.from_mapping(
        observations,
        order_numbers={spec.name: spec.order_number for spec in ORDER_SPECS},
        barycentric_velocities_km_s={
            spec.name: spec.barycentric_velocity_km_s for spec in ORDER_SPECS
        },
        systemic_velocities_km_s={
            spec.name: spec.systemic_velocity_km_s for spec in ORDER_SPECS
        },
        masks={spec.name: spec.order_mask for spec in ORDER_SPECS},
        lsf_resolving_powers={
            spec.name: spec.lsf_resolving_power for spec in ORDER_SPECS
        },
        data_model_filters={
            spec.name: LinearDataModelFilter(
                spec.filter_matrix,
                name=f"{spec.name}-fixed-linear-filter",
                output_wavelength=spec.filter_output_wavelengths,
                output_wavelength_unit=WAVELENGTH_UNIT,
            )
            for spec in ORDER_SPECS
        },
        lsf_kernel_support_sigma=4.0,
        pixel_integration=True,
        infer_pixel_edges=False,
        name="synthetic-two-order-hrs-contract",
        metadata={
            "benchmark_scope": "synthetic operator contract",
            "cross_order_covariance": "not included",
        },
    )


def _native_spectrum(grid: SpectralGrid, values: np.ndarray) -> Spectrum:
    """Build one native synthetic model spectrum."""

    return Spectrum(
        spectral_grid=grid,
        values=values,
        unit=FLUX_UNIT,
        observable=OBSERVABLE,
    )


def _model_values(spec: _OrderSpec, grid: SpectralGrid, *, trial: bool) -> np.ndarray:
    """Return a smooth continuum plus two narrow synthetic lines."""

    wavelength = np.asarray(grid.values, dtype=float)
    midpoint = 0.5 * (spec.native_min + spec.native_max)
    span = spec.native_max - spec.native_min
    values = 1.0 + 0.0015 * (wavelength - midpoint) / span
    for index, (centre, width, amplitude) in enumerate(
        zip(spec.line_centres, spec.line_widths, spec.line_amplitudes, strict=True)
    ):
        if trial:
            amplitude += spec.trial_amplitude_offsets[index]
        values += amplitude * np.exp(-0.5 * ((wavelength - centre) / width) ** 2)
    return np.asarray(values, dtype=float)


def _input_covariance(spec: _OrderSpec) -> np.ndarray:
    """Build one positive-definite input detector covariance."""

    index = np.arange(len(spec.wavelengths), dtype=float)
    correlation = spec.input_correlation ** np.abs(index[:, None] - index[None, :])
    return (spec.input_sigma**2) * correlation


def _order_report(
    spec: _OrderSpec,
    native_grid: SpectralGrid,
    truth_native: Spectrum,
    trial_native: Spectrum,
    prepared_order: object,
    truth_response: Spectrum,
    trial_response: Spectrum,
    filtered_observation: Observation,
    filtered_model: Spectrum,
    filtered_data: np.ndarray,
    filtered_truth: np.ndarray,
    output_mask: np.ndarray,
    input_covariance: np.ndarray,
) -> dict[str, object]:
    """Evaluate and describe one order's prepared operator contract."""

    prepared_filter = prepared_order.data_model_filter
    if prepared_filter is None:
        raise RuntimeError("contract benchmark requires a prepared linear filter")
    effective_matrix = np.asarray(prepared_filter.effective_matrix, dtype=float)
    propagated_covariance = np.asarray(
        prepared_filter.propagate_covariance(input_covariance),
        dtype=float,
    )
    expected_covariance = effective_matrix @ input_covariance @ effective_matrix.T
    covariance_propagation_passed = bool(
        np.allclose(propagated_covariance, expected_covariance, rtol=1.0e-12, atol=1.0e-16)
    )

    covariance_kind: str
    covariance_operator: DenseCovariance | PositiveSemidefiniteCovariance
    if spec.name == "order_blue":
        covariance_kind = "DenseCovariance"
        covariance_operator = DenseCovariance(
            propagated_covariance,
            name=f"{spec.name}-filtered-dense-covariance",
        )
    else:
        covariance_kind = "PositiveSemidefiniteCovariance"
        covariance_operator = PositiveSemidefiniteCovariance(
            propagated_covariance,
            name=f"{spec.name}-filtered-psd-covariance",
            rank_rtol=1.0e-12,
            support_rtol=1.0e-9,
        )

    likelihood = CorrelatedGaussianLikelihood(covariance_operator)
    chi_square = likelihood.chi_square(filtered_model, filtered_observation)
    residual = np.asarray(filtered_data - filtered_model.values, dtype=float)
    if covariance_kind == "DenseCovariance":
        numpy_chi_square = float(residual @ np.linalg.solve(propagated_covariance, residual))
    else:
        numpy_chi_square = float(
            residual
            @ np.linalg.pinv(propagated_covariance, rcond=1.0e-12)
            @ residual
        )
    chi_square_difference = abs(float(chi_square) - numpy_chi_square)
    chi_square_passed = bool(
        np.isclose(chi_square, numpy_chi_square, rtol=1.0e-10, atol=1.0e-12)
    )

    direct_filtered_data = effective_matrix @ (
        np.asarray(truth_response.values, dtype=float)
        + np.asarray(spec.raw_residual, dtype=float)
    )
    direct_filtered_truth = effective_matrix @ truth_response.values
    direct_filtered_trial = effective_matrix @ trial_response.values
    same_filter_passed = bool(
        np.array_equal(filtered_data, direct_filtered_data)
        and np.array_equal(filtered_truth, direct_filtered_truth)
        and np.array_equal(filtered_model.values, direct_filtered_trial)
        and np.array_equal(filtered_observation.flux, filtered_data)
    )
    response_repeat_truth = prepared_order.response.observe(truth_native)
    response_repeat_trial = prepared_order.response.observe(trial_native)
    same_response_passed = bool(
        np.array_equal(response_repeat_truth.values, truth_response.values)
        and np.array_equal(response_repeat_trial.values, trial_response.values)
    )
    expected_order = (
        "relativistic-doppler",
        "gaussian-high-resolution",
        "pixel-bin-integration",
    )
    response_operator_names = tuple(
        str(getattr(operator, "name", type(operator).__name__))
        for operator in prepared_order.response.prepared_operators
    )
    response_order_passed = response_operator_names == expected_order
    effective_mask = np.asarray(prepared_order.order.valid_mask, dtype=bool)
    expected_total_velocity = (
        spec.barycentric_velocity_km_s
        + spec.systemic_velocity_km_s
        + RUNTIME_RADIAL_VELOCITY_KM_S
    )
    doppler_factor = float(
        np.sqrt(
            (1.0 + expected_total_velocity / 299_792.458)
            / (1.0 - expected_total_velocity / 299_792.458)
        )
    )
    first_operator = prepared_order.response.prepared_operators[0]
    observed_doppler_velocity = float(getattr(first_operator, "velocity_km_s", 0.0))
    retained_rank = int(covariance_operator.effective_rank)
    expected_rank = int(np.linalg.matrix_rank(propagated_covariance, tol=1.0e-12))

    checksum_inputs = {
        "observation_wavelengths_sha256": _array_checksum(spec.wavelengths),
        "pixel_edges_sha256": _array_checksum(spec.pixel_edges),
        "observation_mask_sha256": _array_checksum(spec.observation_mask),
        "order_mask_sha256": _array_checksum(spec.order_mask),
        "effective_mask_sha256": _array_checksum(effective_mask),
        "native_grid_sha256": _array_checksum(native_grid.values),
        "filter_matrix_sha256": _array_checksum(spec.filter_matrix),
        "effective_filter_matrix_sha256": _array_checksum(effective_matrix),
        "filter_output_wavelengths_sha256": _array_checksum(
            spec.filter_output_wavelengths
        ),
        "input_covariance_sha256": _array_checksum(input_covariance),
        "filtered_covariance_sha256": _array_checksum(propagated_covariance),
        "truth_native_spectrum_sha256": _array_checksum(truth_native.values),
        "trial_native_spectrum_sha256": _array_checksum(trial_native.values),
        "raw_residual_sha256": _array_checksum(spec.raw_residual),
    }
    return {
        "order_number": spec.order_number,
        "n_input_pixels": len(spec.wavelengths),
        "n_output_pixels": int(filtered_model.values.size),
        "wavelength_unit": WAVELENGTH_UNIT,
        "wavelengths": list(spec.wavelengths),
        "pixel_edges": list(spec.pixel_edges),
        "observation_mask": list(spec.observation_mask),
        "order_mask": list(spec.order_mask),
        "effective_mask": effective_mask.tolist(),
        "lsf": {
            "resolving_power": spec.lsf_resolving_power,
            "kernel_support_sigma": 4.0,
            "response_stage": "Gaussian LSF after Doppler and before pixel integration",
        },
        "velocity": {
            "barycentric_velocity_km_s": spec.barycentric_velocity_km_s,
            "systemic_velocity_km_s": spec.systemic_velocity_km_s,
            "runtime_radial_velocity_km_s": RUNTIME_RADIAL_VELOCITY_KM_S,
            "fixed_velocity_km_s": spec.barycentric_velocity_km_s
            + spec.systemic_velocity_km_s,
            "total_velocity_km_s": expected_total_velocity,
            "observed_doppler_velocity_km_s": observed_doppler_velocity,
            "doppler_factor": doppler_factor,
            "sign_convention": (
                "positive velocity shifts model wavelengths longer; fixed terms and "
                "runtime term add before one relativistic Doppler stage"
            ),
            "formula": "D=sqrt((1+beta)/(1-beta)); lambda_observed=D*lambda_rest",
        },
        "response_order": list(response_operator_names),
        "required_response_order": list(expected_order),
        "filter": {
            "name": prepared_filter.name,
            "shape": list(effective_matrix.shape),
            "effective_matrix_rank": int(np.linalg.matrix_rank(effective_matrix)),
            "same_fixed_operator_for_data_and_model": same_filter_passed,
        },
        "covariance": {
            "input_shape": list(input_covariance.shape),
            "filtered_shape": list(propagated_covariance.shape),
            "operator": covariance_kind,
            "retained_rank": retained_rank,
            "independent_numpy_rank": expected_rank,
            "propagation_formula": "C_filtered = F C_input F.T",
            "propagation_passed": covariance_propagation_passed,
        },
        "chi_square": {
            "robert": float(chi_square),
            "numpy_independent": numpy_chi_square,
            "absolute_difference": chi_square_difference,
            "method": (
                "np.linalg.solve for DenseCovariance"
                if covariance_kind == "DenseCovariance"
                else "np.linalg.pinv(rcond=1e-12) for PositiveSemidefiniteCovariance"
            ),
            "passed": chi_square_passed,
        },
        "operator_contract": {
            "same_response_operator_reused": same_response_passed,
            "same_filter_operator_data_model": same_filter_passed,
            "response_order_passed": response_order_passed,
            "output_mask_all_valid": bool(np.all(output_mask)),
            "filtered_observation_matches_filtered_data": bool(
                np.array_equal(filtered_observation.flux, filtered_data)
            ),
        },
        "checksum_inputs": checksum_inputs,
    }


def run_benchmark() -> dict[str, object]:
    """Run the deterministic two-order contract benchmark in memory."""

    started = perf_counter()
    if len(ORDER_SPECS) != 2:
        raise RuntimeError("contract benchmark must contain exactly two order specifications")
    if len({spec.name for spec in ORDER_SPECS}) != 2:
        raise RuntimeError("contract benchmark order names must be unique")

    native_grids = {spec.name: _native_grid(spec) for spec in ORDER_SPECS}
    placeholder_observations = {
        spec.name: _observation(spec, np.zeros(len(spec.wavelengths)))
        for spec in ORDER_SPECS
    }
    placeholder_collection = _collection(placeholder_observations)
    placeholder_prepared = placeholder_collection.prepare(
        native_grids,
        {
            spec.name: RUNTIME_RADIAL_VELOCITY_KM_S
            for spec in ORDER_SPECS
        },
    )
    truth_native = {
        spec.name: _native_spectrum(
            native_grids[spec.name],
            _model_values(spec, native_grids[spec.name], trial=False),
        )
        for spec in ORDER_SPECS
    }
    truth_response = {
        spec.name: placeholder_prepared.order(spec.name).response.observe(
            truth_native[spec.name]
        )
        for spec in ORDER_SPECS
    }
    data_observations = {
        spec.name: _observation(
            spec,
            np.asarray(truth_response[spec.name].values, dtype=float)
            + np.asarray(spec.raw_residual, dtype=float),
        )
        for spec in ORDER_SPECS
    }
    # The zero-flux observations above carry the final detector coordinates,
    # masks, and bins.  Their prepared response/filter is reused for the
    # actual data values and every trial model below.  Flux values do not enter
    # response preparation.
    prepared = placeholder_prepared

    order_reports: dict[str, dict[str, object]] = {}
    for spec in ORDER_SPECS:
        prepared_order = prepared.order(spec.name)
        truth_response_final = prepared_order.response.observe(truth_native[spec.name])
        trial_native = _native_spectrum(
            native_grids[spec.name],
            _model_values(spec, native_grids[spec.name], trial=True),
        )
        trial_response = prepared_order.response.observe(trial_native)
        filtered_data, filtered_truth, output_mask = prepared_order.apply_data_model(
            data_observations[spec.name].flux,
            truth_response_final.values,
        )
        if prepared_order.data_model_filter is None:  # pragma: no cover - fixed above
            raise RuntimeError("contract benchmark requires a prepared linear filter")
        filtered_observation = (
            prepared_order.data_model_filter.operator.apply_to_observation(
                data_observations[spec.name]
            )
        )
        filtered_model = prepared_order.observe(trial_native)
        order_reports[spec.name] = _order_report(
            spec,
            native_grids[spec.name],
            truth_native[spec.name],
            trial_native,
            prepared_order,
            truth_response_final,
            trial_response,
            filtered_observation,
            filtered_model,
            filtered_data,
            filtered_truth,
            output_mask,
            _input_covariance(spec),
        )

    order_names = tuple(spec.name for spec in ORDER_SPECS)
    order_gate_values = {
        name: {
            "response_filter_contract": bool(
                report["operator_contract"]["same_response_operator_reused"]
                and report["operator_contract"]["same_filter_operator_data_model"]
                and report["operator_contract"]["response_order_passed"]
            ),
            "covariance_FCFT": bool(report["covariance"]["propagation_passed"]),
            "correlated_chi_square": bool(report["chi_square"]["passed"]),
            "retained_rank_matches_numpy": bool(
                report["covariance"]["retained_rank"]
                == report["covariance"]["independent_numpy_rank"]
            ),
            "output_mask_valid": bool(report["operator_contract"]["output_mask_all_valid"]),
        }
        for name, report in order_reports.items()
    }
    all_order_gates = [
        passed
        for values in order_gate_values.values()
        for passed in values.values()
    ]
    peak_rss = _peak_rss_bytes()
    thread_values = _thread_values()
    resource_gates = {
        "cpu_threads_at_most_three": all(
            1 <= value <= 3 for value in thread_values.values()
        ),
        "rss_below_2_gib": peak_rss < MAX_RSS_BYTES,
    }
    input_checksum_payload = {
        name: report["checksum_inputs"] for name, report in order_reports.items()
    }
    input_checksum = sha256(
        json.dumps(input_checksum_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    total_chi_square = float(
        sum(float(report["chi_square"]["robert"]) for report in order_reports.values())
    )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "pass" if all(all_order_gates) and all(resource_gates.values()) else "fail",
        "passed": bool(all(all_order_gates) and all(resource_gates.values())),
        "benchmark": "high-resolution multi-order observation/covariance contract",
        "scope": "synthetic operator contract",
        "not_real_target_science_validation": True,
        "not_cross_order_covariance": True,
        "opacity_data_used": False,
        "sampler_used": False,
        "command": " ".join(sys.argv),
        "software": {
            "robert_exoplanets": __version__,
            "numpy": np.__version__,
        },
        "runtime_radial_velocity_km_s": RUNTIME_RADIAL_VELOCITY_KM_S,
        "response_and_filter_contract": {
            "response_order": [
                "Doppler",
                "Gaussian LSF",
                "pixel integration",
                "fixed linear data/model filter",
            ],
            "data_model_statement": (
                "The prepared response maps native truth and trial models to the "
                "detector grid. The same prepared fixed filter then maps detector "
                "data and every response-mapped trial model."
            ),
            "fixed_filter_prepared_once_per_order": True,
        },
        "covariance_assumption": {
            "per_order_likelihoods": True,
            "orders_are_independent": True,
            "cross_order_covariance_included": False,
            "statement": (
                "The total chi-square is the sum of separate order chi-squares. "
                "No covariance block connects the two orders."
            ),
        },
        "orders": order_reports,
        "order_gates": order_gate_values,
        "total_chi_square": total_chi_square,
        "resource": {
            "thread_values": thread_values,
            "cpu_guard_passed": resource_gates["cpu_threads_at_most_three"],
            "peak_rss_bytes": peak_rss,
            "rss_limit_bytes": MAX_RSS_BYTES,
            "rss_guard_passed": resource_gates["rss_below_2_gib"],
            "rule": "one process, numerical thread variables <= 3, RSS < 2 GiB",
        },
        "gates": {
            "two_named_distinct_orders": order_names == ("order_blue", "order_red"),
            "response_filter_contract": bool(
                all(
                    values["response_filter_contract"]
                    for values in order_gate_values.values()
                )
            ),
            "covariance_FCFT": bool(
                all(values["covariance_FCFT"] for values in order_gate_values.values())
            ),
            "correlated_chi_square_numpy": bool(
                all(
                    values["correlated_chi_square"]
                    for values in order_gate_values.values()
                )
            ),
            "retained_ranks_recorded": bool(
                all(values["retained_rank_matches_numpy"] for values in order_gate_values.values())
            ),
            "doppler_sign_and_order_recorded": bool(
                all(
                    report["velocity"]["sign_convention"]
                    and report["response_order"] == report["required_response_order"]
                    for report in order_reports.values()
                )
            ),
            "independent_order_covariance_assumption_recorded": True,
            "resources": bool(all(resource_gates.values())),
        },
        "input_checksum_sha256": input_checksum,
        "elapsed_seconds": perf_counter() - started,
    }
    return report


def write_report(report: Mapping[str, object], path: Path = REPORT_PATH) -> None:
    """Write one compact JSON report atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    """Run and write the benchmark report."""

    report = run_benchmark()
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
