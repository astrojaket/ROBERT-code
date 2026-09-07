"""Bounded shared-VMR ROBERT retrieval for the real WASP-77Ab data.

This example is the physical follow-up to the fixed-template Smith
comparison.  It uses the public August et al. (2023) NIRSpec G395H table and
the cleaned Smith et al. (2024) IGRINS 2020-12-14 cube.  The IGRINS orders
inside 1.90--2.45 micron are retained as the documented K-band subset.

The two instruments use one :class:`AtmosphereBuilder` instance.  H2O and CO
come from the tabulated petitRADTRANS line-by-line tables.  LRS and HRS use
different, explicit sampling strides.  The real-grid report's LRS stride 25
is retained as the accuracy-preferred choice, while the laptop joint process
uses report-validated stride 100 to stay below its resident-memory limit.  The
physical
ROBERT state contains VMR profiles only.  No mass-fraction retrieval value is
accepted here.  A pRT boundary conversion is not needed for this workflow.

The H-minus continuum is composed from the H-minus, neutral-H, and electron
VMRs in ``FreeChemistry``.  The selected 1.90--5.17 micron window is beyond
the John bound-free cutoff at 1.6421 micron, so bound-free opacity is zero in
this window.  Free-free opacity is controlled by a fixed neutral-H VMR and a
retrieved electron VMR.  Fixing neutral H removes the H*electron product
degeneracy.  The retrieved H-minus VMR is therefore prior-dominated in this
window; it is retained because the same state is used for future shorter
wavelength LRS work.  The strict H-minus fit temperature range is enforced by
the temperature prior.

The default command is a dry run.  ``--preflight`` performs source checks,
builds the real operators, and runs a small injection/null operator check.
Only ``--run-pymultinest`` starts sampling.  That path uses PyMultiNest with
the MultiNest backend, two fixed seeds, one MPI process, and the repository
thread and memory limits.  No other sampler is supported by this example.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Final, Mapping, Sequence


ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.fetch_wasp77ab_phoenix_subset import (  # noqa: E402
    PHOENIX_BASE_URL,
    PHOENIX_FILES,
)

THREAD_VARIABLES: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
MAX_THREADS: Final = 6


def _clamp_thread_environment() -> None:
    """Clamp numerical thread controls before importing NumPy."""

    for name in THREAD_VARIABLES:
        raw = os.environ.get(name, str(MAX_THREADS))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = MAX_THREADS
        os.environ[name] = str(min(MAX_THREADS, max(1, value)))


_clamp_thread_environment()
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "robert-matplotlib"),
)

import numpy as np  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402

from examples import wasp77ab_target as target  # noqa: E402
from robert_exoplanets import (  # noqa: E402
    AtmosphereBuilder,
    BackgroundGasMixture,
    CompositionMeanMolecularWeight,
    FreeChemistry,
    GaussianLikelihood,
    GaussianHighResolutionResponse,
    HeterogeneousLikelihood,
    HeterogeneousLikelihoodComponent,
    HeterogeneousRetrievalProblem,
    HighResolutionEmissionTemplate,
    IsothermalTemperatureProfile,
    LineByLineOpacityProvider,
    LineByLineTable,
    Observation,
    ObservationCollection,
    ParameterizedEmissionForwardModel,
    ParameterizedEmissionModelConfig,
    PressureGrid,
    RetrievalParameter,
    RetrievalParameterSet,
    RotationalBroadeningResponse,
    Spectrum,
    TimeResolvedHighResolutionLikelihood,
    TimeResolvedHighResolutionObservation,
    TopHatObservationResponse,
    UniformPrior,
    HMinusContinuumConfig,
    load_august2023_wasp77ab,
    load_nemesispy_cia_table,
    load_smith2024_wasp77ab_hrs,
)
from robert_exoplanets.core import (  # noqa: E402
    RobertCoverageError,
    RobertValidationError,
)
from robert_exoplanets.instruments import PreparedSpectralOperator  # noqa: E402
from robert_exoplanets.io.smith2024_wasp77ab import (  # noqa: E402
    SMITH2024_WASP77AB_CUBE14_MD5,
    SMITH2024_WASP77AB_CUBE14_SHA256,
    SMITH2024_WASP77AB_CUBE14_SIZE,
    SMITH2024_WASP77AB_INFO_MD5,
    SMITH2024_WASP77AB_INFO_SHA256,
    SMITH2024_WASP77AB_INFO_SIZE,
)


HRS_MODE: Final = "hrs_only"
LRS_MODE: Final = "lrs_only"
JOINT_MODE: Final = "joint"
ALL_MODES: Final = (HRS_MODE, LRS_MODE, JOINT_MODE)

DEFAULT_SMITH_DATA_ROOT: Final = ROOT / "external_data" / "wasp77ab_smith2024"
DEFAULT_NIRSPEC_DATA_ROOT: Final = target.DATA_DIRECTORY
DEFAULT_REPORT_PATH: Final = (
    ROOT / "docs" / "data" / "wasp77ab_robert_joint_20260831.json"
)
DEFAULT_OUTPUT_ROOT: Final = (
    ROOT / "examples" / "outputs" / "wasp77ab_robert_joint"
)
DEFAULT_STRIDE_REPORT: Final = ROOT / "docs" / "data" / "wasp77ab_lbl_sampling_20260831.json"
DEFAULT_H2O_TABLE: Final = (
    ROOT
    / "external_data/petitRADTRANS/input_data/opacities/lines/line_by_line/H2O/1H2-16O"
    / "1H2-16O__POKAZATEL.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)
DEFAULT_CO_TABLE: Final = (
    ROOT
    / "external_data/petitRADTRANS/input_data/opacities/lines/line_by_line/CO/12C-16O"
    / "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)

SEEDS: Final = (24680, 24681)
N_LIVE_POINTS: Final = 64
MAX_ITER: Final = 0
EVIDENCE_TOLERANCE: Final = 0.5
SAMPLING_EFFICIENCY: Final = 0.8
MPI_PROCESSES: Final = 1
PROCESS_RSS_HARD_LIMIT_BYTES: Final = 2 * 1024**3
DEFAULT_PROCESS_RSS_LIMIT_BYTES: Final = int(1.9 * 1024**3)
OPACITY_MEMORY_LIMIT_BYTES: Final = 1023 * 1024**2
INVALID_LOGLIKE_FLOOR: Final = -1.0e100

HRS_K_BAND_RANGE_MICRON: Final = (1.90, 2.45)
NIRSPEC_RANGE_MICRON: Final = (2.808, 5.168)
HRS_SOURCE_MARGIN_MICRON: Final = 0.020
LRS_SOURCE_MARGIN_MICRON: Final = 0.020
HRS_KP_PRIOR_KM_S: Final = (172.0, 212.0)
HRS_DVSYS_PRIOR_KM_S: Final = (-20.0, 20.0)
SAFE_HRS_LBL_STRIDE: Final = 2
SAFE_LRS_LBL_STRIDE: Final = 50
# The finest passing LRS stride in the sampling benchmark is 25.  The real
# joint preflight retains too much memory at that setting, however.  Stride
# 100 is the smallest benchmark-validated production setting that keeps the
# full NRS1/NRS2 data and the shared VMR model on a laptop.
LRS_MEMORY_SAFE_STRIDE: Final = 100
LRS_MEMORY_SAFE_RMS_LIMIT_SIGMA: Final = 0.25
LRS_MEMORY_SAFE_MAX_LIMIT_SIGMA: Final = 1.0
ATMOSPHERE_LAYERS: Final = 32
PRESSURE_FIRST_CENTER_BAR: Final = 1.0e-5
PRESSURE_LAST_CENTER_BAR: Final = 100.0
# The Gray/pRT free-free fit is only defined for T >= 2500 K.  Keep the
# retrieval prior inside that coverage when strict H-minus extrapolation is
# selected below.  This is an engineering comparison, not Smith's PT prior.
TEMPERATURE_PRIOR_K: Final = (2500.0, 2900.0)
FIXED_NEUTRAL_H_VMR: Final = 0.1
HRS_ROTATION_KM_S: Final = float(target.HRS_VSIN_I_KM_S)
HRS_LSF_RESOLVING_POWER: Final = float(target.HRS_INSTRUMENT_RESOLVING_POWER)
HRS_LIMB_DARKENING: Final = 0.6

H2O_TABLE_SHA256: Final = (
    "9a79513fb92dfa369f85abb273257ae2b2100a4a386169b9bfff8a5b74249b73"
)
CO_TABLE_SHA256: Final = (
    "5f77209da3ea67d5a697b06379b1de2b8fe106adf7ea0abfb451c675084087ad"
)

PARAMETER_NAMES: Final = {
    "temperature": "temperature_K",
    "H2O": "log10_H2O_VMR",
    "CO": "log10_CO_VMR",
    "H-": "log10_Hminus_VMR",
    "H": "log10_H_VMR",
    "e-": "log10_electron_VMR",
}
HRS_KP_PARAMETER: Final = "Kp"
HRS_DVSYS_PARAMETER: Final = "dVsys"
HRS_SCALE_PARAMETER: Final = "log10_a_hrs"
LRS_SCALE_PARAMETER: Final = "lrs_scale"

HMINUS_CONFIG: Final = HMinusContinuumConfig(
    temperature_extrapolation="raise",
    spectral_extrapolation="raise",
    metadata={
        "bound_free_cutoff_micron": "1.6421",
        "window_bound_free_status": "zero throughout 1.90-5.17 micron",
        "free_free_control": "fixed neutral-H and retrieved electron VMR profiles",
        "fixed_neutral_h_vmr": str(FIXED_NEUTRAL_H_VMR),
        "retrieved_hminus_vmr_status": "prior_dominated_in_selected_window",
        "strict_fit_temperature_min_K": "2500",
    },
)


def _peak_rss_bytes() -> int:
    """Return peak resident memory in bytes."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _thread_values() -> dict[str, int]:
    """Return all numerical thread values after clamping."""

    return {name: int(os.environ[name]) for name in THREAD_VARIABLES}


def _mpi_world_size() -> int:
    """Return MPI world size without requiring mpi4py."""

    try:
        from mpi4py import MPI
    except ImportError:
        return 1
    return int(MPI.COMM_WORLD.Get_size())


def resource_record(
    *,
    max_memory_bytes: int = DEFAULT_PROCESS_RSS_LIMIT_BYTES,
    check_peak: bool = True,
) -> dict[str, object]:
    """Record the strict process, MPI, and thread resource gate."""

    limit = int(max_memory_bytes)
    if not 0 < limit < PROCESS_RSS_HARD_LIMIT_BYTES:
        raise ValueError("max_memory_bytes must be positive and below 2 GiB")
    peak = _peak_rss_bytes() if check_peak else None
    mpi_size = _mpi_world_size() if check_peak else MPI_PROCESSES
    threads = _thread_values()
    threads_pass = all(1 <= value <= MAX_THREADS for value in threads.values())
    mpi_pass = mpi_size == MPI_PROCESSES
    process_pass = peak is None or peak < limit
    return {
        "thread_values": threads,
        "thread_limit": MAX_THREADS,
        "mpi_processes_configured": MPI_PROCESSES,
        "mpi_world_size_observed": mpi_size,
        "threads_pass": threads_pass,
        "mpi_pass": mpi_pass,
        "peak_rss_bytes": peak,
        "configured_process_rss_limit_bytes": limit,
        "process_rss_hard_limit_bytes": PROCESS_RSS_HARD_LIMIT_BYTES,
        "process_rss_pass": process_pass,
        "gate_passed": bool(threads_pass and mpi_pass and process_pass),
    }


def _sha256(path: Path) -> str:
    """Hash a file with a bounded memory buffer."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path, *, name: str, role: str, expected_sha256: str | None = None) -> dict[str, object]:
    """Return a bounded source identity record."""

    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    verified = expected_sha256 is None or actual_sha256 == expected_sha256
    if not verified:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "name": name,
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "verified": verified,
    }


def _resolve_pysyn_cdbs_root(root: Path | str | None) -> Path:
    """Resolve the explicit or environment-provided STScI data root."""

    value = os.environ.get("PYSYN_CDBS") if root is None else str(root)
    if value is None or not value.strip():
        raise FileNotFoundError(
            "PHOENIX mode requires an explicit --pysyn-cdbs-root or PYSYN_CDBS"
        )
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"PYSYN_CDBS root was not found or is not a directory: {resolved}"
        )
    return resolved


def _phoenix_source_records(root: Path | str | None) -> dict[str, Mapping[str, object]]:
    """Verify and record the exact minimal STScI PHOENIX subset."""

    resolved = _resolve_pysyn_cdbs_root(root)
    records: dict[str, Mapping[str, object]] = {}
    for spec in PHOENIX_FILES:
        path = resolved / "grid" / "phoenix" / spec.name
        if not path.is_file():
            raise FileNotFoundError(
                f"PHOENIX subset file was not found: {path}"
            )
        measured_size = path.stat().st_size
        if measured_size != spec.size_bytes:
            raise ValueError(
                f"{spec.name} size mismatch: expected {spec.size_bytes}, "
                f"got {measured_size}"
            )
        record = _source_record(
            path,
            name=spec.name,
            role=spec.role,
            expected_sha256=spec.sha256,
        )
        record.update(
            {
                "expected_size_bytes": spec.size_bytes,
                "relative_path": f"grid/phoenix/{spec.name}",
                "official_url": spec.content_url,
                "pysyn_cdbs_root": str(resolved),
                "official_base_url": PHOENIX_BASE_URL,
            }
        )
        key = f"phoenix_{Path(spec.name).stem}"
        records[key] = record
    return records


def _select_k_band_orders(order_wavelengths: ArrayLike) -> tuple[int, ...]:
    """Return real IGRINS orders wholly inside the declared K window."""

    values = np.asarray(order_wavelengths, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("order_wavelengths must be a two-dimensional grid")
    lower, upper = HRS_K_BAND_RANGE_MICRON
    selected = tuple(
        int(index)
        for index, row in enumerate(values)
        if float(np.min(row)) >= lower and float(np.max(row)) <= upper
    )
    if not selected:
        raise ValueError("the Smith cube has no complete K-band order")
    return selected


def select_k_band_observation(
    observation: TimeResolvedHighResolutionObservation,
) -> tuple[TimeResolvedHighResolutionObservation, tuple[int, ...]]:
    """Keep the 1.90--2.45 micron Smith orders as a separate HRS product."""

    order_indices = _select_k_band_orders(observation.order_wavelengths)
    selected = TimeResolvedHighResolutionObservation.from_arrays(
        observation.order_wavelengths[list(order_indices)],
        observation.flux[list(order_indices)],
        observation.phase,
        observation.fixed_velocity_km_s,
        observation.time_bjd,
        mask=observation.mask[list(order_indices)],
        airmass=observation.airmass,
        humidity_percent=observation.humidity_percent,
        median_snr=observation.median_snr,
        wavelength_unit=observation.wavelength_unit,
        flux_unit=observation.flux_unit,
        observable=observation.observable,
        instrument=observation.instrument,
        name=f"{observation.name} K-band subset",
        metadata={
            **dict(observation.metadata),
            "k_band_subset": "1.90-2.45 micron",
            "selected_order_indices": ",".join(map(str, order_indices)),
            "selected_order_count": str(len(order_indices)),
            "source_public_cube_is_cleaned": "Smith cube14v3 already discards bad orders and 200 edge pixels",
        },
    )
    return selected, order_indices


def _pressure_grid() -> PressureGrid:
    return PressureGrid.from_log_centers(
        PRESSURE_FIRST_CENTER_BAR,
        PRESSURE_LAST_CENTER_BAR,
        ATMOSPHERE_LAYERS,
        unit="bar",
        name="WASP-77Ab real joint ROBERT pressure grid",
    )


def _molecular_masses() -> dict[str, float]:
    return {
        "H2": 2.01588,
        "He": 4.002602,
        "H2O": 18.01528,
        "CO": 28.0101,
        "H-": 1.00054858,
        "H": 1.00794,
        "e-": 5.485799096e-4,
    }


def build_atmosphere_builder(
    *,
    pressure_grid: PressureGrid | None = None,
) -> AtmosphereBuilder:
    """Build one shared VMR-only atmosphere parameterisation."""

    pressure = _pressure_grid() if pressure_grid is None else pressure_grid
    chemistry = FreeChemistry(
        active_species=("H2O", "CO", "H-", "H", "e-"),
        background=BackgroundGasMixture({"H2": 0.85, "He": 0.15}),
        fixed_mixing_ratios={"H": FIXED_NEUTRAL_H_VMR},
        parameter_names={
            species: parameter
            for species, parameter in PARAMETER_NAMES.items()
            if species not in {"temperature", "H"}
        },
        parameter_mode="log10",
        fill_background=True,
        excess_policy="raise",
        metadata={
            "convention": "volume_mixing_ratio",
            "mass_fraction_parameters": "none",
            "hminus_control": "H- VMR, fixed H VMR, and retrieved e- VMR",
            "fixed_neutral_h_vmr": str(FIXED_NEUTRAL_H_VMR),
        },
    )
    temperature = IsothermalTemperatureProfile(
        parameter_name=PARAMETER_NAMES["temperature"],
        name="WASP-77Ab bounded shared isothermal atmosphere",
    )
    return AtmosphereBuilder(
        pressure_grid=pressure,
        temperature_profile=temperature,
        chemistry_model=chemistry,
        mean_molecular_weight_model=CompositionMeanMolecularWeight(
            molecular_masses=_molecular_masses(),
            normalization="require",
        ),
        opacity_free_species=("H-", "H", "e-"),
    )


def _stride_from_report(report_path: Path | None, *, mode: str) -> tuple[int, str]:
    """Read a passing measured stride, or use an explicit safe provisional one."""

    default = SAFE_HRS_LBL_STRIDE if mode == HRS_MODE else SAFE_LRS_LBL_STRIDE
    report_branch = "hrs" if mode == HRS_MODE else "lrs"
    if report_path is None or not Path(report_path).is_file():
        return default, "safe_default_no_measured_stride_report"
    try:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default, "safe_default_stride_report_unreadable"

    # A selected stride is only usable when the benchmark itself passed.  A
    # stale or failed report may still contain a numeric selection from a
    # partial run, and accepting that value would turn diagnostic output into
    # an unverified retrieval setting.
    if not isinstance(payload, Mapping) or payload.get("status") != "pass":
        return default, "safe_default_stride_report_not_passing"
    if payload.get("record_type") != "wasp77ab_real_grid_lbl_sampling_convergence":
        return default, "safe_default_stride_report_wrong_record_type"
    target_record = payload.get("target")
    if not isinstance(target_record, Mapping) or target_record.get("name") != "WASP-77Ab":
        return default, "safe_default_stride_report_wrong_target"

    def walk(value: object) -> int | None:
        if isinstance(value, Mapping):
            candidate = value.get("selected_finest_passing_stride")
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                try:
                    candidate_float = float(candidate)
                    candidate_int = int(candidate)
                except (TypeError, ValueError, OverflowError):
                    candidate_float = float("nan")
                    candidate_int = 0
                if (
                    np.isfinite(candidate_float)
                    and candidate_int == candidate
                    and candidate_int > 0
                ):
                    return candidate_int
            for key, child in value.items():
                key_text = str(key).lower()
                if key_text in {"selection", "stride_selection", report_branch, mode}:
                    found = walk(child)
                    if found is not None:
                        return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = walk(child)
                if found is not None:
                    return found
        return None

    selected = walk(payload.get(report_branch))
    if selected is None:
        return default, "safe_default_stride_report_has_no_selection"
    return selected, f"measured_stride_report:{Path(report_path)}"


def _lrs_memory_safe_stride_from_report(
    report_path: Path | None,
    *,
    accuracy_preferred_stride: int,
    accuracy_source: str,
) -> tuple[int, str, Mapping[str, str]]:
    """Select the report-validated laptop LRS stride.

    The convergence report keeps stride 25 as the accuracy-preferred choice.
    A real joint ROBERT process also retains the HRS and LRS preparations at
    the same time, so it uses stride 100 to keep the measured preflight below
    the process limit.  This override is accepted only when the report
    contains the stride-100 candidate, its own numerical gates, and the
    global resource gates.  A missing or stale report fails closed instead of
    silently using an unvalidated sparse grid.
    """

    if report_path is None or not Path(report_path).is_file():
        raise RobertValidationError(
            "real LRS production stride requires a passing report that validates "
            f"stride {LRS_MEMORY_SAFE_STRIDE}"
        )
    path = Path(report_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RobertValidationError(
            "real LRS production stride report is unreadable"
        ) from error
    if not isinstance(payload, Mapping):
        raise RobertValidationError("real LRS production stride report must be an object")
    if payload.get("status") != "pass":
        raise RobertValidationError(
            "real LRS production stride requires a passing convergence report"
        )
    if payload.get("record_type") != "wasp77ab_real_grid_lbl_sampling_convergence":
        raise RobertValidationError(
            "real LRS production stride report has the wrong record type"
        )
    target_record = payload.get("target")
    if not isinstance(target_record, Mapping) or target_record.get("name") != "WASP-77Ab":
        raise RobertValidationError(
            "real LRS production stride report has the wrong target"
        )
    acceptance = payload.get("acceptance")
    required_acceptance = (
        "lrs_finest_passing_stride_selected",
        "lrs_reference_and_candidates_complete",
        "opacity_estimates_below_1023_MiB",
        "process_rss_below_1_9_GiB",
        "thread_values_between_1_and_3",
    )
    if not isinstance(acceptance, Mapping) or any(
        acceptance.get(key) is not True for key in required_acceptance
    ):
        raise RobertValidationError(
            "real LRS production stride report does not pass its resource and "
            "candidate acceptance gates"
        )

    lrs_record = payload.get("lrs")
    if not isinstance(lrs_record, Mapping):
        raise RobertValidationError("real LRS production stride report has no LRS branch")
    candidate_strides = lrs_record.get("candidate_strides")
    numeric_candidates: set[int] = set()
    if isinstance(candidate_strides, (list, tuple)):
        for value in candidate_strides:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            try:
                value_float = float(value)
                value_int = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value_float) and value_int == value and value_int > 0:
                numeric_candidates.add(value_int)
    if LRS_MEMORY_SAFE_STRIDE not in numeric_candidates:
        raise RobertValidationError(
            f"real LRS production stride report does not test stride {LRS_MEMORY_SAFE_STRIDE}"
        )
    metrics = lrs_record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RobertValidationError("real LRS production stride report has no LRS metrics")
    candidate = metrics.get(str(LRS_MEMORY_SAFE_STRIDE))
    if candidate is None:
        candidate = metrics.get(LRS_MEMORY_SAFE_STRIDE)
    if not isinstance(candidate, Mapping) or candidate.get("gate_pass") is not True:
        raise RobertValidationError(
            f"real LRS stride {LRS_MEMORY_SAFE_STRIDE} does not pass its benchmark gate"
        )
    try:
        rms = float(candidate["rms_sigma"])
        maximum = float(candidate["max_absolute_sigma"])
        rms_limit = float(candidate["gate_rms_limit_sigma"])
        maximum_limit = float(candidate["gate_max_limit_sigma"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RobertValidationError(
            f"real LRS stride {LRS_MEMORY_SAFE_STRIDE} has incomplete metrics"
        ) from error
    if (
        not all(np.isfinite(value) for value in (rms, maximum, rms_limit, maximum_limit))
        or rms_limit > LRS_MEMORY_SAFE_RMS_LIMIT_SIGMA
        or maximum_limit > LRS_MEMORY_SAFE_MAX_LIMIT_SIGMA
        or rms > rms_limit
        or maximum > maximum_limit
    ):
        raise RobertValidationError(
            f"real LRS stride {LRS_MEMORY_SAFE_STRIDE} fails its declared accuracy gates"
        )
    estimates = lrs_record.get("max_preparation_estimate_bytes")
    if not isinstance(estimates, Mapping):
        raise RobertValidationError(
            "real LRS production stride report has no preparation estimates"
        )
    estimate = estimates.get(str(LRS_MEMORY_SAFE_STRIDE))
    if estimate is None:
        estimate = estimates.get(LRS_MEMORY_SAFE_STRIDE)
    try:
        estimate_bytes = int(estimate)
    except (TypeError, ValueError, OverflowError) as error:
        raise RobertValidationError(
            f"real LRS stride {LRS_MEMORY_SAFE_STRIDE} has no valid memory estimate"
        ) from error
    if estimate_bytes <= 0 or estimate_bytes >= OPACITY_MEMORY_LIMIT_BYTES:
        raise RobertValidationError(
            f"real LRS stride {LRS_MEMORY_SAFE_STRIDE} reaches the opacity memory limit"
        )

    preferred_estimate = estimates.get(str(accuracy_preferred_stride))
    if preferred_estimate is None:
        preferred_estimate = estimates.get(accuracy_preferred_stride)
    try:
        savings = max(int(preferred_estimate) - estimate_bytes, 0)
    except (TypeError, ValueError, OverflowError):
        savings = 0
    policy = (
        "report_validated_laptop_memory_safe_stride"
        if accuracy_preferred_stride == LRS_MEMORY_SAFE_STRIDE
        else "report_validated_laptop_memory_safe_override"
    )
    source = (
        f"{accuracy_source};{policy}:{accuracy_preferred_stride}->"
        f"{LRS_MEMORY_SAFE_STRIDE};rms_sigma={rms:.17g};"
        f"max_absolute_sigma={maximum:.17g};estimate_bytes={estimate_bytes}"
    )
    metadata = {
        "accuracy_preferred_stride": str(int(accuracy_preferred_stride)),
        "accuracy_preferred_source": str(accuracy_source),
        "production_stride": str(LRS_MEMORY_SAFE_STRIDE),
        "policy": policy,
        "report": str(path),
        "candidate_rms_sigma": f"{rms:.17g}",
        "candidate_max_absolute_sigma": f"{maximum:.17g}",
        "candidate_rms_limit_sigma": f"{rms_limit:.17g}",
        "candidate_max_limit_sigma": f"{maximum_limit:.17g}",
        "candidate_estimate_bytes": str(estimate_bytes),
        "estimated_savings_bytes_vs_accuracy_preferred": str(savings),
    }
    return LRS_MEMORY_SAFE_STRIDE, source, metadata


def _line_by_line_tables(
    *,
    h2o_table: Path,
    co_table: Path,
) -> Mapping[str, LineByLineTable]:
    """Inspect the two external LBL tables once for both model branches."""

    return {
        "H2O": LineByLineTable.from_hdf5(
            h2o_table,
            species="H2O",
            checksum=False,
        ),
        "CO": LineByLineTable.from_hdf5(
            co_table,
            species="CO",
            checksum=False,
        ),
    }


def _provider(
    *,
    h2o_table: Path,
    co_table: Path,
    bounds: tuple[float, float],
    stride: int,
    tables: Mapping[str, LineByLineTable] | None = None,
) -> LineByLineOpacityProvider:
    """Create a bounded pRT-table LBL provider."""

    options = {
        "max_memory_bytes": OPACITY_MEMORY_LIMIT_BYTES,
        "wavelength_bounds_micron": bounds,
        "native_sampling_stride": int(stride),
        "max_cached_slices": 0,
    }
    if tables is None:
        return LineByLineOpacityProvider.from_hdf_paths(
            {"H2O": h2o_table, "CO": co_table},
            checksum=False,
            **options,
        )
    return LineByLineOpacityProvider(tables=tables, **options)


def _guard_opacity_memory(provider: LineByLineOpacityProvider, grid: object) -> int:
    estimate = int(provider.estimate_memory_bytes(grid, species=("H2O", "CO")))
    if estimate >= OPACITY_MEMORY_LIMIT_BYTES:
        raise MemoryError(
            f"LBL estimate {estimate} B reaches the strict {OPACITY_MEMORY_LIMIT_BYTES} B limit"
        )
    return estimate


def _observation_grid_for_order(
    wavelengths: ArrayLike,
    *,
    order_index: int,
) -> Observation:
    values = np.asarray(wavelengths, dtype=float)
    return Observation.from_arrays(
        values,
        np.zeros(values.size, dtype=float),
        np.ones(values.size, dtype=float),
        wavelength_unit="micron",
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
        instrument=f"Smith-IGRINS-K-order-{order_index}",
    )


def _hrs_velocity_bounds(
    observation: TimeResolvedHighResolutionObservation,
) -> tuple[float, float]:
    """Return velocity extrema over the complete HRS prior rectangle."""

    phase_sine = np.sin(2.0 * np.pi * np.asarray(observation.phase, dtype=float))
    kp_lower, kp_upper = HRS_KP_PRIOR_KM_S
    dVsys_lower, dVsys_upper = HRS_DVSYS_PRIOR_KM_S
    fixed = np.asarray(observation.fixed_velocity_km_s, dtype=float)
    kp_for_lower = np.where(phase_sine >= 0.0, kp_lower, kp_upper)
    kp_for_upper = np.where(phase_sine >= 0.0, kp_upper, kp_lower)
    lower = fixed + dVsys_lower + kp_for_lower * phase_sine
    upper = fixed + dVsys_upper + kp_for_upper * phase_sine
    lower_bound = float(np.min(lower))
    upper_bound = float(np.max(upper))
    if (
        not np.isfinite(lower_bound)
        or not np.isfinite(upper_bound)
        or abs(lower_bound) >= 299_792.458
        or abs(upper_bound) >= 299_792.458
    ):
        raise RobertValidationError(
            "HRS velocity prior extrema must be finite and below the speed of light"
        )
    return lower_bound, upper_bound


def _hrs_query_wavelength_bounds(
    observation: TimeResolvedHighResolutionObservation,
) -> tuple[float, float]:
    """Return rest-grid wavelengths sampled at every allowed HRS velocity."""

    velocity_lower, velocity_upper = _hrs_velocity_bounds(observation)
    wavelengths = np.asarray(observation.order_wavelengths, dtype=float)
    sampled = np.concatenate(
        (
            wavelengths * (1.0 - velocity_lower / 299_792.458),
            wavelengths * (1.0 - velocity_upper / 299_792.458),
        )
    )
    lower = float(np.min(sampled))
    upper = float(np.max(sampled))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0:
        raise RobertValidationError(
            "HRS velocity prior produces invalid template query bounds"
        )
    return lower, upper


def _prepare_hrs_broadening(
    native_grid: object,
    observation: TimeResolvedHighResolutionObservation,
) -> tuple[PreparedSpectralOperator, ...]:
    """Prepare Gray rotation and IGRINS Gaussian LSF on the LBL grid."""

    rotation = RotationalBroadeningResponse(
        projected_velocity_km_s=HRS_ROTATION_KM_S,
        limb_darkening=HRS_LIMB_DARKENING,
    ).prepare_on_grid(native_grid)
    gaussian = GaussianHighResolutionResponse(
        resolving_power=HRS_LSF_RESOLVING_POWER,
        kernel_support=4.0,
    ).prepare_on_grid(rotation.target_grid)
    query_lower, query_upper = _hrs_query_wavelength_bounds(observation)
    output = np.asarray(gaussian.target_grid.values, dtype=float)
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, query_upper)
    if (
        float(np.min(output)) > query_lower + tolerance
        or float(np.max(output)) < query_upper - tolerance
    ):
        raise RobertCoverageError(
            "HRS response output does not cover all allowed Doppler samples"
        )
    return rotation, gaussian


@dataclass(frozen=True)
class RobertJointInputs:
    """Prepared real data, shared atmosphere, and mode-specific RT state."""

    hrs_observation: TimeResolvedHighResolutionObservation
    nirspec_observations: ObservationCollection
    hrs_model: ParameterizedEmissionForwardModel
    lrs_model: ParameterizedEmissionForwardModel
    hrs_broadening: tuple[PreparedSpectralOperator, ...]
    lrs_responses: Mapping[str, object]
    source_hashes: Mapping[str, Mapping[str, object]]
    hrs_order_indices: tuple[int, ...]
    hrs_stride: int
    lrs_stride: int
    stride_sources: Mapping[str, str]
    opacity_estimates: Mapping[str, int]
    hminus_enabled: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)
    memory_checkpoints: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.hrs_observation, TimeResolvedHighResolutionObservation):
            raise RobertValidationError("hrs_observation has an unexpected type")
        if not isinstance(self.nirspec_observations, ObservationCollection):
            raise RobertValidationError("nirspec_observations has an unexpected type")
        if not isinstance(self.hrs_model, ParameterizedEmissionForwardModel):
            raise RobertValidationError("hrs_model has an unexpected type")
        if not isinstance(self.lrs_model, ParameterizedEmissionForwardModel):
            raise RobertValidationError("lrs_model has an unexpected type")
        if self.hrs_model.atmosphere_builder is not self.lrs_model.atmosphere_builder:
            raise RobertValidationError("HRS and LRS models must share one AtmosphereBuilder")
        if not isinstance(self.hminus_enabled, bool):
            raise RobertValidationError("hminus_enabled must be boolean")
        object.__setattr__(self, "lrs_responses", dict(self.lrs_responses))
        object.__setattr__(self, "source_hashes", dict(self.source_hashes))
        object.__setattr__(self, "stride_sources", dict(self.stride_sources))
        object.__setattr__(self, "opacity_estimates", dict(self.opacity_estimates))
        checkpoints = {
            str(name): int(value)
            for name, value in self.memory_checkpoints.items()
        }
        if any(value < 0 for value in checkpoints.values()):
            raise RobertValidationError("memory checkpoints must be non-negative")
        object.__setattr__(self, "memory_checkpoints", checkpoints)


def _real_source_hashes(
    *,
    smith_root: Path,
    hrs: TimeResolvedHighResolutionObservation,
    lrs: ObservationCollection,
    h2o_table: Path,
    co_table: Path,
    stride_report: Path | None = None,
    lbl_source_records: Mapping[str, Mapping[str, object]] | None = None,
    phoenix_source_records: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Mapping[str, object]]:
    """Record strict source identities without retaining bulk arrays."""

    metadata = hrs.metadata
    records: dict[str, Mapping[str, object]] = {
        "smith_cube14v3": {
            "name": "cube14v3.pic",
            "role": "Smith et al. 2024 cleaned pre-eclipse IGRINS cube",
            "path": str(smith_root / "cube14v3.pic"),
            "size_bytes": int(metadata["cube_size_bytes"]),
            "sha256": str(metadata["cube_sha256"]),
            "expected_size_bytes": SMITH2024_WASP77AB_CUBE14_SIZE,
            "expected_md5": SMITH2024_WASP77AB_CUBE14_MD5,
            "expected_sha256": SMITH2024_WASP77AB_CUBE14_SHA256,
            "verified": True,
        },
        "smith_info14": {
            "name": "20201214_info.csv",
            "role": "Smith et al. 2024 frame phases and fixed RV terms",
            "path": str(smith_root / "20201214_info.csv"),
            "size_bytes": int(metadata["info_size_bytes"]),
            "sha256": str(metadata["info_sha256"]),
            "expected_size_bytes": SMITH2024_WASP77AB_INFO_SIZE,
            "expected_md5": SMITH2024_WASP77AB_INFO_MD5,
            "expected_sha256": SMITH2024_WASP77AB_INFO_SHA256,
            "verified": True,
        },
    }
    table_path = Path(str(lrs.datasets[0].observation.metadata["source_path"]))
    lrs_meta = lrs.datasets[0].observation.metadata
    records["august_nirspec_table"] = {
        "name": table_path.name,
        "role": "August et al. 2023 NIRSpec G395H comparison table",
        "path": str(table_path),
        "size_bytes": table_path.stat().st_size,
        "sha256": str(lrs_meta["checksum_sha256"]),
        "expected_sha256": target.NIRSPEC_TABLE.sha256,
        "verified": True,
        "detectors": ["NRS1", "NRS2"],
    }
    if lbl_source_records is None:
        lbl_source_records = {
            "h2o_lbl": _source_record(
                h2o_table,
                name=h2o_table.name,
                role="pRT H2O POKAZATEL R=1e6 tabulated LBL cross sections",
                expected_sha256=H2O_TABLE_SHA256,
            ),
            "co_lbl": _source_record(
                co_table,
                name=co_table.name,
                role="pRT CO HITEMP R=1e6 tabulated LBL cross sections",
                expected_sha256=CO_TABLE_SHA256,
            ),
        }
    records.update(
        {
            str(name): dict(record)
            for name, record in lbl_source_records.items()
        }
    )
    if phoenix_source_records is not None:
        records.update(
            {
                str(name): dict(record)
                for name, record in phoenix_source_records.items()
            }
        )
    if stride_report is not None and Path(stride_report).is_file():
        records["lbl_stride_report"] = _source_record(
            Path(stride_report),
            name=Path(stride_report).name,
            role="passing real-grid LBL stride-selection benchmark",
        )
    cia = load_nemesispy_cia_table()
    records["cia"] = {
        "name": "NEMESISPY CIA table",
        "role": "H2-H2/H2-He collision-induced absorption",
        "source_project": cia.metadata.get("source_project", "NEMESISPY"),
        "sha256": cia.metadata.get("checksum_sha256", ""),
        "verified": True,
    }
    return records


def load_real_inputs(
    *,
    smith_data_root: Path = DEFAULT_SMITH_DATA_ROOT,
    nirspec_data_root: Path = DEFAULT_NIRSPEC_DATA_ROOT,
    h2o_table: Path = DEFAULT_H2O_TABLE,
    co_table: Path = DEFAULT_CO_TABLE,
    stride_report: Path | None = DEFAULT_STRIDE_REPORT,
    stellar_spectrum_model: str = "phoenix",
    pysyn_cdbs_root: Path | str | None = None,
    hminus_enabled: bool = True,
) -> RobertJointInputs:
    """Load real Smith/August data and prepare shared-VMR ROBERT models."""

    memory_checkpoints: dict[str, int] = {}

    def checkpoint(name: str) -> None:
        memory_checkpoints[str(name)] = _peak_rss_bytes()

    if not isinstance(hminus_enabled, bool):
        raise ValueError("hminus_enabled must be boolean")
    stellar_model = str(stellar_spectrum_model).strip().lower()
    if stellar_model not in {"phoenix", "blackbody"}:
        raise ValueError("stellar_spectrum_model must be 'phoenix' or 'blackbody'")
    phoenix_root: Path | None = None
    phoenix_source_records: dict[str, Mapping[str, object]] = {}
    if stellar_model == "phoenix":
        # Verify the small official subset before any LBL provider or opacity
        # preparation can allocate memory.  Blackbody mode deliberately skips
        # this check and does not require PYSYN_CDBS.
        phoenix_root = _resolve_pysyn_cdbs_root(pysyn_cdbs_root)
        phoenix_source_records = _phoenix_source_records(phoenix_root)
    smith_root = Path(smith_data_root).expanduser()
    hrs_full = load_smith2024_wasp77ab_hrs(
        smith_root / "cube14v3.pic",
        info_path=smith_root / "20201214_info.csv",
        verify_checksum=True,
        verify_sha256=True,
    )
    hrs, order_indices = select_k_band_observation(hrs_full)
    # ``select_k_band_observation`` copies the source identity metadata into
    # the retained K-band object.  Release the full cleaned cube before the
    # two LBL preparations allocate their hyperslabs; it is not part of the
    # real retrieval once the documented orders have been selected.
    del hrs_full
    nirspec = load_august2023_wasp77ab(Path(nirspec_data_root), verify_checksum=True)
    if nirspec.names != ("nirspec_g395h_nrs1", "nirspec_g395h_nrs2"):
        raise RobertValidationError("real NIRSpec data must retain NRS1 and NRS2 separately")
    checkpoint("source_load")
    h2o_path = Path(h2o_table).expanduser()
    co_path = Path(co_table).expanduser()
    if not h2o_path.is_file() or not co_path.is_file():
        raise FileNotFoundError("both H2O and CO LBL tables are required")

    # Verify the large external tables before preparing any hyperslabs or
    # response operators.  The digest is streamed in bounded chunks.  This
    # prevents a wrong table from consuming the real-run memory budget before
    # the identity failure is reported.
    lbl_source_records = {
        "h2o_lbl": _source_record(
            h2o_path,
            name=h2o_path.name,
            role="pRT H2O POKAZATEL R=1e6 tabulated LBL cross sections",
            expected_sha256=H2O_TABLE_SHA256,
        ),
        "co_lbl": _source_record(
            co_path,
            name=co_path.name,
            role="pRT CO HITEMP R=1e6 tabulated LBL cross sections",
            expected_sha256=CO_TABLE_SHA256,
        ),
    }

    hrs_stride, hrs_source = _stride_from_report(stride_report, mode=HRS_MODE)
    lrs_accuracy_stride, lrs_accuracy_source = _stride_from_report(
        stride_report,
        mode=LRS_MODE,
    )
    (
        lrs_stride,
        lrs_source,
        lrs_stride_metadata,
    ) = _lrs_memory_safe_stride_from_report(
        stride_report,
        accuracy_preferred_stride=lrs_accuracy_stride,
        accuracy_source=lrs_accuracy_source,
    )
    k_lower = float(np.min(hrs.order_wavelengths)) - HRS_SOURCE_MARGIN_MICRON
    k_upper = float(np.max(hrs.order_wavelengths)) + HRS_SOURCE_MARGIN_MICRON
    l_lower, l_upper = NIRSPEC_RANGE_MICRON
    lbl_tables = _line_by_line_tables(
        h2o_table=h2o_path,
        co_table=co_path,
    )
    checkpoint("table_metadata")
    hrs_provider = _provider(
        h2o_table=h2o_path,
        co_table=co_path,
        bounds=(k_lower, k_upper),
        stride=hrs_stride,
        tables=lbl_tables,
    )
    checkpoint("hrs_provider")
    lrs_provider = _provider(
        h2o_table=h2o_path,
        co_table=co_path,
        bounds=(l_lower - LRS_SOURCE_MARGIN_MICRON, l_upper + LRS_SOURCE_MARGIN_MICRON),
        stride=lrs_stride,
        tables=lbl_tables,
    )
    checkpoint("lrs_provider")
    hrs_grid = hrs_provider.native_spectral_grid(
        sampling=hrs_stride,
        wavelength_bounds_micron=(k_lower, k_upper),
        reference_species="H2O",
        name=f"WASP-77Ab K-band LBL stride {hrs_stride}",
    )
    checkpoint("hrs_grid")
    lrs_grid = lrs_provider.native_spectral_grid(
        sampling=lrs_stride,
        wavelength_bounds_micron=(l_lower - LRS_SOURCE_MARGIN_MICRON, l_upper + LRS_SOURCE_MARGIN_MICRON),
        reference_species="H2O",
        name=f"WASP-77Ab NIRSpec LBL stride {lrs_stride}",
    )
    checkpoint("lrs_grid")
    hrs_estimate = _guard_opacity_memory(hrs_provider, hrs_grid)
    checkpoint("hrs_prep")
    lrs_estimate = _guard_opacity_memory(lrs_provider, lrs_grid)
    checkpoint("lrs_prep")
    builder = build_atmosphere_builder()
    model_config = ParameterizedEmissionModelConfig(
        opacity_species=("H2O", "CO"),
        include_rayleigh=True,
        cia_normal_hydrogen=True,
        cia_temperature_extrapolation="clip",
        cia_spectral_extrapolation="zero",
        gas_combination="sum_by_g",
        thermal_integration_backend="numba",
        stellar_spectrum_model=stellar_model,
        hminus_continuum=HMINUS_CONFIG if hminus_enabled else None,
        aggregate_additional_optical_depths=True,
        metadata={
            "target": "WASP-77Ab",
            "composition_convention": "volume_mixing_ratio",
            "mass_fraction_parameters": "none",
            "continuum_amplitude_parameter": "none",
        },
    )
    cia = load_nemesispy_cia_table()
    previous_pysyn_cdbs = os.environ.get("PYSYN_CDBS")
    if phoenix_root is not None:
        os.environ["PYSYN_CDBS"] = str(phoenix_root)
    try:
        hrs_model = ParameterizedEmissionForwardModel(
            planet=target.PLANET,
            star=target.STAR,
            spectral_grid=hrs_grid,
            atmosphere_builder=builder,
            opacity_provider=hrs_provider,
            config=model_config,
            cia_table=cia,
        )
        checkpoint("hrs_model_preparation")
        lrs_model = ParameterizedEmissionForwardModel(
            planet=target.PLANET,
            star=target.STAR,
            spectral_grid=lrs_grid,
            atmosphere_builder=builder,
            opacity_provider=lrs_provider,
            config=model_config,
            cia_table=cia,
        )
        checkpoint("lrs_model_preparation")
    finally:
        if phoenix_root is not None:
            if previous_pysyn_cdbs is None:
                os.environ.pop("PYSYN_CDBS", None)
            else:
                os.environ["PYSYN_CDBS"] = previous_pysyn_cdbs
    broadening = _prepare_hrs_broadening(hrs_grid, hrs)
    checkpoint("broadening")
    hrs_velocity_lower, hrs_velocity_upper = _hrs_velocity_bounds(hrs)
    hrs_query_lower, hrs_query_upper = _hrs_query_wavelength_bounds(hrs)
    lrs_responses = {
        dataset.name: TopHatObservationResponse().prepare(dataset.observation)
        for dataset in nirspec.datasets
    }
    source_hashes = _real_source_hashes(
        smith_root=smith_root,
        hrs=hrs,
        lrs=nirspec,
        h2o_table=h2o_path,
        co_table=co_path,
        stride_report=stride_report,
        lbl_source_records=lbl_source_records,
        phoenix_source_records=phoenix_source_records,
    )
    checkpoint("load_complete")
    return RobertJointInputs(
        hrs_observation=hrs,
        nirspec_observations=nirspec,
        hrs_model=hrs_model,
        lrs_model=lrs_model,
        hrs_broadening=broadening,
        lrs_responses=lrs_responses,
        source_hashes=source_hashes,
        hrs_order_indices=order_indices,
        hrs_stride=hrs_stride,
        lrs_stride=lrs_stride,
        stride_sources={"hrs": hrs_source, "lrs": lrs_source},
        opacity_estimates={"hrs": hrs_estimate, "lrs": lrs_estimate},
        hminus_enabled=hminus_enabled,
        metadata={
            "stellar_spectrum_model": stellar_model,
            "pysyn_cdbs_root": "" if phoenix_root is None else str(phoenix_root),
            "phoenix_subset_verified": str(stellar_model == "phoenix").lower(),
            "hrs_velocity_prior_bounds_km_s": (
                f"{hrs_velocity_lower:.17g},{hrs_velocity_upper:.17g}"
            ),
            "hrs_template_query_bounds_micron": (
                f"{hrs_query_lower:.17g},{hrs_query_upper:.17g}"
            ),
            "lrs_accuracy_preferred_stride": str(lrs_accuracy_stride),
            "lrs_production_stride": str(lrs_stride),
            "lrs_stride_policy": lrs_stride_metadata["policy"],
            "lrs_stride_report": lrs_stride_metadata["report"],
            "lrs_stride_candidate_rms_sigma": lrs_stride_metadata[
                "candidate_rms_sigma"
            ],
            "lrs_stride_candidate_max_absolute_sigma": lrs_stride_metadata[
                "candidate_max_absolute_sigma"
            ],
            "lrs_stride_candidate_estimate_bytes": lrs_stride_metadata[
                "candidate_estimate_bytes"
            ],
            "lrs_stride_estimated_savings_bytes": lrs_stride_metadata[
                "estimated_savings_bytes_vs_accuracy_preferred"
            ],
            "hminus_window": "1.90-5.17 micron",
            "hminus_enabled": str(bool(hminus_enabled)).lower(),
            "hminus_bound_free": (
                "zero beyond 1.6421 micron" if hminus_enabled else "disabled"
            ),
            "hminus_free_free": (
                "controlled by fixed H and retrieved e- VMR"
                if hminus_enabled
                else "disabled"
            ),
        },
        memory_checkpoints=memory_checkpoints,
    )


def _apply_broadening(
    spectrum: Spectrum,
    operators: Sequence[PreparedSpectralOperator],
) -> Spectrum:
    """Apply the fixed HRS Gray plus Gaussian operators."""

    output = spectrum
    for operator in operators:
        output = operator.observe(output)
    return output


@dataclass(frozen=True)
class SharedRobertForward:
    """Evaluate HRS and LRS outputs from one atmospheric state."""

    inputs: RobertJointInputs
    mode: str = JOINT_MODE

    def __post_init__(self) -> None:
        if self.mode not in ALL_MODES:
            raise ValueError(f"mode must be one of {ALL_MODES}")

    @property
    def required_parameters(self) -> tuple[str, ...]:
        names = list(self.inputs.hrs_model.required_parameters)
        if self.mode in {HRS_MODE, JOINT_MODE}:
            names.extend((HRS_KP_PARAMETER, HRS_DVSYS_PARAMETER, HRS_SCALE_PARAMETER))
        if self.mode in {LRS_MODE, JOINT_MODE}:
            names.append(LRS_SCALE_PARAMETER)
        if len(set(names)) != len(names):
            raise RobertValidationError("real joint retrieval parameter names must be unique")
        return tuple(names)

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, object]:
        model_parameters = {
            name: float(parameters[name])
            for name in self.inputs.hrs_model.required_parameters
        }
        atmosphere = self.inputs.hrs_model.atmosphere_builder.build(model_parameters)
        output: dict[str, object] = {}
        if self.mode in {HRS_MODE, JOINT_MODE}:
            native = self.inputs.hrs_model.evaluate_atmosphere(atmosphere, model_parameters)
            broadened = _apply_broadening(native, self.inputs.hrs_broadening)
            output["hrs"] = HighResolutionEmissionTemplate(
                wavelength=broadened.spectral_grid.values,
                planet_flux=broadened.values,
                stellar_flux=np.ones(broadened.spectral_grid.size, dtype=float),
                wavelength_unit=broadened.spectral_grid.unit,
                flux_ratio_scale=1.0,
                name="ROBERT WASP-77Ab K-band LBL HRS template",
                metadata={
                    "response_stages": "Gray rotational broadening, Gaussian IGRINS LSF",
                    "opacity_mode": "tabulated_line_by_line",
                    "composition_convention": "volume_mixing_ratio",
                    "flux_ratio_source": (
                        "ROBERT eclipse_depth already includes (Rp/Rs)^2 and "
                        "the configured stellar spectrum"
                    ),
                    "stellar_flux_carrier": "unity because planet_flux is eclipse_depth",
                },
            )
        if self.mode in {LRS_MODE, JOINT_MODE}:
            native = self.inputs.lrs_model.evaluate_atmosphere(atmosphere, model_parameters)
            scale = float(parameters[LRS_SCALE_PARAMETER])
            if not np.isfinite(scale) or scale <= 0.0:
                raise RobertValidationError("LRS scale must be finite and positive")
            for name, response in self.inputs.lrs_responses.items():
                observed = response.observe(native)
                output[name] = Spectrum(
                    spectral_grid=observed.spectral_grid,
                    values=scale * observed.values,
                    unit=observed.unit,
                    observable=observed.observable,
                    metadata={
                        **dict(observed.metadata),
                        "shared_atmosphere": "true",
                        "composition_convention": "volume_mixing_ratio",
                        "lrs_scale_parameter": LRS_SCALE_PARAMETER,
                    },
                )
        return output


def _parameter_specs(mode: str) -> tuple[RetrievalParameter, ...]:
    """Return conservative bounded priors for a real-data proof run."""

    chemistry = (
        RetrievalParameter("temperature_K", UniformPrior(*TEMPERATURE_PRIOR_K), unit="K"),
        RetrievalParameter("log10_H2O_VMR", UniformPrior(-8.0, -2.0), unit="dex"),
        RetrievalParameter("log10_CO_VMR", UniformPrior(-8.0, -2.0), unit="dex"),
        RetrievalParameter("log10_Hminus_VMR", UniformPrior(-14.0, -2.0), unit="dex"),
        RetrievalParameter("log10_electron_VMR", UniformPrior(-14.0, -2.0), unit="dex"),
    )
    extras: list[RetrievalParameter] = []
    if mode in {HRS_MODE, JOINT_MODE}:
        extras.extend(
            (
                RetrievalParameter(
                    HRS_KP_PARAMETER,
                    UniformPrior(*HRS_KP_PRIOR_KM_S),
                    unit="km s^-1",
                ),
                RetrievalParameter(
                    HRS_DVSYS_PARAMETER,
                    UniformPrior(*HRS_DVSYS_PRIOR_KM_S),
                    unit="km s^-1",
                ),
                RetrievalParameter(HRS_SCALE_PARAMETER, UniformPrior(-2.0, 2.0), unit="log10"),
            )
        )
    if mode in {LRS_MODE, JOINT_MODE}:
        extras.append(RetrievalParameter(LRS_SCALE_PARAMETER, UniformPrior(0.5, 1.5)))
    return chemistry + tuple(extras)


def _lrs_components(observations: ObservationCollection) -> tuple[HeterogeneousLikelihoodComponent, ...]:
    gaussian = GaussianLikelihood(
        include_normalization=False,
        offset_parameter=None,
        jitter_parameter=None,
        uncertainty_scale_parameter=None,
    )
    return tuple(
        HeterogeneousLikelihoodComponent(
            name=dataset.name,
            prediction_key=dataset.name,
            likelihood=gaussian,
            observation=dataset.observation,
            metadata={
                "detector": str(dataset.metadata.get("detector", "unknown")),
                "covariance": "published point uncertainties; no cross-detector covariance published",
                "uncertainty_model": (
                    "symmetric Gaussian sigma is the mean of absolute published "
                    "asymmetric errors; original columns remain in observation metadata"
                ),
            },
        )
        for dataset in observations.datasets
    )


def build_real_problem(
    inputs: RobertJointInputs,
    *,
    mode: str = JOINT_MODE,
    hminus_enabled: bool | None = None,
) -> HeterogeneousRetrievalProblem:
    """Build one HRS-only, LRS-only, or joint real-data problem."""

    if mode not in ALL_MODES:
        raise ValueError(f"mode must be one of {ALL_MODES}")
    enabled = inputs.hminus_enabled if hminus_enabled is None else bool(hminus_enabled)
    if enabled != inputs.hminus_enabled:
        raise RobertValidationError(
            "hminus_enabled must match the continuum state used to build the inputs"
        )
    components: list[HeterogeneousLikelihoodComponent] = []
    if mode in {HRS_MODE, JOINT_MODE}:
        prepared = TimeResolvedHighResolutionLikelihood(
            n_components=target.PRIMARY_HRS_PC_COUNT,
            kp_parameter=HRS_KP_PARAMETER,
            dVsys_parameter=HRS_DVSYS_PARAMETER,
            dphi_parameter="dphi_fixed_zero",
            scale_parameter=HRS_SCALE_PARAMETER,
            scale_is_log10=True,
            doppler_mode="smith_nonrelativistic",
            flux_ratio_scale=1.0,
            name="ROBERT WASP-77Ab Smith PCA HRS likelihood",
        ).prepare(inputs.hrs_observation)
        components.append(
            HeterogeneousLikelihoodComponent(
                name="hrs",
                prediction_key="hrs",
                likelihood=prepared,
                metadata={
                    "method": "Smith exact order-wise PCA plus Brogi-Line statistic",
                    "k_band_subset": "1.90-2.45 micron",
                    "effective_residual_rank": str(prepared.effective_residual_rank),
                },
            )
        )
    if mode in {LRS_MODE, JOINT_MODE}:
        components.extend(_lrs_components(inputs.nirspec_observations))
    return HeterogeneousRetrievalProblem(
        name=f"wasp77ab-robert-{mode}",
        parameters=RetrievalParameterSet(_parameter_specs(mode)),
        forward_model=SharedRobertForward(inputs, mode=mode),
        likelihood=HeterogeneousLikelihood(
            tuple(components),
            name=f"wasp77ab-robert-{mode}-likelihood",
        ),
        invalid_loglike=INVALID_LOGLIKE_FLOOR,
        metadata={
            "target": "WASP-77Ab",
            "data_role": "Smith et al. 2024 IGRINS K subset plus August et al. 2023 NIRSpec G395H",
            "atmosphere": "one shared AtmosphereBuilder instance",
            "composition_convention": "volume_mixing_ratio",
            "mass_fraction_parameters": "none",
            "opacity": "tabulated H2O POKAZATEL and CO HITEMP LBL cross sections",
            "thermal_integration_backend": "numba",
            "aggregate_additional_optical_depths": "true",
            "continuum": "H2-H2/H2-He CIA plus physical H-minus bound-free/free-free",
            "hminus_enabled": str(enabled).lower(),
            "hminus_bound_free": (
                "zero throughout 1.90-5.17 micron beyond 1.6421 micron cutoff"
                if enabled
                else "disabled"
            ),
            "hminus_free_free": (
                "controlled by fixed H and retrieved electron VMR"
                if enabled
                else "disabled"
            ),
            "hminus_vmr_identifiability": "prior-dominated in selected window",
            "continuum_amplitude_parameter": "none",
            "truth_recovery_claim": "false for real data",
        },
        opacity_identifiers={
            "H2O": H2O_TABLE_SHA256,
            "CO": CO_TABLE_SHA256,
        },
    )


def _null_template(template: HighResolutionEmissionTemplate) -> HighResolutionEmissionTemplate:
    """Make a flat template with the same exact response grid."""

    return HighResolutionEmissionTemplate(
        wavelength=template.wavelength,
        planet_flux=np.zeros(template.wavelength.size, dtype=float),
        stellar_flux=np.ones(template.wavelength.size, dtype=float),
        wavelength_unit=template.wavelength_unit,
        name="flat-null-template",
    )


def _loglike_terms_with_checkpoints(
    problem: HeterogeneousRetrievalProblem,
    prediction: Mapping[str, object],
    parameters: Mapping[str, float],
    checkpoints: dict[str, int],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Evaluate components with exact dispatch and record high-water memory."""

    terms: dict[str, float] = {}
    for component in problem.likelihood.components:
        if component.prediction_key not in prediction:
            raise RobertValidationError(
                "heterogeneous forward prediction is missing component key "
                f"'{component.prediction_key}'"
            )
        if component.observation is None:
            result = component.likelihood.loglike(
                prediction[component.prediction_key],
                parameters,
            )
        else:
            result = component.likelihood.loglike(
                prediction[component.prediction_key],
                component.observation,
                parameters,
            )
        try:
            value = float(result)
        except (TypeError, ValueError, OverflowError) as error:
            raise RobertValidationError(
                f"likelihood component '{component.name}' returned a non-scalar "
                "log likelihood"
            ) from error
        if not np.isfinite(value):
            raise RobertValidationError(
                f"likelihood component '{component.name}' returned a non-finite "
                "log likelihood"
            )
        terms[component.name] = value
        checkpoints[f"{prefix}{component.name}_loglike"] = _peak_rss_bytes()
    return terms


def run_operator_preflight(
    problem: HeterogeneousRetrievalProblem,
    *,
    parameters: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Run finite-model and null checks through the real operators.

    This is an operator preflight only.  It does not claim recovery from an
    injected synthetic data set or from the real observations.  The HRS
    injection check reuses the exact streaming log-likelihood statistic and
    does not retain full diagnostic model cubes.  Low-overhead ``ru_maxrss``
    checkpoints identify the high-water stage without retaining arrays.
    """

    values = {
        parameter.name: float(parameter.prior.lower + parameter.prior.upper) / 2.0
        for parameter in problem.parameters.parameters
    }
    if parameters is not None:
        values.update({str(key): float(value) for key, value in parameters.items()})
    memory_checkpoints: dict[str, int] = {}
    prediction = problem.predict(values)
    memory_checkpoints["prediction"] = _peak_rss_bytes()
    terms = _loglike_terms_with_checkpoints(
        problem,
        prediction,
        values,
        memory_checkpoints,
    )
    finite_prediction = all(np.isfinite(value) for value in terms.values())
    injection_record: dict[str, object] = {"applicable": False}
    finite_injection_operator = True
    hrs_template = prediction.get("hrs")
    for component in problem.likelihood.components:
        if component.name != "hrs" or not isinstance(
            hrs_template,
            HighResolutionEmissionTemplate,
        ):
            continue
        injection_record["applicable"] = True
        try:
            # ``loglike_by_component`` above has already run the exact
            # streaming Smith injection/reprocessing path.  Reusing its
            # finite statistic avoids ``evaluate_model`` here, which would
            # retain five full HRS diagnostic cubes during preflight.
            statistic = float(terms[component.name])
            finite_injection_operator = bool(np.isfinite(statistic))
            injection_record.update(
                {
                    "finite": finite_injection_operator,
                    "statistic": statistic,
                    "operator": "streaming exact Smith/Brogi-Line loglike",
                    "derived_from": "prediction_loglike_by_component",
                    "full_diagnostic_cubes_stored": False,
                }
            )
            if not finite_injection_operator:
                injection_record["error"] = "streaming HRS statistic is not finite"
        except (
            RobertValidationError,
            KeyError,
            ValueError,
            FloatingPointError,
            TypeError,
        ) as error:
            finite_injection_operator = False
            injection_record["error"] = f"{type(error).__name__}: {error}"
            injection_record["finite"] = finite_injection_operator
        break
    null_terms: dict[str, float] = {}
    null_prediction = dict(prediction)
    if "hrs" in null_prediction and isinstance(null_prediction["hrs"], HighResolutionEmissionTemplate):
        null_prediction["hrs"] = _null_template(null_prediction["hrs"])
    for key, value in null_prediction.items():
        if isinstance(value, Spectrum):
            null_prediction[key] = Spectrum(
                spectral_grid=value.spectral_grid,
                values=np.zeros_like(value.values),
                unit=value.unit,
                observable=value.observable,
            )
    memory_checkpoints["null_prediction"] = _peak_rss_bytes()
    try:
        null_terms = _loglike_terms_with_checkpoints(
            problem,
            null_prediction,
            values,
            memory_checkpoints,
            prefix="null_",
        )
    except (RobertValidationError, KeyError, ValueError, FloatingPointError, TypeError) as error:
        return {
            "status": "fail",
            "message": f"null operator check failed: {type(error).__name__}: {error}",
            "prediction_loglike_by_component": terms,
            "null_loglike_by_component": {},
            "finite_prediction": finite_prediction,
            "injection_operator": injection_record,
            "finite_injection_operator": finite_injection_operator,
            "finite_null": False,
            "memory_checkpoints": memory_checkpoints,
        }
    finite_null = all(np.isfinite(value) for value in null_terms.values())
    hrs_rank = 0
    for component in problem.likelihood.components:
        if component.name == "hrs":
            value = getattr(component.likelihood, "effective_residual_rank", 0)
            hrs_rank = int(value() if callable(value) else value)
            break
    return {
        "status": (
            "pass"
            if finite_prediction and finite_injection_operator and finite_null
            else "fail"
        ),
        "message": (
            "real HRS/LRS operators accept finite model, streaming HRS injection, "
            "and null predictions; this is not injection recovery"
        ),
        "prediction_loglike_by_component": terms,
        "injection_operator": injection_record,
        "null_loglike_by_component": null_terms,
        "finite_prediction": finite_prediction,
        "finite_injection_operator": finite_injection_operator,
        "finite_null": finite_null,
        "n_hrs_points": hrs_rank,
        "memory_checkpoints": memory_checkpoints,
    }


def _weighted_quantile(values: ArrayLike, quantile: float, weights: ArrayLike | None) -> float:
    numbers = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.quantile(numbers, quantile))
    sample_weights = np.asarray(weights, dtype=float)
    order = np.argsort(numbers)
    sorted_values = numbers[order]
    sorted_weights = sample_weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.sum(sorted_weights)
    return float(np.interp(float(quantile), cumulative, sorted_values))


def _component_effective_rank(
    component: HeterogeneousLikelihoodComponent,
) -> int | None:
    """Read a child likelihood's exact retained residual rank."""

    value = getattr(component.likelihood, "effective_residual_rank", None)
    if value is None:
        return None
    try:
        if callable(value):
            if component.observation is None:
                return None
            value = value(component.observation)
        rank = int(value)
    except (TypeError, ValueError, OverflowError, RobertValidationError):
        return None
    return rank if rank > 0 else None


def _finite_summary(values: ArrayLike) -> dict[str, float | int]:
    """Return scalar moments without retaining the input array."""

    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RobertValidationError("diagnostic summary requires finite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def _lrs_residual_summary(
    component: HeterogeneousLikelihoodComponent,
    prediction: object,
    parameters: Mapping[str, float],
) -> dict[str, object]:
    """Summarise one LRS residual with the child likelihood's exact hooks.

    ``effective_inputs`` is used only for the unweighted residual and model
    summaries.  The weighted statistic always comes from the child
    likelihood's exact ``chi_square`` method.  This prevents a covariance
    child from being silently reduced to diagonal uncertainties.
    """

    observation = component.observation
    effective_inputs = getattr(component.likelihood, "effective_inputs", None)
    chi_square_method = getattr(component.likelihood, "chi_square", None)
    rank_method = getattr(component.likelihood, "effective_residual_rank", None)
    if observation is None:
        raise RobertValidationError(
            f"LRS component '{component.name}' must expose an observation"
        )
    if not callable(effective_inputs):
        raise RobertValidationError(
            f"LRS component '{component.name}' has no exact effective_inputs hook"
        )
    if not callable(chi_square_method):
        raise RobertValidationError(
            f"LRS component '{component.name}' has no exact chi_square hook; "
            "diagonal fallback is forbidden"
        )
    if not callable(rank_method):
        raise RobertValidationError(
            f"LRS component '{component.name}' has no exact retained-rank hook"
        )

    model, data, _ = effective_inputs(prediction, observation, parameters)
    model_values = np.asarray(model, dtype=float)
    data_values = np.asarray(data, dtype=float)
    if model_values.shape != data_values.shape:
        raise RobertValidationError(
            f"LRS component '{component.name}' returned mismatched effective arrays"
        )
    residual = data_values - model_values
    residual_summary = _finite_summary(residual)
    model_summary = _finite_summary(model_values)
    data_summary = _finite_summary(data_values)
    weighted_chi_square = float(
        chi_square_method(prediction, observation, parameters)
    )
    if (
        not np.isfinite(weighted_chi_square)
        or weighted_chi_square < 0.0
    ):
        raise RobertValidationError(
            f"LRS component '{component.name}' returned an invalid chi-square"
        )
    try:
        retained_rank = int(rank_method(observation))
    except (TypeError, ValueError, OverflowError, RobertValidationError) as error:
        raise RobertValidationError(
            f"LRS component '{component.name}' returned an invalid retained rank"
        ) from error
    if retained_rank <= 0:
        raise RobertValidationError(
            f"LRS component '{component.name}' retained rank must be positive"
        )
    return {
        "status": "pass",
        "count": residual_summary["count"],
        "n_points": residual_summary["count"],
        "residual_rms": residual_summary["rms"],
        "rms": residual_summary["rms"],
        "residual_mean": residual_summary["mean"],
        "residual_minimum": residual_summary["minimum"],
        "residual_maximum": residual_summary["maximum"],
        "residual_max_abs": float(np.max(np.abs(residual))),
        "chi_square": weighted_chi_square,
        "weighted_chi_square": weighted_chi_square,
        "effective_rank": retained_rank,
        "effective_retained_rank": retained_rank,
        "model_summary": model_summary,
        "data_summary": data_summary,
        "weighted_statistic_source": (
            f"{type(component.likelihood).__name__}.chi_square"
        ),
        "diagonal_fallback_used": False,
    }


def summarize_post_run_diagnostics(
    problem: object,
    *,
    best_fit_parameters: Mapping[str, float],
    posterior: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    """Create compact exact diagnostics for one sampler best fit.

    The function accepts a structural problem object so tests can use small
    fakes.  It evaluates the real likelihood components once at the best-fit
    parameter mapping.  It never stores HRS or LRS residual arrays in the
    report.  HRS is reported as the Smith/Brogi-Line log-likelihood statistic,
    not as a chi-square.  LRS weighted chi-square values come from each child
    likelihood's exact ``chi_square`` method; no diagonal likelihood is
    constructed as a fallback.
    """

    posterior_values = {} if posterior is None else {
        str(name): dict(interval)
        for name, interval in posterior.items()
    }
    abundance_names = {
        "H2O": "log10_H2O_VMR",
        "CO": "log10_CO_VMR",
        "H-": "log10_Hminus_VMR",
        "e-": "log10_electron_VMR",
    }
    abundance_intervals = {
        species: posterior_values.get(parameter_name)
        for species, parameter_name in abundance_names.items()
    }
    base: dict[str, object] = {
        "status": "unavailable",
        "best_fit_log_likelihood": None,
        "best_fit_log_likelihood_by_component": {},
        "per_component": {},
        "lrs": {},
        "hrs": None,
        "rv": {
            "Kp_km_s": None,
            "dVsys_km_s": None,
            "parameter_names": {
                "Kp": HRS_KP_PARAMETER,
                "dVsys": HRS_DVSYS_PARAMETER,
            },
        },
        "abundance_posterior_intervals": abundance_intervals,
        "full_arrays_stored": False,
    }
    if not isinstance(best_fit_parameters, Mapping) or not best_fit_parameters:
        base["reason"] = "best_fit_parameters are missing"
        return base
    try:
        parameters = {
            str(name): float(value)
            for name, value in best_fit_parameters.items()
        }
        if not all(np.isfinite(value) for value in parameters.values()):
            raise RobertValidationError("best-fit parameters must be finite")
        base["rv"] = {
            "Kp_km_s": parameters.get(HRS_KP_PARAMETER),
            "dVsys_km_s": parameters.get(HRS_DVSYS_PARAMETER),
            "parameter_names": {
                "Kp": HRS_KP_PARAMETER,
                "dVsys": HRS_DVSYS_PARAMETER,
            },
        }
        prediction = problem.predict(parameters)
        likelihood = problem.likelihood
        terms = likelihood.loglike_by_component(prediction, parameters)
    except (
        RobertValidationError,
        KeyError,
        ValueError,
        TypeError,
        FloatingPointError,
        OverflowError,
    ) as error:
        base["status"] = "fail"
        base["reason"] = f"best-fit diagnostic evaluation failed: {type(error).__name__}: {error}"
        return base

    component_records: dict[str, dict[str, object]] = {}
    lrs_records: dict[str, dict[str, object]] = {}
    hrs_record: dict[str, object] | None = None
    diagnostic_errors: list[str] = []
    for component in likelihood.components:
        name = str(component.name)
        try:
            loglike = float(terms[name])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            diagnostic_errors.append(
                f"{name}: missing best-fit log likelihood ({type(error).__name__})"
            )
            continue
        if not np.isfinite(loglike):
            diagnostic_errors.append(f"{name}: best-fit log likelihood is not finite")
            continue
        rank = _component_effective_rank(component)
        component_record: dict[str, object] = {
            "prediction_key": component.prediction_key,
            "loglike": loglike,
            "effective_retained_rank": rank,
            "observation_embedded": component.observation is None,
        }
        is_hrs = name == "hrs" or callable(
            getattr(component.likelihood, "cross_correlation", None)
        )
        if is_hrs:
            local: dict[str, object] = {
                "status": "pass" if rank is not None else "fail",
                "statistic_name": "smith_brogi_line_log_likelihood",
                "statistic_kind": "log_likelihood_not_chi_square",
                "statistic_value": loglike,
                "loglike": loglike,
                "chi_square": None,
                "chi_square_defined": False,
                "retained_rank": rank,
                "effective_retained_rank": rank,
            }
            if rank is None:
                local["error"] = "HRS likelihood did not expose a positive retained rank"
                diagnostic_errors.append(str(local["error"]))
            velocity_method = getattr(component.likelihood, "velocity", None)
            if callable(velocity_method):
                try:
                    velocity = np.asarray(velocity_method(parameters), dtype=float)
                    local["velocity_summary_km_s"] = _finite_summary(velocity)
                except (
                    RobertValidationError,
                    ValueError,
                    TypeError,
                    FloatingPointError,
                    OverflowError,
                ) as error:
                    local["velocity_summary_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
            ccf_method = getattr(component.likelihood, "cross_correlation", None)
            if callable(ccf_method):
                try:
                    ccf = np.asarray(ccf_method(prediction[component.prediction_key], parameters), dtype=float)
                    finite_ccf = ccf[np.isfinite(ccf)]
                    if finite_ccf.size:
                        local["ccf_summary"] = _finite_summary(finite_ccf)
                except (
                    RobertValidationError,
                    KeyError,
                    ValueError,
                    TypeError,
                    FloatingPointError,
                    OverflowError,
                ) as error:
                    local["ccf_summary_error"] = f"{type(error).__name__}: {error}"
            component_record["hrs_local_likelihood"] = local
            hrs_record = local
        else:
            try:
                residual = _lrs_residual_summary(
                    component,
                    prediction[component.prediction_key],
                    parameters,
                )
            except (
                RobertValidationError,
                KeyError,
                ValueError,
                TypeError,
                FloatingPointError,
                OverflowError,
            ) as error:
                residual = {
                    "status": "fail",
                    "error": f"{type(error).__name__}: {error}",
                    "diagonal_fallback_used": False,
                }
                diagnostic_errors.append(f"{name}: {residual['error']}")
            component_record["lrs_residual"] = residual
            lrs_records[name] = residual
        component_records[name] = component_record

    base["status"] = "pass" if not diagnostic_errors else "fail"
    base["best_fit_log_likelihood"] = float(
        np.sum(np.fromiter(terms.values(), dtype=float))
    )
    base["best_fit_log_likelihood_by_component"] = {
        name: float(value) for name, value in terms.items()
    }
    base["per_component"] = component_records
    base["lrs"] = lrs_records
    base["hrs"] = hrs_record
    base["full_arrays_stored"] = False
    if diagnostic_errors:
        base["errors"] = diagnostic_errors
    return base


def _posterior_summary(result: object) -> dict[str, Mapping[str, float]]:
    samples = np.asarray(getattr(result, "samples"), dtype=float)
    names = tuple(getattr(result, "parameter_names"))
    weights = getattr(result, "weights", None)
    return {
        str(name): {
            "median": _weighted_quantile(samples[:, index], 0.5, weights),
            "lower_90": _weighted_quantile(samples[:, index], 0.05, weights),
            "upper_90": _weighted_quantile(samples[:, index], 0.95, weights),
        }
        for index, name in enumerate(names)
    }


def _metadata_boolean(value: object, *, name: str) -> bool:
    """Parse one strict metadata boolean without Python truth-value coercion."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise RobertValidationError(f"metadata field '{name}' must be true or false")


def _run_record(
    result: object,
    *,
    problem: HeterogeneousRetrievalProblem,
    seed: int,
    elapsed: float,
    resources: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    metadata = getattr(result, "metadata", {})
    method = str(getattr(result, "method", "")).strip().lower()
    raw_calls = metadata.get("likelihood_evaluations") if hasattr(metadata, "get") else None
    try:
        calls = None if raw_calls is None else int(raw_calls)
    except (TypeError, ValueError):
        calls = None
    best_fit_parameters = {
        str(name): float(value)
        for name, value in getattr(result, "best_fit_parameters", {}).items()
    }
    posterior = _posterior_summary(result)
    diagnostics = summarize_post_run_diagnostics(
        problem,
        best_fit_parameters=best_fit_parameters,
        posterior=posterior,
    )
    diagnostics_passed = diagnostics.get("status") == "pass"
    return {
        "seed": int(seed),
        "output_dir": str(output_dir),
        "method": method,
        "sampler_method_pass": method == "multinest",
        "status": (
            "pass"
            if (
                bool(getattr(result, "converged", False))
                and method == "multinest"
                and diagnostics_passed
            )
            else "fail"
        ),
        "converged": bool(getattr(result, "converged", False)),
        "message": str(getattr(result, "message", "")),
        "calls": calls,
        "wall_time_seconds": float(elapsed),
        "peak_rss_bytes": resources.get("peak_rss_bytes"),
        "resource_gate_passed": bool(resources.get("gate_passed", False)),
        "diagnostics_gate_passed": diagnostics_passed,
        "log_evidence": None if getattr(result, "log_evidence", None) is None else float(result.log_evidence),
        "log_evidence_error": None if getattr(result, "log_evidence_error", None) is None else float(result.log_evidence_error),
        "best_fit_log_likelihood": diagnostics.get("best_fit_log_likelihood"),
        "best_fit_parameters": best_fit_parameters,
        "posterior": posterior,
        "diagnostics": diagnostics,
    }


def run_sampler_pair(
    problem: HeterogeneousRetrievalProblem,
    *,
    mode: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_memory_bytes: int = DEFAULT_PROCESS_RSS_LIMIT_BYTES,
    sampler_runner: object | None = None,
) -> dict[str, object]:
    """Run the two fixed PyMultiNest seeds for one real-data mode."""

    if mode not in ALL_MODES:
        raise ValueError(f"mode must be one of {ALL_MODES}")
    before = resource_record(max_memory_bytes=max_memory_bytes)
    if not bool(before["gate_passed"]):
        raise RuntimeError("resource contract failed before PyMultiNest")
    if sampler_runner is None:
        from robert_exoplanets.retrieval.samplers import run_multinest

        sampler_runner = run_multinest
    if not callable(sampler_runner):
        raise TypeError("sampler_runner must be callable")
    records: list[dict[str, object]] = []
    hminus_state = _metadata_boolean(
        problem.metadata.get("hminus_enabled", True),
        name="hminus_enabled",
    )
    for seed in SEEDS:
        output_dir = Path(output_root) / mode / f"seed-{seed}"
        started = perf_counter()
        try:
            result = sampler_runner(
                problem,
                output_dir=output_dir,
                n_live_points=N_LIVE_POINTS,
                max_iter=MAX_ITER,
                evidence_tolerance=EVIDENCE_TOLERANCE,
                sampling_efficiency=SAMPLING_EFFICIENCY,
                mpi_nprocs=MPI_PROCESSES,
                seed=seed,
                resume=False,
                verbose=False,
                invalid_loglike_floor=INVALID_LOGLIKE_FLOOR,
            )
        except Exception as error:
            resources = resource_record(max_memory_bytes=max_memory_bytes)
            records.append(
                {
                    "seed": int(seed),
                    "hminus_enabled": hminus_state,
                    "output_dir": str(output_dir),
                    "method": "",
                    "sampler_method_pass": False,
                    "status": "fail",
                    "converged": False,
                    "message": f"{type(error).__name__}: {error}",
                    "calls": None,
                    "wall_time_seconds": max(perf_counter() - started, 0.0),
                    "peak_rss_bytes": resources.get("peak_rss_bytes"),
                    "resource_gate_passed": bool(resources.get("gate_passed", False)),
                    "diagnostics_gate_passed": False,
                    "log_evidence": None,
                    "log_evidence_error": None,
                    "best_fit_log_likelihood": None,
                    "best_fit_parameters": {},
                    "posterior": {},
                    "diagnostics": {
                        "status": "unavailable",
                        "reason": "sampler did not return a result",
                        "full_arrays_stored": False,
                    },
                }
            )
            if not bool(records[-1]["resource_gate_passed"]):
                break
            continue
        resources = resource_record(max_memory_bytes=max_memory_bytes)
        record = _run_record(
            result,
            problem=problem,
            seed=seed,
            elapsed=max(perf_counter() - started, 0.0),
            resources=resources,
            output_dir=output_dir,
        )
        record["hminus_enabled"] = hminus_state
        records.append(record)
        if not bool(record["resource_gate_passed"]):
            break
    if len(records) == 2 and all(record["log_evidence"] is not None for record in records):
        delta = abs(float(records[0]["log_evidence"]) - float(records[1]["log_evidence"]))
        errors = [record["log_evidence_error"] for record in records]
        combined_error = None if any(value is None for value in errors) else float(np.hypot(*[float(value) for value in errors]))
    else:
        delta = None
        combined_error = None
    gate = bool(
        len(records) == 2
        and all(
            bool(record["converged"])
            and bool(record["sampler_method_pass"])
            and bool(record["resource_gate_passed"])
            and bool(record.get("diagnostics_gate_passed", False))
            for record in records
        )
        and delta is not None
        and delta <= EVIDENCE_TOLERANCE
    )
    return {
        "mode": mode,
        "hminus_enabled": hminus_state,
        "runs": records,
        "evidence_comparison": {
            "seed_pair": list(SEEDS),
            "delta_log_evidence": delta,
            "combined_error": combined_error,
            "repeatability_limit": EVIDENCE_TOLERANCE,
            "gate_passed": gate,
        },
        "gate_passed": gate,
    }


def _contract_record(mode: str, *, hminus_enabled: bool = True) -> dict[str, object]:
    names = [parameter.name for parameter in _parameter_specs(mode)]
    components = ["hrs"] if mode == HRS_MODE else [dataset for dataset in ("nirspec_g395h_nrs1", "nirspec_g395h_nrs2")]
    if mode == JOINT_MODE:
        components = ["hrs", "nirspec_g395h_nrs1", "nirspec_g395h_nrs2"]
    return {
        "name": f"wasp77ab-robert-{mode}",
        "parameter_names": names,
        "component_names": components,
        "opacity_mode": "tabulated_H2O_CO_line_by_line",
        "lrs_opacity_mode": "tabulated_LBL_on_native_grid_then_top_hat_bins",
        "composition_convention": "volume_mixing_ratio",
        "mass_fraction_parameters": "none",
        "neutral_h_vmr": {
            "retrieval": "fixed",
            "value": FIXED_NEUTRAL_H_VMR,
            "reason": "break the H*electron free-free degeneracy",
        },
        "hminus": {
            "enabled": hminus_enabled,
            "species": ["H-", "H", "e-"],
            "bound_free_cutoff_micron": 1.6421,
            "bound_free_in_selected_window": "zero" if hminus_enabled else "disabled",
            "free_free_controls": ["fixed H", "e-"] if hminus_enabled else [],
            "hminus_vmr_identifiability": "prior_dominated",
            "continuum_amplitude_parameter": "none",
        },
    }


def write_report(report: Mapping[str, object], path: Path) -> None:
    """Write a finite JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_workflow(
    *,
    smith_data_root: Path = DEFAULT_SMITH_DATA_ROOT,
    nirspec_data_root: Path = DEFAULT_NIRSPEC_DATA_ROOT,
    h2o_table: Path = DEFAULT_H2O_TABLE,
    co_table: Path = DEFAULT_CO_TABLE,
    stride_report: Path | None = DEFAULT_STRIDE_REPORT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path | None = None,
    modes: Sequence[str] = ALL_MODES,
    stellar_spectrum_model: str = "phoenix",
    pysyn_cdbs_root: Path | str | None = None,
    preflight: bool = False,
    run_sampler: bool = False,
    dry_run: bool = True,
    max_memory_bytes: int = DEFAULT_PROCESS_RSS_LIMIT_BYTES,
    sampler_runner: object | None = None,
    hminus_enabled: bool = True,
) -> dict[str, object]:
    """Run contract, real preflight, or the bounded two-seed sampler."""

    selected_modes = ALL_MODES if tuple(modes) == ("all",) else tuple(modes)
    if not selected_modes or any(mode not in ALL_MODES for mode in selected_modes):
        raise ValueError(f"modes must be drawn from {ALL_MODES}")
    if len(set(selected_modes)) != len(selected_modes):
        raise ValueError("modes must be unique")
    if not isinstance(hminus_enabled, bool):
        raise ValueError("hminus_enabled must be boolean")
    stellar_model = str(stellar_spectrum_model).strip().lower()
    if stellar_model not in {"phoenix", "blackbody"}:
        raise ValueError("stellar_spectrum_model must be 'phoenix' or 'blackbody'")
    if preflight or run_sampler:
        dry_run = False
    inputs: RobertJointInputs | None = None
    records: dict[str, object] = {}
    preflight_records: dict[str, object] = {}
    memory_checkpoints: dict[str, int] = {}
    if preflight or run_sampler:
        inputs = load_real_inputs(
            smith_data_root=Path(smith_data_root),
            nirspec_data_root=Path(nirspec_data_root),
            h2o_table=Path(h2o_table),
            co_table=Path(co_table),
            stride_report=stride_report,
            stellar_spectrum_model=stellar_model,
            pysyn_cdbs_root=pysyn_cdbs_root,
            hminus_enabled=hminus_enabled,
        )
        memory_checkpoints.update(
            {
                str(name): int(value)
                for name, value in inputs.memory_checkpoints.items()
            }
        )
        for mode in selected_modes:
            problem = build_real_problem(
                inputs,
                mode=mode,
                hminus_enabled=hminus_enabled,
            )
            memory_checkpoints[f"{mode}_likelihood_preparation"] = _peak_rss_bytes()
            preflight_records[mode] = run_operator_preflight(problem)
            mode_checkpoints = preflight_records[mode].get("memory_checkpoints", {})
            if isinstance(mode_checkpoints, Mapping):
                memory_checkpoints.update(
                    {
                        f"{mode}_{name}": int(value)
                        for name, value in mode_checkpoints.items()
                    }
                )
            if run_sampler:
                records[mode] = run_sampler_pair(
                    problem,
                    mode=mode,
                    output_root=Path(output_root),
                    max_memory_bytes=max_memory_bytes,
                    sampler_runner=sampler_runner,
                )
    resources = resource_record(max_memory_bytes=max_memory_bytes, check_peak=not dry_run)
    resources["memory_checkpoints"] = dict(memory_checkpoints)
    if run_sampler:
        status = "pass" if records and all(bool(value.get("gate_passed")) for value in records.values()) and bool(resources["gate_passed"]) else "fail"
    elif preflight:
        status = "preflight_pass" if preflight_records and all(value.get("status") == "pass" for value in preflight_records.values()) and bool(resources["gate_passed"]) else "fail"
    else:
        status = "dry_run"
    source_hashes = {} if inputs is None else dict(inputs.source_hashes)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": status,
        "claim": "real-data ROBERT shared-VMR workflow for WASP-77Ab; real data have no truth-recovery claim",
        "target": {
            "name": "WASP-77Ab",
            "paper": target.SMITH_PAPER_URL,
            "zenodo": target.SMITH_ZENODO_URL,
            "nirspec_input_role": "August et al. 2023 public NIRSpec table used for the Smith comparison",
        },
        "stellar_policy": {
            "model": stellar_model,
            "pysyn_cdbs_required": stellar_model == "phoenix",
            "pysyn_cdbs_root": (
                None
                if inputs is None
                else inputs.metadata.get("pysyn_cdbs_root") or None
            ),
            "phoenix_subset": (
                "three exact STScI files verified before opacity preparation"
                if stellar_model == "phoenix"
                else "not required for explicit blackbody mode"
            ),
        },
        "sampler_policy": {
            "sampler": "PyMultiNest",
            "backend": "MultiNest",
            "seeds": list(SEEDS),
            "n_live_points": N_LIVE_POINTS,
            "max_iter": MAX_ITER,
            "evidence_tolerance": EVIDENCE_TOLERANCE,
            "sampling_efficiency": SAMPLING_EFFICIENCY,
            "mpi_processes": MPI_PROCESSES,
            "thread_variables": list(THREAD_VARIABLES),
            "max_threads": MAX_THREADS,
        },
        "composition_policy": {
            "robert_convention": "volume_mixing_ratio",
            "mass_fraction_parameters_in_robert": False,
            "mass_fraction_parameters": False,
            "shared_atmosphere": True,
            "chemistry": ["H2O", "CO", "H-", "H", "e-", "H2", "He"],
            "fixed_neutral_h_vmr": FIXED_NEUTRAL_H_VMR,
        },
        "opacity_policy": {
            "hrs": "tabulated H2O POKAZATEL and CO HITEMP R=1e6 LBL",
            "lrs": "same tabulated LBL tables on a measured/provisional sparse native grid, then top-hat NIRSpec bins",
            "lrs_production_stride_policy": (
                "report-validated stride 100 for the laptop joint process; "
                "the finer report selection remains recorded as accuracy-preferred"
            ),
            "lrs_uncertainty_model": (
                "symmetric Gaussian sigma from the mean of absolute published "
                "asymmetric errors; original errors remain in source metadata"
            ),
            "opacity_memory_limit_bytes": OPACITY_MEMORY_LIMIT_BYTES,
            "stride_report": str(stride_report) if stride_report is not None else None,
        },
        "hminus_policy": {
            "enabled": hminus_enabled,
            "window_micron": list((1.90, 5.17)),
            "bound_free_cutoff_micron": 1.6421,
            "bound_free_in_window": "zero" if hminus_enabled else "disabled",
            "free_free_control": (
                "fixed H and retrieved electron VMR" if hminus_enabled else "disabled"
            ),
            "hminus_vmr_status": "prior_dominated_in_selected_window",
            "continuum_amplitude_parameter": "none",
            "same_parameterization_for_on_off_comparison": True,
        },
        "modes": list(selected_modes),
        "problems": {
            mode: _contract_record(mode, hminus_enabled=hminus_enabled)
            for mode in selected_modes
        } if inputs is None else {
            mode: {
                **_contract_record(mode, hminus_enabled=hminus_enabled),
                "parameter_names": [
                    parameter.name for parameter in _parameter_specs(mode)
                ],
            }
            for mode in selected_modes
        },
        "data": {
            "hrs_subset": "Smith et al. 2024 2020-12-14 pre-eclipse IGRINS orders wholly inside 1.90-2.45 micron",
            "lrs_detectors": ["NRS1", "NRS2"],
            "source_hashes": source_hashes,
            "hrs_order_indices": None if inputs is None else list(inputs.hrs_order_indices),
            "hrs_point_count": None if inputs is None else inputs.hrs_observation.n_points,
            "lrs_point_count": None if inputs is None else inputs.nirspec_observations.n_points,
            "hrs_response": {
                "rotation_vsin_km_s": HRS_ROTATION_KM_S,
                "gaussian_resolving_power": HRS_LSF_RESOLVING_POWER,
                "stage_order": "Gray rotation then Gaussian LSF",
                "velocity_prior_bounds_km_s": None
                if inputs is None
                else inputs.metadata.get("hrs_velocity_prior_bounds_km_s"),
                "template_query_bounds_micron": None
                if inputs is None
                else inputs.metadata.get("hrs_template_query_bounds_micron"),
                "coverage_guard": "output response grid covers all sampled wavelengths",
            },
        },
        "stride_selection": {
            "hrs": None if inputs is None else inputs.hrs_stride,
            "lrs": None if inputs is None else inputs.lrs_stride,
            "lrs_accuracy_preferred": None
            if inputs is None
            else int(inputs.metadata["lrs_accuracy_preferred_stride"]),
            "lrs_production": None
            if inputs is None
            else int(inputs.metadata["lrs_production_stride"]),
            "lrs_policy": None
            if inputs is None
            else inputs.metadata["lrs_stride_policy"],
            "lrs_validation": None
            if inputs is None
            else {
                "report": inputs.metadata["lrs_stride_report"],
                "rms_sigma": float(
                    inputs.metadata["lrs_stride_candidate_rms_sigma"]
                ),
                "max_absolute_sigma": float(
                    inputs.metadata["lrs_stride_candidate_max_absolute_sigma"]
                ),
                "preparation_estimate_bytes": int(
                    inputs.metadata["lrs_stride_candidate_estimate_bytes"]
                ),
                "estimated_savings_bytes_vs_accuracy_preferred": int(
                    inputs.metadata["lrs_stride_estimated_savings_bytes"]
                ),
            },
            "sources": {} if inputs is None else dict(inputs.stride_sources),
            "release_claim": False if inputs is None or any("safe_default" in source for source in inputs.stride_sources.values()) else True,
        },
        "preflight": preflight_records,
        "runs": records,
        "diagnostics": {
            "posterior_predictive": (
                "compact best-fit model summaries are recorded per seed; full "
                "posterior-predictive arrays are not retained"
            ),
            "residual_products": (
                "compact per-component residual/model summaries are recorded per "
                "seed; full residual arrays are not retained"
            ),
            "best_fit_log_likelihood_by_component": (
                "recorded in each seed diagnostic using the exact heterogeneous "
                "likelihood dispatch"
            ),
            "hrs_statistic": (
                "Smith/Brogi-Line local log-likelihood and retained rank are "
                "recorded per seed; this statistic is not a chi-square"
            ),
            "hrs_chi_square": "not defined for the Smith PCA/Brogi-Line statistic",
            "lrs_residuals": (
                "NRS1/NRS2 residual RMS, exact child-likelihood weighted "
                "chi-square, and retained rank are recorded per seed; no "
                "diagonal covariance fallback is used"
            ),
            "lrs_chi_square": (
                "recorded per detector from the actual Gaussian likelihood "
                "chi_square hook"
            ),
            "rv_diagnostic": (
                "best-fit HRS Kp/dVsys and compact per-frame velocity summaries "
                "are recorded per seed; no separate RV time-series product is "
                "produced"
            ),
            "abundance_posterior_intervals": (
                "CO, H2O, H-, and electron log10-VMR 90-percent intervals are "
                "recorded per seed"
            ),
            "real_data_truth_recovery": False,
        },
        "resources": resources,
        "memory_checkpoints": dict(memory_checkpoints),
        "real_data_loaded": inputs is not None,
        "hminus_enabled": hminus_enabled,
    }
    if report_path is not None:
        write_report(report, Path(report_path))
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *ALL_MODES), default="all")
    parser.add_argument("--smith-data-root", type=Path, default=DEFAULT_SMITH_DATA_ROOT)
    parser.add_argument("--nirspec-data-root", type=Path, default=DEFAULT_NIRSPEC_DATA_ROOT)
    parser.add_argument("--h2o-table", type=Path, default=DEFAULT_H2O_TABLE)
    parser.add_argument("--co-table", type=Path, default=DEFAULT_CO_TABLE)
    parser.add_argument("--stride-report", type=Path, default=DEFAULT_STRIDE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--stellar-spectrum-model", choices=("phoenix", "blackbody"), default="phoenix")
    parser.add_argument(
        "--pysyn-cdbs-root",
        type=Path,
        default=None,
        help="explicit STScI PYSYN_CDBS root containing grid/phoenix (PHOENIX mode only)",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-pymultinest", action="store_true")
    parser.add_argument(
        "--hminus",
        choices=("on", "off"),
        default="on",
        help="enable or disable physical H-minus opacity while retaining its VMR parameter",
    )
    parser.add_argument("--max-memory-gib", type=float, default=DEFAULT_PROCESS_RSS_LIMIT_BYTES / 1024**3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = ALL_MODES if args.mode == "all" else (args.mode,)
    report = run_workflow(
        smith_data_root=args.smith_data_root,
        nirspec_data_root=args.nirspec_data_root,
        h2o_table=args.h2o_table,
        co_table=args.co_table,
        stride_report=args.stride_report,
        output_root=args.output_root,
        report_path=args.report,
        modes=modes,
        stellar_spectrum_model=args.stellar_spectrum_model,
        pysyn_cdbs_root=args.pysyn_cdbs_root,
        preflight=args.preflight,
        run_sampler=args.run_pymultinest,
        dry_run=not (args.preflight or args.run_pymultinest),
        max_memory_bytes=int(float(args.max_memory_gib) * 1024**3),
        hminus_enabled=args.hminus == "on",
    )
    print(json.dumps({"status": report["status"], "modes": report["modes"]}, sort_keys=True))
    return 0 if report["status"] in {"dry_run", "preflight_pass", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
