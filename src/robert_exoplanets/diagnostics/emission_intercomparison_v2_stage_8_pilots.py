"""Controlled Stage-8 contract and archived future-study timing pilots.

This module deliberately contains no radiative-transfer implementation.  It
freezes the current common-denominator study and preserves the broader timing
pilot contract as resource evidence for a future scattering study.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


TRACK = "track_b_native_scattering"
SCHEMA_VERSION = "2.0.0"
PRIMARY_CELLS = 80
CELL_COUNTS = (40, 80, 160)
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
PILOT_PROFILE = "pg14_non_inverted"
PILOT_PLACEMENT = "deck_top_10mbar_slope_0"
OMEGA0_LADDER = (0.5, 0.9, 0.99)
TAU_LADDER = (0.1, 1.0, 10.0, 100.0)
ACCEPTED_MODERATE_TAU = (0.1, 1.0)
UNRESOLVED_STRESS_TAU = (10.0, 100.0)
ASYMMETRY_LADDER = (0.3, 0.6, 0.9)
DELTA_M_OPTIONS = (False, True)
REFERENCE_ANGLE_COUNTS = (16, 32)
FLOAT_PRECISION = "float64"
TIMING_CLOCK = "time.perf_counter"
ASSEMBLY_OVERHEAD_FACTOR = 1.25
SHARDS_PER_PATH = len(PROFILES) * len(CELL_COUNTS)
SUBSECTION_WALL_LIMIT_S = 7200.0
COMBINED_WALL_LIMIT_S = 21600.0
RSS_AVAILABLE_FRACTION_LIMIT = 0.60
CURRENT_STAGE_8_STATUS = "controlled_isotropic_grey_aerosol_study"
FUTURE_STUDY_STATUS = "future_study"
CONTROLLED_PROFILE_GRIDS = (
    ("pg14_non_inverted", 40),
    ("pg14_non_inverted", 80),
    ("pg14_non_inverted", 160),
    ("isothermal", 80),
)
CONTROLLED_SHARDS_PER_PATH = len(CONTROLLED_PROFILE_GRIDS)
CONTROLLED_STATES_PER_SHARD = 3
CONTROLLED_CASES_PER_PATH = CONTROLLED_SHARDS_PER_PATH * CONTROLLED_STATES_PER_SHARD


@dataclass(frozen=True)
class PilotPath:
    """One genuine native framework/solver path or an explicit boundary."""

    subsection: Literal["8A", "8B", "8C", "8D", "8E"]
    framework: Literal["robert", "picaso", "petitradtrans"]
    path: str
    supported: bool
    full_case_count: int
    stream_or_angle_count: int
    pilot_omega0: float
    pilot_tau: float
    pilot_g: float
    delta_m: bool | None
    capability_note: str


@dataclass(frozen=True)
class PlannedCloudScope:
    """Archived material-specific scope for a future cloud study."""

    material: str
    optical_constants: str
    comparison: str
    shared_definition: str
    varied_definition: str
    common_cloud_tensors_allowed: bool


PLANNED_CLOUD_SCOPE = PlannedCloudScope(
    material="MgSiO3",
    optical_constants="data/optical_constants/exo_skryer/MgSiO3.txt",
    comparison="targeted independent native-code cloud-parameterization interpretation",
    shared_definition="material identity and measured refractive-index provenance",
    varied_definition="each framework's genuinely supported native cloud parameterization",
    common_cloud_tensors_allowed=False,
)


@dataclass(frozen=True)
class ControlledCloudState:
    """One state in the current idealized Stage-8 cloud experiment."""

    name: Literal["clear", "absorbing_cloud", "scattering_cloud"]
    cloud_tau_at_5_micron: float
    single_scattering_albedo: float
    asymmetry_factor: float


@dataclass(frozen=True)
class ControlledSolverPath:
    """Simplest completed native scattering path for one framework."""

    framework: Literal["robert", "picaso", "petitradtrans"]
    path: str
    solver: str
    stream_count: int | None
    emission_angle_count: int | None
    delta_m: bool | None


CONTROLLED_CLOUD_STATES = (
    ControlledCloudState("clear", 0.0, 0.0, 0.0),
    ControlledCloudState("absorbing_cloud", 1.0, 0.0, 0.0),
    ControlledCloudState("scattering_cloud", 1.0, 0.9, 0.0),
)

CONTROLLED_SOLVER_PATHS = (
    ControlledSolverPath("robert", "sh4_p3_isotropic", "SH4/P3", 4, 8, False),
    ControlledSolverPath("picaso", "sh4_isotropic", "SH4", 4, None, False),
    ControlledSolverPath(
        "petitradtrans",
        "feautrier_isotropic",
        "Feautrier isotropic scattering",
        None,
        8,
        None,
    ),
)


def _path(
    subsection: Literal["8A", "8B", "8C", "8D", "8E"],
    framework: Literal["robert", "picaso", "petitradtrans"],
    path: str,
    *,
    cases: int,
    order: int,
    omega0: float,
    tau: float,
    g: float,
    delta_m: bool | None,
    note: str,
    supported: bool = True,
) -> PilotPath:
    return PilotPath(
        subsection,
        framework,
        path,
        supported,
        cases if supported else 0,
        order,
        omega0,
        tau,
        g,
        delta_m,
        note,
    )


# Frozen before timing and now retained for a future study. Counts include
# three profiles, three pressure grids, controls, science cases, plots, and
# assembly through the overhead factor.
PILOT_PATHS = (
    _path("8A", "robert", "native_absorption", cases=18, order=8, omega0=0.0, tau=1.0, g=0.0, delta_m=None, note="exact omega0=0 regression and scattering plumbing"),
    _path("8A", "picaso", "sh4_exact_omega0_zero", cases=18, order=4, omega0=0.0, tau=1.0, g=0.0, delta_m=False, note="native resort-rebin SH4 absorption regression"),
    _path("8A", "petitradtrans", "native_ck_absorption", cases=18, order=8, omega0=0.0, tau=1.0, g=0.0, delta_m=None, note="native correlated-k absorption regression"),
    _path("8B", "robert", "toon_isotropic", cases=126, order=2, omega0=0.9, tau=1.0, g=0.0, delta_m=None, note="native two-stream isotropic ladder"),
    _path("8B", "robert", "sh4_p3_isotropic", cases=126, order=4, omega0=0.9, tau=1.0, g=0.0, delta_m=False, note="native SH4/P3 isotropic ladder"),
    _path("8B", "picaso", "toon_isotropic", cases=126, order=2, omega0=0.9, tau=1.0, g=0.0, delta_m=False, note="native Toon isotropic ladder"),
    _path("8B", "picaso", "sh4_isotropic", cases=126, order=4, omega0=0.9, tau=1.0, g=0.0, delta_m=False, note="native SH4 isotropic ladder"),
    _path("8B", "petitradtrans", "feautrier_isotropic", cases=126, order=8, omega0=0.9, tau=1.0, g=0.0, delta_m=None, note="native Feautrier isotropic additional scattering"),
    _path("8C", "robert", "toon_anisotropic", cases=45, order=2, omega0=0.9, tau=1.0, g=0.6, delta_m=None, note="native two-stream anisotropy; no delta-M switch"),
    _path("8C", "robert", "sh4_delta_off", cases=45, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=False, note="native SH4/P3 anisotropy"),
    _path("8C", "robert", "sh4_delta_on", cases=45, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=True, note="native SH4/P3 anisotropy with delta-M"),
    _path("8C", "picaso", "toon_delta_off", cases=45, order=2, omega0=0.9, tau=1.0, g=0.6, delta_m=False, note="native Toon anisotropy"),
    _path("8C", "picaso", "toon_delta_on", cases=45, order=2, omega0=0.9, tau=1.0, g=0.6, delta_m=True, note="native Toon delta-Eddington"),
    _path("8C", "picaso", "sh4_delta_off", cases=45, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=False, note="native SH4 anisotropy"),
    _path("8C", "picaso", "sh4_delta_on", cases=45, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=True, note="native SH4 delta-Eddington"),
    _path("8C", "petitradtrans", "arbitrary_anisotropy_delta_m", cases=0, order=0, omega0=0.9, tau=1.0, g=0.6, delta_m=None, note="additional scattering callback is isotropic; no arbitrary g, delta-M, or order control", supported=False),
    _path("8D", "robert", "spectral_hg_sh4", cases=27, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=False, note="wavelength-dependent omega0/g HG moments"),
    _path("8D", "robert", "physical_mie_exact_moments_sh4", cases=27, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=False, note="shared physical Mie inputs with exact native phase moments"),
    _path("8D", "picaso", "spectral_hg_sh4", cases=27, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=False, note="wavelength-dependent native cloud omega0/g"),
    _path("8D", "picaso", "physical_mie_virga_hg_sh4", cases=27, order=4, omega0=0.9, tau=1.0, g=0.6, delta_m=False, note="shared physical n/k/size inputs through genuine Virga Mie, represented natively by g"),
    _path("8D", "petitradtrans", "spectral_isotropic_feautrier", cases=27, order=8, omega0=0.9, tau=1.0, g=0.0, delta_m=None, note="wavelength-dependent omega0 with isotropic additional scattering"),
    _path("8D", "picaso", "native_atmospheric_microphysics", cases=0, order=0, omega0=0.9, tau=1.0, g=0.6, delta_m=None, note="local reference installation lacks required native cloud-atmosphere assets", supported=False),
    _path("8D", "petitradtrans", "native_microphysical_cloud", cases=0, order=0, omega0=0.9, tau=1.0, g=0.6, delta_m=None, note="local input-data tree contains no cloud opacity assets", supported=False),
    _path("8E", "robert", "high_order_reference", cases=0, order=0, omega0=0.99, tau=10.0, g=0.0, delta_m=None, note="no genuine 16+ stream, adding/doubling, matrix-operator, or Monte Carlo solver", supported=False),
    _path("8E", "picaso", "high_order_reference", cases=0, order=0, omega0=0.99, tau=10.0, g=0.0, delta_m=None, note="installed API supports Toon two-stream and SH4 only", supported=False),
    _path("8E", "petitradtrans", "feautrier_16_angles", cases=27, order=16, omega0=0.99, tau=10.0, g=0.0, delta_m=None, note="native Feautrier with 16-point emission-angle quadrature"),
    _path("8E", "petitradtrans", "feautrier_32_angles", cases=27, order=32, omega0=0.99, tau=10.0, g=0.0, delta_m=None, note="native Feautrier with 32-point emission-angle quadrature"),
)
FUTURE_STUDY_PATHS = PILOT_PATHS


def supported_paths() -> tuple[PilotPath, ...]:
    """Return only paths that receive cold/warm timing pilots."""

    return tuple(item for item in PILOT_PATHS if item.supported)


def validate_frozen_contract() -> None:
    """Reject accidental Track-A, duplicate, imprecise, or incomplete contracts."""

    if TRACK != "track_b_native_scattering" or "track_a" in TRACK:
        raise ValueError("Stage 8 pilots must be Track B only")
    keys = {(item.subsection, item.framework, item.path) for item in PILOT_PATHS}
    if len(keys) != len(PILOT_PATHS):
        raise ValueError("pilot path identifiers must be unique")
    if {item.subsection for item in PILOT_PATHS} != {"8A", "8B", "8C", "8D", "8E"}:
        raise ValueError("every Stage-8 subsection must be represented")
    if any(item.full_case_count <= 0 for item in PILOT_PATHS if item.supported):
        raise ValueError("every supported path needs a non-zero full matrix projection")
    if any(item.full_case_count != 0 for item in PILOT_PATHS if not item.supported):
        raise ValueError("unsupported paths must schedule zero cases")
    if OMEGA0_LADDER != (0.5, 0.9, 0.99) or TAU_LADDER != (0.1, 1.0, 10.0, 100.0):
        raise ValueError("the isotropic ladder changed")
    if ASYMMETRY_LADDER != (0.3, 0.6, 0.9) or DELTA_M_OPTIONS != (False, True):
        raise ValueError("the anisotropy/delta-M ladder changed")
    if FLOAT_PRECISION != "float64":
        raise ValueError("Stage-8 science pilots require float64")


def project_path_resources(
    *,
    cold_wall_s: float,
    warm_wall_s: float,
    warm_setup_s: float,
    warm_case_s: float,
    peak_rss_bytes: int,
    available_memory_bytes: int,
    retained_tensor_bytes: int,
    full_case_count: int,
) -> dict[str, int | float | bool]:
    """Project one path using the frozen one-profile/grid shard contract.

    Cold-start penalty is paid once, setup once per nine profile/grid shards,
    and warm per-case cost once per full case.  The 25% factor reserves plot,
    serialization, integrity-index, and report assembly time.  Memory assumes a
    shard retains every case product, in addition to measured pilot RSS.
    """

    values = (cold_wall_s, warm_wall_s, warm_setup_s, warm_case_s)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("timing inputs must be finite and non-negative")
    if min(peak_rss_bytes, available_memory_bytes, retained_tensor_bytes, full_case_count) < 0:
        raise ValueError("resource and count inputs must be non-negative")
    cold_penalty = max(cold_wall_s - warm_wall_s, 0.0)
    raw_wall = cold_penalty + SHARDS_PER_PATH * warm_setup_s + full_case_count * warm_case_s
    projected_wall = raw_wall * ASSEMBLY_OVERHEAD_FACTOR
    max_cases_per_shard = math.ceil(full_case_count / SHARDS_PER_PATH)
    projected_peak = peak_rss_bytes + max(max_cases_per_shard - 1, 0) * retained_tensor_bytes
    memory_fraction = (
        projected_peak / available_memory_bytes if available_memory_bytes > 0 else math.inf
    )
    return {
        "cold_penalty_s": cold_penalty,
        "raw_wall_time_s": raw_wall,
        "projected_wall_time_s": projected_wall,
        "projected_peak_rss_bytes": projected_peak,
        "available_memory_bytes_at_decision": available_memory_bytes,
        "projected_peak_fraction_of_available": memory_fraction,
        "laptop_wall_safe": projected_wall <= SUBSECTION_WALL_LIMIT_S,
        "laptop_memory_safe": memory_fraction <= RSS_AVAILABLE_FRACTION_LIMIT,
        "recommended_shards": SHARDS_PER_PATH,
        "max_cases_per_shard": max_cases_per_shard,
    }


def project_controlled_study_resources(
    *,
    cold_wall_s: float,
    warm_wall_s: float,
    warm_case_s: float,
    peak_rss_bytes: int,
    available_memory_bytes: int,
    native_wavelength_count: int,
) -> dict[str, int | float | bool]:
    """Project the current 12-case controlled study for one framework.

    Each of four profile/grid shards performs its three cloud states
    sequentially after one native setup. Only spectra remain resident between
    states; complete per-case diagnostics are serialized before the next state.
    """

    values = (cold_wall_s, warm_wall_s, warm_case_s)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("timing inputs must be finite and non-negative")
    if min(
        peak_rss_bytes,
        available_memory_bytes,
        native_wavelength_count,
    ) < 0:
        raise ValueError("resource and native-size inputs must be non-negative")
    cold_penalty = max(cold_wall_s - warm_wall_s, 0.0)
    additional_cases = CONTROLLED_CASES_PER_PATH - CONTROLLED_SHARDS_PER_PATH
    raw_wall = (
        cold_penalty
        + CONTROLLED_SHARDS_PER_PATH * warm_wall_s
        + additional_cases * warm_case_s
    )
    projected_wall = raw_wall * ASSEMBLY_OVERHEAD_FACTOR
    retained_spectra_bytes = (
        (CONTROLLED_STATES_PER_SHARD - 1) * native_wavelength_count * 8
    )
    projected_peak = peak_rss_bytes + retained_spectra_bytes
    fraction = (
        projected_peak / available_memory_bytes
        if available_memory_bytes > 0
        else math.inf
    )
    return {
        "cold_penalty_s": cold_penalty,
        "raw_wall_time_s": raw_wall,
        "projected_wall_time_s": projected_wall,
        "projected_peak_rss_bytes": projected_peak,
        "available_memory_bytes_at_decision": available_memory_bytes,
        "projected_peak_fraction_of_available": fraction,
        "strict_laptop_memory_gate": fraction <= RSS_AVAILABLE_FRACTION_LIMIT,
        "operationally_feasible_if_serial": projected_peak < available_memory_bytes,
        "recommended_shards": CONTROLLED_SHARDS_PER_PATH,
        "cases_per_shard": CONTROLLED_STATES_PER_SHARD,
    }


def contract_payload() -> dict[str, object]:
    """Return the archived broad pilot contract for a future study."""

    validate_frozen_contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": 8,
        "status": FUTURE_STUDY_STATUS,
        "track": TRACK,
        "precision": FLOAT_PRECISION,
        "timing_clock": TIMING_CLOCK,
        "primary_cells": PRIMARY_CELLS,
        "cell_counts": list(CELL_COUNTS),
        "profiles": list(PROFILES),
        "pilot_profile": PILOT_PROFILE,
        "pilot_placement": PILOT_PLACEMENT,
        "omega0_ladder": list(OMEGA0_LADDER),
        "tau_ladder": list(TAU_LADDER),
        "accepted_moderate_tau": list(ACCEPTED_MODERATE_TAU),
        "unresolved_stress_tau": list(UNRESOLVED_STRESS_TAU),
        "asymmetry_ladder": list(ASYMMETRY_LADDER),
        "delta_m_options": list(DELTA_M_OPTIONS),
        "reference_angle_counts": list(REFERENCE_ANGLE_COUNTS),
        "assembly_overhead_factor": ASSEMBLY_OVERHEAD_FACTOR,
        "shards_per_path": SHARDS_PER_PATH,
        "thresholds": {
            "subsection_wall_limit_s": SUBSECTION_WALL_LIMIT_S,
            "combined_wall_limit_s": COMBINED_WALL_LIMIT_S,
            "rss_available_fraction_limit": RSS_AVAILABLE_FRACTION_LIMIT,
        },
        "projection_formula": "1.25 * (max(cold-warm,0) + 9*warm_setup + full_cases*warm_case)",
        "memory_formula": "pilot_peak_rss + (ceil(full_cases/9)-1)*retained_tensor_bytes",
        "paths": [asdict(item) for item in PILOT_PATHS],
    }


def planned_cloud_scope_payload() -> dict[str, object]:
    """Return the MgSiO3-only scope now deferred to a future study."""

    return asdict(PLANNED_CLOUD_SCOPE)


def controlled_study_payload() -> dict[str, object]:
    """Return the authoritative current Stage-8 controlled-study contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": 8,
        "status": CURRENT_STAGE_8_STATUS,
        "track": TRACK,
        "science_question": (
            "For the same moderate grey cloud, how much does enabling exact "
            "isotropic scattering change each native thermal-emission spectrum?"
        ),
        "cloud_description": "idealized grey aerosol; no material-specific claim",
        "cloud_top_pressure_bar": 1.0e-2,
        "cloud_bottom_pressure_bar": 100.0,
        "cloud_reference_wavelength_micron": 5.0,
        "cloud_extinction_slope": 0.0,
        "profile_grids": [
            {"profile": profile, "cells": cells}
            for profile, cells in CONTROLLED_PROFILE_GRIDS
        ],
        "states": [asdict(item) for item in CONTROLLED_CLOUD_STATES],
        "solver_paths": [asdict(item) for item in CONTROLLED_SOLVER_PATHS],
        "cases_per_framework": CONTROLLED_CASES_PER_PATH,
        "total_native_cases": CONTROLLED_CASES_PER_PATH
        * len(CONTROLLED_SOLVER_PATHS),
        "common_tensor_allowed": False,
        "primary_metric": "scattering_cloud_minus_absorbing_cloud",
        "projection_formula": (
            "1.25 * (max(cold-warm,0) + 4*warm + 8*warm_case)"
        ),
    }
