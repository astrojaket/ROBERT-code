"""Frozen run-matrix helpers for emission intercomparison V2 Stage 9.

This module is deliberately free of forward-model imports.  It defines and
validates the Stage-9 science matrix, uncertainty tiers, priors,
and scheduler shards without executing any science workload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from robert_exoplanets.core import RobertValidationError


SCHEMA_VERSION = "2.0.0"
STAGE = 9
TRACK = "track_b_native_retrieval"
FRAMEWORKS = ("robert", "picaso", "petitradtrans")
NOISE_TIERS_PPM = (30, 60, 100)
NOISE_REALIZATIONS = ("mean",)
MPI_RANKS_PER_RETRIEVAL = 12
THREADS_PER_RANK = 1

# Barstow et al. (2020) did not randomize the synthetic spectral points.  Keep
# this exported mapping empty so downstream code cannot silently add a draw.
GAUSSIAN_NOISE_SEEDS: dict[str, int] = {}

MULTINEST_SETTINGS = {
    "n_live_points": 400,
    "evidence_tolerance": 0.5,
    "sampling_efficiency": 0.8,
    "max_iter": 0,
    "resume": True,
    "importance_nested_sampling": True,
    "multimodal": True,
    "n_iter_before_update": 100,
    "verbose": False,
    "mpi_nprocs": MPI_RANKS_PER_RETRIEVAL,
    "invalid_loglike_floor": -1.0e100,
}


@dataclass(frozen=True)
class ParameterDefinition:
    """One frozen Stage-9 uniform-prior retrieval parameter."""

    name: str
    lower: float
    upper: float
    truth: float
    label: str
    unit: str | None = None

    def __post_init__(self) -> None:
        values = (float(self.lower), float(self.upper), float(self.truth))
        if not all(math.isfinite(value) for value in values):
            raise RobertValidationError(
                f"non-finite parameter definition for {self.name}"
            )
        if not self.lower < self.upper or not self.lower <= self.truth <= self.upper:
            raise RobertValidationError(
                f"invalid prior/truth relationship for {self.name}"
            )


@dataclass(frozen=True)
class ScenarioDefinition:
    """One Stage-9 injection/retrieval atmospheric scenario."""

    name: str
    profile: str
    cloud: str
    gamma2_truth: float
    cloud_tau_truth: float | None = None
    cloud_top_pressure_bar_truth: float | None = None
    cloud_single_scattering_albedo_truth: float | None = None

    @property
    def cloudy(self) -> bool:
        return self.cloud != "clear"


SCENARIOS = (
    ScenarioDefinition("clear_non_inverted", "pg14_non_inverted", "clear", 0.5),
    ScenarioDefinition("clear_inverted", "pg14_inverted", "clear", 10.0),
    ScenarioDefinition(
        "grey_absorbing_non_inverted",
        "pg14_non_inverted",
        "grey_absorbing",
        0.5,
        cloud_tau_truth=1.0,
        cloud_top_pressure_bar_truth=1.0e-2,
        cloud_single_scattering_albedo_truth=0.0,
    ),
    ScenarioDefinition(
        "grey_scattering_non_inverted",
        "pg14_non_inverted",
        "grey_isotropic_scattering",
        0.5,
        cloud_tau_truth=1.0,
        cloud_top_pressure_bar_truth=1.0e-2,
        cloud_single_scattering_albedo_truth=0.9,
    ),
)

_COMMON_GAS_TRUTH = {
    "H2O": 3.222572565623962e-4,
    "CO": 4.59870844773289e-4,
    "CO2": 6.734289181697181e-8,
    "CH4": 3.861237147673902e-8,
}


def parameter_definitions(
    scenario: ScenarioDefinition | str,
) -> tuple[ParameterDefinition, ...]:
    """Return ordered priors for a scenario; area scale is intentionally absent."""

    item = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
    parameters = [
        ParameterDefinition("log10_kappa_ir", -4.5, -1.5, -3.0, "log10 kappa_IR"),
        ParameterDefinition("log10_gamma1", -2.0, -0.7, -1.0, "log10 gamma_1"),
        ParameterDefinition(
            "log10_gamma2", -0.6, 1.3, math.log10(item.gamma2_truth), "log10 gamma_2"
        ),
        ParameterDefinition("T_irr_k", 1000.0, 2000.0, 1500.0, "T_irr", "K"),
        ParameterDefinition("alpha", 0.0, 1.0, 0.5, "alpha"),
    ]
    parameters.extend(
        ParameterDefinition(
            f"log10_vmr_{name}", -10.0, -2.0, math.log10(value), f"log10 VMR({name})"
        )
        for name, value in _COMMON_GAS_TRUTH.items()
    )
    if item.cloudy:
        parameters.extend(
            (
                ParameterDefinition(
                    "log10_cloud_tau_5um", -1.0, 1.0, 0.0, "log10 tau_cloud(5 um)"
                ),
                ParameterDefinition(
                    "log10_cloud_top_pressure_bar",
                    -3.0,
                    -1.0,
                    -2.0,
                    "log10 P_cloud,top",
                    "bar",
                ),
            )
        )
    if item.cloud == "grey_isotropic_scattering":
        parameters.append(
            ParameterDefinition(
                "cloud_single_scattering_albedo", 0.5, 0.99, 0.9, "omega_0"
            )
        )
    return tuple(parameters)


@dataclass(frozen=True)
class RunDefinition:
    """One immutable retrieval in the Stage-9 production matrix."""

    run_id: str
    scenario: str
    injector: str
    retriever: str
    noise_ppm: int
    noise_id: str
    control: str
    shard_id: str
    sampler_seed: int
    mpi_ranks: int = MPI_RANKS_PER_RETRIEVAL
    threads_per_rank: int = THREADS_PER_RANK

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parameters"] = [
            asdict(item) for item in parameter_definitions(self.scenario)
        ]
        payload["sampler"] = {"engine": "multinest", **MULTINEST_SETTINGS}
        return payload


def scenario_by_name(name: str) -> ScenarioDefinition:
    """Resolve a frozen scenario name."""

    for item in SCENARIOS:
        if item.name == name:
            return item
    raise RobertValidationError(f"unknown Stage-9 scenario: {name}")


def directed_cross_framework_pairs() -> tuple[tuple[str, str], ...]:
    """Return all six directed non-self injector/retriever pairs."""

    return tuple(
        (left, right) for left in FRAMEWORKS for right in FRAMEWORKS if left != right
    )


def _sampler_seed(parts: Iterable[object]) -> int:
    digest = hashlib.sha256(
        "|".join(str(item) for item in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _run_id(
    scenario: str, injector: str, retriever: str, noise_ppm: int, noise_id: str
) -> str:
    return (
        f"{scenario}__inj-{injector}__ret-{retriever}__{noise_ppm:03d}ppm__{noise_id}"
    )


def build_run_matrix() -> tuple[RunDefinition, ...]:
    """Build the frozen 72-run matrix in deterministic scheduler order."""

    runs: list[RunDefinition] = []
    for scenario in SCENARIOS:
        for retriever in FRAMEWORKS:
            shard = f"{retriever}__{scenario.name}"
            for injector, directed_retriever in directed_cross_framework_pairs():
                if directed_retriever != retriever:
                    continue
                for noise_ppm in NOISE_TIERS_PPM:
                    for noise_id in NOISE_REALIZATIONS:
                        run_id = _run_id(
                            scenario.name, injector, retriever, noise_ppm, noise_id
                        )
                        runs.append(
                            RunDefinition(
                                run_id,
                                scenario.name,
                                injector,
                                retriever,
                                noise_ppm,
                                noise_id,
                                "directed_cross_framework",
                                shard,
                                _sampler_seed((run_id, "multinest")),
                            )
                        )
    validate_run_matrix(runs)
    return tuple(runs)


def validate_run_matrix(runs: Iterable[RunDefinition]) -> None:
    """Enforce the frozen counts and boundaries."""

    values = tuple(runs)
    if len(values) != 72 or len({item.run_id for item in values}) != 72:
        raise RobertValidationError("Stage 9 requires exactly 72 unique retrievals")
    shard_counts: dict[str, int] = {}
    for run in values:
        shard_counts[run.shard_id] = shard_counts.get(run.shard_id, 0) + 1
        if run.mpi_ranks != 12 or run.threads_per_rank != 1:
            raise RobertValidationError(
                "every Stage-9 retrieval requires 12 ranks and one thread per rank"
            )
        names = {item.name for item in parameter_definitions(run.scenario)}
        if "area_scale" in names or "log10_area_scale" in names:
            raise RobertValidationError("area scale is excluded from Stage 9")
        if run.control != "directed_cross_framework":
            raise RobertValidationError(
                "Stage 9 contains directed cross-retrievals only"
            )
        if run.noise_id != "mean":
            raise RobertValidationError("Stage 9 uses unperturbed spectral means only")
    if len(shard_counts) != 12 or set(shard_counts.values()) != {6}:
        raise RobertValidationError("Stage 9 requires twelve 6-retrieval shards")


def noise_vector_key(scenario: str, noise_ppm: int, noise_id: str) -> None:
    """Validate the Barstow-style mean realization and return no noise key."""

    scenario_by_name(scenario)
    if noise_ppm not in NOISE_TIERS_PPM:
        raise RobertValidationError(f"unsupported Stage-9 noise tier: {noise_ppm}")
    if noise_id == "mean":
        return None
    raise RobertValidationError(f"unsupported Stage-9 noise realization: {noise_id}")


def frozen_contract_payload(*, common_contract_sha256: str) -> dict[str, object]:
    """Return the JSON-serializable frozen setup payload."""

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "track": TRACK,
        "status": "approved_setup_not_yet_executed",
        "common_contract_sha256": common_contract_sha256,
        "science_question": (
            "How do native ROBERT, PICASO, and petitRADTRANS emission forward models bias "
            "and broaden atmospheric and grey-cloud retrievals under directed model mismatch?"
        ),
        "frameworks": list(FRAMEWORKS),
        "scenarios": [asdict(item) for item in SCENARIOS],
        "parameters_by_scenario": {
            item.name: [asdict(value) for value in parameter_definitions(item)]
            for item in SCENARIOS
        },
        "noise": {
            "tiers_ppm": list(NOISE_TIERS_PPM),
            "realizations": list(NOISE_REALIZATIONS),
            "gaussian_seeds": GAUSSIAN_NOISE_SEEDS,
            "spectral_points_randomized": False,
            "data_realization": "unperturbed_native_mean",
            "interpretation": "nonzero error envelopes enter the Gaussian likelihood",
            "protocol_reference": (
                "Barstow et al. 2020, MNRAS 493, 4884; doi:10.1093/mnras/staa548"
            ),
        },
        "matrix": {
            "directed_cross_framework_pairs": [
                list(item) for item in directed_cross_framework_pairs()
            ],
            "self_controls": [],
            "retrieval_count": 72,
            "shard_count": 12,
            "retrievals_per_shard": 6,
        },
        "sampler": {"engine": "multinest", **MULTINEST_SETTINGS},
        "execution": {
            "cluster": "glamdring",
            "scheduler_queue": "redwood",
            "mpi_ranks_per_retrieval": MPI_RANKS_PER_RETRIEVAL,
            "threads_per_rank": THREADS_PER_RANK,
            "nested_mpirun_forbidden_under_addqueue": True,
        },
        "cloud_contract": {
            "reference_wavelength_micron": 5.0,
            "cloud_bottom_pressure_bar": 100.0,
            "extinction_slope": 0.0,
            "asymmetry_factor": 0.0,
            "delta_m": False,
            "cloud_parameters_retrieved_natively": True,
        },
        "fixed": {
            "internal_temperature_k": 100.0,
            "normalization_parameters_retrieved": [],
            "radii_gravity_and_eclipse_normalization": "version_2_common_contract",
            "likelihood": "independent Gaussian with fixed ppm uncertainty",
        },
        "scope_exclusions": [
            "Track A shared-tensor retrievals",
            "MgSiO3 or other condensate microphysics",
            "Mie optical properties",
            "anisotropic scattering",
            "wavelength-dependent cloud extinction",
            "high-order scattering methods",
            "fabricated shared opacity or cloud tensors",
        ],
        "retention": {
            "native_injection_means": 12,
            "standard_normal_noise_vectors": 0,
            "per_likelihood_spectra": False,
            "all_posterior_predictive_spectra": False,
            "canonical_posterior_compressed": True,
            "raw_multinest_chains_archived_after_validation": True,
            "failed_restart_files_preserved": True,
        },
    }


def write_frozen_contract(path: str | Path, *, common_contract_sha256: str) -> Path:
    """Write the deterministic frozen contract without running science code."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            frozen_contract_payload(common_contract_sha256=common_contract_sha256),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def matrix_summary(runs: Iterable[RunDefinition] | None = None) -> Mapping[str, object]:
    """Return compact matrix counts for manifests and diagnostics."""

    values = tuple(build_run_matrix() if runs is None else runs)
    return {
        "retrievals": len(values),
        "directed_cross_framework": sum(
            item.control == "directed_cross_framework" for item in values
        ),
        "self_retrieval_controls": sum(
            item.control == "self_retrieval_control" for item in values
        ),
        "shards": len({item.shard_id for item in values}),
        "retrievals_per_shard": 6,
    }


__all__ = [
    "FRAMEWORKS",
    "GAUSSIAN_NOISE_SEEDS",
    "MPI_RANKS_PER_RETRIEVAL",
    "MULTINEST_SETTINGS",
    "NOISE_REALIZATIONS",
    "NOISE_TIERS_PPM",
    "ParameterDefinition",
    "RunDefinition",
    "SCENARIOS",
    "ScenarioDefinition",
    "build_run_matrix",
    "directed_cross_framework_pairs",
    "frozen_contract_payload",
    "matrix_summary",
    "noise_vector_key",
    "parameter_definitions",
    "scenario_by_name",
    "validate_run_matrix",
    "write_frozen_contract",
]
