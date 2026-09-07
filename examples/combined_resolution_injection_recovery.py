"""Run a deterministic combined low/high-resolution emission recovery.

The low-resolution branch uses the bundled correlated-k tables. The
high-resolution branch uses full R=1e6 CO and H2O line-by-line tables, then
applies Doppler shift, rotational broadening, a Gaussian LSF, and pixel
integration. Both branches share one atmosphere evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from typing import Mapping

# Apply the laptop policy before numerical libraries load.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    _requested_threads = os.environ.get(_thread_variable)
    if _requested_threads is None:
        _requested_threads = "3"
    else:
        try:
            _requested_threads = str(min(3, max(1, int(_requested_threads))))
        except ValueError:
            _requested_threads = "3"
    os.environ[_thread_variable] = _requested_threads
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib")
)

import numpy as np  # noqa: E402

from robert_exoplanets import (  # noqa: E402
    ExoKOpacitySource,
    ExoKTableBinning,
    FreeChemistry,
    GaussianLikelihood,
    MultiDatasetRetrievalProblem,
    Observation,
    ParameterizedEmissionFactoryConfig,
    ParameterizedEmissionModelConfig,
    Planet,
    PressureGrid,
    RetrievalParameter,
    RetrievalParameterSet,
    SpectralGrid,
    Star,
    UniformPrior,
    build_multi_dataset_emission_model,
    bundled_k_table_paths,
    run_retrieval,
)
from robert_exoplanets.forward.factory import LineByLineOpacitySource  # noqa: E402
from robert_exoplanets.forward.high_resolution import (  # noqa: E402
    ParameterizedMultiDatasetResponseForwardModel,
    VelocityParameterizedHighResolutionResponse,
)
from robert_exoplanets.likelihoods import (  # noqa: E402
    MixedMultiDatasetLikelihood,
    PolynomialContinuumLikelihood,
)
from robert_exoplanets.validation import (  # noqa: E402
    evaluate_multi_dataset_injection_recovery,
    inject_spectrum_collection,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = ROOT / "external_data" / "petitRADTRANS" / "input_data"
H2O_TABLE = (
    INPUT_DATA
    / "opacities/lines/line_by_line/H2O/1H2-16O/"
    "1H2-16O__POKAZATEL.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)
CO_TABLE = (
    INPUT_DATA
    / "opacities/lines/line_by_line/CO/12C-16O/"
    "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)
EXPECTED_SHA256 = {
    "H2O": "9a79513fb92dfa369f85abb273257ae2b2100a4a386169b9bfff8a5b74249b73",
    "CO": "5f77209da3ea67d5a697b06379b1de2b8fe106adf7ea0abfb451c675084087ad",
}
OUTPUT_DIR = ROOT / "examples" / "outputs" / "combined_resolution_recovery"
REPORT_PATH = ROOT / "docs" / "data" / "combined_resolution_recovery_20260830.json"
# These are strict upper bounds.  The process and one-opacity-slice estimates
# must remain below, not equal to, the laptop policy limits.
MAX_MEMORY_BYTES = 2 * 1024**3
OPACITY_MEMORY_BYTES = 900 * 1024**2
DEFAULT_MAX_MEMORY_BYTES = 1900 * 1024**2
DEFAULT_PYMULTINEST_LIVE_POINTS = 64
DEFAULT_PYMULTINEST_MAX_ITER = 0
DEFAULT_PYMULTINEST_EVIDENCE_TOLERANCE = 0.5
DEFAULT_PYMULTINEST_SAMPLING_EFFICIENCY = 0.8
DEFAULT_PYMULTINEST_TRUTH_TOLERANCE_KM_S = 1.0
DEFAULT_PYMULTINEST_MPI_NPROCS = 1
LBL_WINDOW_MICRON = (2.2984, 2.3042)
OBSERVED_WINDOW_MICRON = (2.29885, 2.30355)
TRUTH = {
    "temperature_K": 1500.0,
    "log10_h2o_vmr": -3.3,
    "log10_co_vmr": -3.0,
    "radial_velocity_km_s": 5.0,
}


@dataclass(frozen=True)
class LogLinearTemperatureProfile:
    """One-parameter non-isothermal profile for this controlled injection."""

    reference_pressure_bar: float = 0.1
    gradient_K_per_decade: float = 120.0
    parameter_name: str = "temperature_K"
    name: str = "log-linear-validation-profile"

    def required_parameters(self) -> tuple[str, ...]:
        return (self.parameter_name,)

    def evaluate(
        self,
        parameters: Mapping[str, float],
        pressure_grid: PressureGrid,
    ) -> np.ndarray:
        if pressure_grid.unit != "bar":
            raise ValueError("log-linear validation profile requires pressure in bar")
        reference_temperature = float(parameters[self.parameter_name])
        temperature = reference_temperature + self.gradient_K_per_decade * np.log10(
            pressure_grid.centers / self.reference_pressure_bar
        )
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("log-linear validation profile produced invalid temperatures")
        temperature.setflags(write=False)
        return temperature


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Gaussian injection noise in units of the stated uncertainties (default: 1).",
    )
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=DEFAULT_MAX_MEMORY_BYTES / 1024**3,
        help="Strict process RSS guard; values must be below 2 GiB.",
    )
    parser.add_argument(
        "--run-pymultinest",
        action="store_true",
        help=(
            "Run two resource-bounded one-parameter PyMultiNest proof runs "
            "after optimal estimation."
        ),
    )
    parser.add_argument(
        "--pymultinest-live-points",
        type=int,
        default=DEFAULT_PYMULTINEST_LIVE_POINTS,
    )
    parser.add_argument(
        "--pymultinest-max-iter",
        type=int,
        default=DEFAULT_PYMULTINEST_MAX_ITER,
    )
    parser.add_argument(
        "--pymultinest-evidence-tolerance",
        type=float,
        default=DEFAULT_PYMULTINEST_EVIDENCE_TOLERANCE,
    )
    parser.add_argument(
        "--pymultinest-sampling-efficiency",
        type=float,
        default=DEFAULT_PYMULTINEST_SAMPLING_EFFICIENCY,
        help="PyMultiNest sampling efficiency for the proof runs.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _log_binned_grid(lower: float, upper: float, n_bins: int, name: str) -> SpectralGrid:
    edges = np.geomspace(lower, upper, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return SpectralGrid(
        values=centers,
        bin_edges=edges,
        unit="micron",
        role="observed",
        name=name,
    )


def _constant_resolving_power_grid(
    lower: float,
    upper: float,
    resolving_power: float,
) -> SpectralGrid:
    count = int(np.floor(resolving_power * np.log(upper / lower))) + 1
    centers = lower * np.exp(np.arange(count, dtype=float) / resolving_power)
    centers = centers[centers <= upper]
    log_centers = np.log(centers)
    log_edges = np.empty(centers.size + 1)
    log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
    log_edges[0] = log_centers[0] - 0.5 * (
        log_centers[1] - log_centers[0]
    )
    log_edges[-1] = log_centers[-1] + 0.5 * (
        log_centers[-1] - log_centers[-2]
    )
    return SpectralGrid(
        values=centers,
        bin_edges=np.exp(log_edges),
        unit="micron",
        role="observed",
        name="synthetic-R100000-pixels",
    )


def _dummy_observation(grid: SpectralGrid) -> Observation:
    return Observation.from_arrays(
        grid.values,
        np.zeros(grid.size),
        np.ones(grid.size),
        wavelength_bin_edges=grid.bin_edges,
        wavelength_unit=grid.unit,
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
        instrument="synthetic K-band R=100000",
    )


def _build_forward_model(
    *,
    native_sampling_stride: int = 1,
) -> tuple[
    ParameterizedMultiDatasetResponseForwardModel,
    dict[str, object],
]:
    pressure_grid = PressureGrid.from_log_centers(
        1.0e-5,
        100.0,
        48,
        unit="bar",
        name="shared combined-resolution pressure grid",
    )
    temperature_profile = LogLinearTemperatureProfile()
    chemistry = FreeChemistry(
        active_species=("H2O", "CO"),
        parameter_names={
            "H2O": "log10_h2o_vmr",
            "CO": "log10_co_vmr",
        },
        parameter_mode="log10",
    )
    planet = Planet(
        name="synthetic hot Jupiter",
        radius_m=7.1492e7,
        gravity_m_s2=15.0,
    )
    star = Star(
        name="synthetic Sun-like star",
        radius_m=6.957e8,
        effective_temperature_k=5500.0,
    )
    model_config = ParameterizedEmissionModelConfig(
        opacity_species=("H2O", "CO"),
        include_rayleigh=False,
        gas_combination="random_overlap",
        thermal_integration_backend="numpy",
        stellar_spectrum_model="blackbody",
        compute_diagnostics=False,
        metadata={"validation_case": "combined_resolution_injection"},
    )
    low_grid = _log_binned_grid(1.0, 5.0, 48, "synthetic-low-resolution")
    low_source = ExoKOpacitySource(
        species=("H2O", "CO"),
        paths=bundled_k_table_paths(("H2O", "CO")),
        name="bundled-R100-H2O-CO",
    )
    high_source = LineByLineOpacitySource(
        species=("H2O", "CO"),
        paths={"H2O": H2O_TABLE, "CO": CO_TABLE},
        checksum=False,
        max_memory_bytes=OPACITY_MEMORY_BYTES,
        wavelength_bounds_micron=LBL_WINDOW_MICRON,
        native_sampling_stride=native_sampling_stride,
        max_cached_slices=1,
    )
    high_grid = high_source.native_spectral_grid()
    low_factory = ParameterizedEmissionFactoryConfig(
        planet=planet,
        star=star,
        temperature_profile=temperature_profile,
        chemistry_model=chemistry,
        opacity_source=low_source,
        model=model_config,
        pressure_grid=pressure_grid,
        mean_molecular_weight=2.3217263131165278,
        opacity_binning=ExoKTableBinning(num=64),
    )
    high_factory = ParameterizedEmissionFactoryConfig(
        planet=planet,
        star=star,
        temperature_profile=temperature_profile,
        chemistry_model=chemistry,
        opacity_source=high_source,
        model=model_config,
        pressure_grid=pressure_grid,
        mean_molecular_weight=2.3217263131165278,
        opacity_binning=None,
    )
    native_model = build_multi_dataset_emission_model(
        {"low_resolution": low_factory, "high_resolution": high_factory},
        spectral_grids={"low_resolution": low_grid, "high_resolution": high_grid},
    )
    high_observation_grid = _constant_resolving_power_grid(
        *OBSERVED_WINDOW_MICRON,
        resolving_power=100_000.0,
    )
    high_response = VelocityParameterizedHighResolutionResponse(
        observation=_dummy_observation(high_observation_grid),
        velocity_parameter="radial_velocity_km_s",
        resolving_power=100_000.0,
        projected_rotation_km_s=4.0,
        limb_darkening=0.3,
        kernel_support_sigma=4.0,
        max_cached_responses=2,
    )
    model = ParameterizedMultiDatasetResponseForwardModel(
        native_model=native_model,
        runtime_responses={"high_resolution": high_response},
    )
    preparation = {
        "pressure_layers": pressure_grid.n_layers,
        "low_resolution_points": low_grid.size,
        "high_resolution_native_points": high_grid.size,
        "high_resolution_observed_points": high_observation_grid.size,
        "native_sampling_stride": native_sampling_stride,
        "high_response": high_response,
    }
    return model, preparation


def _parameters() -> RetrievalParameterSet:
    return RetrievalParameterSet(
        (
            RetrievalParameter(
                "temperature_K", UniformPrior(1200.0, 1800.0), unit="K"
            ),
            RetrievalParameter("log10_h2o_vmr", UniformPrior(-4.5, -2.5)),
            RetrievalParameter("log10_co_vmr", UniformPrior(-4.5, -2.5)),
            RetrievalParameter(
                "radial_velocity_km_s",
                UniformPrior(-12.0, 12.0),
                unit="km/s",
            ),
        )
    )


def _pymultinest_proof_problem(
    problem: MultiDatasetRetrievalProblem,
) -> tuple[MultiDatasetRetrievalProblem, dict[str, object]]:
    """Build the resource-bounded PyMultiNest proof problem.

    The proof keeps the complete combined low/high-resolution likelihood and
    forward model.  It samples only radial velocity, while temperature and
    both abundances stay at their injected values.  This is a deliberately
    narrow test of the high-resolution response, shared data-set likelihood,
    and sampler/evidence plumbing.  It is not evidence for the full
    four-parameter atmospheric posterior.
    """

    proof_parameters = RetrievalParameterSet(
        (
            RetrievalParameter(
                "radial_velocity_km_s",
                UniformPrior(-12.0, 12.0),
                unit="km/s",
            ),
        )
    )

    response_model = problem.forward_model
    native_model = getattr(response_model, "native_model", None)
    fixed_responses = getattr(response_model, "fixed_responses", {})
    runtime_responses = getattr(response_model, "runtime_responses", {})
    if not callable(native_model):
        raise RuntimeError(
            "the PyMultiNest proof requires a parameterized response wrapper with native_model"
        )
    native_truth = native_model(TRUTH)
    if not isinstance(native_truth, Mapping):
        raise RuntimeError("the cached native proof model must return named spectra")

    def proof_forward(parameters: Mapping[str, float]) -> Mapping[str, object]:
        trial = dict(TRUTH)
        trial["radial_velocity_km_s"] = float(parameters["radial_velocity_km_s"])
        output: dict[str, object] = {}
        for name, spectrum in native_truth.items():
            transformed = spectrum
            if name in fixed_responses:
                transformed = fixed_responses[name].observe(transformed)
            if name in runtime_responses:
                transformed = runtime_responses[name].observe(transformed, trial)
            output[name] = transformed
        return output

    proof_problem = MultiDatasetRetrievalProblem(
        name=f"{problem.name}-pymultinest-rv-proof",
        observations=problem.observations,
        parameters=proof_parameters,
        forward_model=proof_forward,
        likelihood=problem.likelihood,
        metadata={
            **dict(problem.metadata),
            "sampling_proof": "one-dimensional radial-velocity PyMultiNest slice",
            "sampling_proof_fixed_parameters": "temperature_K,log10_h2o_vmr,log10_co_vmr",
        },
        opacity_identifiers=problem.opacity_identifiers,
    )
    return proof_problem, {
        "native_spectra_cached": True,
        "cache_state": "one_shared_native_atmosphere_at_injected_parameters",
        "atmosphere_evaluations_per_sampler_call": 0,
        "runtime_response_only": True,
        "cached_dataset_names": sorted(str(name) for name in native_truth),
    }


def _run_pymultinest_proof(
    problem: MultiDatasetRetrievalProblem,
    *,
    output_dir: Path,
    seed: int,
    n_live_points: int,
    max_iter: int,
    evidence_tolerance: float,
    sampling_efficiency: float,
) -> dict[str, object]:
    """Run two independent resource-bounded PyMultiNest checks and apply a gate."""

    proof_problem, cache_record = _pymultinest_proof_problem(problem)
    run_records: list[dict[str, object]] = []
    started = perf_counter()
    benchmark_started = perf_counter()
    benchmark_loglike = proof_problem.log_likelihood_from_vector(
        [TRUTH["radial_velocity_km_s"]]
    )
    benchmark_seconds = perf_counter() - benchmark_started
    if not np.isfinite(benchmark_loglike):
        raise RuntimeError("cached PyMultiNest proof likelihood is not finite at injection truth")
    for run_index, run_seed in enumerate((int(seed), int(seed) + 1), start=1):
        run_directory = output_dir / f"seed_{run_seed}"
        run_started = perf_counter()
        try:
            result = run_retrieval(
                proof_problem,
                method="pymultinest",
                output_dir=run_directory,
                seed=run_seed,
                n_live_points=n_live_points,
                max_iter=max_iter,
                evidence_tolerance=evidence_tolerance,
                sampling_efficiency=sampling_efficiency,
                resume=False,
                verbose=False,
                mpi_nprocs=DEFAULT_PYMULTINEST_MPI_NPROCS,
            )
            evidence = result.log_evidence
            evidence_error = result.log_evidence_error
            best_velocity = result.best_fit_parameters.get(
                "radial_velocity_km_s"
            )
            run_records.append(
                {
                    "run_index": run_index,
                    "seed": run_seed,
                    "status": "completed",
                    "converged": bool(result.converged),
                    "message": result.message,
                    "likelihood_evaluations": int(
                        result.inference_result.metadata.get(
                            "likelihood_evaluations", "0"
                        )
                    ),
                    "likelihood_callback_evaluations": int(
                        result.inference_result.metadata.get(
                            "likelihood_callback_evaluations", "0"
                        )
                    ),
                    "best_fit_radial_velocity_km_s": best_velocity,
                    "absolute_truth_error_km_s": (
                        None
                        if best_velocity is None
                        else abs(float(best_velocity) - TRUTH["radial_velocity_km_s"])
                    ),
                    "log_evidence": evidence,
                    "log_evidence_error": evidence_error,
                    "elapsed_seconds": perf_counter() - run_started,
                }
            )
        except Exception as error:  # report a proof failure; do not hide it
            run_records.append(
                {
                    "run_index": run_index,
                    "seed": run_seed,
                    "status": "failed",
                    "converged": False,
                    "message": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": perf_counter() - run_started,
                }
            )

    completed = [record for record in run_records if record["status"] == "completed"]
    converged = bool(completed) and all(bool(record["converged"]) for record in completed)
    evidences = [
        float(record["log_evidence"])
        for record in completed
        if record.get("log_evidence") is not None
        and np.isfinite(float(record["log_evidence"]))
    ]
    evidence_difference = (
        None if len(evidences) != 2 else abs(evidences[0] - evidences[1])
    )
    evidence_agreement = (
        evidence_difference is not None
        and evidence_difference <= evidence_tolerance
    )
    truth_errors = [
        float(record["absolute_truth_error_km_s"])
        for record in completed
        if record.get("absolute_truth_error_km_s") is not None
        and np.isfinite(float(record["absolute_truth_error_km_s"]))
    ]
    truth_recovery = bool(truth_errors) and all(
        error <= DEFAULT_PYMULTINEST_TRUTH_TOLERANCE_KM_S for error in truth_errors
    )
    gate_passed = bool(
        len(completed) == 2
        and converged
        and evidence_agreement
        and truth_recovery
    )
    return {
        "status": "pass" if gate_passed else "fail",
        "gate_passed": gate_passed,
        "scope": "one-dimensional radial-velocity PyMultiNest slice",
        "fixed_parameters": [
            "temperature_K",
            "log10_h2o_vmr",
            "log10_co_vmr",
        ],
        "full_four_parameter_nested_sampling": "not_run",
        "forward_cache": {
            **cache_record,
            "single_truth_likelihood_seconds": benchmark_seconds,
            "single_truth_likelihood": benchmark_loglike,
        },
        "limitation": (
            "This resource-bounded PyMultiNest proof validates the combined likelihood "
            "and RV response. "
            "It does not demonstrate convergence or evidence accuracy for the full "
            "four-parameter atmospheric posterior."
        ),
        "gate": {
            "required_completed_runs": 2,
            "required_converged_each_run": True,
            "evidence_absolute_difference_max": evidence_tolerance,
            "radial_velocity_absolute_error_max_km_s": (
                DEFAULT_PYMULTINEST_TRUTH_TOLERANCE_KM_S
            ),
            "converged": converged,
            "evidence_agreement": evidence_agreement,
            "truth_recovery": truth_recovery,
            "evidence_absolute_difference": evidence_difference,
        },
        "settings": {
            "n_live_points": n_live_points,
            "max_iter": max_iter,
            "evidence_tolerance": evidence_tolerance,
            "sampling_efficiency": sampling_efficiency,
            "resume": False,
            "verbose": False,
            "mpi_nprocs": DEFAULT_PYMULTINEST_MPI_NPROCS,
            "seed_pair": [int(seed), int(seed) + 1],
        },
        "runs": run_records,
        "elapsed_seconds": perf_counter() - started,
    }


def _reduced_chi_square_gate(noise_scale: float) -> tuple[tuple[float, float], str]:
    """Return the fit gate appropriate for the requested injection noise."""

    if noise_scale == 0.0:
        return (0.0, 1.0e-8), "noise-free near-zero chi-square gate"
    expected = noise_scale**2
    return (
        (0.5 * expected, 1.5 * expected),
        "Gaussian noise gate: reduced chi-square expected near noise_scale^2",
    )


def main() -> dict[str, object]:
    args = _parse_args()
    workflow_started = perf_counter()
    max_memory_bytes = int(float(args.max_memory_gib) * 1024**3)
    if max_memory_bytes <= 0 or max_memory_bytes >= MAX_MEMORY_BYTES:
        raise ValueError("max-memory-gib must be positive and below 2 GiB")
    if OPACITY_MEMORY_BYTES >= 1 * 1024**3:
        raise RuntimeError("the opacity memory guard must remain below 1 GiB")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(args.noise_scale) or args.noise_scale < 0.0:
        raise ValueError("noise-scale must be finite and non-negative")
    if args.pymultinest_live_points < 50:
        raise ValueError("pymultinest-live-points must be at least 50")
    if args.pymultinest_max_iter != 0:
        raise ValueError("pymultinest-max-iter must be 0 for natural convergence")
    if (
        not np.isfinite(args.pymultinest_evidence_tolerance)
        or args.pymultinest_evidence_tolerance <= 0.0
    ):
        raise ValueError("pymultinest-evidence-tolerance must be finite and positive")
    if (
        not np.isfinite(args.pymultinest_sampling_efficiency)
        or not 0.0 < args.pymultinest_sampling_efficiency <= 1.0
    ):
        raise ValueError("pymultinest-sampling-efficiency must be in (0, 1]")

    table_records = {}
    for species, path in {"H2O": H2O_TABLE, "CO": CO_TABLE}.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing external LBL table: {path}")
        checksum = _file_sha256(path)
        if checksum != EXPECTED_SHA256[species]:
            raise ValueError(f"unexpected {species} LBL checksum")
        table_records[species] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": checksum,
        }

    start = perf_counter()
    forward_model, preparation = _build_forward_model()
    truth_spectra = forward_model(TRUTH)
    preparation_seconds = perf_counter() - start
    low_uncertainty = np.full(truth_spectra["low_resolution"].values.shape, 5.0e-6)
    high_uncertainty = np.full(
        truth_spectra["high_resolution"].values.shape,
        2.0e-6,
    )
    observations = inject_spectrum_collection(
        truth_spectra,
        {
            "low_resolution": low_uncertainty,
            "high_resolution": high_uncertainty,
        },
        seed=args.seed,
        noise_scale=args.noise_scale,
        instruments={
            "low_resolution": "synthetic broadband R~30",
            "high_resolution": "synthetic K-band R=100000",
        },
        metadata={"validation_case": "combined_resolution_injection"},
    )
    observations_by_name = {
        dataset.name: dataset.observation for dataset in observations.datasets
    }
    reduced_chi_square_bounds, fit_gate_description = _reduced_chi_square_gate(
        args.noise_scale
    )
    likelihood = MixedMultiDatasetLikelihood(
        {
            "low_resolution": GaussianLikelihood(
                offset_parameter=None,
                jitter_parameter=None,
            ),
            "high_resolution": PolynomialContinuumLikelihood(degree=1).prepare(
                observations_by_name["high_resolution"]
            ),
        }
    )
    problem = MultiDatasetRetrievalProblem(
        name="combined-low-high-resolution-emission-injection",
        observations=observations,
        parameters=_parameters(),
        forward_model=forward_model,
        likelihood=likelihood,
        metadata={
            "clear_atmosphere": "true",
            "low_resolution_opacity": "bundled_R100_correlated_k",
            "high_resolution_opacity": "petitRADTRANS_R1e6_line_by_line",
            "high_resolution_response": "Doppler+rotation+Gaussian_LSF+pixels",
        },
        opacity_identifiers={
            "H2O_LBL_sha256": EXPECTED_SHA256["H2O"],
            "CO_LBL_sha256": EXPECTED_SHA256["CO"],
        },
    )
    start = perf_counter()
    oe_result = run_retrieval(
        problem,
        method="optimal_estimation",
        output_dir=OUTPUT_DIR / "optimal_estimation",
        initial_state=[1450.0, -3.6, -3.4, 0.0],
        max_iterations=12,
        convergence_tolerance=2.0e-4,
        finite_difference_fraction=2.0e-3,
        damping=1.0e-8,
    )
    oe_seconds = perf_counter() - start
    best_fit_spectra = problem.model_spectra(oe_result.best_fit_parameters)
    recovery = evaluate_multi_dataset_injection_recovery(
        case_name="combined-low-high-resolution-emission-injection",
        truth=TRUTH,
        estimates=oe_result.best_fit_parameters,
        absolute_tolerances={
            "temperature_K": 80.0,
            "log10_h2o_vmr": 0.25,
            "log10_co_vmr": 0.25,
            "radial_velocity_km_s": 1.0,
        },
        observations=observations,
        best_fit_spectra=best_fit_spectra,
        likelihood=likelihood,
        seed=args.seed,
        parameter_order=problem.parameter_names,
        posterior_covariance=oe_result.inference_result.covariance,
        inference_converged=oe_result.converged,
        reduced_chi_square_bounds=reduced_chi_square_bounds,
        metadata={
            "noise_scale": f"{args.noise_scale:.12g}",
            "likelihood": "Gaussian low resolution + projected high resolution",
            "fit_gate": fit_gate_description,
        },
    )
    nested_record: dict[str, object]
    if args.run_pymultinest:
        nested_record = _run_pymultinest_proof(
            problem,
            output_dir=OUTPUT_DIR / "pymultinest_proof",
            seed=args.seed,
            n_live_points=args.pymultinest_live_points,
            max_iter=args.pymultinest_max_iter,
            evidence_tolerance=args.pymultinest_evidence_tolerance,
            sampling_efficiency=args.pymultinest_sampling_efficiency,
        )
    else:
        nested_record = {
            "status": "not_run",
            "gate_passed": None,
            "scope": "one-dimensional radial-velocity PyMultiNest slice",
            "full_four_parameter_nested_sampling": "not_run",
            "limitation": (
                "Use --run-pymultinest to run the resource-bounded two-seed proof. "
                "The default command reports the noisy OE recovery only."
            ),
        }

    peak_rss = _peak_rss_bytes()
    report = recovery.to_mapping()
    memory_passed = bool(peak_rss < max_memory_bytes)
    sampling_passed = (
        None if not args.run_pymultinest else bool(nested_record["gate_passed"])
    )
    recovery_passed = bool(recovery.passed)
    if args.run_pymultinest:
        status = "pass" if recovery_passed and memory_passed and sampling_passed else "fail"
    else:
        status = (
            "pass_recovery_only_sampling_not_run"
            if recovery_passed and memory_passed
            else "fail"
        )
    report.update(
        {
            "status": status,
            "command": " ".join(sys.argv),
            "recovery_passed": recovery_passed,
            "memory_guard_passed": memory_passed,
            "sampling_gate_passed": sampling_passed,
            "table_inputs": table_records,
            "truth": TRUTH,
            "best_fit": dict(oe_result.best_fit_parameters),
            "preparation": {
                key: value
                for key, value in preparation.items()
                if key != "high_response"
            },
            "timings_seconds": {
                "model_preparation_and_truth": preparation_seconds,
                "optimal_estimation": oe_seconds,
                "total_workflow": perf_counter() - workflow_started,
            },
            "resources": {
                "cpu_thread_limit": 3,
                "max_memory_bytes": max_memory_bytes,
                "opacity_memory_limit_bytes": OPACITY_MEMORY_BYTES,
                "measured_peak_rss_bytes": peak_rss,
                "process_limit_rule": "measured_peak_rss < max_memory_bytes < 2 GiB",
                "opacity_limit_rule": "estimated_opacity_slice < 1 GiB",
                "cached_velocity_responses": preparation[
                    "high_response"
                ].cached_response_count,
            },
            "pymultinest": nested_record,
        }
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = REPORT_PATH.with_name(
        f".{REPORT_PATH.name}.{os.getpid()}.tmp"
    )
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(REPORT_PATH)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "fail":
        raise SystemExit("combined-resolution injection recovery failed")
    return report


if __name__ == "__main__":
    main()
