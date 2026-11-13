#!/usr/bin/env python3
"""Qualify supervised MultiNest MPI failures and native checkpoint resume.

Run this on a single Linux compute node inside an allocation. The parent
process starts every MultiNest world through ROBERT's callback supervisor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from typing import Iterator

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robert_exoplanets import (  # noqa: E402
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    run_multinest,
)
from robert_exoplanets.core import RobertConfigError  # noqa: E402
from robert_exoplanets.retrieval.samplers.callback_supervisor import (  # noqa: E402
    supervise_retrieval_command,
)
from robert_exoplanets.retrieval.samplers.multinest import (  # noqa: E402
    _native_likelihood_evaluations,
)


INJECTED_LEVEL = 0.72
DEFAULT_RANKS = 4
DEFAULT_LIVE_POINTS = 64
DEFAULT_CHECKPOINT_ITERATIONS = 256


def _mpi_identity() -> tuple[int, int]:
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError("mpi4py is required for this qualification") from exc
    communicator = MPI.COMM_WORLD
    return int(communicator.Get_rank()), int(communicator.Get_size())


def _problem(*, failing_rank: int | None = None) -> RetrievalProblem:
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        flux=[INJECTED_LEVEL, INJECTED_LEVEL],
        uncertainty=[0.1, 0.1],
    )

    def forward(parameters):
        if failing_rank is not None:
            rank, _ = _mpi_identity()
            if rank == failing_rank:
                raise RuntimeError("controlled non-root callback failure")
        return Spectrum.from_arrays(
            observation.wavelength,
            np.full(observation.n_points, parameters["level"]),
            unit="eclipse_depth",
            observable="eclipse_depth",
        )

    return RetrievalProblem(
        name="multinest-mpi-qualification",
        observation=observation,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("level", UniformPrior(0.0, 2.0)),)
        ),
        forward_model=forward,
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    return float(np.interp(probability, cumulative, ordered_values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _worker(args: argparse.Namespace) -> int:
    rank, world_size = _mpi_identity()
    if world_size != args.ranks:
        raise RuntimeError(
            f"worker sees {world_size} MPI ranks, expected {args.ranks}"
        )

    result = run_multinest(
        _problem(failing_rank=1 if args.scenario == "callback-failure" else None),
        output_dir=args.output_dir,
        n_live_points=args.live_points,
        evidence_tolerance=args.evidence_tolerance,
        max_iter=args.max_iterations,
        resume=args.resume,
        mpi_nprocs=args.ranks,
        seed=args.seed,
        verbose=False,
        n_iter_before_update=32,
    )

    if rank == 0:
        weights = np.array(result.weights, dtype=float, copy=True)
        weights /= weights.sum()
        samples = np.asarray(result.samples[:, 0], dtype=float)
        mean = float(np.sum(weights * samples))
        variance = float(np.sum(weights * (samples - mean) ** 2))
        resume_file = Path(args.output_dir) / "chains" / "1-resume.dat"
        evaluations = _native_likelihood_evaluations(resume_file)
        summary = {
            "scenario": args.scenario,
            "mpi_ranks": world_size,
            "seed": args.seed,
            "max_iterations": args.max_iterations,
            "resume": args.resume,
            "sample_count": int(samples.size),
            "weight_effective_sample_size": float(1.0 / np.sum(weights**2)),
            "weighted_mean": mean,
            "weighted_standard_deviation": float(np.sqrt(variance)),
            "weighted_quantiles_02_50_98": [
                _weighted_quantile(samples, weights, q) for q in (0.025, 0.5, 0.975)
            ],
            "weighted_quantiles_16_50_84": [
                _weighted_quantile(samples, weights, q) for q in (0.16, 0.5, 0.84)
            ],
            "best_fit_level": float(result.best_fit_parameters["level"]),
            "log_evidence": float(result.log_evidence),
            "log_evidence_error": float(result.log_evidence_error),
            "converged": bool(result.converged),
            "native_likelihood_evaluations": evaluations,
            "resume_file": str(resume_file),
            "resume_file_sha256": _sha256(resume_file),
        }
        _write_json(args.summary_path, summary)
    return 0


@contextmanager
def _phase_timeout(seconds: float) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        raise ValueError("phase timeout must be positive and needs POSIX setitimer")

    def raise_timeout(signum: int, frame: object) -> None:
        del signum, frame
        raise TimeoutError(f"qualification phase exceeded {seconds:g} seconds")

    old_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _supervised_run(
    args: argparse.Namespace,
    *,
    scenario: str,
    output_dir: Path,
    summary_path: Path,
    resume: bool,
    max_iterations: int,
) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--scenario",
        scenario,
        "--output-dir",
        str(output_dir),
        "--summary-path",
        str(summary_path),
        "--ranks",
        str(args.ranks),
        "--live-points",
        str(args.live_points),
        "--evidence-tolerance",
        str(args.evidence_tolerance),
        "--max-iterations",
        str(max_iterations),
        "--seed",
        str(args.seed),
        "--resume",
        "true" if resume else "false",
    ]
    environment = dict(os.environ)
    environment["ROBERT_MPI_RANKS"] = str(args.ranks)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"

    with _phase_timeout(args.phase_timeout_seconds):
        return supervise_retrieval_command(
            command,
            fallback_output_dir=output_dir,
            mpi_nprocs=args.ranks,
            mpi_launcher=args.mpi_launcher,
            environment=environment,
        )


def _weighted_result_checks(
    resumed: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    mean_resumed = float(resumed["weighted_mean"])
    mean_reference = float(reference["weighted_mean"])
    sigma_resumed = float(resumed["weighted_standard_deviation"])
    sigma_reference = float(reference["weighted_standard_deviation"])
    ess_resumed = float(resumed["weight_effective_sample_size"])
    ess_reference = float(reference["weight_effective_sample_size"])

    mean_mc_error = np.sqrt(
        sigma_resumed**2 / max(ess_resumed, 1.0)
        + sigma_reference**2 / max(ess_reference, 1.0)
    )
    mean_limit = max(0.05, 4.0 * float(mean_mc_error))
    median_resumed = float(resumed["weighted_quantiles_16_50_84"][1])
    median_reference = float(reference["weighted_quantiles_16_50_84"][1])
    median_limit = max(0.08, 4.0 * float(mean_mc_error))
    logz_resumed = float(resumed["log_evidence"])
    logz_reference = float(reference["log_evidence"])
    logz_error = np.hypot(
        float(resumed["log_evidence_error"]),
        float(reference["log_evidence_error"]),
    )
    logz_limit = max(0.5, 4.0 * float(logz_error))

    checks = {
        "both_runs_converged": bool(resumed["converged"] and reference["converged"]),
        "resumed_mean_matches_reference": abs(mean_resumed - mean_reference) <= mean_limit,
        "resumed_median_matches_reference": abs(median_resumed - median_reference) <= median_limit,
        "resumed_evidence_matches_reference": abs(logz_resumed - logz_reference) <= logz_limit,
        "injected_value_within_resumed_95_percent_interval": (
            float(resumed["weighted_quantiles_02_50_98"][0])
            <= INJECTED_LEVEL
            <= float(resumed["weighted_quantiles_02_50_98"][2])
        ),
        "best_fit_near_injected_value": (
            abs(float(resumed["best_fit_level"]) - INJECTED_LEVEL) <= 0.15
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "differences": {
            "mean_absolute": abs(mean_resumed - mean_reference),
            "mean_limit": mean_limit,
            "median_absolute": abs(median_resumed - median_reference),
            "median_limit": median_limit,
            "log_evidence_absolute": abs(logz_resumed - logz_reference),
            "log_evidence_limit": logz_limit,
            "injected_level": INJECTED_LEVEL,
        },
    }


def _qualify(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    failure_dir = output_root / "callback-failure"
    restart_dir = output_root / "checkpoint-resume"
    reference_dir = output_root / "uninterrupted-reference"
    failure_summary = output_root / "callback-failure-summary.json"
    capped_summary = output_root / "checkpoint-capped-summary.json"
    resumed_summary = output_root / "checkpoint-resumed-summary.json"
    reference_summary = output_root / "reference-summary.json"

    failure_code = _supervised_run(
        args,
        scenario="callback-failure",
        output_dir=failure_dir,
        summary_path=failure_summary,
        resume=False,
        max_iterations=0,
    )
    failure_status_path = failure_dir / "sampler_status.json"
    if not failure_status_path.is_file():
        raise RuntimeError("callback failure did not write sampler status")
    failure_status = json.loads(failure_status_path.read_text(encoding="utf-8"))
    failure_checks = {
        "supervisor_returned_failure": failure_code == 1,
        "status_reports_callback_failure": failure_status.get("callback_failure") is True,
        "failure_came_from_rank_one": failure_status.get("callback_rank") == 1,
        "all_mpi_ranks_stopped": failure_status.get(
            "callback_supervisor_cleanup_verified"
        ) is True,
        "resume_marked_unsafe": failure_status.get("multinest_resume_safe") is False,
        "partial_results_not_published": not (failure_dir / "result.json").exists(),
    }
    if not all(failure_checks.values()):
        raise RuntimeError(f"MPI callback failure checks failed: {failure_checks}")

    try:
        run_multinest(
            _problem(),
            output_dir=failure_dir,
            n_live_points=args.live_points,
            evidence_tolerance=args.evidence_tolerance,
            max_iter=0,
            resume=True,
            mpi_nprocs=1,
            seed=args.seed,
            verbose=False,
        )
    except RobertConfigError as exc:
        failure_checks["unsafe_restart_refused"] = "unsafe to resume" in str(exc)
    else:
        failure_checks["unsafe_restart_refused"] = False
    if not all(failure_checks.values()):
        raise RuntimeError(f"unsafe restart check failed: {failure_checks}")

    capped_code = _supervised_run(
        args,
        scenario="checkpoint-capped",
        output_dir=restart_dir,
        summary_path=capped_summary,
        resume=False,
        max_iterations=args.checkpoint_iterations,
    )
    capped_status = json.loads(
        (restart_dir / "sampler_status.json").read_text(encoding="utf-8")
    )
    resume_file = restart_dir / "chains" / "1-resume.dat"
    if capped_code != 0 or not resume_file.is_file():
        raise RuntimeError("capped MPI run did not leave a resumable MultiNest checkpoint")
    checkpoint_evaluations = _native_likelihood_evaluations(resume_file)
    checkpoint_hash = _sha256(resume_file)
    if checkpoint_evaluations is None:
        raise RuntimeError("could not read the native MultiNest checkpoint counter")

    resumed_code = _supervised_run(
        args,
        scenario="checkpoint-resumed",
        output_dir=restart_dir,
        summary_path=resumed_summary,
        resume=True,
        max_iterations=0,
    )
    resumed_status = json.loads(
        (restart_dir / "sampler_status.json").read_text(encoding="utf-8")
    )
    if resumed_code != 0 or not resumed_summary.is_file():
        raise RuntimeError("uncapped MPI restart did not finish successfully")
    resumed = json.loads(resumed_summary.read_text(encoding="utf-8"))
    resumed_evaluations = resumed.get("native_likelihood_evaluations")
    reuse_checks = {
        "capped_run_reached_iteration_limit": (
            capped_status.get("state") == "iteration_limit_reached"
        ),
        "restart_status_reports_resume": resumed_status.get("resume") is True,
        "native_checkpoint_counter_increased": (
            isinstance(resumed_evaluations, int)
            and resumed_evaluations > checkpoint_evaluations
        ),
        "native_checkpoint_changed": (
            resumed.get("resume_file_sha256") != checkpoint_hash
        ),
        "resumed_run_converged": resumed.get("converged") is True,
    }
    if not all(reuse_checks.values()):
        raise RuntimeError(f"native checkpoint reuse checks failed: {reuse_checks}")

    reference_code = _supervised_run(
        args,
        scenario="uninterrupted-reference",
        output_dir=reference_dir,
        summary_path=reference_summary,
        resume=False,
        max_iterations=0,
    )
    if reference_code != 0 or not reference_summary.is_file():
        raise RuntimeError("uninterrupted MPI reference run did not finish")
    reference = json.loads(reference_summary.read_text(encoding="utf-8"))
    comparison = _weighted_result_checks(resumed, reference)

    report = {
        "schema_version": 1,
        "status": "passed" if comparison["passed"] else "failed",
        "mpi_ranks": args.ranks,
        "mpi_launcher": str(Path(args.mpi_launcher).expanduser()),
        "seed": args.seed,
        "live_points": args.live_points,
        "checkpoint_iterations": args.checkpoint_iterations,
        "evidence_tolerance": args.evidence_tolerance,
        "phase_timeout_seconds": args.phase_timeout_seconds,
        "injected_level": INJECTED_LEVEL,
        "callback_failure": {
            "checks": failure_checks,
            "status": failure_status,
        },
        "checkpoint_resume": {
            "checks": reuse_checks,
            "checkpoint_native_likelihood_evaluations": checkpoint_evaluations,
            "resumed_native_likelihood_evaluations": resumed_evaluations,
            "checkpoint_sha256_before_resume": checkpoint_hash,
            "resumed_summary": resumed,
            "uninterrupted_reference": reference,
            "posterior_comparison": comparison,
        },
    }
    _write_json(output_root / "qualification.json", report)
    if not comparison["passed"]:
        raise RuntimeError("checkpoint restart did not match the uninterrupted reference")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mpi-launcher", default=os.environ.get("ROBERT_MPIEXEC"))
    parser.add_argument("--ranks", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--live-points", type=int, default=DEFAULT_LIVE_POINTS)
    parser.add_argument(
        "--checkpoint-iterations",
        type=int,
        default=DEFAULT_CHECKPOINT_ITERATIONS,
    )
    parser.add_argument("--evidence-tolerance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=241113)
    parser.add_argument("--phase-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scenario",
        choices=[
            "callback-failure",
            "checkpoint-capped",
            "checkpoint-resumed",
            "uninterrupted-reference",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--summary-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--max-iterations", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume",
        choices=["true", "false"],
        default="false",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    if args.worker:
        if args.scenario is None or args.summary_path is None:
            parser.error("internal worker arguments are incomplete")
        args.output_dir = args.output_dir.expanduser().resolve()
        args.summary_path = args.summary_path.expanduser().resolve()
        args.resume = args.resume == "true"
        _worker(args)
        return 0

    if args.mpi_launcher is None:
        parser.error("set ROBERT_MPIEXEC or pass --mpi-launcher")
    if args.ranks < 2:
        parser.error("--ranks must be at least 2")
    if args.live_points < 20:
        parser.error("--live-points must be at least 20")
    if args.checkpoint_iterations <= 0:
        parser.error("--checkpoint-iterations must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.evidence_tolerance <= 0:
        parser.error("--evidence-tolerance must be positive")
    if args.phase_timeout_seconds <= 0:
        parser.error("--phase-timeout-seconds must be positive")
    if not args.output_dir.expanduser().resolve().parent.is_dir():
        parser.error("the output directory parent must already exist")

    try:
        report = _qualify(args)
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.is_dir():
            _write_json(output_dir / "qualification-failure.json", failure)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
