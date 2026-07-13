"""Lightweight benchmark of ROBERT's two hybrid retrieval workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from robert_exoplanets import (
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    Spectrum,
    UniformPrior,
    run_nested_sampling_then_oe,
    run_oe_then_nested_sampling,
)


def _problem(*, detailed: bool = False) -> RetrievalProblem:
    wavelength = np.linspace(2.9, 5.1, 36)
    x = (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
    truth = 2.7e-3 + 8.0e-4 * x + 3.5e-4 * (np.square(x) - np.mean(np.square(x)))
    uncertainty = np.full(wavelength.size, 7.5e-5)
    observation = Observation.from_arrays(
        wavelength=wavelength,
        flux=truth,
        uncertainty=uncertainty,
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
    )
    parameters = [
        RetrievalParameter("baseline", UniformPrior(1.0e-3, 5.0e-3)),
        RetrievalParameter("slope", UniformPrior(-3.0e-3, 3.0e-3)),
    ]
    if detailed:
        parameters.append(RetrievalParameter("curvature", UniformPrior(-3.0e-3, 3.0e-3)))

    def forward(values):
        model = values["baseline"] + values["slope"] * x
        if detailed:
            model += values["curvature"] * (np.square(x) - np.mean(np.square(x)))
        return Spectrum.from_arrays(
            wavelength,
            model,
            unit=observation.flux_unit,
            observable=observation.observable,
        )

    return RetrievalProblem(
        name="hybrid-polynomial-detailed" if detailed else "hybrid-polynomial",
        observation=observation,
        parameters=RetrievalParameterSet(tuple(parameters)),
        forward_model=forward,
        metadata={"physics": "retrieval_workflow_benchmark_only"},
    )


def main() -> dict[str, object]:
    args = _parser().parse_args()
    output = Path(args.output_dir).expanduser()
    nested_settings = {
        "min_num_live_points": args.live_points,
        "max_ncalls": args.max_ncalls,
        "dlogz": args.dlogz,
        "min_ess": args.min_ess,
        "resume": "overwrite",
        "show_status": False,
    }

    started = time.perf_counter()
    oe_nested = run_oe_then_nested_sampling(
        _problem(),
        output_dir=output / "oe_then_nested",
        prior_sigma=4.0,
        oe_kwargs={"max_iterations": 6},
        nested_kwargs=nested_settings,
        seed=args.seed,
    )
    first_seconds = time.perf_counter() - started

    started = time.perf_counter()
    nested_oe = run_nested_sampling_then_oe(
        _problem(),
        oe_problem=_problem(detailed=True),
        oe_state_overrides={"curvature": 0.0},
        output_dir=output / "nested_then_oe",
        nested_kwargs=nested_settings,
        oe_kwargs={"max_iterations": 6},
        require_nested_convergence=args.require_convergence,
        seed=args.seed + 1,
    )
    second_seconds = time.perf_counter() - started

    report = {
        "benchmark_model": "polynomial workflow surrogate (not retrieval physics)",
        "oe_then_nested": {
            "seconds": first_seconds,
            "oe_converged": oe_nested.optimal_estimation.converged,
            "nested_converged": oe_nested.nested_sampling.converged,
            "nested_message": oe_nested.nested_sampling.message,
            "best_fit": dict(oe_nested.nested_sampling.best_fit_parameters),
            "refined_prior_bounds": dict(oe_nested.prior_bounds),
        },
        "nested_then_oe": {
            "seconds": second_seconds,
            "nested_converged": nested_oe.nested_sampling.converged,
            "nested_message": nested_oe.nested_sampling.message,
            "oe_converged": nested_oe.optimal_estimation.converged,
            "oe_best_fit": dict(nested_oe.optimal_estimation.best_fit_parameters),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "benchmark.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Wrote {path}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="examples/outputs/hybrid_retrieval_benchmark")
    parser.add_argument("--live-points", type=int, default=40)
    parser.add_argument("--max-ncalls", type=int, default=2500)
    parser.add_argument("--dlogz", type=float, default=2.0)
    parser.add_argument("--min-ess", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--require-convergence", action="store_true")
    return parser


if __name__ == "__main__":
    main()
