"""Run a multi-gas HAT-P-32b-like RT injection-recovery validation case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from retrieve_hat_p_32b_rt import (
    DEFAULT_KTA_DIR,
    DEFAULT_PT_CSV,
    build_hat_p_32b_forward_model,
)
from robert_exoplanets import (
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    UniformPrior,
    evaluate_injection_recovery,
    inject_spectrum,
    run_retrieval,
    write_injection_recovery_report,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hat_p_32b_injection_recovery"
OPACITY_SPECIES = ("H2O", "CO2", "NH3")
INJECTED_PARAMETERS = {
    "log_h2o": -3.5,
    "log_co2": -4.5,
    "log_nh3": -4.0,
    "temperature_offset": 80.0,
    "radius_scale": 1.03,
}
ABSOLUTE_RECOVERY_TOLERANCES = {
    "log_h2o": 0.75,
    "log_co2": 0.75,
    "log_nh3": 0.75,
    "temperature_offset": 75.0,
    "radius_scale": 0.04,
}


def main() -> dict[str, object]:
    """Run the validation case and return its report mapping."""

    args = _parser().parse_args()
    pt_csv = Path(args.pt_csv).expanduser()
    kta_dir = Path(args.kta_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    _validate_inputs(pt_csv, kta_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(args.wavelength_min, args.wavelength_max, args.n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    uncertainty = np.full(args.n_bins, args.uncertainty_ppm * 1.0e-6)
    template_observation = Observation.from_arrays(
        wavelength=bin_centres,
        wavelength_bin_edges=bin_edges,
        flux=np.zeros(args.n_bins),
        uncertainty=uncertainty,
        instrument="synthetic JWST/NIRSpec G395H-like bins",
    )
    forward_model = build_hat_p_32b_forward_model(
        template_observation,
        pt_csv=pt_csv,
        kta_dir=kta_dir,
        include_rayleigh=not args.no_rayleigh,
        exok_num=args.exok_num,
        opacity_species=OPACITY_SPECIES,
    )
    truth_spectrum = forward_model(INJECTED_PARAMETERS)
    observation = inject_spectrum(
        truth_spectrum,
        uncertainty,
        seed=args.seed,
        noise_scale=args.noise_scale,
        instrument=template_observation.instrument,
        metadata={
            "case": "HAT-P-32b-like multi-gas emission",
            "bin_definition": "explicit_uniform_wavelength_edges",
        },
    )
    parameters = _parameters()
    problem = RetrievalProblem(
        name="hat-p-32b-multigas-injection-recovery",
        observation=observation,
        parameters=parameters,
        forward_model=forward_model,
        metadata={
            **dict(forward_model.manifest_metadata),
            "validation": "injection_recovery",
            "synthetic_bins": "explicit_uniform_wavelength_edges",
        },
        opacity_identifiers=forward_model.opacity_identifiers,
    )
    if args.method == "optimal_estimation":
        result = run_retrieval(
            problem,
            method="optimal_estimation",
            output_dir=output_dir / "optimal_estimation",
            max_iterations=args.max_iterations,
            damping=args.damping,
        )
    else:
        result = run_retrieval(
            problem,
            method="ultranest",
            output_dir=output_dir / "ultranest",
            seed=args.seed,
            min_num_live_points=args.live_points,
            max_ncalls=args.max_ncalls,
            dlogz=args.dlogz,
            resume=args.resume,
            mpi_nprocs=1,
            show_status=True,
        )

    best_fit_spectrum = problem.model_spectrum(result.best_fit_parameters)
    report = evaluate_injection_recovery(
        case_name="HAT-P-32b-like multi-gas emission",
        truth=INJECTED_PARAMETERS,
        estimates=result.best_fit_parameters,
        absolute_tolerances=ABSOLUTE_RECOVERY_TOLERANCES,
        observation=observation,
        best_fit_spectrum=best_fit_spectrum,
        seed=args.seed,
        parameter_order=problem.parameter_names,
        posterior_covariance=_posterior_covariance(result),
        inference_converged=result.converged,
        reduced_chi_square_bounds=tuple(args.reduced_chi_square_bounds),
        metadata={
            "method": result.method,
            "retrieval_config_hash": result.manifest.config_hash,
            "opacity_species": ",".join(OPACITY_SPECIES),
            "noise_scale": f"{args.noise_scale:.12g}",
            "uncertainty_ppm": f"{args.uncertainty_ppm:.12g}",
        },
    )
    report_path = write_injection_recovery_report(
        report,
        output_dir / f"hat_p_32b_{args.method}_injection_recovery.json",
    )
    truth_path = output_dir / "injected_case.json"
    truth_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "truth": INJECTED_PARAMETERS,
                "absolute_recovery_tolerances": ABSOLUTE_RECOVERY_TOLERANCES,
                "wavelength_bin_edges_micron": bin_edges.tolist(),
                "uncertainty_ppm": args.uncertainty_ppm,
                "noise_scale": args.noise_scale,
                "opacity_identifiers": dict(problem.opacity_identifiers),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plot_path = output_dir / f"hat_p_32b_{args.method}_injection_recovery.png"
    _plot(plot_path, observation, truth_spectrum, best_fit_spectrum, report.reduced_chi_square)
    print(f"Validation passed: {report.passed}")
    print(f"Inference converged: {report.inference_converged}")
    print(f"Reduced chi-square: {report.reduced_chi_square:.4f}")
    print(f"Wrote {report_path}")
    print(f"Wrote {truth_path}")
    print(f"Wrote {plot_path}")
    return report.to_mapping()


def _parameters() -> RetrievalParameterSet:
    return RetrievalParameterSet(
        (
            RetrievalParameter("log_h2o", UniformPrior(-7.0, -1.0)),
            RetrievalParameter("log_co2", UniformPrior(-8.0, -2.0)),
            RetrievalParameter("log_nh3", UniformPrior(-8.0, -2.0)),
            RetrievalParameter("temperature_offset", UniformPrior(-250.0, 250.0), unit="K"),
            RetrievalParameter("radius_scale", UniformPrior(0.85, 1.15)),
        )
    )


def _posterior_covariance(result) -> np.ndarray | None:
    inference_result = result.inference_result
    covariance = getattr(inference_result, "covariance", None)
    if covariance is not None:
        return np.asarray(covariance, dtype=float)
    samples = np.asarray(getattr(inference_result, "samples", np.empty((0, 0))), dtype=float)
    if samples.shape[0] < 2:
        return None
    weights = getattr(inference_result, "weights", None)
    if weights is None:
        return np.asarray(np.cov(samples, rowvar=False), dtype=float)
    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights = normalized_weights / np.sum(normalized_weights)
    return np.asarray(np.cov(samples, rowvar=False, aweights=normalized_weights), dtype=float)


def _validate_inputs(pt_csv: Path, kta_dir: Path) -> None:
    required = [pt_csv, *(kta_dir / f"{species}_emission_R1000.kta" for species in OPACITY_SPECIES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing injection-recovery inputs:\n" + "\n".join(missing))


def _plot(path: Path, observation, truth_spectrum, best_fit_spectrum, reduced_chi_square: float) -> None:
    residual_sigma = (observation.flux - best_fit_spectrum.values) / observation.uncertainty
    fig, (spectrum_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
        constrained_layout=True,
    )
    spectrum_axis.errorbar(
        observation.wavelength,
        observation.flux * 1.0e6,
        yerr=observation.uncertainty * 1.0e6,
        fmt=".",
        color="#222222",
        ecolor="#999999",
        label="Injected observation",
    )
    spectrum_axis.plot(
        truth_spectrum.spectral_grid.values,
        truth_spectrum.values * 1.0e6,
        color="#4c78a8",
        label="Injected truth",
    )
    spectrum_axis.plot(
        best_fit_spectrum.spectral_grid.values,
        best_fit_spectrum.values * 1.0e6,
        color="#f58518",
        label="Retrieved best fit",
    )
    spectrum_axis.set_ylabel("Eclipse depth [ppm]")
    spectrum_axis.set_title(f"Multi-gas injection recovery — reduced χ² = {reduced_chi_square:.3f}")
    spectrum_axis.grid(alpha=0.25)
    spectrum_axis.legend(frameon=False)
    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    residual_axis.plot(observation.wavelength, residual_sigma, ".", color="#4c78a8")
    residual_axis.set_xlabel("Wavelength [micron]")
    residual_axis.set_ylabel("Residual [σ]")
    residual_axis.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-csv", default=str(DEFAULT_PT_CSV))
    parser.add_argument("--kta-dir", default=str(DEFAULT_KTA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--method", choices=("optimal_estimation", "ultranest"), default="optimal_estimation")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--n-bins", type=int, default=48)
    parser.add_argument("--wavelength-min", type=float, default=2.85)
    parser.add_argument("--wavelength-max", type=float, default=5.15)
    parser.add_argument("--uncertainty-ppm", type=float, default=30.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--exok-num", type=int, default=300)
    parser.add_argument("--no-rayleigh", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--live-points", type=int, default=100)
    parser.add_argument("--max-ncalls", type=int, default=10000)
    parser.add_argument("--dlogz", type=float, default=0.5)
    parser.add_argument("--resume", default="overwrite")
    parser.add_argument("--reduced-chi-square-bounds", nargs=2, type=float, default=(0.5, 1.5))
    return parser


if __name__ == "__main__":
    main()
