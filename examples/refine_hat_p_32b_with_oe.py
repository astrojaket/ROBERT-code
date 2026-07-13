"""Refine a converged HAT-P-32b nested retrieval with a spline-profile OE sounding."""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from robert_exoplanets import (
    MadhusudhanSeager2009TemperatureProfile,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    SplineTemperatureProfile,
    UniformPrior,
    build_parameterized_clear_sky_emission_model,
    load_emission_observation_npz,
    load_nested_sampler_result,
    nested_posterior_oe_prior,
    run_optimal_estimation_from_nested_result,
)

if __package__:
    from .hat_p_32b_fastchem_config import OBSERVATION_NPZ, RESULTS_DIR, make_model_config
else:
    from hat_p_32b_fastchem_config import OBSERVATION_NPZ, RESULTS_DIR, make_model_config

DEFAULT_NESTED_OUTPUT = Path("retrieval_runs/hat_p_32b_10k_4x2_20260712")
DEFAULT_OUTPUT = Path("retrieval_runs/hat_p_32b_spline_oe")


def main() -> dict[str, object]:
    args = _parser().parse_args()
    nested = load_nested_sampler_result(args.nested_output)
    if not nested.converged and not args.allow_unconverged:
        raise RuntimeError(
            f"HAT-P-32b nested retrieval is not converged ({nested.message}). "
            "Wait for the monitored run, or pass --allow-unconverged for diagnostics only."
        )

    observation = load_emission_observation_npz(
        OBSERVATION_NPZ,
        instrument="JWST/NIRSpec G395H",
    )
    base_config = make_model_config(n_layers=args.layers)
    pressure_grid = base_config.pressure_grid
    if pressure_grid is None:
        raise RuntimeError("HAT-P-32b spline OE requires the configured pressure grid")
    knot_pressure = np.geomspace(
        float(np.min(pressure_grid.centers)),
        float(np.max(pressure_grid.centers)),
        args.knots,
    )
    knot_names = tuple(f"temperature_{index}" for index in range(args.knots))
    spline = SplineTemperatureProfile(
        knot_pressure=knot_pressure,
        parameter_names=knot_names,
        pressure_unit=pressure_grid.unit,
        extrapolation="clip",
    )
    spline_config = replace(base_config, temperature_profile=spline)
    nested_forward_model = build_parameterized_clear_sky_emission_model(
        base_config,
        spectral_grid=observation.spectral_grid,
    )
    forward_model = build_parameterized_clear_sky_emission_model(
        spline_config,
        spectral_grid=observation.spectral_grid,
    )
    parameters = RetrievalParameterSet(
        (
            RetrievalParameter("metallicity", UniformPrior(-1.0, 2.0), unit="dex"),
            RetrievalParameter("CtoO", UniformPrior(0.0, 1.0)),
            *(
                RetrievalParameter(name, UniformPrior(300.0, 3500.0), unit="K")
                for name in knot_names
            ),
        )
    )
    problem = RetrievalProblem(
        name="hat-p-32b-fastchem-spline-oe-sounding",
        observation=observation,
        parameters=parameters,
        forward_model=forward_model,
        metadata={
            **dict(forward_model.manifest_metadata),
            "hybrid_workflow": "converged_nested_sampling_then_spline_optimal_estimation",
            "nested_source": str(Path(args.nested_output).expanduser()),
        },
        opacity_identifiers=forward_model.opacity_identifiers,
    )

    madhu = MadhusudhanSeager2009TemperatureProfile(
        pressure_unit=pressure_grid.unit,
        reference_pressure=1.0e-6,
    )
    nested_profile = madhu.evaluate(nested.best_fit_parameters, pressure_grid)
    knot_initial = _interpolate_profile(pressure_grid.centers, nested_profile, knot_pressure)
    overrides = dict(zip(knot_names, knot_initial, strict=True))
    _, prior_covariance, _ = nested_posterior_oe_prior(
        nested,
        problem,
        state_overrides=overrides,
    )
    prior_covariance = np.array(prior_covariance, copy=True)
    log_knots = np.log10(knot_pressure)
    temperature_covariance = args.temperature_prior_sigma**2 * np.exp(
        -np.abs(log_knots[:, None] - log_knots[None, :]) / args.correlation_length_dex
    )
    temperature_slice = slice(2, 2 + args.knots)
    prior_covariance[temperature_slice, temperature_slice] = temperature_covariance

    output = Path(args.output_dir).expanduser()
    oe_result, prior_state = run_optimal_estimation_from_nested_result(
        nested,
        oe_problem=problem,
        output_dir=output,
        state_overrides=overrides,
        require_convergence=not args.allow_unconverged,
        prior_covariance=prior_covariance,
        oe_kwargs={
            "max_iterations": args.max_iterations,
            "damping": args.damping,
            "finite_difference_fraction": args.finite_difference_fraction,
        },
    )
    oe_profile = spline.evaluate(oe_result.best_fit_parameters, pressure_grid)
    nested_spectrum = nested_forward_model(nested.best_fit_parameters)
    oe_spectrum = problem.model_spectrum(oe_result.best_fit_parameters)
    reference_pressure, reference_temperature = _reference_temperature()
    nested_rmse = _profile_rmse(pressure_grid.centers, nested_profile, reference_pressure, reference_temperature)
    oe_rmse = _profile_rmse(pressure_grid.centers, oe_profile, reference_pressure, reference_temperature)
    report = {
        "nested_output": str(Path(args.nested_output).expanduser()),
        "nested_converged": nested.converged,
        "nested_message": nested.message,
        "oe_converged": oe_result.converged,
        "oe_message": oe_result.message,
        "n_temperature_knots": args.knots,
        "knot_pressure_bar": knot_pressure.tolist(),
        "prior_state": prior_state,
        "oe_best_fit": dict(oe_result.best_fit_parameters),
        "reference_profile": "bundled NEMESIS median T(P)",
        "nested_parametric_profile_rmse_K": nested_rmse,
        "oe_spline_profile_rmse_K": oe_rmse,
        "oe_improves_reference_rmse": oe_rmse < nested_rmse,
        "nested_parametric_reduced_chi_square": _reduced_chi_square(observation, nested_spectrum.values),
        "oe_spline_reduced_chi_square": _reduced_chi_square(observation, oe_spectrum.values),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "hat_p_32b_spline_oe.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez(
        output / "hat_p_32b_spline_profiles.npz",
        pressure_bar=pressure_grid.centers,
        nested_temperature_K=nested_profile,
        oe_temperature_K=oe_profile,
        reference_pressure_bar=reference_pressure,
        reference_temperature_K=reference_temperature,
        knot_pressure_bar=knot_pressure,
    )
    _plot(output / "hat_p_32b_spline_oe.png", pressure_grid.centers, nested_profile, oe_profile, reference_pressure, reference_temperature, knot_pressure)
    _plot_spectrum(
        output / "hat_p_32b_spline_oe_spectrum.png",
        observation,
        nested_spectrum,
        oe_spectrum,
    )
    _plot_nested_posterior(
        output / "hat_p_32b_nested_posterior.png",
        Path(args.nested_output).expanduser() / "result_arrays.npz",
        nested.parameter_names,
        converged=nested.converged,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def _interpolate_profile(pressure, temperature, target_pressure):
    order = np.argsort(pressure)
    return np.interp(np.log10(target_pressure), np.log10(np.asarray(pressure)[order]), np.asarray(temperature)[order])


def _reference_temperature():
    with np.load(RESULTS_DIR / "quench_study_emission_TP_band.npz", allow_pickle=False) as archive:
        return np.array(archive["pressure_bar"], copy=True), np.array(archive["T_med"], copy=True)


def _profile_rmse(pressure, temperature, reference_pressure, reference_temperature) -> float:
    interpolated = _interpolate_profile(reference_pressure, reference_temperature, np.asarray(pressure))
    return float(np.sqrt(np.mean(np.square(np.asarray(temperature) - interpolated))))


def _reduced_chi_square(observation, model_values) -> float:
    residual = (observation.flux - np.asarray(model_values)) / observation.uncertainty
    return float(np.mean(np.square(residual)))


def _plot(path, pressure, nested_temperature, oe_temperature, reference_pressure, reference_temperature, knots) -> None:
    fig, axis = plt.subplots(figsize=(5.8, 6.2), constrained_layout=True)
    axis.plot(reference_temperature, reference_pressure, color="#222222", lw=1.8, label="Bundled reference median")
    axis.plot(nested_temperature, pressure, color="#4c78a8", lw=1.6, label="Nested parametric T(P)")
    axis.plot(oe_temperature, pressure, color="#f58518", lw=1.8, label="OE spline T(P)")
    axis.scatter(_interpolate_profile(pressure, oe_temperature, knots), knots, color="#f58518", s=20, zorder=3)
    axis.set(yscale="log", xlabel="Temperature [K]", ylabel="Pressure [bar]", title="HAT-P-32b nested → spline OE sounding")
    axis.invert_yaxis()
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_spectrum(path: Path, observation, nested_spectrum, oe_spectrum) -> None:
    nested_residual = (observation.flux - nested_spectrum.values) / observation.uncertainty
    oe_residual = (observation.flux - oe_spectrum.values) / observation.uncertainty
    fig, (spectrum_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.5),
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
        label="JWST observation",
    )
    spectrum_axis.plot(
        nested_spectrum.spectral_grid.values,
        nested_spectrum.values * 1.0e6,
        color="#4c78a8",
        lw=1.6,
        label="Nested parametric best fit",
    )
    spectrum_axis.plot(
        oe_spectrum.spectral_grid.values,
        oe_spectrum.values * 1.0e6,
        color="#f58518",
        lw=1.8,
        label="OE spline best fit",
    )
    spectrum_axis.set(ylabel="Eclipse depth [ppm]", title="HAT-P-32b best-fit emission spectra")
    spectrum_axis.grid(alpha=0.25)
    spectrum_axis.legend(frameon=False)
    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    residual_axis.plot(observation.wavelength, nested_residual, ".", color="#4c78a8", label="Nested")
    residual_axis.plot(observation.wavelength, oe_residual, ".", color="#f58518", label="OE")
    residual_axis.set(xlabel="Wavelength [micron]", ylabel="Residual [σ]")
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_nested_posterior(path: Path, arrays_path: Path, parameter_names, *, converged: bool) -> None:
    with np.load(arrays_path, allow_pickle=False) as arrays:
        samples = np.asarray(arrays["samples"])
        weights = np.asarray(arrays["weights"])
    labels = ("[M/H]", "C/O", "P1", "P2", "P3", "T0 [K]", "α1", "α2")
    count = samples.shape[1]
    fig, axes = plt.subplots(count, count, figsize=(12.0, 12.0), constrained_layout=True)
    for row in range(count):
        for column in range(count):
            axis = axes[row, column]
            if row < column:
                axis.set_visible(False)
            elif row == column:
                axis.hist(samples[:, column], bins=30, weights=weights, color="#4c78a8", alpha=0.85)
            else:
                axis.scatter(samples[:, column], samples[:, row], s=3, c="#4c78a8", alpha=0.3, linewidths=0)
            if row == count - 1 and row >= column:
                axis.set_xlabel(labels[column], fontsize=8)
            else:
                axis.set_xticklabels([])
            if column == 0 and row > 0:
                axis.set_ylabel(labels[row], fontsize=8)
            else:
                axis.set_yticklabels([])
    state = "converged" if converged else "unconverged — diagnostic only"
    fig.suptitle(f"HAT-P-32b nested posterior ({state})", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-output", default=str(DEFAULT_NESTED_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--knots", type=int, default=6)
    parser.add_argument("--layers", type=int, default=100)
    parser.add_argument("--temperature-prior-sigma", type=float, default=250.0)
    parser.add_argument("--correlation-length-dex", type=float, default=1.5)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--finite-difference-fraction", type=float, default=1.0e-3)
    parser.add_argument("--allow-unconverged", action="store_true")
    return parser


if __name__ == "__main__":
    main()
