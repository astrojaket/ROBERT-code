#!/usr/bin/env python3
"""Run a deferred OE refinement from a completed nested-sampling result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from robert_exoplanets.atmosphere import (
    IsothermalTemperatureProfile,
    MadhusudhanSeager2009TemperatureProfile,
    ParmentierGuillot2014TemperatureProfile,
    SplineTemperatureProfile,
    TabulatedTemperatureProfile,
)
from robert_exoplanets.core import PressureGrid, RobertConfigError
from robert_exoplanets.io.configured_tasks import (
    build_problem,
    configured_temperature_prior_covariance,
    load_observations,
    smoke_evaluation,
    write_config_snapshot,
)
from robert_exoplanets.io.task_config import (
    TaskConfig,
    initialize_task_directories,
    load_task_config,
)
from robert_exoplanets.retrieval import (
    load_nested_sampler_result,
    nested_posterior_oe_prior,
    run_optimal_estimation_from_nested_result,
)


GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11


def main() -> None:
    args = _parser().parse_args()
    nested_config = load_task_config(args.nested_config)
    oe_config = load_task_config(args.oe_config)
    nested_result_dir = (
        args.nested_result_dir
        or nested_config.outputs.directory / "multinest"
    ).expanduser()
    output_dir = (
        args.output_dir or oe_config.outputs.directory / "optimal_estimation"
    ).expanduser()

    nested_result = load_nested_sampler_result(nested_result_dir)
    if nested_result.method != "multinest":
        raise RobertConfigError(
            f"expected a completed MultiNest result, found {nested_result.method!r}"
        )

    initialize_task_directories(oe_config)
    oe_problem = build_problem(oe_config, load_observations(oe_config))
    temperature_overrides = layer_temperature_overrides(
        nested_config,
        nested_result.best_fit_parameters,
        oe_problem,
    )
    prior_state, prior_covariance, _ = nested_posterior_oe_prior(
        nested_result,
        oe_problem,
        state_overrides=temperature_overrides,
        covariance_floor_fraction=args.covariance_floor_fraction,
    )
    prior_covariance, temperature_pressure = configured_temperature_prior_covariance(
        oe_config,
        oe_problem,
        prior_covariance,
    )
    prior_path = oe_config.outputs.directory / "multinest_to_oe_prior.npz"
    np.savez(
        prior_path,
        parameter_names=np.asarray(oe_problem.parameter_names),
        prior_state=prior_state,
        prior_covariance=prior_covariance,
        temperature_parameter_names=np.asarray(
            _spline_temperature_profile(oe_problem).parameter_names
        ),
        temperature_pressure_bar=temperature_pressure,
    )
    smoke = smoke_evaluation(oe_problem)
    write_config_snapshot(oe_config, args.oe_config)
    (oe_config.outputs.directory / "smoke_evaluation.json").write_text(
        json.dumps(smoke, indent=2), encoding="utf-8"
    )

    result, transferred = run_optimal_estimation_from_nested_result(
        nested_result,
        oe_problem=oe_problem,
        output_dir=output_dir,
        state_overrides=temperature_overrides,
        require_convergence=not args.allow_unconverged,
        covariance_floor_fraction=args.covariance_floor_fraction,
        prior_covariance=prior_covariance,
        oe_kwargs={
            "max_iterations": oe_config.sampler.oe_max_iterations,
            "convergence_tolerance": oe_config.sampler.oe_convergence_tolerance,
            "finite_difference_fraction": (
                oe_config.sampler.oe_finite_difference_fraction
            ),
            "damping": oe_config.sampler.oe_damping,
        },
    )
    handoff = {
        "workflow": "completed_multinest_to_optimal_estimation",
        "nested_configuration": str(args.nested_config.expanduser().resolve()),
        "nested_result_directory": str(nested_result_dir.resolve()),
        "oe_configuration": str(args.oe_config.expanduser().resolve()),
        "oe_output_directory": str(output_dir.resolve()),
        "nested_converged": nested_result.converged,
        "nested_message": nested_result.message,
        "transferred_prior_state": transferred,
        "temperature_state_overrides": temperature_overrides,
        "temperature_prior_sigma_k": (
            oe_config.sampler.oe_temperature_prior_sigma_k
        ),
        "temperature_correlation_length_dex": (
            oe_config.sampler.oe_temperature_correlation_length_dex
        ),
        "prior_arrays": str(prior_path.resolve()),
        "oe_converged": result.converged,
        "oe_message": result.message,
    }
    handoff_path = oe_config.outputs.directory / "multinest_to_oe_handoff.json"
    handoff_path.write_text(
        json.dumps(handoff, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"OE converged={result.converged}; message={result.message}; "
        f"handoff={handoff_path}",
        flush=True,
    )


def layer_temperature_overrides(
    nested_config: TaskConfig,
    nested_best_fit: Mapping[str, float],
    oe_problem: object,
) -> dict[str, float]:
    """Evaluate the nested T-P profile at every OE spline knot."""

    builder = _atmosphere_builder(oe_problem)
    target_profile = _spline_temperature_profile(oe_problem)
    if target_profile.knot_temperature is not None:
        raise RobertConfigError("the OE spline temperatures must be retrieved parameters")
    if target_profile.pressure_unit != "bar" or builder.pressure_grid.unit != "bar":
        raise RobertConfigError(
            "deferred layer-temperature handoff currently requires pressure in bar"
        )

    source_grid = _pressure_grid(nested_config)
    source_profile = _temperature_profile(nested_config)
    source_temperature = source_profile.evaluate(nested_best_fit, source_grid)
    order = np.argsort(source_grid.centers)
    source_log_pressure = np.log10(source_grid.centers[order])
    target_log_pressure = np.log10(target_profile.knot_pressure)
    if (
        target_log_pressure[0] < source_log_pressure[0]
        or target_log_pressure[-1] > source_log_pressure[-1]
    ):
        raise RobertConfigError(
            "OE temperature knots extend outside the nested temperature grid"
        )
    target_temperature = np.interp(
        target_log_pressure,
        source_log_pressure,
        source_temperature[order],
    )
    return {
        name: float(value)
        for name, value in zip(
            target_profile.parameter_names or (), target_temperature, strict=True
        )
    }


def _atmosphere_builder(oe_problem: object):
    forward_model = getattr(oe_problem, "forward_model", None)
    builder = getattr(forward_model, "atmosphere_builder", None)
    if builder is None:
        raise RobertConfigError(
            "layer-temperature handoff requires an emission problem with a shared "
            "atmosphere builder"
        )
    return builder


def _spline_temperature_profile(oe_problem: object) -> SplineTemperatureProfile:
    profile = _atmosphere_builder(oe_problem).temperature_profile
    if not isinstance(profile, SplineTemperatureProfile):
        raise RobertConfigError(
            "the OE configuration must use a retrieved spline temperature profile"
        )
    if profile.knot_temperature is not None:
        raise RobertConfigError("the OE spline temperatures must be retrieved parameters")
    return profile


def _pressure_grid(config: TaskConfig) -> PressureGrid:
    item = config.atmosphere.pressure
    return PressureGrid.from_log_centers(
        item.bottom_bar,
        item.top_bar,
        n_layers=item.layers,
        unit="bar",
        name=f"{config.bodies.planet.name} configured pressure grid",
    )


def _temperature_profile(config: TaskConfig):
    item = config.atmosphere.temperature
    if item.model == "parmentier_guillot_2014":
        planet = config.bodies.planet
        gravity = planet.gravity_m_s2
        if gravity is None:
            gravity = (
                GRAVITATIONAL_CONSTANT_M3_KG_S2
                * float(planet.mass_kg)
                / planet.radius_m**2
            )
        return ParmentierGuillot2014TemperatureProfile(
            gravity=float(gravity),
            internal_temperature=item.internal_temperature_k,
        )
    if item.model == "isothermal":
        return IsothermalTemperatureProfile(
            temperature=item.temperature_k,
            parameter_name=item.parameter_name,
        )
    if item.model == "tabulated":
        return TabulatedTemperatureProfile.from_csv(
            item.profile_path,
            pressure_column=item.pressure_column,
            temperature_column=item.temperature_column,
            pressure_unit=item.pressure_unit,
            extrapolation=item.extrapolation,
        )
    if item.model == "madhusudhan_seager_2009":
        return MadhusudhanSeager2009TemperatureProfile(
            pressure_unit=item.pressure_unit,
            reference_pressure=item.reference_pressure,
            p1_parameter_name=item.p1_parameter,
            p2_parameter_name=item.p2_parameter,
            p3_parameter_name=item.p3_parameter,
            t0_parameter_name=item.t0_parameter,
            alpha1_parameter_name=item.alpha1_parameter,
            alpha2_parameter_name=item.alpha2_parameter,
        )
    return SplineTemperatureProfile(
        knot_pressure=np.asarray(item.knot_pressure, dtype=float),
        knot_temperature=(
            None
            if item.knot_temperature_k is None
            else np.asarray(item.knot_temperature_k, dtype=float)
        ),
        parameter_names=item.parameter_names,
        pressure_unit=item.pressure_unit,
        extrapolation=item.extrapolation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-config", type=Path, required=True)
    parser.add_argument("--oe-config", type=Path, required=True)
    parser.add_argument(
        "--nested-result-dir",
        type=Path,
        help="completed result directory (default: <nested outputs>/multinest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="OE result directory (default: <OE outputs>/optimal_estimation)",
    )
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="allow diagnostic OE initialization from a non-converged MultiNest result",
    )
    parser.add_argument("--covariance-floor-fraction", type=float, default=1.0e-4)
    return parser


if __name__ == "__main__":
    main()
