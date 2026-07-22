"""Deferred completed-MultiNest to layer-resolved OE workflow tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from robert_exoplanets.atmosphere import SplineTemperatureProfile
from robert_exoplanets.io.configured_tasks import (
    configured_temperature_prior_covariance,
    describe_config,
)
from robert_exoplanets.io.task_config import TaskConfig, load_task_config
from robert_exoplanets.retrieval import log_pressure_correlated_covariance
from run_oe_from_nested import (
    _pressure_grid,
    _temperature_profile,
    layer_temperature_overrides,
)


ROOT = Path(__file__).resolve().parents[1]
NESTED_CONFIG = (
    ROOT / "configurations" / "wasp69b_mie_catalog_pg14_R1000_multinest.yaml"
)
OE_CONFIG = (
    ROOT
    / "configurations"
    / "wasp69b_mie_catalog_layer_by_layer_R1000_optimal_estimation.yaml"
)


def test_layer_by_layer_oe_configuration_matches_pressure_grid() -> None:
    config = load_task_config(OE_CONFIG)
    temperature = config.atmosphere.temperature

    assert config.sampler.engine == "optimal_estimation"
    assert config.runtime.mpi_processes == 1
    assert config.sampler.oe_temperature_prior_sigma_k == 250.0
    assert config.sampler.oe_temperature_correlation_length_dex == 1.5
    assert "correlated_temperature_prior=250 K/1.5 dex" in describe_config(config)
    assert temperature.model == "spline"
    assert len(temperature.knot_pressure) == config.atmosphere.pressure.layers == 80
    assert temperature.parameter_names == tuple(f"T_{index:02d}" for index in range(80))
    np.testing.assert_allclose(
        temperature.knot_pressure,
        np.logspace(2.0, -6.0, 80),
        rtol=5.0e-12,
    )
    assert len(config.parameters) == 87
    assert set(temperature.parameter_names).issubset(
        item.name for item in config.parameters
    )
    assert not {
        "kappa_IR",
        "gamma1",
        "gamma2",
        "T_irr",
        "alpha",
    }.intersection(item.name for item in config.parameters)


def test_pg14_best_fit_initializes_every_oe_temperature_layer() -> None:
    nested_config = load_task_config(NESTED_CONFIG)
    oe_config = load_task_config(OE_CONFIG)
    target_profile = _temperature_profile(oe_config)
    assert isinstance(target_profile, SplineTemperatureProfile)
    target_grid = _pressure_grid(oe_config)
    oe_problem = SimpleNamespace(
        forward_model=SimpleNamespace(
            atmosphere_builder=SimpleNamespace(
                temperature_profile=target_profile,
                pressure_grid=target_grid,
            )
        )
    )
    best_fit = {
        "kappa_IR": 0.01,
        "gamma1": 0.5,
        "gamma2": 2.0,
        "T_irr": 1500.0,
        "alpha": 0.4,
    }

    overrides = layer_temperature_overrides(nested_config, best_fit, oe_problem)
    source_temperature = _temperature_profile(nested_config).evaluate(
        best_fit, _pressure_grid(nested_config)
    )

    assert tuple(overrides) == target_profile.parameter_names
    assert len(overrides) == 80
    np.testing.assert_allclose(
        [overrides[f"T_{index:02d}"] for index in range(80)],
        source_temperature,
        rtol=1.0e-12,
    )


def test_log_pressure_temperature_covariance_has_requested_smoothing() -> None:
    covariance = log_pressure_correlated_covariance(
        [1.0, 10.0**1.5, 1000.0],
        standard_deviation=250.0,
        correlation_length_dex=1.5,
    )

    np.testing.assert_allclose(np.diag(covariance), 250.0**2)
    assert covariance[0, 1] == pytest.approx(250.0**2 / np.e)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_configured_temperature_prior_replaces_only_temperature_block() -> None:
    config = load_task_config(OE_CONFIG)
    profile = _temperature_profile(config)
    names = tuple(item.name for item in config.parameters)
    problem = SimpleNamespace(
        parameter_names=names,
        ndim=len(names),
        parameters=SimpleNamespace(
            parameters=tuple(
                SimpleNamespace(approximate_standard_deviation=1.0)
                for _ in names
            )
        ),
        forward_model=SimpleNamespace(
            atmosphere_builder=SimpleNamespace(temperature_profile=profile)
        ),
    )

    covariance, pressure = configured_temperature_prior_covariance(config, problem)
    temperature_indices = np.asarray(
        [names.index(name) for name in profile.parameter_names], dtype=int
    )
    temperature_block = covariance[np.ix_(temperature_indices, temperature_indices)]

    np.testing.assert_allclose(np.diag(temperature_block), 250.0**2)
    assert np.all(np.linalg.eigvalsh(temperature_block) > 0.0)
    np.testing.assert_allclose(pressure, profile.knot_pressure)
    assert covariance[names.index("metallicity"), names.index("metallicity")] == 1.0
    assert covariance[names.index("metallicity"), temperature_indices[0]] == 0.0


def test_temperature_prior_sigma_and_correlation_length_are_paired() -> None:
    config = load_task_config(OE_CONFIG)
    raw = config.model_dump(mode="python")
    raw["sampler"]["oe_temperature_correlation_length_dex"] = None

    with pytest.raises(ValidationError, match="configured together"):
        TaskConfig.model_validate(raw)
