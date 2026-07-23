"""Regression tests for the matched WASP-69b/WASP-80b retrieval matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from robert_exoplanets.io.task_config import configured_regions, load_task_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configurations" / "retrievals"
MODELS = {
    "clear": ("one_region", 0),
    "one-region": ("one_region", 0),
    "diluted": ("diluted_one_region", 1),
    "two-region": ("two_region", 1),
}


@pytest.mark.parametrize(
    ("planet", "datasets", "parameter_offset"),
    [
        ("WASP-69b", ("f322w2", "avg", "f444w", "lrs"), 1),
        ("WASP-80b", ("f322w2", "f444w", "lrs"), 0),
    ],
)
@pytest.mark.parametrize(("model_name", "model_expectation"), MODELS.items())
def test_wasp_retrieval_matrix_is_queue_ready(
    planet: str,
    datasets: tuple[str, ...],
    parameter_offset: int,
    model_name: str,
    model_expectation: tuple[str, int],
) -> None:
    config = load_task_config(
        MATRIX / planet / model_name / "configuration.yaml"
    )
    disk_model, extra_parameter_count = model_expectation
    base_parameter_count = {
        "clear": 7,
        "one-region": 11,
        "diluted": 11,
        "two-region": 20,
    }[model_name]

    assert config.run.name == model_name
    assert config.observations.datasets == datasets
    assert config.disk_emission.model == disk_model
    assert config.sampler.engine == "multinest"
    assert config.sampler.live_points == 400
    assert config.sampler.max_calls is None
    assert config.runtime.mpi_processes == "auto"
    assert config.opacity.resolution == "R1000"
    assert len(config.parameters) == (
        base_parameter_count + parameter_offset + extra_parameter_count
    )

    regions = configured_regions(config)
    assert len(regions) == (2 if model_name == "two-region" else 1)
    if model_name == "clear":
        assert regions[0].clouds.model == "none"
    else:
        assert all(region.clouds.model == "mie_catalog" for region in regions)


@pytest.mark.parametrize("planet", ["WASP-69b", "WASP-80b"])
def test_two_region_matrix_uses_independent_temperature_and_cloud_parameters(
    planet: str,
) -> None:
    config = load_task_config(
        MATRIX / planet / "two-region" / "configuration.yaml"
    )
    hot, cold = configured_regions(config)

    assert hot.atmosphere.chemistry == cold.atmosphere.chemistry
    assert hot.atmosphere.temperature.kappa_ir_parameter == "kappa_IR"
    assert cold.atmosphere.temperature.kappa_ir_parameter == "cold_kappa_IR"
    assert hot.clouds.log10_mass_fraction_parameter == "log_cloud_mass_fraction"
    assert (
        cold.clouds.log10_mass_fraction_parameter
        == "cold_log_cloud_mass_fraction"
    )
