"""Regression tests for the matched WASP-69b/WASP-80b retrieval matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from robert_exoplanets.io.task_config import configured_regions, load_task_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configurations" / "targets"
MODELS = {
    "clear": ("one_region", 7, "none"),
    "one-region": ("one_region", 11, "mie_catalog"),
    "diluted": ("diluted_one_region", 12, "mie_catalog"),
    "two-region": ("two_region", 21, "mie_catalog"),
    "n-k": ("one_region", 23, "mie_direct_nk"),
    "diluted-n-k": ("diluted_one_region", 24, "mie_direct_nk"),
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
    model_expectation: tuple[str, int, str],
) -> None:
    config = load_task_config(
        MATRIX / planet / "retrievals" / model_name / "configuration.yaml"
    )
    disk_model, base_parameter_count, cloud_model = model_expectation

    assert config.run.name == model_name
    assert config.observations.datasets == datasets
    assert config.disk_emission.model == disk_model
    assert config.sampler.engine == "multinest"
    assert config.sampler.live_points == 1000
    assert config.runtime.mpi_processes == 128
    assert config.opacity.resolution == "R1000"
    assert len(config.parameters) == base_parameter_count + parameter_offset

    regions = configured_regions(config)
    assert len(regions) == (2 if model_name == "two-region" else 1)
    assert all(region.clouds.model == cloud_model for region in regions)


@pytest.mark.parametrize("planet", ["WASP-69b", "WASP-80b"])
def test_two_region_matrix_uses_independent_temperature_and_cloud_parameters(
    planet: str,
) -> None:
    config = load_task_config(
        MATRIX / planet / "retrievals" / "two-region" / "configuration.yaml"
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


@pytest.mark.parametrize(
    ("planet", "observation_directory"),
    [
        ("WASP-69b", "wasp69b_schlawin2024"),
        ("WASP-80b", "wasp80b_wiser2025"),
    ],
)
def test_matrix_paths_resolve_from_each_nested_configuration(
    planet: str, observation_directory: str
) -> None:
    config = load_task_config(
        MATRIX / planet / "retrievals" / "clear" / "configuration.yaml"
    )

    assert config.observations.path.resolve() == ROOT / "data" / observation_directory
    assert config.atmosphere.chemistry.fastchem_path.resolve() == (
        ROOT / "data" / "chemistry" / "fastchem"
    )
    assert config.opacity.path.resolve() == ROOT / "opacity_data" / "ktables_exomol"
