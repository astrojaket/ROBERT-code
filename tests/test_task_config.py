"""Tests for the strict user-facing task configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from robert_exoplanets.io.task_config import (
    TaskConfig,
    initialize_task_directories,
    load_task_config,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configurations" / "wasp69b_clear_R1000.yaml"
DEFAULTS = tuple(sorted((ROOT / "configurations").glob("wasp*.yaml")))


def test_wasp69b_example_exposes_complete_native_mode_run() -> None:
    config = load_task_config(EXAMPLE)

    assert config.schema_version == 1
    assert config.observations.datasets == ("f322w2", "f444w", "lrs")
    assert config.opacity.resolution == "R1000"
    assert config.opacity.species == ("H2O", "CO2", "CO", "CH4", "NH3", "HCN")
    assert config.sampler.live_points == 400
    assert config.sampler.max_calls == 200_000
    assert config.runtime.mpi_processes == "auto"
    assert config.runtime.scratch_directory.is_absolute()
    assert config.outputs.directory.is_absolute()


def test_unknown_configuration_field_is_rejected() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["sampler"]["live_pointz"] = 400

    with pytest.raises(ValidationError, match="live_pointz"):
        TaskConfig.model_validate(raw)


def test_opacity_species_must_exist_in_chemistry() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["opacity"]["species"] = [*raw["opacity"]["species"], "TiO"]

    with pytest.raises(ValidationError, match="TiO"):
        TaskConfig.model_validate(raw)


def test_initialize_creates_only_configured_writable_directories(tmp_path: Path) -> None:
    config = load_task_config(EXAMPLE)
    output = tmp_path / "project" / "run"
    cache = tmp_path / "cache"
    scratch = tmp_path / "scratch"
    config = config.model_copy(
        update={
            "outputs": config.outputs.model_copy(update={"directory": output}),
            "opacity": config.opacity.model_copy(update={"cache_directory": cache}),
            "runtime": config.runtime.model_copy(update={"scratch_directory": scratch}),
        }
    )

    created = initialize_task_directories(config)

    assert set(created) == {
        output,
        cache,
        cache / "R1000",
        scratch,
        scratch / "numba",
        scratch / "matplotlib",
    }
    assert all(path.is_dir() for path in created)


def test_tabulated_temperature_profile_has_an_explicit_path() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["temperature"] = {
        "model": "tabulated",
        "profile_path": Path("inputs/pt.csv"),
        "extrapolation": "clip",
    }

    parsed = TaskConfig.model_validate(raw)

    assert parsed.atmosphere.temperature.model == "tabulated"
    assert parsed.atmosphere.temperature.profile_path == Path("inputs/pt.csv")


def test_all_shipped_wasp_defaults_resolve_and_validate() -> None:
    assert len(DEFAULTS) == 11
    for path in DEFAULTS:
        config = load_task_config(path)
        assert config.run.name.startswith(("wasp69b-", "wasp80b-"))
        assert config.opacity.resolution == "R1000"


def test_direct_nk_default_replaces_catalogue_cloud_fields() -> None:
    config = load_task_config(
        ROOT / "configurations" / "wasp69b_mie_direct_nk_pg14_R1000.yaml"
    )

    assert config.clouds.model == "mie_direct_nk"
    assert len(config.clouds.real_index_parameter_names) == 6
