"""Tests for the strict user-facing task configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from robert_exoplanets.io.configured_tasks import (
    load_observations as load_configured_observations,
)
from robert_exoplanets.io.task_config import (
    TaskConfig,
    initialize_task_directories,
    load_task_config,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configurations" / "wasp69b_cloud_free_R1000.yaml"
TEMPLATE = ROOT / "configurations" / "TEMPLATE_all_supported_options.yaml"
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
    assert config.plotting.enabled is False


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


def test_initialize_creates_only_configured_writable_directories(
    tmp_path: Path,
) -> None:
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
    assert len(DEFAULTS) == 19
    for path in DEFAULTS:
        config = load_task_config(path)
        assert config.run.name.startswith(("wasp69b-", "wasp80b-"))
        assert config.opacity.resolution == "R1000"


@pytest.mark.parametrize(
    ("suffix", "engine"),
    [
        ("multinest", "multinest"),
        ("optimal_estimation", "optimal_estimation"),
        ("optimal_estimation_to_ultranest", "optimal_estimation_to_ultranest"),
        ("optimal_estimation_to_multinest", "optimal_estimation_to_multinest"),
    ],
)
@pytest.mark.parametrize(
    "scenario", ["cloud_free_native_pg14", "mie_catalog_pg14"]
)
def test_wasp69b_inference_benchmarks_only_change_run_controls(
    scenario: str, suffix: str, engine: str
) -> None:
    baseline = load_task_config(
        ROOT / "configurations" / f"wasp69b_{scenario}_R1000.yaml"
    )
    benchmark = load_task_config(
        ROOT / "configurations" / f"wasp69b_{scenario}_R1000_{suffix}.yaml"
    )

    assert benchmark.sampler.engine == engine
    assert baseline.plotting.enabled is True
    assert benchmark.plotting.enabled is True
    for section in (
        "bodies",
        "observations",
        "atmosphere",
        "clouds",
        "opacity",
        "radiative_transfer",
        "likelihood",
        "parameters",
    ):
        assert getattr(benchmark, section) == getattr(baseline, section)


def test_direct_nk_default_replaces_catalogue_cloud_fields() -> None:
    config = load_task_config(
        ROOT / "configurations" / "wasp69b_mie_direct_nk_pg14_R1000.yaml"
    )

    assert config.clouds.model == "mie_direct_nk"
    assert len(config.clouds.real_index_parameter_names) == 6


def test_complete_template_uses_housekeeping_for_internal_paths() -> None:
    config = load_task_config(TEMPLATE)

    assert config.clouds.model == "none"
    assert config.housekeeping is not None
    assert config.observations.path == config.housekeeping.observations_directory
    assert (
        config.atmosphere.chemistry.fastchem_path
        == config.housekeeping.fastchem_directory
    )
    assert config.opacity.path == config.housekeeping.k_table_directory
    assert config.opacity.cache_directory == config.housekeeping.opacity_cache_directory
    assert config.outputs.directory == config.housekeeping.output_directory
    assert config.runtime.scratch_directory == config.housekeeping.scratch_directory
    assert config.plotting.enabled is False
    assert config.plotting.dataset_colors["f322w2"] == "#20639b"


def test_yaml_supports_named_dataset_offsets_and_uncertainty_inflation() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["observations"]["dataset_options"] = {
        "f322w2": {
            "offset_parameter": "nircam_offset",
            "uncertainty_scale_parameter": "nircam_error_scale",
        },
        "f444w": {"offset_parameter": "nircam_offset"},
        "lrs": {"uncertainty_scale": 1.15, "jitter_parameter": "miri_jitter"},
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "nircam_offset",
            "prior": {"type": "uniform", "lower": -1, "upper": 1},
        },
        {
            "name": "nircam_error_scale",
            "prior": {"type": "log_uniform", "lower": 0.5, "upper": 5},
        },
        {
            "name": "miri_jitter",
            "prior": {"type": "log_uniform", "lower": 1e-8, "upper": 1e-3},
        },
    ]

    parsed = TaskConfig.model_validate(raw)

    assert (
        parsed.observations.dataset_options["f322w2"].offset_parameter
        == "nircam_offset"
    )
    assert parsed.observations.dataset_options["lrs"].uncertainty_scale == 1.15

    local_observations = parsed.observations.model_copy(
        update={"path": ROOT / "data" / "wasp69b_schlawin2024"}
    )
    loaded = load_configured_observations(
        parsed.model_copy(update={"observations": local_observations})
    )
    assert loaded.datasets[0].offset_parameter == "nircam_offset"
    assert loaded.datasets[0].uncertainty_scale_parameter == "nircam_error_scale"
    assert loaded.datasets[1].offset_parameter == "nircam_offset"
    assert loaded.datasets[2].uncertainty_scale == 1.15
    assert loaded.datasets[2].jitter_parameter == "miri_jitter"


@pytest.mark.parametrize(
    ("temperature", "names"),
    [
        (
            {"model": "madhusudhan_seager_2009"},
            ("P1", "P2", "P3", "T0", "alpha1", "alpha2"),
        ),
        (
            {
                "model": "spline",
                "knot_pressure": [1e-6, 1e-2, 100.0],
                "parameter_names": ["T_top", "T_mid", "T_deep"],
                "extrapolation": "clip",
            },
            ("T_top", "T_mid", "T_deep"),
        ),
    ],
)
def test_yaml_supports_retrieved_madhu_and_spline_profiles(temperature, names) -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["temperature"] = temperature
    pg14_names = {"kappa_IR", "gamma1", "gamma2", "T_irr", "alpha"}
    raw["parameters"] = [
        item for item in raw["parameters"] if item["name"] not in pg14_names
    ]
    raw["parameters"] = [
        *raw["parameters"],
        *(
            {"name": name, "prior": {"type": "uniform", "lower": 0.01, "upper": 3000.0}}
            for name in names
        ),
    ]

    parsed = TaskConfig.model_validate(raw)

    assert parsed.atmosphere.temperature.model == temperature["model"]


def test_yaml_supports_free_chemistry() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"] = {
        "model": "free",
        "species": ["H2O", "CO2", "CO", "CH4", "NH3", "HCN"],
        "parameter_mode": "log10",
        "parameter_names": {
            "H2O": "log_H2O",
            "CO2": "log_CO2",
            "CO": "log_CO",
            "CH4": "log_CH4",
            "NH3": "log_NH3",
            "HCN": "log_HCN",
        },
        "background_species": ["H2", "He"],
        "background_fractions": [0.8547, 0.1453],
    }
    raw["parameters"] = [
        item
        for item in raw["parameters"]
        if item["name"] not in {"metallicity", "CtoO"}
    ]
    raw["parameters"] = [
        *raw["parameters"],
        *(
            {
                "name": f"log_{species}",
                "prior": {"type": "uniform", "lower": -12, "upper": -1},
            }
            for species in ("H2O", "CO2", "CO", "CH4", "NH3", "HCN")
        ),
    ]

    parsed = TaskConfig.model_validate(raw)

    assert parsed.atmosphere.chemistry.model == "free"
