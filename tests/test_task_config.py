"""Tests for the strict user-facing task configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from robert_exoplanets.io.configured_tasks import (
    load_observations as load_configured_observations,
    prepare_opacity,
)
from robert_exoplanets.io.task_config import (
    TaskConfig,
    initialize_task_directories,
    load_task_config,
)
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configurations" / "wasp69b_cloud_free_R1000.yaml"
TEMPLATE = ROOT / "configurations" / "TEMPLATE_all_supported_options.yaml"
TRANSMISSION = (
    ROOT / "configurations" / "synthetic_transmission_injection_recovery_multinest.yaml"
)
MULTISPECIES_TRANSMISSION = (
    ROOT
    / "configurations"
    / "synthetic_six_molecule_transmission_injection_recovery_multinest.yaml"
)
CLOUDY_MULTISPECIES_TRANSMISSION = (
    ROOT
    / "configurations"
    / "synthetic_six_molecule_cloudy_transmission_injection_recovery_multinest.yaml"
)
DEFAULTS = tuple(sorted((ROOT / "configurations").glob("wasp*.yaml")))


def test_wasp69b_example_exposes_complete_native_mode_run() -> None:
    config = load_task_config(EXAMPLE)

    assert config.schema_version == 2
    assert config.observations.datasets == ("f322w2", "f444w", "lrs")
    assert config.opacity.resolution == "R1000"
    assert config.opacity.species == ("H2O", "CO2", "CO", "CH4", "NH3", "SO2")
    assert config.atmosphere.chemistry.constant_log10_vmr_parameters == {
        "SO2": "log_SO2"
    }
    assert config.sampler.live_points == 400
    assert config.sampler.max_calls is None
    assert config.runtime.mpi_processes == "auto"
    assert config.runtime.scratch_directory.is_absolute()
    assert config.outputs.directory.is_absolute()
    assert config.plotting.enabled is False
    assert config.bodies.star.spectrum_model == "phoenix"
    assert config.bodies.star.log_g_cgs == 4.5
    assert config.bodies.star.metallicity_dex == 0.0


def test_yaml_can_select_blackbody_stellar_spectrum() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["bodies"]["star"]["spectrum_model"] = "blackbody"

    parsed = TaskConfig.model_validate(raw)

    assert parsed.bodies.star.spectrum_model == "blackbody"


def test_yaml_configures_transmission_and_real_exomol_h2o() -> None:
    config = load_task_config(TRANSMISSION)

    assert config.observations.loader == "robert_npz"
    assert config.opacity.format == "exomol_cross_section_hdf"
    assert config.opacity.binning.g_points == 8
    assert config.radiative_transfer.model == "transmission"
    assert config.radiative_transfer.reference_pressure_bar == 1.0
    assert config.radiative_transfer.radius_scale_parameter == "radius_scale"
    assert config.radiative_transfer.gravity_model == "inverse_square"
    assert config.sampler.engine == "multinest"
    assert config.sampler.live_points == 40
    assert config.runtime.mpi_processes == 2


def test_yaml_configures_six_molecule_transmission_recovery() -> None:
    config = load_task_config(MULTISPECIES_TRANSMISSION)

    expected_species = ("H2O", "CO", "CO2", "CH4", "NH3", "HCN")
    assert config.opacity.species == expected_species
    assert config.atmosphere.chemistry.species == expected_species
    assert tuple(config.atmosphere.chemistry.parameter_names) == expected_species
    assert tuple(item.name for item in config.parameters) == (
        "log_H2O",
        "log_CO",
        "log_CO2",
        "log_CH4",
        "log_NH3",
        "log_HCN",
        "radius_scale",
    )
    assert config.radiative_transfer.gas_combination == "random_overlap"
    assert config.sampler.live_points == 40
    assert config.runtime.mpi_processes == 2


def test_yaml_configures_cloudy_six_molecule_transmission_recovery() -> None:
    config = load_task_config(CLOUDY_MULTISPECIES_TRANSMISSION)

    assert config.clouds.model == "deck_haze"
    assert config.clouds.haze_slope_parameter == "haze_slope"
    assert tuple(item.name for item in config.parameters)[-4:] == (
        "log_cloud_top_pressure_bar",
        "log_cloud_optical_depth",
        "log_haze_mass_extinction",
        "haze_slope",
    )
    assert config.sampler.live_points == 50
    assert config.runtime.mpi_processes == 2


@pytest.mark.parametrize("source", [MULTISPECIES_TRANSMISSION, EXAMPLE])
def test_deck_haze_yaml_is_shared_by_transmission_and_emission(source: Path) -> None:
    config = load_task_config(source)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["clouds"] = {"model": "deck_haze"}
    raw["parameters"] = [
        *raw["parameters"],
        *(
            {
                "name": name,
                "prior": {"type": "uniform", "lower": lower, "upper": upper},
            }
            for name, lower, upper in (
                ("log_cloud_top_pressure_bar", -4.0, 1.0),
                ("log_cloud_optical_depth", -2.0, 3.0),
                ("log_haze_mass_extinction", -12.0, -2.0),
                ("haze_slope", -8.0, 2.0),
            )
        ),
    ]

    parsed = TaskConfig.model_validate(raw)

    assert parsed.clouds.model == "deck_haze"
    assert parsed.clouds.multiple_scattering_backend == "sh4"


def test_transmission_radius_parameter_must_be_in_retrieval_parameters() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["parameters"] = [
        item for item in raw["parameters"] if item["name"] != "radius_scale"
    ]

    with pytest.raises(ValidationError, match="radius_scale"):
        TaskConfig.model_validate(raw)


def test_real_cross_section_hdf_is_correlated_inside_observation_bins(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    config = load_task_config(TRANSMISSION)
    source = tmp_path / "source"
    source.mkdir()
    with h5py.File(source / "H2O.h5", "w") as handle:
        handle.create_dataset("p", data=[1.0e-5, 1.0])
        handle.create_dataset("t", data=[1000.0, 1200.0])
        handle.create_dataset("bin_edges", data=np.linspace(3000.0, 12000.0, 128))
        cross_sections = handle.create_dataset(
            "xsecarr",
            data=np.geomspace(1.0e-30, 1.0e-20, 2 * 2 * 128).reshape(2, 2, 128),
        )
        cross_sections.attrs["units"] = "cm^2/molecule"
        handle.create_dataset("DOI", data=[b"test-doi"])
        handle.create_dataset("key_iso_ll", data=[b"test-line-list"])
    cache = tmp_path / "cache"
    config = config.model_copy(
        update={
            "opacity": config.opacity.model_copy(
                update={"path": source, "cache_directory": cache}
            )
        }
    )
    observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        wavelength_bin_edges=[0.85, 1.3, 2.5],
        flux=[0.01, 0.01],
        uncertainty=[1.0e-5, 1.0e-5],
        flux_unit="transit_depth",
        observable="transit_depth",
    )
    observations = ObservationCollection(
        datasets=(
            ObservationDataset(name="synthetic_transit", observation=observation),
        )
    )

    prepare_opacity(config, observations)

    with np.load(cache / "R100" / "synthetic_transit_H2O.npz") as saved:
        assert saved["kcoeff"].shape == (2, 2, 2, 8)
        assert np.isclose(saved["g_weights"].sum(), 1.0)
        assert str(saved["source_line_list"]) == "test-line-list"
        assert str(saved["spectral_preparation"]) == (
            "exomol_cross_section_wavelength_weighted_k"
        )


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


def test_mie_catalog_configuration_is_valid_for_transmission() -> None:
    config = load_task_config(
        ROOT / "configurations" / "wasp69b_mie_catalog_pg14_R1000.yaml"
    )
    raw = deepcopy(config.model_dump(mode="python"))
    raw["radiative_transfer"]["model"] = "transmission"

    parsed = TaskConfig.model_validate(raw)

    assert parsed.radiative_transfer.model == "transmission"
    assert parsed.clouds.model == "mie_catalog"


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
    assert baseline.sampler.live_points == 400
    if engine != "optimal_estimation":
        assert benchmark.sampler.live_points == 400
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
    assert config.sampler.max_calls is None
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
        "species": ["H2O", "CO2", "CO", "CH4", "NH3", "SO2"],
        "parameter_mode": "log10",
        "parameter_names": {
            "H2O": "log_H2O",
            "CO2": "log_CO2",
            "CO": "log_CO",
            "CH4": "log_CH4",
            "NH3": "log_NH3",
            "SO2": "log_SO2",
        },
        "background_species": ["H2", "He"],
        "background_fractions": [0.8547, 0.1453],
    }
    raw["parameters"] = [
        item
        for item in raw["parameters"]
        if item["name"] not in {"metallicity", "CtoO", "log_SO2"}
    ]
    raw["parameters"] = [
        *raw["parameters"],
        *(
            {
                "name": f"log_{species}",
                "prior": {"type": "uniform", "lower": -12, "upper": -1},
            }
            for species in ("H2O", "CO2", "CO", "CH4", "NH3", "SO2")
        ),
    ]

    parsed = TaskConfig.model_validate(raw)

    assert parsed.atmosphere.chemistry.model == "free"
