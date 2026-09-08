"""Tests for the strict user-facing task configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from robert_exoplanets.io.configured_tasks import (
    _configured_science_sha256,
    _load_cached_table,
    _preparation_fingerprint,

    build_problem,
    load_observations as load_configured_observations,
    prepare_opacity,
)
from robert_exoplanets.io.task_config import (
    SamplerConfig,
    TaskConfig,
    configured_regions,
    initialize_task_directories,
    load_task_config,
)
from robert_exoplanets.instruments import (
    Observation,
    ObservationCollection,
    ObservationDataset,
)
from robert_exoplanets.core import RobertConfigError
from robert_exoplanets.retrieval.samplers.multinest import MULTINEST_MAX_SEED

from robert_exoplanets.opacity import CorrelatedKTable
from robert_exoplanets.retrieval.manifest import build_run_manifest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configurations" / "emission.yaml"
TEMPLATE = ROOT / "configurations" / "quickstart.yaml"
TRANSMISSION = ROOT / "configurations" / "transmission.yaml"
HDF_TRANSMISSION = (
    ROOT / "tests" / "fixtures" / "configurations" / "synthetic_transmission.yaml"
)
MULTISPECIES_TRANSMISSION = (
    ROOT / "tests" / "fixtures" / "configurations" / "six_gas_transmission.yaml"
)
CLOUDY_MULTISPECIES_TRANSMISSION = (
    ROOT
    / "tests"
    / "fixtures"
    / "configurations"
    / "six_gas_cloudy_transmission.yaml"
)
CONFIGURATIONS = ROOT / "configurations"
DEFAULTS = tuple(sorted(CONFIGURATIONS.glob("*.yaml")))
SHIPPED_CONFIGURATIONS = tuple(
    sorted(
        path
        for path in (
            *CONFIGURATIONS.glob("*.yaml"),
            *(ROOT / "tests" / "fixtures" / "configurations").glob("*.yaml"),
        )
        if "outputs" not in path.parts and "opacity_cache" not in path.parts
    )
)


def test_emission_example_exposes_complete_native_mode_run() -> None:
    config = load_task_config(EXAMPLE)

    assert config.schema_version == 2
    assert config.observations.datasets == ("f322w2", "f444w", "lrs")
    assert config.opacity.resolution == "R1000"
    assert config.opacity.path.resolve() == ROOT / "opacity_data" / "ktables_exomol"
    assert config.opacity.species == ("H2O", "CO2", "CO", "CH4", "NH3", "SO2")
    assert config.atmosphere.chemistry.constant_log10_vmr_parameters == {
        "SO2": "log_SO2"
    }
    assert config.sampler.live_points == 400
    assert config.radiative_transfer.sh4_boundary_backend == "auto"
    assert config.runtime.mpi_processes == "auto"
    assert config.runtime.scratch_directory.is_absolute()
    assert config.outputs.directory.is_absolute()
    assert config.plotting.enabled is True
    assert config.plotting.posterior_predictive_samples == 100
    assert config.bodies.star.spectrum_model == "phoenix"
    assert config.bodies.star.log_g_cgs == 4.5
    assert config.bodies.star.metallicity_dex == 0.0


@pytest.mark.parametrize("name", ["emission", "cloudy_emission", "optimal_estimation", "two_region_emission"])
def test_published_emission_examples_exclude_derived_overlap_average(name) -> None:
    config = load_task_config(CONFIGURATIONS / f"{name}.yaml")
    observations = load_configured_observations(config)
    assert tuple(dataset.name for dataset in observations.datasets) == (
        "f322w2", "f444w", "lrs"
    )


def test_yaml_rejects_direct_self_extension_with_actionable_message(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configuration.yaml"
    source.write_text("extends: configuration.yaml\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="configuration extends itself.*must not contain an extends entry",
    ):
        load_task_config(source)

def test_yaml_configures_arbitrary_molecular_pressure_quenching() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["quenching"] = {
        "model": "pressure_quench",
        "preset": "custom",
        "groups": [
            {
                "pressure_parameter": "log_Pq_CO2",
                "species": ["CO2"],
            },
            {
                "pressure_parameter": "log_Pq_water_carbon",
                "species": ["H2O", "CO"],
            },
        ],
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "log_Pq_CO2",
            "unit": "log10(bar)",
            "prior": {"type": "uniform", "lower": -5.5, "upper": 2.0},
        },
        {
            "name": "log_Pq_water_carbon",
            "unit": "log10(bar)",
            "prior": {"type": "uniform", "lower": -4.0, "upper": 1.0},
        },
    ]

    parsed = TaskConfig.model_validate(raw)

    quenching = parsed.atmosphere.chemistry.quenching
    assert quenching is not None
    assert quenching.preset == "custom"
    assert quenching.groups[0].species == ("CO2",)
    assert quenching.groups[1].species == ("H2O", "CO")


def test_yaml_configures_taylor_2026_grouped_quench_preset() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["quenching"] = {
        "model": "pressure_quench",
        "preset": "taylor_2026_hot_jupiter_element_grouped",
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "log_Pq_C",
            "prior": {"type": "uniform", "lower": -5.5, "upper": 2.0},
        },
        {
            "name": "log_Pq_N",
            "prior": {"type": "uniform", "lower": -5.5, "upper": 2.0},
        },
    ]

    parsed = TaskConfig.model_validate(raw)
    groups = parsed.atmosphere.chemistry.quenching.resolved_groups()

    assert groups[0].pressure_parameter == "log_Pq_C"
    assert groups[0].species == ("H2O", "CO", "CO2", "CH4")
    assert groups[1].species == ("NH3",)
    assert all("N2" not in group.species for group in groups)


@pytest.mark.parametrize(
    ("quenching", "message"),
    [
        (
            {
                "model": "pressure_quench",
                "groups": [
                    {"pressure_parameter": "log Pq", "species": ["CO2"]}
                ],
            },
            "pressure_parameter",
        ),
        (
            {
                "model": "pressure_quench",
                "groups": [
                    {"pressure_parameter": "log_Pq_1", "species": ["CO2"]},
                    {"pressure_parameter": "log_Pq_2", "species": ["CO2"]},
                ],
            },
            "only one quench group",
        ),
        (
            {
                "model": "pressure_quench",
                "groups": [
                    {"pressure_parameter": "log_Pq_X", "species": ["C2H2"]}
                ],
            },
            "missing from FastChem labels",
        ),
        (
            {
                "model": "pressure_quench",
                "groups": [
                    {"pressure_parameter": "metallicity", "species": ["CO2"]}
                ],
            },
            "collide with FastChem parameters",
        ),
        (
            {
                "model": "pressure_quench",
                "groups": [
                    {
                        "pressure_parameter": "log_Pq_CO2",
                        "species": ["CO2"],
                        "interpolation": "nearest",
                    }
                ],
            },
            "Extra inputs are not permitted",
        ),
    ],
)
def test_yaml_rejects_malformed_pressure_quenching(quenching, message) -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["quenching"] = quenching

    with pytest.raises(ValidationError, match=message):
        TaskConfig.model_validate(raw)


def test_yaml_requires_every_quench_pressure_prior() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["quenching"] = {
        "model": "pressure_quench",
        "groups": [
            {"pressure_parameter": "log_Pq_CO2", "species": ["CO2"]}
        ],
    }

    with pytest.raises(ValidationError, match="log_Pq_CO2"):
        TaskConfig.model_validate(raw)


def test_yaml_rejects_quenching_for_constant_with_altitude_free_chemistry() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["quenching"] = {
        "model": "pressure_quench",
        "groups": [
            {"pressure_parameter": "log_Pq_H2O", "species": ["H2O"]}
        ],
    }

    with pytest.raises(ValidationError, match="quenching"):
        TaskConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        (
            {"type": "log_uniform", "lower": 0.01, "upper": 1.0},
            "already logarithmic",
        ),
        (
            {"type": "uniform", "lower": -7.0, "upper": 1.0},
            "pressure grid layer-center domain",
        ),
    ],
)
def test_yaml_validates_quench_prior_semantics_and_grid_domain(prior, message) -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["quenching"] = {
        "model": "pressure_quench",
        "groups": [
            {"pressure_parameter": "log_Pq_CO2", "species": ["CO2"]}
        ],
    }
    raw["parameters"] = [
        *raw["parameters"],
        {"name": "log_Pq_CO2", "prior": prior},
    ]

    with pytest.raises(ValidationError, match=message):
        TaskConfig.model_validate(raw)


def test_configured_fastchem_is_wrapped_by_pressure_quench_decorator(
    monkeypatch,
) -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["bodies"]["star"]["spectrum_model"] = "blackbody"
    raw["atmosphere"]["chemistry"]["quenching"] = {
        "model": "pressure_quench",
        "groups": [
            {"pressure_parameter": "log_Pq_CO2", "species": ["CO2"]}
        ],
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "log_Pq_CO2",
            "prior": {"type": "uniform", "lower": -5.5, "upper": 2.0},
        },
    ]
    parsed = TaskConfig.model_validate(raw)
    observation = Observation.from_arrays(
        wavelength=[3.0, 4.0],
        wavelength_bin_edges=[2.5, 3.5, 4.5],
        flux=[1.0e-3, 1.0e-3],
        uncertainty=[1.0e-5, 1.0e-5],
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
    )
    observations = ObservationCollection(
        datasets=(ObservationDataset(name="nircam", observation=observation),)
    )

    def fake_table(_config, _dataset_name, species):
        return CorrelatedKTable(
            species=species,
            pressure_bar=np.array([1.0e-6, 100.0]),
            temperature_K=np.array([500.0, 2500.0]),
            wavenumber_cm_inverse=10000.0 / observation.spectral_grid.values,
            g_samples=np.array([0.5]),
            g_weights=np.array([1.0]),
            kcoeff=np.full((2, 2, 2, 1), 1.0e-24),
            metadata={"checksum_sha256": f"configured-{species}"},
        )

    monkeypatch.setattr(
        "robert_exoplanets.io.configured_tasks._load_cached_table",
        fake_table,
    )
    monkeypatch.setattr(
        "robert_exoplanets.io.configured_tasks.load_nemesispy_cia_table",
        lambda: None,
    )

    problem = build_problem(parsed, observations)
    manifest = build_run_manifest(
        problem,
        method="unit-test",
        settings={},
        random_seed=7,
    )

    assert "log_Pq_CO2" in problem.parameter_names
    assert problem.metadata["chemistry_quench_scheme"] == "pressure_quench"
    assert problem.metadata["chemistry_quench_groups"] == "log_Pq_CO2:CO2"
    assert problem.metadata["chemistry_quench_pressure_semantics"] == "log10(P_q/bar)"
    assert manifest.problem_metadata["chemistry_quench_scheme"] == "pressure_quench"
    assert manifest.problem_metadata["chemistry_quench_closure_policy"] == (
        "no_renormalization"
    )


def test_yaml_can_select_blackbody_stellar_spectrum() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["bodies"]["star"]["spectrum_model"] = "blackbody"

    parsed = TaskConfig.model_validate(raw)

    assert parsed.bodies.star.spectrum_model == "blackbody"


def _two_region_raw() -> dict:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    cold_species = list(raw["opacity"]["species"])
    raw["disk_emission"] = {
        "model": "two_region",
        "hot_fraction_parameter": "hot_area_fraction",
        "cold_region": {
            "atmosphere": {
                "temperature": {"model": "isothermal", "temperature_k": 900.0},
                "chemistry": {
                    "model": "free",
                    "species": cold_species,
                    "fixed_mixing_ratios": {
                        species: 1.0e-8 for species in cold_species
                    },
                    "background_species": ["H2", "He"],
                },
            },
            "clouds": {
                "model": "deck_haze",
                "log10_cloud_top_pressure_bar_parameter": "cold_cloud_top",
                "log10_cloud_optical_depth_parameter": "cold_cloud_tau",
                "log10_haze_mass_extinction_parameter": "cold_haze_extinction",
                "haze_slope_parameter": "cold_haze_slope",
            },
        },
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "hot_area_fraction",
            "prior": {"type": "uniform", "lower": 0.0, "upper": 1.0},
        },
        *(
            {
                "name": name,
                "prior": {"type": "uniform", "lower": -12.0, "upper": 4.0},
            }
            for name in (
                "cold_cloud_top",
                "cold_cloud_tau",
                "cold_haze_extinction",
                "cold_haze_slope",
            )
        ),
    ]
    return raw


def test_yaml_resolves_independent_two_region_physics() -> None:
    parsed = TaskConfig.model_validate(_two_region_raw())

    hot, cold = configured_regions(parsed)

    assert parsed.disk_emission.model == "two_region"
    assert hot.atmosphere.temperature.model == "parmentier_guillot_2014"
    assert hot.atmosphere.chemistry.model == "fastchem_equilibrium"
    assert hot.clouds.model == "none"
    assert cold.atmosphere.pressure is parsed.atmosphere.pressure
    assert cold.atmosphere.temperature.model == "isothermal"
    assert cold.atmosphere.chemistry.model == "free"
    assert cold.clouds.model == "deck_haze"


def test_two_region_yaml_requires_the_area_fraction_parameter() -> None:
    raw = _two_region_raw()
    raw["parameters"] = [
        item for item in raw["parameters"] if item["name"] != "hot_area_fraction"
    ]

    with pytest.raises(ValidationError, match="hot_area_fraction"):
        TaskConfig.model_validate(raw)


def test_yaml_accepts_diluted_emission_and_rejects_it_for_transmission() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["disk_emission"] = {
        "model": "diluted_one_region",
        "dilution_parameter": "dayside_dilution",
    }
    raw["parameters"] = (
        *raw["parameters"],
        {
            "name": "dayside_dilution",
            "prior": {"type": "uniform", "lower": 0.0, "upper": 1.0},
        },
    )

    parsed = TaskConfig.model_validate(raw)
    assert parsed.disk_emission.model == "diluted_one_region"

    raw["radiative_transfer"]["model"] = "transmission"
    with pytest.raises(ValidationError, match="require.*emission"):
        TaskConfig.model_validate(raw)


def test_yaml_defaults_writable_paths_to_the_configuration_directory(
    tmp_path: Path,
) -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="json"))
    raw.pop("paths", None)
    raw.pop("outputs")
    raw["runtime"].pop("scratch_directory")
    raw["opacity"].pop("cache_directory")
    source = tmp_path / "configuration.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    parsed = load_task_config(source)

    assert parsed.outputs.directory == tmp_path / "outputs"
    assert parsed.runtime.scratch_directory == tmp_path / "scratch"
    assert parsed.opacity.cache_directory == tmp_path / "opacity_cache"


def test_sampler_defaults_to_multinest() -> None:
    assert SamplerConfig().engine == "multinest"


def test_opacity_defaults_to_r1000_but_preserves_explicit_resolution() -> None:
    raw = load_task_config(EXAMPLE).model_dump(mode="python")
    raw["opacity"].pop("resolution")
    assert TaskConfig.model_validate(raw).opacity.resolution == "R1000"
    raw["opacity"]["resolution"] = "R100"
    assert TaskConfig.model_validate(raw).opacity.resolution == "R100"


def test_resume_science_identity_excludes_budget_but_includes_fixed_physics() -> None:
    config = load_task_config(EXAMPLE)
    identity = _configured_science_sha256(config)
    changed_budget = config.model_copy(update={"sampler": config.sampler.model_copy(
        update={"dlogz": 0.1, "multinest_max_iterations": 2000}
    )})
    assert _configured_science_sha256(changed_budget) == identity
    changed_star = config.model_copy(update={"bodies": config.bodies.model_copy(
        update={"star": config.bodies.star.model_copy(
            update={"effective_temperature_k": 4800.0}
        )}
    )})
    assert _configured_science_sha256(changed_star) != identity


def test_yaml_configures_transmission_and_real_exomol_h2o() -> None:
    config = load_task_config(HDF_TRANSMISSION)

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



def test_yaml_configures_poseidon_rackham_stellar_contamination() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["stellar_contamination"] = {
        "model": "poseidon_rackham",
        "regions": [
            {
                "name": "cool_spot",
                "kind": "spot",
                "temperature_k": 4800.0,
                "covering_fraction_parameter": "f_spot",
            },
            {
                "name": "hot_facula",
                "kind": "facula",
                "temperature_k": 5700.0,
                "covering_fraction": 0.1,
            },
        ],
        "transit_chord_temperature_k": 5250.0,
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "f_spot",
            "prior": {"type": "uniform", "lower": 0.0, "upper": 0.5},
        },
    ]

    parsed = TaskConfig.model_validate(raw)
    round_trip = TaskConfig.model_validate(parsed.model_dump(mode="python"))

    assert parsed.stellar_contamination is not None
    assert parsed.stellar_contamination.regions[0].kind == "spot"
    assert parsed.stellar_contamination.regions[1].covering_fraction == 0.1
    assert parsed.stellar_contamination.transit_chord_temperature_k == 5250.0
    assert round_trip == parsed


def test_stellar_contamination_requires_valid_temperature_and_fraction_priors() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["stellar_contamination"] = {
        "regions": [
            {
                "name": "cool_spot",
                "kind": "spot",
                "temperature_k": 4800.0,
                "covering_fraction_parameter": "f_spot",
            }
        ]
    }
    with pytest.raises(ValidationError, match="f_spot"):
        TaskConfig.model_validate(raw)

    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "f_spot",
            "prior": {"type": "uniform", "lower": -0.1, "upper": 0.5},
        },
    ]
    with pytest.raises(ValidationError, match=r"within \[0, 1\]"):
        TaskConfig.model_validate(raw)

    raw["parameters"][-1]["prior"] = {
        "type": "uniform",
        "lower": 0.0,
        "upper": 0.5,
    }
    raw["stellar_contamination"]["regions"][0]["temperature_k"] = 6000.0
    with pytest.raises(ValidationError, match="cooler than the photosphere"):
        TaskConfig.model_validate(raw)


def test_stellar_contamination_mixture_prior_must_close() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["stellar_contamination"] = {
        "regions": [
            {
                "name": "spot",
                "kind": "spot",
                "temperature_k": 4800.0,
                "covering_fraction_parameter": "f_spot",
            },
            {
                "name": "facula",
                "kind": "facula",
                "temperature_k": 5700.0,
                "covering_fraction_parameter": "f_fac",
            },
        ]
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "f_spot",
            "prior": {"type": "uniform", "lower": 0.0, "upper": 0.7},
        },
        {
            "name": "f_fac",
            "prior": {"type": "uniform", "lower": 0.0, "upper": 0.4},
        },
    ]

    with pytest.raises(ValidationError, match="sum to at most one"):
        TaskConfig.model_validate(raw)


def test_stellar_contamination_is_rejected_for_emission() -> None:
    config = load_task_config(EXAMPLE)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["stellar_contamination"] = {"regions": []}

    with pytest.raises(ValidationError, match="only supported for transmission"):
        TaskConfig.model_validate(raw)


def test_configured_multi_dataset_tsle_parameter_changes_each_spectrum(
    monkeypatch,
) -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["bodies"]["star"]["spectrum_model"] = "blackbody"
    raw["stellar_contamination"] = {
        "regions": [
            {
                "name": "spot",
                "kind": "spot",
                "temperature_k": 4800.0,
                "covering_fraction_parameter": "f_spot",
            }
        ]
    }
    raw["parameters"] = [
        *raw["parameters"],
        {
            "name": "f_spot",
            "prior": {"type": "uniform", "lower": 0.0, "upper": 0.3},
        },
    ]
    parsed = TaskConfig.model_validate(raw)
    observations = ObservationCollection(
        datasets=(
            ObservationDataset(
                name="blue",
                observation=Observation.from_arrays(
                    [1.0, 1.5],
                    [0.01, 0.01],
                    [1.0e-4, 1.0e-4],
                    wavelength_bin_edges=[0.9, 1.2, 1.8],
                    flux_unit="transit_depth",
                    observable="transit_depth",
                ),
            ),
            ObservationDataset(
                name="red",
                observation=Observation.from_arrays(
                    [3.0, 4.0],
                    [0.01, 0.01],
                    [1.0e-4, 1.0e-4],
                    wavelength_bin_edges=[2.5, 3.5, 4.5],
                    flux_unit="transit_depth",
                    observable="transit_depth",
                ),
            ),
        )
    )

    def fake_table(_config, dataset, species):
        dataset_name = dataset.name
        wavelength = (
            np.array([1.0, 1.5])
            if dataset_name == "blue"
            else np.array([3.0, 4.0])
        )
        return CorrelatedKTable(
            species=species,
            pressure_bar=np.array([1.0e-6, 100.0]),
            temperature_K=np.array([500.0, 2000.0]),
            wavenumber_cm_inverse=10000.0 / wavelength,
            g_samples=np.array([0.5]),
            g_weights=np.array([1.0]),
            kcoeff=np.full((2, 2, 2, 1), 1.0e-24),
            metadata={"checksum_sha256": f"{dataset_name}-{species}"},
        )

    monkeypatch.setattr(
        "robert_exoplanets.io.configured_tasks._load_cached_table",
        fake_table,
    )
    monkeypatch.setattr(
        "robert_exoplanets.io.configured_tasks.load_nemesispy_cia_table",
        lambda: None,
    )
    problem = build_problem(parsed, observations)
    baseline = problem.parameters.vector_to_mapping(problem.parameters.midpoint_vector())
    baseline["f_spot"] = 0.0
    spotted = {**baseline, "f_spot": 0.2}

    homogeneous_spectra = problem.model_spectra(baseline)
    spotted_spectra = problem.model_spectra(spotted)

    assert set(spotted_spectra) == {"blue", "red"}
    assert all(
        np.all(spotted_spectra[name].values > homogeneous_spectra[name].values)
        for name in spotted_spectra
    )
    assert problem.metadata["stellar_contamination"] == "enabled"
    assert problem.metadata["stellar_contamination_required_parameters"] == "f_spot"
    assert problem.opacity_identifiers == {
        "blue:H2O": "blue-H2O",
        "red:H2O": "red-H2O",
    }



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
    config = load_task_config(HDF_TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["parameters"] = [
        item for item in raw["parameters"] if item["name"] != "radius_scale"
    ]

    with pytest.raises(ValidationError, match="radius_scale"):
        TaskConfig.model_validate(raw)


def test_yaml_configures_joint_centered_log_ratio_chemistry() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["background_species"] = ["H2"]
    raw["atmosphere"]["chemistry"]["background_fractions"] = [1.0]
    for parameter in raw["parameters"]:
        if parameter["name"] == "log_H2O":
            parameter["prior"] = {
                "type": "centered_log_ratio",
                "lower": -12.0,
                "upper": 0.0,
                "group": "atmosphere",
            }

    parsed = TaskConfig.model_validate(raw)

    prior = parsed.parameters[0].prior
    assert prior.type == "centered_log_ratio"
    assert prior.group == "atmosphere"
    assert parsed.atmosphere.chemistry.background_species == ("H2",)


def test_yaml_configures_phantom_background_and_fitted_molecular_weight() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    chemistry = raw["atmosphere"]["chemistry"]
    chemistry["background_species"] = ["phantom"]
    chemistry["background_fractions"] = [1.0]
    chemistry["phantom_species"] = "phantom"
    chemistry["phantom_mean_molecular_weight_parameter"] = "phantom_mmw"
    for parameter in raw["parameters"]:
        if parameter["name"] == "log_H2O":
            parameter["prior"] = {
                "type": "centered_log_ratio",
                "lower": -12.0,
                "upper": 0.0,
                "group": "atmosphere",
            }
    raw["parameters"] = (
        *raw["parameters"],
        {
            "name": "phantom_mmw",
            "prior": {"type": "uniform", "lower": 2.3, "upper": 100.0},
        },
    )

    parsed = TaskConfig.model_validate(raw)

    assert parsed.atmosphere.chemistry.phantom_species == "phantom"
    assert (
        parsed.atmosphere.chemistry.phantom_mean_molecular_weight_parameter
        == "phantom_mmw"
    )


def test_clr_and_phantom_configuration_is_geometry_agnostic() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    chemistry = raw["atmosphere"]["chemistry"]
    chemistry["background_species"] = ["phantom"]
    chemistry["background_fractions"] = [1.0]
    chemistry["phantom_species"] = "phantom"
    chemistry["phantom_mean_molecular_weight_parameter"] = "phantom_mmw"
    raw["parameters"] = [
        parameter
        for parameter in raw["parameters"]
        if parameter["name"] != "radius_scale"
    ]
    raw["parameters"][0]["prior"] = {
        "type": "centered_log_ratio",
        "lower": -12.0,
        "upper": 0.0,
    }
    raw["parameters"].append(
        {
            "name": "phantom_mmw",
            "prior": {"type": "uniform", "lower": 2.3, "upper": 100.0},
        }
    )
    raw["radiative_transfer"]["model"] = "emission"
    raw["radiative_transfer"]["radius_scale_parameter"] = None

    parsed = TaskConfig.model_validate(raw)

    assert parsed.radiative_transfer.model == "emission"
    assert parsed.atmosphere.chemistry.phantom_species == "phantom"
    assert parsed.parameters[0].prior.type == "centered_log_ratio"


def test_yaml_rejects_optimal_estimation_for_centered_log_ratio_prior() -> None:
    config = load_task_config(TRANSMISSION)
    raw = deepcopy(config.model_dump(mode="python"))
    raw["atmosphere"]["chemistry"]["background_species"] = ["H2"]
    raw["atmosphere"]["chemistry"]["background_fractions"] = [1.0]
    raw["parameters"][0]["prior"] = {
        "type": "centered_log_ratio",
        "lower": -12.0,
        "upper": 0.0,
    }
    raw["sampler"]["engine"] = "optimal_estimation"

    with pytest.raises(ValidationError, match="direct nested sampling"):
        TaskConfig.model_validate(raw)


def _cross_section_cache_fixture(
    tmp_path: Path,
) -> tuple[TaskConfig, ObservationCollection, Path]:
    h5py = pytest.importorskip("h5py")
    config = load_task_config(HDF_TRANSMISSION)
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
    return config, observations, cache


def test_real_cross_section_hdf_is_correlated_inside_observation_bins(
    tmp_path: Path,
) -> None:
    config, observations, cache = _cross_section_cache_fixture(tmp_path)

    prepare_opacity(config, observations)

    with np.load(cache / "R100" / "synthetic_transit_H2O.npz") as saved:
        assert saved["kcoeff"].shape == (2, 2, 2, 8)
        assert np.isclose(saved["g_weights"].sum(), 1.0)
        assert str(saved["source_line_list"]) == "test-line-list"
        assert int(saved["opacity_cache_schema_version"]) == 2
        assert str(saved["preparation_fingerprint"]) == _preparation_fingerprint(
            config, observations.datasets[0], "H2O"
        )
        assert str(saved["spectral_preparation"]) == (
            "exomol_cross_section_wavelength_weighted_k"
        )


def test_opacity_cache_fingerprint_covers_grid_and_binning_settings(
    tmp_path: Path,
) -> None:
    config, observations, cache = _cross_section_cache_fixture(tmp_path)
    prepare_opacity(config, observations)
    path = cache / "R100" / "synthetic_transit_H2O.npz"
    with np.load(path, allow_pickle=False) as saved:
        original_fingerprint = str(saved["preparation_fingerprint"])

    changed_source_config = config.model_copy(
        update={
            "opacity": config.opacity.model_copy(
                update={"path": tmp_path / "different-source"}
            )
        }
    )
    assert _preparation_fingerprint(
        changed_source_config, observations.datasets[0], "H2O"
    ) != original_fingerprint

    changed_config = config.model_copy(
        update={
            "opacity": config.opacity.model_copy(
                update={
                    "binning": config.opacity.binning.model_copy(
                        update={"g_points": 4, "remove_zeros": False}
                    )
                }
            )
        }
    )
    changed_observation = Observation.from_arrays(
        wavelength=[1.0, 2.0],
        wavelength_bin_edges=[0.80, 1.35, 2.5],
        flux=[0.01, 0.01],
        uncertainty=[1.0e-5, 1.0e-5],
        flux_unit="transit_depth",
        observable="transit_depth",
    )
    changed_dataset = ObservationDataset(
        name="synthetic_transit", observation=changed_observation
    )
    changed_observations = ObservationCollection(datasets=(changed_dataset,))

    with pytest.raises(RobertConfigError, match="stale"):
        _load_cached_table(config, changed_dataset, "H2O")
    prepare_opacity(changed_config, changed_observations)

    with np.load(path, allow_pickle=False) as saved:
        assert str(saved["preparation_fingerprint"]) != original_fingerprint
        assert saved["g_samples"].size == 4
        assert bool(saved["binning_remove_zeros"]) is False


def test_opacity_cache_without_current_schema_requires_repreparation(
    tmp_path: Path,
) -> None:
    config, observations, cache = _cross_section_cache_fixture(tmp_path)
    prepare_opacity(config, observations)
    path = cache / "R100" / "synthetic_transit_H2O.npz"
    with np.load(path, allow_pickle=False) as saved:
        legacy = {
            name: saved[name]
            for name in saved.files
            if name != "opacity_cache_schema_version"
        }
    np.savez_compressed(path, **legacy)

    with pytest.raises(RobertConfigError, match="stale|unsupported schema"):
        _load_cached_table(config, observations.datasets[0], "H2O")


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


def test_all_public_configurations_resolve_and_validate() -> None:
    assert len(DEFAULTS) == 7
    resolutions = []
    for path in DEFAULTS:
        config = load_task_config(path)
        resolutions.append(config.opacity.resolution)
    assert set(resolutions) == {"R100", "R1000"}


def test_all_shipped_multinest_seeds_fit_legacy_fortran_range() -> None:
    for path in SHIPPED_CONFIGURATIONS:
        config = load_task_config(path)
        if "multinest" not in config.sampler.engine or config.sampler.seed is None:
            continue
        assert config.sampler.seed <= MULTINEST_MAX_SEED, path


def test_mie_catalog_configuration_is_valid_for_transmission() -> None:
    config = load_task_config(ROOT / "configurations" / "cloudy_emission.yaml")
    raw = deepcopy(config.model_dump(mode="python"))
    raw["radiative_transfer"]["model"] = "transmission"

    parsed = TaskConfig.model_validate(raw)

    assert parsed.radiative_transfer.model == "transmission"
    assert parsed.clouds.model == "mie_catalog"


def test_optimal_estimation_keeps_emission_science_configuration() -> None:
    baseline = load_task_config(ROOT / "configurations" / "emission.yaml")
    benchmark = load_task_config(ROOT / "configurations" / "optimal_estimation.yaml")

    assert baseline.sampler.engine == "multinest"
    assert benchmark.sampler.engine == "optimal_estimation"
    assert benchmark.sampler.oe_max_iterations == 8
    for section in (
        "bodies",
        "observations",
        "atmosphere",
        "clouds",
        "disk_emission",
        "radiative_transfer",
        "likelihood",
        "parameters",
    ):
        assert getattr(benchmark, section) == getattr(baseline, section)
    assert benchmark.opacity.format == baseline.opacity.format
    assert benchmark.opacity.resolution == baseline.opacity.resolution
    assert benchmark.opacity.species == baseline.opacity.species
    assert benchmark.opacity.binning == baseline.opacity.binning


def test_public_quickstart_uses_one_top_level_path_block() -> None:
    config = load_task_config(TEMPLATE)

    assert config.clouds.model == "none"
    assert config.paths is not None
    assert config.observations.path == config.paths.observations_directory
    assert config.atmosphere.chemistry.model == "free"
    assert config.paths.fastchem_directory is None
    assert config.paths.k_table_directory is None
    assert config.opacity.path.is_dir()
    assert config.opacity.cache_directory.resolve() == ROOT / "examples" / "outputs" / "r100_validation" / "emission" / "opacity_cache"
    assert config.outputs.directory.resolve() == ROOT / "examples" / "outputs" / "r100_validation" / "emission"
    assert config.runtime.scratch_directory.resolve() == ROOT / "examples" / "outputs" / "r100_validation" / "emission" / "scratch"
    assert config.plotting.enabled is True
    assert config.plotting.dataset_colors["synthetic_emission"] == "mediumpurple"


def test_public_configurations_have_distinct_writable_output_roots() -> None:
    output_roots = {
        load_task_config(path).outputs.directory.resolve() for path in DEFAULTS
    }

    assert len(output_roots) == len(DEFAULTS)


def test_public_rocky_clr_configuration_keeps_l9859b_data_matched_to_planet() -> None:
    config = load_task_config(ROOT / "configurations" / "rocky_transmission_clr.yaml")

    assert config.bodies.planet.name == "L 98-59 b"
    assert config.observations.loader == "bello_arufe2025_l9859b"
    assert config.observations.path.resolve() == (
        ROOT / "data" / "observations" / "l98_59b_bello_arufe2025"
    )
    assert config.opacity.resolution == "R1000"
    assert config.sampler.engine == "multinest"
    assert all(
        parameter.prior.type == "centered_log_ratio"
        for parameter in config.parameters[:3]
    )


def test_public_transmission_configuration_is_a_bundled_r100_free_chemistry_check() -> None:
    config = load_task_config(ROOT / "configurations" / "transmission.yaml")

    assert config.opacity.resolution == "R100"
    assert config.opacity.format == "exomol_kta"
    assert config.atmosphere.chemistry.model == "free"
    assert config.radiative_transfer.model == "transmission"


def test_legacy_housekeeping_path_block_is_rejected() -> None:
    raw = load_task_config(TEMPLATE).model_dump(mode="python")
    raw["housekeeping"] = raw.pop("paths")

    with pytest.raises(ValidationError, match="housekeeping"):
        TaskConfig.model_validate(raw)


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
