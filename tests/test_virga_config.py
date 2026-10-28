"""Tests for the fixed-control Virga task configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robert_exoplanets.io import configured_tasks
from robert_exoplanets.io.task_config import (
    VirgaEddySedCloudConfig,
    configured_regions,
    load_task_config,
)


ROOT = Path(__file__).resolve().parents[1]
EMISSION = ROOT / "configurations" / "emission.yaml"
TWO_REGION_EMISSION = ROOT / "configurations" / "two_region_emission.yaml"
TRANSMISSION = ROOT / "configurations" / "transmission.yaml"


def _virga_cloud(mie_directory: Path | None = None) -> dict:
    cloud = {
        "model": "virga_eddy_sed",
        "condensibles": ["KCl"],
        "gas_mass_mixing_ratios": {"KCl": 1.0e-4},
        "fsed": 2.0,
        "kz_cm2_s": 1.0e9,
        "metallicity": 1.0,
        "geometric_stddev": 2.0,
        "supersaturation": 0.0,
        "multiple_scattering_backend": "sh4",
    }
    if mie_directory is not None:
        cloud["mie_directory"] = str(mie_directory)
    return cloud


def _write_config(
    tmp_path: Path,
    base_path: Path,
    *,
    cloud: dict,
    virga_mie_directory: Path | None = None,
    gravity_model: str = "constant",
    extra_parameters: list[dict] | None = None,
) -> Path:
    raw = load_task_config(base_path).model_dump(mode="json")
    raw["clouds"] = cloud
    raw["radiative_transfer"]["gravity_model"] = gravity_model
    if extra_parameters:
        raw["parameters"].extend(extra_parameters)
    if virga_mie_directory is not None:
        raw.setdefault("paths", {})["virga_mie_directory"] = str(virga_mie_directory)
    source = tmp_path / "task.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return source


def _make_mie_table(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "KCl.mieff").write_text("fixture\n", encoding="utf-8")
    return directory


def _retrieval_parameter(
    name: str,
    lower: float,
    upper: float,
    *,
    unit: str | None = None,
    prior_type: str = "uniform",
) -> dict:
    parameter = {
        "name": name,
        "prior": {"type": prior_type, "lower": lower, "upper": upper},
    }
    if unit is not None:
        parameter["unit"] = unit
    return parameter


def test_load_and_factory_build_fixed_virga_cloud_from_yaml(tmp_path: Path) -> None:
    mie_directory = _make_mie_table(tmp_path / "mie")
    source = _write_config(
        tmp_path,
        EMISSION,
        cloud=_virga_cloud(mie_directory),
    )

    config = load_task_config(source)
    cloud = configured_tasks._cloud_model(config.clouds)

    assert isinstance(config.clouds, VirgaEddySedCloudConfig)
    assert config.clouds.mie_directory == mie_directory
    assert isinstance(cloud, configured_tasks.VirgaEddySedCloudModel)
    assert cloud.fsed == 2.0
    assert cloud.kz_cm2_s == 1.0e9
    assert cloud.gas_mass_mixing_ratios == {"KCl": 1.0e-4}
    assert cloud.required_parameters == ()
    assert cloud.manifest_metadata["cloud_provider"] == "virga-exo"


def test_global_virga_mie_path_fills_top_level_and_regional_clouds(
    tmp_path: Path,
) -> None:
    mie_directory = _make_mie_table(tmp_path / "shared-mie")
    top_level_source = _write_config(
        tmp_path,
        EMISSION,
        cloud=_virga_cloud(),
        virga_mie_directory=mie_directory,
    )

    top_level_config = load_task_config(top_level_source)

    assert isinstance(top_level_config.clouds, VirgaEddySedCloudConfig)
    assert top_level_config.clouds.mie_directory == mie_directory

    raw = load_task_config(TWO_REGION_EMISSION).model_dump(mode="json")
    raw.setdefault("paths", {})["virga_mie_directory"] = str(mie_directory)
    raw["disk_emission"]["hot_region"]["clouds"] = _virga_cloud()
    source = tmp_path / "two-region.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    regional_config = load_task_config(source)
    regions = {region.name: region for region in configured_regions(regional_config)}

    assert regions["hot"].clouds.mie_directory == mie_directory
    assert regions["cold"].clouds.model == "mie_catalog"


def test_virga_configures_retrieved_fsed_and_log10_kzz(tmp_path: Path) -> None:
    mie_directory = _make_mie_table(tmp_path / "mie")
    cloud = _virga_cloud(mie_directory)
    cloud.pop("fsed")
    cloud.pop("kz_cm2_s")
    cloud["fsed_parameter"] = "cloud_fsed"
    cloud["log10_kz_cm2_s_parameter"] = "cloud_log10_kzz"
    source = _write_config(
        tmp_path,
        EMISSION,
        cloud=cloud,
        extra_parameters=[
            _retrieval_parameter("cloud_fsed", 0.1, 10.0, unit="dimensionless"),
            _retrieval_parameter(
                "cloud_log10_kzz",
                4.0,
                12.0,
                unit="log10(cm^2/s)",
            ),
        ],
    )

    config = load_task_config(source)
    cloud_model = configured_tasks._cloud_model(config.clouds)

    assert isinstance(config.clouds, VirgaEddySedCloudConfig)
    assert cloud_model.required_parameters == ("cloud_fsed", "cloud_log10_kzz")
    assert cloud_model.fsed is None
    assert cloud_model.kz_cm2_s is None
    assert cloud_model.manifest_metadata["cloud_fsed_parameter"] == "cloud_fsed"
    assert (
        cloud_model.manifest_metadata["cloud_log10_kz_cm2_s_parameter"]
        == "cloud_log10_kzz"
    )


def test_task_config_requires_a_prior_for_each_retrieved_virga_control(
    tmp_path: Path,
) -> None:
    cloud = _virga_cloud(_make_mie_table(tmp_path / "mie"))
    cloud.pop("fsed")
    cloud["fsed_parameter"] = "cloud_fsed"
    source = _write_config(tmp_path, EMISSION, cloud=cloud)

    with pytest.raises(
        ValidationError, match="required model parameters are missing: cloud_fsed"
    ):
        load_task_config(source)


@pytest.mark.parametrize(
    ("control", "updates", "message"),
    [
        ("fsed", {"fsed": None, "fsed_parameter": None}, "fsed"),
        ("fsed", {"fsed_parameter": "cloud_fsed"}, "fsed"),
        ("fsed", {"fsed": None, "fsed_parameter": "   "}, "fsed"),
        ("Kzz", {"kz_cm2_s": None, "log10_kz_cm2_s_parameter": None}, "Kzz"),
        ("Kzz", {"log10_kz_cm2_s_parameter": "cloud_log10_kzz"}, "Kzz"),
        (
            "Kzz",
            {"kz_cm2_s": None, "log10_kz_cm2_s_parameter": "   "},
            "non-empty name",
        ),
    ],
)
def test_virga_config_requires_one_valid_source_for_each_control(
    tmp_path: Path,
    control: str,
    updates: dict,
    message: str,
) -> None:
    cloud = _virga_cloud(tmp_path)
    cloud.update(updates)

    with pytest.raises(ValidationError, match=message):
        VirgaEddySedCloudConfig.model_validate(cloud)


@pytest.mark.parametrize(
    ("control", "parameter", "message"),
    [
        (
            "fsed_parameter",
            _retrieval_parameter("cloud_fsed", 0.0, 10.0, unit="dimensionless"),
            "positive uniform or log-uniform bounds",
        ),
        (
            "fsed_parameter",
            _retrieval_parameter("cloud_fsed", 0.1, 10.0, unit="cm/s"),
            "dimensionless unit",
        ),
        (
            "log10_kz_cm2_s_parameter",
            _retrieval_parameter("cloud_log10_kzz", 4.0, 12.0, unit="m^2/s"),
            r"unit must describe log10\(cm\^2/s\)",
        ),
        (
            "log10_kz_cm2_s_parameter",
            _retrieval_parameter(
                "cloud_log10_kzz", 4.0, 12.0, unit="dex", prior_type="log_uniform"
            ),
            "requires a uniform prior",
        ),
        (
            "log10_kz_cm2_s_parameter",
            _retrieval_parameter("cloud_log10_kzz", -301.0, 12.0, unit="dex"),
            r"must lie within \[-300, 300\]",
        ),
    ],
)
def test_virga_retrieval_prior_units_and_bounds_are_validated(
    tmp_path: Path,
    control: str,
    parameter: dict,
    message: str,
) -> None:
    mie_directory = _make_mie_table(tmp_path / "mie")
    cloud = _virga_cloud(mie_directory)
    cloud.pop("fsed")
    cloud.pop("kz_cm2_s")
    cloud["fsed_parameter"] = "cloud_fsed"
    cloud["log10_kz_cm2_s_parameter"] = "cloud_log10_kzz"
    name = parameter["name"]
    cloud[control] = name
    source = _write_config(
        tmp_path,
        EMISSION,
        cloud=cloud,
        extra_parameters=[
            _retrieval_parameter("cloud_fsed", 0.1, 10.0, unit="dimensionless"),
            _retrieval_parameter("cloud_log10_kzz", 4.0, 12.0, unit="log10(cm^2/s)"),
        ],
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["parameters"][-2 if control == "fsed_parameter" else -1] = parameter
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_task_config(source)


def test_virga_retrieval_parameter_names_must_not_collide_with_other_models(
    tmp_path: Path,
) -> None:
    mie_directory = _make_mie_table(tmp_path / "mie")
    cloud = _virga_cloud(mie_directory)
    cloud.pop("fsed")
    cloud["fsed_parameter"] = "gamma1"
    source = _write_config(
        tmp_path,
        EMISSION,
        cloud=cloud,
    )

    with pytest.raises(ValidationError, match="collide with chemistry or temperature"):
        load_task_config(source)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gas_mass_mixing_ratios": {"KCl": 1.0, "Na2S": 1.0}}, "exactly one value"),
        ({"gas_mass_mixing_ratios": {"KCl": 1.1}}, "sum to no more than one"),
        ({"geometric_stddev": 1.0}, "greater than one"),
        ({"fsed": float("inf")}, "finite"),
    ],
)
def test_virga_config_rejects_invalid_fixed_controls(
    overrides: dict,
    message: str,
) -> None:
    cloud = _virga_cloud(Path("."))
    cloud.update(overrides)

    with pytest.raises(ValidationError, match=message):
        VirgaEddySedCloudConfig.model_validate(cloud)


def test_transmission_virga_requires_constant_gravity_in_config(tmp_path: Path) -> None:
    cloud = _virga_cloud(tmp_path / "mie")
    source = _write_config(
        tmp_path,
        TRANSMISSION,
        cloud=cloud,
        gravity_model="inverse_square",
    )

    with pytest.raises(
        ValidationError, match="require radiative_transfer.gravity_model=constant"
    ):
        load_task_config(source)


def test_transmission_virga_accepts_explicit_constant_gravity(tmp_path: Path) -> None:
    source = _write_config(
        tmp_path,
        TRANSMISSION,
        cloud=_virga_cloud(tmp_path / "mie"),
        gravity_model="constant",
    )

    config = load_task_config(source)

    assert config.radiative_transfer.gravity_model == "constant"
    assert isinstance(config.clouds, VirgaEddySedCloudConfig)
