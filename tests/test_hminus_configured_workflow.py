"""Strict-YAML wiring for the shared H-minus continuum option."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets import (
    HMinusContinuumConfig,
    Observation,
    ObservationCollection,
    ObservationDataset,
)
from robert_exoplanets.core import RobertConfigError, RobertDataError
from robert_exoplanets.io import configured_tasks
from robert_exoplanets.io.task_config import TaskConfig, load_task_config


ROOT = Path(__file__).resolve().parents[1]


def _configured_hminus(*, rt_model: str, chemistry_model: str) -> TaskConfig:
    raw = deepcopy(
        load_task_config(ROOT / "configurations" / "emission.yaml").model_dump(
            mode="python"
        )
    )
    rt = raw["radiative_transfer"]
    rt["hminus_continuum"] = {}
    if rt_model == "transmission":
        for key in (
            "geometry",
            "thermal_integration_backend",
            "sh4_boundary_backend",
        ):
            rt.pop(key, None)
        rt.update(
            model="transmission",
            reference_pressure_bar=1.0,
            gravity_model="constant",
            impact_quadrature_order=4,
        )
    else:
        for key in (
            "reference_pressure_bar",
            "radius_scale_parameter",
            "gravity_model",
            "impact_quadrature_order",
        ):
            rt.pop(key, None)
        rt["model"] = "emission"

    chemistry = raw["atmosphere"]["chemistry"]
    if chemistry_model == "fastchem_equilibrium":
        chemistry["full_equilibrium_composition"] = True
        chemistry["constant_log10_vmr_parameters"] = {}
    else:
        species = tuple(raw["opacity"]["species"]) + ("H-", "H", "e-")
        chemistry.clear()
        chemistry.update(
            model="free",
            species=species,
            fixed_mixing_ratios={name: 1.0e-8 for name in species},
        )
    return TaskConfig.model_validate(raw)


@pytest.mark.parametrize("rt_model", ["emission", "transmission"])
@pytest.mark.parametrize("chemistry_model", ["fastchem_equilibrium", "free"])
def test_shared_hminus_yaml_config_roundtrips_for_both_modes(
    rt_model: str, chemistry_model: str
) -> None:
    config = _configured_hminus(rt_model=rt_model, chemistry_model=chemistry_model)
    roundtrip = TaskConfig.model_validate(config.model_dump(mode="python"))

    assert roundtrip.radiative_transfer.model == rt_model
    assert roundtrip.radiative_transfer.hminus_continuum is not None
    assert (
        configured_tasks._hminus_continuum_config(
            roundtrip.radiative_transfer.hminus_continuum
        )
        == HMinusContinuumConfig()
    )


@pytest.mark.parametrize("chemistry_model", ["fastchem_equilibrium", "free"])
def test_enabled_hminus_requires_explicit_species_profiles(
    chemistry_model: str,
) -> None:
    raw = _configured_hminus(
        rt_model="emission", chemistry_model=chemistry_model
    ).model_dump(mode="python")
    chemistry = raw["atmosphere"]["chemistry"]
    if chemistry_model == "fastchem_equilibrium":
        chemistry["full_equilibrium_composition"] = False
        chemistry["species"] = [
            item
            for item in chemistry["species"]
            if item["label"] not in {"H-", "H", "e-"}
        ]
    else:
        chemistry["species"] = tuple(
            name for name in chemistry["species"] if name not in {"H-", "H", "e-"}
        )
        chemistry["fixed_mixing_ratios"] = {}

    with pytest.raises(
        ValueError, match="H-minus continuum requires explicit chemistry profiles"
    ):
        TaskConfig.model_validate(raw)


def test_fastchem_hminus_uses_native_negative_ion_identifier() -> None:
    raw = _configured_hminus(
        rt_model="transmission", chemistry_model="fastchem_equilibrium"
    ).model_dump(mode="python")
    species = list(raw["atmosphere"]["chemistry"]["species"])
    species.append({"label": "H-", "fastchem_name": "H-"})
    raw["atmosphere"]["chemistry"]["species"] = species

    with pytest.raises(ValueError, match="must map to FastChem species H1-"):
        TaskConfig.model_validate(raw)


def test_full_fastchem_state_rejects_constant_vmr_overrides() -> None:
    raw = _configured_hminus(
        rt_model="emission", chemistry_model="fastchem_equilibrium"
    ).model_dump(mode="python")
    raw["atmosphere"]["chemistry"]["constant_log10_vmr_parameters"] = {"SO2": "log_SO2"}

    with pytest.raises(
        ValueError,
        match="full_equilibrium_composition does not allow constant VMR overrides",
    ):
        TaskConfig.model_validate(raw)


@pytest.mark.parametrize("rt_model", ["emission", "transmission"])
def test_configured_factory_passes_hminus_to_selected_forward_model(
    rt_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _configured_hminus(
        rt_model=rt_model, chemistry_model="fastchem_equilibrium"
    )
    observation = Observation.from_arrays(
        wavelength=[2.0, 2.5],
        flux=[1.0e-3, 1.1e-3],
        uncertainty=[1.0e-4, 1.0e-4],
        instrument="NIRCam",
    )
    observations = ObservationCollection((ObservationDataset("f322w2", observation),))
    monkeypatch.setattr(
        configured_tasks,
        "_opacity_providers",
        lambda _config, _observations, *, planet: {"f322w2": object()},
    )
    monkeypatch.setattr(
        configured_tasks,
        "ParameterizedEmissionFactoryConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        configured_tasks,
        "ParameterizedTransmissionFactoryConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(configured_tasks, "load_nemesispy_cia_table", lambda: None)
    captured: list[object] = []

    class StubRegionalModel:
        models = {
            "f322w2": SimpleNamespace(opacity_identifiers={}, manifest_metadata={})
        }

        def __call__(self, parameters):
            return {}

    if rt_model == "emission":

        def capture(factories, *, spectral_grids):
            captured.append(factories["f322w2"].model.hminus_continuum)
            return StubRegionalModel()

        monkeypatch.setattr(
            configured_tasks, "build_multi_dataset_emission_model", capture
        )
    else:

        def capture(factory, *, spectral_grid):
            captured.append(factory.model.hminus_continuum)
            return SimpleNamespace(opacity_identifiers={}, manifest_metadata={})

        monkeypatch.setattr(
            configured_tasks, "build_parameterized_transmission_model", capture
        )

    configured_tasks.build_problem(config, observations)

    assert captured == [HMinusContinuumConfig()]


def test_mode_specific_radiative_transfer_options_are_checked_only_when_explicit() -> (
    None
):
    raw = _configured_hminus(
        rt_model="transmission", chemistry_model="free"
    ).model_dump(mode="python")
    raw["radiative_transfer"]["geometry"] = {"model": "gauss_legendre_disk"}

    with pytest.raises(ValueError, match="emission-only radiative_transfer options"):
        TaskConfig.model_validate(raw)


def _write_npz(path: Path, wavelength: list[float], edges: list[float]) -> None:
    np.savez(
        path,
        wavelength=np.asarray(wavelength),
        wavelength_bin_edges=np.asarray(edges),
        data=np.asarray([1.0, 1.1]),
        err=np.asarray([0.1, 0.1]),
        wavelength_unit=np.asarray("micron"),
        flux_unit=np.asarray("eclipse_depth"),
        observable=np.asarray("eclipse_depth"),
    )


def test_robert_npz_directory_keeps_named_detector_edges_separate(
    tmp_path: Path,
) -> None:
    raw = deepcopy(
        load_task_config(ROOT / "configurations" / "emission.yaml").model_dump(
            mode="python"
        )
    )
    observations_dir = tmp_path / "observations"
    observations_dir.mkdir()
    _write_npz(observations_dir / "nrs1.npz", [2.1, 2.2], [2.0, 2.15, 2.3])
    _write_npz(observations_dir / "nrs2.npz", [4.1, 4.2], [4.0, 4.15, 4.3])
    raw["observations"].update(
        loader="robert_npz",
        path=observations_dir,
        datasets=("nrs1", "nrs2"),
        verify_checksum=False,
    )

    config = TaskConfig.model_validate(raw)
    config = TaskConfig.model_validate(config.model_dump(mode="python"))
    observations = configured_tasks.load_observations(config)

    assert tuple(dataset.name for dataset in observations.datasets) == ("nrs1", "nrs2")
    np.testing.assert_array_equal(
        observations.datasets[0].observation.spectral_grid.bin_edges,
        [2.0, 2.15, 2.3],
    )
    np.testing.assert_array_equal(
        observations.datasets[1].observation.spectral_grid.bin_edges,
        [4.0, 4.15, 4.3],
    )
    assert observations.metadata["source_path"] == str(observations_dir)
    assert "nrs1=" in observations.metadata["dataset_paths"]
    assert "nrs2=" in observations.metadata["dataset_paths"]


def test_robert_npz_single_file_and_ambiguous_multi_dataset_paths(
    tmp_path: Path,
) -> None:
    raw = deepcopy(
        load_task_config(ROOT / "configurations" / "emission.yaml").model_dump(
            mode="python"
        )
    )
    source = tmp_path / "legacy.npz"
    _write_npz(source, [2.1, 2.2], [2.0, 2.15, 2.3])
    raw["observations"].update(
        loader="robert_npz",
        path=source,
        datasets=("legacy",),
        verify_checksum=False,
    )
    single = configured_tasks.load_observations(TaskConfig.model_validate(raw))
    assert len(single.datasets) == 1
    assert single.metadata == {"source_path": str(source)}

    raw["observations"]["datasets"] = ("nrs1", "nrs2")
    config = TaskConfig.model_validate(raw)
    with pytest.raises(
        RobertConfigError,
        match="multiple datasets requires observations.path to be a directory",
    ):
        configured_tasks.load_observations(config)

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    raw["observations"]["path"] = missing_dir
    config = TaskConfig.model_validate(raw)
    with pytest.raises(RobertDataError, match="observation NPZ does not exist"):
        configured_tasks.load_observations(config)
