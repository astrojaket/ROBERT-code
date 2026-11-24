"""Tests for fixed tabulated VMR chemistry profiles."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import yaml

from robert_exoplanets import (
    BackgroundGasMixture,
    CorrelatedKTable,
    Observation,
    ObservationCollection,
    ObservationDataset,
    PressureGrid,
    TabulatedChemistry,
)
from robert_exoplanets.atmosphere import CompositionMeanMolecularWeight
from robert_exoplanets.io import configured_tasks
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3", "SO2", "H2S")
BACKGROUND = BackgroundGasMixture({"H2": 0.85, "He": 0.15})


def _centred_pressure_grid(centers: np.ndarray) -> PressureGrid:
    log_centers = np.log(centers)
    log_edges = np.empty(centers.size + 1)
    log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
    log_edges[0] = 2.0 * log_centers[0] - log_edges[1]
    log_edges[-1] = 2.0 * log_centers[-1] - log_edges[-2]
    return PressureGrid(edges=np.exp(log_edges), centers=centers, unit="bar")


def _profile_values() -> dict[str, np.ndarray]:
    return {
        "H2O": np.array([1.0e-4, 1.0e-3, 1.0e-2]),
        **{
            species: np.full(3, (index + 1) * 1.0e-5)
            for index, species in enumerate(PROFILE_SPECIES[1:])
        },
    }


def _write_profile(path: Path) -> None:
    values = _profile_values()
    columns = ("Pressure", *PROFILE_SPECIES)
    rows = []
    for row_index, pressure in enumerate((0.1, 1.0, 22.535)):
        rows.append(
            ",".join(
                [f"{pressure:g}"]
                + [f"{values[species][row_index]:.12g}" for species in PROFILE_SPECIES]
            )
        )
    path.write_text(",".join(columns) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _chemistry() -> TabulatedChemistry:
    return TabulatedChemistry(
        pressure=np.array([0.1, 1.0, 22.535]),
        mixing_ratios=_profile_values(),
        pressure_unit="bar",
        background=BACKGROUND,
        endpoint_policy="constant",
    )


def test_tabulated_chemistry_interpolates_log_vmr_and_clips_constant_endpoints() -> (
    None
):
    centers = np.array([200.0, 22.535, 1.0, np.sqrt(0.1), 0.01])
    grid = _centred_pressure_grid(centers)

    composition = _chemistry().evaluate({}, grid, np.full(grid.n_layers, 1400.0))

    assert tuple(composition) == PROFILE_SPECIES + ("H2", "He")
    np.testing.assert_allclose(
        composition["H2O"], [1.0e-2, 1.0e-2, 1.0e-3, np.sqrt(1.0e-7), 1.0e-4]
    )
    total_vmr = np.sum(np.stack(tuple(composition.values())), axis=0)
    np.testing.assert_allclose(
        total_vmr, np.ones(grid.n_layers), rtol=0.0, atol=2.0e-16
    )
    np.testing.assert_allclose(
        composition["H2"],
        0.85
        * (
            1.0
            - np.sum(np.stack([composition[name] for name in PROFILE_SPECIES]), axis=0)
        ),
    )
    assert all(not values.flags.writeable for values in composition.values())
    assert _chemistry().required_parameters() == ()


@pytest.mark.parametrize(
    ("unit", "scale"),
    (("bar", 1.0), ("Pa", 1.0e5), ("mbar", 1.0e3)),
)
def test_tabulated_chemistry_converts_profile_and_grid_pressure_units(
    unit: str,
    scale: float,
) -> None:
    centers = np.array([200.0, 22.535, 1.0, np.sqrt(0.1), 0.01])
    bar_grid = _centred_pressure_grid(centers)
    target_grid = PressureGrid(
        edges=bar_grid.edges * 1.0e5,
        centers=bar_grid.centers * 1.0e5,
        unit="Pa",
    )
    chemistry = TabulatedChemistry(
        pressure=np.array([0.1, 1.0, 22.535]) * scale,
        mixing_ratios=_profile_values(),
        pressure_unit=unit,
        background=BACKGROUND,
        endpoint_policy="constant",
    )

    bar_composition = chemistry.evaluate(
        {}, bar_grid, np.full(bar_grid.n_layers, 1400.0)
    )
    converted_grid_composition = chemistry.evaluate(
        {}, target_grid, np.full(target_grid.n_layers, 1400.0)
    )

    for species in PROFILE_SPECIES + ("H2", "He"):
        np.testing.assert_allclose(
            converted_grid_composition[species],
            bar_composition[species],
            rtol=1.0e-13,
            atol=0.0,
        )


def test_tabulated_chemistry_csv_records_source_checksum_and_column_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chemistry.csv"
    _write_profile(path)

    chemistry = TabulatedChemistry.from_csv(
        path,
        pressure_column="Pressure",
        species=PROFILE_SPECIES,
        pressure_unit="bar",
        background=BACKGROUND,
        endpoint_policy="constant",
    )

    assert chemistry.source_path == path
    assert chemistry.metadata["source_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert chemistry.metadata["pressure_unit"] == "bar"
    assert chemistry.metadata["pressure_column"] == "Pressure"
    assert chemistry.metadata["endpoint_policy"] == "constant"


def test_tabulated_chemistry_rejects_nonpositive_vmr_and_mass_excess() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        TabulatedChemistry(
            pressure=np.array([1.0, 10.0]),
            mixing_ratios={"H2O": np.array([1.0e-4, 0.0])},
            pressure_unit="bar",
            background=BACKGROUND,
            endpoint_policy="constant",
        )

    with pytest.raises(ValueError, match="sum to no more than one"):
        TabulatedChemistry(
            pressure=np.array([1.0, 10.0]),
            mixing_ratios={
                "H2O": np.array([0.6, 0.6]),
                "CO": np.array([0.5, 0.5]),
            },
            pressure_unit="bar",
            background=BACKGROUND,
            endpoint_policy="constant",
        )

    with pytest.raises(ValueError, match="finite and positive"):
        TabulatedChemistry(
            pressure=np.array([1.0, 10.0]),
            mixing_ratios={"H2O": np.array([1.0e-4, np.nan])},
            pressure_unit="bar",
            background=BACKGROUND,
            endpoint_policy="constant",
        )

    with pytest.raises(ValueError, match="pressure values must be unique"):
        TabulatedChemistry(
            pressure=np.array([1.0, 1.0]),
            mixing_ratios={"H2O": np.array([1.0e-4, 2.0e-4])},
            pressure_unit="bar",
            background=BACKGROUND,
            endpoint_policy="constant",
        )

    with pytest.raises(ValueError, match="pressure values must be finite"):
        TabulatedChemistry(
            pressure=np.array([1.0, np.inf]),
            mixing_ratios={"H2O": np.array([1.0e-4, 2.0e-4])},
            pressure_unit="bar",
            background=BACKGROUND,
            endpoint_policy="constant",
        )


def _tiny_table(species: str) -> CorrelatedKTable:
    return CorrelatedKTable(
        species=species,
        pressure_bar=np.array([1.0e-3, 1.0]),
        temperature_K=np.array([500.0, 1500.0]),
        wavenumber_cm_inverse=np.array([5000.0, 4000.0]),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        kcoeff=np.full((2, 2, 2, 1), 1.0e-24),
    )


def _parameter(
    name: str,
    *,
    prior_type: str = "uniform",
    lower: float,
    upper: float,
    value: float,
) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "prior": {"type": prior_type, "lower": lower, "upper": upper},
    }


def test_tabulated_chemistry_config_builds_six_parameter_transmission_problem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = tmp_path / "chemistry.csv"
    _write_profile(profile_path)
    base = load_task_config(ROOT / "configurations" / "transmission.yaml")
    raw = deepcopy(base.model_dump(mode="json"))
    raw["run"]["name"] = "tabulated-chemistry-config-test"
    raw["atmosphere"]["pressure"]["bottom_bar"] = 200.0
    raw["atmosphere"]["temperature"] = {
        "model": "parmentier_guillot_2014",
        "internal_temperature_k": 100.0,
    }
    raw["atmosphere"]["chemistry"] = {
        "model": "tabulated",
        "profile_path": profile_path.name,
        "pressure_column": "Pressure",
        "pressure_unit": "bar",
        "species": list(PROFILE_SPECIES),
        "background_fractions": {"H2": 0.85, "He": 0.15},
        "endpoint_policy": "constant",
    }
    raw["opacity"]["species"] = list(PROFILE_SPECIES)
    raw["radiative_transfer"]["radius_scale_parameter"] = "radius_scale"
    raw["parameters"] = [
        _parameter("radius_scale", lower=0.99, upper=1.01, value=1.0),
        _parameter(
            "kappa_IR", prior_type="log_uniform", lower=1.0e-5, upper=1.0, value=0.02
        ),
        _parameter(
            "gamma1", prior_type="log_uniform", lower=1.0e-4, upper=100.0, value=0.5
        ),
        _parameter(
            "gamma2", prior_type="log_uniform", lower=1.0e-4, upper=100.0, value=1.5
        ),
        _parameter("T_irr", lower=800.0, upper=2200.0, value=1500.0),
        _parameter("alpha", lower=0.0, upper=1.0, value=0.5),
    ]
    config_path = tmp_path / "configuration.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_task_config(config_path)

    observation = Observation.from_arrays(
        wavelength=[2.0, 2.5],
        flux=[1.0e-3, 1.1e-3],
        uncertainty=[1.0e-4, 1.0e-4],
        instrument="test",
    )
    observations = ObservationCollection(
        (ObservationDataset("synthetic_transmission", observation),)
    )
    monkeypatch.setattr(
        configured_tasks,
        "_load_cached_table",
        lambda _config, _dataset, species: _tiny_table(species),
    )
    monkeypatch.setattr(configured_tasks, "load_nemesispy_cia_table", lambda: None)

    problem = configured_tasks.build_problem(config, observations)

    assert problem.ndim == 6
    assert problem.parameter_names == (
        "radius_scale",
        "kappa_IR",
        "gamma1",
        "gamma2",
        "T_irr",
        "alpha",
    )
    model = problem.forward_model.models["synthetic_transmission"]
    assert model.atmosphere_builder.chemistry_model.species == PROFILE_SPECIES + (
        "H2",
        "He",
    )
    assert isinstance(model.atmosphere_builder.chemistry_model, TabulatedChemistry)
    assert isinstance(
        model.atmosphere_builder.mean_molecular_weight_model,
        CompositionMeanMolecularWeight,
    )
    assert model.atmosphere_builder.chemistry_model.required_parameters() == ()
    assert (
        model.manifest_metadata["chemistry_source_sha256"]
        == sha256(profile_path.read_bytes()).hexdigest()
    )
    assert (
        problem.metadata["chemistry_source_sha256"]
        == sha256(profile_path.read_bytes()).hexdigest()
    )
    assert model.pressure_grid.centers[0] == pytest.approx(200.0)
    atmosphere = model.atmosphere_builder.build(
        {
            "radius_scale": 1.0,
            "kappa_IR": 0.02,
            "gamma1": 0.5,
            "gamma2": 1.5,
            "T_irr": 1500.0,
            "alpha": 0.5,
        }
    )
    assert atmosphere.species == PROFILE_SPECIES + ("H2", "He")
    np.testing.assert_allclose(
        np.sum(np.stack(tuple(atmosphere.composition.values())), axis=0),
        np.ones(atmosphere.n_layers),
        rtol=0.0,
        atol=2.0e-16,
    )
    assert np.all(atmosphere.mean_molecular_weight > 0.0)


def test_tabulated_chemistry_schema_requires_pressure_unit_and_h2_he_background(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "chemistry.csv"
    _write_profile(profile_path)
    base = load_task_config(ROOT / "configurations" / "transmission.yaml")
    raw = deepcopy(base.model_dump(mode="json"))
    raw["atmosphere"]["chemistry"] = {
        "model": "tabulated",
        "profile_path": str(profile_path),
        "pressure_column": "Pressure",
        "pressure_unit": "bar",
        "species": list(PROFILE_SPECIES),
        "background_fractions": {"H2": 0.85, "He": 0.15},
        "endpoint_policy": "constant",
    }
    raw["opacity"]["species"] = list(PROFILE_SPECIES)
    config = type(base).model_validate(raw)
    assert config.atmosphere.chemistry.pressure_unit == "bar"

    missing_unit = deepcopy(raw)
    del missing_unit["atmosphere"]["chemistry"]["pressure_unit"]
    with pytest.raises(ValueError, match="pressure_unit"):
        type(base).model_validate(missing_unit)

    invalid_background = deepcopy(raw)
    invalid_background["atmosphere"]["chemistry"]["background_fractions"] = {
        "H2": 0.85,
        "N2": 0.15,
    }
    with pytest.raises(ValueError, match="define H2 and He"):
        type(base).model_validate(invalid_background)
