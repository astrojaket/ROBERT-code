"""Tests for model-setup factories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import AtmosphereModelSetup, Planet, build_atmosphere_setup
from robert_exoplanets.atmosphere import (
    CompositionMeanMolecularWeight,
    FreeChemistry,
    ParmentierGuillot2014TemperatureProfile,
    TabulatedTemperatureProfile,
)
from robert_exoplanets.core import RobertConfigError
from robert_exoplanets.io import (
    build_chemistry_model_from_config,
    build_pressure_grid_from_config,
    build_temperature_profile_from_config,
)


def hat_p_32b_style_config() -> dict:
    return {
        "pressure_grid": {
            "n_layers": 8,
            "p_top_bar": 1.0e-6,
            "p_bot_bar": 100.0,
        },
        "temperature_profile": {
            "type": "guillot14",
            "values": {
                "kappa_IR": 0.02,
                "gamma1": 0.5,
                "gamma2": 1.5,
                "T_irr": 1500.0,
                "alpha": 0.5,
                "T_int": 200.0,
            },
        },
        "molecules": {
            "free": {
                "names": ["H2O", "CO"],
                "inactive": {"names": ["H2", "He"]},
                "log": True,
                "values": {
                    "H2O": 1.0e-3,
                    "CO": 1.0e-4,
                },
            },
        },
    }


def test_build_pressure_grid_from_hat_style_config() -> None:
    grid = build_pressure_grid_from_config(hat_p_32b_style_config())

    assert grid.n_layers == 8
    assert grid.unit == "bar"
    assert grid.centers[0] < grid.centers[-1]


def test_build_atmosphere_setup_from_hat_style_config() -> None:
    planet = Planet(name="HAT-P-32b", radius_m=1.4e8, gravity_m_s2=4.3)

    setup = build_atmosphere_setup(hat_p_32b_style_config(), planet=planet)

    assert isinstance(setup, AtmosphereModelSetup)
    assert isinstance(setup.temperature_profile, ParmentierGuillot2014TemperatureProfile)
    assert isinstance(setup.chemistry_model, FreeChemistry)
    assert isinstance(setup.mean_molecular_weight_model, CompositionMeanMolecularWeight)
    assert setup.default_parameters["kappa_IR"] == 0.02
    assert setup.default_parameters["H2O"] == 1.0e-3

    atmosphere = setup.build_atmosphere_builder().build(setup.default_parameters)

    assert atmosphere.n_layers == 8
    assert atmosphere.species == ("H2O", "CO", "H2", "He")
    np.testing.assert_allclose(atmosphere.composition["H2O"], np.full(8, 1.0e-3))
    assert np.all(atmosphere.temperature > 0.0)
    assert np.all(atmosphere.mean_molecular_weight > 2.0)


def test_temperature_factory_can_load_tabulated_profile_from_emission_block(tmp_path: Path) -> None:
    profile_path = tmp_path / "pt.csv"
    profile_path.write_text(
        "pressure_bar,temperature_K\n"
        "1.0e-6,1000.0\n"
        "1.0,1500.0\n",
        encoding="utf-8",
    )

    profile, parameters = build_temperature_profile_from_config(
        {
            "temperature_profile": {"type": "tabulated"},
            "emission": {"pt_profile_csv": str(profile_path)},
        }
    )

    assert isinstance(profile, TabulatedTemperatureProfile)
    assert parameters == {}


def test_temperature_factory_rejects_unsupported_guillot_shortcut() -> None:
    with pytest.raises(RobertConfigError, match="not implemented"):
        build_temperature_profile_from_config({"temperature_profile": {"type": "guillot"}})


def test_chemistry_factory_parses_inactive_fractions() -> None:
    chemistry, parameters = build_chemistry_model_from_config(
        {
            "molecules": {
                "free": {
                    "names": ["H2O"],
                    "values": {"H2O": 1.0e-3},
                    "inactive": {
                        "names": ["H2", "He"],
                        "fractions": {"H2": 3.0, "He": 1.0},
                    },
                },
            }
        }
    )

    assert isinstance(chemistry, FreeChemistry)
    assert parameters == {"H2O": 1.0e-3}
    assert chemistry.background is not None
    assert chemistry.background.fractions == {"H2": 0.75, "He": 0.25}


def test_chemistry_factory_requires_active_species() -> None:
    with pytest.raises(RobertConfigError, match="active species"):
        build_chemistry_model_from_config({"molecules": {"free": {"names": []}}})
