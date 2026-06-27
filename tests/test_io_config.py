"""Tests for typed configuration skeletons."""

from __future__ import annotations

import pytest

from robert_exoplanets.bodies import Planet
from robert_exoplanets.instruments import Observation
from robert_exoplanets.io import RobertConfig
from robert_exoplanets.core import RobertConfigError


def test_robert_config_holds_validated_domain_objects() -> None:
    planet = Planet(name="WASP-43b", radius_m=7.0e7, gravity_m_s2=45.0)
    observation = Observation.from_arrays(
        wavelength=[5.0, 6.0],
        flux=[1.0e-4, 1.1e-4],
        uncertainty=[1.0e-5, 1.0e-5],
    )

    config = RobertConfig(
        run_name="wasp43b-smoke",
        planet=planet,
        observations=(observation,),
    )

    assert config.planet is planet
    assert config.observations == (observation,)


def test_robert_config_requires_observations() -> None:
    planet = Planet(name="WASP-43b", radius_m=7.0e7, gravity_m_s2=45.0)

    with pytest.raises(RobertConfigError, match="observation"):
        RobertConfig(run_name="missing-observation", planet=planet, observations=())
