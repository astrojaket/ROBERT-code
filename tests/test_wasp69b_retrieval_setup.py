"""Regression tests for the runnable WASP-69b retrieval configuration."""

import numpy as np

from examples.retrieve_wasp69b_nircam_clear import parameters
from examples.wasp69b_target import PLANET, PLANET_GRAVITY_M_S2, STAR


def test_wasp69b_clear_retrieval_uses_requested_chemistry_priors() -> None:
    bounds = {parameter.name: parameter.bounds for parameter in parameters().parameters}

    assert bounds["metallicity"] == (-1.0, 2.0)
    assert bounds["CtoO"] == (0.0, 1.0)


def test_wasp69b_target_parameters_have_one_shared_gravity() -> None:
    expected = 6.67430e-11 * PLANET.mass_kg / PLANET.radius_m**2

    assert PLANET.name == "WASP-69b"
    assert STAR.name == "WASP-69"
    np.testing.assert_allclose(PLANET_GRAVITY_M_S2, expected, rtol=0.0, atol=0.0)
