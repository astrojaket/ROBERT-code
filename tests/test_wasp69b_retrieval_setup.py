"""Regression tests for the runnable WASP-69b retrieval configuration."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from examples import benchmark_wasp69b_multi_instrument
from examples.retrieve_wasp69b_nircam_cloud_free import parameters
from examples.wasp69b_target import PLANET, PLANET_GRAVITY_M_S2, STAR


ROOT = Path(__file__).resolve().parents[1]


def test_wasp69b_cloud_free_retrieval_uses_requested_chemistry_priors() -> None:
    bounds = {parameter.name: parameter.bounds for parameter in parameters().parameters}

    assert bounds["metallicity"] == (-1.0, 2.0)
    assert bounds["CtoO"] == (0.0, 1.0)


def test_wasp69b_target_parameters_have_one_shared_gravity() -> None:
    expected = 6.67430e-11 * PLANET.mass_kg / PLANET.radius_m**2

    assert PLANET.name == "WASP-69b"
    assert STAR.name == "WASP-69"
    np.testing.assert_allclose(PLANET_GRAVITY_M_S2, expected, rtol=0.0, atol=0.0)


def test_multi_instrument_benchmark_owns_its_native_hdf_selection() -> None:
    assert set(benchmark_wasp69b_multi_instrument.PATTERNS) == set(
        benchmark_wasp69b_multi_instrument.SPECIES
    )
    assert benchmark_wasp69b_multi_instrument.PRT_DATA.name == "input_data"


@pytest.mark.parametrize(
    ("module_name", "configuration"),
    [
        (
            "examples.retrieve_wasp69b_nircam_cloud_free",
            "targets/WASP-69b/wasp69b_cloud_free_nircam_pg14_R1000.yaml",
        ),
        (
            "examples.retrieve_wasp69b_mie_cloud",
            "targets/WASP-69b/wasp69b_mie_catalog_pg14_R1000.yaml",
        ),
        (
            "examples.retrieve_wasp69b_cloud_free_native_modes",
            "targets/WASP-69b/wasp69b_cloud_free_native_pg14_R1000.yaml",
        ),
        (
            "examples.retrieve_wasp80b_nircam_cloud_free",
            "targets/WASP-80b/wasp80b_cloud_free_nircam_pg14_R1000.yaml",
        ),
        (
            "examples.retrieve_wasp80b_mie_cloud",
            "targets/WASP-80b/wasp80b_mie_catalog_pg14_R1000.yaml",
        ),
        (
            "examples.retrieve_wasp80b_cloud_free_native_modes",
            "targets/WASP-80b/wasp80b_cloud_free_native_pg14_R1000.yaml",
        ),
    ],
)
def test_retrieval_entry_points_use_current_yaml_workflows(
    module_name: str,
    configuration: str,
) -> None:
    module = importlib.import_module(module_name)

    assert module.DEFAULT_CONFIG == ROOT / "configurations" / configuration
    assert callable(module.main)
