"""Regression tests for the runnable WASP-69b retrieval configuration."""

from examples.retrieve_wasp69b_nircam_clear import parameters


def test_wasp69b_clear_retrieval_uses_requested_chemistry_priors() -> None:
    bounds = {parameter.name: parameter.bounds for parameter in parameters().parameters}

    assert bounds["metallicity"] == (-1.0, 2.0)
    assert bounds["CtoO"] == (0.0, 1.0)
