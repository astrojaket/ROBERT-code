"""Parity tests for the independent optional JAX conservative-RORR backend."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets.rt.jax_random_overlap import (
    jax_random_overlap_device_info,
    jax_random_overlap_species_tau,
)
from robert_exoplanets.rt.random_overlap import random_overlap_species_tau
from robert_exoplanets.rt.thermal_integration import (
    integrate_thermal_emission_spectrum,
)


def _enable_jax_float64():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def test_jax_conservative_rorr_matches_numpy_for_unsorted_distributions() -> None:
    _enable_jax_float64()
    rng = np.random.default_rng(26071947)
    tau = np.exp(rng.uniform(-12.0, 3.0, size=(4, 3, 5, 8)))
    weights = np.arange(1.0, 9.0)
    weights /= np.sum(weights)

    reference = random_overlap_species_tau(tau, weights, backend="numpy")
    accelerated = jax_random_overlap_species_tau(tau, weights)

    np.testing.assert_allclose(accelerated, reference, rtol=8.0e-13, atol=2.0e-13)


def test_jax_conservative_rorr_matches_numpy_cutoff_behavior() -> None:
    _enable_jax_float64()
    tau = np.array(
        [
            [[[[1.0e-16, 2.0e-16, 3.0e-16, 4.0e-16]]]],
            [[[[0.1, 0.2, 0.4, 0.8]]]],
            [[[[0.05, 0.1, 0.3, 0.9]]]],
        ]
    ).reshape(3, 1, 1, 4)
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    reference = random_overlap_species_tau(tau, weights, backend="numpy")
    accelerated = jax_random_overlap_species_tau(tau, weights)

    np.testing.assert_allclose(accelerated, reference, rtol=5.0e-14, atol=5.0e-14)
    assert any(device.startswith("cpu:") for device in jax_random_overlap_device_info())


def test_jax_single_species_preserves_values_below_multispecies_cutoff() -> None:
    _enable_jax_float64()
    tau = np.array([[[[1.0e-18, 2.0e-18, 4.0e-18]]]])
    weights = np.array([0.2, 0.3, 0.5])

    result = jax_random_overlap_species_tau(tau, weights, platform="cpu")

    np.testing.assert_array_equal(result, tau[0])


def test_jax_rorr_produces_same_thermal_spectrum_as_numpy_reference() -> None:
    _enable_jax_float64()
    rng = np.random.default_rng(26072026)
    species_tau = np.sort(
        np.exp(rng.uniform(-10.0, 1.0, size=(3, 4, 6, 8))), axis=-1
    )
    weights = np.arange(1.0, 9.0)
    weights /= np.sum(weights)
    source = np.exp(rng.uniform(1.0, 3.0, size=(4, 6)))
    paths = np.array([[1.0, 1.0, 1.0, 1.0], [1.8, 1.8, 1.8, 1.8]])
    point_weights = np.array([0.4, 0.6])

    numpy_tau = random_overlap_species_tau(
        species_tau, weights, backend="numpy"
    )
    jax_tau = jax_random_overlap_species_tau(
        species_tau, weights, platform="cpu"
    )
    numpy_spectrum = integrate_thermal_emission_spectrum(
        numpy_tau,
        source,
        weights,
        paths,
        point_weights,
        backend="numpy",
    )
    jax_spectrum = integrate_thermal_emission_spectrum(
        jax_tau,
        source,
        weights,
        paths,
        point_weights,
        backend="numpy",
    )

    np.testing.assert_allclose(
        jax_spectrum.radiance,
        numpy_spectrum.radiance,
        rtol=2.0e-12,
        atol=0.0,
    )
