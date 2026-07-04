"""Tests for random-overlap correlated-k gas combination."""

from __future__ import annotations

import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    random_overlap_species_tau,
    random_overlap_tau_vectors,
    rank_rebin_distribution,
)


def test_rank_rebin_distribution_sorts_and_averages_into_target_bins() -> None:
    values = np.array([12.0, 1.0, 11.0, 2.0])
    weights = np.full(4, 0.25)
    target = np.array([0.5, 0.5])

    rebinned = rank_rebin_distribution(values, weights, target)

    np.testing.assert_allclose(rebinned, [1.5, 11.5])


def test_random_overlap_tau_vectors_combines_two_species_by_ranked_outer_sum() -> None:
    tau = np.array(
        [
            [0.0, 10.0],
            [1.0, 2.0],
        ]
    )
    weights = np.array([0.5, 0.5])

    combined = random_overlap_tau_vectors(tau, weights)

    np.testing.assert_allclose(combined, [1.5, 11.5])
    np.testing.assert_allclose(
        np.sum(combined * weights),
        np.sum(tau[0] * weights) + np.sum(tau[1] * weights),
    )


def test_random_overlap_species_tau_preserves_single_species_distribution() -> None:
    species_tau = np.array([[[[1.0e-14, 2.0e-14], [3.0e-14, 4.0e-14]]]])

    combined = random_overlap_species_tau(species_tau, [0.4, 0.6])

    np.testing.assert_allclose(combined, species_tau[0])


def test_assemble_gas_optical_depth_can_use_random_overlap_combination() -> None:
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3]),
        centers=np.array([1.0e-4]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([1000.0], unit="cm^-1", role="opacity")
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([1000.0]),
        composition={
            "H2O": np.array([1.0]),
            "CO": np.array([1.0]),
        },
        mean_molecular_weight=2.3,
    )
    kcoeff = np.array(
        [
            [[[0.0, 10.0]]],
            [[[1.0, 2.0]]],
        ]
    )
    opacity = _evaluated_opacity(pressure_grid, spectral_grid, ("H2O", "CO"), kcoeff)

    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=10.0,
        gas_combination="random_overlap",
    )

    expected = random_overlap_species_tau(gas_tau.species_tau, gas_tau.g_weights)
    np.testing.assert_allclose(gas_tau.total_tau, expected)
    assert gas_tau.metadata["gas_combination"] == "random_overlap"


def _evaluated_opacity(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    species: tuple[str, ...],
    kcoeff: np.ndarray,
) -> EvaluatedCorrelatedKOpacity:
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test-correlated-k",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=species,
        g_samples=np.array([0.25, 0.75]),
        g_weights=np.array([0.5, 0.5]),
        cache_key="test-random-overlap-cache-key",
    )
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit="m^2/molecule",
    )
