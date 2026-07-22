"""Tests for reproducible flat-spectrum ensemble helpers."""

from pathlib import Path

import numpy as np

from robert_exoplanets.io import load_bello_arufe2025_l9859b
from robert_exoplanets.validation import (
    abundance_constraint,
    closed_composition,
    composition_mean_molecular_weight,
    constant_resolving_power_grid,
    generate_flat_spectrum_ensemble,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "observations" / "l98_59b_bello_arufe2025"


def test_constant_resolving_power_grid_has_exact_span_and_convention() -> None:
    grid = constant_resolving_power_grid(3.0, 5.0, 100.0)

    assert grid.n_bins == 51
    assert grid.edges_micron[0] == 3.0
    assert grid.edges_micron[-1] == 5.0
    np.testing.assert_allclose(
        grid.centers_micron / np.diff(grid.edges_micron),
        grid.effective_resolving_power,
    )
    np.testing.assert_allclose(grid.effective_resolving_power, 99.8379573)


def test_flat_ensemble_is_reproducible_and_uses_published_medians() -> None:
    source = load_bello_arufe2025_l9859b(DATA)
    first, metadata = generate_flat_spectrum_ensemble(source, n_realizations=2, seed=17)
    second, _ = generate_flat_spectrum_ensemble(source, n_realizations=2, seed=17)

    np.testing.assert_array_equal(first[0].flux, second[0].flux)
    assert not np.array_equal(first[0].flux, first[1].flux)
    np.testing.assert_allclose(first[0].uncertainty, 3.7166e-5)
    np.testing.assert_allclose(metadata["median_transit_depth_ppm"], 619.782)
    np.testing.assert_allclose(metadata["median_uncertainty_ppm"], 37.166)


def test_closure_mmw_and_constraint_diagnostic() -> None:
    vmr = closed_composition(
        {"H2O": np.log10(np.array([0.1, 0.2])), "CO2": np.log10(np.array([0.3, 0.2]))},
        closure_species="H2S",
    )
    np.testing.assert_allclose(vmr["H2S"], [0.6, 0.6])
    mmw = composition_mean_molecular_weight(vmr)
    assert np.all((mmw > 18.0) & (mmw < 45.0))

    diagnostic = abundance_constraint(
        np.array([0.02, 0.03, 0.04]), np.ones(3), threshold=0.01, credibility=0.95
    )
    assert diagnostic["constrained"] is True
