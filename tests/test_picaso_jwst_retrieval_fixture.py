from __future__ import annotations

import numpy as np

from examples.picaso_jwst_transmission_retrieval import (
    N_LAYERS,
    build_picaso_contract,
    jwst_grid,
)


def test_jwst_grid_is_contiguous_and_has_three_uncertainty_floors() -> None:
    wavelength, edges, uncertainty_ppm = jwst_grid()

    assert wavelength.shape == (72,)
    assert edges.shape == (73,)
    assert np.all(np.diff(edges) > 0.0)
    assert np.allclose(wavelength, np.sqrt(edges[:-1] * edges[1:]))
    assert set(uncertainty_ppm) == {20.0, 25.0, 35.0}


def test_picaso_contract_matches_retrieval_truth_and_radius_anchor() -> None:
    contract = build_picaso_contract()

    assert contract["gas_vmr"].shape == (N_LAYERS, 4)
    assert np.allclose(contract["gas_vmr"][0], [1.0e-3, 3.0e-4, 2.0e-5, 4.0e-6])
    assert np.all(contract["temperature_level_k"] == 1100.0)
    assert float(contract["reference_pressure_bar"]) == 10.0
    assert float(contract["pressure_edges_bar"][-1]) == 10.0
    assert np.all(contract["he_vmr"] > 0.0)
