"""Tests for the published Wiser et al. WASP-80b emission spectrum."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples import wasp80b_target
from robert_exoplanets import load_wiser2025_wasp80b


DATA = Path(__file__).resolve().parents[1] / "data" / "wasp80b_wiser2025"


def test_wasp80b_loader_preserves_modes_widths_and_asymmetric_errors() -> None:
    collection = load_wiser2025_wasp80b(DATA)

    assert collection.names == ("f322w2", "f444w", "lrs")
    assert [dataset.observation.n_points for dataset in collection.datasets] == [
        100,
        72,
        27,
    ]
    assert collection.n_points == 199
    assert collection.datasets[-1].offset_parameter == "miri_offset"
    first = collection.datasets[0].observation
    np.testing.assert_allclose(first.wavelength_bin_edges[:2], [2.45, 2.465])
    negative = np.fromstring(first.metadata["published_error_negative"], sep=",")
    positive = np.fromstring(first.metadata["published_error_positive"], sep=",")
    np.testing.assert_allclose(first.uncertainty, 0.5 * (negative + positive))


def test_wasp80b_target_config_points_to_versioned_data() -> None:
    configured = wasp80b_target.load_observations(miri_offset_parameter=None)

    assert wasp80b_target.DATA_DIRECTORY == DATA
    assert wasp80b_target.PLANET.name == "WASP-80b"
    assert wasp80b_target.STAR.name == "WASP-80"
    assert configured.datasets[-1].offset_parameter is None
