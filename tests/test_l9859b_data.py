"""Tests for the published L 98-59 b transmission spectrum."""

from pathlib import Path

import numpy as np

from robert_exoplanets.io import load_bello_arufe2025_l9859b


DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "observations"
    / "l98_59b_bello_arufe2025"
)


def test_l9859b_loader_splits_detectors_and_converts_ppm() -> None:
    collection = load_bello_arufe2025_l9859b(DATA)

    assert collection.names == ("nrs1", "nrs2")
    assert [dataset.observation.n_points for dataset in collection.datasets] == [
        84,
        134,
    ]
    assert collection.n_points == 218
    assert collection.datasets[1].offset_parameter is None
    first = collection.datasets[0].observation
    np.testing.assert_allclose(first.flux[0], 663.408e-6)
    np.testing.assert_allclose(first.uncertainty[0], 35.651e-6)
    np.testing.assert_allclose(first.wavelength_bin_edges[:2], [2.87, 2.88])
    assert first.metadata["doi"] == "10.3847/2041-8213/adaf22"


def test_l9859b_loader_can_assign_nrs2_offset() -> None:
    collection = load_bello_arufe2025_l9859b(
        DATA, miri_offset_parameter="nrs2_offset"
    )

    assert collection.datasets[0].offset_parameter is None
    assert collection.datasets[1].offset_parameter == "nrs2_offset"


def test_l9859b_checksum_ignores_terminal_blank_line_normalization(tmp_path) -> None:
    source = DATA / "L9859b_combined_spectrum_eureka.txt"
    normalized = source.read_bytes().rstrip(b"\r\n") + b"\n\n"
    (tmp_path / source.name).write_bytes(normalized)

    collection = load_bello_arufe2025_l9859b(tmp_path)

    assert collection.n_points == 218
