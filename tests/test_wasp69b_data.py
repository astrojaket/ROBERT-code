"""Tests for the published WASP-69b emission spectrum."""

from pathlib import Path

from robert_exoplanets import load_schlawin2024_wasp69b

DATA = Path(__file__).resolve().parents[1] / "data" / "wasp69b_schlawin2024"


def test_wasp69b_loader_preserves_instrument_datasets() -> None:
    collection = load_schlawin2024_wasp69b(DATA)

    assert collection.names == ("f322w2", "avg", "f444w", "lrs")
    assert [dataset.observation.n_points for dataset in collection.datasets] == [144, 6, 102, 28]
    assert collection.n_points == 280
    assert collection.datasets[-1].offset_parameter == "miri_offset"
    assert all(dataset.observation.wavelength_bin_edges is not None for dataset in collection.datasets)


def test_wasp69b_loader_can_disable_miri_offset() -> None:
    collection = load_schlawin2024_wasp69b(DATA, miri_offset_parameter=None)

    assert collection.datasets[-1].offset_parameter is None
