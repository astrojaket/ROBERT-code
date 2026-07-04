"""Tests for ROBERT-native opacity archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.opacity import (
    GridCoverage,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityDatabase,
    OpacityMode,
    OpacityStorageFormat,
    RobertOpacityArchive,
    SpectralCoverage,
    inspect_robert_npy_directory,
    inspect_robert_npz_archive,
    load_robert_npy_directory,
    load_robert_npz_archive,
    write_robert_npy_directory,
    write_robert_npz_archive,
)


def test_robert_npy_directory_round_trips_metadata_arrays_and_mmap(tmp_path: Path) -> None:
    database = _tiny_database()
    arrays = {
        "kcoeff": np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5),
        "g_weights": np.linspace(0.1, 1.0, 5),
    }
    archive_path = tmp_path / "H2O.robert-opacity"

    written = write_robert_npy_directory(
        archive_path,
        database=database,
        arrays=arrays,
        metadata={"purpose": "test"},
    )

    assert isinstance(written, RobertOpacityArchive)
    assert (archive_path / "manifest.json").is_file()
    assert (archive_path / "kcoeff.npy").is_file()
    inspected = inspect_robert_npy_directory(archive_path)
    assert inspected.products[0].source == OpacityDataSource.ROBERT_ARCHIVE
    assert inspected.products[0].storage_format == OpacityStorageFormat.ROBERT_NPY_DIRECTORY
    assert inspected.products[0].metadata["converted_from_source"] == "exomol_op"

    loaded = load_robert_npy_directory(archive_path)
    np.testing.assert_allclose(loaded.arrays["kcoeff"], arrays["kcoeff"])
    assert loaded.metadata["purpose"] == "test"

    mapped = load_robert_npy_directory(archive_path, mmap_mode="r")
    assert isinstance(mapped.arrays["kcoeff"], np.memmap)
    np.testing.assert_allclose(mapped.arrays["g_weights"], arrays["g_weights"])


def test_robert_npy_directory_requires_overwrite_for_existing_archive(tmp_path: Path) -> None:
    database = _tiny_database()
    archive_path = tmp_path / "archive"
    write_robert_npy_directory(archive_path, database=database, arrays={"opacity": np.ones((2, 2))})

    with pytest.raises(RobertDataError, match="already exist"):
        write_robert_npy_directory(archive_path, database=database, arrays={"opacity": np.ones((2, 2))})

    write_robert_npy_directory(
        archive_path,
        database=database,
        arrays={"opacity": np.full((2, 2), 2.0)},
        overwrite=True,
    )
    loaded = load_robert_npy_directory(archive_path)
    np.testing.assert_allclose(loaded.arrays["opacity"], np.full((2, 2), 2.0))


def test_robert_npz_archive_round_trips_compact_exchange_file(tmp_path: Path) -> None:
    database = _tiny_database()
    arrays = {"opacity": np.arange(12, dtype=np.float64).reshape(3, 4)}
    archive_path = tmp_path / "opacity_archive.npz"

    write_robert_npz_archive(
        archive_path,
        database=database,
        arrays=arrays,
        compressed=True,
        metadata={"purpose": "exchange"},
    )

    inspected = inspect_robert_npz_archive(archive_path)
    assert inspected.products[0].storage_format == OpacityStorageFormat.ROBERT_NPZ
    assert inspected.products[0].compression == "zip_deflate"

    loaded = load_robert_npz_archive(archive_path)
    np.testing.assert_allclose(loaded.arrays["opacity"], arrays["opacity"])
    assert loaded.metadata["purpose"] == "exchange"


def _tiny_database() -> OpacityDatabase:
    product = OpacityDataProduct(
        species=("H2O",),
        mode=OpacityMode.CORRELATED_K,
        source=OpacityDataSource.EXOMOL_OP,
        storage_format=OpacityStorageFormat.NEMESIS_KTA,
        spectral_coverage=SpectralCoverage(1000.0, 5000.0, unit="cm^-1", n_points=4),
        grid_coverage=GridCoverage(
            pressure_min=1.0e-6,
            pressure_max=1.0,
            temperature_min=300.0,
            temperature_max=2000.0,
            n_pressure=2,
            n_temperature=3,
        ),
        g_ordinates=5,
        native_shape=(2, 3, 4, 5),
    )
    return OpacityDatabase(products=(product,), name="tiny")
