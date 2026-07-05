"""Tests for opacity metadata and coverage checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robert_exoplanets.core import PressureGrid, RobertCoverageError, SpectralGrid
from robert_exoplanets.opacity import (
    GridCoverage,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityDatabase,
    OpacityMode,
    OpacityStorageFormat,
    SpectralCoverage,
    inspect_robert_npz_archive,
    pressure_values_in_unit,
    spectral_grid_values_in_unit,
)


def test_opacity_database_reports_correlated_k_coverage() -> None:
    pressure_grid = PressureGrid.logspace(1.0e-5, 1.0, n_layers=3)
    spectral_grid = SpectralGrid.from_array([1.0, 2.0, 3.0], unit="micron")
    product = OpacityDataProduct(
        species=("H2O",),
        mode=OpacityMode.CORRELATED_K,
        source=OpacityDataSource.EXOMOL_OP,
        storage_format=OpacityStorageFormat.KTA_BINARY,
        spectral_coverage=SpectralCoverage(3000.0, 11000.0, unit="cm^-1"),
        grid_coverage=GridCoverage(
            pressure_min=1.0e-6,
            pressure_max=10.0,
            temperature_min=500.0,
            temperature_max=2500.0,
        ),
        g_ordinates=20,
    )
    database = OpacityDatabase(products=(product,))

    report = database.coverage(
        species=("H2O",),
        mode=OpacityMode.CORRELATED_K,
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        temperature=np.array([1000.0, 1100.0, 1200.0]),
    )

    assert report.valid
    assert report.covered_species == ("H2O",)
    assert report.missing_species == ()
    assert report.product_ids == (product.product_id,)


def test_opacity_database_rejects_unknown_or_missing_coverage() -> None:
    product = OpacityDataProduct(
        species=("CO",),
        mode="correlated_k",
        source="exomol_op",
        storage_format="kta_binary",
    )
    database = OpacityDatabase(products=(product,))
    spectral_grid = SpectralGrid.from_array([1.0, 2.0], unit="micron")

    report = database.coverage(
        species=("CO", "H2O"),
        mode="correlated_k",
        spectral_grid=spectral_grid,
    )

    assert not report.valid
    assert report.missing_species == ("CO", "H2O")
    assert "spectral coverage is unknown" in report.reasons["CO"]
    assert "no opacity product" in report.reasons["H2O"]
    with pytest.raises(RobertCoverageError, match="opacity coverage is incomplete"):
        database.validate_coverage(
            species=("CO",),
            mode="correlated_k",
            spectral_grid=spectral_grid,
        )


def test_opacity_database_serializes_to_manifest_mapping(tmp_path) -> None:
    product = OpacityDataProduct(
        species=("CH4",),
        mode="opacity_sampling",
        source="exomol",
        storage_format="exomol_cross_section",
        spectral_coverage=SpectralCoverage(0.5, 5.0, unit="micron", n_points=100),
        grid_coverage=GridCoverage(temperature_min=300.0, temperature_max=2000.0),
        compression="npz",
        native_shape=(4, 100),
    )
    database = OpacityDatabase(products=(product,), name="synthetic")
    restored = OpacityDatabase.from_mapping(database.to_mapping())

    assert restored.name == "synthetic"
    assert restored.products[0].species == ("CH4",)
    assert restored.products[0].mode == OpacityMode.OPACITY_SAMPLING

    archive_path = tmp_path / "opacity_archive.npz"
    np.savez_compressed(archive_path, manifest_json=json.dumps(database.to_mapping()))

    archive_database = inspect_robert_npz_archive(archive_path)

    assert archive_database.products[0].storage_format == OpacityStorageFormat.EXOMOL_CROSS_SECTION


def test_spectral_and_pressure_unit_conversions() -> None:
    spectral_grid = SpectralGrid.from_array([1.0, 2.0], unit="micron")
    pressure_grid = PressureGrid.logspace(1.0e-3, 1.0, n_layers=2, unit="bar")

    np.testing.assert_allclose(spectral_grid_values_in_unit(spectral_grid, "cm^-1"), [10000.0, 5000.0])
    np.testing.assert_allclose(
        pressure_values_in_unit(pressure_grid.centers, "bar", "mbar"),
        pressure_grid.centers * 1000.0,
    )
