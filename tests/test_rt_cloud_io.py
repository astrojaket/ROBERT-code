"""Tests for cloud optical-property interchange readers."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from robert_exoplanets import (
    CloudOpticalProperties,
    PressureGrid,
    SpectralGrid,
    compare_cloud_optical_properties,
    load_cloud_optical_properties_csv,
    load_cloud_optical_properties_npz,
    write_cloud_optical_properties_npz,
)
from robert_exoplanets.core import RobertDataError, RobertValidationError


def test_cloud_npz_roundtrip_preserves_optical_properties(tmp_path) -> None:
    cloud = _cloud()
    path = tmp_path / "cloud_properties.npz"

    write_cloud_optical_properties_npz(cloud, path)
    loaded = load_cloud_optical_properties_npz(path, name="loaded cloud")

    comparison = compare_cloud_optical_properties(cloud, loaded)
    assert comparison.max_abs_extinction_tau == pytest.approx(0.0)
    assert comparison.max_abs_single_scattering_albedo == pytest.approx(0.0)
    assert comparison.max_abs_asymmetry_factor == pytest.approx(0.0)
    assert loaded.metadata["source_format"] == "npz_cloud_optical_properties"


def test_cloud_npz_reader_accepts_picaso_virga_aliases(tmp_path) -> None:
    path = tmp_path / "virga_like.npz"
    np.savez(
        path,
        pressure_bar=np.array([1.0e-4, 1.0e-2]),
        wavelength_micron=np.array([1.0, 2.0]),
        tau_ext=np.array([[0.1, 0.2], [0.3, 0.4]]),
        omega0=np.array([[0.5, 0.6], [0.7, 0.8]]),
        g=np.array([[0.0, 0.1], [0.2, 0.3]]),
    )

    cloud = load_cloud_optical_properties_npz(path)

    np.testing.assert_allclose(cloud.extinction_tau, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(cloud.single_scattering_albedo, [[0.5, 0.6], [0.7, 0.8]])
    np.testing.assert_allclose(cloud.asymmetry_factor, [[0.0, 0.1], [0.2, 0.3]])


def test_cloud_csv_reader_loads_long_table(tmp_path) -> None:
    path = tmp_path / "cloud_properties.csv"
    rows = [
        {"pressure_bar": 1.0e-4, "wavelength_micron": 1.0, "tau_ext": 0.1, "omega0": 0.5, "g": 0.0},
        {"pressure_bar": 1.0e-4, "wavelength_micron": 2.0, "tau_ext": 0.2, "omega0": 0.6, "g": 0.1},
        {"pressure_bar": 1.0e-2, "wavelength_micron": 1.0, "tau_ext": 0.3, "omega0": 0.7, "g": 0.2},
        {"pressure_bar": 1.0e-2, "wavelength_micron": 2.0, "tau_ext": 0.4, "omega0": 0.8, "g": 0.3},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cloud = load_cloud_optical_properties_csv(path)

    np.testing.assert_allclose(cloud.extinction_tau, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(cloud.single_scattering_albedo, [[0.5, 0.6], [0.7, 0.8]])
    np.testing.assert_allclose(cloud.asymmetry_factor, [[0.0, 0.1], [0.2, 0.3]])
    assert cloud.metadata["source_format"] == "csv_cloud_optical_properties"


def test_cloud_csv_reader_rejects_missing_cells(tmp_path) -> None:
    path = tmp_path / "incomplete.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("pressure_bar,wavelength_micron,tau_ext\n")
        handle.write("1e-4,1.0,0.1\n")
        handle.write("1e-2,2.0,0.4\n")

    with pytest.raises(RobertDataError, match="missing pressure/wavelength"):
        load_cloud_optical_properties_csv(path)


def test_cloud_comparison_rejects_grid_mismatch() -> None:
    cloud = _cloud()
    other = CloudOpticalProperties(
        name="other",
        extinction_tau=np.ones((2, 2)),
        pressure_grid=cloud.pressure_grid,
        spectral_grid=SpectralGrid.from_array([1.0, 3.0], unit="micron", role="opacity"),
    )

    with pytest.raises(RobertValidationError, match="spectral grids"):
        compare_cloud_optical_properties(cloud, other)


def _cloud() -> CloudOpticalProperties:
    pressure_grid = PressureGrid.logspace(1.0e-5, 1.0e-1, 2, unit="bar")
    spectral_grid = SpectralGrid.from_array([1.0, 2.0], unit="micron", role="opacity")
    return CloudOpticalProperties(
        name="roundtrip cloud",
        extinction_tau=np.array([[0.1, 0.2], [0.3, 0.4]]),
        pressure_grid=pressure_grid,
        spectral_grid=spectral_grid,
        single_scattering_albedo=np.array([[0.5, 0.6], [0.7, 0.8]]),
        asymmetry_factor=np.array([[0.0, 0.1], [0.2, 0.3]]),
    )
