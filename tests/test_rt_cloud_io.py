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
    load_picaso_cloud_optical_properties,
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


def test_picaso_cloud_reader_loads_index_table_with_pressure_and_wave_files(tmp_path) -> None:
    cloud_path = tmp_path / "jupiter_like.cld"
    cloud_path.write_text(
        "\n".join(
            [
                "lvl wv opd g0 w0 sigma",
                "1 1 0.1 0.2 0.5 0.0",
                "1 2 0.2 0.3 0.6 0.0",
                "2 1 0.3 0.4 0.7 0.0",
                "2 2 0.4 0.5 0.8 0.0",
            ]
        ),
        encoding="utf-8",
    )
    pressure_path = tmp_path / "jupiter.pt"
    pressure_path.write_text(
        "\n".join(
            [
                "pressure temperature",
                "1e-5 900.0",
                "1e-3 1000.0",
                "1e-1 1100.0",
            ]
        ),
        encoding="utf-8",
    )
    wave_grid_path = tmp_path / "wave_EGP.dat"
    wave_grid_path.write_text(
        "\n".join(
            [
                "i micron. wavenumber idum",
                "1 2.0 5000.0 0.0",
                "2 1.0 10000.0 0.0",
            ]
        ),
        encoding="utf-8",
    )

    cloud = load_picaso_cloud_optical_properties(
        cloud_path,
        pressure_path=pressure_path,
        wave_grid_path=wave_grid_path,
    )

    np.testing.assert_allclose(cloud.pressure_grid.edges, [1.0e-5, 1.0e-3, 1.0e-1])
    np.testing.assert_allclose(cloud.pressure_grid.centers, [1.0e-4, 1.0e-2])
    np.testing.assert_allclose(cloud.spectral_grid.values, [1.0, 2.0])
    np.testing.assert_allclose(cloud.extinction_tau, [[0.2, 0.1], [0.4, 0.3]])
    np.testing.assert_allclose(cloud.single_scattering_albedo, [[0.6, 0.5], [0.8, 0.7]])
    np.testing.assert_allclose(cloud.asymmetry_factor, [[0.3, 0.2], [0.5, 0.4]])
    assert cloud.metadata["source_format"] == "picaso_cld_cloud_optical_properties"
    assert cloud.metadata["layer_coordinate"] == "index"


def test_picaso_cloud_reader_loads_physical_pressure_and_wavenumber_columns(tmp_path) -> None:
    cloud_path = tmp_path / "virga_like.cld"
    cloud_path.write_text(
        "\n".join(
            [
                "pressure wavenumber opd w0 g0",
                "1e-4 5000.0 0.1 0.5 0.2",
                "1e-4 10000.0 0.2 0.6 0.3",
                "1e-2 5000.0 0.3 0.7 0.4",
                "1e-2 10000.0 0.4 0.8 0.5",
            ]
        ),
        encoding="utf-8",
    )

    cloud = load_picaso_cloud_optical_properties(cloud_path)

    np.testing.assert_allclose(cloud.pressure_grid.centers, [1.0e-4, 1.0e-2])
    np.testing.assert_allclose(cloud.spectral_grid.values, [1.0, 2.0])
    np.testing.assert_allclose(cloud.extinction_tau, [[0.2, 0.1], [0.4, 0.3]])
    assert cloud.metadata["spectral_kind"] == "wavenumber"
    assert cloud.metadata["layer_coordinate"] == "pressure"


def test_picaso_cloud_reader_requires_coordinates_for_index_files(tmp_path) -> None:
    cloud_path = tmp_path / "index_only.cld"
    cloud_path.write_text(
        "\n".join(
            [
                "lvl wv opd g0 w0",
                "1 1 0.1 0.2 0.5",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RobertDataError, match="layer indices"):
        load_picaso_cloud_optical_properties(cloud_path)


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
