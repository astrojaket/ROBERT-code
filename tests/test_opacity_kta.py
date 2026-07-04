"""Tests for NEMESIS `.kta` opacity readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robert_exoplanets.opacity import (
    OpacityDataSource,
    OpacityStorageFormat,
    convert_kta_to_robert_archive,
    inspect_robert_npy_directory,
    load_robert_npy_directory,
    read_kta,
    read_kta_header,
)


def test_read_kta_header_reads_dimensions_and_coverage(tmp_path: Path) -> None:
    path, expected = _write_synthetic_kta(tmp_path / "H2O_test.kta")

    header = read_kta_header(path, checksum=True)

    assert header.n_pressure == 2
    assert header.n_temperature == 3
    assert header.n_wavelength == 4
    assert header.n_g == 2
    assert header.native_shape == expected.shape
    assert header.stored_shape == (4, 2, 3, 2)
    assert header.checksum_sha256
    np.testing.assert_allclose(header.pressure_bar, [1.0e-5, 1.0])
    np.testing.assert_allclose(header.temperature_K, [500.0, 1000.0, 1500.0])
    np.testing.assert_allclose(header.wavenumber_cm_inverse, [1000.0, 2000.0, 3000.0, 4000.0])
    assert header.spectral_coverage.min_value == 1000.0
    assert header.spectral_coverage.max_value == 4000.0
    assert header.grid_coverage.temperature_max == 1500.0


def test_read_kta_round_trips_kcoefficients(tmp_path: Path) -> None:
    path, expected = _write_synthetic_kta(tmp_path / "CO_test.kta")

    table = read_kta(path)

    np.testing.assert_allclose(table.kcoeff, expected)
    assert table.unit == "cm^2/molecule"
    assert table.header.molecule_id == 1


def test_convert_kta_to_robert_archive_writes_native_directory(tmp_path: Path) -> None:
    path, expected = _write_synthetic_kta(tmp_path / "CH4_test.kta")
    archive_path = tmp_path / "CH4.robert-opacity"

    database = convert_kta_to_robert_archive(
        path,
        archive_path,
        species="CH4",
        archive="npy",
        overwrite=True,
    )

    assert database.products[0].source == OpacityDataSource.ROBERT_ARCHIVE
    assert database.products[0].storage_format == OpacityStorageFormat.ROBERT_NPY_DIRECTORY
    inspected = inspect_robert_npy_directory(archive_path)
    assert inspected.products[0].species == ("CH4",)
    loaded = load_robert_npy_directory(archive_path)
    np.testing.assert_allclose(loaded.arrays["kcoeff"], expected)
    np.testing.assert_allclose(loaded.arrays["g_weights"], [0.4, 0.6])


def _write_synthetic_kta(path: Path) -> tuple[Path, np.ndarray]:
    pressure = np.array([1.0e-5, 1.0], dtype=np.float32)
    temperature = np.array([500.0, 1000.0, 1500.0], dtype=np.float32)
    wavenumber = np.array([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float64)
    wavelength = (10000.0 / wavenumber).astype(np.float32)
    g_samples = np.array([0.25, 0.75], dtype=np.float32)
    g_weights = np.array([0.4, 0.6], dtype=np.float32)
    kcoeff = (
        1.0e-30
        + np.arange(
            pressure.size * temperature.size * wavenumber.size * g_samples.size,
            dtype=np.float64,
        ).reshape(pressure.size, temperature.size, wavenumber.size, g_samples.size)
        * 1.0e-32
    )

    n_pressure, n_temperature, n_wavelength, n_g = kcoeff.shape
    irec0 = 11 + 2 * n_g + 2 + n_pressure + n_temperature + n_wavelength
    with path.open("wb") as handle:
        handle.write(np.int32(irec0).tobytes())
        handle.write(np.int32(n_wavelength).tobytes())
        handle.write(np.float32(wavelength[-1]).tobytes())
        handle.write(np.float32(-1.0).tobytes())
        handle.write(np.float32(0.0).tobytes())
        handle.write(np.int32(n_pressure).tobytes())
        handle.write(np.int32(n_temperature).tobytes())
        handle.write(np.int32(n_g).tobytes())
        handle.write(np.int32(1).tobytes())
        handle.write(np.int32(0).tobytes())
        handle.write(g_samples.tobytes())
        handle.write(g_weights.tobytes())
        handle.write(np.float32(0.0).tobytes())
        handle.write(np.float32(0.0).tobytes())
        handle.write(pressure.tobytes())
        handle.write(temperature.tobytes())
        handle.write(wavelength[::-1].astype(np.float32).tobytes())
        stored_kcoeff = kcoeff[:, :, ::-1, :].transpose(2, 0, 1, 3) * 1.0e20
        handle.write(stored_kcoeff.astype(np.float32).tobytes())
    return path, kcoeff
