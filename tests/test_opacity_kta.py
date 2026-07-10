"""Tests for `.kta` opacity readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.core import RobertValidationError, SpectralGrid
from robert_exoplanets.opacity import (
    CorrelatedKOpacityProvider,
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


def test_read_kta_rejects_nonfinite_kcoefficients_by_default(tmp_path: Path) -> None:
    path, _expected = _write_synthetic_kta(
        tmp_path / "CO_incomplete_test.kta",
        nonfinite_index=(0, 1, 2, 1),
    )

    with pytest.raises(RobertValidationError, match="non-finite"):
        read_kta(path)


def test_read_kta_can_floor_nonfinite_kcoefficients_for_runtime(tmp_path: Path) -> None:
    missing_index = (0, 1, 2, 1)
    path, expected = _write_synthetic_kta(
        tmp_path / "CO_floor_test.kta",
        nonfinite_index=missing_index,
    )
    fill_value = 1.0e-300

    table = read_kta(path, nonfinite_policy="floor", nonfinite_fill_value=fill_value)

    expected = expected.copy()
    expected[missing_index] = fill_value
    np.testing.assert_allclose(table.kcoeff, expected)
    assert table.metadata["kcoeff_nonfinite_policy"] == "floor"
    assert table.metadata["kcoeff_nonfinite_replaced"] == "1"
    assert table.metadata["kcoeff_nan_replaced"] == "1"


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


def test_convert_kta_to_robert_archive_preserves_floor_policy_metadata(tmp_path: Path) -> None:
    missing_index = (0, 1, 2, 1)
    path, expected = _write_synthetic_kta(
        tmp_path / "NH3_incomplete_test.kta",
        nonfinite_index=missing_index,
    )
    archive_path = tmp_path / "NH3.robert-opacity"

    database = convert_kta_to_robert_archive(
        path,
        archive_path,
        species="NH3",
        archive="npy",
        overwrite=True,
        nonfinite_policy="floor",
    )

    expected = expected.copy()
    expected[missing_index] = 1.0e-300
    assert database.products[0].metadata["kcoeff_nonfinite_policy"] == "floor"
    assert database.products[0].metadata["kcoeff_nonfinite_replaced"] == "1"
    loaded = load_robert_npy_directory(archive_path)
    np.testing.assert_allclose(loaded.arrays["kcoeff"], expected)


def test_exok_bins_correlated_k_distributions_to_observation_bins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "numba-cache"
    cache_dir.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache_dir))
    import numba

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir))
    pytest.importorskip("exo_k")
    path, _expected = _write_synthetic_kta(tmp_path / "H2O_exok_test.kta")
    provider = CorrelatedKOpacityProvider.from_kta_paths({"H2O": path})
    wavenumber_centres = np.array([3500.0, 2500.0, 1500.0])
    wavenumber_edges = np.array([4000.0, 3000.0, 2000.0, 1000.0])
    target = SpectralGrid(
        values=10000.0 / wavenumber_centres,
        bin_edges=10000.0 / wavenumber_edges,
        unit="micron",
        role="observed",
    )

    binned = provider.bin_to_spectral_grid(target, num=50)
    table = binned.tables["H2O"]

    assert table.kcoeff.shape == (2, 3, 3, 2)
    np.testing.assert_allclose(table.wavenumber_cm_inverse, wavenumber_centres)
    assert table.metadata["spectral_preparation"] == "exo_k_bin_down"


def test_exok_replaces_zero_coefficients_before_correlated_k_binning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "numba-cache"
    cache_dir.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache_dir))
    import numba

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir))
    pytest.importorskip("exo_k")
    path, _expected = _write_synthetic_kta(
        tmp_path / "CO_exok_zero_test.kta",
        zero_index=(0, 0, 1, 0),
    )
    provider = CorrelatedKOpacityProvider.from_kta_paths({"CO": path})
    target = SpectralGrid(
        values=10000.0 / np.array([3500.0, 2500.0, 1500.0]),
        bin_edges=10000.0 / np.array([4000.0, 3000.0, 2000.0, 1000.0]),
        unit="micron",
        role="observed",
    )

    table = provider.bin_to_spectral_grid(target).tables["CO"]

    assert np.all(np.isfinite(table.kcoeff))
    assert np.all(table.kcoeff > 0.0)
    assert table.metadata["exo_k_remove_zeros"] == "true"
    assert table.metadata["exo_k_zeros_replaced"] == "1"
    assert float(table.metadata["exo_k_zero_floor"]) > 0.0


def test_provider_discovers_arbitrary_species_from_exomol_kta_directory(tmp_path: Path) -> None:
    _write_synthetic_kta(tmp_path / "TiO_custom_resolution.kta")
    _write_synthetic_kta(tmp_path / "VO_custom_resolution.kta")

    provider = CorrelatedKOpacityProvider.from_exomol_kta_directory(
        tmp_path,
        species=("TiO", "VO"),
    )

    assert provider.species == ("TiO", "VO")
    assert provider.tables["TiO"].metadata["source_path"].endswith("TiO_custom_resolution.kta")


def test_provider_requires_explicit_selection_for_duplicate_species_files(tmp_path: Path) -> None:
    _write_synthetic_kta(tmp_path / "H2O_R100.kta")
    _write_synthetic_kta(tmp_path / "H2O_R1000.kta")

    with pytest.raises(RobertValidationError, match="explicit selection"):
        CorrelatedKOpacityProvider.from_exomol_kta_directory(tmp_path)


def test_provider_loads_and_bins_exomol_hdf5_through_exok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "numba-cache"
    cache_dir.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache_dir))
    import numba

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir))
    exok = pytest.importorskip("exo_k")
    kta_path, _expected = _write_synthetic_kta(tmp_path / "TiO_source.kta")
    hdf5_path = tmp_path / "TiO_exomol.h5"
    native = exok.Ktable(filename=str(kta_path), mol="TiO", remove_zeros=False)
    native.write_hdf5(filename=str(hdf5_path), exomol_units=True)

    provider = CorrelatedKOpacityProvider.from_exok_paths({"TiO": hdf5_path})
    target = SpectralGrid(
        values=10000.0 / np.array([3500.0, 2500.0, 1500.0]),
        bin_edges=10000.0 / np.array([4000.0, 3000.0, 2000.0, 1000.0]),
        unit="micron",
        role="observed",
    )
    binned = provider.bin_to_spectral_grid(target)

    assert provider.tables["TiO"].metadata["source_format"] == "exo_k:h5"
    assert binned.tables["TiO"].kcoeff.shape == (2, 3, 3, 2)
    assert np.all(np.isfinite(binned.tables["TiO"].kcoeff))


def _write_synthetic_kta(
    path: Path,
    *,
    nonfinite_index: tuple[int, int, int, int] | None = None,
    zero_index: tuple[int, int, int, int] | None = None,
) -> tuple[Path, np.ndarray]:
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
    if nonfinite_index is not None:
        kcoeff[nonfinite_index] = np.nan
    if zero_index is not None:
        kcoeff[zero_index] = 0.0

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
