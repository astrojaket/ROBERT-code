"""Tests for native-grid correlated-k opacity evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    PressureGrid,
    SpectralGrid,
)
from robert_exoplanets.core import RobertCoverageError, RobertValidationError
from robert_exoplanets.opacity import (
    GridCoverage,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityDatabase,
    OpacityMode,
    OpacityStorageFormat,
    SpectralCoverage,
    write_robert_npy_directory,
)


def test_correlated_k_table_loads_petitradtrans_hdf(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "water.ktable.petitRADTRANS.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("p", data=[1.0e-5, 1.0])
        handle.create_dataset("t", data=[500.0, 1000.0])
        handle.create_dataset("bin_centers", data=[1000.0, 2000.0])
        handle.create_dataset("samples", data=[0.25, 0.75])
        handle.create_dataset("weights", data=[0.4, 0.6])
        coefficients = handle.create_dataset("kcoeff", data=np.ones((2, 2, 2, 2)))
        coefficients.attrs["units"] = "cm^2/molecule"
        handle.create_dataset("DOI", data=[b"10.0000/example"])
        handle.create_dataset("method", data=[b"petit_samples"])

    table = CorrelatedKTable.from_petitradtrans_hdf(path, species="H2O")

    assert table.kcoeff.shape == (2, 2, 2, 2)
    assert table.unit == "cm^2/molecule"
    assert table.metadata["source_format"] == "petitradtrans_hdf5"
    assert table.metadata["doi"] == "10.0000/example"
    np.testing.assert_allclose(table.wavelength_micron, [10.0, 5.0])


def test_correlated_k_table_correlates_exomol_cross_sections_in_bins(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "H2O.h5"
    wavenumber = np.linspace(3000.0, 12000.0, 128)
    native = np.geomspace(1.0e-30, 1.0e-20, 2 * 2 * 128).reshape(2, 2, 128)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("p", data=[1.0e-5, 1.0])
        handle.create_dataset("t", data=[1000.0, 1200.0])
        handle.create_dataset("bin_edges", data=wavenumber)
        cross_sections = handle.create_dataset("xsecarr", data=native)
        cross_sections.attrs["units"] = "cm^2/molecule"
        handle.create_dataset("DOI", data=[b"test-doi"])
        handle.create_dataset("key_iso_ll", data=[b"test-line-list"])
    grid = SpectralGrid(
        values=np.array([1.0, 2.0]),
        bin_edges=np.array([0.85, 1.3, 2.5]),
        unit="micron",
        role="observed",
    )

    table = CorrelatedKTable.from_exomol_cross_section_hdf(
        path,
        species="H2O",
        spectral_grid=grid,
        g_points=8,
    )

    assert table.kcoeff.shape == (2, 2, 2, 8)
    assert table.metadata["doi"] == "test-doi"
    assert table.metadata["line_list"] == "test-line-list"
    assert table.metadata["g_points"] == "8"
    assert len(table.metadata["checksum_sha256"]) == 64
    assert np.all(np.diff(table.kcoeff, axis=-1) >= 0.0)


def test_exomol_cross_section_correlation_requires_bin_edges(tmp_path: Path) -> None:
    grid = SpectralGrid.from_array([1.0, 2.0], unit="micron", role="observed")

    with pytest.raises(RobertValidationError, match="bin edges"):
        CorrelatedKTable.from_exomol_cross_section_hdf(
            tmp_path / "missing.h5",
            species="H2O",
            spectral_grid=grid,
        )


def test_correlated_k_provider_evaluates_exact_native_grid_points() -> None:
    table = _tiny_table("H2O")
    provider = CorrelatedKOpacityProvider({"H2O": table})
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array(
        [1000.0, 3000.0], unit="cm^-1", role="opacity"
    )
    atmosphere = _atmosphere(pressure_grid, temperature=[500.0, 1500.0])
    prepared = provider.prepare(spectral_grid, pressure_grid, species=("H2O",))

    evaluated = provider.evaluate(atmosphere, prepared)

    assert evaluated.kcoeff.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(evaluated.kcoeff[0, 0, 0], table.kcoeff[0, 0, 0])
    np.testing.assert_allclose(evaluated.kcoeff[0, 0, 1], table.kcoeff[0, 0, 2])
    np.testing.assert_allclose(evaluated.kcoeff[0, 1, 0], table.kcoeff[1, 2, 0])
    np.testing.assert_allclose(evaluated.kcoeff[0, 1, 1], table.kcoeff[1, 2, 2])
    np.testing.assert_allclose(prepared.g_weights, [0.4, 0.6])


def test_correlated_k_provider_supports_wavelength_spectral_grid() -> None:
    table = _tiny_table("H2O")
    provider = CorrelatedKOpacityProvider({"H2O": table})
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array([10.0, 5.0], unit="micron", role="opacity")
    atmosphere = _atmosphere(pressure_grid, temperature=[500.0, 1000.0])
    prepared = provider.prepare(spectral_grid, pressure_grid, species=("H2O",))

    evaluated = provider.evaluate(atmosphere, prepared)

    np.testing.assert_allclose(evaluated.kcoeff[0, 0, 0], table.kcoeff[0, 0, 0])
    np.testing.assert_allclose(evaluated.kcoeff[0, 1, 1], table.kcoeff[1, 1, 1])


def test_correlated_k_provider_rejects_off_grid_temperature() -> None:
    table = _tiny_table("H2O")
    provider = CorrelatedKOpacityProvider({"H2O": table})
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array(
        [1000.0, 2000.0], unit="cm^-1", role="opacity"
    )
    atmosphere = _atmosphere(pressure_grid, temperature=[500.0, 1250.0])
    prepared = provider.prepare(spectral_grid, pressure_grid, species=("H2O",))

    report = provider.coverage(atmosphere, prepared)

    assert not report.valid
    assert "temperature value 1250" in report.reasons["H2O"]
    with pytest.raises(RobertCoverageError, match="temperature value 1250"):
        provider.evaluate(atmosphere, prepared)


def test_correlated_k_provider_rejects_off_grid_spectral_grid() -> None:
    table = _tiny_table("H2O")
    provider = CorrelatedKOpacityProvider({"H2O": table})
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array(
        [1100.0, 2000.0], unit="cm^-1", role="opacity"
    )

    with pytest.raises(RobertCoverageError, match="spectral value 1100"):
        provider.prepare(spectral_grid, pressure_grid, species=("H2O",))


def test_correlated_k_table_normalizes_close_float32_g_weight_sum() -> None:
    g_weights = np.array([0.2, 0.3, 0.49999996])
    table = CorrelatedKTable(
        species="H2O",
        pressure_bar=np.array([1.0e-3]),
        temperature_K=np.array([1000.0]),
        wavenumber_cm_inverse=np.array([2000.0]),
        g_samples=np.array([0.2, 0.5, 0.8]),
        g_weights=g_weights,
        kcoeff=np.ones((1, 1, 1, 3)) * 1.0e-30,
    )

    np.testing.assert_allclose(table.g_weights, g_weights / np.sum(g_weights))
    np.testing.assert_allclose(np.sum(table.g_weights), 1.0)


def test_correlated_k_provider_interpolates_log_pressure_temperature_log_k() -> None:
    table = _interpolation_table("H2O")
    provider = CorrelatedKOpacityProvider(
        {"H2O": table},
        interpolation="log_pressure_temperature_log_k",
    )
    pressure = 1.0e-3
    temperature = 1000.0
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-4, 1.0e-2]),
        centers=np.array([pressure]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([2000.0], unit="cm^-1", role="opacity")
    atmosphere = _atmosphere(pressure_grid, temperature=[temperature])
    prepared = provider.prepare(spectral_grid, pressure_grid, species=("H2O",))

    evaluated = provider.evaluate(atmosphere, prepared)

    expected = np.exp(_linear_log_k(np.log10(pressure), temperature))
    assert evaluated.kcoeff.shape == (1, 1, 1, 1)
    assert prepared.metadata["interpolation"] == "log_pressure_temperature_log_k"
    assert evaluated.metadata["interpolation"] == "log_pressure_temperature_log_k"
    np.testing.assert_array_equal(prepared.spectral_indices["H2O"], [0])
    np.testing.assert_allclose(evaluated.kcoeff[0, 0, 0, 0], expected, rtol=1.0e-12)


def test_correlated_k_cached_log_coefficients_match_uncached_interpolation() -> None:
    table = _interpolation_table("H2O")
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-4, 1.0e-2]),
        centers=np.array([1.0e-3]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([2000.0], unit="cm^-1", role="opacity")
    atmosphere = _atmosphere(pressure_grid, temperature=[1000.0])
    cached = CorrelatedKOpacityProvider(
        {"H2O": table},
        interpolation="log_pressure_temperature_log_k",
    )
    uncached = CorrelatedKOpacityProvider(
        {"H2O": table},
        interpolation="log_pressure_temperature_log_k",
        cache_log_kcoeff=False,
    )
    cached_prepared = cached.prepare(spectral_grid, pressure_grid, species=("H2O",))
    uncached_prepared = uncached.prepare(spectral_grid, pressure_grid, species=("H2O",))

    cached_values = cached.evaluate(atmosphere, cached_prepared).kcoeff
    uncached_values = uncached.evaluate(atmosphere, uncached_prepared).kcoeff

    assert cached_prepared.metadata["log_kcoeff_cache"] == "enabled"
    assert uncached_prepared.metadata["log_kcoeff_cache"] == "disabled"
    np.testing.assert_array_equal(cached_values, uncached_values)


def test_correlated_k_interpolation_rejects_out_of_range_temperature() -> None:
    table = _interpolation_table("H2O")
    provider = CorrelatedKOpacityProvider(
        {"H2O": table},
        interpolation="log_pressure_temperature_log_k",
    )
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-4, 1.0e-2]),
        centers=np.array([1.0e-3]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([2000.0], unit="cm^-1", role="opacity")
    atmosphere = _atmosphere(pressure_grid, temperature=[1800.0])
    prepared = provider.prepare(spectral_grid, pressure_grid, species=("H2O",))

    report = provider.coverage(atmosphere, prepared)

    assert not report.valid
    assert "temperature values are outside" in report.reasons["H2O"]
    with pytest.raises(RobertCoverageError, match="temperature values are outside"):
        provider.evaluate(atmosphere, prepared)


def test_correlated_k_clip_policy_matches_nemesispy_boundary_clamping() -> None:
    table = _interpolation_table("H2O")
    provider = CorrelatedKOpacityProvider(
        {"H2O": table},
        interpolation="log_pressure_temperature_log_k_clip",
    )
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-7, 1.0e-5]),
        centers=np.array([1.0e-6]),
        unit="bar",
    )
    spectral_grid = SpectralGrid.from_array([2000.0], unit="cm^-1", role="opacity")
    atmosphere = _atmosphere(pressure_grid, temperature=[1800.0])
    prepared = provider.prepare(spectral_grid, pressure_grid, species=("H2O",))

    evaluated = provider.evaluate(atmosphere, prepared)

    expected = np.exp(
        _linear_log_k(
            np.log10(np.min(table.pressure_bar)),
            np.max(table.temperature_K),
        )
    )
    assert provider.coverage(atmosphere, prepared).valid
    assert prepared.metadata["interpolation"] == "log_pressure_temperature_log_k_clip"
    np.testing.assert_allclose(evaluated.kcoeff[0, 0, 0, 0], expected, rtol=1.0e-12)


def test_correlated_k_cache_key_includes_interpolation_policy() -> None:
    table = _tiny_table("H2O")
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array(
        [1000.0, 2000.0], unit="cm^-1", role="opacity"
    )

    exact = CorrelatedKOpacityProvider({"H2O": table}, interpolation="exact").prepare(
        spectral_grid,
        pressure_grid,
        species=("H2O",),
    )
    interpolated = CorrelatedKOpacityProvider(
        {"H2O": table},
        interpolation="log_pressure_temperature_log_k",
    ).prepare(
        spectral_grid,
        pressure_grid,
        species=("H2O",),
    )

    assert exact.cache_key != interpolated.cache_key


def test_correlated_k_table_loads_from_robert_archive(tmp_path: Path) -> None:
    source = _tiny_table("CO")
    archive_path = tmp_path / "CO.robert-opacity"
    write_robert_npy_directory(
        archive_path,
        database=_tiny_database("CO", source),
        arrays={
            "kcoeff": source.kcoeff,
            "pressure_bar": source.pressure_bar,
            "temperature_K": source.temperature_K,
            "wavenumber_cm-1": source.wavenumber_cm_inverse,
            "wavelength_micron": source.wavelength_micron,
            "g_samples": source.g_samples,
            "g_weights": source.g_weights,
        },
    )

    loaded = CorrelatedKTable.from_robert_archive(archive_path, species="CO")

    assert loaded.species == "CO"
    np.testing.assert_allclose(loaded.kcoeff, source.kcoeff)
    np.testing.assert_allclose(loaded.pressure_bar, source.pressure_bar)


def _tiny_table(species: str) -> CorrelatedKTable:
    pressure = np.array([1.0e-5, 1.0])
    temperature = np.array([500.0, 1000.0, 1500.0])
    wavenumber = np.array([1000.0, 2000.0, 3000.0])
    g_samples = np.array([0.25, 0.75])
    g_weights = np.array([0.4, 0.6])
    kcoeff = (
        1.0e-30
        + np.arange(
            pressure.size * temperature.size * wavenumber.size * g_samples.size,
            dtype=float,
        ).reshape(pressure.size, temperature.size, wavenumber.size, g_samples.size)
        * 1.0e-32
    )
    return CorrelatedKTable(
        species=species,
        pressure_bar=pressure,
        temperature_K=temperature,
        wavenumber_cm_inverse=wavenumber,
        g_samples=g_samples,
        g_weights=g_weights,
        kcoeff=kcoeff,
    )


def _interpolation_table(species: str) -> CorrelatedKTable:
    pressure = np.array([1.0e-5, 1.0])
    temperature = np.array([500.0, 1500.0])
    wavenumber = np.array([2000.0])
    g_samples = np.array([0.5])
    g_weights = np.array([1.0])
    log_k = _linear_log_k(
        np.log10(pressure)[:, None, None, None],
        temperature[None, :, None, None],
    )
    return CorrelatedKTable(
        species=species,
        pressure_bar=pressure,
        temperature_K=temperature,
        wavenumber_cm_inverse=wavenumber,
        g_samples=g_samples,
        g_weights=g_weights,
        kcoeff=np.exp(log_k),
    )


def _linear_log_k(
    log10_pressure: np.ndarray | float, temperature: np.ndarray | float
) -> np.ndarray | float:
    return -72.0 + 0.35 * log10_pressure + 1.0e-3 * temperature


def _pressure_grid() -> PressureGrid:
    return PressureGrid(
        edges=np.array([1.0e-6, 1.0e-4, 10.0]),
        centers=np.array([1.0e-5, 1.0]),
        unit="bar",
    )


def _atmosphere(
    pressure_grid: PressureGrid, temperature: list[float]
) -> AtmosphereState:
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.asarray(temperature),
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-3)},
        mean_molecular_weight=np.full(pressure_grid.n_layers, 2.3),
    )


def _tiny_database(species: str, table: CorrelatedKTable) -> OpacityDatabase:
    product = OpacityDataProduct(
        species=(species,),
        mode=OpacityMode.CORRELATED_K,
        source=OpacityDataSource.EXOMOL_OP,
        storage_format=OpacityStorageFormat.KTA_BINARY,
        spectral_coverage=SpectralCoverage(1000.0, 3000.0, unit="cm^-1", n_points=3),
        grid_coverage=GridCoverage(
            pressure_min=float(np.min(table.pressure_bar)),
            pressure_max=float(np.max(table.pressure_bar)),
            temperature_min=float(np.min(table.temperature_K)),
            temperature_max=float(np.max(table.temperature_K)),
            n_pressure=table.pressure_bar.size,
            n_temperature=table.temperature_K.size,
        ),
        g_ordinates=table.g_weights.size,
        native_shape=table.kcoeff.shape,
    )
    return OpacityDatabase(products=(product,), name=f"{species}-tiny")
