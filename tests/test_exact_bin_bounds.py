"""Exact per-bin spectral bounds with gaps and overlaps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    EmissionForwardModel,
    EmissionModelConfig,
    FreeChemistry,
    IsothermalTemperatureProfile,
    Observation,
    ParameterizedTransmissionFactoryConfig,
    ParameterizedTransmissionModelConfig,
    Planet,
    PressureGrid,
    SpectralGrid,
    Spectrum,
    Star,
    TopHatObservationResponse,
    build_parameterized_transmission_model,
    load_observation_npz,
    load_observation_table,
    save_observation_npz,
)
from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.instruments import PixelIntegrationResponse
from robert_exoplanets.opacity import (
    spectral_grid_bin_bounds_in_unit,
    spectral_grid_values_in_unit,
)
from robert_exoplanets.opacity.exok import (
    _ensure_writable_numba_cache,
    bin_correlated_k_table_with_exok,
)
from robert_exoplanets.rt.extinction import _hminus_bound_free_bin_cross_section


def _bounds_grid() -> SpectralGrid:
    return SpectralGrid(
        values=np.array([1.2, 1.8, 3.2]),
        bin_edges=np.array([[0.8, 1.6], [1.4, 2.2], [2.8, 3.6]]),
        unit="micron",
        role="observed",
    )


def _observation(grid: SpectralGrid) -> Observation:
    return Observation(
        wavelength=grid.values,
        wavelength_unit=grid.unit,
        wavelength_bin_edges=grid.bin_edges,
        flux=np.array([1.0, 1.1, 1.2]),
        uncertainty=np.full(grid.size, 0.01),
        flux_unit="transit_depth",
        observable="transit_depth",
    )


def _provider(grid: SpectralGrid) -> CorrelatedKOpacityProvider:
    pressure = np.array([0.3, 3.0])
    temperature = np.array([700.0, 1300.0])
    tables = {}
    for index, species in enumerate(("H2O", "CO"), start=1):
        kcoeff = np.full((2, 2, grid.size, 1), index * 1.0e-24)
        tables[species] = CorrelatedKTable(
            species=species,
            pressure_bar=pressure,
            temperature_K=temperature,
            wavenumber_cm_inverse=spectral_grid_values_in_unit(grid, "cm^-1"),
            g_samples=np.array([0.5]),
            g_weights=np.array([1.0]),
            kcoeff=kcoeff,
        )
    return CorrelatedKOpacityProvider(
        tables=tables,
        interpolation="log_pressure_temperature_log_k",
    )


def _write_small_kta(path: Path) -> Path:
    pressure = np.array([1.0e-5, 1.0], dtype=np.float32)
    temperature = np.array([500.0, 1000.0], dtype=np.float32)
    wavenumber = np.arange(1000.0, 18000.0, 1000.0)
    wavelength = (10000.0 / wavenumber).astype(np.float32)
    g_samples = np.array([0.25, 0.75], dtype=np.float32)
    g_weights = np.array([0.4, 0.6], dtype=np.float32)
    coefficient = (
        1.0e-30
        + np.arange(
            pressure.size * temperature.size * wavenumber.size * g_samples.size,
            dtype=float,
        ).reshape(pressure.size, temperature.size, wavenumber.size, g_samples.size)
        * 1.0e-32
    )
    n_pressure, n_temperature, n_wavelength, n_g = coefficient.shape
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
        stored = coefficient[:, :, ::-1, :].transpose(2, 0, 1, 3) * 1.0e20
        handle.write(stored.astype(np.float32).tobytes())
    return path


def _write_small_regular_kta(path: Path) -> Path:
    pressure = np.array([1.0e-5, 1.0], dtype=np.float32)
    temperature = np.array([500.0, 1000.0], dtype=np.float32)
    wavelength_min = np.float32(0.5)
    wavelength_step = np.float32(0.25)
    n_wavelength = 24
    g_samples = np.array([0.25, 0.75], dtype=np.float32)
    g_weights = np.array([0.4, 0.6], dtype=np.float32)
    coefficient = (
        1.0e-30
        + np.arange(
            pressure.size * temperature.size * n_wavelength * g_samples.size,
            dtype=float,
        ).reshape(pressure.size, temperature.size, n_wavelength, g_samples.size)
        * 1.0e-32
    )
    n_pressure, n_temperature, _, n_g = coefficient.shape
    # Regular KTA files store the wavelength minimum and step in the header,
    # with no explicit wavelength array. Data start immediately after it.
    irec0 = 13 + 2 * n_g + n_pressure + n_temperature
    with path.open("wb") as handle:
        handle.write(np.int32(irec0).tobytes())
        handle.write(np.int32(n_wavelength).tobytes())
        handle.write(wavelength_min.tobytes())
        handle.write(wavelength_step.tobytes())
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
        stored = coefficient[:, :, ::-1, :].transpose(2, 0, 1, 3) * 1.0e20
        handle.write(stored.astype(np.float32).tobytes())
    return path


def test_observation_npz_round_trip_preserves_per_bin_pairs(tmp_path: Path) -> None:
    observation = _observation(_bounds_grid())
    path = save_observation_npz(observation, tmp_path / "observation.npz")

    restored = load_observation_npz(path)

    np.testing.assert_array_equal(restored.wavelength, observation.wavelength)
    np.testing.assert_array_equal(
        restored.wavelength_bin_edges,
        observation.wavelength_bin_edges,
    )
    assert restored.wavelength_bin_edges.shape == (3, 2)
    assert restored.spectral_grid.bin_bounds is not None


def test_text_table_import_keeps_published_gaps_and_overlaps(tmp_path: Path) -> None:
    source = tmp_path / "published.txt"
    source.write_text(
        "wave depth sigma low high\n"
        "1.2 0.01 0.001 0.8 1.6\n"
        "1.8 0.02 0.001 1.4 2.2\n"
        "3.2 0.03 0.001 2.8 3.6\n",
        encoding="utf-8",
    )

    observation = load_observation_table(
        source,
        wavelength_column="wave",
        flux_column="depth",
        uncertainty_column="sigma",
        bin_low_column="low",
        bin_high_column="high",
        observable="transit_depth",
    )

    np.testing.assert_array_equal(
        observation.wavelength_bin_edges,
        np.array([[0.8, 1.6], [1.4, 2.2], [2.8, 3.6]]),
    )


def test_per_bin_validation_accepts_overlap_and_gap_but_checks_each_pair() -> None:
    grid = _bounds_grid()
    np.testing.assert_array_equal(
        grid.bin_bounds,
        np.array([[0.8, 1.6], [1.4, 2.2], [2.8, 3.6]]),
    )
    for invalid_bounds in (
        np.array([[0.8, 1.6], [1.9, 1.7], [2.8, 3.6]]),
        np.array([[0.8, 1.6], [1.4, 2.2], [3.3, 3.6]]),
        np.ones((2, 2)),
        np.array([[0.8, np.inf], [1.4, 2.2], [2.8, 3.6]]),
    ):
        with pytest.raises(RobertValidationError):
            SpectralGrid(values=grid.values, bin_edges=invalid_bounds)


def test_bound_conversion_preserves_requested_centres_and_row_order() -> None:
    wavelength_grid = _bounds_grid()
    wavenumber_bounds = spectral_grid_bin_bounds_in_unit(
        wavelength_grid,
        "cm^-1",
    )
    wavenumber_centres = spectral_grid_values_in_unit(
        wavelength_grid,
        "cm^-1",
    )
    assert wavenumber_bounds is not None
    np.testing.assert_allclose(
        wavenumber_bounds,
        np.array([[10000.0 / 1.6, 10000.0 / 0.8],
                  [10000.0 / 2.2, 10000.0 / 1.4],
                  [10000.0 / 3.6, 10000.0 / 2.8]]),
    )
    wavenumber_grid = SpectralGrid(
        values=wavenumber_centres,
        unit="cm^-1",
        bin_edges=wavenumber_bounds,
    )
    restored_bounds = spectral_grid_bin_bounds_in_unit(
        wavenumber_grid,
        "micron",
    )
    assert restored_bounds is not None
    np.testing.assert_allclose(restored_bounds, wavelength_grid.bin_bounds)

    reversed_grid = SpectralGrid(
        values=wavelength_grid.values[::-1],
        unit="micron",
        bin_edges=wavelength_grid.bin_edges[::-1],
    )
    np.testing.assert_array_equal(
        reversed_grid.bin_bounds,
        wavelength_grid.bin_bounds[::-1],
    )


def test_tophat_and_pixel_response_integrate_exact_pairs() -> None:
    observation = _observation(_bounds_grid())
    source_grid = SpectralGrid(
        values=np.linspace(0.5, 4.0, 351),
        unit="micron",
    )
    source_values = 2.0 * source_grid.values + 1.0
    spectrum = Spectrum(
        spectral_grid=source_grid,
        values=source_values,
        unit=observation.flux_unit,
        observable=observation.observable,
    )
    expected = 2.0 * np.mean(observation.wavelength_bin_bounds, axis=1) + 1.0

    top_hat = TopHatObservationResponse().prepare(observation).observe(spectrum)
    pixel = PixelIntegrationResponse().prepare(observation, source_grid).observe(spectrum)

    np.testing.assert_allclose(top_hat.values, expected, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(pixel.values, expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(pixel.spectral_grid.bin_edges, observation.wavelength_bin_edges)


def test_hminus_bin_average_matches_individual_bounds_and_cutoff() -> None:
    bounds = np.array([[15000.0, 16000.0], [16000.0, 16500.0], [17000.0, 18000.0]])
    exact = _hminus_bound_free_bin_cross_section(bounds)
    singles = np.array(
        [
            _hminus_bound_free_bin_cross_section(pair).item()
            for pair in bounds
        ]
    )

    np.testing.assert_array_equal(exact, singles)
    assert exact[1] > 0.0
    assert exact[2] == 0.0


def test_correlated_k_exact_windows_match_independent_exok_bins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "numba-cache"
    cache.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    _ensure_writable_numba_cache()
    source = _write_small_kta(tmp_path / "H2O_exact_windows.kta")
    provider = CorrelatedKOpacityProvider.from_kta_paths({"H2O": source})
    grid = _bounds_grid()
    binned = provider.bin_to_spectral_grid(grid, num=50, remove_zeros=False)
    table = binned.tables["H2O"]

    expected_centres = spectral_grid_values_in_unit(grid, "cm^-1")
    np.testing.assert_allclose(table.wavenumber_cm_inverse, expected_centres)
    np.testing.assert_allclose(
        table.wavelength_micron,
        spectral_grid_values_in_unit(grid, "micron"),
    )
    assert table.kcoeff.shape == (2, 2, grid.size, 2)
    assert table.metadata["spectral_preparation"] == "exo_k_bin_down_exact_windows"
    for index, pair in enumerate(grid.bin_bounds):
        single_grid = SpectralGrid(
            values=np.array([grid.values[index]]),
            unit=grid.unit,
            bin_edges=pair,
        )
        single = provider.bin_to_spectral_grid(
            single_grid,
            num=50,
            remove_zeros=False,
        ).tables["H2O"]
        np.testing.assert_allclose(
            table.kcoeff[:, :, index, :],
            single.kcoeff[:, :, 0, :],
            rtol=1.0e-6,
            atol=0.0,
        )


def test_regular_kta_exact_windows_match_legacy_single_bin_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "numba-cache"
    cache.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    _ensure_writable_numba_cache()
    source = _write_small_regular_kta(tmp_path / "H2O_regular_windows.kta")
    provider = CorrelatedKOpacityProvider.from_kta_paths({"H2O": source})
    grid = _bounds_grid()
    table = provider.bin_to_spectral_grid(grid, num=50, remove_zeros=False).tables[
        "H2O"
    ]

    for index, pair in enumerate(grid.bin_bounds):
        single_grid = SpectralGrid(
            values=np.array([grid.values[index]]),
            unit=grid.unit,
            bin_edges=pair,
        )
        single = provider.bin_to_spectral_grid(
            single_grid,
            num=50,
            remove_zeros=False,
        ).tables["H2O"]
        np.testing.assert_allclose(
            table.kcoeff[:, :, index, :],
            single.kcoeff[:, :, 0, :],
            rtol=1.0e-6,
            atol=0.0,
        )


def test_exact_windows_reject_cached_table_with_non_native_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "numba-cache"
    cache.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    _ensure_writable_numba_cache()
    source = _write_small_kta(tmp_path / "H2O_non_native.kta")
    provider = CorrelatedKOpacityProvider.from_kta_paths({"H2O": source})
    native_table = provider.tables["H2O"]
    changed_wavenumber = np.array(native_table.wavenumber_cm_inverse, copy=True)
    changed_wavenumber += 1.0
    non_native_table = replace(
        native_table,
        wavenumber_cm_inverse=changed_wavenumber,
        wavelength_micron=10000.0 / changed_wavenumber,
    )

    with pytest.raises(RobertValidationError, match="native KTA spectral grid"):
        bin_correlated_k_table_with_exok(
            non_native_table,
            _bounds_grid(),
            num=50,
            remove_zeros=False,
        )


def test_emission_and_transmission_keep_exact_bin_grid() -> None:
    grid = _bounds_grid()
    provider = _provider(grid)
    pressure = PressureGrid(
        edges=np.array([0.1, 1.0, 10.0]),
        centers=np.array([0.3, 3.0]),
        unit="bar",
    )
    emission = EmissionForwardModel(
        planet=Planet(name="Synthetic b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="Synthetic", radius_m=7.0e8, effective_temperature_k=5500.0),
        spectral_grid=grid,
        pressure_grid=pressure,
        base_temperature_K=np.array([900.0, 1100.0]),
        opacity_provider=provider,
        config=EmissionModelConfig(
            opacity_species=("H2O", "CO"),
            log_vmr_parameters={"H2O": "log_h2o", "CO": "log_co"},
            include_rayleigh=False,
            thermal_integration_backend="numpy",
            stellar_spectrum_model="blackbody",
            temperature_offset_parameter="temperature_offset",
            radius_scale_parameter="radius_scale",
        ),
    )
    emission_spectrum = emission(
        {
            "log_h2o": -3.0,
            "log_co": -4.0,
            "temperature_offset": 0.0,
            "radius_scale": 1.0,
        }
    )
    np.testing.assert_array_equal(emission_spectrum.spectral_grid.bin_edges, grid.bin_edges)

    radius = 8.0e7
    gravity = 12.0
    mass = gravity * radius**2 / 6.67430e-11
    transmission = build_parameterized_transmission_model(
        ParameterizedTransmissionFactoryConfig(
            planet=Planet(name="Synthetic b", radius_m=radius, mass_kg=mass),
            star=Star(name="Synthetic star", radius_m=7.0e8),
            temperature_profile=IsothermalTemperatureProfile(parameter_name="temperature"),
            chemistry_model=FreeChemistry(
                active_species=("H2O", "CO"),
                parameter_names={"H2O": "log_h2o", "CO": "log_co"},
                parameter_mode="log10",
            ),
            opacity_source=provider,
            opacity_binning=None,
            pressure_grid=pressure,
            model=ParameterizedTransmissionModelConfig(
                opacity_species=("H2O", "CO"),
                reference_pressure_bar=1.0,
                include_rayleigh=False,
                gas_combination="sum_by_g",
            ),
        ),
        spectral_grid=grid,
    )
    transmission_spectrum = transmission(
        {"temperature": 1000.0, "log_h2o": -3.0, "log_co": -4.0}
    )
    np.testing.assert_array_equal(
        transmission_spectrum.spectral_grid.bin_edges,
        grid.bin_edges,
    )
