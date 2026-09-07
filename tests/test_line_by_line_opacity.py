"""Tests for tabulated line-by-line opacity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import (
    PressureGrid,
    RobertCoverageError,
    RobertValidationError,
    SpectralGrid,
)
from robert_exoplanets.opacity.line_by_line import (
    EvaluatedLineByLineOpacity,
    LineByLineOpacityProvider,
    LineByLineTable,
    PreparedLineByLineOpacity,
)
from robert_exoplanets.rt import assemble_gas_optical_depth
from robert_exoplanets.rt.optical_depth import (
    assemble_line_by_line_gas_optical_depth,
)
from robert_exoplanets.forward._atmospheric import evaluate_gas_optical_depth


def test_hdf5_table_and_narrow_physical_wavelength_evaluation(
    tmp_path: Path,
) -> None:
    path = _write_hdf5_fixture(tmp_path / "H2O-lbl.h5")

    table = LineByLineTable.from_hdf5(path, species="H2O", checksum=False)

    assert table.metadata["source_format"] == "line_by_line_hdf5"
    assert table.metadata["storage_strategy"] == "hdf5_hyperslab"
    assert table.metadata["spectral_coordinate"] == "physical_wavelength"
    assert table.native_shape == (2, 2, 4)
    assert table.native_nbytes == 2 * 2 * 4 * 8
    np.testing.assert_array_equal(table.wavelength_micron, [1.0, 1.1, 1.2, 1.3])

    provider = LineByLineOpacityProvider(
        {"H2O": table},
        max_memory_bytes=1024 * 1024,
    )
    spectral_grid = SpectralGrid.from_array(
        [1.2, 1.0],
        unit="micron",
        role="opacity",
    )
    pressure_grid = PressureGrid(
        edges=np.array([1.0, 10.0]),
        centers=np.array([np.sqrt(10.0)]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([750.0]),
        composition={"H2O": np.array([1.0e-4])},
        mean_molecular_weight=np.array([2.3]),
    )

    assert provider.estimate_memory_bytes(spectral_grid, ["H2O"]) > 0
    prepared = provider.prepare(spectral_grid, pressure_grid, ("H2O",))
    evaluated = provider.evaluate(atmosphere, prepared)

    assert prepared.metadata["opacity_mode"] == "line_by_line"
    assert prepared.metadata["physical_sample_axis"] == "singleton"
    assert prepared.metadata["source_read"] == "exact_narrow_window"
    assert prepared.g_samples.shape == (1,)
    assert prepared.g_weights.shape == (1,)
    np.testing.assert_allclose(prepared.g_samples, [0.5])
    np.testing.assert_allclose(prepared.g_weights, [1.0])
    np.testing.assert_array_equal(prepared.spectral_indices["H2O"], [2, 0])
    assert prepared.log_cross_sections["H2O"].shape == (2, 2, 2)
    assert evaluated.kcoeff.shape == (1, 1, 2, 1)
    assert evaluated.metadata["spectral_coordinate"] == "physical_wavelength"

    gas_optical_depth = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=10.0,
        gas_combination="random_overlap",
    )
    assert gas_optical_depth.metadata["gas_combination"] == "sum_by_g"
    assert gas_optical_depth.metadata["requested_gas_combination"] == "random_overlap"

    expected = np.sqrt(10.0) * np.exp(0.75) * np.array([1.2, 1.0]) * 1.0e-25
    np.testing.assert_allclose(evaluated.kcoeff[0, 0, :, 0], expected)
    assert not evaluated.kcoeff.flags.writeable


def test_hdf5_source_accepts_singleton_source_axis_and_unit_conversion(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "CO-lbl-singleton.h5"
    pressure = np.array([1000.0, 10000.0])
    temperature = np.array([500.0, 1000.0])
    wavelength_nm = np.array([1000.0, 1100.0])
    values = np.ones((2, 2, 2, 1)) * 1.0e-24
    with h5py.File(path, "w") as handle:
        p = handle.create_dataset("p", data=pressure)
        p.attrs["units"] = "mbar"
        t = handle.create_dataset("t", data=temperature)
        t.attrs["units"] = "K"
        wavelength = handle.create_dataset("wavelength", data=wavelength_nm)
        wavelength.attrs["units"] = "nm"
        cross_section = handle.create_dataset("xsecarr", data=values)
        cross_section.attrs["units"] = "m^2/molecule"

    table = LineByLineTable.from_hdf5(path, species="CO", checksum=False)

    assert table.native_shape == (2, 2, 2, 1)
    np.testing.assert_allclose(table.pressure_bar, [1.0, 10.0])
    np.testing.assert_allclose(table.wavelength_micron, [1.0, 1.1])
    assert table.unit == "m^2/molecule"


def test_petitradtrans_bin_edges_and_different_pt_axes(tmp_path: Path) -> None:
    h2o_path = _write_prt_fixture(
        tmp_path / "H2O_all_iso_HITEMP.h5",
        pressure=[1.0, 10.0],
        temperature=[500.0, 1000.0],
        wavenumber=[10000.0, 11000.0, 12000.0],
        scale=1.0,
    )
    co_path = _write_prt_fixture(
        tmp_path / "CO_all_iso_HITEMP.h5",
        pressure=[1.0, 3.0, 10.0],
        temperature=[400.0, 700.0, 1000.0],
        # Official pRT files can repeat the first wavenumber sample.
        wavenumber=[9000.0, 9000.0, 10000.0, 11000.0, 13000.0],
        scale=2.0,
    )

    provider = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": h2o_path, "CO": co_path},
        checksum=False,
        max_memory_bytes=1024 * 1024,
    )
    h2o_table = provider.tables["H2O"]
    co_table = provider.tables["CO"]

    assert h2o_table.metadata["source_spectral_coordinate"] == "physical_wavenumber"
    assert h2o_table.metadata["source_spectral_dataset"] == "bin_edges"
    assert h2o_table.metadata["source_spectral_unit"] == "cm^-1"
    np.testing.assert_allclose(
        h2o_table.wavelength_micron,
        10000.0 / np.array([10000.0, 11000.0, 12000.0]),
    )
    assert h2o_table.pressure_bar.size != co_table.pressure_bar.size
    assert h2o_table.temperature_K.size != co_table.temperature_K.size
    assert np.count_nonzero(np.diff(co_table.wavelength_micron) == 0.0) == 1

    spectral_grid = provider.native_spectral_grid()
    np.testing.assert_allclose(
        spectral_grid.values,
        10000.0 / np.array([10000.0, 11000.0]),
    )
    pressure_grid = _pressure_grid()
    atmosphere = _atmosphere(
        pressure_grid,
        [750.0],
        {"H2O": 1.0e-4, "CO": 2.0e-4},
    )

    prepared = provider.prepare(spectral_grid, pressure_grid, ("H2O", "CO"))
    evaluated = provider.evaluate(atmosphere, prepared)
    mixture = provider.evaluate_mixture(atmosphere, prepared)

    assert prepared.log_cross_sections["H2O"].shape == (2, 2, 2)
    assert prepared.log_cross_sections["CO"].shape == (3, 3, 2)
    assert evaluated.kcoeff.shape == (2, 1, 2, 1)
    assert mixture.cross_section.shape == (1, 2)
    np.testing.assert_allclose(
        evaluated.kcoeff[1, 0, :, 0],
        2.0
        * np.sqrt(10.0)
        * np.exp(0.75)
        * spectral_grid.values
        * 1.0e-25,
    )
    np.testing.assert_allclose(
        mixture.cross_section[0],
        evaluated.kcoeff[0, 0, :, 0] * 1.0e-4
        + evaluated.kcoeff[1, 0, :, 0] * 2.0e-4,
    )


def test_npz_roundtrip_reads_compact_table_and_direct_mixture(tmp_path: Path) -> None:
    path = _write_npz_fixture(tmp_path / "H2O-lbl.npz", scale=1.0)
    provider = LineByLineOpacityProvider.from_npz_paths(
        {"H2O": path},
        checksum=False,
        max_memory_bytes=1024 * 1024,
    )
    spectral_grid = provider.native_spectral_grid(
        wavelength_bounds_micron=(1.0, 1.2),
        sampling=2,
    )
    pressure_grid = _pressure_grid()
    atmosphere = _atmosphere(pressure_grid, [750.0], {"H2O": 2.0e-4})

    prepared = provider.prepare(spectral_grid, pressure_grid, ("H2O",))
    evaluated = provider.evaluate(atmosphere, prepared)
    mixture = provider.evaluate_mixture(atmosphere, prepared)

    np.testing.assert_allclose(spectral_grid.values, [1.0, 1.2])
    assert prepared.metadata["storage_strategy"] == "npz_guarded_full_member"
    assert prepared.metadata["source_read"] == "guarded_source_read"
    np.testing.assert_allclose(
        mixture.cross_section,
        evaluated.kcoeff[0, :, :, 0] * 2.0e-4,
    )
    assert mixture.metadata["species_arrays_retained"] == "false"
    assert mixture.metadata["assembly_backend"] == "fused_low_memory_direct_sum"


@pytest.mark.parametrize("method_name", ("evaluate", "evaluate_mixture"))
def test_direct_provider_evaluation_requires_vmr_composition(
    tmp_path: Path,
    method_name: str,
) -> None:
    path = _write_npz_fixture(tmp_path / "H2O-lbl.npz", scale=1.0)
    provider = LineByLineOpacityProvider.from_npz_paths(
        {"H2O": path},
        checksum=False,
        max_memory_bytes=1024 * 1024,
    )
    pressure_grid = _pressure_grid()
    atmosphere = replace(
        _atmosphere(pressure_grid, [750.0], {"H2O": 2.0e-4}),
        composition_convention="mass_fraction",
    )
    prepared = provider.prepare(
        provider.native_spectral_grid(wavelength_bounds_micron=(1.0, 1.2)),
        pressure_grid,
        ("H2O",),
    )

    with pytest.raises(RobertValidationError, match="volume_mixing_ratio"):
        getattr(provider, method_name)(atmosphere, prepared)


def test_prepared_slices_are_cached_with_a_bounded_lru(tmp_path: Path) -> None:
    path = _write_hdf5_fixture(tmp_path / "H2O-lbl.h5")
    provider = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": path},
        checksum=False,
        max_cached_slices=1,
        max_memory_bytes=1024 * 1024,
    )
    pressure_grid = _pressure_grid()
    first_grid = SpectralGrid.from_array([1.0], unit="micron", role="opacity")
    second_grid = SpectralGrid.from_array([1.1], unit="micron", role="opacity")

    first = provider.prepare(first_grid, pressure_grid, ("H2O",))
    assert provider.cached_slice_count == 1
    assert provider.cached_slice_bytes > 0
    assert provider.prepare(first_grid, pressure_grid, ("H2O",)) is first

    second = provider.prepare(second_grid, pressure_grid, ("H2O",))
    assert second is not first
    assert provider.cached_slice_count == 1
    assert provider.cached_slice_bytes == int(
        second.metadata["estimated_memory_bytes"]
    )
    provider.clear_prepared_cache()
    assert provider.cached_slice_count == 0
    assert provider.cached_slice_bytes == 0


def test_configured_window_and_stride_are_explicit_and_not_called_accurate(
    tmp_path: Path,
) -> None:
    path = _write_hdf5_fixture(tmp_path / "H2O-lbl.h5")
    provider = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": path},
        checksum=False,
        wavelength_bounds_micron=(1.0, 1.2),
        native_sampling_stride=2,
    )
    grid = provider.native_spectral_grid()
    np.testing.assert_allclose(grid.values, [1.0, 1.2])
    assert grid.metadata["native_sampling_stride"] == "2"
    assert grid.metadata["native_sampling_accuracy"] == (
        "requires_convergence_validation"
    )
    prepared = provider.prepare(grid, _pressure_grid(), ("H2O",))
    assert prepared.metadata["native_sampling_stride"] == "2"
    with pytest.raises(RobertCoverageError, match="configured line-by-line"):
        provider.prepare(
            SpectralGrid.from_array([1.3], unit="micron", role="opacity"),
            _pressure_grid(),
            ("H2O",),
        )


def test_fused_line_by_line_tau_has_no_species_opacity_cube(tmp_path: Path) -> None:
    paths = {
        species: _write_hdf5_fixture(
            tmp_path / f"{species}-lbl.h5",
            scale=scale,
        )
        for species, scale in (("H2O", 1.0), ("CO", 2.0))
    }
    provider = LineByLineOpacityProvider.from_hdf_paths(
        paths,
        checksum=False,
        max_memory_bytes=1024 * 1024,
    )
    pressure_grid = _pressure_grid()
    spectral_grid = SpectralGrid.from_array(
        [1.0, 1.1],
        unit="micron",
        role="opacity",
    )
    atmosphere = _atmosphere(
        pressure_grid,
        [750.0],
        {"H2O": 1.0e-4, "CO": 2.0e-4},
    )
    prepared = provider.prepare(spectral_grid, pressure_grid, ("H2O", "CO"))

    fused = assemble_line_by_line_gas_optical_depth(
        atmosphere,
        provider,
        prepared,
        gravity_m_s2=10.0,
    )
    via_forward_helper = evaluate_gas_optical_depth(
        provider,
        prepared,
        atmosphere,
        gravity_m_s2=10.0,
        gas_combination="random_overlap",
        retain_species_tau=False,
    )

    assert fused.species_tau is None
    assert fused.metadata["assembly_backend"] == "fused_lbl_direct_sum"
    assert fused.metadata["gas_combination"] == "sum_by_g"
    np.testing.assert_allclose(via_forward_helper.total_tau, fused.total_tau)
    np.testing.assert_allclose(
        fused.total_tau[:, :, 0],
        provider.evaluate_mixture(atmosphere, prepared).cross_section
        * 1.0e-4
        * fused.layer_column_density_molecules_m2[:, None],
    )


def test_memory_guard_rejects_before_source_read(tmp_path: Path) -> None:
    path = _write_hdf5_fixture(tmp_path / "H2O-lbl.h5")
    provider = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": path},
        checksum=False,
        max_memory_bytes=1,
    )
    spectral_grid = SpectralGrid.from_array(
        [1.0, 1.1],
        unit="micron",
        role="opacity",
    )

    with pytest.raises(RobertValidationError, match="estimated memory"):
        provider.prepare(spectral_grid, _pressure_grid(), ("H2O",))


def test_strict_coverage_and_clip_policy_are_explicit(tmp_path: Path) -> None:
    path = _write_hdf5_fixture(tmp_path / "H2O-lbl.h5")
    spectral_grid = SpectralGrid.from_array([1.0], unit="micron", role="opacity")
    pressure_grid = _pressure_grid()
    out_of_range = _atmosphere(pressure_grid, [1800.0], {"H2O": 1.0e-4})

    strict = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": path},
        checksum=False,
        max_memory_bytes=1024 * 1024,
    )
    prepared = strict.prepare(spectral_grid, pressure_grid, ("H2O",))
    report = strict.coverage(out_of_range, prepared)
    assert not report.valid
    assert "temperature values are outside" in report.reasons["H2O"]
    with pytest.raises(RobertCoverageError, match="temperature values are outside"):
        strict.evaluate(out_of_range, prepared)

    clipped = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": path},
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=False,
        max_memory_bytes=1024 * 1024,
    )
    clipped_prepared = clipped.prepare(spectral_grid, pressure_grid, ("H2O",))
    clipped_evaluated = clipped.evaluate(out_of_range, clipped_prepared)
    expected = np.sqrt(10.0) * np.exp(1.0) * 1.0 * 1.0e-25
    np.testing.assert_allclose(clipped_evaluated.kcoeff[0, 0, 0, 0], expected)


def test_invalid_physical_grid_and_source_shape_are_rejected(tmp_path: Path) -> None:
    path = _write_hdf5_fixture(tmp_path / "H2O-lbl.h5")
    with pytest.raises(RobertCoverageError, match="exact subset"):
        LineByLineOpacityProvider.from_hdf_paths(
            {"H2O": path},
            checksum=False,
        ).prepare(
            SpectralGrid.from_array([1.05], unit="micron", role="opacity"),
            _pressure_grid(),
            ("H2O",),
        )

    h5py = pytest.importorskip("h5py")
    invalid = tmp_path / "invalid-lbl.h5"
    with h5py.File(invalid, "w") as handle:
        handle.create_dataset("pressure_bar", data=[1.0, 10.0])
        handle.create_dataset("temperature_K", data=[500.0, 1000.0])
        handle.create_dataset("wavelength_micron", data=[1.0, 1.1])
        cross_section = handle.create_dataset(
            "cross_section",
            data=np.ones((2, 2, 3)),
        )
        cross_section.attrs["units"] = "cm^2/molecule"
    with pytest.raises(RobertValidationError, match="shape"):
        LineByLineTable.from_hdf5(invalid, species="H2O", checksum=False)


def test_prepared_and_evaluated_types_reject_non_singleton_sample_axis(
    tmp_path: Path,
) -> None:
    path = _write_hdf5_fixture(tmp_path / "H2O-lbl.h5")
    provider = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": path},
        checksum=False,
    )
    prepared = provider.prepare(
        SpectralGrid.from_array([1.0], unit="micron", role="opacity"),
        _pressure_grid(),
        ("H2O",),
    )

    with pytest.raises(RobertValidationError, match="singleton"):
        PreparedLineByLineOpacity(
            provider_name=prepared.provider_name,
            spectral_grid=prepared.spectral_grid,
            pressure_grid=prepared.pressure_grid,
            species=prepared.species,
            g_samples=np.array([0.25, 0.75]),
            g_weights=np.array([0.5, 0.5]),
            cache_key="bad",
            log_cross_sections=prepared.log_cross_sections,
        )

    with pytest.raises(RobertValidationError, match="species x layers"):
        EvaluatedLineByLineOpacity(
            prepared=prepared,
            kcoeff=np.ones((1, 1, 1)),
        )


def _pressure_grid() -> PressureGrid:
    return PressureGrid(
        edges=np.array([1.0, 10.0]),
        centers=np.array([np.sqrt(10.0)]),
        unit="bar",
    )


def _atmosphere(
    pressure_grid: PressureGrid,
    temperature: list[float],
    composition: dict[str, float],
) -> AtmosphereState:
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.asarray(temperature),
        composition={
            species: np.full(pressure_grid.n_layers, value)
            for species, value in composition.items()
        },
        mean_molecular_weight=np.full(pressure_grid.n_layers, 2.3),
    )


def _write_hdf5_fixture(path: Path, *, scale: float = 1.0) -> Path:
    h5py = pytest.importorskip("h5py")
    pressure = np.array([1.0, 10.0])
    temperature = np.array([500.0, 1000.0])
    wavelength = np.array([1.0, 1.1, 1.2, 1.3])
    values = np.empty((2, 2, 4))
    for pressure_index, pressure_value in enumerate(pressure):
        for temperature_index, temperature_value in enumerate(temperature):
            values[pressure_index, temperature_index] = (
                scale
                * pressure_value
                * np.exp(temperature_value / 1000.0)
                * wavelength
                * 1.0e-25
            )
    with h5py.File(path, "w") as handle:
        p = handle.create_dataset("pressure_bar", data=pressure)
        p.attrs["units"] = "bar"
        t = handle.create_dataset("temperature_K", data=temperature)
        t.attrs["units"] = "K"
        w = handle.create_dataset("wavelength_micron", data=wavelength)
        w.attrs["units"] = "micron"
        cross_section = handle.create_dataset("cross_section", data=values)
        cross_section.attrs["units"] = "cm^2/molecule"
        handle.attrs["line_list"] = "synthetic-line-list"
    return path


def _write_npz_fixture(path: Path, *, scale: float = 1.0) -> Path:
    pressure = np.array([1.0, 10.0])
    temperature = np.array([500.0, 1000.0])
    wavelength = np.array([1.0, 1.1, 1.2, 1.3])
    values = np.empty((2, 2, 4))
    for pressure_index, pressure_value in enumerate(pressure):
        for temperature_index, temperature_value in enumerate(temperature):
            values[pressure_index, temperature_index] = (
                scale
                * pressure_value
                * np.exp(temperature_value / 1000.0)
                * wavelength
                * 1.0e-25
            )
    np.savez(
        path,
        pressure_bar=pressure,
        temperature_K=temperature,
        wavelength_micron=wavelength,
        cross_section=values,
        metadata_json=np.asarray('{"line_list": "synthetic-npz"}'),
    )
    return path


def _write_prt_fixture(
    path: Path,
    *,
    pressure: list[float],
    temperature: list[float],
    wavenumber: list[float],
    scale: float,
) -> Path:
    """Write a small petitRADTRANS-style p/t/bin_edges/xsecarr file."""

    h5py = pytest.importorskip("h5py")
    pressure_array = np.asarray(pressure, dtype=float)
    temperature_array = np.asarray(temperature, dtype=float)
    wavenumber_array = np.asarray(wavenumber, dtype=float)
    wavelength = 10000.0 / wavenumber_array
    values = np.empty(
        (pressure_array.size, temperature_array.size, wavenumber_array.size),
    )
    for pressure_index, pressure_value in enumerate(pressure_array):
        for temperature_index, temperature_value in enumerate(temperature_array):
            values[pressure_index, temperature_index] = (
                scale
                * pressure_value
                * np.exp(temperature_value / 1000.0)
                * wavelength
                * 1.0e-25
            )
    with h5py.File(path, "w") as handle:
        p = handle.create_dataset("p", data=pressure_array)
        p.attrs["units"] = "bar"
        t = handle.create_dataset("t", data=temperature_array)
        t.attrs["units"] = "K"
        bins = handle.create_dataset("bin_edges", data=wavenumber_array)
        bins.attrs["units"] = "cm^-1"
        cross_section = handle.create_dataset("xsecarr", data=values)
        cross_section.attrs["units"] = "cm^2/molecule"
    return path
