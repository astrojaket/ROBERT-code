"""Tests for ExoMol cross-section opacity sampling."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    ExoMolOpacitySamplingSource,
    OpacitySamplingProvider,
    OpacitySamplingTable,
    PressureGrid,
    assemble_gas_optical_depth,
    assemble_opacity_sampling_gas_optical_depth,
)
from robert_exoplanets.core import RobertCoverageError


def test_exomol_table_loads_axes_and_metadata(tmp_path) -> None:
    path = _write_exomol_fixture(tmp_path / "H2O.h5")

    table = OpacitySamplingTable.from_exomol_hdf(
        path, species="H2O", checksum=False
    )

    assert table.species == "H2O"
    assert table.unit == "cm^2/molecule"
    np.testing.assert_array_equal(table.pressure_bar, [1.0, 10.0])
    np.testing.assert_array_equal(table.temperature_K, [500.0, 1000.0])
    np.testing.assert_array_equal(table.wavenumber_cm_inverse, [1000.0, 2000.0, 3000.0])
    assert table.metadata["source_format"] == "exomol_taurex_hdf5"
    assert table.metadata["line_list"] == "1H2-16O__TEST"

    source = ExoMolOpacitySamplingSource(
        species=("H2O",), paths={"H2O": path}, checksum=False
    )
    assert isinstance(source.load(), OpacitySamplingProvider)


def test_opacity_sampling_interpolates_log_cross_section_and_sums_species(tmp_path) -> None:
    paths = {
        species: _write_exomol_fixture(tmp_path / f"{species}.h5", scale=scale)
        for species, scale in (("H2O", 1.0), ("CO", 2.0))
    }
    provider = OpacitySamplingProvider.from_exomol_paths(paths, checksum=False)
    spectral_grid = provider.native_spectral_grid()
    pressure_grid = PressureGrid(
        edges=np.array([1.0, 10.0]),
        centers=np.array([np.sqrt(10.0)]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([750.0]),
        composition={"H2O": np.array([1.0e-4]), "CO": np.array([2.0e-4])},
        mean_molecular_weight=np.array([2.3]),
    )

    prepared = provider.prepare(spectral_grid, pressure_grid, ("H2O", "CO"))
    assert prepared.metadata["maturity"] == "beta"
    evaluated = provider.evaluate(atmosphere, prepared)
    assert evaluated.metadata["maturity"] == "beta"

    base = np.sqrt(10.0) * np.exp(0.75) * 1.0e-25
    # native_spectral_grid returns increasing wavelength, hence reversed wavenumber.
    np.testing.assert_allclose(evaluated.kcoeff[0, 0, :, 0], base * np.array([3.0, 2.0, 1.0]))
    np.testing.assert_allclose(evaluated.kcoeff[1], 2.0 * evaluated.kcoeff[0])

    gas_tau = assemble_gas_optical_depth(
        atmosphere,
        evaluated,
        gravity_m_s2=10.0,
        gas_combination="random_overlap",
    )
    assert gas_tau.metadata["opacity_mode"] == "opacity_sampling"
    assert gas_tau.metadata["gas_combination"] == "sum_by_g"
    assert gas_tau.metadata["requested_gas_combination"] == "random_overlap"
    np.testing.assert_allclose(gas_tau.total_tau, np.sum(gas_tau.species_tau, axis=0))

    fused = assemble_opacity_sampling_gas_optical_depth(
        atmosphere,
        provider,
        prepared,
        gravity_m_s2=10.0,
    )
    np.testing.assert_allclose(fused.total_tau, gas_tau.total_tau, rtol=2.0e-13)
    assert fused.species_tau is None
    assert fused.metadata["species_tau_diagnostics"] == "disabled"
    assert fused.metadata["maturity"] == "beta"


def test_native_grid_stride_and_coverage_are_explicit(tmp_path) -> None:
    path = _write_exomol_fixture(tmp_path / "H2O.h5")
    provider = OpacitySamplingProvider.from_exomol_paths(
        {"H2O": path}, checksum=False
    )
    spectral_grid = provider.native_spectral_grid(sampling=2)
    np.testing.assert_allclose(spectral_grid.values, [10.0 / 3.0, 10.0])

    pressure_grid = PressureGrid(
        edges=np.array([0.01, 0.1]), centers=np.array([np.sqrt(0.001)]), unit="bar"
    )
    with pytest.raises(RobertCoverageError, match="pressure values are outside"):
        provider.prepare(spectral_grid, pressure_grid, ("H2O",))


def _write_exomol_fixture(path, *, scale: float = 1.0):
    h5py = pytest.importorskip("h5py")
    pressure = np.array([1.0, 10.0])
    temperature = np.array([500.0, 1000.0])
    wavenumber = np.array([1000.0, 2000.0, 3000.0])
    values = np.empty((2, 2, 3))
    for p_index, p_value in enumerate(pressure):
        for t_index, t_value in enumerate(temperature):
            values[p_index, t_index] = (
                scale
                * p_value
                * np.exp(t_value / 1000.0)
                * np.arange(1.0, 4.0)
                * 1.0e-25
            )
    with h5py.File(path, "w") as handle:
        p = handle.create_dataset("p", data=pressure)
        p.attrs["units"] = "bar"
        t = handle.create_dataset("t", data=temperature)
        t.attrs["units"] = "kelvin"
        w = handle.create_dataset("bin_edges", data=wavenumber)
        w.attrs["units"] = "wavenumbers"
        xsec = handle.create_dataset("xsecarr", data=values)
        xsec.attrs["units"] = "cm^2/molecule"
        handle.create_dataset("DOI", data=np.array([b"10.0000/test"]))
        handle.create_dataset("key_iso_ll", data=np.array([b"1H2-16O__TEST"]))
        handle.create_dataset("mol_name", data=np.array([b"H2O"]))
    return path
