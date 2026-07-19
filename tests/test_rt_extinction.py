"""Tests for CIA and Rayleigh layer optical depths."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    CiaTable,
    EvaluatedCorrelatedKOpacity,
    LayerOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    cia_optical_depth,
    rayleigh_scattering_optical_depth,
    read_cia_table,
    load_nemesispy_cia_table,
)
from robert_exoplanets.core import RobertCoverageError, RobertValidationError


def test_rayleigh_scattering_optical_depth_is_positive_and_larger_at_short_wavelengths() -> None:
    gas_tau = _gas_tau(SpectralGrid.from_array([0.5, 1.0, 2.0], unit="micron", role="opacity"))

    rayleigh = rayleigh_scattering_optical_depth(gas_tau)

    assert isinstance(rayleigh, LayerOpticalDepth)
    assert rayleigh.kind == "scattering_extinction"
    assert rayleigh.tau.shape == (gas_tau.atmosphere.n_layers, gas_tau.spectral_grid.size)
    assert np.all(rayleigh.tau > 0.0)
    assert np.all(rayleigh.tau[:, 0] > rayleigh.tau[:, 1])
    assert np.all(rayleigh.tau[:, 1] > rayleigh.tau[:, 2])
    assert rayleigh.phase_function_moments is not None
    np.testing.assert_allclose(
        rayleigh.phase_function_moments[:, 0, 0],
        [1.0, 0.0, 0.5, 0.0, 0.0],
    )


def test_rayleigh_mixture_adds_species_cross_sections_not_refractivities() -> None:
    gas_tau = _gas_tau(SpectralGrid.from_array([0.5], unit="micron", role="opacity"))

    rayleigh = rayleigh_scattering_optical_depth(gas_tau)

    wavelength_m = 0.5e-6
    inverse_micron = 2.0
    refractivity_h2 = 13.58e-5 * (1.0 + 7.52e-3 * inverse_micron**2)
    refractivity_he = 3.48e-5 * (1.0 + 2.30e-3 * inverse_micron**2)
    n0 = 1.01325e5 / (1.380649e-23 * 273.15)
    common = 32.0 * np.pi**3 / (3.0 * (n0 * wavelength_m**2) ** 2)
    expected_cross_section = common * (
        0.84 * refractivity_h2**2 + 0.159 * refractivity_he**2
    )
    expected = gas_tau.layer_column_density_molecules_m2 * expected_cross_section
    np.testing.assert_allclose(rayleigh.tau[:, 0], expected)
    assert rayleigh.metadata["mixture_rule"] == (
        "number-weighted sum of molecular cross-sections"
    )


def test_rayleigh_requires_composition_or_explicit_background_assumption() -> None:
    gas_tau = _gas_tau(SpectralGrid.from_array([0.5], unit="micron", role="opacity"))
    atmosphere = AtmosphereState(
        pressure_grid=gas_tau.pressure_grid,
        temperature=gas_tau.atmosphere.temperature,
        composition={"H2O": np.full(gas_tau.atmosphere.n_layers, 1.0)},
        mean_molecular_weight=18.01528,
    )
    opacity = _evaluated_opacity(
        gas_tau.pressure_grid,
        gas_tau.spectral_grid,
        np.zeros((1, gas_tau.atmosphere.n_layers, 1, 1)),
    )
    no_background = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)

    with pytest.raises(RobertValidationError, match="explicit H2/He composition"):
        rayleigh_scattering_optical_depth(no_background)

    assumed = rayleigh_scattering_optical_depth(
        no_background,
        default_h2_fraction_of_h2_he=0.85,
    )
    assert np.all(assumed.tau > 0.0)
    assert assumed.metadata["composition_fallback"] == "explicit_H2/He"


def test_phantom_background_has_zero_rayleigh_scattering() -> None:
    gas_tau = _gas_tau(SpectralGrid.from_array([0.5], unit="micron", role="opacity"))
    atmosphere = AtmosphereState(
        pressure_grid=gas_tau.pressure_grid,
        temperature=gas_tau.atmosphere.temperature,
        composition={
            "H2O": np.full(gas_tau.atmosphere.n_layers, 0.1),
            "phantom": np.full(gas_tau.atmosphere.n_layers, 0.9),
        },
        mean_molecular_weight=25.0,
        metadata={"opacity_free_species": "phantom"},
    )
    opacity = _evaluated_opacity(
        gas_tau.pressure_grid,
        gas_tau.spectral_grid,
        np.zeros((1, gas_tau.atmosphere.n_layers, 1, 1)),
    )
    phantom = assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)

    rayleigh = rayleigh_scattering_optical_depth(phantom)

    np.testing.assert_allclose(rayleigh.tau, 0.0)
    assert rayleigh.metadata["composition_fallback"] == (
        "zero_H2/He_with_opacity_free_background"
    )


def test_cia_optical_depth_uses_h2_h2_and_h2_he_pairs() -> None:
    spectral_grid = SpectralGrid.from_array([5.0, 10.0], unit="micron", role="opacity")
    gas_tau = _gas_tau(spectral_grid)
    wavenumber = np.array([0.0, 1000.0, 2000.0, 3000.0])
    temperature = np.array([500.0, 1500.0])
    k_cia = np.zeros((4, 2, 4), dtype=float)
    k_cia[2, :, :] = 2.0
    k_cia[3, :, :] = 1.0
    cia_table = CiaTable(
        wavenumber_cm_inverse=wavenumber,
        temperature_K=temperature,
        k_cia=k_cia,
        pair_order=(
            "H2-H2_equilibrium",
            "H2-He_equilibrium",
            "H2-H2_normal",
            "H2-He_normal",
        ),
    )

    cia = cia_optical_depth(gas_tau, cia_table)

    assert cia.kind == "absorption_continuum"
    assert cia.tau.shape == (gas_tau.atmosphere.n_layers, spectral_grid.size)
    assert np.all(cia.tau > 0.0)
    assert "H2-H2_normal" in cia.metadata["active_pairs"]
    assert "H2-He_normal" in cia.metadata["active_pairs"]


def test_petitradtrans_cia_hdf_loader_and_log_interpolation(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "h2-h2.ciatable.petitRADTRANS.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("t", data=[500.0, 1500.0])
        handle.create_dataset("wavenumbers", data=[1000.0, 2000.0])
        alpha = handle.create_dataset("alpha", data=[[1.0, 2.0], [100.0, 200.0]])
        alpha.attrs["units"] = "cm^-1"
        handle.create_dataset("mol_name", data=[b"H2", b"H2"])
        handle.create_dataset("mol_mass", data=[2.01588, 2.01588])
        handle.create_dataset("DOI", data=[b"10.0000/example"])
    table = CiaTable.from_petitradtrans_hdf(path, collision_pair="H2-H2")
    gas_tau = _gas_tau(SpectralGrid.from_array([10.0, 5.0], unit="micron", role="opacity"))

    cia = cia_optical_depth(
        gas_tau,
        table,
        coefficient_interpolation="log",
    )

    assert table.metadata["source_format"] == "petitradtrans_cia_hdf5"
    assert table.metadata["doi"] == "10.0000/example"
    np.testing.assert_allclose(table.k_cia[0], table.k_cia[2])
    assert np.all(table.k_cia[1] == 0.0)
    assert np.all(cia.tau > 0.0)
    assert cia.metadata["coefficient_interpolation"] == "log"


def test_cia_optical_depth_requires_explicit_extrapolation_policy() -> None:
    spectral_grid = SpectralGrid.from_array([2.0, 10.0], unit="micron", role="opacity")
    gas_tau = _gas_tau(spectral_grid)
    cia_table = CiaTable(
        wavenumber_cm_inverse=[1000.0, 2000.0],
        temperature_K=[1000.0, 1100.0],
        k_cia=np.ones((4, 2, 2)),
        pair_order=("eq_h2", "eq_he", "H2-H2_normal", "H2-He_normal"),
    )

    with pytest.raises(RobertCoverageError):
        cia_optical_depth(gas_tau, cia_table)

    result = cia_optical_depth(
        gas_tau,
        cia_table,
        temperature_extrapolation="clip",
        spectral_extrapolation="zero",
    )
    assert result.metadata["temperature_extrapolation"] == "clip"
    assert result.metadata["spectral_extrapolation"] == "zero"


def test_read_cia_table_reads_fortran_records(tmp_path) -> None:
    path = tmp_path / "tiny_cia.tab"
    temperatures = np.array([500.0, 1000.0], dtype="<f8")
    coefficients = np.arange(4 * 2 * 4, dtype="<f4")
    _write_fortran_record(path, temperatures.tobytes(), mode="wb")
    _write_fortran_record(path, coefficients.tobytes(), mode="ab")

    table = read_cia_table(path, dnu=25.0, n_pairs=4, endian="<")

    assert table.temperature_K.tolist() == [500.0, 1000.0]
    np.testing.assert_allclose(table.wavenumber_cm_inverse, [0.0, 25.0, 50.0, 75.0])
    expected = coefficients.reshape(4, 2, 4).transpose(2, 1, 0)
    np.testing.assert_allclose(table.k_cia, expected)


def test_vendored_nemesispy_cia_table_has_recorded_provenance() -> None:
    table = load_nemesispy_cia_table()

    assert table.n_pairs == 9
    assert table.temperature_K[0] == pytest.approx(200.0)
    assert table.temperature_K[-1] == pytest.approx(3800.0)
    assert table.metadata["source_tag"] == "v1.0.1"
    assert table.metadata["checksum_sha256"] == (
        "5b519f02f98b205f20628ee5ec7f2829528d0bd356b449c4221ba8b2ef86ea0e"
    )


def _gas_tau(spectral_grid: SpectralGrid):
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 1.0e-1]),
        centers=np.array([1.0e-4, 1.0e-2]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([900.0, 1200.0]),
        composition={
            "H2O": np.full(pressure_grid.n_layers, 1.0e-3),
            "H2": np.full(pressure_grid.n_layers, 0.84),
            "He": np.full(pressure_grid.n_layers, 0.159),
        },
        mean_molecular_weight=2.3,
    )
    opacity = _evaluated_opacity(
        pressure_grid,
        spectral_grid,
        np.zeros((1, pressure_grid.n_layers, spectral_grid.size, 1)),
    )
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)


def _evaluated_opacity(
    pressure_grid: PressureGrid,
    spectral_grid: SpectralGrid,
    kcoeff: np.ndarray,
) -> EvaluatedCorrelatedKOpacity:
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test-correlated-k",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="test-extinction-cache-key",
    )
    return EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=kcoeff,
        unit="m^2/molecule",
    )


def _write_fortran_record(path, payload: bytes, *, mode: str) -> None:
    with path.open(mode) as handle:
        marker = struct.pack("<i", len(payload))
        handle.write(marker)
        handle.write(payload)
        handle.write(marker)
