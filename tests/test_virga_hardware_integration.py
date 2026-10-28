"""Opt-in integration test against the installed Virga cloud model."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import OpticalConstantsCatalog
from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, SpectralGrid
from robert_exoplanets.forward.virga_cloud import VirgaEddySedCloudModel
from robert_exoplanets.opacity import (
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
)
from robert_exoplanets.rt import assemble_gas_optical_depth


pytestmark = pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_VIRGA_TESTS") != "1",
    reason="set ROBERT_RUN_VIRGA_TESTS=1 to run the optional Virga integration test",
)


def test_virga_cloud_adapter_matches_direct_virga_calculation(
    tmp_path: Path,
) -> None:
    """Compare ROBERT's mapping with independent output from real Virga."""

    virga = pytest.importorskip(
        "virga.justdoit",
        reason="install virga-exo 2.x to run the Virga integration test",
    )
    catalog = OpticalConstantsCatalog(
        Path(__file__).resolve().parents[1]
        / "data"
        / "optical_constants"
        / "exo_skryer"
    )
    refractive_index = catalog.load("MgSiO3")
    source_checksum = refractive_index.metadata.get("source_sha256")
    assert source_checksum is not None and len(source_checksum) == 64

    mie_directory = _prepare_virga_mie_table(
        virga,
        refractive_index,
        tmp_path,
    )
    pressure_grid = _pressure_grid()
    temperature_edges_k = np.linspace(1800.0, 850.0, pressure_grid.n_layers + 1)
    temperature_k = 0.5 * (temperature_edges_k[:-1] + temperature_edges_k[1:])
    spectral_grid = SpectralGrid(
        values=np.array([10000.0, 5000.0, 2500.0, 1250.0]),
        unit="cm^-1",
        role="native",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=temperature_k,
        temperature_edges=temperature_edges_k,
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-3)},
        mean_molecular_weight=2.3,
    )
    gas_optical_depth = _gas_optical_depth(
        atmosphere,
        spectral_grid,
    )

    fsed = 0.5
    kz_cm2_s = 1.0e8
    metallicity = 1.0
    gas_mass_mixing_ratio = 2.75e-3
    geometric_stddev = 2.0
    model = VirgaEddySedCloudModel(
        condensibles=("MgSiO3",),
        gas_mass_mixing_ratios={"MgSiO3": gas_mass_mixing_ratio},
        mie_directory=mie_directory,
        fsed=fsed,
        kz_cm2_s=kz_cm2_s,
        metallicity=metallicity,
        geometric_stddev=geometric_stddev,
    )
    (adapter_cloud,) = model.evaluate(gas_optical_depth, parameters={})

    direct = _run_direct_virga(
        virga,
        mie_directory=mie_directory,
        pressure_grid=pressure_grid,
        temperature_edges_k=temperature_edges_k,
        fsed=fsed,
        kz_cm2_s=kz_cm2_s,
        metallicity=metallicity,
        gas_mass_mixing_ratio=gas_mass_mixing_ratio,
        geometric_stddev=geometric_stddev,
    )

    target_wavelength_micron = 10000.0 / spectral_grid.values
    direct_tau = _interpolate_direct_output(
        direct,
        "opd_per_layer",
        target_wavelength_micron,
    )[::-1]
    direct_ssa = _interpolate_direct_output(
        direct,
        "single_scattering",
        target_wavelength_micron,
    )[::-1]
    direct_asymmetry = _interpolate_direct_output(
        direct,
        "asymmetry",
        target_wavelength_micron,
    )[::-1]

    # Virga pressure layers are arithmetic midpoints between the exact edges
    # supplied by ROBERT. This test also checks reversed-pressure row mapping.
    ascending_edges_bar = np.sort(pressure_grid.edges)
    expected_virga_pressure_bar = 0.5 * (
        ascending_edges_bar[:-1] + ascending_edges_bar[1:]
    )
    np.testing.assert_allclose(
        direct["pressure"],
        expected_virga_pressure_bar,
        rtol=1.0e-12,
    )
    assert direct["pressure_unit"] == "bar"
    assert direct["wave_unit"] == "micron"

    # The cloud must be physically active for this comparison. This prevents a
    # vacuous pass where both paths return only clear-atmosphere values.
    condensate_mmr = np.asarray(direct["condensate_mmr"], dtype=float)
    assert condensate_mmr.shape == (pressure_grid.n_layers, 1)
    assert float(np.max(condensate_mmr)) > 0.0
    assert float(np.max(direct_tau)) > 0.0

    np.testing.assert_allclose(adapter_cloud.extinction_tau, direct_tau, rtol=2.0e-12)
    np.testing.assert_allclose(
        adapter_cloud.single_scattering_albedo,
        direct_ssa,
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        adapter_cloud.asymmetry_factor,
        direct_asymmetry,
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert adapter_cloud.metadata["virga_version"] == _virga_version()
    assert adapter_cloud.metadata["cloud_virga_do_virtual"] == "false"


def _prepare_virga_mie_table(
    virga,
    refractive_index,
    tmp_path: Path,
) -> Path:
    """Build a small real Virga Mie table from the bundled MgSiO3 data."""

    refrind_directory = tmp_path / "refrind"
    mie_directory = tmp_path / "mie"
    refrind_directory.mkdir()
    mie_directory.mkdir()
    wavelength = np.asarray(refractive_index.wavelength_micron, dtype=float)
    real_index = np.asarray(refractive_index.real_index, dtype=float)
    imaginary_index = np.asarray(refractive_index.imaginary_index, dtype=float)
    refrind_table = np.column_stack(
        (np.arange(1, wavelength.size + 1), wavelength, real_index, imaginary_index)
    )
    np.savetxt(
        refrind_directory / "MgSiO3.refrind",
        refrind_table,
        fmt=("%d", "%.10g", "%.10g", "%.10g"),
    )

    virga.calc_mie_db(
        "MgSiO3",
        virga_dir=str(refrind_directory),
        dir_out=str(mie_directory),
        rmin=1.0e-8,
        rmax=1.0e-2,
        nradii=20,
        logspace=True,
        aggregates=False,
        fort_calc_mie=False,
    )
    assert (mie_directory / "MgSiO3.mieff").is_file()
    assert (mie_directory / "MgSiO3.mieff").stat().st_size > 0
    return mie_directory


def _pressure_grid() -> PressureGrid:
    edges = np.logspace(1.0, -5.0, 13)
    return PressureGrid(
        edges=edges,
        centers=np.sqrt(edges[:-1] * edges[1:]),
        unit="bar",
        name="Virga integration pressure grid",
    )


def _gas_optical_depth(
    atmosphere: AtmosphereState,
    spectral_grid: SpectralGrid,
):
    pressure_grid = atmosphere.pressure_grid
    prepared = PreparedCorrelatedKOpacity(
        provider_name="virga-integration-test-opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="virga-integration-test-opacity",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.full(
            (1, pressure_grid.n_layers, spectral_grid.size, 1),
            1.0e-25,
        ),
    )
    return assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=15.0,
    )


def _run_direct_virga(
    virga,
    *,
    mie_directory: Path,
    pressure_grid: PressureGrid,
    temperature_edges_k: np.ndarray,
    fsed: float,
    kz_cm2_s: float,
    metallicity: float,
    gas_mass_mixing_ratio: float,
    geometric_stddev: float,
):
    import astropy.units as units

    # Virga sorts ptk points to increasing pressure. Give it the same ordered
    # pressure-edge state that the adapter builds from ROBERT's grid.
    pressure_bar = np.sort(np.asarray(pressure_grid.edges, dtype=float))
    temperature_sorted = np.asarray(temperature_edges_k, dtype=float)[::-1]
    virga_atmosphere = virga.Atmosphere(
        ["MgSiO3"],
        fsed=fsed,
        mh=metallicity,
        mmw=2.3,
        sig=geometric_stddev,
        param="const",
        verbose=False,
        supsat=0.0,
        gas_mmr={"MgSiO3": gas_mass_mixing_ratio},
    )
    virga_atmosphere.gravity(
        gravity=1500.0,
        gravity_unit=units.cm / units.s**2,
    )
    virga_atmosphere.ptk(
        df={
            "pressure": pressure_bar,
            "temperature": temperature_sorted,
            "kz": np.full(pressure_bar.size, kz_cm2_s),
        },
        kz_min=kz_cm2_s,
    )
    return virga.compute(
        virga_atmosphere,
        directory=str(mie_directory),
        as_dict=True,
        do_virtual=False,
    )


def _interpolate_direct_output(
    result: dict[str, object],
    key: str,
    target_wavelength_micron: np.ndarray,
) -> np.ndarray:
    source_wavelength = np.asarray(result["wave"], dtype=float)
    if result["wave_unit"] != "micron":
        raise AssertionError(f"unsupported direct Virga wave unit: {result['wave_unit']}")
    source_order = np.argsort(source_wavelength)
    source_wavelength = source_wavelength[source_order]
    source_values = np.asarray(result[key], dtype=float)[:, source_order]
    assert float(np.min(target_wavelength_micron)) >= float(source_wavelength[0])
    assert float(np.max(target_wavelength_micron)) <= float(source_wavelength[-1])
    return np.vstack(
        [
            np.interp(
                np.log(target_wavelength_micron),
                np.log(source_wavelength),
                row,
            )
            for row in source_values
        ]
    )


def _virga_version() -> str:
    try:
        return importlib.metadata.version("virga-exo")
    except importlib.metadata.PackageNotFoundError:
        return importlib.metadata.version("virga")
