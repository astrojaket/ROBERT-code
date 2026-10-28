"""Opt-in real Virga and ROBERT emission-forward integration test."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np
import pytest

from robert_exoplanets import (
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    FreeChemistry,
    ParameterizedEmissionForwardModel,
    ParameterizedEmissionModelConfig,
    Planet,
    PressureGrid,
    SpectralGrid,
    Star,
    TabulatedTemperatureProfile,
)
from robert_exoplanets.atmosphere import AtmosphereBuilder
from robert_exoplanets.forward.virga_cloud import VirgaEddySedCloudModel


pytestmark = pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_VIRGA_TESTS") != "1",
    reason="set ROBERT_RUN_VIRGA_TESTS=1 to run the optional Virga integration test",
)


def test_real_virga_cloud_controls_change_emission_spectrum(
    tmp_path: Path,
    record_property: pytest.RecordProperty,
) -> None:
    """Run real Virga inside the NumPy emission forward model."""

    virga = pytest.importorskip(
        "virga.justdoit",
        reason="install virga-exo 2.x to run the Virga forward integration test",
    )
    from robert_exoplanets import OpticalConstantsCatalog

    repository = Path(__file__).resolve().parents[1]
    catalog = OpticalConstantsCatalog(
        repository / "data" / "optical_constants" / "exo_skryer"
    )
    refractive_index = catalog.load("MgSiO3")
    assert len(refractive_index.metadata["source_sha256"]) == 64
    mie_directory = _prepare_mie_table(virga, refractive_index, tmp_path)

    layers = 8
    pressure_edges = np.logspace(1.0, -5.0, layers + 1)
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=np.sqrt(pressure_edges[:-1] * pressure_edges[1:]),
        unit="bar",
        name="Virga forward integration pressure grid",
    )
    spectral_grid = SpectralGrid(
        values=np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]),
        unit="micron",
        role="native",
        name="Virga forward integration wavelengths",
    )
    chemistry = FreeChemistry(
        active_species=("H2O",),
        parameter_names={"H2O": "log_h2o"},
        parameter_mode="log10",
    )
    atmosphere_builder = AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=TabulatedTemperatureProfile(
            pressure=np.array([9.0e-6, 11.0]),
            temperature=np.array([850.0, 1800.0]),
            pressure_unit="bar",
            extrapolation="raise",
            name="small Virga test profile",
        ),
        chemistry_model=chemistry,
        mean_molecular_weight=2.3,
    )
    atmosphere = atmosphere_builder.build({"log_h2o": -3.0})
    np.testing.assert_allclose(
        atmosphere.composition["H2O"],
        np.full(layers, 1.0e-3),
    )
    np.testing.assert_allclose(
        atmosphere.mean_molecular_weight,
        np.full(layers, 2.3),
    )
    assert atmosphere.temperature_edges is not None
    assert atmosphere.temperature_edges.shape == (layers + 1,)

    provider = _opacity_provider(spectral_grid)
    cloud_model = VirgaEddySedCloudModel(
        condensibles=("MgSiO3",),
        gas_mass_mixing_ratios={"MgSiO3": 2.75e-3},
        mie_directory=mie_directory,
        fsed=None,
        fsed_parameter="cloud_fsed",
        kz_cm2_s=None,
        log10_kz_cm2_s_parameter="log10_cloud_kzz",
        metallicity=1.0,
        geometric_stddev=2.0,
        multiple_scattering_backend="sh4",
    )
    model = ParameterizedEmissionForwardModel(
        planet=Planet(
            name="Virga integration planet",
            radius_m=7.0e7,
            gravity_m_s2=15.0,
        ),
        star=Star(
            name="Virga integration star",
            radius_m=7.0e8,
            effective_temperature_k=5500.0,
        ),
        spectral_grid=spectral_grid,
        atmosphere_builder=atmosphere_builder,
        opacity_provider=provider,
        config=ParameterizedEmissionModelConfig(
            opacity_species=("H2O",),
            include_rayleigh=False,
            thermal_integration_backend="numpy",
            stellar_spectrum_model="blackbody",
            compute_diagnostics=False,
        ),
        cloud_model=cloud_model,
    )
    baseline_parameters = {
        "log_h2o": -3.0,
        "cloud_fsed": 0.5,
        "log10_cloud_kzz": 8.0,
    }
    fsed_parameters = {**baseline_parameters, "cloud_fsed": 2.0}
    kz_parameters = {**baseline_parameters, "log10_cloud_kzz": 10.0}

    baseline, baseline_seconds = _timed_forward(model, baseline_parameters)
    fsed_changed, fsed_seconds = _timed_forward(model, fsed_parameters)
    kz_changed, kz_seconds = _timed_forward(model, kz_parameters)

    for spectrum in (baseline, fsed_changed, kz_changed):
        assert spectrum.observable == "eclipse_depth"
        assert spectrum.values.shape == (spectral_grid.size,)
        assert np.all(np.isfinite(spectrum.values))
        assert np.all(spectrum.values > 0.0)
    assert np.max(np.abs(fsed_changed.values - baseline.values)) > 1.0e-12
    assert np.max(np.abs(kz_changed.values - baseline.values)) > 1.0e-12
    assert model.manifest_metadata["cloud_provider"] == "virga-exo"
    assert model.manifest_metadata["cloud_multiple_scattering_backend"] == "sh4"
    assert model.manifest_metadata["cloud_virga_do_virtual"] == "false"
    assert model.manifest_metadata["mean_molecular_weight_model"] == "fixed_scalar"

    record_property("virga_version", _virga_version())
    record_property("mie_table_sha256", _sha256(mie_directory / "MgSiO3.mieff"))
    record_property("baseline_forward_seconds", baseline_seconds)
    record_property("fsed_perturbed_forward_seconds", fsed_seconds)
    record_property("kzz_perturbed_forward_seconds", kz_seconds)
    record_property(
        "max_abs_flux_delta_from_fsed",
        float(np.max(np.abs(fsed_changed.values - baseline.values))),
    )
    record_property(
        "max_abs_flux_delta_from_kzz",
        float(np.max(np.abs(kz_changed.values - baseline.values))),
    )


def _prepare_mie_table(
    virga,
    refractive_index,
    tmp_path: Path,
) -> Path:
    """Create a small Virga Mie table from the bundled MgSiO3 data."""

    import numpy as np

    refrind_directory = tmp_path / "refrind"
    mie_directory = tmp_path / "mie"
    refrind_directory.mkdir()
    mie_directory.mkdir()
    refrind = np.column_stack(
        (
            np.arange(1, refractive_index.wavelength_micron.size + 1),
            refractive_index.wavelength_micron,
            refractive_index.real_index,
            refractive_index.imaginary_index,
        )
    )
    np.savetxt(
        refrind_directory / "MgSiO3.refrind",
        refrind,
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
    mie_file = mie_directory / "MgSiO3.mieff"
    assert mie_file.is_file() and mie_file.stat().st_size > 0
    return mie_directory


def _opacity_provider(spectral_grid: SpectralGrid) -> CorrelatedKOpacityProvider:
    n_wavelengths = spectral_grid.size
    wavenumber = 10000.0 / spectral_grid.values
    wavelength_scale = np.array([1.0, 0.7, 0.45, 0.3, 0.2, 0.13, 0.08])
    kcoeff = np.broadcast_to(
        (2.0e-23 * wavelength_scale)[None, None, :, None],
        (2, 2, n_wavelengths, 1),
    ).copy()
    table = CorrelatedKTable(
        species="H2O",
        pressure_bar=np.array([1.0e-5, 10.0]),
        temperature_K=np.array([500.0, 2000.0]),
        wavenumber_cm_inverse=wavenumber,
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        kcoeff=kcoeff,
        metadata={"fixture": "small real-Virga forward integration"},
    )
    return CorrelatedKOpacityProvider(
        tables={"H2O": table},
        interpolation="log_pressure_temperature_log_k",
    )


def _timed_forward(model, parameters: dict[str, float]):
    start = time.perf_counter()
    spectrum = model(parameters)
    return spectrum, time.perf_counter() - start


def _virga_version() -> str:
    try:
        return importlib.metadata.version("virga-exo")
    except importlib.metadata.PackageNotFoundError:
        return importlib.metadata.version("virga")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
