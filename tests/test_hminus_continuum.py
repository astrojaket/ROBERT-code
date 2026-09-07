"""Focused H-minus continuum and retrieval-source tests."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from robert_exoplanets import (
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    FreeChemistry,
    HMinusContinuumConfig,
    LineByLineOpacityProvider,
    ParameterizedEmissionForwardModel,
    ParameterizedEmissionModelConfig,
    Planet,
    PressureGrid,
    SpectralGrid,
    Star,
    TabulatedTemperatureProfile,
)
from robert_exoplanets.atmosphere import AtmosphereBuilder


def test_retrieved_hminus_changes_lrs_and_lbl_hrs_spectra(tmp_path: Path) -> None:
    """The same explicit chemistry source must affect both opacity modes."""

    h5py = pytest.importorskip("h5py")
    pressure_grid = PressureGrid(
        edges=np.array([0.1, 1.0, 10.0]),
        centers=np.array([0.316227766, 3.16227766]),
        unit="bar",
    )
    chemistry = FreeChemistry(
        active_species=("H2O", "H-", "H", "e-"),
        parameter_names={
            "H2O": "log_h2o",
            "H-": "log_hminus",
            "H": "log_h",
            "e-": "log_electron",
        },
        parameter_mode="log10",
    )
    builder = AtmosphereBuilder(
        pressure_grid=pressure_grid,
        temperature_profile=TabulatedTemperatureProfile(
            pressure=np.array([0.1, 10.0]),
            temperature=np.array([2000.0, 4000.0]),
            pressure_unit="bar",
        ),
        chemistry_model=chemistry,
        mean_molecular_weight=2.3,
    )
    config = ParameterizedEmissionModelConfig(
        opacity_species=("H2O",),
        include_rayleigh=False,
        thermal_integration_backend="numpy",
        stellar_spectrum_model="blackbody",
        hminus_continuum=HMinusContinuumConfig(),
    )
    lrs_grid = SpectralGrid(
        values=np.array([1.0, 2.0]),
        bin_edges=np.array([0.8, 1.5, 2.5]),
        unit="micron",
        role="observed",
    )
    lrs_model = ParameterizedEmissionForwardModel(
        planet=Planet(name="H-minus b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="H-minus star", radius_m=7.0e8, effective_temperature_k=5500.0),
        spectral_grid=lrs_grid,
        atmosphere_builder=builder,
        opacity_provider=_correlated_provider(lrs_grid, pressure_grid),
        config=config,
    )

    hrs_grid = SpectralGrid.from_array(
        [1.0, 1.1, 2.0],
        unit="micron",
        role="native",
    )
    h5_path = tmp_path / "H2O-lbl.h5"
    with h5py.File(h5_path, "w") as handle:
        pressure = handle.create_dataset("pressure_bar", data=[0.1, 10.0])
        pressure.attrs["units"] = "bar"
        temperature = handle.create_dataset("temperature_K", data=[1000.0, 4000.0])
        temperature.attrs["units"] = "K"
        wavelength = handle.create_dataset(
            "wavelength_micron",
            data=hrs_grid.values,
        )
        wavelength.attrs["units"] = "micron"
        cross_section = handle.create_dataset(
            "cross_section",
            data=np.full((2, 2, hrs_grid.size), 1.0e-24),
        )
        cross_section.attrs["units"] = "cm^2/molecule"
    hrs_provider = LineByLineOpacityProvider.from_hdf_paths(
        {"H2O": h5_path},
        checksum=False,
        max_memory_bytes=1024 * 1024,
    )
    hrs_model = ParameterizedEmissionForwardModel(
        planet=lrs_model.planet,
        star=lrs_model.star,
        spectral_grid=hrs_grid,
        atmosphere_builder=builder,
        opacity_provider=hrs_provider,
        config=config,
    )

    baseline = {
        "log_h2o": -3.0,
        "log_hminus": -16.0,
        "log_h": -6.0,
        "log_electron": -12.0,
    }
    enhanced = {**baseline, "log_hminus": -10.0}
    lrs_baseline = lrs_model(baseline)
    lrs_enhanced = lrs_model(enhanced)
    hrs_baseline = hrs_model(baseline)
    hrs_enhanced = hrs_model(enhanced)

    streamed_config = replace(config, aggregate_additional_optical_depths=True)
    streamed_lrs_model = ParameterizedEmissionForwardModel(
        planet=lrs_model.planet,
        star=lrs_model.star,
        spectral_grid=lrs_grid,
        atmosphere_builder=builder,
        opacity_provider=_correlated_provider(lrs_grid, pressure_grid),
        config=streamed_config,
    )
    streamed_lrs = streamed_lrs_model(baseline)

    assert np.all(np.isfinite(lrs_baseline.values))
    assert np.all(np.isfinite(hrs_baseline.values))
    assert not np.allclose(lrs_baseline.values, lrs_enhanced.values)
    assert not np.allclose(hrs_baseline.values, hrs_enhanced.values)
    np.testing.assert_allclose(streamed_lrs.values, lrs_baseline.values, rtol=2.0e-13, atol=0.0)
    assert lrs_model.manifest_metadata["include_hminus_continuum"] == "true"
    assert hrs_model.manifest_metadata["hminus_continuum_species"] == "H-,H,e-"
    assert streamed_lrs_model.manifest_metadata["aggregate_additional_optical_depths"] == "true"


def _correlated_provider(
    spectral_grid: SpectralGrid,
    pressure_grid: PressureGrid,
) -> CorrelatedKOpacityProvider:
    table = CorrelatedKTable(
        species="H2O",
        pressure_bar=np.asarray(pressure_grid.centers),
        temperature_K=np.array([1000.0, 4000.0]),
        wavenumber_cm_inverse=10000.0 / spectral_grid.values,
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        kcoeff=np.full(
            (pressure_grid.n_layers, 2, spectral_grid.size, 1),
            1.0e-24,
        ),
    )
    return CorrelatedKOpacityProvider(
        tables={"H2O": table},
        interpolation="log_pressure_temperature_log_k",
    )
