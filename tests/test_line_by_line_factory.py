"""Focused tests for the line-by-line forward factory source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    EmissionFactoryConfig,
    EmissionModelConfig,
    IsothermalTemperatureProfile,
    Planet,
    Star,
    build_emission_model,
)
from robert_exoplanets.core import RobertConfigError
from robert_exoplanets.forward.factory import LineByLineOpacitySource
from robert_exoplanets.opacity import LineByLineOpacityProvider


def test_line_by_line_source_builds_normal_emission_factory(tmp_path: Path) -> None:
    path = _write_table(tmp_path / "H2O.h5")
    source = LineByLineOpacitySource(
        species=("H2O",),
        paths={"H2O": path},
        wavelength_bounds_micron=(2.0, 2.1),
        native_sampling_stride=2,
        max_cached_slices=1,
        checksum=False,
    )

    native_grid = source.native_spectral_grid()
    np.testing.assert_allclose(native_grid.values, [2.0, 2.1])
    provider = source.load()
    assert isinstance(provider, LineByLineOpacityProvider)
    assert provider.native_sampling_stride == 2
    assert provider.wavelength_bounds_micron == (2.0, 2.1)
    assert provider.max_cached_slices == 1

    config = EmissionFactoryConfig(
        planet=Planet(name="LBL b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="LBL star", radius_m=7.0e8, effective_temperature_k=5500.0),
        temperature_profile=IsothermalTemperatureProfile(parameter_name="T_iso"),
        temperature_parameters={"T_iso": 1000.0},
        opacity_source=source,
        model=EmissionModelConfig(
            opacity_species=("H2O",),
            log_vmr_parameters={"H2O": "log_h2o"},
            include_rayleigh=False,
            thermal_integration_backend="numpy",
            stellar_spectrum_model="blackbody",
        ),
    )
    model = build_emission_model(config, spectral_grid=native_grid)

    spectrum = model(
        {
            "T_iso": 1000.0,
            "log_h2o": -3.0,
            "temperature_offset": 0.0,
            "radius_scale": 1.0,
        }
    )
    assert np.all(np.isfinite(spectrum.values))
    assert model.manifest_metadata["factory_opacity_source_type"] == "line_by_line"
    assert model.manifest_metadata["factory_opacity_native_sampling_stride"] == "2"
    assert model.manifest_metadata["factory_opacity_native_sampling_accuracy"] == (
        "requires_convergence_validation"
    )
    assert model.manifest_metadata["factory_exo_k_binning"] == "disabled"
    assert model.prepared_opacity.metadata["native_sampling_stride"] == "2"


def test_line_by_line_source_validates_window_and_stride() -> None:
    with pytest.raises(RobertConfigError, match="wavelength_bounds_micron"):
        LineByLineOpacitySource(
            species=("H2O",),
            paths={"H2O": "water.h5"},
            wavelength_bounds_micron=(2.0, 2.0),
        )
    with pytest.raises(RobertConfigError, match="native_sampling_stride"):
        LineByLineOpacitySource(
            species=("H2O",),
            paths={"H2O": "water.h5"},
            native_sampling_stride=0,
        )


def _write_table(path: Path) -> Path:
    h5py = pytest.importorskip("h5py")
    pressure = np.array([0.3, 3.0])
    temperature = np.array([500.0, 1500.0])
    wavelength = np.array([2.0, 2.05, 2.1, 2.15])
    values = np.empty((2, 2, 4))
    for pressure_index, pressure_value in enumerate(pressure):
        for temperature_index, temperature_value in enumerate(temperature):
            values[pressure_index, temperature_index] = (
                pressure_value
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
    return path
