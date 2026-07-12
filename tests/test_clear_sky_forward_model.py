"""Tests for the reusable parameterized clear-sky emission model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robert_exoplanets import (
    ClearSkyEmissionForwardModel,
    ClearSkyEmissionModelConfig,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    Planet,
    PressureGrid,
    SpectralGrid,
    Star,
)
from robert_exoplanets.core import RobertValidationError


def _provider(spectral_grid: SpectralGrid) -> CorrelatedKOpacityProvider:
    wavenumber = 10000.0 / spectral_grid.values
    tables = {}
    for index, species in enumerate(("H2O", "CO"), start=1):
        tables[species] = CorrelatedKTable(
            species=species,
            pressure_bar=np.array([0.3, 3.0]),
            temperature_K=np.array([500.0, 1500.0]),
            wavenumber_cm_inverse=wavenumber,
            g_samples=np.array([0.5]),
            g_weights=np.array([1.0]),
            kcoeff=np.full((2, 2, 2, 1), index * 1.0e-24),
            metadata={"checksum_sha256": f"checksum-{species}"},
        )
    return CorrelatedKOpacityProvider(
        tables=tables,
        interpolation="log_pressure_temperature_log_k",
    )


def _model() -> ClearSkyEmissionForwardModel:
    spectral_grid = SpectralGrid(
        values=np.array([2.0, 4.0]),
        bin_edges=np.array([1.5, 3.0, 5.0]),
        unit="micron",
        role="observed",
    )
    pressure_grid = PressureGrid(
        edges=np.array([0.1, 1.0, 10.0]),
        centers=np.array([0.3, 3.0]),
        unit="bar",
    )
    return ClearSkyEmissionForwardModel(
        planet=Planet(name="Synthetic b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="Synthetic", radius_m=7.0e8, effective_temperature_k=5500.0),
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        base_temperature_K=np.array([900.0, 1100.0]),
        opacity_provider=_provider(spectral_grid),
        config=ClearSkyEmissionModelConfig(
            opacity_species=("H2O", "CO"),
            log_vmr_parameters={"H2O": "log_h2o", "CO": "log_co"},
            include_rayleigh=False,
            thermal_integration_backend="numpy",
            metadata={"fixture": "two-gas"},
        ),
    )


def test_clear_sky_emission_model_evaluates_multi_gas_eclipse_depth() -> None:
    model = _model()

    spectrum = model(
        {
            "log_h2o": -3.0,
            "log_co": -4.0,
            "temperature_offset": 20.0,
            "radius_scale": 1.02,
        }
    )

    assert spectrum.observable == "eclipse_depth"
    np.testing.assert_array_equal(spectrum.spectral_grid.values, model.spectral_grid.values)
    assert np.all(np.isfinite(spectrum.values))
    assert np.all(spectrum.values > 0.0)
    assert model.required_parameters == (
        "log_h2o",
        "log_co",
        "temperature_offset",
        "radius_scale",
    )
    assert model.opacity_identifiers == {
        "H2O": "checksum-H2O",
        "CO": "checksum-CO",
    }
    assert model.manifest_metadata["fixture"] == "two-gas"
    assert len(model.manifest_metadata["pressure_grid_sha256"]) == 64
    assert len(model.manifest_metadata["spectral_grid_sha256"]) == 64
    assert model.manifest_metadata["log_vmr_parameters"] == "H2O:log_h2o,CO:log_co"


def test_diagnostics_free_fused_model_matches_diagnostic_reference() -> None:
    pytest.importorskip("numba")
    fused = _model()
    reference = replace(
        fused,
        config=replace(fused.config, compute_diagnostics=True),
    )
    parameters = {
        "log_h2o": -3.0,
        "log_co": -4.0,
        "temperature_offset": 20.0,
        "radius_scale": 1.02,
    }

    fused_spectrum = fused(parameters)
    reference_spectrum = reference(parameters)

    np.testing.assert_allclose(
        fused_spectrum.values,
        reference_spectrum.values,
        rtol=2.0e-13,
        atol=0.0,
    )


def test_clear_sky_emission_model_rejects_missing_or_invalid_parameters() -> None:
    model = _model()

    with pytest.raises(RobertValidationError, match="parameters are missing"):
        model({"log_h2o": -3.0})

    with pytest.raises(RobertValidationError, match="sum below one"):
        model(
            {
                "log_h2o": 0.0,
                "log_co": -4.0,
                "temperature_offset": 0.0,
                "radius_scale": 1.0,
            }
        )

    with pytest.raises(RobertValidationError, match="radius scale"):
        model(
            {
                "log_h2o": -3.0,
                "log_co": -4.0,
                "temperature_offset": 0.0,
                "radius_scale": 0.0,
            }
        )


def test_clear_sky_emission_config_requires_species_parameter_alignment() -> None:
    with pytest.raises(RobertValidationError, match="keys must match"):
        ClearSkyEmissionModelConfig(
            opacity_species=("H2O", "CO"),
            log_vmr_parameters={"H2O": "log_h2o"},
        )


def test_clear_sky_emission_model_can_derive_planet_gravity() -> None:
    model = _model()
    derived = ClearSkyEmissionForwardModel(
        planet=Planet(name="Mass-radius b", radius_m=7.0e7, mass_kg=1.0e27),
        star=model.star,
        spectral_grid=model.spectral_grid,
        pressure_grid=model.pressure_grid,
        base_temperature_K=model.base_temperature_K,
        opacity_provider=model.opacity_provider,
        config=model.config,
        geometry=model.geometry,
    )

    assert derived.gravity_m_s2 == pytest.approx(6.67430e-11 * 1.0e27 / (7.0e7) ** 2)
