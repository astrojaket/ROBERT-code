"""Tests for the parameterized transmission forward-model foundation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robert_exoplanets import (
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    FreeChemistry,
    IsothermalTemperatureProfile,
    ParameterizedDeckHazeCloudModel,
    ParameterizedTransmissionFactoryConfig,
    ParameterizedTransmissionModelConfig,
    Planet,
    PressureGrid,
    PressureQuenchChemistry,
    QuenchGroup,
    SpectralGrid,
    Spectrum,
    Star,
    StellarContaminationModel,
    StellarHeterogeneity,
    build_parameterized_transmission_model,
)
from robert_exoplanets.core import RobertValidationError


def _spectral_grid() -> SpectralGrid:
    return SpectralGrid(
        values=np.array([1.0, 2.0]),
        bin_edges=np.array([0.8, 1.4, 2.5]),
        unit="micron",
        role="observed",
    )


def _pressure_grid() -> PressureGrid:
    return PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 1.0e-1, 10.0]),
        centers=np.array([1.0e-4, 1.0e-2, 1.0]),
        unit="bar",
    )


def _provider() -> CorrelatedKOpacityProvider:
    grid = _spectral_grid()
    pressure = _pressure_grid()
    return CorrelatedKOpacityProvider(
        tables={
            "H2O": CorrelatedKTable(
                species="H2O",
                pressure_bar=pressure.centers,
                temperature_K=np.array([700.0, 1300.0]),
                wavenumber_cm_inverse=10000.0 / grid.values,
                g_samples=np.array([0.5]),
                g_weights=np.array([1.0]),
                kcoeff=np.array(
                    [
                        [[[2.0e-24], [5.0e-25]], [[2.0e-24], [5.0e-25]]],
                        [[[2.0e-24], [5.0e-25]], [[2.0e-24], [5.0e-25]]],
                        [[[2.0e-24], [5.0e-25]], [[2.0e-24], [5.0e-25]]],
                    ]
                ),
                metadata={"checksum_sha256": "h2o-transmission-test"},
            )
        },
        interpolation="log_pressure_temperature_log_k",
    )


def _factory() -> ParameterizedTransmissionFactoryConfig:
    radius = 8.0e7
    gravity = 12.0
    mass = gravity * radius**2 / 6.67430e-11
    return ParameterizedTransmissionFactoryConfig(
        planet=Planet(
            name="Transmission b",
            radius_m=radius,
            mass_kg=mass,
        ),
        star=Star(name="Transmission star", radius_m=7.0e8),
        temperature_profile=IsothermalTemperatureProfile(
            parameter_name="temperature"
        ),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            parameter_names={"H2O": "log_h2o"},
            parameter_mode="log10",
        ),
        opacity_source=_provider(),
        opacity_binning=None,
        pressure_grid=_pressure_grid(),
        model=ParameterizedTransmissionModelConfig(
            opacity_species=("H2O",),
            reference_pressure_bar=10.0,
            radius_scale_parameter="radius_scale",
            gravity_model="inverse_square",
            include_rayleigh=False,
            gas_combination="sum_by_g",
        ),
    )


def test_parameterized_transmission_factory_builds_transit_depth() -> None:
    model = build_parameterized_transmission_model(
        _factory(),
        spectral_grid=_spectral_grid(),
    )
    parameters = {
        "temperature": 1000.0,
        "log_h2o": -3.0,
        "radius_scale": 1.0,
    }

    result = model.evaluate_result(parameters)
    spectrum = model(parameters)

    assert spectrum.observable == "transit_depth"
    assert spectrum.unit == "transit_depth"
    assert np.all(np.isfinite(spectrum.values))
    assert np.all(spectrum.values > 0.0)
    np.testing.assert_array_equal(spectrum.values, result.transit_depth.values)
    assert result.path_geometry is not None
    assert result.path_geometry.metadata["gravity_model"] == (
        "inverse_square_layer_center_fixed_point"
    )
    assert spectrum.metadata["forward_model"] == "parameterized_transmission"
    assert model.manifest_metadata["reference_pressure_bar"] == "10"
    assert model.opacity_identifiers == {"H2O": "h2o-transmission-test"}


def test_transmission_discovers_quench_parameters_and_manifest_metadata() -> None:
    base = _factory()
    chemistry = PressureQuenchChemistry(
        base_model=base.chemistry_model,
        groups=(QuenchGroup("log_Pq_H2O", ("H2O",)),),
    )
    model = build_parameterized_transmission_model(
        replace(base, chemistry_model=chemistry),
        spectral_grid=_spectral_grid(),
    )

    spectrum = model(
        {
            "temperature": 1000.0,
            "log_h2o": -3.0,
            "log_Pq_H2O": -2.0,
            "radius_scale": 1.0,
        }
    )

    assert model.required_parameters == (
        "temperature",
        "log_h2o",
        "log_Pq_H2O",
        "radius_scale",
    )
    assert model.manifest_metadata["chemistry_quench_scheme"] == "pressure_quench"
    assert model.manifest_metadata["chemistry_quench_groups"] == "log_Pq_H2O:H2O"
    assert np.all(np.isfinite(spectrum.values))


def test_transmission_depth_responds_to_abundance_and_reference_radius() -> None:
    model = build_parameterized_transmission_model(
        _factory(),
        spectral_grid=_spectral_grid(),
    )
    baseline = {
        "temperature": 1000.0,
        "log_h2o": -6.0,
        "radius_scale": 1.0,
    }

    low_abundance = model(baseline)
    high_abundance = model({**baseline, "log_h2o": -2.0})
    larger_radius = model({**baseline, "radius_scale": 1.01})

    assert np.all(high_abundance.values > low_abundance.values)
    assert np.all(larger_radius.values > low_abundance.values)


def test_transmission_factory_applies_parameterized_stellar_contamination() -> None:
    grid = _spectral_grid()
    photosphere = Spectrum(
        spectral_grid=grid,
        values=np.array([4.0, 4.0]),
        unit="W m^-3 sr^-1",
        observable="stellar_spectral_radiance",
    )
    spot = StellarHeterogeneity(
        name="spot",
        kind="spot",
        spectrum=Spectrum(
            spectral_grid=grid,
            values=np.array([1.0, 3.0]),
            unit="W m^-3 sr^-1",
            observable="stellar_spectral_radiance",
        ),
        covering_fraction_parameter="f_spot",
    )
    component = StellarContaminationModel(photosphere, (spot,))
    clear = build_parameterized_transmission_model(
        _factory(),
        spectral_grid=grid,
    )
    contaminated = build_parameterized_transmission_model(
        replace(_factory(), stellar_contamination=component),
        spectral_grid=grid,
    )
    shared = {
        "temperature": 1000.0,
        "log_h2o": -3.0,
        "radius_scale": 1.0,
    }

    clear_spectrum = clear(shared)
    contaminated_spectrum = contaminated({**shared, "f_spot": 0.2})
    epsilon = component.evaluate({"f_spot": 0.2}).contamination_factor.values

    np.testing.assert_allclose(contaminated_spectrum.values, clear_spectrum.values * epsilon)
    assert contaminated.required_parameters[-1] == "f_spot"
    assert contaminated.manifest_metadata["stellar_contamination"] == "enabled"
    assert contaminated_spectrum.metadata["stellar_contamination_application_order"] == (
        "native_transit_depth_before_instrument_response"
    )


def test_disabled_stellar_contamination_preserves_transmission_exactly() -> None:
    first = build_parameterized_transmission_model(
        _factory(),
        spectral_grid=_spectral_grid(),
    )
    second = build_parameterized_transmission_model(
        replace(_factory(), stellar_contamination=None),
        spectral_grid=_spectral_grid(),
    )
    parameters = {
        "temperature": 1000.0,
        "log_h2o": -3.0,
        "radius_scale": 1.0,
    }

    np.testing.assert_array_equal(first(parameters).values, second(parameters).values)
    assert first(parameters).metadata == second(parameters).metadata


def test_transmission_uses_shared_parameterized_deck_haze_model() -> None:
    clear_factory = _factory()
    cloud_model = ParameterizedDeckHazeCloudModel(
        deck_single_scattering_albedo=0.0,
        haze_single_scattering_albedo=1.0,
    )
    cloudy = build_parameterized_transmission_model(
        replace(clear_factory, cloud_model=cloud_model),
        spectral_grid=_spectral_grid(),
    )
    clear = build_parameterized_transmission_model(
        clear_factory,
        spectral_grid=_spectral_grid(),
    )
    parameters = {
        "temperature": 1000.0,
        "log_h2o": -6.0,
        "radius_scale": 1.0,
        "log_cloud_top_pressure_bar": -2.0,
        "log_cloud_optical_depth": 1.0,
        "log_haze_mass_extinction": -5.0,
        "haze_slope": -4.0,
    }

    cloudy_spectrum = cloudy(parameters)
    clear_spectrum = clear(
        {name: parameters[name] for name in clear.required_parameters}
    )

    assert cloudy.required_parameters == tuple(parameters)
    assert cloudy.manifest_metadata["cloud_geometry_independent"] == "true"
    assert np.all(cloudy_spectrum.values > clear_spectrum.values)


def test_transmission_model_validates_radius_and_gravity_choices() -> None:
    with pytest.raises(RobertValidationError, match="gravity_model"):
        ParameterizedTransmissionModelConfig(
            opacity_species=("H2O",),
            reference_pressure_bar=10.0,
            gravity_model="arbitrary",
        )
    with pytest.raises(RobertValidationError, match="quadrature"):
        ParameterizedTransmissionModelConfig(
            opacity_species=("H2O",),
            reference_pressure_bar=10.0,
            impact_quadrature_order=2.5,
        )

    model = build_parameterized_transmission_model(
        _factory(),
        spectral_grid=_spectral_grid(),
    )
    with pytest.raises(RobertValidationError, match="radius scale"):
        model(
            {
                "temperature": 1000.0,
                "log_h2o": -3.0,
                "radius_scale": 0.0,
            }
        )
