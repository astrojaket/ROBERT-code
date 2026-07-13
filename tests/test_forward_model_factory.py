"""Tests for typed Python forward-model configuration and construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    EmissionFactoryConfig,
    EmissionModelConfig,
    CorrelatedKOpacityProvider,
    CorrelatedKTable,
    ExoKOpacitySource,
    ExoKTableBinning,
    FreeChemistry,
    IsothermalTemperatureProfile,
    Planet,
    ParameterizedEmissionFactoryConfig,
    ParameterizedGreyCloudEmissionForwardModel,
    ParameterizedRefractiveIndexCloudEmissionForwardModel,
    ParameterizedEmissionModelConfig,
    RefractiveIndexCloudConfig,
    RefractiveIndexSpectrum,
    GreyScatteringCloudConfig,
    MultiDatasetEmissionForwardModel,
    SpectralGrid,
    Star,
    build_emission_model,
    build_multi_dataset_emission_model,
    build_parameterized_emission_model,
    pressure_grid_from_opacity,
)
from robert_exoplanets.core import RobertConfigError, RobertValidationError
from robert_exoplanets.atmosphere import AtmosphereBuilder


def _spectral_grid(*, include_edges: bool = True) -> SpectralGrid:
    return SpectralGrid(
        values=np.array([2.0, 4.0]),
        bin_edges=np.array([1.5, 3.0, 5.0]) if include_edges else None,
        unit="micron",
        role="observed",
    )


def _provider() -> CorrelatedKOpacityProvider:
    grid = _spectral_grid()
    return CorrelatedKOpacityProvider(
        tables={
            "H2O": CorrelatedKTable(
                species="H2O",
                pressure_bar=np.array([0.3, 3.0]),
                temperature_K=np.array([500.0, 1500.0]),
                wavenumber_cm_inverse=10000.0 / grid.values,
                g_samples=np.array([0.5]),
                g_weights=np.array([1.0]),
                kcoeff=np.full((2, 2, 2, 1), 1.0e-24),
            )
        },
        interpolation="log_pressure_temperature_log_k",
    )


def _factory_config() -> EmissionFactoryConfig:
    return EmissionFactoryConfig(
        planet=Planet(name="Configured b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(
            name="Configured star",
            radius_m=7.0e8,
            effective_temperature_k=5500.0,
        ),
        temperature_profile=IsothermalTemperatureProfile(parameter_name="temperature"),
        temperature_parameters={"temperature": 1000.0},
        opacity_source=_provider(),
        opacity_binning=None,
        model=EmissionModelConfig(
            opacity_species=("H2O",),
            log_vmr_parameters={"H2O": "log_h2o"},
            include_rayleigh=False,
            thermal_integration_backend="numpy",
        ),
    )


def test_factory_builds_evaluable_model_from_python_objects() -> None:
    model = build_emission_model(
        _factory_config(), spectral_grid=_spectral_grid()
    )

    spectrum = model(
        {
            "log_h2o": -3.0,
            "temperature_offset": 0.0,
            "radius_scale": 1.0,
        }
    )

    np.testing.assert_allclose(model.pressure_grid.centers, [0.3, 3.0])
    np.testing.assert_allclose(model.base_temperature_K, [1000.0, 1000.0])
    assert model.pressure_grid.metadata["species"] == "H2O"
    assert model.manifest_metadata["factory_configuration_interface"] == "typed_python"
    assert (
        model.manifest_metadata["factory_temperature_parameters"] == "temperature:1000"
    )
    assert model.manifest_metadata["factory_pressure_grid_source"] == "opacity_centers"
    assert model.manifest_metadata["factory_exo_k_binning"] == "disabled"
    assert np.all(np.isfinite(spectrum.values))
    assert spectrum.observable == "eclipse_depth"


def test_pressure_grid_factory_supports_descending_opacity_centers() -> None:
    provider = _provider()
    table = provider.tables["H2O"]
    descending = CorrelatedKOpacityProvider(
        tables={
            "H2O": CorrelatedKTable(
                species="H2O",
                pressure_bar=table.pressure_bar[::-1],
                temperature_K=table.temperature_K,
                wavenumber_cm_inverse=table.wavenumber_cm_inverse,
                g_samples=table.g_samples,
                g_weights=table.g_weights,
                kcoeff=table.kcoeff[::-1],
            )
        },
        interpolation=provider.interpolation,
    )

    grid = pressure_grid_from_opacity(descending)

    assert grid.orientation == "decreasing"
    np.testing.assert_allclose(grid.centers, [3.0, 0.3])


def test_factory_config_validates_temperature_and_opacity_species() -> None:
    base = _factory_config()
    with pytest.raises(RobertConfigError, match="temperature parameters are missing"):
        EmissionFactoryConfig(
            planet=base.planet,
            star=base.star,
            temperature_profile=base.temperature_profile,
            opacity_source=base.opacity_source,
            opacity_binning=None,
            model=base.model,
        )

    with pytest.raises(RobertConfigError, match="missing model species"):
        EmissionFactoryConfig(
            planet=base.planet,
            star=base.star,
            temperature_profile=base.temperature_profile,
            temperature_parameters=base.temperature_parameters,
            opacity_source=base.opacity_source,
            opacity_binning=None,
            model=EmissionModelConfig(
                opacity_species=("CO",),
                log_vmr_parameters={"CO": "log_co"},
            ),
        )


def test_exok_source_requires_one_source_and_matching_paths() -> None:
    with pytest.raises(RobertConfigError, match="exactly one"):
        ExoKOpacitySource(species=("H2O",))
    with pytest.raises(RobertConfigError, match="keys must match"):
        ExoKOpacitySource(
            species=("H2O", "CO"),
            paths={"H2O": Path("water.h5")},
        )

    source = ExoKOpacitySource(
        species=("H2O",),
        paths={"H2O": Path("water.h5")},
    )
    assert source.paths == {"H2O": Path("water.h5")}
    with pytest.raises(TypeError):
        source.paths["CO"] = Path("co.h5")  # type: ignore[index]

    directory_source = ExoKOpacitySource(
        species=("H2O",),
        directory=Path("ktables"),
        resolution=1000,
    )
    assert directory_source.resolution == "R1000"
    with pytest.raises(RobertConfigError, match="only valid"):
        ExoKOpacitySource(
            species=("H2O",),
            paths={"H2O": Path("water.h5")},
            resolution="R1000",
        )


def test_exok_binning_requires_observation_bin_edges() -> None:
    with pytest.raises(RobertConfigError, match="bin edges"):
        ExoKTableBinning().apply(_provider(), _spectral_grid(include_edges=False))
    with pytest.raises(RobertConfigError, match="positive integer"):
        ExoKTableBinning(num=0)


def test_pressure_grid_inference_requires_two_points() -> None:
    grid = _spectral_grid()
    provider = CorrelatedKOpacityProvider(
        tables={
            "H2O": CorrelatedKTable(
                species="H2O",
                pressure_bar=np.array([1.0]),
                temperature_K=np.array([1000.0]),
                wavenumber_cm_inverse=10000.0 / grid.values,
                g_samples=np.array([0.5]),
                g_weights=np.array([1.0]),
                kcoeff=np.full((1, 1, 2, 1), 1.0e-24),
            )
        }
    )

    with pytest.raises(RobertValidationError, match="at least two"):
        pressure_grid_from_opacity(provider)


def test_parameterized_factory_evaluates_temperature_and_chemistry_at_runtime() -> None:
    config = ParameterizedEmissionFactoryConfig(
        planet=Planet(name="Parameterized b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="Parameterized", radius_m=7.0e8, effective_temperature_k=5500.0),
        temperature_profile=IsothermalTemperatureProfile(parameter_name="T_iso"),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            parameter_names={"H2O": "log_h2o"},
            parameter_mode="log10",
        ),
        opacity_source=_provider(),
        opacity_binning=None,
        model=ParameterizedEmissionModelConfig(
            opacity_species=("H2O",),
            include_rayleigh=False,
            thermal_integration_backend="numpy",
        ),
    )
    model = build_parameterized_emission_model(
        config,
        spectral_grid=_spectral_grid(),
    )

    cool = model({"T_iso": 900.0, "log_h2o": -3.0})
    hot = model({"T_iso": 1200.0, "log_h2o": -3.0})

    assert model.required_parameters == ("T_iso", "log_h2o")
    assert model.manifest_metadata["factory_parameterization"] == (
        "runtime_temperature_and_chemistry"
    )
    assert np.all(hot.values > cool.values)


def test_shared_atmosphere_multi_dataset_model_matches_independent_models(
    monkeypatch,
) -> None:
    config = ParameterizedEmissionFactoryConfig(
        planet=Planet(name="Shared b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="Shared", radius_m=7.0e8, effective_temperature_k=5500.0),
        temperature_profile=IsothermalTemperatureProfile(parameter_name="T_iso"),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            parameter_names={"H2O": "log_h2o"},
            parameter_mode="log10",
        ),
        opacity_source=_provider(),
        opacity_binning=None,
        model=ParameterizedEmissionModelConfig(
            opacity_species=("H2O",),
            include_rayleigh=False,
            thermal_integration_backend="numpy",
        ),
    )
    first = build_parameterized_emission_model(
        config,
        spectral_grid=_spectral_grid(),
    )
    independent_second = build_parameterized_emission_model(
        config,
        spectral_grid=_spectral_grid(),
    )
    parameters = {"T_iso": 1000.0, "log_h2o": -3.0}
    independent = {
        "first": first(parameters),
        "second": independent_second(parameters),
    }
    build_calls = 0
    original_build = AtmosphereBuilder.build

    def counted_build(builder, values=None):
        nonlocal build_calls
        build_calls += 1
        return original_build(builder, values)

    monkeypatch.setattr(AtmosphereBuilder, "build", counted_build)
    shared = build_multi_dataset_emission_model(
        {"first": config, "second": config},
        spectral_grids={
            "first": _spectral_grid(),
            "second": _spectral_grid(),
        },
    )

    prediction = shared(parameters)

    assert build_calls == 1
    np.testing.assert_array_equal(
        prediction["first"].values, independent["first"].values
    )
    np.testing.assert_array_equal(
        prediction["second"].values, independent["second"].values
    )


def test_shared_atmosphere_multi_dataset_model_rejects_distinct_builders() -> None:
    config = ParameterizedEmissionFactoryConfig(
        planet=Planet(name="Shared b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="Shared", radius_m=7.0e8, effective_temperature_k=5500.0),
        temperature_profile=IsothermalTemperatureProfile(parameter_name="T_iso"),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            parameter_names={"H2O": "log_h2o"},
            parameter_mode="log10",
        ),
        opacity_source=_provider(),
        opacity_binning=None,
        model=ParameterizedEmissionModelConfig(
            opacity_species=("H2O",),
            include_rayleigh=False,
            thermal_integration_backend="numpy",
        ),
    )
    first = build_parameterized_emission_model(
        config,
        spectral_grid=_spectral_grid(),
    )
    second = build_parameterized_emission_model(
        config,
        spectral_grid=_spectral_grid(),
    )

    with pytest.raises(RobertValidationError, match="same AtmosphereBuilder"):
        MultiDatasetEmissionForwardModel({"first": first, "second": second})


def test_parameterized_grey_cloud_model_wraps_existing_regional_hardware() -> None:
    config = ParameterizedEmissionFactoryConfig(
        planet=Planet(name="Generic b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="Generic star", radius_m=7.0e8, effective_temperature_k=5500.0),
        temperature_profile=IsothermalTemperatureProfile(parameter_name="T_iso"),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            parameter_names={"H2O": "log_h2o"},
            parameter_mode="log10",
        ),
        opacity_source=_provider(),
        opacity_binning=None,
        model=ParameterizedEmissionModelConfig(
            opacity_species=("H2O",),
            include_rayleigh=False,
            thermal_integration_backend="numpy",
        ),
    )
    clear = build_parameterized_emission_model(
        config,
        spectral_grid=_spectral_grid(),
    )
    cloudy = ParameterizedGreyCloudEmissionForwardModel(
        planet=clear.planet,
        star=clear.star,
        spectral_grid=clear.spectral_grid,
        atmosphere_builder=clear.atmosphere_builder,
        opacity_provider=clear.opacity_provider,
        config=clear.config,
        cia_table=clear.cia_table,
        geometry=clear.geometry,
        cloud=GreyScatteringCloudConfig(
            log10_mass_extinction_parameter="log_kappa_cloud",
            multiple_scattering_backend="sh4",
        ),
    )

    spectrum = cloudy({"T_iso": 1000.0, "log_h2o": -3.0, "log_kappa_cloud": -2.0})

    assert cloudy.required_parameters == ("T_iso", "log_h2o", "log_kappa_cloud")
    assert cloudy.manifest_metadata["cloud_multiple_scattering_backend"] == "sh4"
    assert cloudy.manifest_metadata["cloud_spectrum_only"] == "true"
    assert cloudy.manifest_metadata["cloud_sh4_spectrum_backend"] == "numpy"
    assert cloudy.manifest_metadata["planet_name"] == "Generic b"
    assert spectrum.observable == "eclipse_depth"
    assert spectrum.metadata["rt_solver"] == "sh4_spectrum_only"
    assert np.all(np.isfinite(spectrum.values))


def test_parameterized_refractive_index_cloud_model_retrieves_n_k_and_particles() -> (
    None
):
    config = ParameterizedEmissionFactoryConfig(
        planet=Planet(name="Generic b", radius_m=7.0e7, gravity_m_s2=20.0),
        star=Star(name="Generic star", radius_m=7.0e8, effective_temperature_k=5500.0),
        temperature_profile=IsothermalTemperatureProfile(parameter_name="T_iso"),
        chemistry_model=FreeChemistry(
            active_species=("H2O",),
            parameter_names={"H2O": "log_h2o"},
            parameter_mode="log10",
        ),
        opacity_source=_provider(),
        opacity_binning=None,
        model=ParameterizedEmissionModelConfig(
            opacity_species=("H2O",),
            include_rayleigh=False,
            thermal_integration_backend="numpy",
        ),
    )
    clear = build_parameterized_emission_model(
        config,
        spectral_grid=_spectral_grid(),
    )
    cloudy = ParameterizedRefractiveIndexCloudEmissionForwardModel(
        planet=clear.planet,
        star=clear.star,
        spectral_grid=clear.spectral_grid,
        atmosphere_builder=clear.atmosphere_builder,
        opacity_provider=clear.opacity_provider,
        config=clear.config,
        cia_table=clear.cia_table,
        geometry=clear.geometry,
        cloud=RefractiveIndexCloudConfig(
            refractive_index_wavelength_micron=(1.5, 5.0),
            real_index_parameter_names=("cloud_n_short", "cloud_n_long"),
            log10_imaginary_index_parameter_names=(
                "cloud_logk_short",
                "cloud_logk_long",
            ),
            log10_condensate_mass_fraction_parameter="log_cloud_mass_fraction",
            log10_effective_radius_micron_parameter="log_cloud_radius",
            particle_density_kg_m3=3000.0,
            geometric_stddev=1.0,
            quadrature_points=1,
        ),
    )
    parameters = {
        "T_iso": 1000.0,
        "log_h2o": -3.0,
        "cloud_n_short": 1.5,
        "cloud_n_long": 1.6,
        "cloud_logk_short": -3.0,
        "cloud_logk_long": -2.0,
        "log_cloud_mass_fraction": -6.0,
        "log_cloud_radius": -1.0,
    }

    spectrum = cloudy(parameters)

    assert cloudy.required_parameters == tuple(parameters)
    assert cloudy.manifest_metadata["cloud_refractive_index_parameterization"] == (
        "nodal_n_log10_k"
    )
    assert cloudy.manifest_metadata["cloud_phase_function_closure"] == (
        "exact_mie_legendre_moments_through_l4"
    )
    assert cloudy.manifest_metadata["cloud_spectrum_only"] == "true"
    assert cloudy.manifest_metadata["cloud_sh4_spectrum_backend"] == "numpy"
    assert spectrum.observable == "eclipse_depth"
    assert np.all(np.isfinite(spectrum.values))


def test_parameterized_mie_cloud_accepts_fixed_catalog_style_refractive_index() -> None:
    fixed = RefractiveIndexSpectrum(
        wavelength_micron=(1.0, 6.0),
        real_index=(1.5, 1.7),
        imaginary_index=(1.0e-3, 1.0e-2),
        name="fixed test material",
    )
    cloud = RefractiveIndexCloudConfig(
        refractive_index_wavelength_micron=(),
        real_index_parameter_names=(),
        log10_imaginary_index_parameter_names=(),
        log10_condensate_mass_fraction_parameter="log_q_cloud",
        log10_effective_radius_micron_parameter="log_r_cloud",
        particle_density_kg_m3=3000.0,
        fixed_refractive_index=fixed,
    )

    assert cloud.required_parameters == ("log_q_cloud", "log_r_cloud")
    assert cloud.fixed_refractive_index is fixed
