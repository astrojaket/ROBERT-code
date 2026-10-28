"""Tests for physical cloud-profile diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    ParameterizedDeckHazeCloudModel,
    ParameterizedMieCloudModel,
    PressureGrid,
    RefractiveIndexSpectrum,
    SpectralGrid,
    grey_cloud_deck,
)
from robert_exoplanets.diagnostics.cloud_profiles import (
    CloudProfileUnavailableError,
    cloud_profiles,
)
from robert_exoplanets.forward.clouds import pressure_slab_layer_fractions
from robert_exoplanets.forward import virga_cloud
from robert_exoplanets.forward.virga_cloud import VirgaEddySedCloudModel


def _atmosphere() -> AtmosphereState:
    pressure_grid = PressureGrid.logspace(1.0e-5, 1.0, 4, unit="bar")
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.full(pressure_grid.n_layers, 1200.0),
        temperature_edges=np.full(pressure_grid.n_layers + 1, 1200.0),
        composition={"H2": np.full(pressure_grid.n_layers, 0.85)},
        mean_molecular_weight=np.full(pressure_grid.n_layers, 2.3),
    )


def _mie_cloud() -> ParameterizedMieCloudModel:
    return ParameterizedMieCloudModel(
        refractive_index_wavelength_micron=(),
        real_index_parameter_names=(),
        log10_imaginary_index_parameter_names=(),
        fixed_refractive_index=RefractiveIndexSpectrum(
            wavelength_micron=[1.0, 5.0],
            real_index=[1.5, 1.5],
            imaginary_index=[1.0e-3, 1.0e-3],
        ),
        log10_condensate_mass_fraction_parameter="log_mass",
        log10_effective_radius_micron_parameter="log_radius",
        log10_cloud_top_pressure_bar_parameter="log_top",
        log10_cloud_base_pressure_bar_parameter="log_base",
        particle_density_kg_m3=3000.0,
    )


def test_mie_profiles_use_fractional_slab_mass_and_radius_in_microns() -> None:
    atmosphere = _atmosphere()
    cloud = _mie_cloud()
    parameters = {
        "log_mass": -4.0,
        "log_radius": -0.7,
        "log_top": -3.0,
        "log_base": -1.0,
    }

    profiles = cloud_profiles(cloud, atmosphere, parameters)

    expected_fraction = pressure_slab_layer_fractions(
        atmosphere.pressure_grid,
        top_pressure_bar=1.0e-3,
        base_pressure_bar=1.0e-1,
    )
    np.testing.assert_allclose(
        profiles["condensate_mass_fraction"]["values"],
        expected_fraction * 1.0e-4,
    )
    np.testing.assert_allclose(
        profiles["particle_radius"]["values"],
        np.full(atmosphere.n_layers, 10.0**-0.7),
    )
    assert profiles["condensate_mass_fraction"]["unit"] == "kg/kg"
    assert profiles["particle_radius"]["unit"] == "micron"
    assert not profiles["particle_radius"]["values"].flags.writeable


def test_deck_haze_profiles_use_existing_layer_optical_depth_equations() -> None:
    atmosphere = _atmosphere()
    cloud = ParameterizedDeckHazeCloudModel(haze_reference_wavelength_micron=2.0)
    owner = SimpleNamespace(cloud_model=cloud, gravity_m_s2=10.0)
    parameters = {
        "log_cloud_top_pressure_bar": -3.0,
        "log_cloud_optical_depth": np.log10(0.6),
        "log_haze_mass_extinction": np.log10(2.0),
        "haze_slope": -4.0,
    }

    profiles = cloud_profiles(owner, atmosphere, parameters)

    deck_name = "deck_extinction_optical_depth_at_2_micron"
    haze_name = "haze_extinction_optical_depth_at_2_micron"
    reference_grid = SpectralGrid.from_array([2.0], unit="micron", role="diagnostic")
    expected_deck = grey_cloud_deck(
        atmosphere.pressure_grid,
        reference_grid,
        cloud_top_pressure=1.0e-3,
        cloud_top_pressure_unit="bar",
        optical_depth=0.6,
    )
    pressure_edges_pa = atmosphere.pressure_grid.edges * 1.0e5
    expected_haze = 2.0 * 0.1 * np.abs(np.diff(pressure_edges_pa)) / 10.0

    np.testing.assert_allclose(
        profiles[deck_name]["values"], expected_deck.extinction_tau[:, 0]
    )
    np.testing.assert_allclose(profiles[haze_name]["values"], expected_haze)
    assert profiles[deck_name]["unit"] == "dimensionless"
    assert profiles[haze_name]["unit"] == "dimensionless"
    assert reference_grid.role == "diagnostic"


def test_transmission_owner_uses_its_hydrostatic_gravity_profile() -> None:
    atmosphere = _atmosphere()
    cloud = ParameterizedDeckHazeCloudModel()
    owner = SimpleNamespace(
        cloud_model=cloud,
        planet=SimpleNamespace(radius_m=7.0e7),
        config=SimpleNamespace(
            radius_scale_parameter=None,
            gravity_model="constant",
            reference_pressure_bar=1.0e-2,
        ),
        _reference_gravity=lambda radius: 10.0,
    )
    parameters = {
        "log_cloud_top_pressure_bar": -3.0,
        "log_cloud_optical_depth": -1.0,
        "log_haze_mass_extinction": 0.0,
        "haze_slope": -4.0,
    }

    profiles = cloud_profiles(owner, atmosphere, parameters)

    pressure_edges_pa = atmosphere.pressure_grid.edges * 1.0e5
    expected_haze = 0.1 * np.abs(np.diff(pressure_edges_pa)) / 10.0
    np.testing.assert_allclose(
        profiles["haze_extinction_optical_depth_at_1_micron"]["values"],
        expected_haze,
    )


def test_cloud_free_returns_empty_and_custom_cloud_is_explicitly_unavailable() -> None:
    atmosphere = _atmosphere()

    assert cloud_profiles(SimpleNamespace(cloud_model=None), atmosphere, {}) == {}

    with pytest.raises(
        CloudProfileUnavailableError, match="legacy cloud configuration"
    ):
        cloud_profiles(
            SimpleNamespace(cloud_model=None, cloud=object()), atmosphere, {}
        )

    with pytest.raises(CloudProfileUnavailableError, match="unsupported cloud model"):
        cloud_profiles(SimpleNamespace(cloud_model=object()), atmosphere, {})


def test_virga_profiles_use_physical_species_arrays_and_forward_owner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mie_directory = tmp_path / "mie"
    mie_directory.mkdir()
    (mie_directory / "MgSiO3.mieff").write_text("test Mie table", encoding="utf-8")
    cloud = VirgaEddySedCloudModel(
        condensibles=("MgSiO3",),
        gas_mass_mixing_ratios={"MgSiO3": 1.0e-4},
        mie_directory=mie_directory,
        fsed=None,
        fsed_parameter="virga_fsed",
        kz_cm2_s=None,
        log10_kz_cm2_s_parameter="virga_log10_kzz",
    )
    atmosphere_objects = []

    class FakeAtmosphere:
        def __init__(self, condensibles, **kwargs):
            self.condensibles = tuple(condensibles)
            self.constructor_kwargs = kwargs
            self.gravity_kwargs = None
            self.ptk_kwargs = None
            atmosphere_objects.append(self)

        def gravity(self, **kwargs):
            self.gravity_kwargs = kwargs

        def ptk(self, **kwargs):
            self.ptk_kwargs = kwargs

    gas_properties = SimpleNamespace(
        MgSiO3=lambda mmw, *, mh=1.0, gas_mmr=0.0: (
            100.4,
            float(gas_mmr) * 100.4 / mmw,
            1.0,
        )
    )

    def compute(virga_atmosphere, **kwargs):
        assert virga_atmosphere in atmosphere_objects
        grid = _atmosphere().pressure_grid
        ascending_edges = np.sort(grid.edges)
        return {
            "pressure": 0.5 * (ascending_edges[:-1] + ascending_edges[1:]),
            "pressure_unit": "bar",
            "condensate_mmr": np.arange(1.0, grid.n_layers + 1.0)[:, None] * 1.0e-5,
            "droplet_eff_r": np.arange(10.0, 10.0 + grid.n_layers)[:, None],
            "r_units": "micron",
            "condensibles": ["MgSiO3"],
        }

    fake = SimpleNamespace(
        Atmosphere=FakeAtmosphere,
        compute=compute,
        gas_properties=gas_properties,
    )
    monkeypatch.setattr(virga_cloud, "_load_virga", lambda: (fake, "2.0.2"))
    atmosphere = _atmosphere()
    owner = SimpleNamespace(cloud_model=cloud, gravity_m_s2=15.0)
    parameters = {"virga_fsed": 0.6, "virga_log10_kzz": 8.0}

    profiles = cloud_profiles(owner, atmosphere, parameters)

    np.testing.assert_allclose(
        profiles["condensate_mass_mixing_ratio_MgSiO3"]["values"],
        np.arange(1.0, atmosphere.n_layers + 1.0) * 1.0e-5,
    )
    np.testing.assert_allclose(
        profiles["effective_particle_radius_MgSiO3"]["values"],
        np.arange(10.0, 10.0 + atmosphere.n_layers),
    )
    assert profiles["condensate_mass_mixing_ratio_MgSiO3"]["unit"] == "kg/kg"
    assert profiles["effective_particle_radius_MgSiO3"]["unit"] == "micron"
    assert not profiles["condensate_mass_mixing_ratio_MgSiO3"]["values"].flags.writeable
    assert not profiles["effective_particle_radius_MgSiO3"]["values"].flags.writeable
    assert len(atmosphere_objects) == 1
    assert atmosphere_objects[0].constructor_kwargs["fsed"] == pytest.approx(0.6)
    assert atmosphere_objects[0].ptk_kwargs["kz_min"] == pytest.approx(1.0e8)
    assert atmosphere_objects[0].gravity_kwargs["gravity"] == pytest.approx(1500.0)

    with pytest.raises(
        CloudProfileUnavailableError,
        match="owning forward model's physical gravity profile",
    ):
        cloud_profiles(cloud, atmosphere, parameters)
