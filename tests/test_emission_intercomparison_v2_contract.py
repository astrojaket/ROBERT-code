"""Focused tests for the emission intercomparison Version-2 common contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    BOLTZMANN_CONSTANT_J_K,
    SPEED_OF_LIGHT_M_S,
    Version2CommonContract,
    build_version_2_common_contract,
    flux_conserving_bin_mean,
    isolated_molecule_composition,
    load_version_2_common_contract,
    payload_sha256,
    planck_surface_flux_w_m2_m,
)


@pytest.fixture(scope="module")
def contract() -> Version2CommonContract:
    return build_version_2_common_contract()


def test_source_values_convert_to_frozen_si(contract: Version2CommonContract) -> None:
    constants = contract.constants
    measurements = contract.measurements

    assert measurements["planet_mass"].si_value == pytest.approx(
        0.477 * constants["M_J_kg"], rel=2e-16
    )
    assert measurements["planet_radius"].si_value == pytest.approx(
        1.932 * constants["R_J_m"], rel=2e-16
    )
    assert measurements["stellar_mass"].si_value == pytest.approx(
        1.286 * constants["M_sun_kg"], rel=2e-16
    )
    assert measurements["stellar_radius"].si_value == pytest.approx(
        1.583 * constants["R_sun_m"], rel=2e-16
    )
    assert measurements["semimajor_axis"].si_value == pytest.approx(
        0.05135 * constants["AU_m"], rel=2e-16
    )
    assert measurements["distance"].si_value == pytest.approx(
        405.908 * constants["pc_m"], rel=2e-13
    )
    assert measurements["orbital_period"].si_value == pytest.approx(
        3.73548546 * constants["day_s"], rel=0.0, abs=0.0
    )


def test_derived_wasp17_quantities_are_frozen(contract: Version2CommonContract) -> None:
    derived = contract.derived

    assert derived["surface_gravity_m_s2"] == pytest.approx(3.1673143851664745)
    assert derived["radius_ratio"] == pytest.approx(0.12541851392694642)
    assert derived["projected_area_ratio"] == pytest.approx(0.015729803635643653)
    assert derived[
        "equilibrium_temperature_full_redistribution_zero_albedo_k"
    ] == pytest.approx(1753.657719, abs=1e-6)
    assert derived["substellar_irradiation_temperature_k"] == pytest.approx(
        2480.046530, abs=1e-6
    )


def test_blackbody_has_explicit_units_and_stable_limits(
    contract: Version2CommonContract,
) -> None:
    wavelength = np.array([0.3, 12.0, 1.0e5])
    temperature = 6550.0
    flux = planck_surface_flux_w_m2_m(wavelength, temperature)
    rayleigh_jeans = (
        np.pi
        * 2.0
        * SPEED_OF_LIGHT_M_S
        * BOLTZMANN_CONSTANT_J_K
        * temperature
        / (wavelength[-1] * 1.0e-6) ** 4
    )

    assert np.all(np.isfinite(flux))
    assert np.all(flux > 0.0)
    assert flux[-1] == pytest.approx(rayleigh_jeans, rel=2e-5)
    assert contract.to_dict()["stellar_blackbody"]["surface_flux_unit"] == "W m^-3"
    assert contract.stellar_surface_flux_native_w_m2_m.flags.writeable is False


def test_vmr_sum_background_rule_and_mean_molecular_weight(
    contract: Version2CommonContract,
) -> None:
    assert sum(contract.composition_vmr.values()) == 1.0
    active = sum(
        contract.composition_vmr[name] for name in ("H2O", "CO", "CO2", "CH4")
    )
    remainder = 1.0 - active
    assert contract.composition_vmr["H2"] == pytest.approx(0.8547 * remainder)
    assert contract.composition_vmr["He"] == pytest.approx(0.1453 * remainder)
    assert contract.mean_molecular_weight_u == pytest.approx(
        2.321438174776293, rel=0.0, abs=2e-12
    )
    assert "not_runtime_chemistry" in contract.to_dict()["composition"]["provenance_state"]


def test_pg14_profiles_have_frozen_shapes_and_reproducible_checksums(
    contract: Version2CommonContract,
) -> None:
    rebuilt = build_version_2_common_contract()
    non_inverted = contract.temperature_profiles_k["pg14_non_inverted_80_cells"]
    inverted = contract.temperature_profiles_k["pg14_inverted_80_cells"]

    assert non_inverted.shape == inverted.shape == (80,)
    assert np.all(np.diff(non_inverted) > 0.0)
    assert np.any(np.diff(inverted) > 0.0)
    assert np.any(np.diff(inverted) < 0.0)
    for name, values in contract.temperature_profiles_k.items():
        np.testing.assert_array_equal(values, rebuilt.temperature_profiles_k[name])
    assert len(contract.pg14_implementation_sha256) == 64
    assert len(contract.pg14_method_sha256) == 64
    assert contract.pg14_implementation_sha256 == rebuilt.pg14_implementation_sha256
    assert contract.pg14_method_sha256 == rebuilt.pg14_method_sha256


def test_pressure_edges_centers_and_framework_mappings(
    contract: Version2CommonContract,
) -> None:
    assert tuple(grid.n_cells for grid in contract.pressure_grids) == (40, 80, 160)
    assert contract.primary_pressure_grid.n_cells == 80
    for grid in contract.pressure_grids:
        assert grid.edges_bar.shape == (grid.n_cells + 1,)
        assert grid.centers_bar.shape == (grid.n_cells,)
        np.testing.assert_array_equal(grid.picaso_levels_bar, grid.edges_bar)
        np.testing.assert_array_equal(grid.petitradtrans_nodes_bar, grid.centers_bar)
        np.testing.assert_allclose(
            grid.centers_bar, np.sqrt(grid.edges_bar[:-1] * grid.edges_bar[1:])
        )
        assert grid.orientation == "top_to_bottom_increasing_pressure"


def test_r100_grid_edges_and_flux_conservation(contract: Version2CommonContract) -> None:
    spectral = contract.spectral
    wavelength = spectral.native_reference_wavelength_micron
    linear = 2.5 + 0.75 * wavelength
    binned = flux_conserving_bin_mean(
        wavelength, linear, spectral.r100_edges_micron
    )
    widths = np.diff(spectral.r100_edges_micron)

    assert spectral.r100_edges_micron[0] == 0.3
    assert spectral.r100_edges_micron[-1] == 12.0
    assert spectral.r100_centers_micron.size == 369
    assert np.all(np.diff(spectral.r100_edges_micron) > 0.0)
    assert np.sum(binned * widths) == pytest.approx(
        np.trapezoid(linear, wavelength), rel=2e-14
    )
    expected = 2.5 + 0.75 * 0.5 * (
        spectral.r100_edges_micron[:-1] + spectral.r100_edges_micron[1:]
    )
    np.testing.assert_allclose(binned, expected, rtol=2e-14)


def test_picaso_correlated_k_assets_and_environment_are_frozen(
    contract: Version2CommonContract,
) -> None:
    expected = {
        "CH4": "1474ca8c5236c9f7571aabb08b9d983ac8d244a55561a3348d078e9c7b31758f",
        "CO2": "83feeddf1f3de9f385c6dc21650636af8a95ae5619aadcfaa43e51e9ae2d1510",
        "CO": "96be68a9b1dce6e645c1fea28026fdf8b5f00dec20e6abda2f99693421f65cec",
        "H2O": "9be15e41e59dc8689fb2f4d0992c8cef07dcc4eb3bd86a0b537d981f906b6672",
    }
    opacity = contract.to_dict()["opacity_contract"]

    assert {
        species: asset.sha256
        for species, asset in contract.picaso_correlated_k_assets.items()
    } == expected
    assert opacity["picaso_primary_molecular_representation_from_stage_2"] == (
        "correlated_k_resort_rebin"
    )
    assert opacity["picaso_opacity_sampling_role"] == (
        "retired_not_run_under_0p3_to_12_contract"
    )
    assert opacity["asset_family"]["zenodo_doi"] == "10.5281/zenodo.18644980"
    assert opacity["asset_family"]["covers_version_2_domain"] is True
    environment = opacity["version_2_picaso_environment"]
    assert environment["interpreter"] == "/opt/miniconda3/envs/picaso-v4/bin/python"
    assert environment["versions"]["picaso"] == "4.0"
    assert environment["correlated_k_smoke"]["status"] == "pass"
    assert environment["correlated_k_smoke"]["taugas_shape"] == [40, 661, 8]
    assert environment["correlated_k_smoke"][
        "finite_positive_bins_in_frozen_domain"
    ] == 613
    assert environment["reference"]["official_git_commit"] == (
        "0369089372f748609dd0233e6de9361af31a38cf"
    )
    assert opacity["historical_version_1_picaso_environment"][
        "version_2_molecular_use_forbidden"
    ] is True


def test_schema_round_trip_and_checksum(contract: Version2CommonContract) -> None:
    payload = contract.to_dict()
    serialized = json.loads(json.dumps(payload))
    restored = Version2CommonContract.from_dict(serialized)

    assert restored.to_dict() == payload
    without_checksum = dict(payload)
    supplied = without_checksum.pop("contract_sha256")
    assert supplied == payload_sha256(without_checksum)
    serialized["physical_constants"]["G_m3_kg_s2"] = 1.0
    with pytest.raises(RobertValidationError, match="checksum mismatch"):
        Version2CommonContract.from_dict(serialized)


def test_contract_is_deeply_immutable(contract: Version2CommonContract) -> None:
    with pytest.raises(TypeError):
        contract.constants["G_m3_kg_s2"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError):
        contract.primary_pressure_grid.edges_bar[0] = 1.0
    with pytest.raises(ValueError):
        contract.temperature_profiles_k["isothermal_80_cells"][0] = 1.0
    with pytest.raises(FrozenInstanceError):
        contract.isothermal_temperature_k = 1.0  # type: ignore[misc]


def test_authoritative_contract_loader_and_isolated_molecule_fill() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/data/emission_intercomparison/version_2/common_contract.json"
    )
    loaded = load_version_2_common_contract(path)
    active_species = ("H2O", "CO", "CO2", "CH4")
    reference_background_ratio = (
        loaded.composition_vmr["H2"] / loaded.composition_vmr["He"]
    )

    for species in active_species:
        composition = isolated_molecule_composition(loaded, species)
        assert composition[species] == loaded.composition_vmr[species]
        assert sum(composition.values()) == pytest.approx(1.0, abs=2e-16)
        assert composition["H2"] / composition["He"] == pytest.approx(
            reference_background_ratio
        )
        assert all(
            composition[other] == 0.0
            for other in active_species
            if other != species
        )
        with pytest.raises(TypeError):
            composition["H2"] = 0.0  # type: ignore[index]

    with pytest.raises(RobertValidationError, match="unsupported frozen molecule"):
        isolated_molecule_composition(loaded, "NH3")
    with pytest.raises(RobertValidationError, match="finite and positive"):
        isolated_molecule_composition(loaded, "H2O", abundance_scale=0.0)
