"""Tests for cloud-type-agnostic refractive-index and Mie cloud physics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    OpticalConstantsCatalog,
    ParameterizedCloudModel,
    ParameterizedMieCloudModel,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    RefractiveIndexSpectrum,
    SpectralGrid,
    assemble_gas_optical_depth,
    load_refractive_index_csv,
    load_refractive_index_table,
    lognormal_mie_optics,
    mie_cloud_from_mass_fraction,
    mie_efficiencies,
    mie_phase_function_moments,
    refractive_index_from_parameters,
)
from robert_exoplanets.core import RobertValidationError


def test_refractive_index_interpolates_n_and_positive_k_in_log_space() -> None:
    index = RefractiveIndexSpectrum(
        wavelength_micron=[1.0, 4.0],
        real_index=[1.4, 1.6],
        imaginary_index=[1.0e-4, 1.0e-2],
        name="synthetic material",
    )

    evaluated = index.evaluate([2.0])

    np.testing.assert_allclose(evaluated.real, [1.5])
    np.testing.assert_allclose(evaluated.imag, [1.0e-3])
    with pytest.raises(RobertValidationError, match="outside refractive-index coverage"):
        index.evaluate([0.5])
    clipped = index.evaluate([0.5], extrapolation="clip")
    np.testing.assert_allclose(clipped, [1.4 + 1.0e-4j])


def test_refractive_index_readers_support_headerless_and_csv_tables(tmp_path: Path) -> None:
    text_path = tmp_path / "material.txt"
    text_path.write_text("1000 1.5 0.01\n2000 1.6 0.02\n", encoding="utf-8")
    csv_path = tmp_path / "material.csv"
    csv_path.write_text(
        "wavelength_nm,n,k\n1000,1.5,0.01\n2000,1.6,0.02\n",
        encoding="utf-8",
    )

    text_index = load_refractive_index_table(text_path, wavelength_unit="nm")
    csv_index = load_refractive_index_csv(csv_path)

    np.testing.assert_allclose(text_index.wavelength_micron, [1.0, 2.0])
    np.testing.assert_allclose(csv_index.wavelength_micron, [1.0, 2.0])
    np.testing.assert_allclose(text_index.real_index, csv_index.real_index)
    np.testing.assert_allclose(text_index.imaginary_index, csv_index.imaginary_index)


def test_exo_skryer_catalog_selects_material_and_records_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = OpticalConstantsCatalog(root / "data" / "optical_constants" / "exo_skryer")

    material = catalog.load("MgSiO3")

    assert len(catalog.materials) == 44
    assert "MgSiO3" in catalog.materials
    assert "White" not in catalog.materials
    assert material.name == "MgSiO3"
    assert material.metadata["source_format"] == "exo_skryer_nk_data"
    assert len(material.metadata["source_sha256"]) == 64
    np.testing.assert_allclose(material.wavelength_micron[[0, -1]], [0.2, 500.0])


def test_small_particle_mie_solution_matches_rayleigh_limit() -> None:
    size_parameter = 1.0e-4
    refractive_index = 1.5 + 0.01j
    polarizability = (refractive_index**2 - 1.0) / (refractive_index**2 + 2.0)
    expected_scattering = (8.0 / 3.0) * size_parameter**4 * abs(polarizability) ** 2
    expected_absorption = 4.0 * size_parameter * polarizability.imag

    extinction, scattering, asymmetry = mie_efficiencies(size_parameter, refractive_index)

    assert extinction == pytest.approx(expected_absorption + expected_scattering)
    assert scattering == pytest.approx(expected_scattering)
    assert asymmetry == 0.0


def test_nonabsorbing_mie_particle_conserves_extinction_as_scattering() -> None:
    extinction, scattering, asymmetry = mie_efficiencies(1.0, 1.5 + 0.0j)

    assert extinction == pytest.approx(scattering, rel=1.0e-12)
    assert extinction > 0.0
    assert -1.0 <= asymmetry <= 1.0


@pytest.mark.parametrize(
    ("size_parameter", "index", "expected_extinction", "expected_scattering"),
    [
        (0.1, 1.5 + 0.01j, 0.00202731297845712, 0.0000230934857364471),
        (5.0, 1.6 + 0.1j, 2.81140960007493, 1.59241830576119),
        (20.0, 1.3 + 0.02j, 2.37744056002414, 1.50573567769399),
    ],
)
def test_absorbing_mie_efficiencies_match_independent_riccati_bessel_reference(
    size_parameter: float,
    index: complex,
    expected_extinction: float,
    expected_scattering: float,
) -> None:
    extinction, scattering, _ = mie_efficiencies(size_parameter, index)

    assert extinction == pytest.approx(expected_extinction, rel=2.0e-12)
    assert scattering == pytest.approx(expected_scattering, rel=2.0e-12)


def test_mie_phase_moments_match_independent_miepython_reference() -> None:
    moments = mie_phase_function_moments(5.0, 1.6 + 0.1j)

    # miepython 3.2.0, 256-point Gauss-Legendre integration of its normalized
    # unpolarized phase function using the passive n-ik convention.
    expected = np.array(
        [1.0, 2.324914198318782, 3.181268856917814, 3.596168330101814, 3.979001987734556]
    )
    np.testing.assert_allclose(moments, expected, rtol=8.0e-13, atol=2.0e-13)
    assert moments[1] / 3.0 == pytest.approx(
        mie_efficiencies(5.0, 1.6 + 0.1j)[2], rel=2.0e-14
    )


def test_rayleigh_phase_moment_limit_is_recovered() -> None:
    moments = mie_phase_function_moments(1.0e-5, 1.5 + 0.01j)

    np.testing.assert_allclose(moments, [1.0, 0.0, 0.5, 0.0, 0.0])


def test_mie_mass_opacity_and_cloud_tau_follow_particle_and_hydrostatic_mass() -> None:
    spectral_grid = SpectralGrid.from_array([2.0, 4.0], unit="micron", role="opacity")
    index = RefractiveIndexSpectrum(
        wavelength_micron=[1.0, 5.0],
        real_index=[1.6, 1.6],
        imaginary_index=[0.0, 0.0],
    )
    optics = lognormal_mie_optics(
        index,
        spectral_grid,
        effective_radius_micron=0.2,
        geometric_stddev=1.0,
        particle_density_kg_m3=3000.0,
    )
    gas_tau = _zero_gas_tau(spectral_grid)
    cloud = mie_cloud_from_mass_fraction(
        gas_tau,
        optics,
        condensate_mass_fraction=1.0e-4,
    )

    expected = (
        gas_tau.layer_pressure_thickness_pa[:, None]
        / gas_tau.gravity_m_s2
        * 1.0e-4
        * optics.mass_extinction_m2_kg[None, :]
    )
    np.testing.assert_allclose(cloud.extinction_tau, expected)
    np.testing.assert_allclose(cloud.single_scattering_albedo, 1.0, atol=1.0e-12)
    assert np.all(optics.mass_extinction_m2_kg > 0.0)
    assert cloud.metadata["optical_model"] == "homogeneous_sphere_mie"
    np.testing.assert_allclose(
        cloud.phase_function_moments[:, 0],
        optics.phase_function_moments,
    )
    assert optics.metadata["radius_definition"] == "effective_radius=<r3>/<r2>"


def test_retrieved_refractive_index_uses_n_and_log10_k_nodes() -> None:
    index = refractive_index_from_parameters(
        [1.0, 3.0],
        {"n0": 1.4, "n1": 1.6, "logk0": -4.0, "logk1": -2.0},
        real_parameter_names=("n0", "n1"),
        log10_imaginary_parameter_names=("logk0", "logk1"),
    )

    np.testing.assert_allclose(index.real_index, [1.4, 1.6])
    np.testing.assert_allclose(index.imaginary_index, [1.0e-4, 1.0e-2])
    assert index.metadata["parameterization"] == "nodal_n_log10_k"


def test_parameterized_mie_cloud_is_geometry_independent_and_matches_core_physics() -> None:
    spectral_grid = SpectralGrid.from_array([2.0, 4.0], unit="micron", role="opacity")
    index = RefractiveIndexSpectrum(
        wavelength_micron=[1.0, 5.0],
        real_index=[1.6, 1.6],
        imaginary_index=[1.0e-3, 1.0e-3],
        name="shared synthetic particles",
    )
    gas_tau = _zero_gas_tau(spectral_grid)
    model = ParameterizedMieCloudModel(
        refractive_index_wavelength_micron=(),
        real_index_parameter_names=(),
        log10_imaginary_index_parameter_names=(),
        fixed_refractive_index=index,
        log10_condensate_mass_fraction_parameter="log_cloud_mass_fraction",
        log10_effective_radius_micron_parameter="log_cloud_radius_micron",
        particle_density_kg_m3=3000.0,
        geometric_stddev=1.0,
        log10_cloud_top_pressure_bar_parameter="log_cloud_top_pressure_bar",
    )
    parameters = {
        "log_cloud_mass_fraction": -4.0,
        "log_cloud_radius_micron": np.log10(0.2),
        "log_cloud_top_pressure_bar": -3.0,
    }

    (shared_cloud,) = model.evaluate(gas_tau, parameters)
    optics = lognormal_mie_optics(
        index,
        spectral_grid,
        effective_radius_micron=0.2,
        geometric_stddev=1.0,
        particle_density_kg_m3=3000.0,
    )
    expected = mie_cloud_from_mass_fraction(
        gas_tau,
        optics,
        condensate_mass_fraction=np.array([0.0, 1.0e-4]),
    )

    assert isinstance(model, ParameterizedCloudModel)
    np.testing.assert_allclose(shared_cloud.extinction_tau, expected.extinction_tau)
    np.testing.assert_allclose(
        shared_cloud.single_scattering_albedo,
        expected.single_scattering_albedo,
    )
    assert model.manifest_metadata["cloud_geometry_independent"] == "true"


def _zero_gas_tau(spectral_grid: SpectralGrid):
    pressure_grid = PressureGrid(
        edges=np.array([1.0e-5, 1.0e-3, 1.0e-1]),
        centers=np.array([1.0e-4, 1.0e-2]),
        unit="bar",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([800.0, 1000.0]),
        composition={"H2O": np.full(2, 1.0e-3)},
        mean_molecular_weight=2.3,
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="mie-test",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, 2, spectral_grid.size, 1)),
        unit="m^2/molecule",
    )
    return assemble_gas_optical_depth(atmosphere, opacity, gravity_m_s2=10.0)
