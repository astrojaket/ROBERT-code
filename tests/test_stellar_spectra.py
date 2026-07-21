"""Unit tests for prepared stellar spectra without external atlas data."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    BlackbodyStellarSpectrumModel,
    PhoenixStellarSpectrumModel,
    SpectralGrid,
    Star,
    planck_radiance_wavelength,
    prepare_stellar_spectrum,
)
from robert_exoplanets.core import RobertConfigError, RobertValidationError
from robert_exoplanets.stellar import spectra as stellar_spectra


def test_blackbody_stellar_model_matches_planck_radiance() -> None:
    grid = SpectralGrid.from_array([2.0, 5.0, 10.0], unit="micron")
    star = Star(name="Synthetic", effective_temperature_k=5778.0)

    spectrum = BlackbodyStellarSpectrumModel().prepare(star, grid)

    np.testing.assert_allclose(
        spectrum.values,
        planck_radiance_wavelength(grid.values, 5778.0),
        rtol=1.0e-13,
    )
    assert spectrum.unit == "W m^-3 sr^-1"
    assert spectrum.observable == "stellar_spectral_radiance"
    assert spectrum.metadata["stellar_model"] == "blackbody"


def test_phoenix_is_default_and_requires_complete_stellar_parameters() -> None:
    grid = SpectralGrid.from_array([2.0], unit="micron")
    incomplete = Star(name="Incomplete", effective_temperature_k=5778.0)

    with pytest.raises(RobertValidationError, match="metallicity_dex"):
        prepare_stellar_spectrum(incomplete, grid)
    with pytest.raises(RobertConfigError, match="phoenix.*blackbody"):
        prepare_stellar_spectrum(incomplete, grid, model="kurucz")


def test_phoenix_surface_flux_is_bin_averaged_and_converted_to_radiance(
    monkeypatch,
) -> None:
    from astropy import units as u

    class FakePhoenixSpectrum:
        waveset = np.linspace(1.0e4, 3.0e4, 21) * u.AA

        def __call__(self, wavelength, *, flux_unit):
            wavelength_angstrom = wavelength.to_value(u.AA)
            # Linear FLAM surface flux: exact bin averages are 15 and 25.
            return (wavelength_angstrom / 1.0e3) * flux_unit

    monkeypatch.setattr(
        stellar_spectra,
        "_load_phoenix_source_spectrum",
        lambda *_: FakePhoenixSpectrum(),
    )
    grid = SpectralGrid(
        values=np.array([1.5, 2.5]),
        bin_edges=np.array([1.0, 2.0, 3.0]),
        unit="micron",
    )
    star = Star(
        name="G2V",
        effective_temperature_k=5778.0,
        metallicity_dex=0.0,
        log_g_cgs=4.44,
    )

    spectrum = PhoenixStellarSpectrumModel(
        bolometric_normalization=False
    ).prepare(star, grid)

    np.testing.assert_allclose(spectrum.values, np.array([15.0, 25.0]) * 1.0e7 / np.pi)
    assert spectrum.metadata["spectral_sampling"] == "flux_conserving_bin_average"
    assert spectrum.metadata["surface_flux_convention"] == (
        "phoenix_f_lambda_divided_by_pi"
    )


def test_phoenix_loader_requires_stsci_reference_root(monkeypatch) -> None:
    stellar_spectra._load_phoenix_source_spectrum.cache_clear()
    monkeypatch.delenv("PYSYN_CDBS", raising=False)

    with pytest.raises(RobertConfigError, match="PYSYN_CDBS"):
        stellar_spectra._load_phoenix_source_spectrum(5778.0, 0.0, 4.44)


def test_phoenix_preparation_rejects_spectral_extrapolation(monkeypatch) -> None:
    from astropy import units as u

    class FakePhoenixSpectrum:
        waveset = np.linspace(1.0e4, 3.0e4, 21) * u.AA

        def __call__(self, wavelength, *, flux_unit):
            return np.ones_like(wavelength.to_value(u.AA)) * flux_unit

    monkeypatch.setattr(
        stellar_spectra,
        "_load_phoenix_source_spectrum",
        lambda *_: FakePhoenixSpectrum(),
    )
    star = Star(
        name="G2V",
        effective_temperature_k=5778.0,
        metallicity_dex=0.0,
        log_g_cgs=4.44,
    )

    with pytest.raises(RobertValidationError, match="outside PHOENIX coverage"):
        PhoenixStellarSpectrumModel(bolometric_normalization=False).prepare(
            star,
            SpectralGrid.from_array([0.5, 2.0], unit="micron"),
        )
