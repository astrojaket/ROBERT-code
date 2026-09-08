"""Regression against the compact official-POSEIDON TSLE oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robert_exoplanets import SpectralGrid, Spectrum
from robert_exoplanets.stellar import StellarContaminationModel, StellarHeterogeneity


ORACLE = (
    Path(__file__).parent
    / "fixtures"
    / "poseidon_stellar_contamination_v1_4_0.json"
)


def _planck(wavelength_micron: np.ndarray, temperature_k: float) -> np.ndarray:
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    wavelength_m = wavelength_micron * 1.0e-6
    return (2.0 * h * c**2 / wavelength_m**5) / np.expm1(
        h * c / (wavelength_m * k * temperature_k)
    )


def _stellar(grid: SpectralGrid, values: np.ndarray) -> Spectrum:
    return Spectrum(
        spectral_grid=grid,
        values=values,
        unit="W m^-3 sr^-1",
        observable="stellar_spectral_radiance",
    )


def _planet_depth(wavelength_micron: np.ndarray) -> np.ndarray:
    return (
        0.008
        + 2.5e-4 * np.exp(-0.5 * ((wavelength_micron - 1.4) / 0.12) ** 2)
        + 1.7e-4 * np.exp(-0.5 * ((wavelength_micron - 4.3) / 0.25) ** 2)
        + 3.0e-5 * np.sin(2.0 * np.pi * np.log(wavelength_micron / 0.6))
    )


def test_robert_matches_official_poseidon_v1_4_transform_oracle() -> None:
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    tolerances = oracle["tolerances"]
    factor_residuals = []
    depth_residuals = []

    assert oracle["passed"] is True
    assert oracle["poseidon"]["version"] == "1.4.0"
    assert oracle["poseidon"]["commit"] == (
        "d1632214c9f3087e367a8d752454f3668fc30e18"
    )
    for case in oracle["cases"]:
        wavelength = np.asarray(case["sample_wavelength_micron"], dtype=float)
        grid = SpectralGrid.from_array(wavelength, unit="micron", role="rt_native")
        photosphere = _stellar(grid, _planck(wavelength, 5200.0))
        regions = tuple(
            StellarHeterogeneity(
                name=kind,
                kind=kind,
                spectrum=_stellar(grid, _planck(wavelength, temperature)),
                covering_fraction=fraction,
            )
            for kind, temperature, fraction in zip(
                (
                    ()
                    if case["name"] == "homogeneous"
                    else (
                        ("spot",)
                        if case["name"] == "cool_spot"
                        else (
                            ("facula",)
                            if case["name"] == "hot_facula"
                            else ("spot", "facula")
                        )
                    )
                ),
                case["temperatures_k"],
                case["fractions"],
                strict=True,
            )
        )
        factor = StellarContaminationModel(
            photosphere,
            regions,
        ).evaluate().contamination_factor.values
        expected_factor = np.asarray(case["poseidon_contamination_factor"])
        expected_depth = np.asarray(case["poseidon_observed_depth"])
        factor_residuals.append(factor - expected_factor)
        depth_residuals.append(factor * _planet_depth(wavelength) - expected_depth)

    factor_residual = np.concatenate(factor_residuals)
    depth_residual = np.concatenate(depth_residuals)
    assert np.max(np.abs(factor_residual)) <= tolerances["max_abs_factor_residual"]
    assert np.max(np.abs(depth_residual)) <= tolerances["max_abs_depth_residual"]
    assert np.sqrt(np.mean(depth_residual**2)) <= tolerances["rms_depth_residual"]
