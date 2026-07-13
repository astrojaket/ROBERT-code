"""Tests for conversion into the portable ROBERT observation format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robert_exoplanets import (
    convert_emission_observation_table,
    load_emission_observation_npz,
    load_emission_observation_table,
)


def test_convert_named_csv_from_ppm_and_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "unity.csv"
    source.write_text(
        "wavelength_nm,eclipse_ppm,error_ppm,bin_low_nm,bin_high_nm\n"
        "3100,1000,100,3000,3200\n"
        "3300,1200,120,3200,3400\n",
        encoding="utf-8",
    )
    output = tmp_path / "unity.npz"

    convert_emission_observation_table(
        source,
        output,
        delimiter=",",
        wavelength_column="wavelength_nm",
        flux_column="eclipse_ppm",
        uncertainty_column="error_ppm",
        bin_low_column="bin_low_nm",
        bin_high_column="bin_high_nm",
        wavelength_input_unit="nm",
        flux_input_unit="ppm",
        instrument="JWST/NIRSpec-G395H",
    )
    observation = load_emission_observation_npz(output)

    np.testing.assert_allclose(observation.wavelength, [3.1, 3.3])
    np.testing.assert_allclose(observation.flux, [1.0e-3, 1.2e-3])
    np.testing.assert_allclose(observation.uncertainty, [1.0e-4, 1.2e-4])
    np.testing.assert_allclose(observation.wavelength_bin_edges, [3.0, 3.2, 3.4])
    assert observation.instrument == "JWST/NIRSpec-G395H"
    assert observation.metadata["source_flux_unit"] == "ppm"


def test_table_loader_sorts_rows_and_infers_edges(tmp_path: Path) -> None:
    source = tmp_path / "spectrum.txt"
    source.write_text(
        "wave depth sigma\n4.0 0.2 0.02\n3.0 0.1 0.01\n",
        encoding="utf-8",
    )

    observation = load_emission_observation_table(
        source,
        wavelength_column="wave",
        flux_column="depth",
        uncertainty_column="sigma",
        flux_input_unit="percent",
    )

    np.testing.assert_allclose(observation.wavelength, [3.0, 4.0])
    np.testing.assert_allclose(observation.flux, [1.0e-3, 2.0e-3])
    np.testing.assert_allclose(observation.wavelength_bin_edges, [2.5, 3.5, 4.5])
