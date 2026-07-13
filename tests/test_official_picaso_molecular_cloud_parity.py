"""Regression tests for the official PICASO molecular-opacity benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from examples.benchmark_official_picaso_molecular_cloud_parity import (
    _bin_mean,
    _bin_ratio,
    _interpolate_wavenumber,
    _make_science_contract,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "validation" / "official_picaso_molecular_cloud_parity"


def test_science_contract_has_normalized_explicit_vmr_state() -> None:
    contract = _make_science_contract(32, 12, 16)

    assert int(contract["contract_schema_version"]) == 2
    assert contract["gas_vmr"].shape == (16, 4)
    total = (
        np.sum(contract["gas_vmr"], axis=1)
        + float(contract["h2_vmr"])
        + contract["he_vmr"]
    )
    np.testing.assert_allclose(total, 1.0, rtol=0.0, atol=2.0e-16)
    assert np.all(contract["he_vmr"] > 0.0)


def test_wavenumber_interpolation_preserves_linear_wavenumber_function() -> None:
    source_wavelength = np.array([1.0, 2.0, 4.0])
    source_values = 3.0 + 2.0 * (1.0e4 / source_wavelength)
    target_wavelength = np.array([1.25, 1.6, 3.2])

    interpolated = _interpolate_wavenumber(
        source_wavelength, source_values, target_wavelength
    )

    np.testing.assert_allclose(
        interpolated,
        3.0 + 2.0 * (1.0e4 / target_wavelength),
    )


def test_flux_conserving_bin_mean_preserves_constant() -> None:
    x = np.geomspace(1.0, 12.0, 1000)
    edges = np.geomspace(1.0, 12.0, 37)

    result = _bin_mean(x, np.full(x.size, 7.5), edges)

    np.testing.assert_allclose(result, 7.5, rtol=0.0, atol=2.0e-15)


def test_flux_weighted_ratio_binning_preserves_constant_ratio() -> None:
    x = np.geomspace(1.0, 12.0, 1000)
    edges = np.geomspace(1.0, 12.0, 37)
    denominator = x**-4

    result = _bin_ratio(x, 0.003 * denominator, denominator, edges)

    np.testing.assert_allclose(result, 0.003, rtol=0.0, atol=2.0e-18)


def test_versioned_official_picaso_reference_passes_acceptance() -> None:
    report = json.loads(
        (REFERENCE / "official_picaso_molecular_cloud_parity.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["schema_version"] == 1
    assert report["sampling"]["opacity_stride"] == 1
    assert report["sampling"]["picaso_native_samples"] == 37273
    assert report["metrics"]["acceptance"]["all_pass"] is True
    assert (
        report["metrics"]["cloud_mass_extinction"]["rms_relative_difference"]
        < 2.0e-6
    )
    assert report["metrics"]["cloudy_emission_difference_ppm"]["rms"] < 25.0
    assert report["metrics"]["cloudy_transmission_difference_ppm"]["rms"] < 250.0


def test_full_native_sampling_is_required_for_paper_reference() -> None:
    convergence = json.loads(
        (REFERENCE / "sampling_convergence.json").read_text(encoding="utf-8")
    )

    assert convergence["reference_stride"] == 1
    assert convergence["cases"]["2"]["robert"]["cloudy_emission_rms_ppm"] > 10.0
    assert convergence["cases"]["2"]["picaso"]["cloudy_transmission_rms_ppm"] > 20.0


def test_versioned_official_picaso_reference_checksums() -> None:
    checksums = json.loads((REFERENCE / "checksums.json").read_text(encoding="utf-8"))

    for name, expected in checksums.items():
        assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == expected
