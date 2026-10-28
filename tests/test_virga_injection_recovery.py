"""Lightweight contracts for the optional Virga self-injection workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from robert_exoplanets.io.task_config import load_task_config
from scripts.virga_injection_recovery import (
    default_grid,
    grid_from_npz,
    synthetic_flux,
    validate_grid,
    validate_injection_config,
    weighted_parameter_recovery,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configurations" / "emission.yaml"


def test_default_grid_has_expected_native_count_and_coverage() -> None:
    wavelength, edges = default_grid()

    assert wavelength.shape == (301,)
    assert edges.shape == (302,)
    assert wavelength[0] == pytest.approx(0.6039311351970722)
    assert wavelength[-1] == pytest.approx(11.95087089337295)
    assert edges[0] < wavelength[0] < wavelength[-1] < edges[-1]
    assert np.all(np.diff(wavelength) > 0.0)
    assert np.all(np.diff(edges) > 0.0)


def test_grid_npz_reads_only_centres_and_edges(tmp_path: Path) -> None:
    wavelength, edges = default_grid()
    path = tmp_path / "grid_only.npz"
    # Deliberately omit flux and uncertainty. The file is a grid template only.
    np.savez(path, wavelength=wavelength, wavelength_bin_edges=edges)

    loaded_wavelength, loaded_edges, checksum = grid_from_npz(path)

    np.testing.assert_array_equal(loaded_wavelength, wavelength)
    np.testing.assert_array_equal(loaded_edges, edges)
    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()


def test_grid_npz_ignores_existing_flux_and_uncertainty(tmp_path: Path) -> None:
    wavelength, edges = default_grid()
    path = tmp_path / "with_decoy_spectrum.npz"
    np.savez(
        path,
        wavelength=wavelength,
        wavelength_bin_edges=edges,
        data=np.full(wavelength.size, 123.0),
        err=np.full(wavelength.size, 456.0),
    )

    loaded_wavelength, loaded_edges, _ = grid_from_npz(path)
    np.testing.assert_array_equal(loaded_wavelength, wavelength)
    np.testing.assert_array_equal(loaded_edges, edges)


def test_grid_validation_rejects_bad_edges() -> None:
    wavelength, edges = default_grid()
    edges = edges.copy()
    edges[10] = edges[9]

    with pytest.raises(ValueError, match="increase strictly"):
        validate_grid(wavelength, edges)


def test_noise_is_off_by_default_and_reproducible_when_enabled() -> None:
    model = np.linspace(0.001, 0.002, 301)
    data, uncertainty = synthetic_flux(model)
    np.testing.assert_array_equal(data, model)
    np.testing.assert_allclose(uncertainty, np.full(301, 100.0e-6), rtol=0.0, atol=1e-19)

    with pytest.raises(ValueError, match="seed is required"):
        synthetic_flux(model, add_noise=True)
    first, first_error = synthetic_flux(model, add_noise=True, seed=1729)
    second, second_error = synthetic_flux(model, add_noise=True, seed=1729)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_error, second_error)
    assert not np.array_equal(first, model)


def test_weighted_parameter_recovery_uses_weighted_median() -> None:
    samples = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([0.05, 0.15, 0.80])
    result = weighted_parameter_recovery(
        ("x",), samples, weights, {"x": 1.8}
    )

    assert result["x"]["weighted_median"] == pytest.approx(1.375)
    assert result["x"]["truth_in_95_percent_interval"]


def test_config_requires_virga_and_fitted_pg14(tmp_path: Path) -> None:
    raw = load_task_config(BASE_CONFIG).model_dump(mode="json")
    raw["observations"]["datasets"] = ["f322w2"]
    mie_directory = tmp_path / "mie"
    mie_directory.mkdir()
    (mie_directory / "KCl.mieff").write_text("test table\n", encoding="utf-8")
    raw["clouds"] = {
        "model": "virga_eddy_sed",
        "condensibles": ["KCl"],
        "gas_mass_mixing_ratios": {"KCl": 1.0e-4},
        "fsed_parameter": "cloud_fsed",
        "log10_kz_cm2_s_parameter": "cloud_log10_kzz",
        "mie_directory": str(mie_directory),
        "multiple_scattering_backend": "sh4",
    }
    raw["parameters"].extend(
        [
            {
                "name": "cloud_fsed",
                "unit": "dimensionless",
                "prior": {"type": "uniform", "lower": 0.1, "upper": 10.0},
                "value": 2.0,
            },
            {
                "name": "cloud_log10_kzz",
                "unit": "log10(cm2/s)",
                "prior": {"type": "uniform", "lower": 5.0, "upper": 11.0},
                "value": 9.0,
            },
        ]
    )
    source = tmp_path / "virga_r1000.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_task_config(source)

    contract = validate_injection_config(config)

    assert contract["opacity_resolution"] == "R1000"
    assert contract["temperature_models"] == {"primary": "parmentier_guillot_2014"}
    assert set(contract["pg14_fitted_parameter_names"]) == {
        "kappa_IR",
        "gamma1",
        "gamma2",
        "T_irr",
        "alpha",
    }


def test_config_rejects_non_pg14_temperature(tmp_path: Path) -> None:
    raw = load_task_config(BASE_CONFIG).model_dump(mode="json")
    raw["observations"]["datasets"] = ["f322w2"]
    raw["atmosphere"]["temperature"] = {"model": "isothermal", "temperature_k": 1200.0}
    mie_directory = tmp_path / "mie"
    mie_directory.mkdir()
    (mie_directory / "KCl.mieff").write_text("test table\n", encoding="utf-8")
    raw["clouds"] = {
        "model": "virga_eddy_sed",
        "condensibles": ["KCl"],
        "gas_mass_mixing_ratios": {"KCl": 1.0e-4},
        "fsed_parameter": "cloud_fsed",
        "log10_kz_cm2_s_parameter": "cloud_log10_kzz",
        "mie_directory": str(mie_directory),
        "multiple_scattering_backend": "sh4",
    }
    raw["parameters"].extend(
        [
            {
                "name": "cloud_fsed",
                "unit": "dimensionless",
                "prior": {"type": "uniform", "lower": 0.1, "upper": 10.0},
                "value": 2.0,
            },
            {
                "name": "cloud_log10_kzz",
                "unit": "log10(cm2/s)",
                "prior": {"type": "uniform", "lower": 5.0, "upper": 11.0},
                "value": 9.0,
            },
        ]
    )
    source = tmp_path / "isothermal.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_task_config(source)

    with pytest.raises(ValueError, match="fitted PG14"):
        validate_injection_config(config)
