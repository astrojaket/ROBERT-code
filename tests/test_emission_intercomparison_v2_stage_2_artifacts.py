"""Integrity tests for committed Version-2 Stage-2 products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    load_version_2_common_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/emission_intercomparison/version_2"
COMMON = DATA / "common_contract.json"
REPORT = DATA / "stage_2_report.json"
INTEGRITY = DATA / "stage_2_integrity.json"
RESOLUTIONS = (40, 80, 160)
SPECIES = ("H2O", "CO", "CO2", "CH4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage_2_checksums_and_integrity_match_committed_bytes() -> None:
    checksums = json.loads((DATA / "checksums.json").read_text())
    integrity = json.loads(INTEGRITY.read_text())["artifacts"]
    stage_2_names = {name for name in checksums if name.startswith("stage_2_")}

    assert len(stage_2_names) == 33
    for name in stage_2_names:
        path = DATA / name
        assert path.is_file()
        assert checksums[name] == _sha256(path)
        if name != INTEGRITY.name:
            assert integrity[name]["sha256"] == checksums[name]


def test_stage_2_report_preserves_frozen_gates_and_scientific_framing() -> None:
    report = json.loads(REPORT.read_text())
    gates = report["predeclared_acceptance_gates"]
    observed = report["observed_gate_values"]

    assert report["status"] == "out_of_tolerance_closure_regime"
    assert gates["track_a_max_abs_symmetric_relative"] == 5e-4
    assert gates["track_a_max_abs_eclipse_difference_ppm"] == 0.1
    assert gates["track_a_80_to_160_max_abs_eclipse_difference_ppm"] == 0.1
    assert gates["track_a_isothermal_max_abs_eclipse_difference_ppm"] == 0.1
    assert observed["track_a_max_abs_eclipse_difference_ppm"] > gates[
        "track_a_max_abs_eclipse_difference_ppm"
    ]
    assert report["gate_results"][
        "track_a_isothermal_max_abs_eclipse_difference_ppm"
    ] is True
    assert report["pilot"]["authorized_full_matrix"] is True
    assert report["track_a_scope"]["gated_frameworks"] == [
        "robert",
        "petitradtrans",
    ]
    assert "no framework is classified as failed" in report["scientific_framing"]
    assert set(report["species"]) == set(SPECIES)
    assert report["resolutions"] == list(RESOLUTIONS)


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_2_shared_and_native_artifacts_retain_complete_fields(
    n_cells: int,
) -> None:
    expected_cases = 14 if n_cells == 80 else 12
    with np.load(DATA / f"stage_2_shared_tau_{n_cells}_cells.npz") as shared:
        assert shared["layer_tau"].shape[:2] == (expected_cases, n_cells)
        assert shared["layer_tau"].dtype == np.float32

    with np.load(DATA / f"stage_2_picaso_{n_cells}_cells.npz") as picaso:
        assert picaso["native_flux_w_m2_m"].shape == (expected_cases, 661)
        assert picaso["r100_flux_w_m2_m"].shape == (expected_cases, 271)
        assert picaso["layer_tau"].shape == (expected_cases, n_cells, 661, 8)
        assert picaso["normalized_vertical_native"].shape == (
            expected_cases,
            n_cells,
            661,
        )
        assert picaso["native_framework_probe_flux_w_m2_m"].shape == (
            expected_cases,
            661,
        )
        assert np.max(np.abs(picaso["maximum_abs_rayleigh_tau"])) == 0.0
        assert np.max(np.abs(picaso["maximum_abs_cloud_tau"])) == 0.0

    with np.load(DATA / f"stage_2_petitradtrans_{n_cells}_cells.npz") as prt:
        assert prt["native_flux_w_m2_m"].shape[0] == expected_cases
        assert prt["r100_flux_w_m2_m"].shape == (expected_cases, 271)
        assert prt["normalized_vertical_native"].shape[1] == n_cells
        assert "layer_tau" not in prt.files


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_2_opacity_sampling_is_unsmoothed_and_diagnosed(n_cells: int) -> None:
    path = DATA / f"stage_2_picaso_opacity_sampling_{n_cells}_cells.npz"
    with np.load(path) as archive:
        assert archive["native_flux_w_m2_m"].shape == (12, 819)
        assert archive["r100_flux_w_m2_m"].shape == (12, 271)
        assert archive["sample_count_per_r100_bin"].shape == (271,)
        assert archive["within_bin_flux_variance"].shape == (12, 271)
        assert int(np.sum(archive["sample_count_per_r100_bin"])) == 813
        assert archive["smoothing_applied"].item() is False
        assert np.any(archive["within_bin_flux_variance"] > 0.0)


def test_stage_2_exact_compositions_profiles_and_artifact_sharding() -> None:
    common = load_version_2_common_contract(COMMON)
    largest_robert_artifact = 0
    for species in SPECIES:
        path = DATA / f"stage_2_robert_{species}_80_cells.npz"
        largest_robert_artifact = max(largest_robert_artifact, path.stat().st_size)
        with np.load(path) as archive:
            assert set(archive["species_name"].tolist()) == {species}
            assert set(archive["profile_name"].tolist()) == {
                "isothermal",
                "pg14_non_inverted",
                "pg14_inverted",
            }
            reference = archive["abundance_scale"] == 1.0
            species_index = archive["species_name"].tolist().index(species)
            active_index = archive["gas_name"].tolist().index(species)
            assert archive["gas_vmr"][species_index, active_index] == pytest.approx(
                common.composition_vmr[species]
            )
            assert np.all(np.isfinite(archive["mean_molecular_weight_u"][reference]))
            assert archive["pressure_centers_bar"].shape == (80,)
            assert archive["temperature_cells_k"].shape[1] == 80
    assert largest_robert_artifact < 100 * 1024 * 1024


def test_stage_2_sampling_density_product_retains_both_native_spectra() -> None:
    with np.load(DATA / "stage_2_picaso_sampling_density_check.npz") as archive:
        assert archive["primary_native_wavelength_micron"].size == 819
        assert archive["density_native_wavelength_micron"].size == 1638
        assert archive["primary_r100_flux_w_m2_m"].shape == (271,)
        assert archive["density_r100_flux_w_m2_m"].shape == (271,)
        assert archive["smoothing_applied"].item() is False
