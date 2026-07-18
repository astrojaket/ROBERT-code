"""Integrity and schema tests for committed Version-2 Stage-3 products."""

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
REPORT = DATA / "stage_3_report.json"
INTEGRITY = DATA / "stage_3_integrity.json"
RESOLUTIONS = (40, 80, 160)
PROFILES = ("isothermal", "pg14_non_inverted")
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")
FACTOR_CASES = (
    ("molecular_only", False, False, ()),
    ("molecular_plus_h2_h2_cia", True, False, ("H2-H2",)),
    ("molecular_plus_h2_he_cia", False, True, ("H2-He",)),
    (
        "molecular_plus_h2_h2_and_h2_he_cia",
        True,
        True,
        ("H2-H2", "H2-He"),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_mean_molecular_weight_u() -> float:
    payload = json.loads((DATA / "common_contract.json").read_text())
    return float(payload["composition"]["mean_molecular_weight_u_declared"])


def _expected_case_ids(n_cells: int) -> list[str]:
    return [
        f"{profile}_{factor_name}_{n_cells}_cells"
        for profile in PROFILES
        for factor_name, _h2_h2, _h2_he, _pairs in FACTOR_CASES
    ]


def _expected_pairs(n_cells: int) -> dict[str, list[str]]:
    return {
        f"{profile}_{factor_name}_{n_cells}_cells": list(pairs)
        for profile in PROFILES
        for factor_name, _h2_h2, _h2_he, pairs in FACTOR_CASES
    }


def _assert_factorial_and_composition(
    archive: np.lib.npyio.NpzFile, n_cells: int
) -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    gas_names = tuple(common.composition_vmr)
    reference_vmr = np.asarray([common.composition_vmr[name] for name in gas_names])
    repeated_factors = [factor for _profile in PROFILES for factor in FACTOR_CASES]

    assert archive["case_id"].tolist() == _expected_case_ids(n_cells)
    assert archive["profile_name"].tolist() == [
        profile for profile in PROFILES for _factor in FACTOR_CASES
    ]
    assert archive["factor_name"].tolist() == [
        factor_name
        for factor_name, _h2_h2, _h2_he, _pairs in repeated_factors
    ]
    np.testing.assert_array_equal(
        archive["include_h2_h2_cia"],
        [h2_h2 for _name, h2_h2, _h2_he, _pairs in repeated_factors],
    )
    np.testing.assert_array_equal(
        archive["include_h2_he_cia"],
        [h2_he for _name, _h2_h2, h2_he, _pairs in repeated_factors],
    )
    assert archive["gas_name"].tolist() == list(gas_names)
    np.testing.assert_array_equal(
        archive["gas_vmr"], np.broadcast_to(reference_vmr, (8, len(gas_names)))
    )
    np.testing.assert_array_equal(
        archive["mean_molecular_weight_u"],
        np.full(8, _declared_mean_molecular_weight_u()),
    )
    assert archive["molecular_species_name"].tolist() == list(MOLECULAR_SPECIES)
    np.testing.assert_array_equal(
        archive["molecular_species_active"], np.ones((8, 4), dtype=bool)
    )


def test_stage_3_checksums_and_integrity_match_all_committed_bytes() -> None:
    checksums = json.loads((DATA / "checksums.json").read_text())
    integrity = json.loads(INTEGRITY.read_text())["artifacts"]
    stage_3_names = {name for name in checksums if name.startswith("stage_3_")}

    assert len(stage_3_names) == 20
    assert set(integrity) == stage_3_names - {INTEGRITY.name}
    for name in stage_3_names:
        path = DATA / name
        assert path.is_file()
        assert checksums[name] == _sha256(path)
        if name != INTEGRITY.name:
            assert integrity[name]["sha256"] == checksums[name]
            assert integrity[name]["size_bytes"] == path.stat().st_size


def test_stage_3_report_preserves_frozen_gates_status_and_framing() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    report = json.loads(REPORT.read_text())

    assert report["stage"] == 3
    assert report["status"] == "out_of_tolerance_closure_regime"
    assert report["predeclared_acceptance_gates"] == {
        "track_a_max_abs_symmetric_relative": 5.0e-4,
        "track_a_max_abs_eclipse_difference_ppm": 0.1,
        "track_a_80_to_160_max_abs_eclipse_difference_ppm": 0.1,
        "track_a_isothermal_max_abs_eclipse_difference_ppm": 0.1,
        "scattering_single_scattering_albedo_max_abs": 0.0,
        "pilot_projected_wall_time_max_s": 7200.0,
        "pilot_peak_rss_fraction_of_available_max": 0.60,
    }
    declared_mmw = _declared_mean_molecular_weight_u()
    assert report["factorial_contract"]["fixed_mean_molecular_weight_u"] == declared_mmw
    assert report["mean_molecular_weight_attribution"][
        "frozen_full_mixture_mmw_u"
    ] == declared_mmw
    assert report["factorial_contract"]["fixed_composition"] == dict(
        common.composition_vmr
    )
    assert report["factorial_contract"]["molecular_absorbers_always_active"] == list(
        MOLECULAR_SPECIES
    )
    assert report["factorial_contract"]["factor_names"] == [
        factor[0] for factor in FACTOR_CASES
    ]
    assert "no framework is classified as failed" in report["scientific_framing"]
    assert report["track_a_scope"]["gated_frameworks"] == [
        "robert",
        "petitradtrans",
    ]
    assert report["track_b_scope"]["cross_framework_gates"] is None


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_archives_preserve_factorial_spectra_and_vertical_shapes(
    n_cells: int,
) -> None:
    archives = {
        "robert_shared": (3696, DATA / f"stage_3_robert_shared_{n_cells}_cells.npz"),
        "petitradtrans_shared": (
            3696,
            DATA / f"stage_3_petitradtrans_shared_{n_cells}_cells.npz",
        ),
        "robert": (3696, DATA / f"stage_3_robert_{n_cells}_cells.npz"),
        "picaso": (661, DATA / f"stage_3_picaso_{n_cells}_cells.npz"),
        "petitradtrans": (
            3697,
            DATA / f"stage_3_petitradtrans_{n_cells}_cells.npz",
        ),
    }
    for _name, (n_native, path) in archives.items():
        with np.load(path, allow_pickle=False) as archive:
            _assert_factorial_and_composition(archive, n_cells)
            assert archive["native_flux_w_m2_m"].shape == (8, n_native)
            assert archive["r100_flux_w_m2_m"].shape == (8, 369)
            assert archive["normalized_vertical_native"].shape == (
                8,
                n_cells,
                n_native,
            )
            assert archive["normalized_vertical_r100"].shape == (8, n_cells, 369)


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_shared_tau_reconstructs_from_frozen_components(n_cells: int) -> None:
    path = DATA / f"stage_3_shared_tau_{n_cells}_cells.npz"
    with np.load(path, allow_pickle=False) as archive:
        assert archive["case_id"].tolist() == _expected_case_ids(n_cells)
        assert archive["molecular_layer_tau_by_profile"].shape == (
            2,
            n_cells,
            3696,
        )
        assert archive["cia_h2_h2_layer_tau_by_profile"].shape == (
            2,
            n_cells,
            3696,
        )
        assert archive["cia_h2_he_layer_tau_by_profile"].shape == (
            2,
            n_cells,
            3696,
        )
        reconstructed = []
        for profile_index, _profile in enumerate(PROFILES):
            for _name, include_h2_h2, include_h2_he, _pairs in FACTOR_CASES:
                total = np.array(
                    archive["molecular_layer_tau_by_profile"][profile_index],
                    copy=True,
                )
                if include_h2_h2:
                    total += archive["cia_h2_h2_layer_tau_by_profile"][profile_index]
                if include_h2_he:
                    total += archive["cia_h2_he_layer_tau_by_profile"][profile_index]
                reconstructed.append(total)
        np.testing.assert_allclose(
            archive["layer_tau"], reconstructed, rtol=2e-7, atol=0.0
        )


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_robert_retains_full_component_tau_tensors(n_cells: int) -> None:
    with np.load(
        DATA / f"stage_3_robert_{n_cells}_cells.npz", allow_pickle=False
    ) as robert, np.load(
        DATA / f"stage_3_shared_tau_{n_cells}_cells.npz", allow_pickle=False
    ) as shared:
        assert robert["molecular_layer_tau_by_profile"].shape == (
            2,
            n_cells,
            3696,
            16,
        )
        assert robert["cia_h2_h2_layer_tau_by_profile"].shape == (
            2,
            n_cells,
            3696,
        )
        assert robert["cia_h2_he_layer_tau_by_profile"].shape == (
            2,
            n_cells,
            3696,
        )
        collapsed = np.sum(
            robert["molecular_layer_tau_by_profile"]
            * robert["g_weights"][None, None, None, :],
            axis=-1,
        )
        np.testing.assert_allclose(
            collapsed,
            shared["molecular_layer_tau_by_profile"],
            rtol=2e-7,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            robert["cia_h2_h2_layer_tau_by_profile"],
            shared["cia_h2_h2_layer_tau_by_profile"],
        )
        np.testing.assert_array_equal(
            robert["cia_h2_he_layer_tau_by_profile"],
            shared["cia_h2_he_layer_tau_by_profile"],
        )


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_picaso_retains_native_tau_probe_and_exact_pair_metadata(
    n_cells: int,
) -> None:
    with np.load(
        DATA / f"stage_3_picaso_{n_cells}_cells.npz", allow_pickle=False
    ) as archive:
        assert archive["layer_tau"].shape == (8, n_cells, 661, 8)
        assert archive["native_framework_probe_flux_w_m2_m"].shape == (8, 661)
        assert float(archive["maximum_abs_rayleigh_tau"]) == 0.0
        assert float(archive["maximum_abs_cloud_tau"]) == 0.0
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["molecular_species_always_enabled"] == list(MOLECULAR_SPECIES)
        assert metadata["requested_cia_pairs_by_case"] == _expected_pairs(n_cells)
        assert metadata["representation"] == "primary_correlated_k_resort_rebin"
        assert any("exact-omega0=0" in item for item in metadata["limitations"])


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_3_stable_prt_does_not_invent_native_tau(n_cells: int) -> None:
    with np.load(
        DATA / f"stage_3_petitradtrans_{n_cells}_cells.npz", allow_pickle=False
    ) as archive:
        assert "layer_tau" not in archive.files
        assert "g_weights" not in archive.files
        assert "cia_h2_h2_layer_tau_by_profile" not in archive.files
        assert "cia_h2_he_layer_tau_by_profile" not in archive.files
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["requested_cia_pairs_by_case"] == _expected_pairs(n_cells)
        assert any(
            "does not expose native layer optical-depth" in item
            for item in metadata["limitations"]
        )


def test_stage_3_opacity_sampling_products_are_retired() -> None:
    assert not list(DATA.glob("stage_3_picaso_opacity_sampling_*.npz"))
    assert not (DATA / "stage_3_picaso_sampling_density_check.npz").exists()
    report = json.loads(REPORT.read_text())
    assert report["track_b_scope"]["picaso_secondary"] is None


def test_stage_3_capability_boundaries_remain_explicit() -> None:
    report = json.loads(REPORT.read_text())
    boundaries = report["known_warnings_and_capability_boundaries"]

    assert any("optional-Vega" in item for item in boundaries)
    assert any("exact-zero cloud/Rayleigh" in item for item in boundaries)
    assert any("exact-omega0=0 shared-tensor RT remains unsupported" in item for item in boundaries)
    assert any("Stable pRT does not expose native layer optical-depth" in item for item in boundaries)
    assert any("retired" in item for item in boundaries)
    assert any("0.196897 ppm" in item for item in boundaries)
    assert any("Stage-2's measured out-of-tolerance" in item for item in boundaries)
