"""Contract and artifact tests for emission intercomparison Version-2 Stage 5."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    load_version_2_common_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/emission_intercomparison/version_2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_5 = _load(
    "emission_v2_stage_5",
    ROOT / "examples/benchmark_emission_intercomparison_v2_stage_5.py",
)
worker = _load(
    "emission_v2_stage_5_worker",
    ROOT / "examples/run_emission_intercomparison_v2_stage_5_external.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize("n_cells", (40, 80, 160))
def test_stage_5_contract_uses_frozen_profiles_composition_and_localizations(
    n_cells: int,
) -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_5.build_stage_5_contract(common, n_cells)
    assert contract["case_id"].shape == (39,)
    assert contract["temperature_cells_k"].shape == (39, n_cells)
    assert contract["temperature_edges_k"].shape == (39, n_cells + 1)
    np.testing.assert_array_equal(contract["perturbation_centers_bar"], np.geomspace(1e-4, 10, 6))
    assert float(contract["localization_sigma_dex"]) == 0.35
    assert np.count_nonzero(contract["perturbation_sign"] == 0) == 3
    assert np.all(contract["include_h2_h2_cia"])
    assert np.all(contract["include_h2_he_cia"])
    np.testing.assert_allclose(np.sum(contract["gas_vmr"], axis=1), 1.0, rtol=0, atol=5e-16)
    for profile_index, profile in enumerate(stage_5.PROFILES):
        baseline = np.flatnonzero(
            (contract["profile_index"] == profile_index)
            & (contract["perturbation_sign"] == 0)
        )[0]
        np.testing.assert_array_equal(
            contract["temperature_cells_k"][baseline],
            common.temperature_profiles_k[f"{profile}_{n_cells}_cells"],
        )
        for center_index in range(6):
            selected = (
                (contract["profile_index"] == profile_index)
                & (contract["perturbation_center_index"] == center_index)
            )
            minus = np.flatnonzero(selected & (contract["perturbation_sign"] == -1))[0]
            plus = np.flatnonzero(selected & (contract["perturbation_sign"] == 1))[0]
            np.testing.assert_allclose(
                0.5
                * (
                    contract["temperature_cells_k"][minus]
                    + contract["temperature_cells_k"][plus]
                ),
                contract["temperature_cells_k"][baseline],
                rtol=0,
                atol=3e-13,
            )


def test_stage_5_frozen_gates_and_perturbations() -> None:
    assert stage_5.PRIMARY_AMPLITUDE_K == 10.0
    assert stage_5.LINEARITY_AMPLITUDES_K == (5.0, 10.0, 20.0)
    assert stage_5.LOCALIZATION_SIGMA_DEX == 0.35
    assert stage_5.PILOT_PROJECTION_MULTIPLIER == 12.0
    assert stage_5.STAGE_5_ACCEPTANCE_GATES == {
        "track_a_primary_p95_abs_jacobian_difference_over_pair_peak": 0.05,
        "track_a_primary_rms_eclipse_jacobian_difference_ppm_per_k": 0.02,
        "track_a_primary_centroid_rms_difference_dex": 0.15,
        "track_a_primary_response_total_variation_p95": 0.08,
        "track_a_80_to_160_p95_abs_jacobian_difference_over_pair_peak": 0.05,
        "track_a_80_to_160_rms_eclipse_jacobian_difference_ppm_per_k": 0.02,
        "track_a_80_to_160_centroid_rms_difference_dex": 0.15,
        "track_a_80_to_160_response_total_variation_p95": 0.08,
        "track_a_isothermal_baseline_max_abs_eclipse_difference_ppm": 0.1,
        "finite_difference_linearity_p95_relative": 0.02,
        "finite_difference_symmetry_p95_relative": 0.02,
        "exact_zero_normalization_max_abs": 0.0,
        "pilot_projected_wall_time_max_s": 7200.0,
        "pilot_peak_rss_fraction_of_available_max": 0.60,
    }


def test_stage_5_zero_signal_convention_and_localization() -> None:
    normalized, zero = stage_5.normalize_absolute_response(np.zeros((2, 6, 4)))
    assert np.all(normalized == 0.0)
    assert np.all(zero)
    pressure = np.array([0.01, 0.1, 1.0])
    localization = stage_5.temperature_localization(pressure, 0.1)
    assert localization[1] == 1.0
    assert localization[0] == localization[2]


def test_stage_5_worker_rejects_missing_cia() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_5.build_stage_5_contract(common, 40)
    contract["include_h2_he_cia"][0] = False
    with pytest.raises(ValueError, match="fixes both"):
        worker._validate_contract(contract)


def test_stage_5_opacity_sampling_products_are_absent() -> None:
    assert not list(DATA.glob("stage_5_*opacity_sampling*.npz"))
    assert not list(DATA.glob("stage_5_*sampling*.npz"))


def test_stage_5_committed_checksums_integrity_and_blob_sizes() -> None:
    report = DATA / "stage_5_report.json"
    integrity_path = DATA / "stage_5_integrity.json"
    if not report.exists() or not integrity_path.exists():
        pytest.skip("Stage-5 numerical products have not been generated")
    checksums = json.loads((DATA / "checksums.json").read_text())
    names = {name for name in checksums if name.startswith("stage_5_")}
    integrity = json.loads(integrity_path.read_text())["artifacts"]
    assert set(integrity) == names - {integrity_path.name}
    for name in names:
        path = DATA / name
        assert path.is_file()
        assert checksums[name] == _sha256(path)
        assert path.stat().st_size < 100_000_000
        if name != integrity_path.name:
            assert integrity[name]["sha256"] == checksums[name]


def test_stage_5_primary_artifacts_retain_signed_states_and_responses() -> None:
    path = DATA / "stage_5_robert_80_cells.npz"
    if not path.exists():
        pytest.skip("Stage-5 numerical products have not been generated")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["native_flux_w_m2_m"].shape == (39, 3696)
        assert archive["r100_flux_w_m2_m"].shape == (39, 369)
        assert archive["temperature_jacobian_native_w_m2_m_k"].shape == (3, 6, 3696)
        assert archive["temperature_jacobian_r100_w_m2_m_k"].shape == (3, 6, 369)
        assert archive["normalized_absolute_response_r100"].shape == (3, 6, 369)
        total = np.sum(archive["normalized_absolute_response_r100"], axis=1)
        zero = archive["zero_signal_mask_r100"]
        np.testing.assert_allclose(total[~zero], 1.0)
        assert np.all(total[zero] == 0.0)


def test_stage_5_picaso_correction_and_prt_capability_metadata() -> None:
    picaso = DATA / "stage_5_picaso_pg14_non_inverted_80_cells.npz"
    prt = DATA / "stage_5_petitradtrans_pg14_non_inverted_80_cells.npz"
    if not picaso.exists() or not prt.exists():
        pytest.skip("Stage-5 numerical products have not been generated")
    with np.load(picaso, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["absolute_line_vmr_restored_after_resort_rebin"] is True
        assert archive["native_total_taugas"].shape == (13, 80, 661, 8)
        assert float(archive["maximum_abs_rayleigh_tau"]) == 0.0
        assert float(archive["maximum_abs_cloud_tau"]) == 0.0
        assert any("not native SH" in item for item in metadata["limitations"])
    with np.load(prt, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert "native_total_taugas" not in archive.files
        assert any("does not expose" in item for item in metadata["limitations"])
