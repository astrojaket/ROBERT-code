"""Contract and artifact tests for emission intercomparison Version-2 Stage 6."""

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


stage_6 = _load(
    "emission_v2_stage_6",
    ROOT / "examples/benchmark_emission_intercomparison_v2_stage_6.py",
)
worker = _load(
    "emission_v2_stage_6_worker",
    ROOT / "examples/run_emission_intercomparison_v2_stage_6_external.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize("n_cells", (40, 80, 160))
@pytest.mark.parametrize("profile", stage_6.PROFILES)
@pytest.mark.parametrize("species", stage_6.SPECIES)
def test_stage_6_contract_normalizes_composition_and_recomputes_mmw(
    n_cells: int, profile: str, species: str
) -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_6.build_stage_6_contract(
        common, n_cells, profile=profile, target_species=species
    )
    assert contract["case_id"].shape == (13,)
    assert contract["gas_vmr_cells"].shape == (13, n_cells, 6)
    assert contract["gas_vmr_edges"].shape == (13, n_cells + 1, 6)
    np.testing.assert_allclose(
        np.sum(contract["gas_vmr_cells"], axis=-1), 1.0, rtol=0, atol=5e-16
    )
    np.testing.assert_allclose(
        np.sum(contract["gas_vmr_edges"], axis=-1), 1.0, rtol=0, atol=5e-16
    )
    np.testing.assert_allclose(
        contract["mean_molecular_weight_cells"],
        np.sum(contract["gas_vmr_cells"] * contract["gas_mass_u"], axis=-1),
        rtol=0,
        atol=2e-15,
    )
    assert np.all(contract["include_h2_h2_cia"])
    assert np.all(contract["include_h2_he_cia"])
    worker._validate_contract(contract)


def test_stage_6_exact_background_ratio_and_non_target_invariance() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_6.build_stage_6_contract(
        common, 80, profile="pg14_non_inverted", target_species="CO2"
    )
    ratio = contract["gas_vmr_cells"][..., 0] / contract["gas_vmr_cells"][..., 1]
    np.testing.assert_allclose(ratio, 0.8547 / 0.1453, rtol=2e-15)
    for species in ("H2O", "CO", "CH4"):
        index = stage_6.GAS_NAMES.index(species)
        assert np.all(contract["gas_vmr_cells"][..., index] == common.composition_vmr[species])
    assert np.ptp(contract["summed_line_gas_vmr_cells"], axis=0).max() > 0.0


def test_stage_6_target_perturbation_and_edge_cell_mapping() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_6.build_stage_6_contract(
        common, 80, profile="pg14_inverted", target_species="CH4"
    )
    target = stage_6.GAS_NAMES.index("CH4")
    center = 3
    selected = contract["perturbation_center_index"] == center
    minus = np.flatnonzero(selected & (contract["perturbation_sign"] == -1))[0]
    plus = np.flatnonzero(selected & (contract["perturbation_sign"] == 1))[0]
    reference = common.composition_vmr["CH4"]
    expected_cells = reference * 10.0 ** (
        0.1
        * stage_6.composition_localization(
            contract["pressure_centers_bar"], stage_6.CENTERS_BAR[center]
        )
    )
    expected_edges = reference * 10.0 ** (
        0.1
        * stage_6.composition_localization(
            contract["pressure_edges_bar"], stage_6.CENTERS_BAR[center]
        )
    )
    np.testing.assert_allclose(contract["gas_vmr_cells"][plus, :, target], expected_cells)
    np.testing.assert_allclose(contract["gas_vmr_edges"][plus, :, target], expected_edges)
    np.testing.assert_allclose(
        contract["gas_vmr_cells"][plus, :, target]
        * contract["gas_vmr_cells"][minus, :, target],
        reference**2,
        rtol=3e-15,
    )


def test_stage_6_frozen_gates_and_perturbations() -> None:
    assert stage_6.PRIMARY_AMPLITUDE_DEX == 0.10
    assert stage_6.LINEARITY_AMPLITUDES_DEX == (0.05, 0.10, 0.20)
    assert stage_6.LOCALIZATION_SIGMA_DEX == 0.35
    assert stage_6.STAGE_6_ACCEPTANCE_GATES == {
        "track_a_primary_p95_abs_jacobian_difference_over_pair_peak": 0.05,
        "track_a_primary_rms_eclipse_jacobian_difference_ppm_per_dex": 0.50,
        "track_a_primary_centroid_rms_difference_dex": 0.15,
        "track_a_primary_response_total_variation_p95": 0.08,
        "track_a_primary_cross_species_fraction_total_variation_p95": 0.08,
        "track_a_80_to_160_p95_abs_jacobian_difference_over_pair_peak": 0.05,
        "track_a_80_to_160_rms_eclipse_jacobian_difference_ppm_per_dex": 0.50,
        "track_a_80_to_160_centroid_rms_difference_dex": 0.15,
        "track_a_80_to_160_response_total_variation_p95": 0.08,
        "track_a_80_to_160_cross_species_fraction_total_variation_p95": 0.08,
        "finite_difference_linearity_p95_relative": 0.02,
        "finite_difference_symmetry_p95_relative": 0.02,
        "analytic_isothermal_composition_jacobian_max_abs": 0.0,
        "exact_zero_normalization_and_fraction_max_abs": 0.0,
        "pilot_projected_wall_time_max_s": 7200.0,
        "pilot_peak_rss_fraction_of_available_max": 0.60,
    }


def test_stage_6_zero_signal_and_cross_species_conventions() -> None:
    normalized, zero = stage_6.normalize_absolute_response(np.zeros((2, 4, 6, 9)))
    fractions, fraction_zero = stage_6.cross_species_fractions(np.zeros((2, 4, 6, 9)))
    assert np.all(normalized == 0.0)
    assert np.all(fractions == 0.0)
    assert np.all(zero)
    assert np.all(fraction_zero)
    flux = np.ones((13, 7))
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_6.build_stage_6_contract(
        common, 40, profile="isothermal", target_species="H2O"
    )
    response, jacobian, even = stage_6._extract_difference(flux, contract)
    assert np.all(response == 0.0)
    assert np.all(jacobian == 0.0)
    assert np.all(even == 0.0)


def test_stage_6_opacity_sampling_products_are_absent() -> None:
    assert not list(DATA.glob("stage_6_*opacity_sampling*.npz"))
    assert not list(DATA.glob("stage_6_*sampling*.npz"))


def test_stage_6_picaso_correction_is_state_dependent_and_restores_sum() -> None:
    path = DATA / "stage_6_main_pg14_non_inverted_H2O_80_cells_picaso.npz"
    if not path.exists():
        pytest.skip("Stage-6 numerical products have not been generated")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["absolute_line_vmr_restored_after_resort_rebin"] is True
        assert metadata["absolute_line_vmr_correction_state_dependent"] is True
        assert "actual layer summed" in metadata["absolute_line_vmr_correction_algorithm"]
        restored = data["state_dependent_absolute_line_vmr_sum_edges"]
        expected = data["summed_line_gas_vmr_edges"]
        np.testing.assert_array_equal(restored, expected)
        assert np.ptp(restored, axis=0).max() > 0.0


def test_stage_6_products_retain_composition_response_and_coupling_arrays() -> None:
    response_path = DATA / "stage_6_response_pg14_non_inverted_robert_80_cells.npz"
    state_path = DATA / "stage_6_main_pg14_non_inverted_CO2_80_cells_robert.npz"
    tau_path = DATA / "stage_6_main_pg14_non_inverted_CO2_80_cells_shared_tau.npz"
    if not response_path.exists():
        pytest.skip("Stage-6 numerical products have not been generated")
    with np.load(response_path, allow_pickle=False) as response:
        assert response["composition_jacobian_r100_w_m2_m_dex"].shape == (
            4,
            6,
            369,
        )
        assert response["normalized_absolute_response_r100"].shape == (4, 6, 369)
        assert response["cross_species_sensitivity_fraction_r100"].shape == (4, 369)
        assert np.all(np.isfinite(response["composition_jacobian_r100_w_m2_m_dex"]))
    with np.load(state_path, allow_pickle=False) as state:
        np.testing.assert_allclose(
            np.sum(state["gas_vmr_cells"], axis=-1), 1.0, rtol=0, atol=5e-16
        )
        expected_mmw = np.sum(
            state["gas_vmr_cells"] * state["gas_mass_u"], axis=-1
        )
        np.testing.assert_allclose(
            state["mean_molecular_weight_cells"], expected_mmw, rtol=0, atol=2e-15
        )
        assert np.all(state["include_h2_h2_cia"])
        assert np.all(state["include_h2_he_cia"])
    with np.load(tau_path, allow_pickle=False) as tau:
        assert tau["state_dependent_shared_total_mean_layer_tau"].shape[0] == 13
        assert np.ptp(
            tau["state_dependent_shared_total_mean_layer_tau"], axis=0
        ).max() > 0.0


def test_stage_6_analytic_zero_and_capability_boundaries_are_explicit() -> None:
    control_path = DATA / "stage_6_isothermal_analytic_control_picaso.npz"
    report_path = DATA / "stage_6_report.json"
    if not control_path.exists() or not report_path.exists():
        pytest.skip("Stage-6 numerical products have not been generated")
    with np.load(control_path, allow_pickle=False) as control:
        assert bool(control["exact_zero_assignment"])
        assert np.all(control["composition_jacobian_native_w_m2_m_dex"] == 0.0)
        assert np.all(control["composition_jacobian_r100_w_m2_m_dex"] == 0.0)
        assert np.all(control["normalized_absolute_response_r100"] == 0.0)
        assert np.all(control["cross_species_sensitivity_fraction_r100"] == 0.0)
    report = json.loads(report_path.read_text())
    boundaries = report["known_warnings_and_capability_boundaries"]
    assert "no stable layer optical-depth tensor" in boundaries["petitradtrans"]
    assert "absorbing-formal vertical diagnostics" in boundaries["picaso"][-1]
    assert report["track_a_scope"]["gated_frameworks"] == ["robert", "petitradtrans"]
    assert report["track_a_scope"]["picaso"] == "no identical-tensor path or gate"
    assert report["gate_results"]["finite_difference_symmetry_p95_relative"] is False


def test_stage_6_committed_checksums_integrity_and_blob_sizes() -> None:
    report = DATA / "stage_6_report.json"
    integrity_path = DATA / "stage_6_integrity.json"
    if not report.exists() or not integrity_path.exists():
        pytest.skip("Stage-6 numerical products have not been generated")
    checksums = json.loads((DATA / "checksums.json").read_text())
    names = {name for name in checksums if name.startswith("stage_6_")}
    integrity = json.loads(integrity_path.read_text())["artifacts"]
    assert set(integrity) == names - {integrity_path.name}
    for name in names:
        path = DATA / name
        assert path.is_file()
        assert checksums[name] == _sha256(path)
        assert path.stat().st_size < 100_000_000
        if name != integrity_path.name:
            assert integrity[name]["sha256"] == checksums[name]
