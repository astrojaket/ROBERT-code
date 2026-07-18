"""Frozen contract and local-product tests for Version-2 Stage 7."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import numpy as np

from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    load_version_2_common_contract,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_7 import (
    CLOUD_OPTICAL_DEPTHS,
    CLOUD_TOP_PRESSURES_BAR,
    EXTINCTION_SLOPES,
    REFERENCE_WAVELENGTH_MICRON,
    build_cloud_extinction_matrix,
    cloud_definitions,
    fractional_log_pressure_weights,
    pilot_resource_decision,
    power_law_cloud_tau,
    regrid_tabulated_extinction_tau,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/emission_intercomparison/version_2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_7 = _load(
    "emission_v2_stage_7",
    ROOT / "examples/benchmark_emission_intercomparison_v2_stage_7.py",
)
worker = _load(
    "emission_v2_stage_7_worker",
    ROOT / "examples/run_emission_intercomparison_v2_stage_7_external.py",
)


def test_stage_7_matrix_and_exact_absorption_contract() -> None:
    definitions = cloud_definitions()
    assert CLOUD_OPTICAL_DEPTHS == (0.1, 1.0, 10.0, 100.0)
    assert CLOUD_TOP_PRESSURES_BAR == (1.0e-3, 1.0e-2, 1.0e-1)
    assert EXTINCTION_SLOPES == (-4.0, -2.0, 0.0, 2.0)
    assert REFERENCE_WAVELENGTH_MICRON == 5.0
    assert len(definitions) == 50
    assert definitions[0].kind == "clear"
    assert definitions[-1].kind == "archived_tabulated"
    assert all(item.single_scattering_albedo == 0.0 for item in definitions)


def test_fractional_boundary_placement_and_integrated_tau() -> None:
    edges = np.geomspace(1.0e-5, 100.0, 41)
    top = 0.0137
    weights = fractional_log_pressure_weights(edges, top)
    boundary = int(np.searchsorted(edges, top) - 1)
    assert 0.0 < weights[boundary] < np.log(edges[boundary + 1] / edges[boundary]) / np.log(100.0 / top)
    assert np.all(weights[:boundary] == 0.0)
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=0.0, atol=2e-15)
    tau = power_law_cloud_tau(
        edges,
        np.array([1.0, 5.0, 10.0]),
        optical_depth_at_reference=10.0,
        cloud_top_pressure_bar=top,
        extinction_slope=0.0,
    )
    np.testing.assert_allclose(np.sum(tau, axis=0), 10.0, rtol=0.0, atol=3e-14)


def test_reference_wavelength_and_slope_sign() -> None:
    edges = np.geomspace(1.0e-5, 100.0, 81)
    wavelength = np.array([1.0, 5.0, 10.0])
    negative = power_law_cloud_tau(
        edges,
        wavelength,
        optical_depth_at_reference=1.0,
        cloud_top_pressure_bar=1.0e-2,
        extinction_slope=-2.0,
    ).sum(axis=0)
    positive = power_law_cloud_tau(
        edges,
        wavelength,
        optical_depth_at_reference=1.0,
        cloud_top_pressure_bar=1.0e-2,
        extinction_slope=2.0,
    ).sum(axis=0)
    assert negative[0] > negative[1] > negative[2]
    assert positive[0] < positive[1] < positive[2]
    assert negative[1] == positive[1] == 1.0


def test_tabulated_mapping_conserves_pressure_and_retains_exact_zeros() -> None:
    source_edges = np.array([1.0e-5, 1.0e-2, 10.0])
    source_wavelength = np.array([1.0, 5.0, 12.0])
    source = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]])
    target_edges = np.geomspace(1.0e-5, 100.0, 81)
    target_wavelength = np.array([0.3, 1.0, 5.0, 12.0])
    mapped = regrid_tabulated_extinction_tau(
        source_edges,
        source_wavelength,
        source,
        target_edges,
        target_wavelength,
    )
    assert np.all(mapped[:, :2] == 0.0)
    np.testing.assert_allclose(mapped.sum(axis=0)[2:], source.sum(axis=0)[1:], rtol=0.0, atol=2e-14)
    assert np.all(mapped[target_edges[:-1] >= 10.0] == 0.0)


def test_v2_contract_preserves_profiles_composition_and_disabled_scattering() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_7.build_stage_7_contract(common, 80)
    assert contract["case_id"].size == 150
    assert contract["profile_name"].tolist() == [
        "isothermal",
        "pg14_non_inverted",
        "pg14_inverted",
    ]
    np.testing.assert_array_equal(
        contract["temperature_cells_by_profile_k"][1],
        common.temperature_profiles_k["pg14_non_inverted_80_cells"],
    )
    np.testing.assert_array_equal(
        contract["temperature_cells_by_profile_k"][2],
        common.temperature_profiles_k["pg14_inverted_80_cells"],
    )
    np.testing.assert_allclose(np.sum(contract["gas_vmr"], axis=1), 1.0, rtol=0.0, atol=5e-16)
    assert np.all(contract["cloud_single_scattering_albedo"] == 0.0)
    assert np.all(contract["cloud_asymmetry_factor"] == 0.0)
    worker._validate_contract(contract, shared=False)


def test_clear_analytic_control_is_exact_zero_before_normalization() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    definitions = cloud_definitions()
    moderate = next(index for index, item in enumerate(definitions) if item.label == stage_7.PILOT_CLOUD_LABEL)
    contract = stage_7.build_stage_7_contract(
        common,
        40,
        profiles=("isothermal",),
        cloud_indices=np.array([0, moderate]),
    )
    wavelength = common.spectral.native_reference_wavelength_micron
    flux = np.broadcast_to(common.stellar_surface_flux_native_w_m2_m, (2, wavelength.size)).copy()
    vertical = np.full((2, 40, wavelength.size), 1.0 / 40.0, dtype=np.float32)
    payload = {
        "wavelength_micron": wavelength,
        "flux_w_m2_m": flux,
        "normalized_vertical_diagnostic": vertical,
        "runtime_s": np.zeros(2),
    }
    augmented = stage_7._augment_r100(common, contract, payload)
    assert np.all(augmented["r100_cloud_effect_flux_w_m2_m"] == 0.0)
    assert np.all(augmented["r100_cloud_effect_eclipse_ppm"] == 0.0)
    assert np.all(augmented["r100_normalized_cloud_effect"] == 0.0)


def test_shared_tensor_adds_identical_gas_and_cloud_inputs() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    definitions = cloud_definitions()
    moderate = next(index for index, item in enumerate(definitions) if item.label == stage_7.PILOT_CLOUD_LABEL)
    contract = stage_7.build_stage_7_contract(
        common,
        40,
        profiles=("pg14_non_inverted",),
        cloud_indices=np.array([0, moderate]),
    )
    wavelength = np.array([1.0, 5.0])
    cloud = build_cloud_extinction_matrix(
        contract["pressure_edges_bar"],
        wavelength,
        archived_pressure_edges_bar=contract["archived_pressure_edges_bar"],
        archived_wavelength_micron=contract["archived_wavelength_micron"],
        archived_extinction_tau=contract["archived_extinction_tau"],
    )
    robert = {
        "wavelength_micron": wavelength,
        "g_weights": np.array([0.5, 0.5]),
        "molecular_layer_tau_by_profile": np.full((1, 40, 2, 2), 0.25),
        "cia_h2_h2_layer_tau_by_profile": np.full((1, 40, 2), 0.10),
        "cia_h2_he_layer_tau_by_profile": np.full((1, 40, 2), 0.05),
        "native_cloud_extinction_tau_by_cloud": cloud,
    }
    shared = stage_7._shared_contract(contract, robert)
    expected_gas = (
        np.sum(
            robert["molecular_layer_tau_by_profile"][0]
            * robert["g_weights"][None, None, :],
            axis=-1,
        )
        + robert["cia_h2_h2_layer_tau_by_profile"][0]
        + robert["cia_h2_he_layer_tau_by_profile"][0]
    )
    np.testing.assert_array_equal(
        shared["shared_gas_layer_tau_by_profile"][0], expected_gas
    )
    np.testing.assert_array_equal(shared["shared_layer_tau"][0], expected_gas)
    np.testing.assert_array_equal(
        shared["shared_layer_tau"][1], expected_gas + cloud[moderate]
    )


def test_frozen_science_and_resource_gates() -> None:
    assert stage_7.STAGE_7_ACCEPTANCE_GATES["single_scattering_albedo_max_abs"] == 0.0
    assert stage_7.STAGE_7_ACCEPTANCE_GATES["pilot_projected_wall_time_max_s"] == 7200.0
    assert stage_7.STAGE_7_ACCEPTANCE_GATES["pilot_peak_rss_fraction_of_available_max"] == 0.60
    decision = pilot_resource_decision(
        {
            "track_a:robert": {"cold_wall_time_s": 1.0, "warm_wall_time_s": 0.5},
            "track_b:picaso": {"cold_wall_time_s": 2.0, "warm_wall_time_s": 1.0},
        },
        pilot_case_count=2,
        peak_process_tree_rss_bytes=500,
        available_memory_bytes=1000,
    )
    assert decision["continue_full_matrix"]
    unsafe = pilot_resource_decision(
        {"track_b:picaso": {"cold_wall_time_s": 50.0, "warm_wall_time_s": 50.0}},
        pilot_case_count=1,
        peak_process_tree_rss_bytes=700,
        available_memory_bytes=1000,
    )
    assert not unsafe["continue_full_matrix"]


def test_resource_override_is_explicit_and_preserves_frozen_decision() -> None:
    source = (ROOT / "examples/benchmark_emission_intercomparison_v2_stage_7.py").read_text()
    assert '"--override-resource-gate"' in source
    assert '"frozen_decision_preserved": True' in source
    assert "and not args.override_resource_gate" in source
    assert '"complete_matrix_peak_process_tree_rss_bytes"' in source
    assert '"complete_launcher_wall_time_s"' in source


def test_native_capability_boundaries_are_not_fabricated() -> None:
    source = (ROOT / "examples/run_emission_intercomparison_v2_stage_7_external.py").read_text()
    assert '"native_layer_tau_supported": np.array(False)' in source
    assert "absorbing-formal" in source
    assert "native SH contribution" in source
    assert "opacity sampling" not in source.lower()


def test_products_default_to_ignored_local_tree_and_none_are_tracked() -> None:
    assert stage_7.DEFAULT_PRODUCT_ROOT == stage_7.DEFAULT_OUTPUT / "products"
    ignored = subprocess.run(
        ["git", "check-ignore", str(stage_7.DEFAULT_PRODUCT_ROOT / "probe.npz")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "examples/outputs/emission_intercomparison/version_2/stage_7/**"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""


def test_plotting_cli_uses_manuscript_style_and_matplotlib_only() -> None:
    path = ROOT / "examples/plot_emission_intercomparison_v2_stage_7.py"
    source = path.read_text()
    assert "benchmark_style" in source
    assert "ROBERT_COLOR" in source
    assert "mediumpurple" not in source
    assert "#ef6c00" not in source
    assert "#2e7d32" not in source
    assert "matplotlib" in source
    assert "html" not in source.lower()
    completed = subprocess.run(
        [str(stage_7.ROBERT_PYTHON), str(path), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--product-root" in completed.stdout
