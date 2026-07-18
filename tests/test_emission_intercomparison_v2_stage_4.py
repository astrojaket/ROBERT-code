"""Contract and artifact tests for emission intercomparison Version-2 Stage 4."""

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
MODULE_PATH = ROOT / "examples/benchmark_emission_intercomparison_v2_stage_4.py"
SPEC = importlib.util.spec_from_file_location("emission_v2_stage_4", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stage_4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_4)
WORKER_PATH = ROOT / "examples/run_emission_intercomparison_v2_stage_4_external.py"
WORKER_SPEC = importlib.util.spec_from_file_location("emission_v2_stage_4_worker", WORKER_PATH)
assert WORKER_SPEC is not None and WORKER_SPEC.loader is not None
stage_4_worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(stage_4_worker)

RESOLUTIONS = (40, 80, 160)
PROFILES = ("isothermal", "pg14_non_inverted", "pg14_inverted")
MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_4_contract_uses_exact_profiles_composition_and_both_cia(
    n_cells: int,
) -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_4.build_stage_4_contract(common, n_cells)
    grid = next(item for item in common.pressure_grids if item.n_cells == n_cells)
    gas_names = tuple(common.composition_vmr)
    reference_vmr = np.asarray([common.composition_vmr[name] for name in gas_names])

    assert contract["profile_name"].tolist() == list(PROFILES)
    assert contract["case_id"].tolist() == [
        f"{profile}_{stage_4.FIXED_FACTOR}_{n_cells}_cells" for profile in PROFILES
    ]
    np.testing.assert_array_equal(contract["pressure_edges_bar"], grid.edges_bar)
    np.testing.assert_array_equal(contract["pressure_centers_bar"], grid.centers_bar)
    np.testing.assert_array_equal(
        contract["picaso_pressure_levels_bar"], grid.picaso_levels_bar
    )
    np.testing.assert_array_equal(
        contract["petitradtrans_pressure_nodes_bar"], grid.petitradtrans_nodes_bar
    )
    np.testing.assert_array_equal(
        contract["gas_vmr"], np.broadcast_to(reference_vmr, (len(PROFILES), 6))
    )
    assert contract["molecular_species_name"].tolist() == list(MOLECULAR_SPECIES)
    assert np.all(contract["molecular_species_active"])
    assert np.all(contract["include_h2_h2_cia"])
    assert np.all(contract["include_h2_he_cia"])
    for index, profile in enumerate(PROFILES):
        np.testing.assert_array_equal(
            contract["temperature_cells_k"][index],
            common.temperature_profiles_k[f"{profile}_{n_cells}_cells"],
        )


def test_stage_4_gates_and_bands_are_frozen_before_matrix() -> None:
    assert stage_4.STAGE_4_ACCEPTANCE_GATES == {
        "track_a_max_abs_symmetric_relative": 5.0e-4,
        "track_a_max_abs_eclipse_difference_ppm": 0.1,
        "track_a_80_to_160_max_abs_eclipse_difference_ppm": 0.1,
        "track_a_isothermal_max_abs_eclipse_difference_ppm": 0.1,
        "track_a_contribution_centroid_p95_abs_difference_dex": 0.01,
        "track_a_contribution_profile_total_variation_p95": 0.01,
        "scattering_single_scattering_albedo_max_abs": 0.0,
        "pilot_projected_wall_time_max_s": 7200.0,
        "pilot_peak_rss_fraction_of_available_max": 0.60,
    }
    assert stage_4.BAND_WINDOWS_MICRON == {
        "optical": (0.3, 0.8),
        "near_ir_water_band": (1.35, 1.55),
        "near_ir_window": (2.0, 2.3),
        "methane_band": (3.1, 3.6),
        "co_co2_band": (4.2, 5.0),
        "mid_ir_water_band": (5.5, 7.5),
        "mid_ir_window": (8.0, 10.0),
    }


def test_stage_4_worker_rejects_missing_fixed_cia() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    contract = stage_4.build_stage_4_contract(common, 40)
    contract["include_h2_he_cia"][0] = False
    with pytest.raises(ValueError, match="fixes H2-He CIA on"):
        stage_4_worker._validate_contract(contract)


def test_stage_4_band_diagnostics_accept_single_case_arrays() -> None:
    common = load_version_2_common_contract(DATA / "common_contract.json")
    flux = np.ones((1, 369))
    contribution = np.zeros((1, 40, 369))
    contribution[:, 10] = 1.0
    pressure = next(item for item in common.pressure_grids if item.n_cells == 40).centers_bar

    diagnostics = stage_4._band_diagnostics(common, flux, contribution, pressure)

    assert set(diagnostics) == set(stage_4.BAND_WINDOWS_MICRON)
    assert all(item["bin_count"] > 0 for item in diagnostics.values())


def test_stage_4_committed_checksums_and_integrity() -> None:
    report_path = DATA / "stage_4_report.json"
    integrity_path = DATA / "stage_4_integrity.json"
    if not report_path.exists() or not integrity_path.exists():
        pytest.skip("Stage-4 numerical products have not been generated")
    checksums = json.loads((DATA / "checksums.json").read_text())
    stage_names = {name for name in checksums if name.startswith("stage_4_")}
    integrity = json.loads(integrity_path.read_text())["artifacts"]
    assert len(stage_names) == 26
    assert set(integrity) == stage_names - {integrity_path.name}
    for name in stage_names:
        path = DATA / name
        assert path.is_file()
        assert checksums[name] == _sha256(path)
        if name != integrity_path.name:
            assert integrity[name]["sha256"] == checksums[name]
            assert integrity[name]["size_bytes"] == path.stat().st_size
        if path.suffix == ".npz":
            assert path.stat().st_size < 100_000_000


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_4_artifacts_retain_native_r100_and_vertical_products(
    n_cells: int,
) -> None:
    picaso_path = DATA / f"stage_4_picaso_{n_cells}_cells.npz"
    if not picaso_path.exists():
        pytest.skip("Stage-4 numerical products have not been generated")
    paths = {
        "picaso": (picaso_path, 661),
        "petitradtrans": (
            DATA / f"stage_4_petitradtrans_{n_cells}_cells.npz",
            3697,
        ),
        "robert_shared": (
            DATA / f"stage_4_robert_shared_{n_cells}_cells.npz",
            3696,
        ),
        "petitradtrans_shared": (
            DATA / f"stage_4_petitradtrans_shared_{n_cells}_cells.npz",
            3696,
        ),
    }
    for _name, (path, n_native) in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            assert archive["profile_name"].tolist() == list(PROFILES)
            assert archive["native_wavelength_micron"].shape == (n_native,)
            assert archive["native_flux_w_m2_m"].shape == (3, n_native)
            assert archive["native_eclipse_depth"].shape == (3, n_native)
            assert archive["r100_flux_w_m2_m"].shape == (3, 369)
            assert archive["r100_eclipse_depth"].shape == (3, 369)
            assert archive["normalized_vertical_native"].shape == (
                3,
                n_cells,
                n_native,
            )
            assert archive["normalized_vertical_r100"].shape == (3, n_cells, 369)
            assert archive["pressure_centroid_r100_log10_bar"].shape == (3, 369)
            assert archive["peak_pressure_r100_bar"].shape == (3, 369)


@pytest.mark.parametrize("n_cells", RESOLUTIONS)
def test_stage_4_robert_shards_retain_full_component_tensors(n_cells: int) -> None:
    first = DATA / f"stage_4_robert_isothermal_{n_cells}_cells.npz"
    if not first.exists():
        pytest.skip("Stage-4 numerical products have not been generated")
    for profile in PROFILES:
        path = DATA / f"stage_4_robert_{profile}_{n_cells}_cells.npz"
        with np.load(path, allow_pickle=False) as archive:
            assert archive["profile_name"].tolist() == [profile]
            assert archive["molecular_layer_tau_by_profile"].shape == (
                1,
                n_cells,
                3696,
                16,
            )
            assert archive["cia_h2_h2_layer_tau_by_profile"].shape == (
                1,
                n_cells,
                3696,
            )
            assert archive["cia_h2_he_layer_tau_by_profile"].shape == (
                1,
                n_cells,
                3696,
            )


def test_stage_4_picaso_correction_and_capability_metadata() -> None:
    path = DATA / "stage_4_picaso_80_cells.npz"
    if not path.exists():
        pytest.skip("Stage-4 numerical products have not been generated")
    common = load_version_2_common_contract(DATA / "common_contract.json")
    expected_sum = sum(common.composition_vmr[name] for name in MOLECULAR_SPECIES)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["absolute_line_vmr_restored_after_resort_rebin"] is True
        assert metadata["absolute_line_vmr_sum"] == pytest.approx(expected_sum)
        assert archive["layer_tau"].shape == (3, 80, 661, 8)
        assert archive["native_framework_probe_flux_w_m2_m"].shape == (3, 661)
        assert float(archive["maximum_abs_rayleigh_tau"]) == 0.0
        assert float(archive["maximum_abs_cloud_tau"]) == 0.0
        assert any("not native SH" in item for item in metadata["limitations"])


def test_stage_4_prt_does_not_fabricate_native_tau() -> None:
    path = DATA / "stage_4_petitradtrans_80_cells.npz"
    if not path.exists():
        pytest.skip("Stage-4 numerical products have not been generated")
    with np.load(path, allow_pickle=False) as archive:
        assert "layer_tau" not in archive.files
        metadata = json.loads(str(archive["metadata_json"]))
        assert any("does not expose" in item for item in metadata["limitations"])


def test_stage_4_opacity_sampling_products_are_absent() -> None:
    assert not list(DATA.glob("stage_4_picaso_opacity_sampling_*.npz"))
    assert not list(DATA.glob("stage_4_picaso_sampling_*.npz"))
