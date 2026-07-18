"""Frozen contract and isolation tests for Version-2 Stage-8 pilots."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_8_pilots import (
    ACCEPTED_MODERATE_TAU,
    ASSEMBLY_OVERHEAD_FACTOR,
    ASYMMETRY_LADDER,
    CELL_COUNTS,
    CONTROLLED_CASES_PER_PATH,
    CONTROLLED_CLOUD_STATES,
    CONTROLLED_PROFILE_GRIDS,
    CONTROLLED_SHARDS_PER_PATH,
    CONTROLLED_SOLVER_PATHS,
    CURRENT_STAGE_8_STATUS,
    DELTA_M_OPTIONS,
    FLOAT_PRECISION,
    FUTURE_STUDY_STATUS,
    OMEGA0_LADDER,
    PILOT_PATHS,
    PLANNED_CLOUD_SCOPE,
    PROFILES,
    REFERENCE_ANGLE_COUNTS,
    SHARDS_PER_PATH,
    TAU_LADDER,
    TRACK,
    UNRESOLVED_STRESS_TAU,
    contract_payload,
    controlled_study_payload,
    project_controlled_study_resources,
    project_path_resources,
    planned_cloud_scope_payload,
    supported_paths,
    validate_frozen_contract,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "examples/benchmark_emission_intercomparison_v2_stage_8_pilots.py"
WORKER = ROOT / "examples/run_emission_intercomparison_v2_stage_8_pilot.py"
OUTPUT = ROOT / "examples/outputs/emission_intercomparison/version_2/stage_8/pilots"
CONTROLLED_CONTRACT = (
    ROOT
    / "docs/data/emission_intercomparison/version_2/stage_8_controlled_study_contract.json"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_is_track_b_only_and_covers_every_subsection() -> None:
    validate_frozen_contract()
    payload = contract_payload()
    assert TRACK == "track_b_native_scattering"
    assert "track_a" not in TRACK
    assert payload["stage"] == 8
    assert payload["status"] == FUTURE_STUDY_STATUS
    assert {item.subsection for item in PILOT_PATHS} == {"8A", "8B", "8C", "8D", "8E"}
    assert all(item.full_case_count > 0 for item in supported_paths())
    assert all(item.full_case_count == 0 for item in PILOT_PATHS if not item.supported)


def test_scattering_parameters_and_precision_are_exactly_frozen() -> None:
    assert OMEGA0_LADDER == (0.5, 0.9, 0.99)
    assert TAU_LADDER == (0.1, 1.0, 10.0, 100.0)
    assert ACCEPTED_MODERATE_TAU == (0.1, 1.0)
    assert UNRESOLVED_STRESS_TAU == (10.0, 100.0)
    assert ASYMMETRY_LADDER == (0.3, 0.6, 0.9)
    assert DELTA_M_OPTIONS == (False, True)
    assert REFERENCE_ANGLE_COUNTS == (16, 32)
    assert FLOAT_PRECISION == "float64"
    assert CELL_COUNTS == (40, 80, 160)
    assert PROFILES == ("isothermal", "pg14_non_inverted", "pg14_inverted")


def test_mgsio3_scope_is_explicitly_deferred_to_future_study() -> None:
    scope = planned_cloud_scope_payload()
    assert PLANNED_CLOUD_SCOPE.material == "MgSiO3"
    assert scope["optical_constants"] == "data/optical_constants/exo_skryer/MgSiO3.txt"
    assert "independent native-code" in str(scope["comparison"])
    assert "genuinely supported native" in str(scope["varied_definition"])
    assert scope["common_cloud_tensors_allowed"] is False


def test_current_stage_8_is_the_controlled_grey_isotropic_study() -> None:
    payload = controlled_study_payload()
    assert payload["status"] == CURRENT_STAGE_8_STATUS
    assert payload["track"] == TRACK
    assert payload["cloud_description"] == (
        "idealized grey aerosol; no material-specific claim"
    )
    assert payload["cloud_top_pressure_bar"] == 1.0e-2
    assert payload["cloud_bottom_pressure_bar"] == 100.0
    assert payload["cloud_reference_wavelength_micron"] == 5.0
    assert payload["cloud_extinction_slope"] == 0.0
    assert CONTROLLED_PROFILE_GRIDS == (
        ("pg14_non_inverted", 40),
        ("pg14_non_inverted", 80),
        ("pg14_non_inverted", 160),
        ("isothermal", 80),
    )
    assert CONTROLLED_SHARDS_PER_PATH == 4
    assert CONTROLLED_CASES_PER_PATH == 12
    assert payload["total_native_cases"] == 36
    states = {
        item.name: (
            item.cloud_tau_at_5_micron,
            item.single_scattering_albedo,
            item.asymmetry_factor,
        )
        for item in CONTROLLED_CLOUD_STATES
    }
    assert states == {
        "clear": (0.0, 0.0, 0.0),
        "absorbing_cloud": (1.0, 0.0, 0.0),
        "scattering_cloud": (1.0, 0.9, 0.0),
    }
    paths = {(item.framework, item.path) for item in CONTROLLED_SOLVER_PATHS}
    assert paths == {
        ("robert", "sh4_p3_isotropic"),
        ("picaso", "sh4_isotropic"),
        ("petitradtrans", "feautrier_isotropic"),
    }


def test_versioned_controlled_contract_matches_executable_contract() -> None:
    documented = json.loads(CONTROLLED_CONTRACT.read_text())
    executable = controlled_study_payload()
    assert documented["status"] == executable["status"]
    assert documented["science_question"] == executable["science_question"]
    assert documented["states"] == executable["states"]
    assert documented["profile_grids"] == executable["profile_grids"]
    assert documented["solver_paths"] == executable["solver_paths"]
    assert documented["matrix"]["cases_per_framework"] == executable[
        "cases_per_framework"
    ]
    assert documented["matrix"]["total_native_cases"] == executable[
        "total_native_cases"
    ]
    assert documented["future_study"]["status"] == FUTURE_STUDY_STATUS
    common_path = ROOT / documented["common_contract"]["path"]
    assert hashlib.sha256(common_path.read_bytes()).hexdigest() == documented[
        "common_contract"
    ]["file_sha256"]


def test_documented_resource_prediction_replays_frozen_pilot_measurements() -> None:
    documented = json.loads(CONTROLLED_CONTRACT.read_text())
    measurements = {
        "robert": {
            "cold_wall_s": 18.737115750089288,
            "warm_wall_s": 17.768376749940217,
            "warm_case_s": 10.83425354200881,
            "peak_rss_bytes": 6680330240,
            "available_memory_bytes": 11029168128,
            "native_wavelength_count": 3696,
        },
        "picaso": {
            "cold_wall_s": 5.22358312504366,
            "warm_wall_s": 4.981940165976994,
            "warm_case_s": 1.6433335000183433,
            "peak_rss_bytes": 991641600,
            "available_memory_bytes": 11608113152,
            "native_wavelength_count": 661,
        },
        "petitradtrans": {
            "cold_wall_s": 5.090201416052878,
            "warm_wall_s": 4.903907416970469,
            "warm_case_s": 2.6601395410252735,
            "peak_rss_bytes": 2265350144,
            "available_memory_bytes": 11558584320,
            "native_wavelength_count": 3697,
        },
    }
    projected_total = 0.0
    for row in documented["resource_projection"]["frameworks"]:
        result = project_controlled_study_resources(**measurements[row["framework"]])
        assert result["projected_wall_time_s"] == row["projected_wall_time_s"]
        assert result["projected_peak_rss_bytes"] == row[
            "projected_peak_rss_bytes"
        ]
        assert result["projected_peak_fraction_of_available"] == row[
            "peak_fraction_of_available"
        ]
        assert result["strict_laptop_memory_gate"] == row[
            "strict_60_percent_memory_gate"
        ]
        projected_total += float(result["projected_wall_time_s"])
    assert projected_total == documented["resource_projection"][
        "combined_serial_wall_time_s"
    ]


def test_full_case_counts_include_profiles_grids_and_controls() -> None:
    counts = {(item.subsection, item.framework, item.path): item.full_case_count for item in supported_paths()}
    assert counts[("8A", "robert", "native_absorption")] == 18
    assert counts[("8B", "robert", "toon_isotropic")] == 126
    assert counts[("8C", "picaso", "sh4_delta_on")] == 45
    assert counts[("8D", "robert", "physical_mie_exact_moments_sh4")] == 27
    assert counts[("8E", "petitradtrans", "feautrier_32_angles")] == 27


def test_capability_boundaries_are_explicit_and_not_substituted() -> None:
    unsupported = {(item.subsection, item.framework, item.path): item.capability_note for item in PILOT_PATHS if not item.supported}
    assert ("8C", "petitradtrans", "arbitrary_anisotropy_delta_m") in unsupported
    assert ("8D", "picaso", "native_atmospheric_microphysics") in unsupported
    assert ("8D", "petitradtrans", "native_microphysical_cloud") in unsupported
    assert ("8E", "robert", "high_order_reference") in unsupported
    assert ("8E", "picaso", "high_order_reference") in unsupported
    assert all(note.strip() for note in unsupported.values())


def test_projection_and_memory_accounting_follow_frozen_formula() -> None:
    result = project_path_resources(
        cold_wall_s=12.0,
        warm_wall_s=10.0,
        warm_setup_s=2.0,
        warm_case_s=0.5,
        peak_rss_bytes=1_000,
        available_memory_bytes=10_000,
        retained_tensor_bytes=100,
        full_case_count=45,
    )
    assert SHARDS_PER_PATH == 9
    assert ASSEMBLY_OVERHEAD_FACTOR == 1.25
    assert result["cold_penalty_s"] == 2.0
    assert result["raw_wall_time_s"] == 42.5
    assert result["projected_wall_time_s"] == 53.125
    assert result["max_cases_per_shard"] == 5
    assert result["projected_peak_rss_bytes"] == 1_400
    assert result["projected_peak_fraction_of_available"] == 0.14


def test_controlled_study_projection_uses_four_sequential_state_shards() -> None:
    result = project_controlled_study_resources(
        cold_wall_s=12.0,
        warm_wall_s=10.0,
        warm_case_s=0.5,
        peak_rss_bytes=1_000,
        available_memory_bytes=10_000,
        native_wavelength_count=100,
    )
    assert result["cold_penalty_s"] == 2.0
    assert result["raw_wall_time_s"] == 46.0
    assert result["projected_wall_time_s"] == 57.5
    assert result["projected_peak_rss_bytes"] == 2_600
    assert result["projected_peak_fraction_of_available"] == 0.26
    assert result["strict_laptop_memory_gate"]
    assert result["operationally_feasible_if_serial"]
    assert result["recommended_shards"] == 4
    assert result["cases_per_shard"] == 3


def test_pilot_contract_uses_one_accepted_80_cell_stage7_case() -> None:
    launcher = _load("stage8_launcher_for_test", LAUNCHER)
    contract = launcher.build_pilot_contract()
    assert contract["case_id"].size == 1
    assert contract["pressure_centers_bar"].size == 80
    assert str(contract["profile_name"][0]) == "pg14_non_inverted"
    assert str(contract["track"]) == TRACK
    assert bool(contract["pilot_only"])
    cloud = int(contract["case_cloud_index"][0])
    assert str(contract["cloud_label"][cloud]) == "deck_tau1_top10mbar_slope+0"
    assert np.all(contract["cloud_single_scattering_albedo"] == 0.0)


def test_exact_interpreters_and_picaso_environment_are_enforced() -> None:
    source = WORKER.read_text()
    launcher = LAUNCHER.read_text()
    for interpreter in (
        "/opt/miniconda3/envs/robert-exoplanets/bin/python",
        "/opt/miniconda3/envs/picaso-v4/bin/python",
        "/opt/miniconda3/envs/petitradtrans-stable/bin/python",
    ):
        assert interpreter in source or interpreter in launcher
    assert '"picaso_refdata"' in source and '"NUMBA_CACHE_DIR"' in source and '"MPLCONFIGDIR"' in source
    assert 'environment["picaso_refdata"]' in launcher
    assert 'environment["NUMBA_CACHE_DIR"]' in launcher
    assert 'environment["MPLCONFIGDIR"]' in launcher
    assert 'environment["HOME"]' not in source + launcher
    assert "historical PICASO interpreter is forbidden" in source


def test_worker_and_launcher_offer_no_production_or_track_a_mode() -> None:
    source = (WORKER.read_text() + LAUNCHER.read_text()).lower()
    assert "track_a_shared" not in source
    assert '"production_matrix_started": false' in source
    completed = subprocess.run(
        ["/opt/miniconda3/envs/robert-exoplanets/bin/python", str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "production" in completed.stdout.lower()
    assert "--subsection" in completed.stdout


def test_generated_products_are_ignored_and_absent_from_git() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", str(OUTPUT / "probe.npz")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "examples/outputs/emission_intercomparison/version_2/stage_8/**"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""
