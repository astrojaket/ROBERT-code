"""Controlled production-contract tests for Version-2 Stage 8."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import numpy as np

from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_8_pilots import (
    CONTROLLED_CASES_PER_PATH,
    CONTROLLED_CLOUD_STATES,
    CONTROLLED_PROFILE_GRIDS,
    CONTROLLED_SOLVER_PATHS,
    CURRENT_STAGE_8_STATUS,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "examples/benchmark_emission_intercomparison_v2_stage_8.py"
WORKER_PATH = ROOT / "examples/run_emission_intercomparison_v2_stage_8_pilot.py"
PLOTTER_PATH = ROOT / "examples/plot_emission_intercomparison_v2_stage_8.py"
OUTPUT = ROOT / "examples/outputs/emission_intercomparison/version_2/stage_8/controlled_study"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load("stage8_controlled_launcher", LAUNCHER_PATH)


def test_controlled_matrix_is_exactly_36_native_track_b_cases() -> None:
    assert CONTROLLED_PROFILE_GRIDS == (
        ("pg14_non_inverted", 40),
        ("pg14_non_inverted", 80),
        ("pg14_non_inverted", 160),
        ("isothermal", 80),
    )
    assert CONTROLLED_CASES_PER_PATH == 12
    assert len(CONTROLLED_SOLVER_PATHS) == 3
    assert CONTROLLED_CASES_PER_PATH * len(CONTROLLED_SOLVER_PATHS) == 36
    assert tuple(item.name for item in CONTROLLED_CLOUD_STATES) == (
        "clear",
        "absorbing_cloud",
        "scattering_cloud",
    )
    assert launcher.TRACK == "track_b_native_scattering"
    assert launcher.CURRENT_STAGE_8_STATUS == CURRENT_STAGE_8_STATUS


def test_controlled_contract_preserves_accepted_placement_for_every_grid() -> None:
    for profile, cells in CONTROLLED_PROFILE_GRIDS:
        contract = launcher.build_controlled_contract(profile, cells)
        assert contract["case_id"].size == 1
        assert contract["pressure_centers_bar"].size == cells
        assert str(contract["profile_name"][0]) == profile
        assert str(contract["study_status"]) == CURRENT_STAGE_8_STATUS
        assert bool(contract["controlled_study"])
        cloud = int(contract["case_cloud_index"][0])
        assert str(contract["cloud_label"][cloud]) == "deck_tau1_top10mbar_slope+0"
        assert float(contract["cloud_top_pressure_bar"][cloud]) == 1.0e-2
        assert float(contract["cloud_optical_depth_at_reference"][cloud]) == 1.0
        assert float(contract["cloud_extinction_slope"][cloud]) == 0.0


def test_controlled_worker_supports_all_frozen_grid_sizes_and_flux_output() -> None:
    source = WORKER_PATH.read_text()
    assert "{40, 80, 160}" in source
    assert '"flux_w_m2_m"' in source
    assert '"study_status"' in source
    assert '"cloud_state"' in source
    assert 'np.pi * spectrum' in source
    assert "track_a" not in source.lower()


def test_memory_override_is_narrow_recorded_and_never_parallel() -> None:
    source = LAUNCHER_PATH.read_text()
    assert "USER_AUTHORIZED_MEMORY_GATE_OVERRIDE = True" in source
    assert "ROBERT_ABSOLUTE_MIN_AVAILABLE_BYTES" in source
    assert '"user_authorized_memory_override_used"' in source
    assert "subprocess.Popen" in source
    assert "for solver in CONTROLLED_SOLVER_PATHS" in source
    assert "concurrent" not in source.lower()


def test_metrics_are_stable_for_exact_and_scaled_inputs() -> None:
    values = np.array([0.0, 1.0, -2.0, 4.0])
    assert launcher._p95_pair_peak(values, values) == 0.0
    assert launcher._symmetric_p95(values, values) == 0.0
    assert launcher._p95_pair_peak(values, 2.0 * values) > 0.0
    assert launcher._symmetric_p95(values, 2.0 * values) > 0.0


def test_plotter_uses_maintained_style_and_no_colour_only_comparison() -> None:
    source = PLOTTER_PATH.read_text()
    assert "benchmark_style" in source
    assert "ROBERT_COLOR" in source
    assert "REFERENCE_COLOR" in source
    assert "linestyle" in source
    assert "controlled_primary.png" in source
    assert "controlled_convergence.png" in source


def test_controlled_products_are_ignored_and_none_are_tracked() -> None:
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


def test_controlled_cli_has_no_broad_matrix_selector() -> None:
    completed = subprocess.run(
        ["/opt/miniconda3/envs/robert-exoplanets/bin/python", str(LAUNCHER_PATH), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--subsection" not in completed.stdout
    assert "--resume" in completed.stdout
