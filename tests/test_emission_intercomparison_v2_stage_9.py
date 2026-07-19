from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (
    GAUSSIAN_NOISE_SEEDS,
    MULTINEST_SETTINGS,
    SCENARIOS,
    build_run_matrix,
    frozen_contract_payload,
    parameter_definitions,
)


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "docs/data/emission_intercomparison/version_2/common_contract.json"
CONTRACT = (
    ROOT
    / "docs/data/emission_intercomparison/version_2/stage_9_retrieval_contract.json"
)
PREPARE = ROOT / "scripts/prepare_emission_intercomparison_v2_stage_9.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("stage9_prepare_for_tests", PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_stage_9_matrix_counts_and_resources() -> None:
    runs = build_run_matrix()
    assert len(runs) == 504
    assert len({run.run_id for run in runs}) == 504
    assert len({run.shard_id for run in runs}) == 12
    assert {
        sum(item.shard_id == shard for item in runs)
        for shard in {item.shard_id for item in runs}
    } == {42}
    assert sum(run.control == "directed_cross_framework" for run in runs) == 432
    assert sum(run.control == "self_retrieval_control" for run in runs) == 72
    assert {run.mpi_ranks for run in runs} == {12}
    assert {run.threads_per_rank for run in runs} == {1}
    assert MULTINEST_SETTINGS["n_live_points"] == 400
    assert MULTINEST_SETTINGS["mpi_nprocs"] == 12
    assert len(GAUSSIAN_NOISE_SEEDS) == 5


def test_stage_9_parameters_exclude_area_and_retrieve_clouds() -> None:
    dimensions = {
        scenario.name: len(parameter_definitions(scenario)) for scenario in SCENARIOS
    }
    assert dimensions == {
        "clear_non_inverted": 9,
        "clear_inverted": 9,
        "grey_absorbing_non_inverted": 11,
        "grey_scattering_non_inverted": 12,
    }
    for scenario in SCENARIOS:
        names = {item.name for item in parameter_definitions(scenario)}
        assert "area_scale" not in names
        assert "log10_area_scale" not in names
        if scenario.cloudy:
            assert {"log10_cloud_tau_5um", "log10_cloud_top_pressure_bar"} <= names
        if scenario.cloud == "grey_isotropic_scattering":
            assert "cloud_single_scattering_albedo" in names


def test_committed_stage_9_contract_matches_source_of_truth() -> None:
    sha = hashlib.sha256(COMMON.read_bytes()).hexdigest()
    expected = frozen_contract_payload(common_contract_sha256=sha)
    assert json.loads(CONTRACT.read_text(encoding="utf-8")) == expected
    assert expected["scope_exclusions"] == [
        "Track A shared-tensor retrievals",
        "MgSiO3 or other condensate microphysics",
        "Mie optical properties",
        "anisotropic scattering",
        "wavelength-dependent cloud extinction",
        "high-order scattering methods",
        "fabricated shared opacity or cloud tensors",
    ]


def test_directory_generator_creates_and_verifies_complete_tree(tmp_path: Path) -> None:
    module = _load_prepare_module()
    project = tmp_path / "stage9"
    summary = module.prepare(project)
    assert summary["run_count"] == 504
    assert summary["shard_count"] == 12
    assert summary["noise_vector_count"] == 60
    assert len(list((project / "runs").glob("*/*/*/run.json"))) == 504
    assert len(list((project / "shards").glob("*.json"))) == 12
    assert len(list((project / "noise").glob("*/*/*.npz"))) == 60
    assert len(list((project / "injections").glob("*/*"))) >= 12
    module.prepare(project, verify_only=True)

    first = next((project / "noise").glob("*/*/*.npz"))
    with np.load(first, allow_pickle=False) as archive:
        assert archive["standard_normal"].ndim == 1
        assert archive["standard_normal"].size > 100
