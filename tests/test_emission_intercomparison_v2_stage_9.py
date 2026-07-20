from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

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
NATIVE = ROOT / "examples/emission_intercomparison_v2_stage_9_native.py"
MPI_LAUNCHER = (
    ROOT / "scripts/launch_emission_intercomparison_v2_stage_9_mpi.sh"
)
TASK_LAUNCHER = ROOT / "scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
PRODUCTION_LAUNCHER = (
    ROOT / "scripts/submit_emission_intercomparison_v2_stage_9.sh"
)
SHARD_SUBMITTER = (
    ROOT / "scripts/queue_emission_intercomparison_v2_stage_9_shard.sh"
)


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("stage9_prepare_for_tests", PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_native_module():
    spec = importlib.util.spec_from_file_location("stage9_native_for_tests", NATIVE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_stage_9_matrix_counts_and_resources() -> None:
    runs = build_run_matrix()
    assert len(runs) == 72
    assert len({run.run_id for run in runs}) == 72
    assert len({run.shard_id for run in runs}) == 12
    assert {
        sum(item.shard_id == shard for item in runs)
        for shard in {item.shard_id for item in runs}
    } == {6}
    assert sum(run.control == "directed_cross_framework" for run in runs) == 72
    assert sum(run.control == "self_retrieval_control" for run in runs) == 0
    assert {run.mpi_ranks for run in runs} == {12}
    assert {run.threads_per_rank for run in runs} == {1}
    assert MULTINEST_SETTINGS["n_live_points"] == 400
    assert MULTINEST_SETTINGS["mpi_nprocs"] == 12
    assert len(GAUSSIAN_NOISE_SEEDS) == 0
    assert {run.noise_id for run in runs} == {"mean"}


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


def test_picaso_projection_uses_native_bin_support_without_interpolation() -> None:
    module = _load_native_module()
    projected = module._native_bin_overlap_mean(
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
        np.array([2.0, 4.0]),
        np.array([0.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(projected, [3.0, 4.0])

    center, lower, upper, order = module._picaso_native_wavelength_support(
        np.array([1000.0, 2000.0]),
        np.array([1000.0, 1000.0]),
    )
    np.testing.assert_allclose(center, [5.0, 10.0])
    np.testing.assert_allclose(lower, [4.0, 1.0e4 / 1500.0])
    np.testing.assert_allclose(upper, [1.0e4 / 1500.0, 20.0])
    np.testing.assert_array_equal(order, [1, 0])


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
    assert expected["execution"]["scheduler_queue"] == "redwood"
    assert expected["execution"]["addqueue_allocation"] == "single_wrapper_n12"
    assert expected["execution"]["mpi_launcher"] == "conda_mpich_hydra_fork"
    assert expected["execution"]["single_node_required"] is True
    assert expected["noise"]["spectral_points_randomized"] is False
    assert (
        expected["fixed"]["picaso_r100_projection"]
        == "native_wavenumber_bin_support_overlap_no_center_interpolation"
    )


def test_directory_generator_creates_and_verifies_complete_tree(tmp_path: Path) -> None:
    module = _load_prepare_module()
    project = tmp_path / "stage9"
    summary = module.prepare(project)
    assert summary["run_count"] == 72
    assert summary["shard_count"] == 12
    assert summary["noise_vector_count"] == 0
    assert len(list((project / "runs").glob("*/*/*/run.json"))) == 72
    assert len(list((project / "shards").glob("*.json"))) == 12
    assert not (project / "noise").exists()
    assert len(list((project / "injections").glob("*/*"))) >= 12
    module.prepare(project, verify_only=True)


def test_execution_contract_refresh_preserves_pre_science_deployment(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    project = tmp_path / "stage9"
    expected_manifest = module.prepare(project)

    contract_path = project / "contracts" / "stage_9_retrieval_contract.json"
    old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    old_contract["execution"] = {
        "cluster": "glamdring",
        "scheduler_queue": "redwood",
        "mpi_ranks_per_retrieval": 12,
        "threads_per_rank": 1,
        "nested_mpirun_forbidden_under_addqueue": True,
    }
    contract_path.write_text(
        json.dumps(old_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = project / "integrity" / "setup_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_manifest["stage_9_contract_repository_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(old_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    refreshed = module.prepare(project, refresh_execution_contract=True)
    assert refreshed == expected_manifest
    assert json.loads(contract_path.read_text(encoding="utf-8")) == json.loads(
        CONTRACT.read_text(encoding="utf-8")
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest

    injection = (
        project
        / "injections"
        / "robert"
        / "clear_non_inverted"
        / "native_mean.npz"
    )
    injection.touch()
    contract_path.write_text(
        json.dumps(old_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="science products exist"):
        module.prepare(project, refresh_execution_contract=True)


def test_glamdring_launchers_use_one_wrapper_and_conda_mpich() -> None:
    assert MPI_LAUNCHER.stat().st_mode & 0o111
    mpi_text = MPI_LAUNCHER.read_text(encoding="utf-8")
    assert 'PMI*|PMIX*) unset "$name"' in mpi_text
    assert "export HYDRA_LAUNCHER=fork" in mpi_text
    assert "export HYDRA_RMK=user" in mpi_text
    assert "-rmk user" in mpi_text
    assert "-launcher fork" in mpi_text

    task_text = TASK_LAUNCHER.read_text(encoding="utf-8")
    production_text = PRODUCTION_LAUNCHER.read_text(encoding="utf-8")
    assert "launch_emission_intercomparison_v2_stage_9_mpi.sh" in task_text
    assert "launch_emission_intercomparison_v2_stage_9_mpi.sh" in production_text

    shard_text = SHARD_SUBMITTER.read_text(encoding="utf-8")
    assert "addqueue -q redwood -s" in shard_text
    assert "-n 12" in shard_text
    assert "1x12" not in shard_text
