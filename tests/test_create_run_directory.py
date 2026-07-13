"""Tests for self-contained run-directory creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_run_directory import create_run_directory
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configurations" / "wasp69b_cloud_free_R1000.yaml"
TEMPLATE = ROOT / "configurations" / "TEMPLATE_all_supported_options.yaml"


def test_create_run_directory_copies_runners_and_isolates_writable_paths(
    tmp_path: Path,
) -> None:
    run_directory = create_run_directory(
        project_dir=tmp_path / "my_project",
        source_config=SOURCE_CONFIG,
    )
    config = load_task_config(run_directory / "configuration.yaml")

    assert run_directory.name == "wasp69b-cloud-free-native-modes-R1000"
    assert (run_directory / "source_configuration.yaml").is_file()
    assert (run_directory / "run_retrieval.py").is_file()
    assert (run_directory / "run_forward.py").is_file()
    assert config.outputs.directory == run_directory / "outputs"
    assert config.opacity.cache_directory == run_directory / "opacity_cache"
    assert config.runtime.scratch_directory == run_directory / "scratch"
    submission = (run_directory / "submit.sbatch").read_text(encoding="utf-8")
    assert f"#SBATCH --chdir={run_directory}" in submission
    assert f'cd "{run_directory}"' in submission
    assert 'mpirun -np "${SLURM_NTASKS}"' in submission
    assert "--config configuration.yaml" in submission


def test_create_run_directory_refuses_to_mix_runs(tmp_path: Path) -> None:
    kwargs = {"project_dir": tmp_path, "source_config": SOURCE_CONFIG}
    create_run_directory(**kwargs)

    with pytest.raises(FileExistsError, match="already exists"):
        create_run_directory(**kwargs)


def test_create_run_directory_updates_housekeeping_writable_paths(tmp_path: Path) -> None:
    run_directory = create_run_directory(
        project_dir=tmp_path / "my_project",
        source_config=TEMPLATE,
    )
    config = load_task_config(run_directory / "configuration.yaml")

    assert config.housekeeping is not None
    assert config.housekeeping.output_directory == run_directory / "outputs"
    assert config.housekeeping.opacity_cache_directory == run_directory / "opacity_cache"
    assert config.housekeeping.scratch_directory == run_directory / "scratch"
