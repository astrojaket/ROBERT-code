"""Tests for self-contained run-directory creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_run_directory import create_run_directory
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = (
    ROOT
    / "configurations"
    / "targets"
    / "WASP-69b"
    / "wasp69b_cloud_free_R1000.yaml"
)
OE_CONFIG = (
    ROOT
    / "configurations"
    / "targets"
    / "WASP-69b"
    / "wasp69b_cloud_free_native_pg14_R1000_optimal_estimation.yaml"
)
NESTED_CONFIGS = (
    ROOT
    / "configurations"
    / "targets"
    / "WASP-69b"
    / "wasp69b_cloud_free_native_pg14_R1000.yaml",
    ROOT
    / "configurations"
    / "targets"
    / "WASP-69b"
    / "wasp69b_cloud_free_native_pg14_R1000_optimal_estimation_to_multinest.yaml",
)
TEMPLATE = ROOT / "configurations" / "examples" / "TEMPLATE_all_supported_options.yaml"


def test_create_run_directory_copies_runners_and_isolates_writable_paths(
    tmp_path: Path,
) -> None:
    run_directory = create_run_directory(
        project_dir=tmp_path / "my_project",
        source_config=SOURCE_CONFIG,
        slurm_account="dp448",
        slurm_partition="slurm",
        glamdring_ranks=10,
        slurm_mail_user="researcher@example.org",
    )
    config = load_task_config(run_directory / "configuration.yaml")

    assert run_directory.name == "wasp69b-cloud-free-native-modes-R1000"
    assert (run_directory / "source_configuration.yaml").is_file()
    assert (run_directory / "run_retrieval.py").is_file()
    assert (run_directory / "run_oe_from_nested.py").is_file()
    assert (run_directory / "run_forward.py").is_file()
    assert (run_directory / "postprocess_retrieval.py").is_file()
    assert (run_directory / "postprocess_forward.py").is_file()
    assert (run_directory / "submit.sh").stat().st_mode & 0o111
    glamdring_submission = (run_directory / "submit.sh").read_text(encoding="utf-8")
    assert "ENV_PREFIX" in glamdring_submission
    assert '"${PYTHON}" -u run_retrieval.py --config configuration.yaml' in (
        glamdring_submission
    )
    assert "ROBERT_MULTINEST_LIB" in glamdring_submission
    assert '-rmk user -launcher fork -n "${MPI_RANKS}"' in glamdring_submission
    assert "PMI*|PMIX*" in glamdring_submission
    run_readme = (run_directory / "README.md").read_text(encoding="utf-8")
    assert "ROBERT_MPI_RANKS=10" in run_readme
    assert "-s -c \"48 hours\" -n 1x10" in run_readme
    assert config.outputs.directory == run_directory / "outputs"
    assert config.opacity.cache_directory == run_directory / "opacity_cache"
    assert config.runtime.scratch_directory == run_directory / "scratch"
    submission = (run_directory / "submit.sbatch").read_text(encoding="utf-8")
    assert f"#SBATCH --chdir={run_directory}" in submission
    assert f'cd "{run_directory}"' in submission
    assert "#SBATCH --nodes=1" in submission
    assert "#SBATCH --ntasks=128" in submission
    assert "#SBATCH --ntasks-per-node=128" in submission
    assert "#SBATCH --account=dp448" in submission
    assert "#SBATCH --partition=slurm" in submission
    assert "#SBATCH --mail-user=researcher@example.org" in submission
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in submission
    assert 'mpirun -np "${SLURM_NTASKS}"' in submission
    assert (
        'export PYSYN_CDBS="${PYSYN_CDBS:?Set PYSYN_CDBS to the Synphot reference-data root}"'
        in submission
    )
    assert "--config configuration.yaml" in submission


def test_create_run_directory_uses_one_rank_for_optimal_estimation(
    tmp_path: Path,
) -> None:
    run_directory = create_run_directory(
        project_dir=tmp_path / "my_project",
        source_config=OE_CONFIG,
    )

    submission = (run_directory / "submit.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in submission
    assert "#SBATCH --ntasks=1" in submission
    assert "#SBATCH --ntasks-per-node=1" in submission
    assert "#SBATCH --mail-user=" not in submission
    assert "#SBATCH --mail-type=" not in submission
    assert 'mpirun -np "${SLURM_NTASKS}"' in submission
    readme = (run_directory / "README.md").read_text(encoding="utf-8")
    assert "ROBERT_MPI_RANKS=1" in readme


@pytest.mark.parametrize("config_path", NESTED_CONFIGS)
def test_create_run_directory_uses_128_ranks_for_nested_workflows(
    tmp_path: Path,
    config_path: Path,
) -> None:
    run_directory = create_run_directory(
        project_dir=tmp_path / "my_project",
        source_config=config_path,
    )

    submission = (run_directory / "submit.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in submission
    assert "#SBATCH --ntasks=128" in submission
    assert "#SBATCH --ntasks-per-node=128" in submission
    assert "#SBATCH --mail-user=" not in submission
    assert "#SBATCH --mail-type=" not in submission
    readme = (run_directory / "README.md").read_text(encoding="utf-8")
    assert "ROBERT_MPI_RANKS=12" in readme


def test_create_run_directory_refuses_to_mix_runs(tmp_path: Path) -> None:
    kwargs = {"project_dir": tmp_path, "source_config": SOURCE_CONFIG}
    create_run_directory(**kwargs)

    with pytest.raises(FileExistsError, match="already exists"):
        create_run_directory(**kwargs)


def test_create_run_directory_uses_top_level_paths_and_local_writable_defaults(
    tmp_path: Path,
) -> None:
    run_directory = create_run_directory(
        project_dir=tmp_path / "my_project",
        source_config=TEMPLATE,
    )
    config = load_task_config(run_directory / "configuration.yaml")

    assert config.paths is not None
    assert config.paths.project_directory == run_directory
    assert config.outputs.directory == run_directory / "outputs"
    assert config.opacity.cache_directory == run_directory / "opacity_cache"
    assert config.runtime.scratch_directory == run_directory / "scratch"
