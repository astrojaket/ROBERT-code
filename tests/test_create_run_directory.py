"""Tests for self-contained run-directory creation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.create_run_directory import create_run_directory
from robert_exoplanets.io.task_config import load_task_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configurations" / "emission.yaml"
OE_CONFIG = ROOT / "configurations" / "optimal_estimation.yaml"
MULTINEST_CONFIGS = (
    ROOT / "configurations" / "emission.yaml",
    ROOT / "configurations" / "two_region_emission.yaml",
)
TEMPLATE = ROOT / "configurations" / "quickstart.yaml"


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

    assert run_directory.name == "hot-jupiter-emission-r1000"
    assert (run_directory / "source_configuration.yaml").is_file()
    assert (run_directory / "run_retrieval.py").is_file()
    assert (run_directory / "run_oe_from_nested.py").is_file()
    assert (run_directory / "run_forward.py").is_file()
    assert (run_directory / "postprocess_retrieval.py").is_file()
    assert (run_directory / "postprocess_forward.py").is_file()
    for directory_name in ("outputs", "opacity_cache", "scratch"):
        assert (run_directory / directory_name).is_dir()
    assert (run_directory / "submit.sh").stat().st_mode & 0o111
    for filename in (
        "run_retrieval.py",
        "run_oe_from_nested.py",
        "run_forward.py",
        "postprocess_retrieval.py",
        "postprocess_forward.py",
    ):
        wrapper = (run_directory / filename).read_text(encoding="utf-8")
        assert "runpy.run_path" in wrapper
        assert str(ROOT) in wrapper
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
    assert f"#SBATCH --output={run_directory}/slurm-%x-%j.out" in submission
    assert f"#SBATCH --error={run_directory}/slurm-%x-%j.err" in submission
    assert "#SBATCH --output=%x-%j.out" not in submission
    assert "#SBATCH --error=%x-%j.err" not in submission
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


@pytest.mark.parametrize("config_path", MULTINEST_CONFIGS)
def test_create_run_directory_uses_128_ranks_for_multinest_workflows(
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


def test_create_run_directory_rejects_a_destination_inside_the_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside the ROBERT checkout"):
        create_run_directory(
            project_dir=ROOT / "examples" / "outputs",
            source_config=SOURCE_CONFIG,
        )


def _write_source_run_config(
    tmp_path: Path,
    *,
    observation_kind: str = "file",
) -> tuple[Path, Path, Path]:
    source_run = tmp_path / "source-run"
    source_output = source_run / "outputs"
    source_output.mkdir(parents=True)
    observation = source_output / "synthetic_observation.npz"
    if observation_kind == "file":
        observation.write_bytes(b"small synthetic fixture")
    elif observation_kind == "directory":
        observation.mkdir()
        (observation / "large-input-placeholder").write_bytes(b"do not copy")
    elif observation_kind != "missing":
        raise ValueError(observation_kind)

    raw = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    raw["run"]["name"] = "relocated-synthetic"
    raw["paths"] = {
        "observations_directory": str(observation),
        "output_directory": str(source_output),
        "opacity_cache_directory": str(source_run / "opacity_cache"),
        "scratch_directory": str(source_run / "scratch"),
        "project_directory": str(source_run),
    }
    source_config = tmp_path / "source.yaml"
    source_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return source_config, source_run, observation


def test_create_run_directory_relocates_a_source_run_synthetic_file(
    tmp_path: Path,
) -> None:
    source_config, source_run, source_observation = _write_source_run_config(tmp_path)
    run_directory = create_run_directory(
        project_dir=tmp_path / "new-runs",
        source_config=source_config,
    )
    config = load_task_config(run_directory / "configuration.yaml")
    relocated = run_directory / "outputs" / source_observation.name

    assert config.observations.path == relocated
    assert relocated.read_bytes() == source_observation.read_bytes()
    assert source_observation.read_bytes() == b"small synthetic fixture"
    assert str(source_run) not in (run_directory / "configuration.yaml").read_text(
        encoding="utf-8"
    )


def test_create_run_directory_points_missing_synthetic_file_at_new_run(
    tmp_path: Path,
) -> None:
    source_config, source_run, source_observation = _write_source_run_config(
        tmp_path,
        observation_kind="missing",
    )
    run_directory = create_run_directory(
        project_dir=tmp_path / "new-runs",
        source_config=source_config,
    )
    config = load_task_config(run_directory / "configuration.yaml")

    assert config.observations.path == run_directory / "outputs" / source_observation.name
    assert not config.observations.path.exists()
    assert str(source_run) not in (run_directory / "configuration.yaml").read_text(
        encoding="utf-8"
    )


def test_create_run_directory_does_not_copy_source_observation_directories(
    tmp_path: Path,
) -> None:
    source_config, _, _ = _write_source_run_config(
        tmp_path,
        observation_kind="directory",
    )

    with pytest.raises(ValueError, match="cannot copy an observation directory"):
        create_run_directory(
            project_dir=tmp_path / "new-runs",
            source_config=source_config,
        )
