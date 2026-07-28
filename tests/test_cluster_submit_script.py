"""Checks for the addqueue-compatible cluster submission entry point."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def test_addqueue_submit_script_is_executable_and_syntactically_valid() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "submit.sh"
    text = script.read_text(encoding="utf-8")

    assert script.stat().st_mode & 0o111
    assert "ENV_PREFIX" in text
    assert '${ROBERT_CONDA_ROOT:-${HOME}/anaconda3}' in text
    assert "robert-exoplanets" in text
    assert "RUN_DIR" in text
    assert "configuration.yaml" in text
    assert "ROBERT_MULTINEST_LIB" in text
    assert '-rmk user -launcher fork -n "${MPI_RANKS}"' in text
    assert "PMI*|PMIX*" in text
    assert "HYDRA" in text
    assert "sha256sum" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_addqueue_submit_script_starts_one_hydra_world(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    run_directory = tmp_path / "retrieval"
    run_directory.mkdir()
    shutil.copy2(repository / "submit.sh", run_directory / "submit.sh")
    (run_directory / "configuration.yaml").write_text(
        "schema_version: 2\n", encoding="utf-8"
    )
    (run_directory / "run_retrieval.py").write_text("", encoding="utf-8")
    environment_prefix = tmp_path / "anaconda3" / "envs" / "robert-exoplanets"
    bin_directory = environment_prefix / "bin"
    bin_directory.mkdir(parents=True)
    python = bin_directory / "python"
    python.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    mpiexec = bin_directory / "mpiexec"
    mpiexec.write_text(
        "#!/bin/bash\n"
        "if [[ \"${1:-}\" == \"-version\" ]]; then echo 'HYDRA build'; exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"${ROBERT_TEST_MPI_ARGS}\"\n",
        encoding="utf-8",
    )
    mpiexec.chmod(0o755)
    mpi_args = tmp_path / "mpi-args.txt"
    environment = {
        **os.environ,
        "SLURM_CPUS_ON_NODE": "12",
        "PMIX_RANK": "0",
        "ROBERT_CONDA_PREFIX": str(environment_prefix),
        "ROBERT_TEST_MPI_ARGS": str(mpi_args),
    }

    completed = subprocess.run(
        [str(run_directory / "submit.sh")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert mpi_args.read_text(encoding="utf-8").splitlines() == [
        "-rmk",
        "user",
        "-launcher",
        "fork",
        "-n",
        "12",
        str(python),
        "-u",
        "run_retrieval.py",
        "--config",
        "configuration.yaml",
    ]
