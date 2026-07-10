"""Checks for the addqueue-compatible cluster submission entry point."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess


def test_addqueue_submit_script_is_executable_and_syntactically_valid() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "submit.sh"
    text = script.read_text(encoding="utf-8")

    assert script.stat().st_mode & 0o111
    assert "ROBERT_PYTHON" in text
    assert "OMPI_COMM_WORLD_SIZE" in text
    assert "scheduler_mpi" in text
    assert "does not match the addqueue MPI world" in text
    assert '"${retrieval_command[@]}"' in text
    assert "--resume" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_addqueue_submit_script_reuses_existing_mpi_world(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "OMPI_COMM_WORLD_SIZE": "12",
        "OMPI_COMM_WORLD_RANK": "1",
        "ROBERT_NPROCS": "12",
        "ROBERT_MPIRUN": "/path/that/must/not/be/called",
        "ROBERT_OUTPUT_DIR": str(tmp_path / "output"),
        "ROBERT_PYTHON": "/usr/bin/true",
        "ROBERT_REPO_DIR": str(repository),
        "ROBERT_VERIFY_DATA": "0",
    }

    completed = subprocess.run(
        [str(repository / "submit.sh")],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "must/not/be/called" not in completed.stderr


def test_addqueue_submit_script_rejects_process_count_mismatch(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "OMPI_COMM_WORLD_SIZE": "12",
        "OMPI_COMM_WORLD_RANK": "1",
        "ROBERT_NPROCS": "3",
        "ROBERT_OUTPUT_DIR": str(tmp_path / "output"),
        "ROBERT_PYTHON": "/usr/bin/true",
        "ROBERT_REPO_DIR": str(repository),
        "ROBERT_VERIFY_DATA": "0",
    }

    completed = subprocess.run(
        [str(repository / "submit.sh")],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 2
    assert "does not match the addqueue MPI world" in completed.stderr
