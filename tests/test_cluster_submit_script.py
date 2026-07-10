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
    assert "SLURM_NTASKS" in text
    assert "mpirun" not in text
    assert "sha256sum" not in text
    assert "--resume" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_addqueue_submit_script_executes_one_python_process_per_rank(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SLURM_NTASKS": "12",
        "SLURM_PROCID": "1",
        "ROBERT_OUTPUT_DIR": str(tmp_path / "output"),
        "ROBERT_PYTHON": "/usr/bin/true",
    }

    completed = subprocess.run(
        [str(repository / "submit.sh")],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
