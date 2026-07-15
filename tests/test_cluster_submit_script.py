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
    assert "CONDA_DIR" in text
    assert "/mnt/zfsusers/jaketaylor/anaconda3" in text
    assert "robert-exoplanets" in text
    assert "RUN_DIR" in text
    assert "configuration.yaml" in text
    assert "${HOME}/MultiNestNew/lib:${LD_LIBRARY_PATH:-}" in text
    assert "mpirun" not in text
    assert "sha256sum" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_addqueue_submit_script_executes_one_python_process_per_rank(
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
    conda_root = tmp_path / "anaconda3"
    conda_script = conda_root / "etc" / "profile.d" / "conda.sh"
    conda_script.parent.mkdir(parents=True)
    conda_script.write_text("conda() { :; }\n", encoding="utf-8")
    environment = {
        **os.environ,
        "SLURM_NTASKS": "12",
        "SLURM_PROCID": "1",
        "ROBERT_CONDA_ROOT": str(conda_root),
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
