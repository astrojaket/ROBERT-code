"""Checks for the addqueue-compatible cluster submission entry point."""

from __future__ import annotations

from pathlib import Path
import subprocess


def test_addqueue_submit_script_is_executable_and_syntactically_valid() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "submit.sh"
    text = script.read_text(encoding="utf-8")

    assert script.stat().st_mode & 0o111
    assert "ROBERT_PYTHON" in text
    assert 'nprocs="${ROBERT_NPROCS:-${NSLOTS:-12}}"' in text
    assert "import exo_k, h5py, mpi4py, pyfastchem, ultranest" in text
    assert "--resume" in text
    subprocess.run(["bash", "-n", str(script)], check=True)
