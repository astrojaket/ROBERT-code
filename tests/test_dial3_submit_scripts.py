"""Static checks for the DIaL3 WASP-69b submission scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "wasp69b_nircam_clear.sbatch": (2, "retrieve_wasp69b_nircam_clear.py"),
    "wasp69b_clear_native_modes.sbatch": (
        3,
        "retrieve_wasp69b_clear_native_modes.py",
    ),
    "wasp69b_mie_catalog.sbatch": (4, "--cloud-mode catalog"),
    "wasp69b_mie_direct_nk.sbatch": (4, "--cloud-mode direct-nk"),
    "wasp80b_nircam_clear.sbatch": (2, "retrieve_wasp80b_nircam_clear.py"),
    "wasp80b_clear_native_modes.sbatch": (
        3,
        "retrieve_wasp80b_clear_native_modes.py",
    ),
    "wasp80b_mie_catalog.sbatch": (4, "--cloud-mode catalog"),
    "wasp80b_mie_direct_nk.sbatch": (4, "--cloud-mode direct-nk"),
}


def test_dial3_scripts_are_valid_bash_and_use_expected_cases() -> None:
    for name, (tasks, case_marker) in SCRIPTS.items():
        path = ROOT / "slurm" / name
        text = path.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert "#SBATCH --account=CHANGE_ME" in text
        assert "#SBATCH --partition=slurm" in text
        assert f"#SBATCH --ntasks={tasks}" in text
        assert case_marker in text
        assert 'source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/bin/activate"' in text
        assert 'srun --ntasks="${SLURM_NTASKS}"' in text


def test_wasp_entry_points_reach_their_command_line_parser() -> None:
    entry_points = (
        "retrieve_wasp69b_nircam_clear.py",
        "retrieve_wasp69b_clear_native_modes.py",
        "retrieve_wasp69b_mie_cloud.py",
        "retrieve_wasp80b_nircam_clear.py",
        "retrieve_wasp80b_clear_native_modes.py",
        "retrieve_wasp80b_mie_cloud.py",
    )
    for name in entry_points:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "examples" / name), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
