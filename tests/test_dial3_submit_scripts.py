"""Static checks for the DIaL3 WASP submission scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "wasp69b_nircam_clear.sbatch": "retrieve_wasp69b_nircam_clear.py",
    "wasp69b_clear_native_modes.sbatch": "retrieve_wasp69b_clear_native_modes.py",
    "wasp69b_mie_catalog.sbatch": "--cloud-mode catalog",
    "wasp69b_mie_direct_nk.sbatch": "--cloud-mode direct-nk",
    "wasp80b_nircam_clear.sbatch": "retrieve_wasp80b_nircam_clear.py",
    "wasp80b_clear_native_modes.sbatch": "retrieve_wasp80b_clear_native_modes.py",
    "wasp80b_mie_catalog.sbatch": "--cloud-mode catalog",
    "wasp80b_mie_direct_nk.sbatch": "--cloud-mode direct-nk",
}


def test_dial3_scripts_are_valid_bash_and_use_expected_cases() -> None:
    for name, case_marker in SCRIPTS.items():
        path = ROOT / "slurm" / name
        text = path.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert "#SBATCH --account=dp448" in text
        assert "#SBATCH --partition=slurm" in text
        assert "#SBATCH --ntasks=64" in text
        assert case_marker in text
        assert "--kta-path /scratch/dp448/dc-tayl1/ktables_exomol" in text
        assert "--opacity-resolution R1000" in text
        assert "--mpi-processes \"${SLURM_NTASKS}\"" in text
        assert "/scratch/dp448/dc-tayl1/retrieval_runs/" in text
        assert 'source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/etc/profile.d/conda.sh"' in text
        assert 'srun --ntasks="${SLURM_NTASKS}" python -u' in text


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
