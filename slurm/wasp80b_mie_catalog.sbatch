#!/bin/bash -l
#SBATCH --job-name=wasp80b-mie-catalog
#SBATCH --account=CHANGE_ME
#SBATCH --partition=slurm
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/bin/activate" \
  "${ROBERT_CONDA_ENV:-robert-exoplanets}"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

srun --ntasks="${SLURM_NTASKS}" python -u \
  examples/retrieve_wasp80b_mie_cloud.py \
  --cloud-mode catalog \
  --material MgSiO3 \
  --mpi-processes "${SLURM_NTASKS}" \
  --output retrieval_runs/wasp80b_mie_catalog
