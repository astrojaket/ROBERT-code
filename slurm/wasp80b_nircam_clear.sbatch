#!/bin/bash -l
#SBATCH --job-name=wasp80b-nircam-clear
#SBATCH --account=dp448
#SBATCH --partition=slurm
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd /scratch/dp448/dc-tayl1/ROBERT-code
source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ROBERT_CONDA_ENV:-robert-exoplanets}"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/robert-numba-${SLURM_JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

srun --ntasks="${SLURM_NTASKS}" python -u \
  examples/retrieve_wasp80b_nircam_clear.py \
  --kta-path /scratch/dp448/dc-tayl1/ktables_exomol \
  --opacity-resolution R1000 \
  --mpi-processes "${SLURM_NTASKS}" \
  --output "${ROBERT_OUTPUT_DIR:-/scratch/dp448/dc-tayl1/retrieval_runs/wasp80b_nircam_clear}"
