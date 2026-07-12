#!/bin/bash -l
#SBATCH --job-name=wasp80b-native-clear
#SBATCH --account=CHANGE_ME
#SBATCH --partition=slurm
#SBATCH --nodes=1
#SBATCH --ntasks=3
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
  examples/retrieve_wasp80b_clear_native_modes.py \
  --output retrieval_runs/wasp80b_clear_native_modes_optimized_priors_v2
