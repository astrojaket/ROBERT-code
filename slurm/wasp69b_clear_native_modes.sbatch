#!/bin/bash -l
#SBATCH --job-name=wasp69b-clear
#SBATCH --account=dp448
#SBATCH --partition=slurm
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "${ROBERT_CODE_DIR:-${SLURM_SUBMIT_DIR:?submit from the ROBERT-code directory or set ROBERT_CODE_DIR}}"
source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ROBERT_CONDA_ENV:-robert-exoplanets}"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

mpirun -np "${SLURM_NTASKS}" python -u run_retrieval.py \
  --config "${ROBERT_CONFIG:-configurations/wasp69b_clear_R1000.yaml}"
