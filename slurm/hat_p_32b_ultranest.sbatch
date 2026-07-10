#!/usr/bin/env bash
#SBATCH --job-name=robert-hatp32b
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robert-exoplanets

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/robert-numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/robert-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

live_points="${ROBERT_LIVE_POINTS:-400}"
max_ncalls="${ROBERT_MAX_NCALLS:-500000}"
dlogz="${ROBERT_DLOGZ:-0.5}"
layers="${ROBERT_LAYERS:-100}"
output_dir="${ROBERT_OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/retrieval_runs/hat_p_32b_${SLURM_JOB_ID}}"
launcher="${ROBERT_MPI_LAUNCHER:-srun}"

mkdir -p "$output_dir"
command=(
  python -u examples/compare_hat_p_32b_fastchem_retrieval.py
  --method ultranest
  --run-retrieval
  --live-points "$live_points"
  --max-ncalls "$max_ncalls"
  --dlogz "$dlogz"
  --layers "$layers"
  --mpi-nprocs "${SLURM_NTASKS}"
  --output-dir "$output_dir"
)

if [[ "$launcher" == "mpiexec" ]]; then
  mpiexec -n "${SLURM_NTASKS}" "${command[@]}"
else
  srun --ntasks="${SLURM_NTASKS}" "${command[@]}"
fi

if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
  python examples/plot_hat_p_32b_fastchem_comparison.py \
    --robert-result-dir "$output_dir" \
    --output-dir "$output_dir/plots"
fi
