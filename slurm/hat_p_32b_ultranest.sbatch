#!/usr/bin/env bash
#SBATCH --job-name=robert-hatp32b
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f "examples/compare_hat_p_32b_fastchem_retrieval.py" ]]; then
  echo "Submit this script from the ROBERT repository root." >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ROBERT_CONDA_ENV:-robert-exoplanets}"

if [[ "${ROBERT_VERIFY_DATA:-1}" == "1" ]]; then
  (cd examples/data/hat_p_32b && sha256sum --check checksums.sha256)
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/robert-numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/robert-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

live_points="${ROBERT_LIVE_POINTS:-400}"
max_ncalls="${ROBERT_MAX_NCALLS:-500000}"
dlogz="${ROBERT_DLOGZ:-0.5}"
layers="${ROBERT_LAYERS:-100}"
resume="${ROBERT_RESUME:-resume}"
run_name="${ROBERT_RUN_NAME:-hat_p_32b_production}"
output_dir="${ROBERT_OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/retrieval_runs/${run_name}}"
launcher="${ROBERT_MPI_LAUNCHER:-srun}"
launcher_pid=""

handle_preemption() {
  echo "ROBERT received a pre-emption signal at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
  if [[ -n "$launcher_pid" ]]; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
  robert-retrieval-status "$output_dir" || true
  if [[ "${ROBERT_REQUEUE_ON_SIGNAL:-1}" == "1" ]]; then
    echo "Requeueing Slurm job ${SLURM_JOB_ID}."
    scontrol requeue "$SLURM_JOB_ID"
  fi
  exit 0
}

trap handle_preemption USR1

mkdir -p "$output_dir"
if [[ -f "$output_dir/sampler_status.json" ]]; then
  robert-retrieval-status "$output_dir" || true
fi
command=(
  python -u examples/compare_hat_p_32b_fastchem_retrieval.py
  --method ultranest
  --run-retrieval
  --live-points "$live_points"
  --max-ncalls "$max_ncalls"
  --dlogz "$dlogz"
  --resume "$resume"
  --layers "$layers"
  --mpi-nprocs "${SLURM_NTASKS}"
  --output-dir "$output_dir"
)

if [[ "$launcher" == "mpiexec" ]]; then
  mpiexec -n "${SLURM_NTASKS}" "${command[@]}" &
else
  srun --ntasks="${SLURM_NTASKS}" "${command[@]}" &
fi
launcher_pid=$!
set +e
wait "$launcher_pid"
retrieval_exit=$?
set -e
launcher_pid=""

robert-retrieval-status "$output_dir" || true
if [[ "$retrieval_exit" -ne 0 ]]; then
  echo "ROBERT retrieval failed with exit code ${retrieval_exit}." >&2
  exit "$retrieval_exit"
fi

if [[ -f "$output_dir/result.json" ]]; then
  python examples/plot_hat_p_32b_fastchem_comparison.py \
    --robert-result-dir "$output_dir" \
    --output-dir "$output_dir/plots"
fi
