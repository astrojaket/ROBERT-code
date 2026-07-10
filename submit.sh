#!/bin/bash

# HAT-P-32b UltraNest retrieval for clusters using the addqueue/mpirun pattern.

set -uo pipefail

repo_dir="${ROBERT_REPO_DIR:-$PWD}"
python_executable="${ROBERT_PYTHON:-/mnt/zfsusers/jaketaylor/anaconda3/envs/robert-exoplanets/bin/python3}"
mpirun_executable="${ROBERT_MPIRUN:-mpirun}"
mpi_world_size=1
mpi_world_rank=0
if [[ "${OMPI_COMM_WORLD_SIZE:-}" =~ ^[0-9]+$ ]] \
  && (( OMPI_COMM_WORLD_SIZE > 1 )); then
  mpi_world_size="$OMPI_COMM_WORLD_SIZE"
  mpi_world_rank="${OMPI_COMM_WORLD_RANK:-0}"
elif [[ "${PMI_SIZE:-}" =~ ^[0-9]+$ ]] && (( PMI_SIZE > 1 )); then
  mpi_world_size="$PMI_SIZE"
  mpi_world_rank="${PMI_RANK:-0}"
elif [[ "${PMIX_SIZE:-}" =~ ^[0-9]+$ ]] && (( PMIX_SIZE > 1 )); then
  mpi_world_size="$PMIX_SIZE"
  mpi_world_rank="${PMIX_RANK:-0}"
elif [[ -n "${SLURM_PROCID:-}" && "${SLURM_NTASKS:-}" =~ ^[0-9]+$ ]] \
  && (( SLURM_NTASKS > 1 )); then
  mpi_world_size="$SLURM_NTASKS"
  mpi_world_rank="$SLURM_PROCID"
fi
scheduler_mpi=0
if (( mpi_world_size > 1 )); then
  scheduler_mpi=1
fi
if (( scheduler_mpi )); then
  nprocs="${ROBERT_NPROCS:-$mpi_world_size}"
else
  nprocs="${ROBERT_NPROCS:-${NSLOTS:-12}}"
fi
is_leader=0
if [[ "$mpi_world_rank" == "0" ]]; then
  is_leader=1
fi

cd "$repo_dir" || exit 2
if [[ ! -f "examples/compare_hat_p_32b_fastchem_retrieval.py" ]]; then
  echo "Run addqueue from the ROBERT repository root or set ROBERT_REPO_DIR." >&2
  exit 2
fi
if [[ ! -x "$python_executable" ]]; then
  echo "ROBERT Python is not executable: $python_executable" >&2
  echo "Set ROBERT_PYTHON to the Python in your robert-exoplanets environment." >&2
  exit 2
fi
if (( ! scheduler_mpi )) && ! command -v "$mpirun_executable" >/dev/null 2>&1; then
  echo "MPI launcher was not found: $mpirun_executable" >&2
  echo "Set ROBERT_MPIRUN to the cluster's mpirun executable." >&2
  exit 2
fi
if (( scheduler_mpi )) && [[ "$nprocs" != "$mpi_world_size" ]]; then
  echo "ROBERT_NPROCS=${nprocs} does not match the addqueue MPI world (${mpi_world_size})." >&2
  exit 2
fi

export OMP_NUM_THREADS="${ROBERT_THREADS_PER_PROCESS:-1}"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_CACHE_DIR="${ROBERT_NUMBA_CACHE_DIR:-$HOME/.cache/robert/numba}"
export MPLCONFIGDIR="${ROBERT_MPLCONFIGDIR:-$HOME/.cache/robert/matplotlib}"
if [[ -n "${ROBERT_TMPDIR:-}" ]]; then
  export TMPDIR="$ROBERT_TMPDIR"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="$SLURM_TMPDIR"
elif [[ -d /dev/shm && -w /dev/shm ]]; then
  export TMPDIR="/dev/shm/robert-${SLURM_JOB_ID:-${USER:-job}}"
else
  export TMPDIR="/tmp/robert-${SLURM_JOB_ID:-${USER:-job}}"
fi
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"

if (( is_leader )) && [[ "${ROBERT_VERIFY_DATA:-1}" == "1" ]]; then
  (cd examples/data/hat_p_32b && sha256sum --check checksums.sha256) || exit 2
fi

if (( is_leader )); then
  if (( scheduler_mpi )); then
    "$python_executable" -c \
      "import importlib.util; names=('exo_k','h5py','mpi4py','pyfastchem','ultranest','robert_exoplanets'); missing=[name for name in names if importlib.util.find_spec(name) is None]; assert not missing, f'Missing dependencies: {missing}'; print('ROBERT dependencies found')" \
      || exit 2
  else
    "$python_executable" -c \
      "import exo_k, h5py, mpi4py, pyfastchem, ultranest; import robert_exoplanets; print('ROBERT dependencies OK')" \
      || exit 2
  fi
fi

live_points="${ROBERT_LIVE_POINTS:-40}"
max_ncalls="${ROBERT_MAX_NCALLS:-10000}"
dlogz="${ROBERT_DLOGZ:-1.5}"
layers="${ROBERT_LAYERS:-100}"
resume="${ROBERT_RESUME:-resume}"
run_name="${ROBERT_RUN_NAME:-hat_p_32b_cluster_test}"
output_dir="${ROBERT_OUTPUT_DIR:-${repo_dir}/retrieval_runs/${run_name}}"

mkdir -p "$output_dir"
show_status() {
  "$python_executable" -c \
    "import sys; from robert_exoplanets.retrieval.status import main; raise SystemExit(main(sys.argv[1:]))" \
    "$output_dir"
}
if (( is_leader )) && [[ -f "$output_dir/sampler_status.json" ]]; then
  show_status || true
fi

if (( is_leader )); then
  if (( scheduler_mpi )); then
    echo "Using the ${mpi_world_size}-process MPI world supplied by addqueue."
  else
    echo "Starting a new ${nprocs}-process MPI world with ${mpirun_executable}."
  fi
  echo "Output directory: ${output_dir}"
fi
retrieval_command=(
  "$python_executable" -u examples/compare_hat_p_32b_fastchem_retrieval.py
  --method ultranest \
  --run-retrieval \
  --live-points "$live_points" \
  --max-ncalls "$max_ncalls" \
  --dlogz "$dlogz" \
  --resume "$resume" \
  --layers "$layers" \
  --mpi-nprocs "$nprocs" \
  --output-dir "$output_dir"
)
if (( scheduler_mpi )); then
  "${retrieval_command[@]}"
else
  "$mpirun_executable" -n "$nprocs" "${retrieval_command[@]}"
fi
retrieval_exit=$?

if (( is_leader )); then
  show_status || true
fi
if [[ "$retrieval_exit" -ne 0 ]]; then
  echo "ROBERT retrieval failed with exit code ${retrieval_exit}." >&2
  exit "$retrieval_exit"
fi

if (( is_leader )) && [[ -f "$output_dir/result.json" ]]; then
  "$python_executable" examples/plot_hat_p_32b_fastchem_comparison.py \
    --robert-result-dir "$output_dir" \
    --output-dir "$output_dir/plots"
fi
