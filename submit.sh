#!/bin/bash

# HAT-P-32b UltraNest retrieval for clusters using the addqueue/mpirun pattern.

set -uo pipefail

repo_dir="${ROBERT_REPO_DIR:-$PWD}"
python_executable="${ROBERT_PYTHON:-/mnt/zfsusers/jaketaylor/anaconda3/envs/robert-exoplanets/bin/python3}"
mpirun_executable="${ROBERT_MPIRUN:-mpirun}"
nprocs="${ROBERT_NPROCS:-${NSLOTS:-12}}"

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
if ! command -v "$mpirun_executable" >/dev/null 2>&1; then
  echo "MPI launcher was not found: $mpirun_executable" >&2
  echo "Set ROBERT_MPIRUN to the cluster's mpirun executable." >&2
  exit 2
fi

export OMP_NUM_THREADS="${ROBERT_THREADS_PER_PROCESS:-1}"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_CACHE_DIR="${ROBERT_NUMBA_CACHE_DIR:-$HOME/.cache/robert/numba}"
export MPLCONFIGDIR="${ROBERT_MPLCONFIGDIR:-$HOME/.cache/robert/matplotlib}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

if [[ "${ROBERT_VERIFY_DATA:-1}" == "1" ]]; then
  (cd examples/data/hat_p_32b && sha256sum --check checksums.sha256) || exit 2
fi

"$python_executable" -c \
  "import exo_k, h5py, mpi4py, pyfastchem, ultranest; import robert_exoplanets; print('ROBERT dependencies OK')" \
  || exit 2

live_points="${ROBERT_LIVE_POINTS:-40}"
max_ncalls="${ROBERT_MAX_NCALLS:-10000}"
dlogz="${ROBERT_DLOGZ:-1.5}"
layers="${ROBERT_LAYERS:-100}"
resume="${ROBERT_RESUME:-resume}"
run_name="${ROBERT_RUN_NAME:-hat_p_32b_cluster_test}"
output_dir="${ROBERT_OUTPUT_DIR:-${repo_dir}/retrieval_runs/${run_name}}"

mkdir -p "$output_dir"
if [[ -f "$output_dir/sampler_status.json" ]]; then
  "$python_executable" -m robert_exoplanets.retrieval.status "$output_dir" || true
fi

echo "Starting ROBERT with ${nprocs} MPI processes."
echo "Output directory: ${output_dir}"
"$mpirun_executable" -n "$nprocs" \
  "$python_executable" -u examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest \
  --run-retrieval \
  --live-points "$live_points" \
  --max-ncalls "$max_ncalls" \
  --dlogz "$dlogz" \
  --resume "$resume" \
  --layers "$layers" \
  --mpi-nprocs "$nprocs" \
  --output-dir "$output_dir"
retrieval_exit=$?

"$python_executable" -m robert_exoplanets.retrieval.status "$output_dir" || true
if [[ "$retrieval_exit" -ne 0 ]]; then
  echo "ROBERT retrieval failed with exit code ${retrieval_exit}." >&2
  exit "$retrieval_exit"
fi

if [[ -f "$output_dir/result.json" ]]; then
  "$python_executable" examples/plot_hat_p_32b_fastchem_comparison.py \
    --robert-result-dir "$output_dir" \
    --output-dir "$output_dir/plots"
fi
