#!/usr/bin/env bash
# addqueue launches all 12 MPI ranks; do not wrap this command in mpirun/srun.
set -euo pipefail

if [[ -z "${STAGE9_RUN_CONFIG:-}" || -z "${STAGE9_REPOSITORY:-}" || -z "${STAGE9_ENVIRONMENT_PARENT:-}" ]]; then
  echo "STAGE9_RUN_CONFIG, STAGE9_REPOSITORY, and STAGE9_ENVIRONMENT_PARENT are required" >&2
  exit 2
fi

retriever="$(python -c 'import json,os; print(json.load(open(os.environ["STAGE9_RUN_CONFIG"]))["retriever"])')"
case "$retriever" in
  robert) environment_prefix="$STAGE9_ENVIRONMENT_PARENT/robert-stage9" ;;
  picaso) environment_prefix="$STAGE9_ENVIRONMENT_PARENT/robert-stage9-picaso-v4" ;;
  petitradtrans) environment_prefix="$STAGE9_ENVIRONMENT_PARENT/robert-stage9-petitradtrans" ;;
  *) echo "unsupported Stage-9 retriever: $retriever" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export STAGE9_CLUSTER=glamdring
export MPLCONFIGDIR="${STAGE9_PROJECT_ROOT:?}/cache/picaso/matplotlib"
export NUMBA_CACHE_DIR="${STAGE9_PROJECT_ROOT}/cache/picaso/numba"
export picaso_refdata="${STAGE9_PICASO_REFDATA:?}"
export STAGE9_PICASO_CK_DIRECTORY="${STAGE9_PICASO_CK_DIRECTORY:?}"
export STAGE9_PRT_INPUT_DATA="${STAGE9_PRT_INPUT_DATA:?}"

exec "$environment_prefix/bin/python" \
  "$STAGE9_REPOSITORY/examples/run_emission_intercomparison_v2_stage_9_retrieval.py" \
  "$STAGE9_RUN_CONFIG"
