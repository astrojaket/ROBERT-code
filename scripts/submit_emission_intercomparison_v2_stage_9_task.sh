#!/usr/bin/env bash
# Single-wrapper addqueue entry point for approved Stage-9 setup and pilots.
set -euo pipefail

for required in STAGE9_TASK STAGE9_FRAMEWORK STAGE9_PROJECT_ROOT STAGE9_REPOSITORY STAGE9_ENVIRONMENT_PARENT STAGE9_PICASO_REFDATA STAGE9_PICASO_CK_DIRECTORY STAGE9_PRT_INPUT_DATA; do
  if [[ -z "${!required:-}" ]]; then
    echo "$required must be exported before addqueue submission" >&2
    exit 2
  fi
done

case "$STAGE9_FRAMEWORK" in
  robert) environment_prefix="$STAGE9_ENVIRONMENT_PARENT/robert-stage9" ;;
  picaso) environment_prefix="$STAGE9_ENVIRONMENT_PARENT/robert-stage9-picaso-v4" ;;
  petitradtrans) environment_prefix="$STAGE9_ENVIRONMENT_PARENT/robert-stage9-petitradtrans" ;;
  *) echo "unsupported Stage-9 framework: $STAGE9_FRAMEWORK" >&2; exit 2 ;;
esac

export STAGE9_CLUSTER=glamdring
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="$STAGE9_PROJECT_ROOT/cache/picaso/matplotlib"
export NUMBA_CACHE_DIR="$STAGE9_PROJECT_ROOT/cache/picaso/numba"
export picaso_refdata="$STAGE9_PICASO_REFDATA"

python_executable="$environment_prefix/bin/python"
mpi_launcher="$STAGE9_REPOSITORY/scripts/launch_emission_intercomparison_v2_stage_9_mpi.sh"
case "$STAGE9_TASK" in
  preflight)
    exec "$mpi_launcher" "$environment_prefix" 12 "$python_executable" \
      "$STAGE9_REPOSITORY/scripts/preflight_emission_intercomparison_v2_stage_9_glamdring.py" \
      "$STAGE9_FRAMEWORK" "$STAGE9_PROJECT_ROOT"
    ;;
  injection)
    : "${STAGE9_SCENARIO:?STAGE9_SCENARIO is required for injection}"
    exec "$python_executable" \
      "$STAGE9_REPOSITORY/examples/generate_emission_intercomparison_v2_stage_9_injection.py" \
      "$STAGE9_FRAMEWORK" "$STAGE9_SCENARIO" "$STAGE9_PROJECT_ROOT" --approved
    ;;
  forward-pilot)
    : "${STAGE9_SCENARIO:?STAGE9_SCENARIO is required for forward-pilot}"
    : "${STAGE9_PILOT_OUTPUT:?STAGE9_PILOT_OUTPUT is required for forward-pilot}"
    exec "$mpi_launcher" "$environment_prefix" 12 "$python_executable" \
      "$STAGE9_REPOSITORY/examples/pilot_emission_intercomparison_v2_stage_9_forward.py" \
      "$STAGE9_FRAMEWORK" "$STAGE9_SCENARIO" "$STAGE9_PROJECT_ROOT" \
      "$STAGE9_PILOT_OUTPUT" --approved --evaluations "${STAGE9_PILOT_EVALUATIONS:-2}"
    ;;
  retrieval-pilot)
    : "${STAGE9_RUN_CONFIG:?STAGE9_RUN_CONFIG is required for retrieval-pilot}"
    : "${STAGE9_PILOT_OUTPUT:?STAGE9_PILOT_OUTPUT is required for retrieval-pilot}"
    exec "$mpi_launcher" "$environment_prefix" 12 "$python_executable" \
      "$STAGE9_REPOSITORY/examples/run_emission_intercomparison_v2_stage_9_retrieval.py" \
      "$STAGE9_RUN_CONFIG" --pilot-output "$STAGE9_PILOT_OUTPUT" \
      --pilot-max-iter "${STAGE9_PILOT_MAX_ITER:-200}" \
      --pilot-live-points "${STAGE9_PILOT_LIVE_POINTS:-50}"
    ;;
  *)
    echo "STAGE9_TASK must be preflight, injection, forward-pilot, or retrieval-pilot" >&2
    exit 2
    ;;
esac
