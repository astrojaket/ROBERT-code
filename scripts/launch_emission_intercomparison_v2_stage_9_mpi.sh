#!/usr/bin/env bash
# Start one self-contained Conda MPICH world inside a Glamdring single-wrapper job.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 ENVIRONMENT_PREFIX MPI_RANKS PROGRAM [ARGUMENT ...]" >&2
  exit 2
fi

environment_prefix="$1"
mpi_ranks="$2"
shift 2

if [[ "${STAGE9_CLUSTER:-}" != "glamdring" ]]; then
  echo "STAGE9_CLUSTER=glamdring is required for the Stage-9 MPI launcher" >&2
  exit 2
fi
case "$mpi_ranks" in
  1|12) ;;
  *)
    echo "Stage-9 jobs require exactly 1 injection rank or 12 retrieval/pilot ranks" >&2
    exit 2
    ;;
esac
if [[ "${LOADEDMODULES:-}" == *openmpi* ]]; then
  echo "Unload Glamdring's OpenMPI module before using the pinned Conda MPICH stack" >&2
  exit 2
fi

mpiexec_executable="$environment_prefix/bin/mpiexec"
if [[ ! -x "$mpiexec_executable" ]]; then
  echo "Conda MPICH launcher is not executable: $mpiexec_executable" >&2
  exit 2
fi
if ! "$mpiexec_executable" -version 2>&1 | grep -qi 'HYDRA'; then
  echo "Stage-9 requires the Conda MPICH/Hydra mpiexec: $mpiexec_executable" >&2
  exit 2
fi

# addqueue -s -n 12 starts this wrapper once in a 12-core allocation.  Its
# outer Slurm step uses PMIx, whereas the pinned Conda MPICH stack uses Hydra's
# PMI implementation.  Remove the outer PMI/PMIx variables before Hydra starts
# a fresh, internally consistent MPI world.  Keep SLURM_* allocation metadata.
while IFS='=' read -r name _; do
  case "$name" in
    PMI*|PMIX*) unset "$name" ;;
  esac
done < <(env)

export HYDRA_LAUNCHER=fork
export HYDRA_RMK=user

exec "$mpiexec_executable" \
  -rmk user \
  -launcher fork \
  -n "$mpi_ranks" \
  "$@"
