#!/bin/bash

# Oxford Physics Glamdring launcher. Submit exactly one copy of this wrapper:
# addqueue -s -n 1x12 ... ./submit.sh
set -euo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${ROBERT_CONDA_ROOT:-${HOME}/anaconda3}"
CONDA_ENV="${ROBERT_CONDA_ENV:-robert-exoplanets}"
ENV_PREFIX="${ROBERT_CONDA_PREFIX:-${CONDA_ROOT}/envs/${CONDA_ENV}}"
PYTHON="${ENV_PREFIX}/bin/python"
MPIEXEC="${ENV_PREFIX}/bin/mpiexec"
MPI_RANKS="${ROBERT_MPI_RANKS:-${SLURM_CPUS_ON_NODE:-1}}"

if [[ ! "${MPI_RANKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROBERT_MPI_RANKS must be a positive integer, received: ${MPI_RANKS}" >&2
    exit 2
fi
if [[ "${LOADEDMODULES:-}" == *openmpi* ]]; then
    echo "OpenMPI is loaded. Run 'module unload openmpi' before addqueue." >&2
    exit 2
fi
if [[ ! -x "${PYTHON}" || ! -x "${MPIEXEC}" ]]; then
    echo "Missing Python or mpiexec in Conda environment: ${ENV_PREFIX}" >&2
    exit 2
fi
if ! "${MPIEXEC}" -version 2>&1 | grep -qi HYDRA; then
    echo "ROBERT requires the Conda MPICH/Hydra mpiexec on Glamdring." >&2
    exit 2
fi

# addqueue's outer Slurm step supplies PMIx variables. They must not leak into
# the independent Conda MPICH/Hydra world started inside this one wrapper.
while IFS="=" read -r name _; do
    case "${name}" in
        PMI*|PMIX*) unset "${name}" ;;
    esac
done < <(env)

export HYDRA_LAUNCHER=fork
export HYDRA_RMK=user
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

CACHE_ROOT="${SLURM_TMPDIR:-${RUN_DIR}/scratch}"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba-${SLURM_JOB_ID:-local}"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

MULTINEST_LIB="${ROBERT_MULTINEST_LIB:-${HOME}/MultiNestNew/lib}"
if [[ -d "${MULTINEST_LIB}" ]]; then
    export LD_LIBRARY_PATH="${MULTINEST_LIB}:${LD_LIBRARY_PATH:-}"
fi

cd "${RUN_DIR}"
exec "${MPIEXEC}" -rmk user -launcher fork -n "${MPI_RANKS}" \
    "${PYTHON}" -u run_retrieval.py --config configuration.yaml
