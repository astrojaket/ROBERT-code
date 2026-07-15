#!/bin/bash

set -euo pipefail

CONDA_DIR="${ROBERT_CONDA_ROOT:-/mnt/zfsusers/jaketaylor/anaconda3}"
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate robert-exoplanets

# addqueue already launches one copy per rank; do not add mpiexec here.
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="${HOME}/MultiNestNew/lib:${LD_LIBRARY_PATH:-}"

cd "${RUN_DIR}"
exec python -u run_retrieval.py --config configuration.yaml
