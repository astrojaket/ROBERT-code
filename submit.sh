#!/bin/bash

set -euo pipefail

# addqueue launches this script once per Slurm rank. mpi4py discovers the
# resulting world through SLURM_NTASKS; do not launch a second mpiexec layer.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export ROBERT_OUTPUT_DIR="${ROBERT_OUTPUT_DIR:-}"

# UltraNest checkpoint behaviour (the historical CLI called this --resume) is
# configured by sampler.resume in the versioned YAML file.
exec "${ROBERT_PYTHON:-python}" -u run_retrieval.py \
  --config "${ROBERT_CONFIG:-configurations/wasp69b_cloud_free_R1000.yaml}"
