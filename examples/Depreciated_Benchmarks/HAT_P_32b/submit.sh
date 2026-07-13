#!/bin/bash

export OMP_NUM_THREADS=1

exec "${ROBERT_PYTHON:-/mnt/zfsusers/jaketaylor/anaconda3/envs/robert-exoplanets/bin/python3}" \
  -u examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest \
  --run-retrieval \
  --live-points "${ROBERT_LIVE_POINTS:-40}" \
  --max-ncalls "${ROBERT_MAX_NCALLS:-10000}" \
  --dlogz "${ROBERT_DLOGZ:-1.5}" \
  --resume resume \
  --layers "${ROBERT_LAYERS:-100}" \
  --mpi-nprocs "${SLURM_NTASKS:-12}" \
  --output-dir "${ROBERT_OUTPUT_DIR:-$PWD/retrieval_runs/${ROBERT_RUN_NAME:-hat_p_32b_cluster_test}}"
