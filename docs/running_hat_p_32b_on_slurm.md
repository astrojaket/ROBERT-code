# Running the HAT-P-32b Retrieval on Slurm

## Install

Clone ROBERT and create the pinned Conda environment from the repository root:

```bash
git clone git@github.com:astrojaket/ROBERT-code.git
cd ROBERT-code
conda env create --file environment.yml
conda activate robert-exoplanets
```

The environment includes NumPy, Numba, exo_k, FastChem, UltraNest, Open MPI,
mpi4py, plotting, notebooks, and development/test dependencies. The repository
contains the HAT-P-32b observation, comparison posterior, six exo_k-binned
opacity tables, FastChem inputs, and CIA table.

Verify the installation before submitting a long job:

```bash
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/robert-numba"
python -m pytest -q
python examples/compare_hat_p_32b_fastchem_retrieval.py
```

The second command should report a reference-MAP ROBERT chi-square near
`140.81985` and an RMS difference from the NemesisPy MAP spectrum near
`18.78878 ppm`.

## Interactive MPI smoke test

Within an allocated node, test MPI before starting a large retrieval:

```bash
mpiexec -n 4 python examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest --run-retrieval \
  --live-points 40 --max-ncalls 500 --dlogz 1.5 \
  --mpi-nprocs 4 \
  --output-dir retrieval_runs/hat_p_32b_smoke
```

This deliberately bounded run is only an infrastructure test and is not
expected to converge.

## Submit the production job

Review the account, partition, memory, wall time, and task count directives in
`slurm/hat_p_32b_ultranest.sbatch`, then submit:

```bash
sbatch slurm/hat_p_32b_ultranest.sbatch
```

Defaults are 32 MPI tasks, 400 live points, 500,000 likelihood calls, 100
layers, and `dlogz=0.5`. Override them without editing the script:

```bash
ROBERT_MAX_NCALLS=1000000 ROBERT_LIVE_POINTS=600 \
ROBERT_OUTPUT_DIR="$PWD/retrieval_runs/hat_p_32b_long" \
sbatch slurm/hat_p_32b_ultranest.sbatch
```

The script uses `srun` by default. On clusters whose MPI stack requires
`mpiexec` inside a Slurm allocation, submit with:

```bash
ROBERT_MPI_LAUNCHER=mpiexec sbatch slurm/hat_p_32b_ultranest.sbatch
```

Do not interpret the evidence or posterior unless `result.json` reports
`"converged": true`. UltraNest can exceed `--max-ncalls` slightly while
finishing an in-flight parallel batch.

## Manual command

The equivalent command for a custom allocation is:

```bash
mpiexec -n 32 python examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest --run-retrieval \
  --live-points 400 --max-ncalls 500000 --dlogz 0.5 \
  --mpi-nprocs 32 \
  --pressure-top-bar 1e-6 --pressure-bottom-bar 100 --layers 100 \
  --output-dir retrieval_runs/hat_p_32b_production
```

After completion, generate all comparison plots with:

```bash
python examples/plot_hat_p_32b_fastchem_comparison.py \
  --robert-result-dir retrieval_runs/hat_p_32b_production \
  --output-dir retrieval_runs/hat_p_32b_production/plots
```
