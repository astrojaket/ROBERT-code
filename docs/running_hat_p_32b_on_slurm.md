# Running the HAT-P-32b Retrieval on Slurm

## `addqueue` clusters

On clusters where jobs are submitted through `addqueue` and launched with
`mpirun`, ROBERT includes a repository-root `submit.sh` matching that workflow.
The default is the validated 12-process, 40-live-point, 10,000-call test:

```bash
cd ~/ROBERT-code
addqueue -q "planet" -c "Robert-run" -n 12 ./submit.sh
```

The script directly uses
`/mnt/zfsusers/jaketaylor/anaconda3/envs/robert-exoplanets/bin/python3`, verifies
the bundled data and Python dependencies, and writes resumable output to
`retrieval_runs/hat_p_32b_cluster_test`. The process count passed to
`addqueue -n` must match `ROBERT_NPROCS`; both default to 12. For a different
run, export settings before calling `addqueue`, for example:

```bash
export ROBERT_RUN_NAME=hat_p_32b_400live
export ROBERT_NPROCS=12
export ROBERT_LIVE_POINTS=400
export ROBERT_MAX_NCALLS=500000
export ROBERT_DLOGZ=0.5
addqueue -q "planet" -c "Robert-run" -n 12 ./submit.sh
```

Set `ROBERT_PYTHON`, `ROBERT_MPIRUN`, or `ROBERT_REPO_DIR` only if the cluster
paths differ. Reusing the same run name resumes its UltraNest checkpoint;
`ROBERT_MAX_NCALLS` is the cumulative call limit.

Some `addqueue` installations launch `submit.sh` once on every allocated MPI
rank. ROBERT detects the existing MPI world and runs one Python process per
rank without starting a nested `mpirun`. If `submit.sh` is instead executed
once outside MPI, it starts `mpirun -n ROBERT_NPROCS` itself. This prevents an
`N`-core allocation from accidentally launching `N × N` retrieval processes.
Both OpenMPI/PMI rank variables and Slurm's `SLURM_NTASKS`/`SLURM_PROCID`
variables are supported. MPI temporary files use `SLURM_TMPDIR` when supplied,
then node-local `/dev/shm`, rather than a shared `/tmp` filesystem.

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
mpiexec -n 3 python examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest --run-retrieval \
  --live-points 40 --max-ncalls 500 --dlogz 1.5 \
  --resume overwrite \
  --mpi-nprocs 3 \
  --output-dir retrieval_runs/hat_p_32b_smoke
```

This deliberately bounded run is only an infrastructure test and is not
expected to converge.

For the first actual Slurm submission, use the previously validated 40-live
point, 10,000-call setup on three cores and give it a new run name:

```bash
sbatch --ntasks=3 --mem=16G --time=02:00:00 \
  --export=ALL,ROBERT_RUN_NAME=hat_p_32b_smoke_01,ROBERT_LIVE_POINTS=40,ROBERT_MAX_NCALLS=10000,ROBERT_DLOGZ=1.5 \
  slurm/hat_p_32b_ultranest.sbatch
```

This writes to `retrieval_runs/hat_p_32b_smoke_01`. Use a different run name
if that directory already contains a previous smoke test.

## Submit the production job

Review the account, partition, memory, wall time, and task count directives in
`slurm/hat_p_32b_ultranest.sbatch`, then submit:

```bash
sbatch slurm/hat_p_32b_ultranest.sbatch
```

Defaults are 32 MPI tasks, 400 live points, 500,000 likelihood calls, 100
layers, `dlogz=0.5`, and resumable output under
`retrieval_runs/hat_p_32b_production`. Override them without editing the script:

```bash
ROBERT_MAX_NCALLS=1000000 ROBERT_LIVE_POINTS=600 \
ROBERT_RUN_NAME=hat_p_32b_long \
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

## Resume and monitor a run

UltraNest's `--max-ncalls` is a cumulative limit for the run directory, not an
additional allowance for each Slurm submission. To extend a run that stopped at
500,000 calls, submit the same output directory with a higher limit:

```bash
ROBERT_MAX_NCALLS=1000000 ROBERT_RUN_NAME=hat_p_32b_production \
sbatch slurm/hat_p_32b_ultranest.sbatch
```

The script asks Slurm for a three-minute pre-emption warning and automatically
requeues the job. UltraNest periodically flushes its HDF5 point store, so a
hard node loss can discard the last unflushed batch but not the whole run.
Every submission records a separate attempt manifest and event journal while
preserving the original scientific manifest.

Inspect progress from the login node at any time:

```bash
robert-retrieval-status retrieval_runs/hat_p_32b_production
robert-retrieval-status retrieval_runs/hat_p_32b_production --json
```

Never submit two jobs to the same output directory simultaneously. ROBERT uses
an operating-system lock to reject the second job before it can write to the
UltraNest checkpoint. Use `ROBERT_RESUME=overwrite` only when intentionally
starting that directory again from scratch; normally leave it as `resume`.

## Manual command

The equivalent command for a custom allocation is:

```bash
mpiexec -n 32 python examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest --run-retrieval \
  --live-points 400 --max-ncalls 500000 --dlogz 0.5 \
  --resume resume \
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
