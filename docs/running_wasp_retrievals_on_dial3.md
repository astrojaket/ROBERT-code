# Running ROBERT on DiRAC DIaL3

ROBERT is installed at `/scratch/dp448/dc-tayl1/ROBERT-code`, and the ExoMol
KTA root is `/scratch/dp448/dc-tayl1/ktables_exomol`. Pass
`--opacity-resolution R1000` or `R15000` to select the matching subdirectory
and filenames.

The clear native-mode workflow uses NIRCam/F322W2, NIRCam/F444W, and MIRI/LRS
as independent likelihood terms. The NIRCam overlap-average product is omitted.

## Build the conda environment once

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
conda env create -f environment.yml
conda activate robert-exoplanets
export NUMBA_CACHE_DIR=/tmp/${USER}-robert-numba
mkdir -p "$NUMBA_CACHE_DIR"
python -c "import exo_k, mpi4py, ultranest, robert_exoplanets; print('ROBERT environment OK')"
```

For an existing environment after a repository update, use:

```bash
conda env update -n robert-exoplanets -f environment.yml --prune
```

## One-CPU terminal check

Check the live DIaL3 partitions with `sinfo -s`, then request one CPU on the
`slurm` partition:

```bash
srun --account=dp448 --partition=slurm \
  --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --time=02:00:00 --mem=8G --pty bash -l
```

Inside that shell:

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
conda activate robert-exoplanets
mkdir -p /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check
```

Prepare the R1000 opacity cache for F322W2, F444W, and MIRI/LRS:

```bash
python -u /scratch/dp448/dc-tayl1/ROBERT-code/examples/retrieve_wasp69b_clear_native_modes.py \
  --kta-path /scratch/dp448/dc-tayl1/ktables_exomol \
  --opacity-resolution R1000 \
  --prepare-only
```

Write and inspect a deterministic one-CPU smoke configuration:

```bash
python -u /scratch/dp448/dc-tayl1/ROBERT-code/examples/retrieve_wasp69b_clear_native_modes.py \
  --kta-path /scratch/dp448/dc-tayl1/ktables_exomol \
  --opacity-resolution R1000 \
  --mpi-processes 1 \
  --smoke-only \
  --max-ncalls 200 \
  --output /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check

less /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/run_configuration.json
```

Start the small resumable sampler check:

```bash
python -u /scratch/dp448/dc-tayl1/ROBERT-code/examples/retrieve_wasp69b_clear_native_modes.py \
  --kta-path /scratch/dp448/dc-tayl1/ktables_exomol \
  --opacity-resolution R1000 \
  --mpi-processes 1 \
  --max-ncalls 200 \
  --output /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check
```

After UltraNest reports iterations, press `Ctrl-C` once and inspect:

```bash
cat /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/ultranest/sampler_status.json
tail -n 5 /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/ultranest/run_attempts.jsonl
exit
```

## Submit a 64-CPU run

The production scripts request 64 MPI ranks with account `dp448` on the
DIaL3 `slurm` partition and launch them with `srun`. Results default to
`/scratch/dp448/dc-tayl1/retrieval_runs`, outside the clone.

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
sbatch slurm/wasp69b_clear_native_modes.sbatch
```

Choose a project-specific output directory without editing the script:

```bash
sbatch --export=ALL,ROBERT_OUTPUT_DIR=/scratch/dp448/dc-tayl1/my_project/wasp69b_clear_R1000 \
  slurm/wasp69b_clear_native_modes.sbatch
```

Use `squeue -u "$USER"` to monitor jobs and `scancel JOB_ID` to cancel one.
Use a different output directory whenever the target, priors, resolution, or
model changes so incompatible UltraNest checkpoints are never mixed.
