# Running ROBERT on DiRAC COSMA8

ROBERT is installed from `/scratch/dp448/dc-tayl1/ROBERT-code`. The ExoMol KTA
root is `/scratch/dp448/dc-tayl1/ktables_exomol`; pass `--opacity-resolution
R1000` or `R15000` to select its matching subdirectory and filenames.

The clear native-mode workflow uses NIRCam/F322W2, NIRCam/F444W, and MIRI/LRS
as independent likelihood terms. The NIRCam overlap-average product is omitted.

## Build the conda environment once

Run these commands from the repository root. If `conda` is not already on the
path, use `module avail anaconda` and load the site-provided Anaconda module, or
initialize the user's existing Miniconda installation first.

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
conda env create -f environment.yml
conda activate robert-exoplanets
export NUMBA_CACHE_DIR=/tmp/${USER}-robert-numba
mkdir -p "$NUMBA_CACHE_DIR"
python -c "import exo_k, mpi4py, ultranest, robert_exoplanets; print('ROBERT environment OK')"
```

If the environment already exists after a repository update, use:

```bash
conda env update -n robert-exoplanets -f environment.yml --prune
```

## One-CPU terminal check

Do not run the retrieval on a login node. Request one CPU in an interactive
COSMA8 allocation:

```bash
srun --account=dp448 --partition=cosma8-serial \
  --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --time=02:00:00 --mem=8G --pty bash -l
```

Inside that shell, activate ROBERT and create the external test directory:

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
conda activate robert-exoplanets

KTA_ROOT=/scratch/dp448/dc-tayl1/ktables_exomol
CHECK_DIR=/scratch/dp448/dc-tayl1/1_CPU_Terminal_Check
mkdir -p "$CHECK_DIR"
```

Prepare the R1000 tables once. ROBERT validates the requested species and
recompresses their correlated-k distributions onto the F322W2, F444W, and
MIRI/LRS observation bins:

```bash
python -u examples/retrieve_wasp69b_clear_native_modes.py \
  --kta-path "$KTA_ROOT" \
  --opacity-resolution R1000 \
  --prepare-only
```

Next run only the deterministic likelihood smoke check. This writes the full
configuration without starting UltraNest:

```bash
python -u examples/retrieve_wasp69b_clear_native_modes.py \
  --kta-path "$KTA_ROOT" \
  --opacity-resolution R1000 \
  --mpi-processes 1 \
  --smoke-only \
  --max-ncalls 200 \
  --output "$CHECK_DIR"
```

Inspect the generated configuration and verify that the datasets are
`f322w2`, `f444w`, and `lrs` and that the smoke likelihood is finite:

```bash
less "$CHECK_DIR/run_configuration.json"
less environment.yml
less slurm/wasp69b_clear_native_modes.sbatch
```

Finally start the small, resumable sampler check:

```bash
python -u examples/retrieve_wasp69b_clear_native_modes.py \
  --kta-path "$KTA_ROOT" \
  --opacity-resolution R1000 \
  --mpi-processes 1 \
  --max-ncalls 200 \
  --output "$CHECK_DIR"
```

After UltraNest begins reporting iterations, press `Ctrl-C` once and allow
ROBERT to write its interrupted status. Inspect the checkpoint state with:

```bash
cat "$CHECK_DIR/ultranest/sampler_status.json"
tail -n 5 "$CHECK_DIR/ultranest/run_attempts.jsonl"
exit
```

Reusing the same output directory resumes the checkpoint. Use a new directory
for a scientifically different target, prior set, opacity resolution, or model.

## Submit a 64-CPU run

The production scripts request 64 MPI ranks on one shared COSMA8 node using the
`dp448` account and `cosma8-serial` partition. Their default inference outputs
are under `/scratch/dp448/dc-tayl1/retrieval_runs`, outside the clone.

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
sbatch slurm/wasp69b_clear_native_modes.sbatch
```

Choose a project-specific output directory without editing the script:

```bash
sbatch --export=ALL,ROBERT_OUTPUT_DIR=/scratch/dp448/dc-tayl1/my_project/wasp69b_clear_R1000 \
  slurm/wasp69b_clear_native_modes.sbatch
```

Use `squeue -u "$USER"` to monitor the job and `scancel JOB_ID` to cancel it.
Switch both the command-line resolution and output-directory name before an
R15000 run so R1000 and R15000 checkpoints can never be mixed.
