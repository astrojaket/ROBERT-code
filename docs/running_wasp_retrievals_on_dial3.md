# Running ROBERT on DiRAC DIaL3

ROBERT is installed at `/scratch/dp448/dc-tayl1/ROBERT-code`. Machine, data,
opacity, cache, scratch, and output paths are selected in a copied YAML file;
the Python and Slurm runners contain no science paths. Set `resolution` to
`R1000` or `R15000` in YAML to select matching KTA directories and filenames.

The clear native-mode workflow uses NIRCam/F322W2, NIRCam/F444W, and MIRI/LRS
as independent likelihood terms. The NIRCam overlap-average product is omitted.

## Build the conda environment once

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
conda env create -f environment.yml
conda activate robert-exoplanets
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

Inside that shell, copy the example to the terminal-check project and edit all
paths and science choices there. In particular, set `outputs.directory` to
`/scratch/dp448/dc-tayl1/1_CPU_Terminal_Check` and give the run its own
`runtime.scratch_directory`:

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
source "${ROBERT_CONDA_ROOT:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
conda activate robert-exoplanets
mkdir -p /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check
cp configurations/wasp69b_clear_R1000.yaml \
  /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/configuration.yaml
nano /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/configuration.yaml
```

Validate the YAML and create all configured directories:

```bash
python -u run_retrieval.py \
  --config /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/configuration.yaml \
  --validate-only
python -u run_retrieval.py \
  --config /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/configuration.yaml \
  --initialize
```

Prepare the R1000 opacity cache for F322W2, F444W, and MIRI/LRS once, on one
process:

```bash
python -u run_retrieval.py \
  --config /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/configuration.yaml \
  --prepare-opacity
```

Run and inspect one deterministic prior-midpoint likelihood evaluation:

```bash
python -u run_retrieval.py \
  --config /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/configuration.yaml \
  --smoke-only

less /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/resolved_config.yaml
cat /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/smoke_evaluation.json
```

To start a small resumable sampler check, first set a small `sampler.max_calls`
in this copied terminal-check YAML, then run:

```bash
python -u run_retrieval.py \
  --config /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/configuration.yaml
```

After UltraNest reports iterations, press `Ctrl-C` once and inspect:

```bash
cat /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/ultranest/sampler_status.json
tail -n 5 /scratch/dp448/dc-tayl1/1_CPU_Terminal_Check/ultranest/run_attempts.jsonl
exit
```

## Submit a 64-CPU run

The production script requests 64 MPI ranks with account `dp448` on the DIaL3
`slurm` partition. It uses the Conda OpenMPI `mpirun` launcher so all processes
join one `MPI.COMM_WORLD`; using `srun` with this environment launched 64
independent size-one communicators and caused simultaneous writes to the same
UltraNest HDF5 checkpoint. Its only run input is the YAML path.

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
sbatch slurm/wasp69b_clear_native_modes.sbatch
```

Choose any project-specific copied YAML without editing the Slurm script:

```bash
sbatch --export=ALL,ROBERT_CONFIG=/scratch/dp448/dc-tayl1/my_project/configuration.yaml \
  slurm/wasp69b_clear_native_modes.sbatch
```

Use `squeue -u "$USER"` to monitor jobs and `scancel JOB_ID` to cancel one.
Use a different output directory whenever the target, priors, resolution, or
model changes so incompatible UltraNest checkpoints are never mixed. Do not
resume the output directory from the failed 64-writer test: change
`outputs.directory` in the copied YAML to a fresh directory before submitting
again.
