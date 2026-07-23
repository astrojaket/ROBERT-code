# WASP emission-retrieval matrix

These configurations provide the matched MultiNest retrieval matrix used to
compare the Schlawin et al. WASP-69b spectrum with the Wiser et al. WASP-80b
spectrum. The four models are:

| Directory | Disk model | Clouds |
| --- | --- | --- |
| `clear` | one region | none |
| `one-region` | one region | fixed MgSiO3 Mie catalogue |
| `diluted` | diluted one region | fixed MgSiO3 Mie catalogue |
| `two-region` | independent hot and cold regions | independent fixed MgSiO3 Mie cloud structures |

The flexible geometric-albedo and direct-n/k sensitivity models are not part of
this matrix. WASP-69b uses all four disjoint native data segments, including
the six-point NIRCam overlap average. WASP-80b uses its three available native
segments.

## Create the DiRAC run directories

Run these commands from the ROBERT checkout after activating the
`robert-exoplanets` Conda environment. They create the simple directory layout
`WASP-69b/<model>` and `WASP-80b/<model>` beneath the selected project root.

```bash
project_root=/lustre/dirac3/scratch/dp448/dc-tayl1/my_project

for model in clear one-region diluted two-region; do
  python scripts/create_run_directory.py \
    --project-dir "${project_root}/WASP-69b" \
    --config "configurations/retrievals/WASP-69b/${model}/configuration.yaml"
done

for model in clear one-region diluted two-region; do
  python scripts/create_run_directory.py \
    --project-dir "${project_root}/WASP-80b" \
    --config "configurations/retrievals/WASP-80b/${model}/configuration.yaml"
done
```

Each generated model directory contains its resolved `configuration.yaml`,
`submit.sbatch`, runners, and local `outputs/`, `opacity_cache/`, and `scratch/`
locations. Every retrieval explicitly configures 128 MPI processes to match
the 128-task Slurm launcher. Inspect the generated configuration before
submission, then use:

```bash
cd "${project_root}/WASP-69b/clear"
python run_retrieval.py --config configuration.yaml --validate-only
sbatch submit.sbatch
```

Repeat the final two commands for each model directory. Existing run
directories are never overwritten; choose a new project root or move completed
runs before regenerating the matrix.
