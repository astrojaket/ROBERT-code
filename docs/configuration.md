# Configuring and running ROBERT

ROBERT's user interface is a versioned YAML file. The file contains the
science choices; the Python runners contain no target-specific settings. Start
by copying `configurations/wasp69b_cloud_free_R1000.yaml` to a new filename and
editing the copy.

The available WASP-69b/WASP-80b cloud-free, Mie-cloud, and temperature-profile
defaults are catalogued in [configurations/README.md](../configurations/README.md).

The runners may also be copied beside that YAML in an external project
directory. They import the installed `robert-exoplanets` package and do not
depend on their own location:

```bash
cp run_retrieval.py run_forward.py /path/to/my_project/
cd /path/to/my_project
python run_retrieval.py --config configuration.yaml --validate-only
```

## Isolated project directories

For DiRAC runs, create one directory per run beneath a project directory. The
creator uses `run.name` as the directory name, preserves the original YAML,
copies both runners and a 64-rank `submit.sbatch`, and sets the generated
configuration's output, opacity-cache, and scratch paths inside that directory.

```bash
cd /scratch/dp448/dc-tayl1/ROBERT-code
python scripts/create_run_directory.py \
  --project-dir /scratch/dp448/dc-tayl1/my_project \
  --config configurations/wasp69b_cloud_free_R1000.yaml
```

Edit `/scratch/dp448/dc-tayl1/my_project/<run.name>/configuration.yaml`, then
work entirely from that directory:

```bash
cd /scratch/dp448/dc-tayl1/my_project/<run.name>
python run_retrieval.py --config configuration.yaml --validate-only
python run_retrieval.py --config configuration.yaml --prepare-opacity
python run_retrieval.py --config configuration.yaml --smoke-only
sbatch submit.sbatch
```

The major sections are intentionally explicit:

- `bodies`: planetary and stellar parameters;
- `observations`: published-data loader, data path, and selected instrument
  datasets (including `lrs` for MIRI);
- `atmosphere`: pressure grid; a Parmentier-Guillot, isothermal, or external
  tabulated CSV temperature profile; chemistry model; and the FastChem name
  corresponding to each molecule;
- `clouds`: currently `none`, because this runner exposes only the validated
  cloud-free path;
- `opacity`: KTA root, `R1000`/`R15000`, selected opacity molecules, and an
  external cache directory;
- `radiative_transfer`: model, geometry, Rayleigh treatment, gas combination,
  and numerical backend;
- `parameters`: ordered retrieval priors and optional forward-model values;
- `sampler`: inference engine (OE, UltraNest, MultiNest, or OE followed by
  either nested sampler), convergence/iteration controls, resume policy, and
  seed;
- `outputs`: a project directory outside the source checkout; and
- `runtime`: `auto` uses `SLURM_NTASKS` under Slurm and one process otherwise,
  while `scratch_directory` controls runtime caches.

Unknown fields and inconsistent molecule/parameter references are errors. To
inspect the resolved choices without loading data or opacity, run:

```bash
python run_retrieval.py --config configurations/wasp69b_cloud_free_R1000.yaml --validate-only
```

All relative paths are resolved relative to the YAML file, not the runner or
current shell directory. All real actions create the configured output,
opacity-cache, and scratch directories automatically. They can also be created
without loading science data:

```bash
python run_retrieval.py --config configurations/wasp69b_cloud_free_R1000.yaml --initialize
```

Opacity is prepared once, on one process:

```bash
python run_retrieval.py --config configurations/wasp69b_cloud_free_R1000.yaml --prepare-opacity
```

Then run a one-evaluation terminal check or the retrieval:

```bash
python run_retrieval.py --config configurations/wasp69b_cloud_free_R1000.yaml --smoke-only
python run_retrieval.py --config configurations/wasp69b_cloud_free_R1000.yaml
```

The same configuration can produce a deterministic forward model. Each
parameter's `value` is used; if `value` is omitted, ROBERT uses that prior's
midpoint.

```bash
python run_forward.py --config configurations/wasp69b_cloud_free_R1000.yaml
```

Every executed task copies the input YAML and writes a fully resolved YAML to
the output directory. A retrieval's UltraNest checkpoints live in the
`ultranest/` subdirectory. Do not reuse one output directory after changing
the data, priors, molecules, model, or resolution; create a new project/run
directory instead.

A tabulated forward-model P-T profile is selected entirely in YAML:

```yaml
atmosphere:
  temperature:
    model: tabulated
    profile_path: inputs/temperature_profile.csv
    pressure_column: pressure_bar
    temperature_column: temperature_K
    pressure_unit: bar
    extrapolation: clip
```

`profile_path` is resolved relative to this YAML. New chemistry engines such
as a photochemistry emulator should be added as another validated
`atmosphere.chemistry.model` variant with its own explicit input paths; the
runner itself does not need target- or machine-specific edits.

For 64 MPI processes on DiRAC, validate and prepare opacity on the login node,
then submit `slurm/wasp69b_cloud_free_native_modes.sbatch`. Set `ROBERT_CONFIG` when
submitting to select a copied configuration. The batch script uses Conda
the Conda MPICH `mpirun`, not `srun`; ROBERT checks that `MPI.COMM_WORLD` really has
64 processes before UltraNest opens a checkpoint.

```bash
ROBERT_CONFIG=/scratch/dp448/dc-tayl1/configs/my_run.yaml \
  sbatch slurm/wasp69b_cloud_free_native_modes.sbatch
```

After any failed multi-writer launch, select a new `outputs.directory` in YAML.
An HDF5 checkpoint touched by independent writers must not be resumed.
