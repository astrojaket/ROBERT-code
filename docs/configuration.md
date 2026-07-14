# Configuring and running ROBERT

ROBERT's user interface is a versioned YAML file. The file contains the
science choices; the Python runners contain no target-specific settings. Start
by copying `configurations/wasp69b_cloud_free_R1000.yaml` to a new filename and
editing the copy.

Schema version 2 adds explicit stellar `log_g_cgs`, `metallicity_dex`, and
`spectrum_model` fields and makes PHOENIX the default. To migrate a version-1
file, add those three fields under `bodies.star` and set `schema_version: 2`;
use `spectrum_model: blackbody` to reproduce the version-1 stellar treatment.

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
copies both runners and a minimal `submit.sbatch`, and sets the generated
configuration's output, opacity-cache, and scratch paths inside that directory.
The generated script requests one rank on one node for an OE-only run and 128
ranks across two nodes for UltraNest, MultiNest, and either OE-to-nested
workflow. It emails `jake.taylor@physics.ox.ac.uk` when the job begins, ends,
or fails.

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
- `plotting`: optional automatic retrieval/forward post-processing, image
  format, Matplotlib style, sampling display limit, colours, and labels;
- `outputs`: a project directory outside the source checkout; and
- `runtime`: `auto` uses `SLURM_NTASKS` under Slurm and one process otherwise,
  while `scratch_directory` controls runtime caches.

## Stellar spectra

Configured emission models use the STScI PHOENIX atmosphere grid by default.
Set the Synphot reference-data root before preparing a forward model; the path
must be the directory above `grid/`, not the `grid/phoenix` directory itself:

```bash
export PYSYN_CDBS=/Users/jaketaylor/Dropbox/NemesisPy-Docker/grp/redcat/trds
```

The stellar block records all three PHOENIX interpolation coordinates:

```yaml
bodies:
  star:
    name: WASP-69
    radius_m: 565568100.0
    effective_temperature_k: 4750.0
    log_g_cgs: 4.5
    metallicity_dex: 0.15
    spectrum_model: phoenix
```

Use `spectrum_model: blackbody` for the former Planck-spectrum behavior. The
PHOENIX atlas is loaded and flux-conservingly averaged onto every model's
spectral bins during model construction, so no stellar file I/O occurs during
likelihood evaluation. ROBERT converts the tabulated surface flux to radiance
as `F_lambda / pi` and normalizes the finite atlas integral to
`sigma * T_eff**4`; both choices are recorded in spectrum and run metadata.

Unknown fields and inconsistent molecule/parameter references are errors. To
inspect the resolved choices without loading data or opacity, run:

```bash
python run_retrieval.py --config configurations/wasp69b_cloud_free_R1000.yaml --validate-only
```

UltraNest runs to its convergence criteria by default (`sampler.max_calls:
null`). A positive `max_calls` is an optional safety cap for short checks. When
that cap is reached, UltraNest stops proposing likelihood evaluations and may
remain active briefly while all MPI ranks consolidate and write the final
partial result; `sampler_status.json` identifies that finalization state.

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

Set `plotting.enabled: true` to generate fit statistics and figures on rank 0
after a successful retrieval or forward run. Plotting is disabled by default;
it can always be run later with `postprocess_retrieval.py` or
`postprocess_forward.py`. See [Post-processing and plotting](postprocessing.md).

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

For DiRAC, create an isolated project directory and submit its generated
`submit.sbatch`. OE-only scripts request one rank; UltraNest, MultiNest, and
hybrid scripts request 128 ranks across two nodes. The script uses Conda MPICH
`mpirun`, not `srun`, and ROBERT verifies the MPI world before a sampler opens
shared output files.

```bash
cd /scratch/dp448/dc-tayl1/my_project/<run.name>
sbatch submit.sbatch
```

After any failed multi-writer launch, select a new `outputs.directory` in YAML.
An HDF5 checkpoint touched by independent writers must not be resumed.
