# Running retrievals

ROBERT runs optimal estimation, UltraNest, MultiNest, and optimal-estimation
to nested-sampling workflows from one schema-version-2 YAML file. The same
configuration defines the observations, atmosphere, opacity, radiative
transfer, likelihood, priors, inference settings, runtime, and plotting.

MultiNest is the default nested sampler. UltraNest is an explicitly selected
compatibility option.

This guide first creates and checks a run directory, then gives separate
instructions for a local machine, a standard Slurm cluster, and the Oxford
Physics Glamdring cluster.

## 1. Install the complete environment

The supplied Conda environment keeps mpi4py, MPICH, MultiNest, and PyMultiNest
on one ABI:

```bash
conda env create --file environment.yml
conda activate robert-exoplanets
```

Do not mix an OpenMPI-loaded shell with this MPICH environment. Check the
native interfaces before a MultiNest run:

```bash
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
mpiexec -version
python -c "import pymultinest; print('PyMultiNest import: OK')"
```

The MPI library report and `mpiexec` must describe MPICH/Hydra.

## 2. Create one isolated run directory

Copying a YAML and editing it in place is sufficient locally, but isolated
directories make cluster runs reproducible and prevent checkpoints from
different models being combined:

```bash
python scripts/create_run_directory.py \
  --project-dir /path/to/robert-runs \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --slurm-account my-account \
  --slurm-partition compute \
  --slurm-time 48:00:00 \
  --slurm-tasks 32 \
  --glamdring-ranks 12 \
  --slurm-mail-user name@example.org
```

Only `--project-dir` is required. Account, partition, time, rank count, and
mail are site-specific. Without overrides, the generator selects one rank for
optimal estimation, 128 ranks for a standard nested-sampling Slurm job, and 12
single-node ranks for a Glamdring nested-sampling job.

The resulting `/path/to/robert-runs/<run.name>/` contains:

- `source_configuration.yaml`: the untouched input;
- `configuration.yaml`: the fully resolved execution configuration;
- `run_retrieval.py`, `run_forward.py`, and `run_oe_from_nested.py`;
- `postprocess_retrieval.py` and `postprocess_forward.py`;
- `submit.sbatch`: a conventional Slurm job;
- `submit.sh`: the Oxford Glamdring wrapper; and
- `README.md`: commands and paths for that run.

Writable products are local to the directory:

- `outputs/` contains result phases, snapshots, diagnostics, and plots;
- `opacity_cache/` contains tables prepared for the selected datasets; and
- `scratch/` contains runtime and plotting caches.

Use a new directory when changing a model, dataset selection, prior, opacity
resolution, or independent retrieval. Resume only the same inference problem.

## 3. Configure data, priors, and the sampler

Edit `configuration.yaml`. A nested-sampling block may look like:

```yaml
sampler:
  engine: multinest
  live_points: 400
  multinest_max_iterations: 0
  dlogz: 0.5
  sampling_efficiency: 0.8
  importance_nested_sampling: true
  multimodal: true
  iterations_before_update: 100
  resume: resume
  show_status: true
  seed: 2712
  invalid_loglike_floor: -1.0e100

runtime:
  mpi_processes: auto
```

For UltraNest, set `engine: ultranest`; `max_calls: null` runs until the
configured convergence criterion is reached. For MultiNest,
`multinest_max_iterations: 0` means unlimited. ROBERT accepts any
non-negative requested MultiNest seed and records both that seed and the
effective seed passed to the legacy native generator.

Optimal estimation uses:

```yaml
sampler:
  engine: optimal_estimation
  oe_max_iterations: 8
  oe_convergence_tolerance: 1.0e-4
  oe_finite_difference_fraction: 1.0e-4
  oe_damping: 0.0
```

Every parameter referenced by atmosphere, cloud, transmission-radius, regional
emission, or dataset nuisance settings must appear exactly once in
`parameters`. Dataset offsets, jitter, and uncertainty scaling are explicit:

```yaml
observations:
  loader: robert_npz
  path: observation.npz
  datasets: [g395h]
  dataset_options:
    g395h:
      offset_parameter: g395h_offset
      uncertainty_scale_parameter: g395h_error_scale
      jitter_parameter: g395h_jitter
```

The corresponding three names must be in `parameters`.

## 4. Preflight every run

Run these commands from the isolated directory:

```bash
python run_retrieval.py --config configuration.yaml --validate-only
python run_retrieval.py --config configuration.yaml --initialize
python run_retrieval.py --config configuration.yaml --prepare-opacity
python run_retrieval.py --config configuration.yaml --smoke-only
```

They perform distinct checks:

1. `--validate-only` resolves YAML and verifies cross-references without
   loading scientific data.
2. `--initialize` creates output, cache, and scratch directories.
3. `--prepare-opacity` loads the observations and prepares each selected
   dataset's opacity on one process.
4. `--smoke-only` builds the complete model, evaluates the likelihood once at
   the prior midpoint, and exercises manifest serialization.

Do not submit a long run until all four succeed in the same environment and
with the same input paths used by the job.

ROBERT's bundled R=100 tables cover H2O, CO, CO2, CH4, NH3, and HCN from
0.3–15 microns. They are useful for small retrieval demonstrations and
resolution-appropriate HST/WFC3 analyses. Select `opacity.resolution: R100`
and omit `paths.k_table_directory`. Production work should establish
resolution convergence for the data being analysed. Download the R=1000
parents with:

```bash
robert-opacity-download --directory opacity_data/ktables_exomol
```

Set `paths.k_table_directory: ./opacity_data/ktables_exomol` for those tables.
R=15000 data are not distributed with ROBERT; contact Jake Taylor directly for
the validated high-resolution data workflow.

### End-to-end local recovery check

Run `examples/notebooks/r100_emission_transmission_validation.ipynb` before a
larger retrieval. It generates both forward truths, adds seeded Gaussian noise
with 60 ppm uncertainties, runs MultiNest, and compares each posterior directly
with its injected H2O abundance.

The notebook defaults to two MPI ranks. Set `CORES` to 3 or 4 when those cores
are available on a laptop or within an interactive cluster allocation. It
launches MultiNest as an attached foreground subprocess and streams sampling
progress into the cell. Do not submit these short checks to a queue: watching
the run makes MPI, compiled-library, opacity, and convergence problems visible
immediately.

A case passes only when:

- MultiNest reports convergence;
- the injected abundance lies inside the weighted posterior 95% credible
  interval; and
- the reduced chi-square lies between 0.35 and 1.90.

The relatively wide fit interval accounts for the 17 residual degrees of
freedom in this deliberately compact check. Passing validates the installation
and workflow, not the adequacy of a one-molecule R=100 model for unrelated
science data.

## 5. Run locally

### One-process optimal estimation

```bash
conda run -n robert-exoplanets python -u run_retrieval.py \
  --config configuration.yaml
```

### Local MPI nested sampling

Use the `mpiexec` installed inside `robert-exoplanets`:

```bash
conda activate robert-exoplanets
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mpiexec -n 4 python -u run_retrieval.py --config configuration.yaml
```

With `runtime.mpi_processes: auto`, ROBERT reads the MPI/Slurm world. If an
explicit process count is configured, it must equal the launched world size.
Start with one or two ranks for a functional check before using all local
cores.

## 6. Run on a standard Slurm cluster

The generated `submit.sbatch` includes the selected account, partition,
walltime, ranks, working directory, writable Numba/Matplotlib caches, one
thread per rank, and:

```bash
mpirun -np "${SLURM_NTASKS}" \
  python -u run_retrieval.py --config configuration.yaml
```

Review the resource header before submission:

```bash
sed -n '1,80p' submit.sbatch
sbatch --test-only submit.sbatch   # available on many, but not all, Slurm sites
sbatch submit.sbatch
```

Useful Slurm commands are:

```bash
squeue -u "$USER"
scontrol show job JOB_ID
tail -f RUN_NAME-JOB_ID.out
scancel JOB_ID
```

Cluster-specific MPI policy takes precedence over the generic template. The
launcher used by `mpirun`, mpi4py, and a compiled MultiNest library must have a
compatible MPI ABI. Do not load a second MPI module after activating the Conda
environment.

For a PHOENIX stellar model, ensure `PYSYN_CDBS` is exported in the batch
environment. The generated script stops immediately if it is absent.

### Restarting a Slurm run

Keep `sampler.resume: resume`, leave the completed checkpoint files in place,
and submit the same run directory again. ROBERT writes a new retrieval-attempt
event and status record. A changed configuration should use a new directory,
even if the sampler would technically accept the old checkpoint.

## 7. Run on Glamdring (Oxford Physics only)

Glamdring's `addqueue` command starts work through an outer Slurm/PMIx step.
ROBERT's Conda environment uses MPICH/Hydra. The correct topology is:

```text
one addqueue wrapper
└── one Conda MPICH/Hydra world
    ├── rank 0
    ├── rank 1
    └── ...
```

Do not ask `addqueue` to launch one wrapper per rank. Submit exactly one
wrapper with `-s`, reserve one node with `-n 1xN`, and let `submit.sh` launch
all `N` ranks:

```bash
module unload openmpi 2>/dev/null || true
export ROBERT_CONDA_PREFIX="$HOME/anaconda3/envs/robert-exoplanets"
export ROBERT_MPI_RANKS=12
export PYSYN_CDBS=/path/to/synphot/reference-data

addqueue -q redwood -s -c "48 hours" -n 1x12 -m 4 -r ./submit.sh
```

Important Glamdring details:

- `-s` starts a single occurrence of the wrapper.
- `-n 1x12` reserves 12 CPUs on one node. Hydra's `fork` launcher is
  intentionally single-node.
- `-m 4` requests 4 GiB **per CPU**, or 48 GiB total for 12 CPUs. Divide a
  desired total memory by the rank count and round up.
- `-r` permits the queue system's restart/resubmission behavior; ROBERT's
  sampler checkpoints remain controlled by `sampler.resume`.
- The wrapper rejects a loaded OpenMPI module and verifies that its exact
  environment-local `mpiexec` reports Hydra.
- The wrapper clears inherited `PMI*` and `PMIX*` variables before creating the
  Conda MPI world while retaining the useful `SLURM_*` job metadata.
- `OMP`, OpenBLAS, MKL, NumExpr, and Accelerate thread counts are fixed at one
  so each MPI process uses one reserved CPU.
- Matplotlib uses the noninteractive `Agg` backend and writable per-job cache
  directories.
- HDF5 file locking is disabled for Glamdring's shared filesystem. Never run
  two independent jobs in the same ROBERT run directory.

The wrapper defaults to
`$HOME/anaconda3/envs/robert-exoplanets`. Set
`ROBERT_CONDA_PREFIX` when the environment is elsewhere. Alternatively set
`ROBERT_CONDA_ROOT` and `ROBERT_CONDA_ENV`.

After `addqueue`, inspect its response. A successful submission includes:

```text
Sending program's output to file:
```

Treat either of these conditions as a failed submission:

- the output contains `Batch job submission failed`; or
- it never reports an output-file location.

Oxford queue tools include:

```bash
q
showoutput JOB_ID
scancel JOB_ID
```

The current Physics IT `addqueue` syntax and queue descriptions are maintained
in the [Oxford Physics Glamdring documentation](https://itsupport.physics.ox.ac.uk/front/helpdesk.faq.php?id=993).

### Glamdring launch troubleshooting

`OpenMPI is loaded`
: Run `module unload openmpi`, confirm `module list`, and submit again.

`ROBERT requires the Conda MPICH/Hydra mpiexec`
: `ROBERT_CONDA_PREFIX` names the wrong environment or that environment does
  not contain the MPICH build from `environment.yml`.

MPI initializes as several one-rank worlds
: The job was submitted without `-s`, or an outer launcher started several
  wrappers. Cancel it and resubmit one wrapper with `-s -n 1xN`.

`PYSYN_CDBS` is unset
: Export the Synphot root before submission or place the export in the
  run-local wrapper.

No plot is written
: Check the retrieval result first, then the writable `MPLCONFIGDIR`. The
  maintained wrapper creates that directory and forces `MPLBACKEND=Agg`.

## 8. Monitor and inspect a run

ROBERT writes:

- `sampler_status.json`: current sampler state;
- `retrieval_attempts.jsonl`: append-only starts, finishes, failures, and
  restarts;
- `result.json`: portable scalar result and metadata;
- `result_arrays.npz`: samples, weights, likelihoods, or OE state/covariance;
- sampler-native checkpoint directories; and
- source and resolved configuration snapshots.

From another shell:

```bash
robert-retrieval-status outputs/multinest
python -m robert_exoplanets.retrieval.status outputs/multinest
```

The exact path is the sampler phase directory, such as `outputs/ultranest`,
`outputs/multinest`, or `outputs/optimal_estimation`.

## 9. Post-process completed inference

Generate or regenerate diagnostics without rerunning inference:

```bash
python postprocess_retrieval.py --config configuration.yaml
```

Each completed phase receives its own folder under `outputs/plots/`. Products
include:

- fit statistics and a posterior summary;
- a best-fit residual panel;
- posterior marginals, correlations, and a corner plot when dimensionality
  permits;
- temperature-pressure median and central 68% interval;
- observation-grid and native-grid spectral median and central 68% posterior
  predictive interval where supported;
- `posterior_predictive_quantiles.npz`, containing the q16/q50/q84 numerical
  products shown in the interval plots; and
- a plot manifest recording source and appearance settings.

The result's parameter names and order drive all plots. No truth or reference
marker is drawn unless one is explicitly available. Residuals are calculated
from the reported best-fit model; posterior-median markers and intervals are
labeled separately.
