# ROBERT

ROBERT is a general-purpose radiative-transfer and atmospheric-retrieval
framework. It combines typed planet, star, atmosphere, opacity, observation,
instrument, forward-model, likelihood, and inference components behind a
strict YAML workflow.

ROBERT is production-ready for its documented and benchmarked emission,
transmission, and retrieval regimes. The documentation states the validation
boundaries and unsupported scope for each workflow.

ROBERT supports thermal-emission and transmission spectra, correlated-k and
opacity-sampling inputs, equilibrium and free chemistry, cloud-free and cloudy
atmospheres, one- and two-region emission, instrument binning, Gaussian
likelihoods, optimal estimation and PyMultiNest. Runs produce
portable configuration snapshots, manifests, numerical results, diagnostics,
and plots.

The standard YAML runners cover the configured emission and transmission
workflows. Line-by-line opacity, high-resolution observations, covariance
likelihoods, and accelerator paths also have Python APIs and dedicated scripts;
they are not all options in the standard YAML schema. See the
[supported methods](docs/supported_methods.md) and
[development roadmap](docs/architecture/development_roadmap.md) for current
interfaces, validation limits, and next steps.

PyMultiNest is the supported nested sampler. Optimal Estimation supports
deterministic retrievals and future detailed sounding models.

The Python distribution is named `robert-exoplanets`.

## Installation

Install Conda, then run these commands from a ROBERT checkout. Conda supplies
Python 3.12 and the required libraries:

```bash
git clone git@github.com:astrojaket/ROBERT-code.git
cd ROBERT-code
conda env create --file environment.yml
conda activate robert-exoplanets
```

This environment includes the compiled MPICH, MultiNest, and PyMultiNest
libraries as well as FastChem, opacity, plotting, notebook, and test
dependencies.

### First forward model and retrieval

The bundled quickstart needs no external observations, opacity downloads, or
stellar files. Generate its synthetic data, run the forward model, then run
and plot a small PyMultiNest retrieval:

```bash
conda run -n robert-exoplanets python examples/r100_injection_recovery.py \
  --config configurations/quickstart.yaml --generate
conda run -n robert-exoplanets python run_forward.py \
  --config configurations/quickstart.yaml
conda run -n robert-exoplanets python run_retrieval.py \
  --config configurations/quickstart.yaml
conda run -n robert-exoplanets python postprocess_retrieval.py \
  --config configurations/quickstart.yaml
conda run -n robert-exoplanets python examples/r100_injection_recovery.py \
  --config configurations/quickstart.yaml --evaluate
```

Results go to `examples/outputs/r100_validation/emission/`. The last command
checks recovery of the known injected abundance. Repeat these commands with
`configurations/transmission.yaml` for a transit retrieval. See the
[bundled tutorial](docs/tutorials/01_bundled_forward_and_retrieval.md) for each
step and the saved plots.

### Other installation and input choices

For a smaller editable installation into an existing Python 3.10–3.14
environment:

```bash
python -m pip install -e ".[dev,opacity,retrieval]"
```

MultiNest itself is a compiled library and is supplied by `environment.yml`;
the pip retrieval extra installs PyMultiNest and mpi4py. PHOENIX stellar spectra
also require the STScI Synphot reference data:

```bash
export PYSYN_CDBS=/path/to/synphot/reference-data
```

`PYSYN_CDBS` must name the directory containing `grid/phoenix`. A configuration
with `bodies.star.spectrum_model: blackbody` does not require those files.

JWST configurations default to R=1000 tables in
`opacity_data/ktables_exomol/`. These large files stay outside Git and must be
installed separately. See the [external-input tutorial](docs/tutorials/02_external_inputs_and_specialised_retrievals.md)
for the required gases and download limits. Each instrument retains its own
wavelength coverage and bin edges.

ROBERT also includes ready-to-use R=100 correlated-k tables for H2O, CO, CO2, CH4,
NH3, and HCN from 0.3 to 15 microns. They are suitable for quick forward
models and HST/WFC3-scale analyses. Select `opacity.resolution: R100` and omit
`paths.k_table_directory` to use them.

Verify the installation:

```bash
conda run -n robert-exoplanets python -m pytest
conda run -n robert-exoplanets python run_retrieval.py \
  --config configurations/quickstart.yaml \
  --validate-only
```

The bundled quickstart configuration is stored at
`configurations/quickstart.yaml`. It uses the bundled R=100 tables. Use
`configurations/emission.yaml` for the R=1000 WASP-69b science example.

### End-to-end installation validation

Before analysing real observations, run the
[R=100 emission and transmission validation notebook](examples/notebooks/r100_emission_transmission_validation.ipynb).
It uses only data distributed with ROBERT and performs two complete
injection-recovery experiments:

1. generate known emission and transmission forward models;
2. turn them into seeded 1.10–1.70 micron synthetic observations with 60 ppm
   uncertainties;
3. retrieve the injected H2O abundance with MultiNest on two to four local
   MPI ranks; and
4. require the truth to fall inside the posterior 95% credible interval.

Run the retrieval cells in the foreground and monitor their live MultiNest
output. Both tests are intentionally small and should not be submitted to a
batch queue. Representative passing reports are stored under
`data/validation/r100_quickstart/`.

## Forward models

Start with a schema-version-2 YAML configuration. Each parameter may have a
`value`; ROBERT uses the prior midpoint where `value` is omitted.

```bash
conda run -n robert-exoplanets python run_forward.py \
  --config configurations/quickstart.yaml \
  --validate-only

conda run -n robert-exoplanets python run_forward.py \
  --config configurations/quickstart.yaml \
  --prepare-opacity

conda run -n robert-exoplanets python run_forward.py \
  --config configurations/quickstart.yaml
```

The model is written to `outputs/forward_model.npz`. With forward plotting
enabled in YAML, ROBERT also writes fit diagnostics and a spectrum/residual
figure under `outputs/plots/forward/`.

For higher-resolution work, download the checksum-pinned ExoMolOP R=1000
parents into the standard external-data layout:

```bash
robert-opacity-download --directory opacity_data/ktables_exomol
```

This downloads about 1.4 GB and writes `R1000/H2O_R1000.kta`,
`R1000/CO_R1000.kta`, and the corresponding CO2, CH4, NH3, and HCN files.
Point `paths.k_table_directory` at `opacity_data/ktables_exomol`. R=15000 data
are not distributed with ROBERT; contact Jake Taylor directly for access to
the validated high-resolution data workflow.

See [Forward-model generation](docs/forward_models.md) for the full
configuration-to-spectrum workflow, output schema, troubleshooting, a
standalone Python example, and the
[Jupyter notebook](examples/notebooks/configured_forward_model.ipynb).

## Retrievals

Validate, initialize, prepare opacity, and smoke-test a retrieval before
starting inference:

```bash
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --validate-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --initialize
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --prepare-opacity
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --smoke-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml
```

Use `scripts/create_run_directory.py` to generate an isolated directory with
the resolved configuration, runners, post-processors, a standard Slurm script,
and an Oxford Glamdring launcher:

```bash
conda run -n robert-exoplanets python scripts/create_run_directory.py \
  --project-dir /path/to/runs \
  --config configurations/emission.yaml
```

See [Running retrievals](docs/retrievals.md) for detailed local, standard
Slurm, and Oxford-only Glamdring instructions, including MPI selection,
single-node `addqueue` syntax, memory accounting, resume behavior, monitoring,
and post-processing.

## Configuration and data

- [Configuration reference](docs/configuration.md)
- [Portable observation format](docs/data/observation_format.md)
- [Post-processing and plotting](docs/postprocessing.md)
- [Maintained configurations](configurations/README.md)
- [Supported methods](docs/supported_methods.md)
- [Bundled forward and retrieval tutorial](docs/tutorials/01_bundled_forward_and_retrieval.md)
- [External inputs and specialised retrieval tutorial](docs/tutorials/02_external_inputs_and_specialised_retrievals.md)
