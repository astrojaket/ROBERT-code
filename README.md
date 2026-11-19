# ROBERT

ROBERT calculates and fits exoplanet thermal-emission and transmission
spectra. A YAML file specifies the planet, star, atmosphere, gases, clouds,
observations, and parameters. The forward model calculates a spectrum for
specified parameter values on each instrument grid. A retrieval compares
these spectra with the data through a likelihood to estimate the parameters.
ROBERT saves the resolved configuration, input
metadata, numerical results, and plots.

The YAML workflow supports correlated-k radiative transfer, equilibrium
and free chemistry, cloud-free and cloudy models, and one- and two-region
emission. Use PyMultiNest for nested sampling or optimal estimation for
supported Gaussian problems. Python APIs and dedicated scripts also cover
opacity sampling and line-by-line calculations.
[Supported methods and validation limits](docs/supported_methods.md) ·
[Development roadmap](docs/architecture/development_roadmap.md)

## Start here

1. **Install ROBERT.** Use the Conda environment below. Set up GitHub SSH
   access before running the clone command.
2. **Run the bundled R=100 exercise.** Generate a synthetic observation,
   run a forward model, and retrieve the injected water abundance. Check
   the recovery report and plots. [Bundled tutorial](docs/tutorials/01_bundled_forward_and_retrieval.md)
3. **Fit a published JWST spectrum.** Use the included WASP-69b observations.
   Download the six R=1000 ExoMolOP tables, prepare the opacity for each
   instrument, and run the retrieval. [First JWST retrieval with WASP-69b](docs/tutorials/07_first_jwst_retrieval_wasp69b.md)
4. **Set up your target in a new run folder.** Use a reduced transit- or
   eclipse-depth spectrum with uncertainties, units, and bin edges. Keep
   each instrument in its own dataset. Check that the gas tables cover the
   wavelengths, pressures, and temperatures your model uses.
   [Observation format](docs/configuration.md#observations) ·
   [Opacity inputs](docs/theory/opacity_data.md)
5. **Check the model before sampling.** Validate the YAML, prepare opacity,
   and run one likelihood with `--smoke-only`. Inspect a forward spectrum
   on the data grid and review the priors. Then run MultiNest to convergence
   with `multinest_max_iterations: 0`. [Retrieval guide](docs/retrievals.md)

## Installation

Install Conda, then obtain ROBERT and create its environment. Conda supplies
Python 3.12 and the required libraries. The Python distribution is named
`robert-exoplanets`.

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

Keep the checkout as the software directory. Create each simulation in a
separate directory, then edit that run's `configuration.yaml`. From the
ROBERT checkout:

```bash
export ROBERT_CODE="$PWD"
conda run -n robert-exoplanets python scripts/create_run_directory.py \
  --project-dir "$HOME/ROBERT-runs" \
  --config configurations/quickstart.yaml
cd "$HOME/ROBERT-runs/quickstart-emission-r100"
```

The bundled quickstart needs no external observations, opacity downloads, or
stellar files. Generate its synthetic data, run the model, then run and plot
a small PyMultiNest retrieval:

```bash
conda run -n robert-exoplanets python "$ROBERT_CODE/examples/r100_injection_recovery.py" \
  --config configuration.yaml --generate
conda run -n robert-exoplanets python run_forward.py --config configuration.yaml
conda run -n robert-exoplanets python run_retrieval.py --config configuration.yaml
conda run -n robert-exoplanets python postprocess_retrieval.py --config configuration.yaml
conda run -n robert-exoplanets python "$ROBERT_CODE/examples/r100_injection_recovery.py" \
  --config configuration.yaml --evaluate
```

Results go to this run's `outputs/` directory. Its `opacity_cache/` and
`scratch/` also stay outside ROBERT. The last command checks recovery of the
known injected abundance. Create another run from
`configurations/transmission.yaml` for the transit case. See the
[bundled tutorial](docs/tutorials/01_bundled_forward_and_retrieval.md) for each
step and the saved plots.

The environment uses an editable installation: updating the ROBERT checkout
updates the package used by these runs. Keep each run's configuration and
inputs in place. An existing checkpoint still belongs to its recorded code
and model; do not resume it across a change in scientific implementation.

### Other installation and input choices

From the ROBERT checkout, a smaller editable installation into an existing
Python 3.10–3.14 environment is also possible:

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

From the ROBERT checkout, verify the installation:

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

From the isolated run directory:

```bash
conda run -n robert-exoplanets python run_forward.py --config configuration.yaml --validate-only
conda run -n robert-exoplanets python run_forward.py --config configuration.yaml --prepare-opacity
conda run -n robert-exoplanets python run_forward.py --config configuration.yaml
```

The model is written to the run's `outputs/forward_model.npz`. With forward
plotting enabled, diagnostics and figures go to `outputs/plots/forward/`.

For R=1000 work, download the ExoMolOP files once into a shared data directory.
Set `paths.k_table_directory` in each run to that same absolute location. The
[opacity input guide](docs/theory/opacity_data.md) gives the download links,
filenames, supported layouts, and separate high-resolution routes. Preparing
a new instrument grid does not require another source download.

See [Forward-model generation](docs/forward_models.md) for the full
configuration-to-spectrum workflow, output schema, troubleshooting, a
standalone Python example, and the
[Jupyter notebook](examples/notebooks/configured_forward_model.ipynb).

## Retrievals

Use the plain-text [run checklist](RUN_STEPS.txt) for the command order on a
local laptop, DiRAC, or another Slurm cluster. It covers new runs, model checks,
submission, monitoring, checkpoint resume, and posterior plots.

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
- [First JWST retrieval with WASP-69b](docs/tutorials/07_first_jwst_retrieval_wasp69b.md)
- [External inputs and specialised retrieval tutorial](docs/tutorials/02_external_inputs_and_specialised_retrievals.md)

## Development and validation records

- [Code audit and upgrade register](docs/review/57_repository_audit_2026-11-06.md)
- [Local fixes and elemental recovery tests](docs/review/59_local_elemental_recovery_2026-11-06.md)
- [BeAR performance audit and proposed speed tests](docs/review/58_bear_2026_performance_audit.md)
