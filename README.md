# ROBERT

ROBERT is a general-purpose radiative-transfer and atmospheric-retrieval
framework. It combines typed planet, star, atmosphere, opacity, observation,
instrument, forward-model, likelihood, and inference components behind a
strict YAML workflow.

ROBERT supports thermal-emission and transmission spectra, correlated-k and
opacity-sampling inputs, equilibrium and free chemistry, cloud-free and cloudy
atmospheres, one- and two-region emission, instrument binning, Gaussian
likelihoods, optimal estimation, UltraNest, and MultiNest. Runs produce
portable configuration snapshots, manifests, numerical results, diagnostics,
and plots.

The Python distribution is named `robert-exoplanets`.

## Installation

The complete reproducible installation uses Conda and Python 3.12:

```bash
git clone git@github.com:astrojaket/ROBERT-code.git
cd ROBERT-code
conda env create --file environment.yml
conda activate robert-exoplanets
```

This environment includes the compiled MPICH, MultiNest, and PyMultiNest
libraries as well as UltraNest, FastChem, opacity, plotting, notebook, and test
dependencies.

For a smaller editable installation into an existing Python 3.10–3.14
environment:

```bash
python -m pip install -e ".[dev,opacity,retrieval]"
```

MultiNest itself is a compiled library and is supplied by `environment.yml`;
the pip retrieval extra installs UltraNest and mpi4py. PHOENIX stellar spectra
also require the STScI Synphot reference data:

```bash
export PYSYN_CDBS=/path/to/synphot/reference-data
```

`PYSYN_CDBS` must name the directory containing `grid/phoenix`. A configuration
with `bodies.star.spectrum_model: blackbody` does not require those files.

Verify the installation:

```bash
conda run -n robert-exoplanets python -m pytest
conda run -n robert-exoplanets python run_retrieval.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --validate-only
```

## Forward models

Start with a schema-version-2 YAML configuration. Each parameter may have a
`value`; ROBERT uses the prior midpoint where `value` is omitted.

```bash
conda run -n robert-exoplanets python run_forward.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --validate-only

conda run -n robert-exoplanets python run_forward.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --prepare-opacity

conda run -n robert-exoplanets python run_forward.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml
```

The model is written to `outputs/forward_model.npz`. With forward plotting
enabled in YAML, ROBERT also writes fit diagnostics and a spectrum/residual
figure under `outputs/plots/forward/`.

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
  --config configurations/wasp69b_cloud_free_R1000.yaml
```

See [Running retrievals](docs/retrievals.md) for detailed local, standard
Slurm, and Oxford-only Glamdring instructions, including MPI selection,
single-node `addqueue` syntax, memory accounting, resume behavior, monitoring,
and post-processing.

## Configuration and data

- [Configuration reference](docs/configuration.md)
- [Portable observation format](docs/data/observation_format.md)
- [Post-processing and plotting](docs/postprocessing.md)
- [Complete annotated YAML](configurations/TEMPLATE_all_supported_options.yaml)
