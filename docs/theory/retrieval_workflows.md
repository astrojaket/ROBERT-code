# Retrieval Workflows

ROBERT's maintained retrieval interface is the strict, versioned YAML task
workflow. Target-specific science choices belong in configuration files rather
than edited Python benchmark scripts.

## Standard sequence

Start from a supplied cloud-free or cloudy configuration and validate it before
loading observations or opacity:

```bash
python run_retrieval.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --validate-only
```

Prepare the mode-specific correlated-k opacity cache once, then run a midpoint
smoke evaluation:

```bash
python run_retrieval.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --prepare-opacity
python run_retrieval.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml \
  --smoke-only
```

Run the retrieval with the same configuration:

```bash
python run_retrieval.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml
```

For a deterministic forward calculation, ROBERT uses each parameter's YAML
`value`, or the midpoint of its prior when no value is supplied:

```bash
python run_forward.py \
  --config configurations/wasp69b_cloud_free_R1000.yaml
```

## Model orchestration

The configured task builder creates one `ParameterizedEmissionForwardModel`
per dataset. Cloudy configurations wrap the same atmospheric and opacity state
with the selected cloud optical model and multiple-scattering backend.
Multi-instrument runs evaluate temperature and chemistry once per likelihood
call, while retaining independent opacity preparation and radiative transfer
for each instrument mode.

Optimal estimation, UltraNest, and conda-provided MultiNest consume the same
`RetrievalProblem` interface. The `sampler.engine` choice also supports OE
followed by either nested sampler, with bounded priors refined from the OE
state and covariance. Every run records the resolved configuration, opacity
identifiers, runtime and code provenance, settings, random seed, timings,
spectra, and inference products.

See [Configuring and running ROBERT](../configuration.md) for the complete YAML
schema, directory setup, opacity preparation, and MPI/Slurm workflow. The
superseded pre-YAML HAT-P-32b examples are retained only in
`examples/Depreciated_Benchmarks/HAT_P_32b/`.

## Deferred MultiNest-to-OE analysis

A completed MultiNest result can be inspected and post-processed before a
separate OE refinement is authorized. For the WASP-69b Mie-catalogue run, use
the supplied 80-layer spline configuration and point the deferred runner at
the completed MultiNest result directory:

```bash
python run_oe_from_nested.py \
  --nested-config /path/to/multinest-run/configuration.yaml \
  --nested-result-dir /path/to/multinest-run/outputs/multinest \
  --oe-config /path/to/layer-oe-run/configuration.yaml
```

Parameters shared by the two models inherit the MultiNest best fit and
posterior covariance. The PG14 best-fit temperature profile is evaluated at
all 80 pressure-layer centres to initialize the new OE temperature state. The
temperature prior covariance is
`S_ij = sigma_T^2 exp(-|log10(P_i/P_j)| / L)`, with the supplied configuration
recording `sigma_T=250 K` and `L=1.5 dex`. This smooths departures from the
MultiNest profile instead of treating adjacent layer temperatures as
independent. The exact prior state and covariance are saved in
`multinest_to_oe_prior.npz`. MultiNest outputs remain unchanged. By default the
handoff refuses a result that has not reported convergence;
`--allow-unconverged` is available only for explicit diagnostic work.

For a generated one-rank DiRAC OE directory, replace the final
`run_retrieval.py` command in `submit.sbatch` with the deferred command above;
the standard submission command starts an independent midpoint-initialized OE
run and does not consume the MultiNest result.
