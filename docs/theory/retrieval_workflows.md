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

Optimal estimation and UltraNest consume the same `RetrievalProblem`
interface. Every run records the resolved configuration, opacity identifiers,
runtime and code provenance, settings, random seed, spectra, and inference
products.

See [Configuring and running ROBERT](../configuration.md) for the complete YAML
schema, directory setup, opacity preparation, and MPI/Slurm workflow. The
superseded pre-YAML HAT-P-32b examples are retained only in
`examples/Depreciated_Benchmarks/HAT_P_32b/`.
