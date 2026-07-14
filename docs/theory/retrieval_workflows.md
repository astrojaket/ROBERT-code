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

## Pressure-resolved optimal estimation

`VerticalProfileParameterization` defines a temperature, VMR, aerosol, or
cloud-fraction state block on explicit pressure levels. Temperature normally
uses linear coordinates; positive profiles use natural-log coordinates. A
NEMESIS-style exponential prior covariance correlates adjacent pressure levels
over a configurable number of scale heights. `LayerByLayerStateVector` combines
multiple blocks without hiding their parameter names or transforms.

The OE result includes the gain matrix, averaging kernel, measurement-error
covariance, smoothing-error covariance, and finite-difference Jacobian. Use the
averaging kernels and `degrees_of_freedom_for_signal` to decide how much
vertical structure is actually measured. The number of state levels is only a
calculation grid and must not be reported as the vertical resolution.

The HAT-P-32b injection/recovery benchmark and the current NEMESIS comparison
status are documented in
[NEMESIS-informed vertical-profile OE](../review/37_nemesis_vertical_profile_oe.md).

See [Configuring and running ROBERT](../configuration.md) for the complete YAML
schema, directory setup, opacity preparation, and MPI/Slurm workflow. The
superseded pre-YAML HAT-P-32b examples are retained only in
`examples/Depreciated_Benchmarks/HAT_P_32b/`.
