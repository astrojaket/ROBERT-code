# Tutorial 1: Bundled-opacity forward models and retrievals

This tutorial starts with the small R=100 validation cases. It then points to
the clear and cloudy hot-Jupiter emission configurations. Use the same YAML for
a forward model and for a retrieval: `parameters.value` is used by a forward
run, while the prior is used by a retrieval.

Run commands from the ROBERT repository root. The commands below select the
supported environment for every Python call:

```bash
conda run -n robert-exoplanets python --version
export JAX_PLATFORMS=cpu
```

## 1. Validate the two small cases

The maintained [emission validation YAML](../../configurations/quickstart.yaml)
uses the bundled H2O R=100 table, a spline temperature profile, and a
synthetic emission observation. The maintained [transmission validation YAML](../../configurations/transmission.yaml)
uses the same table, an isothermal profile, and a synthetic transmission
observation. Both cases use a blackbody star and a Gaussian likelihood.

Validate both files before preparing any data:

```bash
conda run -n robert-exoplanets python run_forward.py \
  --config configurations/quickstart.yaml \
  --validate-only

conda run -n robert-exoplanets python run_forward.py \
  --config configurations/transmission.yaml \
  --validate-only
```

Generate the synthetic observation and the first opacity cache with the
maintained injection script. The script uses the fixed values in each YAML and
writes ignored files below `examples/outputs/r100_validation/`:

```bash
conda run -n robert-exoplanets python examples/r100_injection_recovery.py \
  --config configurations/quickstart.yaml --generate
conda run -n robert-exoplanets python examples/r100_injection_recovery.py \
  --config configurations/transmission.yaml --generate
```

After generation, run one case through the complete forward workflow:

```bash
CONFIG=configurations/quickstart.yaml

conda run -n robert-exoplanets python run_forward.py \
  --config "$CONFIG" --initialize
conda run -n robert-exoplanets python run_forward.py \
  --config "$CONFIG" --prepare-opacity
conda run -n robert-exoplanets python run_forward.py \
  --config "$CONFIG"
conda run -n robert-exoplanets python postprocess_forward.py \
  --config "$CONFIG"
```

The forward run writes `forward_model.npz` below the configured output
directory. The last command writes the forward fit and residual plots. Use the
transmission YAML in `CONFIG` to repeat the same sequence for transit depth.

## 2. Run a bundled-opacity retrieval

The retrieval runner has four useful preflight actions. `--validate-only`
checks the resolved YAML. `--initialize` creates output, cache, and scratch
directories. `--prepare-opacity` prepares one cache for each selected dataset.
`--smoke-only` builds the model and evaluates one prior-midpoint likelihood.

```bash
CONFIG=configurations/transmission.yaml

conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --validate-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --initialize
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --prepare-opacity
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --smoke-only
```

After these checks pass, run PyMultiNest by omitting the action flag:

```bash
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG"
conda run -n robert-exoplanets python postprocess_retrieval.py \
  --config "$CONFIG"
```

PyMultiNest is the supported nested sampler. Keep `sampler.engine: multinest`
and use a new output directory when you change the data, opacity resolution,
model, or prior. Do not combine checkpoints from different configurations.

With plotting enabled, post-processing uses one weighted-resampled set of 100
posterior parameter vectors for the predictive products. It evaluates the same
draws for every supported product and writes
`posterior_predictive_quantiles.npz`. The file contains the five ordered levels
lower 2-sigma, lower 1-sigma, median, upper 1-sigma, and upper 2-sigma, plus
the draw vectors and the native wavelength or pressure coordinates. The plots
use `mediumpurple` for the median and the two bands. Each observation dataset
keeps its own wavelength grid.

## 3. Change the physical model

Copy a maintained YAML to a new run directory before changing it. The main
sections to edit are:

| Question | YAML section | Typical change |
| --- | --- | --- |
| What is the system? | `bodies` | Planet radius and mass; star radius, temperature, gravity, metallicity, and spectrum model |
| What data are fitted? | `observations` | Loader, data path, and selected independent datasets |
| What is the vertical grid? | `atmosphere.pressure` | Pressure limits and layer count |
| How is temperature set? | `atmosphere.temperature` | Isothermal, spline, tabulated, or Parmentier--Guillot profile |
| How is composition set? | `atmosphere.chemistry` | Free chemistry or FastChem equilibrium chemistry |
| Are clouds present? | `clouds` | `none`, `deck_haze`, `mie_catalog`, or `mie_direct_nk` |
| Which spectrum is solved? | `radiative_transfer` | `emission` or `transmission`, geometry, and solver settings |
| Which opacity is used? | `opacity` and `paths` | Bundled R=100 tables or an external R=1000 table directory |
| What is retrieved? | `parameters` | Values for forward runs and bounded priors for retrievals |

For transmission, also set `reference_pressure_bar`, a gravity model, and the
impact quadrature order. For a free-chemistry model, keep the active species
in `opacity.species` and map each species to its parameter in
`chemistry.parameter_names`. Every parameter referenced by the atmosphere,
clouds, regional emission, or dataset nuisance settings must occur once in
`parameters`.

## 4. Clear, cloudy, and two-region hot-Jupiter emission

The canonical [clear WASP-69b emission configuration](../../configurations/emission.yaml)
uses the native F322W2, F444W, and MIRI/LRS data modes. The [one-region cloudy
configuration](../../configurations/cloudy_emission.yaml)
adds a fixed MgSiO3 Mie catalogue cloud. The [two-region configuration](../../configurations/two_region_emission.yaml)
uses separate hot and cold emission columns and a retrieved hot-area
fraction. Omitted regional fields inherit the top-level atmosphere and cloud
settings.

The two-region block has this form:

```yaml
disk_emission:
  model: two_region
  hot_fraction_parameter: hot_area_fraction
  hot_region: {}
  cold_region:
    atmosphere:
      temperature:
        model: isothermal
        parameter_name: cold_temperature
    clouds:
      model: none

parameters:
  - name: hot_area_fraction
    prior: {type: uniform, lower: 0.0, upper: 1.0}
  - name: cold_temperature
    unit: K
    prior: {type: uniform, lower: 400.0, upper: 2000.0}
```

These WASP files are external-data examples. Their R=1000 opacity tables,
FastChem directory, stellar data, and observation files must exist at the
paths in the YAML before `--prepare-opacity` or `--smoke-only` can succeed.
Use the small R=100 validation cases above to test the command flow without
those files.

Cloudy transmission follows the same command flow. Start from the [maintained
configuration inventory](../../configurations/README.md) and the [configuration
reference](../configuration.md),
set `radiative_transfer.model: transmission`, and set `clouds.model: deck_haze`
with its four cloud parameters. The cloudy transmission example uses external
cross-section HDF files. A clear transmission run can start from
`configurations/transmission.yaml` by keeping `clouds.model: none`.

## 5. Keep instruments independent

List only statistically independent datasets in `observations.datasets`. Each
dataset keeps its own coordinate values, bin edges, unit, and uncertainty. A
multi-instrument fit prepares opacity and evaluates the model on each grid
separately. Do not concatenate NIRCam and MIRI arrays, or add an overlap
average while also fitting the two parent modes, unless that covariance choice
is part of the declared likelihood.
