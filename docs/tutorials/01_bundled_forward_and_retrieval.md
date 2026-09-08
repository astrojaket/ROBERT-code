# Tutorial 1: Bundled-opacity forward models and retrievals

This tutorial starts with the small R=100 validation cases. It then points to
the clear and cloudy hot-Jupiter emission configurations. Use the same YAML for
a forward model and for a retrieval: `parameters.value` is used by a forward
run, while the prior is used by a retrieval.

Keep the source checkout for code and shared inputs. Create every simulation in
an external run directory. The run directory contains its own
`configuration.yaml`, `outputs/`, `opacity_cache/`, and `scratch/` directories;
published observations, bundled opacity tables, FastChem inputs, and the
editable package remain shared inputs and are read-only during a run.

Run source commands from the ROBERT repository root. The commands below select
the supported environment for every Python call:

```bash
conda run -n robert-exoplanets python --version
export JAX_PLATFORMS=cpu
```

## 1. Create and run an external R=100 validation case

The maintained [emission validation YAML](../../configurations/quickstart.yaml)
uses the bundled H2O R=100 table, a spline temperature profile, and a
synthetic emission observation. Create the run from the checkout, then work
only in the copied run directory:

```bash
cd /path/to/ROBERT-code
export ROBERT_CODE=$PWD

conda run -n robert-exoplanets python scripts/create_run_directory.py \
  --project-dir "$HOME/ROBERT-runs" \
  --config configurations/quickstart.yaml
cd "$HOME/ROBERT-runs/quickstart-emission-r100"
```

`create_run_directory.py` creates small forward, retrieval, and post-processing
runner wrappers and writes a resolved `configuration.yaml`. The wrappers
dispatch to the current checkout, so source-code fixes flow into an existing
run. Edit only `configuration.yaml` for run-specific choices. Keep
`source_configuration.yaml`, the runner wrappers, and shared input paths
unchanged. The run's `outputs/`, `opacity_cache/`, and
`scratch/` directories hold all generated observations, checkpoints, caches,
and runtime files.

Validate and initialize the generated configuration, then generate the synthetic
observation with the source-controlled validation script:

```bash
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml --validate-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --validate-only
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml --initialize
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --initialize

conda run -n robert-exoplanets python \
  "$ROBERT_CODE/examples/r100_injection_recovery.py" \
  --config configuration.yaml --generate
```

Run the forward model and its diagnostics from the external run:

```bash
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml --prepare-opacity
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml
conda run -n robert-exoplanets python postprocess_forward.py \
  --config configuration.yaml
```

Run the retrieval preflight, then sample and evaluate the recovery:

```bash
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --prepare-opacity
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --smoke-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml
conda run -n robert-exoplanets python postprocess_retrieval.py \
  --config configuration.yaml
conda run -n robert-exoplanets python \
  "$ROBERT_CODE/examples/r100_injection_recovery.py" \
  --config configuration.yaml --evaluate
```

The forward model and recovery report are written below the external run's
`outputs/` directory. The same workflow applies to the maintained
[transmission validation YAML](../../configurations/transmission.yaml). Give it
its own run directory so its outputs and checkpoints cannot mix with the
emission case:

```bash
cd "$ROBERT_CODE"
conda run -n robert-exoplanets python scripts/create_run_directory.py \
  --project-dir "$HOME/ROBERT-runs" \
  --config configurations/transmission.yaml
cd "$HOME/ROBERT-runs/hot-jupiter-transmission-r100"
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml --validate-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --validate-only
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml --initialize
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --initialize
conda run -n robert-exoplanets python \
  "$ROBERT_CODE/examples/r100_injection_recovery.py" \
  --config configuration.yaml --generate
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml --prepare-opacity
conda run -n robert-exoplanets python run_forward.py \
  --config configuration.yaml
conda run -n robert-exoplanets python postprocess_forward.py \
  --config configuration.yaml
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml --smoke-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config configuration.yaml
conda run -n robert-exoplanets python postprocess_retrieval.py \
  --config configuration.yaml
conda run -n robert-exoplanets python \
  "$ROBERT_CODE/examples/r100_injection_recovery.py" \
  --config configuration.yaml --evaluate
```

### Work from a notebook

Copy a notebook to the external run area before editing it. Keep notebook
outputs and generated figures outside the source checkout, and call validation
scripts through `$ROBERT_CODE` as above. The complete Conda installation uses
an editable package, so source-code fixes are available to the notebook after
restarting its kernel; the run's `configuration.yaml` remains the only file to
edit for run settings. For example, from an external run directory:

```bash
cp "$ROBERT_CODE/examples/notebooks/r100_emission_transmission_validation.ipynb" \
  .
```

## 2. Change the physical model

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

## 3. Clear, cloudy, and two-region hot-Jupiter emission

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
with its four cloud parameters. The `deck_haze` model can use the bundled H2O
R=100 table in the small transmission validation case. Cross-section HDF input
is needed only when the selected YAML sets
`opacity.format: exomol_cross_section_hdf`; see the configuration reference for
the required paths and cloud parameters. A clear transmission run can start from
`configurations/transmission.yaml` by keeping `clouds.model: none`.

## 4. Keep instruments independent

List only statistically independent datasets in `observations.datasets`. Each
dataset keeps its own coordinate values, bin edges, unit, and uncertainty. A
multi-instrument fit prepares opacity and evaluates the model on each grid
separately. Do not concatenate NIRCam and MIRI arrays, or add an overlap
average while also fitting the two parent modes, unless that covariance choice
is part of the declared likelihood.
