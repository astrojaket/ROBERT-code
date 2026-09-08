# Forward-model generation

ROBERT evaluates emission and transmission models from the same validated YAML
used by retrievals. A forward run fixes every configured parameter, evaluates
the atmosphere and radiative-transfer calculation, applies each observation's
spectral bins, and writes a portable NumPy archive. It does not start a sampler.

## New-user emission and transmission check

The recommended first calculation is
`examples/notebooks/r100_emission_transmission_validation.ipynb`. It contains
two independent R=100 forward models:

- thermal emission with a fixed non-isothermal temperature profile and an
  injected `log_H2O = -3.0`; and
- transmission with a fixed reference radius and an injected
  `log_H2O = -3.3`.

Each model uses the bundled H2O table over 18 bins from 1.10–1.70 microns. The
notebook saves the forward model before adding noise, then uses that exact file
to create the synthetic observation. This explicitly tests the same
configuration-to-spectrum path used below while retaining a known answer for
the retrieval stage.

## 1. Install and select the environment

Create the complete environment once from the repository root:

```bash
conda env create --file environment.yml
conda activate robert-exoplanets
```

All examples below can also be run without activating the environment:

```bash
conda run -n robert-exoplanets python ...
```

If the stellar model is PHOENIX, set the STScI Synphot data root:

```bash
export PYSYN_CDBS=/path/to/synphot/reference-data
test -d "$PYSYN_CDBS/grid/phoenix"
```

## 2. Create an external run configuration

For a complete list of fields, read the
[configuration reference](configuration.md). Create a run directory from the
checkout, then edit the generated `configuration.yaml`:

```bash
cd /path/to/ROBERT-code
export ROBERT_CODE=$PWD
conda run -n robert-exoplanets python scripts/create_run_directory.py \
  --project-dir "$HOME/ROBERT-runs" \
  --config configurations/quickstart.yaml
cd "$HOME/ROBERT-runs/quickstart-emission-r100"
```

That example uses the six bundled R=100 molecular tables and a blackbody
stellar spectrum, so it runs without an external opacity or PHOENIX data
directory.

Make all machine-specific input paths valid in the generated configuration:

```yaml
schema_version: 2

paths:
  project_directory: .
  observations_directory: ./data/observations
  fastchem_directory: ./data/fastchem
  optical_constants_directory: ./data/optical_constants

run:
  name: my-forward-model
  description: Cloud-free emission evaluated on three observing modes.
```

Relative paths are resolved from the YAML file. `project_directory: .`
therefore keeps `outputs/`, `scratch/`, and `opacity_cache/` beside the
configuration. Environment variables such as `${ROBERT_DATA_ROOT}` may be used
inside YAML; ROBERT rejects an unresolved variable.

For a first model, use the molecular opacity included with ROBERT:

```yaml
opacity:
  format: exomol_kta
  resolution: R100
  species: [H2O, CO2, CO, CH4, NH3, HCN]
  binning:
    num: 300
    use_rebin: false
    remove_zeros: true
```

The bundled tables cover 0.3–15 microns and retain the complete ExoMolOP
pressure and temperature grid with eight g-points. They require no
`k_table_directory` and are intended for quick forward models and
resolution-appropriate observations such as HST/WFC3.

For R=1000 models, use one shared external ExoMolOP source collection. Download
the six tables supported by the ROBERT downloader once:

```bash
conda run -n robert-exoplanets robert-opacity-download \
  --directory /shared/ROBERT-data/ktables_exomol
```

The command writes canonical `<species>_R1000.kta` files beneath
`/shared/ROBERT-data/ktables_exomol/R1000` and verifies their SHA-256 values.
The maintained hot-Jupiter configurations also request SO2, and the L 98-59 b
CLR configuration requests SO2 and H2S. Download those additional ExoMolOP
files and save them as `SO2_R1000.kta` and `H2S_R1000.kta` in the same selected
directory, following the exact links and checks in
[Opacity Data](theory/opacity_data.md).

An existing flat collection is valid when it has no `R1000/` child. If that
child exists, the loader uses it and does not search the flat files, so do not
run the six-table downloader into a flat collection that contains additional
gases unless the child contains every selected gas. Point the external YAML to
the shared source and keep each run outside the checkout:

```yaml
paths:
  project_directory: .
  k_table_directory: /shared/ROBERT-data/ktables_exomol
```

The source collection can be read-only and is reused by multiple runs.
`--prepare-opacity` writes derived tables to each run's opacity cache and does
not change the source files.

For R=15000 opacity sampling, use the public
`examples/download_exomol_opacity_sampling.py` workflow described in
[Opacity Data](theory/opacity_data.md). These ExoMolOP/TauREx HDF5 cross
sections are separate from R=1000 KTA inputs and from true line-by-line data.
In a configured YAML, `opacity.format: exomol_cross_section_hdf` converts them
to correlated-k tables during preparation; use the Python provider for native
opacity sampling.
For true petitRADTRANS line-by-line inputs, use the public
[petitRADTRANS high-resolution guide](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/high_resolution_spectra.html)
and its documented R=1e6 input-data paths. ROBERT's LBL examples expect those
external files; they do not download them.

The principal forward-model choices are:

- `bodies`: planet radius plus mass or gravity, and stellar radius,
  temperature, gravity, metallicity, and spectrum model;
- `observations`: a published reader or a portable ROBERT NPZ, plus the
  statistically independent datasets to model;
- `atmosphere.pressure`: bottom and top pressure and the number of layers;
- `atmosphere.temperature`: Parmentier–Guillot, Madhusudhan–Seager,
  isothermal, spline, or tabulated profile;
- `atmosphere.chemistry`: FastChem equilibrium chemistry or a free,
  constant-with-altitude composition;
- `clouds`: no cloud, deck plus haze, catalogue Mie optical constants, or
  directly parameterized Mie refractive indices;
- `disk_emission`: one region, a diluted emitting region, or independent hot
  and cold regions;
- `opacity`: ExoMol/exo_k `.kta` tables or cross-section HDF input and target
  correlated-k preparation;
- `radiative_transfer`: emission or transmission, angular geometry, gas
  combination, Rayleigh inclusion, and solver controls;
- `parameters`: the numerical values used by the forward calculation; and
- `plotting`: optional diagnostics and output appearance.

### Fixed forward parameters

Each item in `parameters` contains a prior because the same YAML can be used
for retrieval. Add `value` to state the forward-model value explicitly:

```yaml
parameters:
  - name: metallicity
    unit: dex
    prior: {type: uniform, lower: -1.0, upper: 2.0}
    value: 0.5
  - name: CtoO
    prior: {type: uniform, lower: 0.0, upper: 1.0}
    value: 0.55
```

If `value` is absent, `run_forward.py` uses the prior midpoint. For a
log-uniform prior, that is the geometric midpoint. Explicit values are
recommended for any spectrum intended for comparison or publication.

### Emission and transmission

Thermal emission uses:

```yaml
radiative_transfer:
  model: emission
  geometry:
    model: gauss_legendre_disk
    points: 4
  include_rayleigh: true
  gas_combination: random_overlap
```

Transmission uses:

```yaml
radiative_transfer:
  model: transmission
  reference_pressure_bar: 1.0
  radius_scale_parameter: radius_scale
  gravity_model: inverse_square
  impact_quadrature_order: 8
  include_rayleigh: true
  gas_combination: random_overlap
```

The named `radius_scale` must also appear in `parameters`. Transmission output
is a transit-depth spectrum; emission output is a planet-to-star eclipse-depth
spectrum after applying the configured stellar model.

### Observation bins

A forward task always has an observation definition because its instrument
bins determine the reported spectrum. To ingest a new spectrum, convert a
named-column table:

```bash
python scripts/convert_observation_to_robert.py spectrum.csv observation.npz \
  --delimiter comma \
  --wavelength-column wavelength_um \
  --flux-column depth_ppm \
  --uncertainty-column sigma_ppm \
  --flux-unit ppm \
  --instrument JWST/NIRSpec-G395H
```

Then select it in YAML:

```yaml
observations:
  loader: robert_npz
  path: observation.npz
  datasets: [g395h]
  verify_checksum: false
```

## 3. Validate without loading scientific data

Validation resolves inherited YAML, expands paths, checks all named
parameters, and prints the selected model:

```bash
python run_forward.py --config my_forward_model.yaml --validate-only
```

This is the quickest way to catch a misspelled key, missing parameter,
incompatible cloud/geometry choice, invalid pressure range, or unresolved
environment variable.

To create the writable directories without loading data or opacity:

```bash
python run_forward.py --config my_forward_model.yaml --initialize
```

## 4. Prepare the opacity cache

Correlated-k input is prepared for the selected observation bins. Preparation
is a one-process operation:

```bash
python run_forward.py --config my_forward_model.yaml --prepare-opacity
```

ROBERT writes prepared tables under `opacity_cache/<resolution>/`. The cache
records the species, source table, target bins, quadrature, and preparation
settings. If any of those inputs change, use a new run directory or remove
only the affected cache after checking its path.

Typical preparation errors mean:

- a species in `opacity.species` has no matching source file;
- the source opacity range does not cover all selected wavelength bins;
- `PYSYN_CDBS` is absent for a PHOENIX stellar model;
- the FastChem directory is missing for equilibrium chemistry; or
- a configured optical-constants catalogue is absent for a Mie cloud.

## 5. Evaluate the model

Run:

```bash
python run_forward.py --config my_forward_model.yaml
```

The default product is `outputs/forward_model.npz`. It contains, for every
dataset named `NAME`:

- `NAME_wavelength_micron`: bin centers in microns;
- `NAME_model`: modeled eclipse or transit depth; and
- `parameter_PARAMETER`: every fixed numerical parameter.

ROBERT also writes the source and fully resolved configuration snapshots under
the output directory.

Inspect the archive:

```python
from pathlib import Path

import numpy as np

path = Path("outputs/forward_model.npz")
with np.load(path, allow_pickle=False) as model:
    print(model.files)
    wavelength = model["f322w2_wavelength_micron"]
    eclipse_depth = model["f322w2_model"]
    print(wavelength.shape, eclipse_depth.min(), eclipse_depth.max())
```

## 6. Generate plots

Enable automatic forward plotting:

```yaml
plotting:
  enabled: true
  forward: true
  retrieval: true
  style: default
  image_format: png
  dpi: 180
```

The run writes:

- `outputs/plots/forward/forward_spectrum_residuals.png`;
- `outputs/plots/forward/fit_statistics.json`;
- `outputs/plots/forward/forward_parameters.json`; and
- `outputs/plots/forward/plot_manifest.json`.

Plots can be regenerated without repeating radiative transfer:

```bash
python postprocess_forward.py --config my_forward_model.yaml
```

The residuals compare the model with the configured observation. Information
criteria in the JSON are descriptive for a forward run because its parameters
were prescribed rather than fitted.

## Python API example

The maintained script
`examples/configured_forward_model.py` performs validation, optional opacity
preparation, evaluation, and archive inspection:

```bash
python examples/configured_forward_model.py \
  --config my_forward_model.yaml \
  --prepare-opacity
```

Its core Python workflow is:

```python
from pathlib import Path

import numpy as np

from robert_exoplanets.io.configured_tasks import (
    describe_config,
    load_observations,
    prepare_opacity,
    run_forward_task,
)
from robert_exoplanets.io.task_config import (
    initialize_task_directories,
    load_task_config,
)

source = Path("my_forward_model.yaml")
config = load_task_config(source)
print(describe_config(config))

initialize_task_directories(config)
observations = load_observations(config)
prepare_opacity(config, observations)  # Run once for a new binning/cache.

output = run_forward_task(config, source)
with np.load(output, allow_pickle=False) as archive:
    for dataset in config.observations.datasets:
        wavelength = archive[f"{dataset}_wavelength_micron"]
        spectrum = archive[f"{dataset}_model"]
        print(dataset, wavelength.shape, spectrum.shape)
```

`run_forward_task` reloads the configured observations while constructing the
complete problem. Keeping opacity preparation as a separate explicit step
makes it clear which work is cached and which work is the requested evaluation.

## Jupyter notebook

Open the maintained notebook:

```bash
conda run -n robert-exoplanets jupyter lab \
  examples/notebooks/configured_forward_model.ipynb
```

The notebook walks through configuration validation, path inspection, optional
opacity preparation, forward evaluation, NPZ inspection, and plotting. Change
only its `CONFIG_PATH` cell to use another YAML file.
