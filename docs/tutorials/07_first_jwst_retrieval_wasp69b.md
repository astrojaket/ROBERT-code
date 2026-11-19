# First JWST retrieval with WASP-69b

This exercise fits the published WASP-69b eclipse depths from JWST NIRCam
F322W2, NIRCam F444W, and MIRI/LRS. It uses a cloud-free atmosphere, FastChem
equilibrium chemistry with a free SO2 abundance, and a blackbody stellar
spectrum. The small live-point count is for learning the workflow. The
cloud-free model and stellar approximation limit what you can infer from this
fit.

First install ROBERT and complete the bundled R=100 exercise. That exercise
checks your installation with a known synthetic spectrum before you use
observed data. [Installation](../../README.md#installation) ·
[Bundled forward model and retrieval](01_bundled_forward_and_retrieval.md)

## 1. Find the observations

Start in the ROBERT checkout. The Git checkout already includes
`data/wasp69b_schlawin2024/sp.dat`. The `schlawin2024_wasp69b` loader checks
its checksum. This run uses 274 eclipse depths across three datasets
and keeps their bin edges separate. The six overlap averages in the
280-row table are excluded because they combine measurements already in
the two NIRCam datasets.

`sp.dat` is the native table used in this exercise. The adjacent `spbin.dat`
has 79 points for visualization. For the source tables and original observing
products, use the links below. The original JWST files require reduction to
eclipse depths and uncertainties before a standard ROBERT retrieval can use
them. [Bundled table and checksums](../../data/wasp69b_schlawin2024/PROVENANCE.md) ·
[Published VizieR tables](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/168/104) ·
[Original JWST observations](https://doi.org/10.17909/v2v9-k243)

## 2. Download the six gas opacity tables

Run these commands from the checkout with `curl` and `shasum` available.
Allow about 1.5 GB for the downloads plus space for the prepared opacity
caches and the retrieval results. If you already use a flat opacity folder,
set `KTABLE_ROOT` to a new absolute path for this exercise. When an
`R1000/` child exists, ROBERT reads from the child.

```bash
export ROBERT_CODE="$PWD"
export KTABLE_ROOT="$ROBERT_CODE/opacity_data/ktables_exomol"

conda run -n robert-exoplanets robert-opacity-download \
  --directory "$KTABLE_ROOT" \
  --species H2O CO CO2 NH3
```

The downloader verifies the SHA-256 of these four ExoMolOP tables. Use the
two commands below for CH4 and SO2. The CH4 file is from the ExoMol team's
archived YT34to10 release. Its coefficients differ from the older file that
ROBERT used to derive its bundled R=100 table. The SO2 file is the ExoAmes
table. [YT34to10 release](https://zenodo.org/records/5716864) ·
[ExoAmes release](https://zenodo.org/records/5716834)

```bash
download_table() {
  local file="$KTABLE_ROOT/R1000/$1"
  local checksum="$2"
  local url="$3"
  if [ ! -e "$file" ]; then
    curl --fail --location --retry 2 --output "$file.part" "$url" &&
      printf '%s  %s\n' "$checksum" "$file.part" | shasum -a 256 -c - &&
      mv "$file.part" "$file" || return 1
  fi
  printf '%s  %s\n' "$checksum" "$file" | shasum -a 256 -c -
}

download_table CH4_R1000.kta \
  231564056da3d30cd401dac7b8454917ab0e77948f5a3dded948313627f078f2 \
  'https://zenodo.org/records/5716864/files/12C-1H4__YT34to10.R1000_0.3-50mu.ktable.NEMESIS.kta?download=1'

download_table SO2_R1000.kta \
  dcbde137f1b7e9ed8bb79e0a72a964acfef9b8bfd1163dd2c705227da80a088d \
  'https://www.exomol.com/db/SO2/32S-16O2/ExoAmes/32S-16O2__ExoAmes.R1000_0.3-50mu.ktable.NEMESIS.kta'
```

Continue when both checksum checks print `OK`. The input tree should
look like this:

```text
ROBERT-code/
├── data/wasp69b_schlawin2024/sp.dat
└── opacity_data/ktables_exomol/
    └── R1000/
        ├── CH4_R1000.kta
        ├── CO_R1000.kta
        ├── CO2_R1000.kta
        ├── H2O_R1000.kta
        ├── NH3_R1000.kta
        └── SO2_R1000.kta
```

These are NEMESIS correlated-k tables. The checksums record the exact files
used for this exercise. For your own retrieval, you can select newer ExoMol
tables. Verify the species, format, spectral range, pressure and temperature
coverage, and g ordinates. Record the source and checksum, then prepare a
new opacity cache in a new run. Keep the original input files available
for reproducing earlier runs. [ROBERT opacity input guide](../theory/opacity_data.md)

## 3. Create a run and check its YAML

```bash
conda run -n robert-exoplanets python scripts/create_run_directory.py \
  --project-dir "$HOME/ROBERT-runs/jwst-tutorial" \
  --config configurations/emission.yaml
cd "$HOME/ROBERT-runs/jwst-tutorial/hot-jupiter-emission-r1000"
```

Edit the generated `configuration.yaml`. The run creator resolves the
bundled observations and FastChem paths to absolute paths in the checkout.
Check that these fields match your checkout. Replace `/path/to/ROBERT-code`
with the actual path if you must edit them:

```yaml
paths:
  observations_directory: /path/to/ROBERT-code/data/wasp69b_schlawin2024
  fastchem_directory: /path/to/ROBERT-code/data/chemistry/fastchem
  k_table_directory: /path/to/ROBERT-code/opacity_data/ktables_exomol
```

If you set a different `KTABLE_ROOT`, enter its full path as
`paths.k_table_directory`. Change these five values within the existing
mappings. Keep the other fields, including the star temperature, the three
datasets, and the priors.

| YAML field | Value for this exercise |
| --- | --- |
| `bodies.star.spectrum_model` | `blackbody` |
| `atmosphere.pressure.top_bar` | `0.00001` |
| `sampler.live_points` | `50` |
| `sampler.multinest_max_iterations` | `0` |
| `runtime.mpi_processes` | `1` |

A blackbody lets the exercise run without an external PHOENIX grid.
The pressure of the top edge matches the lower pressure limit of the
downloaded tables. Set the iteration limit to zero so MultiNest runs
without an iteration cap and uses its evidence tolerance to stop.
The source `emission.yaml` has separate settings for the 400-live-point
PHOENIX workflow. Give that workflow its own run directory and review its
additional inputs before use.
[Maintained emission YAML](../../configurations/emission.yaml) ·
[External inputs and PHOENIX setup](02_external_inputs_and_specialised_retrievals.md#1-prepare-an-external-r1000-hot-jupiter-run)

## 4. Prepare opacity and run one forward model

From the run folder, validate the YAML, prepare opacity, and evaluate one
likelihood. Preparation reads the six source tables and saves the result
for each instrument grid.

```bash
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1

conda run -n robert-exoplanets python run_retrieval.py --config configuration.yaml --validate-only
conda run --no-capture-output -n robert-exoplanets python run_retrieval.py --config configuration.yaml --prepare-opacity
conda run --no-capture-output -n robert-exoplanets python run_retrieval.py --config configuration.yaml --smoke-only
conda run -n robert-exoplanets python run_forward.py --config configuration.yaml
```

Check that validation lists `f322w2`, `f444w`, and `lrs`, plus R=1000 and
`max_iterations=unlimited`. The smoke test must report a finite log likelihood.
The forward model uses the `value` fields and writes
`outputs/forward_model.npz`. MultiNest uses the `prior` fields to define
the parameter ranges and distributions. If preparation reports a missing
gas, check the filename and the resolved opacity folder before you start
sampling.

To inspect the forward spectrum and its residuals, plot the saved model.

```bash
conda run -n robert-exoplanets python postprocess_forward.py --config configuration.yaml
```

Open `outputs/plots/forward/forward_spectrum_residuals.png`.

## 5. Run the retrieval and view the plots

```bash
conda run --no-capture-output -n robert-exoplanets python run_retrieval.py --config configuration.yaml
```

Wait for MultiNest to finish and write the automatic plots. If the retrieval
stops early, use the same command to resume with the same YAML and inputs.
Use a new run for a different model or data selection.

In a fresh-checkout test on one macOS CPU, MultiNest took about 32 minutes
and finished after 7,966 likelihood evaluations. This time excludes the
downloads, opacity preparation, and plotting. The corner plot can warn
about sparse contours with 50 live points. Use the saved arrays to inspect
the result of this exercise.

| Product | Where to find it |
| --- | --- |
| Run identity and input metadata | `outputs/multinest/manifest.json` |
| Parameter samples and weights | `outputs/multinest/result_arrays.npz` |
| Fit and residuals | `outputs/plots/multinest/fit_spectrum_residuals.png` |
| Posterior parameter plots | `outputs/plots/multinest/posterior_corner.png` |
| Median model and central 1-sigma and 2-sigma numerical products | `outputs/plots/multinest/posterior_predictive_quantiles.npz` |

To regenerate the plots from the saved result, use the command below. For
this configuration it also evaluates posterior spectra on the native
opacity grid, which takes extra time.

```bash
conda run -n robert-exoplanets python postprocess_retrieval.py --config configuration.yaml
```

Inspect the residuals for each instrument and compare the posterior
distributions with their priors. For your own target, create a new run and set
its observation files, units, bin edges, opacity species, pressure and
temperature coverage, and priors. Run the same validation and smoke tests
before sampling. [Observation input format](../configuration.md#observations)
