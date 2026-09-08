# Configuration reference

ROBERT's command-line interface uses a strict schema-version-2 YAML file.
Unknown keys, inconsistent parameter references, invalid physical ranges, and
unsupported model combinations are errors.

Start with one of the maintained root configurations in
[`configurations/`](../configurations/): `quickstart.yaml` is the bundled
R=100 emission check, `transmission.yaml` is the bundled R=100 transmission
check, and `emission.yaml`, `cloudy_emission.yaml`,
`optimal_estimation.yaml`, `two_region_emission.yaml`, and
`rocky_transmission_clr.yaml` are the R=1000 science examples. The public
files are short, self-contained starting points; this document is the full
option reference when you need to adapt a model.

Distinct PICASO and six-gas regression inputs are kept under
`tests/fixtures/configurations/` and are not public starting configurations.

This schema covers a subset of the Python API. `opacity.format` accepts
`exomol_kta` and `exomol_cross_section_hdf`; `likelihood.model` accepts
`gaussian`. Line-by-line providers, correlated likelihoods, time-resolved
high-resolution data, and device-compiled retrieval problems require Python
construction or their dedicated scripts. Do not add these as YAML keys.

## Paths

Keep machine-specific input locations in one block:

```yaml
schema_version: 2
paths:
  project_directory: .
  observations_directory: ./data/observations
  fastchem_directory: ./data/fastchem
  k_table_directory: ./opacity_data/ktables_exomol
  optical_constants_directory: ./data/optical_constants
```

Relative paths resolve from the YAML file. ROBERT expands `${VARIABLE}` and
rejects undefined variables. Writable directories default to `outputs/`,
`opacity_cache/`, and `scratch/` beneath `project_directory`.
Use the top-level `paths` block for all machine-specific locations. The old
`housekeeping` alias is not supported; move those entries under `paths`.
Omit `k_table_directory` when using the bundled R=100 tables. R=1000 tables
are external and use the directory layout described below.

## Run and bodies

```yaml
run:
  name: unique-run-name
  description: Concise description of data and model.

bodies:
  planet:
    name: Example b
    radius_m: 7.0e7
    mass_kg: 1.0e27
  star:
    name: Example
    radius_m: 5.0e8
    effective_temperature_k: 4800
    log_g_cgs: 4.5
    metallicity_dex: 0.0
    spectrum_model: phoenix
```

A planet may supply `gravity_m_s2` instead of `mass_kg`. Stellar
`spectrum_model` is `phoenix` or `blackbody`.

Transmission TSLE uses the same `bodies.star.spectrum_model` selection. A
complete spot-plus-facula example is:

```yaml
stellar_contamination:
  model: poseidon_rackham
  regions:
    - name: cool_spot
      kind: spot
      temperature_k: 4000.0
      covering_fraction_parameter: f_spot
    - name: hot_facula
      kind: facula
      temperature_k: 5200.0
      covering_fraction_parameter: f_fac
parameters:
  - {name: f_spot, prior: {type: uniform, lower: 0.0, upper: 0.4}}
  - {name: f_fac, prior: {type: uniform, lower: 0.0, upper: 0.4}}
```

Omitting `stellar_contamination` leaves transmission output unchanged. Each
region chooses exactly one fixed `covering_fraction` or
`covering_fraction_parameter`. Fraction priors must be uniform within `[0, 1]`,
and fixed fractions plus all prior upper bounds must sum to at most one. Spot
temperatures must be cooler and facular temperatures hotter than
`bodies.star.effective_temperature_k`. The default omitted chord is the
immaculate photosphere and matches POSEIDON; `transit_chord_temperature_k`
enables ROBERT's broader explicit-chord extension. See [Transit light source
effect and stellar contamination](theory/stellar_contamination.md) for the
equations, assumptions, preparation order, degeneracies, and validation scope.

The WASP-69 benchmark opacity set follows the molecules named in Schlawin et
al.: H2O, CO2, CO, CH4, NH3, and SO2. The first five use FastChem equilibrium
profiles. SO2 follows the paper's PICASO retrieval treatment as a constant
abundance controlled by `log_SO2`:

## Observations

Supported loaders are `robert_npz`, `bello_arufe2025_l9859b`,
`schlawin2024_wasp69b`, and `wiser2025_wasp80b`:

```yaml
observations:
  loader: robert_npz
  path: observation.npz
  datasets: [g395h]
  verify_checksum: false
  dataset_options:
    g395h:
      offset_parameter: g395h_offset
      uncertainty_scale: 1.0
      uncertainty_scale_parameter: g395h_error_scale
      jitter_parameter: g395h_jitter
```

Only selected, statistically independent datasets should be included. Every
named nuisance parameter must appear in `parameters`.

### Pressure-quench retrievals

FastChem equilibrium profiles can be decorated with explicit pressure-quench
groups. Each `pressure_parameter` is `log10(P_q / bar)`, and its prior belongs
in the normal `parameters` list. Bounds are a user and science-case choice;
ROBERT does not impose the Taylor et al. `[-5.5, 2]` range globally. Use a
`uniform` prior for a uniform prior on the logarithmic parameter—a
`log_uniform` prior would apply another logarithm and is not the intended
semantics.

One species in a group gives arbitrary molecular quenching. This example uses
CO2, demonstrating that CH4 and NH3 are not hard-coded:

```yaml
atmosphere:
  chemistry:
    model: fastchem_equilibrium
    fastchem_path: /path/to/fastchem
    species:
      - {label: H2O, fastchem_name: H2O1}
      - {label: CO, fastchem_name: C1O1}
      - {label: CO2, fastchem_name: C1O2}
    quenching:
      model: pressure_quench
      preset: custom
      groups:
        - {pressure_parameter: log_Pq_CO2, species: [CO2]}
parameters:
  - {name: log_Pq_CO2, unit: log10(bar), prior: {type: uniform, lower: -4.0, upper: 1.0}}
```

Multiple species may share a pressure, and multiple independent groups may be
declared:

```yaml
quenching:
  model: pressure_quench
  preset: custom
  groups:
    - {pressure_parameter: log_Pq_C_subset, species: [H2O, CO, CO2]}
    - {pressure_parameter: log_Pq_CH4, species: [CH4]}
    - {pressure_parameter: log_Pq_NH3, species: [NH3]}
```

The Taylor et al. (2026) hot-Jupiter grouped preset is shorter:

```yaml
quenching:
  model: pressure_quench
  preset: taylor_2026_hot_jupiter_element_grouped
parameters:
  - {name: log_Pq_C, unit: log10(bar), prior: {type: uniform, lower: -5.5, upper: 2.0}}
  - {name: log_Pq_N, unit: log10(bar), prior: {type: uniform, lower: -5.5, upper: 2.0}}
```

It applies `log_Pq_C` to H2O, CO, CO2, and CH4, and `log_Pq_N` to
NH3. It deliberately does not add N2. "Elemental" means grouped molecular
profiles here, not an elemental-conservation calculation. Custom Python groups
may include N2 if the base model produces it.

Strict YAML quenching currently supports the FastChem base. The Python
decorator is general for profile-producing `ChemistryModel` implementations;
ROBERT's configured free chemistry is constant with altitude, so quenching it
would be a no-op and is rejected as an unknown configuration field. See
[Chemistry](theory/chemistry.md) for interpolation, bounds, closure,
provenance, and validation-level details.

Free-chemistry nested sampling also accepts a joint CLR prior. All retrieved
abundances must use one shared group; the configured background gas is the
derived final composition category:

## Atmosphere

The atmosphere specifies a pressure grid, temperature profile, and chemistry:

```yaml
atmosphere:
  pressure:
    bottom_bar: 100.0
    top_bar: 1.0e-6
    layers: 80
  temperature:
    model: isothermal
    parameter_name: temperature
  chemistry:
    model: free
    species: [H2O, CO2, CO]
    parameter_mode: log10
    parameter_names: {H2O: log_H2O, CO2: log_CO2, CO: log_CO}
    background_species: [H2, He]
    background_fractions: [0.8547, 0.1453]
    fill_background: true
```

Temperature models are `parmentier_guillot_2014`, `isothermal`, `tabulated`,
`madhusudhan_seager_2009`, and `spline`. Chemistry models are
`fastchem_equilibrium` and `free`. The annotated template lists the required
fields for each choice.

## Clouds and projected emission

Cloud models are `none`, `deck_haze`, `mie_catalog`, and `mie_direct_nk`.
Emission can use one region, a diluted region, or hot and cold regions:

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
```

Regional blocks inherit omitted top-level atmosphere and cloud settings.
Dilution and two-region models require their fraction parameter to have bounds
inside `[0, 1]`.

## Opacity and radiative transfer

```yaml
opacity:
  format: exomol_kta
  resolution: R100
  species: [H2O, CO2, CO]
  binning:
    num: 300
    use_rebin: false
    remove_zeros: true
    g_points: 8

radiative_transfer:
  model: emission
  geometry:
    model: normal_emission
    points: 4
  include_rayleigh: true
  gas_combination: random_overlap
  thermal_integration_backend: auto
  sh4_boundary_backend: auto
```

Opacity formats are `exomol_kta` and `exomol_cross_section_hdf`. Gas
combination is `random_overlap` or `sum_by_g`. Emission geometry is
`normal_emission` or `gauss_legendre_disk`. For cloudy SH4 emission,
`sh4_boundary_backend: auto` selects the compiled batched solve when Numba is
available. Set it to `scipy` to force the scientific reference implementation.

### Molecular opacity locations

R=1000 is the default resolution for JWST work. The maintained science
configurations select the repository's `opacity_data/ktables_exomol/`
directory.
The local 16-table collection was copied from
`Dropbox/NemesisPy-Docker/ktables_exomol/` and verified by SHA-256. Its
`local_copy_manifest.json` records file identities. The directory is ignored
by Git. Direct `SPECIES_R1000.kta` files and a nested `R1000/` directory are
both accepted by the table reader.

For combined instruments, keep a separate observation grid and bin edges for
each dataset. ROBERT prepares the same molecular input tables on each grid;
it does not force all instruments onto one observed resolution. Input opacity
resolution must still be adequate for the finest data being modelled.

ROBERT distributes R=100 correlated-k tables for H2O, CO, CO2, CH4, NH3, and
HCN. They cover 0.3–15 microns on the complete 22-pressure,
27-temperature ExoMolOP grid with eight g-points. No opacity path is needed:

```yaml
paths:
  project_directory: .

opacity:
  format: exomol_kta
  resolution: R100
  species: [H2O, CO2, CO, CH4, NH3, HCN]
```

Use these tables for introductory and rapid forward models or
resolution-appropriate data such as HST/WFC3. For R=1000, download the
checksum-pinned ExoMolOP parents:

```bash
robert-opacity-download --directory opacity_data/ktables_exomol
```

The resulting layout is:

```text
opacity_data/ktables_exomol/
└── R1000/
    ├── H2O_R1000.kta
    ├── CO_R1000.kta
    ├── CO2_R1000.kta
    ├── CH4_R1000.kta
    ├── NH3_R1000.kta
    └── HCN_R1000.kta
```

Then add:

```yaml
paths:
  k_table_directory: ./opacity_data/ktables_exomol

opacity:
  format: exomol_kta
  resolution: R1000
```

The downloader retains the upstream ExoMolOP source identity and verifies the
same SHA-256 checksums used to generate the bundled tables. R=15000 data are
not bundled or downloaded by this command; contact Jake Taylor directly for
the validated high-resolution data workflow.

For transmission, set `model: transmission` and configure
`reference_pressure_bar`, optional `radius_scale_parameter`,
`gravity_model`, and `impact_quadrature_order`.

## Parameters and likelihood

```yaml
likelihood:
  model: gaussian
  include_normalization: true

parameters:
  - name: temperature
    label: Isothermal temperature
    unit: K
    prior: {type: uniform, lower: 500.0, upper: 2500.0}
    value: 1400.0
  - name: log_H2O
    prior: {type: uniform, lower: -12.0, upper: -1.0}
    value: -3.0
```

Prior types are `uniform`, `log_uniform`, and `centered_log_ratio`. Parameter
order is retained in sampler arrays and plots. `value` controls forward runs;
when absent, the prior midpoint is used.

## Sampler, plotting, and runtime

```yaml
sampler:
  engine: multinest
  live_points: 400
  multinest_max_iterations: 0
  dlogz: 0.5
  resume: resume
  seed: 2712

plotting:
  enabled: true
  retrieval: true
  forward: true
  image_format: png
  dpi: 180
  posterior_predictive_samples: 100
  posterior_predictive_seed: 0

runtime:
  mpi_processes: auto
```

MultiNest is the default inference engine. Available engines are
`optimal_estimation`, `multinest`, and `optimal_estimation_to_multinest`.
With `mpi_processes: auto`, ROBERT uses the launched MPI or Slurm world and
otherwise runs on one process.

## Validation

Resolve and inspect a configuration without loading data or opacity:

```bash
python run_retrieval.py --config configuration.yaml --validate-only
```

See [Forward-model generation](forward_models.md) and
[Running retrievals](retrievals.md) for complete execution workflows.
