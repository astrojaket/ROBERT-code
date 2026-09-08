# Tutorial 2: External inputs, quenching, spots, and CLR retrievals

Use this tutorial after the bundled R=100 workflow works. The canonical WASP
target files are hot-Jupiter examples with external observations, FastChem,
R=1000 opacity tables, and PHOENIX stellar spectra. A separate section shows a
terrestrial-style free-chemistry CLR prior and its sampler boundary.

Set `JAX_PLATFORMS=cpu` for local CPU checks, and use the `robert-exoplanets`
Conda environment for every ROBERT command.

## 1. Prepare an external R=1000 hot-Jupiter run

Download the checksum-pinned ExoMolOP R=1000 parents into a project-local
directory:

```bash
conda run -n robert-exoplanets robert-opacity-download \
  --directory opacity_data/ktables_exomol
```

This command obtains only H2O, CO, CO2, CH4, NH3, and HCN R=1000 tables. It
does not obtain SO2 or H2S. The canonical emission files require SO2, and the
rocky CLR file requires SO2 and H2S, so the downloader alone is insufficient
for those runs. Obtain the extra ExoMolOP tables through the established
external-input workflow described in [Opacity Data](../theory/opacity_data.md)
and [Forward-model generation](../forward_models.md), then check the selected
species and source checksums before `--prepare-opacity`. Keep these large
tables outside Git.

The canonical [WASP-69b cloud-free native configuration](../../configurations/emission.yaml)
uses these tables with the native NIRCam and MIRI/LRS datasets and a
400-live-point PyMultiNest run. The
[cloudy entry](../../configurations/cloudy_emission.yaml)
adds fixed MgSiO3 Mie optical constants. Copy one YAML to an isolated run
directory, then set these paths for the local machine:

```yaml
paths:
  observations_directory: /data/wasp69b_schlawin2024
  fastchem_directory: /data/chemistry/fastchem
  k_table_directory: /data/opacity_data/ktables_exomol
  optical_constants_directory: /data/optical_constants/exo_skryer
```

Set `bodies.star.spectrum_model: phoenix` when the PHOENIX reference data are
available. Before model construction, set the Synphot reference-data root:

```bash
export PYSYN_CDBS=/data/synphot/reference-data
test -d "$PYSYN_CDBS/grid/phoenix"
```

Set `spectrum_model: blackbody` for a Planck-spectrum comparison that does not
need PHOENIX files. A PHOENIX model needs the star effective temperature,
`log_g_cgs`, and metallicity values that select a grid point. Check the planet
radius and mass, because they set the gravity and emission geometry.

Run the standard preflight sequence with the copied YAML:

```bash
CONFIG=/path/to/run/configuration.yaml

conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --validate-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --initialize
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --prepare-opacity
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG" --smoke-only
conda run -n robert-exoplanets python run_retrieval.py \
  --config "$CONFIG"
conda run -n robert-exoplanets python postprocess_retrieval.py \
  --config "$CONFIG"
```

The first four commands isolate configuration, directory, opacity, and one
likelihood errors before a long run. The final two commands run and plot the
PyMultiNest result. For an optimal-estimation comparison, use the maintained
[WASP-69b OE configuration](../../configurations/optimal_estimation.yaml)
with `sampler.engine: optimal_estimation`. OE is a separate Gaussian workflow;
do not use it for a CLR prior or for a covariance/profile likelihood.

## 2. Adapt the data and atmosphere

Keep the physical choices explicit when you adapt a target configuration:

1. Set `bodies.planet` and `bodies.star` from the target system.
2. Set `observations.loader`, its path, and the independent names in
   `observations.datasets`.
3. Set `atmosphere.pressure` before choosing temperature and chemistry. A
   quench-pressure prior must stay inside the layer-centre pressure domain.
4. Choose `isothermal`, `spline`, `tabulated`, or Parmentier--Guillot
   temperature parameters.
5. Choose FastChem or free chemistry, then list the same active gases in
   `opacity.species`.
6. Select `clouds.model` and add every cloud parameter to `parameters`.
7. Set the radiative-transfer model and each instrument's grid and bin edges.
8. Set priors, nuisance parameters, sampler controls, and the plotting output.

The model prepares every selected instrument separately. This preserves gaps
between NIRCam and MIRI and preserves each dataset's native wavelength unit.
Use a new opacity cache when the species, resolution, or instrument selection
changes.

## 3. Add pressure quenching to hot-Jupiter FastChem

Pressure quenching decorates FastChem equilibrium profiles. A custom group can
quench one gas or several gases at one pressure:

```yaml
atmosphere:
  chemistry:
    model: fastchem_equilibrium
    fastchem_path: /data/chemistry/fastchem
    species:
      - {label: H2O, fastchem_name: H2O1}
      - {label: CO, fastchem_name: C1O1}
      - {label: CO2, fastchem_name: C1O2}
    quenching:
      model: pressure_quench
      preset: custom
      groups:
        - {pressure_parameter: log_Pq_C, species: [H2O, CO, CO2]}
parameters:
  - name: log_Pq_C
    unit: log10(bar)
    prior: {type: uniform, lower: -5.5, upper: 2.0}
```

The `log_Pq_*` value is `log10(P_q / bar)`. Use a uniform prior on that
logarithmic value. The supported hot-Jupiter grouped preset is shorter:

```yaml
quenching:
  model: pressure_quench
  preset: taylor_2026_hot_jupiter_element_grouped
parameters:
  - {name: log_Pq_C, unit: log10(bar), prior: {type: uniform, lower: -5.5, upper: 2.0}}
  - {name: log_Pq_N, unit: log10(bar), prior: {type: uniform, lower: -5.5, upper: 2.0}}
```

This preset groups H2O, CO, CO2, and CH4 under `log_Pq_C`, and NH3 under
`log_Pq_N`. It does not add N2. Configured free chemistry is constant with
altitude, so adding quenching to it has no supported effect.

## 4. Add stellar spots to transmission

Stellar contamination is a transmission-only transform. A cool spot and a hot
facula can be declared with retrieved covering fractions:

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

Set the spot temperature below the immaculate stellar temperature and the
facula temperature above it. Covering-fraction priors are uniform in `[0, 1]`
and all fixed fractions plus prior upper bounds must be at most one. Keep the
transit-chord convention explicit when a non-immaculate chord is required.

## 5. Use a terrestrial-style CLR composition prior

The maintained [L 98-59 b CLR configuration](../../configurations/rocky_transmission_clr.yaml)
is the runnable terrestrial example. Its loader, planet, data files, and
three-species composition must stay matched. For a terrestrial or
secondary-atmosphere transmission test, use free
chemistry with a background gas as the derived composition category. Every
retrieved abundance must use the same centered-log-ratio prior, bounds, and
group. The block below matches the maintained L 98-59 b chemistry; change the
species, background gas, opacity list, and target data together when adapting
it:

```yaml
atmosphere:
  pressure:
    bottom_bar: 1.0
    top_bar: 1.0e-7
    layers: 80
  chemistry:
    model: free
    species: [SO2, H2S, CO2]
    parameter_mode: log10
    parameter_names: {SO2: log_SO2, H2S: log_H2S, CO2: log_CO2}
    background_species: [H2]
    background_fractions: [1.0]
    fill_background: true

parameters:
  - {name: log_SO2, prior: {type: centered_log_ratio, lower: -12.0, upper: 0.0, group: composition}}
  - {name: log_H2S, prior: {type: centered_log_ratio, lower: -12.0, upper: 0.0, group: composition}}
  - {name: log_CO2, prior: {type: centered_log_ratio, lower: -12.0, upper: 0.0, group: composition}}

sampler:
  engine: multinest
```

Merge these sections into a complete maintained transmission YAML and validate
the full file. A valid CLR draw produces positive, unit-sum abundances and uses
N retrieved coordinates plus the derived background category. Some unit-cube
points produce the explicit invalid sentinel and are rejected before the
likelihood evaluation; inspect sampler diagnostics rather than treating every
cube point as a valid composition. Direct nested sampling supports this joint
transform. Optimal estimation and optimal-estimation-to-MultiNest workflows
reject CLR priors before any forward evaluation. CLR availability in the
emission API does not establish an emission science validation claim.

## 6. Read the posterior products

After a successful run, call `postprocess_retrieval.py` with the same resolved
configuration. The plotting code uses the same 100 weighted-resampled vectors
for spectra, temperature profiles, VMR profiles, and supported cloud profiles.
It writes `posterior_predictive_quantiles.npz` with coordinate values and units,
the vectors under `posterior_draw_vectors`, and five quantile fields in the
order lower 2-sigma, lower 1-sigma, median, upper 1-sigma, upper 2-sigma.
The median and both bands use `mediumpurple`. A product that the forward model
does not expose is reported as unavailable; it is not replaced by a fabricated
spectrum or profile.
