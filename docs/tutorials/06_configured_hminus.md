# Configure the H-minus continuum

Use one optional block at `radiative_transfer.hminus_continuum` for both
emission and transmission. This block uses the existing H-minus continuum
model. It does not calculate chemical equilibrium. It reads the H-minus,
hydrogen, and electron profiles from the selected chemistry model.

## Shared inputs

`robert_npz` accepts one file for one dataset. For separate detector grids, set
`observations.path` to a directory and give each detector a file named after
its dataset. For example, `nrs1.npz` and `nrs2.npz` remain separate spectra
with separate wavelength edges.

```yaml
observations:
  loader: robert_npz
  path: ./spectra
  datasets: [nrs1, nrs2]

atmosphere:
  temperature:
    model: parmentier_guillot_2014
    internal_temperature_k: 100.0
  chemistry:
    model: fastchem_equilibrium
    full_equilibrium_composition: true
    metallicity_parameter: metallicity
    carbon_to_oxygen_parameter: CtoO
    species:
      - {label: H2O, fastchem_name: H2O1}
      - {label: CO, fastchem_name: C1O1}
      - {label: CO2, fastchem_name: C1O2}

radiative_transfer:
  hminus_continuum:
    hminus_species: H-
    hydrogen_species: H
    electron_species: e-
    temperature_extrapolation: zero
    spectral_extrapolation: zero
```

With `full_equilibrium_composition: true`, ROBERT uses every gas species
returned by FastChem when it calculates mean molecular weight. The adapter
maps FastChem `H1-` to the default output label `H-`. The `species` entries in
YAML define the selected chemistry labels used by opacity and retrieval
settings. FastChem supplies the other gas profiles for the mean molecular
weight calculation.

The full state does not allow constant VMR overrides or pressure quenching.
These operations would change the FastChem equilibrium state and its mean
molecular weight. Leave the flag off for the legacy selected-species mode.
With the flag off, an enabled H-minus continuum requires explicit chemistry
entries for H-, H, and e-. In FastChem, map them to native IDs `H1-`, `H`, and
`e-`.

For free chemistry, declare all three profiles. You may retrieve them or set
one or more as fixed mixing ratios. Do not use only an H-minus scale. ROBERT
does not infer H-, H, or e- abundances for a free chemistry model.

```yaml
atmosphere:
  chemistry:
    model: free
    species: [H2O, H-, H, e-]
    parameter_names:
      H2O: log_H2O
      H-: log_Hminus
    fixed_mixing_ratios:
      H: 1.0e-5
      e-: 1.0e-12
    parameter_mode: log10
    background_species: [H2, He]
    fill_background: true
```

Add prior entries for `log_H2O` and `log_Hminus` to `parameters`. The H and
electron values above are example inputs; choose values that match the
retrieval design.

## Emission-only and transmission-only settings

Use the shared temperature, chemistry, opacity, likelihood, sampler, and
H-minus blocks in either mode. Set the radiative-transfer fields for the
selected mode only.

| Mode | Radiative-transfer fields |
| --- | --- |
| Emission only | `model: emission`, optional `geometry`, `thermal_integration_backend`, and `sh4_boundary_backend` |
| Transmission only | `model: transmission`, `reference_pressure_bar`, `gravity_model`, optional `radius_scale_parameter`, and `impact_quadrature_order` |
| Shared | `include_rayleigh`, `gas_combination`, and `hminus_continuum` |

Example mode settings:

```yaml
# Emission only
radiative_transfer:
  model: emission
  geometry:
    model: gauss_legendre_disk
    points: 4
```

```yaml
# Transmission only
radiative_transfer:
  model: transmission
  reference_pressure_bar: 1.0
  gravity_model: inverse_square
  impact_quadrature_order: 8
```

## Validity and defaults

The bound-free contribution is zero above 1.64 microns. The free-free fit is
valid from 0.26 to 11.39 microns and at temperatures of 2500 K or higher.
The default `temperature_extrapolation: zero` keeps the existing zero-opacity
policy below 2500 K. Set it to `raise` to fail when a layer is outside this
temperature range. `spectral_extrapolation: zero` keeps the existing zero
policy outside the free-free wavelength range; `raise` makes that range strict.

The same config block works for emission and transmission. The contribution
uses VMR profiles from the chemistry state and does not add a separate
retrieval parameter for the continuum.
