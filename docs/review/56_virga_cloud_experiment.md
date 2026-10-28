# Experimental Virga cloud provider

ROBERT includes an experimental adapter to Virga's Ackerman–Marley cloud
calculation. It is an optional research path, not a validated replacement for
the existing Mie cloud retrievals. Virga 2.0.2 is the version tested against a
real MgSiO3 example. Install `virga-exo` separately in the
`robert-exoplanets` environment and prepare Virga `.mieff` tables before use.

The provider requires explicit condensable gas mass mixing ratios, Kzz,
sedimentation efficiency, particle-distribution width, and a Mie table for
each condensate. Sedimentation efficiency and log10(Kzz / (cm^2/s)) can be
fixed or retrieved; other Virga controls are fixed. Gravity and
mean molecular weight must be uniform, and pressure-edge temperatures must be
available. Its optical output uses the Virga asymmetry-factor closure rather
than ROBERT's exact Mie phase moments. Compare spectra with a high-stream
reference before interpreting strongly forward-scattering cases.

For example, set the cloud section of a supported task configuration to:

```yaml
clouds:
  model: virga_eddy_sed
  condensibles: [MgSiO3]
  gas_mass_mixing_ratios: {MgSiO3: 0.0001}
  mie_directory: /path/to/prepared/virga-mie
  fsed: 1.0
  kz_cm2_s: 1.0e8
  metallicity: 1.0
  geometric_stddev: 2.0
  supersaturation: 0.0
  multiple_scattering_backend: sh4
```

The abundance is an illustrative input, not a recommended physical prior.
Transmission also requires `radiative_transfer.gravity_model: constant`.
To retrieve the two controls, replace `fsed` and `kz_cm2_s` with
`fsed_parameter` and `log10_kz_cm2_s_parameter`, respectively, and define
their priors in the normal `parameters` list. The former is positive and
dimensionless; the latter is a uniform prior on the base-10 logarithm of Kzz
in cm^2/s. Use scientifically justified bounds for the chosen atmosphere.

FastChem condensate diagnostics are a separate opt-in Python API. They report
FastChem's fictitious condensate number densities and element condensation
fractions. These values are not aerosol particle number densities and are not
automatically converted to condensate mass fractions or connected to Virga.
Rainout is evaluated from high to low pressure. A condensate equilibrium
constant table must be supplied explicitly.

The optional real Virga integration check is enabled by
`ROBERT_RUN_VIRGA_TESTS=1`; it generates an MgSiO3 Mie table from the bundled
optical constants and compares the adapter against a direct Virga call. This
is an interface parity check, not independent cloud-formation validation or a
posterior-recovery result.
