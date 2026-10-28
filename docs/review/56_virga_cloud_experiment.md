# Experimental Virga cloud provider

ROBERT includes an experimental adapter to Virga's Ackerman–Marley cloud
calculation. It is an optional research path, not a validated replacement for
the existing Mie cloud retrievals. Virga 2.0.2 is the version tested against a
real MgSiO3 example. Install `virga-exo` separately in the
`robert-exoplanets` environment and prepare Virga `.mieff` tables before use.

The provider requires explicit condensable gas mass mixing ratios, Kzz,
sedimentation efficiency, particle-distribution width, and a Mie table for
each condensate. Sedimentation efficiency and log10(Kzz / (cm^2/s)) can be
fixed or retrieved; other Virga controls are fixed. Gravity must be uniform,
and pressure-edge temperatures must be available. By default mean molecular
weight must also be uniform. A user may set `mean_molecular_weight_amu` for
Virga only; this is an explicitly recorded approximation and does not replace
the pressure-dependent profile used by ROBERT's gas and radiative transfer.
Its optical output uses the Virga asymmetry-factor closure rather
than ROBERT's exact Mie phase moments. Compare spectra with a high-stream
reference before interpreting strongly forward-scattering cases.

`gas_mass_mixing_ratios` are physical condensable vapor mass ratios in g/g.
Virga 2.0.2 multiplies its explicitly supplied `gas_mmr` by the condensable
molecular weight divided by the atmosphere molecular weight. ROBERT converts
its mass ratio to Virga's pre-conversion input and checks the round trip.
Earlier adapter results produced before this correction used a different
effective abundance and must not be interpreted as that stated mass ratio.

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
gas and condensate densities from the same solve, elemental condensation and
conservation, and mass-equivalent condensate fractions. They do not use the
retrieved metallicity or C/O of the independent gas chemistry wrapper and are
not connected to Virga. Fictitious condensate number densities are not aerosol
particle counts. Rainout is evaluated from high to low pressure; rainout mass
equivalents are not retained cloud-particle profiles. A condensate equilibrium
constant table must be supplied explicitly.

The optional real Virga integration check is enabled by
`ROBERT_RUN_VIRGA_TESTS=1`; it generates an MgSiO3 Mie table from the bundled
optical constants and compares the adapter against a direct Virga call. This
is an interface parity check, not independent cloud-formation validation or a
posterior-recovery result.

## High-stream thermal scattering check

On 2026-10-28, a bounded Redwood job compared ROBERT SH4 with
PythonicDISORT 1.8 using 16, 32 and 64 streams. Four synthetic five-layer
cases used a linear thermal source, a black lower boundary, no incident beam,
and no delta-M scaling. The SH4 upward flux relative to 64 streams differed
by -0.290% (Rayleigh), -0.137% (Henyey–Greenstein g=0.5), -0.0565%
(two-HG mixture), and +0.654% (g=0.9, single-scattering albedo 0.95).
The 32-to-64-stream change was below 1.1e-7 in each case. This comparison
does not validate Mie cloud formation, observed spectra, or retrievals.

The reproducible harness is `scripts/benchmark_sh4_disort.py`; it requires
PythonicDISORT 1.8 as an optional external reference. The archived cluster
report and payload logs are in
`/mnt/extraspace/jaketaylor/robert_sh4_pythonicdisort_redwood_attempt2_20261028`.
