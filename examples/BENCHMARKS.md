# Maintained Forward-Model Benchmarks

PICASO and petitRADTRANS are ROBERT's gold-standard independent forward-model
comparisons. Maintained benchmark plots use a shared purple visual language:
ROBERT and best-fitting model spectra use `mediumpurple`, related ROBERT curves
use darker or lighter purples, and external reference spectra use neutral dark
tones for contrast.

Run each code in its dedicated Conda environment.  The required
`robert-exoplanets`, `picaso`, and `petitradtrans-stable` separation, setup
files, local opacity paths, and smoke tests are documented in
`docs/emission_intercomparison_environments.md`.

## Stellar spectra

- `benchmark_g_star_stellar_spectrum.py`: flux-conserving STScI PHOENIX
  profile for a Sun-like G2V star versus the explicit blackbody fallback,
  including the resulting secondary-eclipse normalization difference.

## PICASO

- `compare_grey_cloud_rt_picaso.py`: controlled grey-cloud radiative transfer.
- `benchmark_sh4_rt.py`: Toon and SH4 scattering comparisons.
- `benchmark_end_to_end_cloud_parity.py`: independent cloud-property and RT
  assembly.
- `benchmark_official_picaso_molecular_cloud_parity.py`: official
  molecular-opacity and cloudy-emission parity.
- `benchmark_emission_intercomparison_stages_1_3.py`: staged ROBERT/PICASO/pRT
  thermal-emission comparison.  Stages 1--2 share optical depth to isolate
  radiative transfer; Stage 3 uses each code's native molecular opacity path.
- `benchmark_emission_intercomparison_v2_stage_1.py`: frozen WASP-17b
  Version-2 common-contract bootstrap and grey/isothermal closure on 40/80/160
  cells.  It runs the exact three interpreters, pilots resources at 80 cells,
  preserves native/R=100 spectra and complete vertical arrays, probes PICASO's
  exact-zero limitation without substituting a non-zero albedo, and retains the
  one predeclared continuous-angle eclipse-gate failure.
- `benchmark_emission_intercomparison_v2_stage_2.py`: frozen single-molecule
  H2O/CO/CO2/CH4 closure at exact common-contract VMRs.  It separates matched
  ROBERT/pRT optical-depth closure from native opacity attribution, keeps
  PICASO resort-rebin correlated-k only over 0.3--12 micron,
  preserves native/R=100 spectral, optical-depth, and vertical arrays, and
  reports the measured out-of-tolerance vertical closure regime.
- `benchmark_emission_intercomparison_v2_stage_3.py`: exact frozen solar-derived fixed-abundance
  H2O/CO/CO2/CH4 mixture crossed with a `2 x 2` H2--H2/H2--He CIA factorial for
  the frozen isothermal and PG14 non-inverted profiles on 40/80/160 cells.  It
  retains the measured out-of-tolerance, vertically converging ROBERT/pRT
  shared-tau regime without classifying a framework as failed, treats native
  differences as attribution only, uses PICASO resort-rebin correlated-k only,
  retires opacity sampling, and records unsupported native tensor
  and vertical-diagnostic interfaces explicitly.
- `benchmark_emission_intercomparison_stage_4.py`: native-opacity thermal
  structure and contribution-function comparison for isothermal, monotonic,
  inverted, and retrieved-like profiles on 40/80/160 vertical grids.
- `benchmark_emission_intercomparison_stage_5.py`: localized thermal-response
  and R=100 temperature-Jacobian comparison for the Stage-4 profiles, with a
  shared-optical-depth RT track and a native temperature-dependent-opacity
  track on the same 40/80/160 grids.
- `benchmark_emission_intercomparison_stage_6.py`: H2O, CO, CO2, and CH4
  abundance-response tensors and signed R=100 composition Jacobians, including
  shared case-specific optical-depth and native-opacity tracks, 40/80/160
  convergence, cross-species sensitivity fractions, and a three-amplitude
  finite-difference linearity audit.
- `benchmark_emission_intercomparison_stage_7.py`: absorbing-cloud placement
  and extinction on the 40/80/160 grids, including the full optical-depth,
  cloud-top, and spectral-slope matrix plus an archived Virga/Mie extinction
  field.  The launcher runs a primary-grid resource pilot, keeps `omega0=0`,
  separates identical-tau Track A from native-cloud Track B, and preserves
  full spectra, cloud effects, vertical profiles, convergence, timings, and
  peak-memory records.

## petitRADTRANS

- `benchmark_petitradtrans3_stable.py`: stable emission and transmission
  comparison.
- `benchmark_petitradtrans3_multispecies_emission.py`: multi-species emission.
- `benchmark_petitradtrans3_multispecies_transmission.py`: multi-species
  transmission.
- `benchmark_petitradtrans4.py`: petitRADTRANS 4 comparison.
- `benchmark_native_hdf_emission_convergence.py`: native-grid emission
  convergence.

The archived HAT-P-32b checks under `Depreciated_Benchmarks/` predate ROBERT's
YAML task workflow and are not maintained or run in CI.
