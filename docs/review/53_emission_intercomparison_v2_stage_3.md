# Emission intercomparison Version 2: Stage 3

## Outcome

The Version-2 Stage-3 scope, factorial design, numerical gates, and resource
limits below were frozen **before** the complete matrix was inspected.  The
resource pilot authorized the run, and the complete matrix is preserved in the
versioned Stage-3 report and numerical artifacts.

The scientific status is `out_of_tolerance_closure_regime`.  The exact-zero
scattering and isothermal analytic-control gates are met, while the maximum
Track-A symmetric-relative, eclipse-depth, and 80-to-160 convergence limits are
exceeded.  The shared ROBERT/pRT differences decrease by about fourfold with
each vertical-grid doubling, identifying another vertically converging closure
regime rather than a framework failure.

Stage 3 measures multi-species and collision-induced-absorption closure.  It
reports agreement, differences, representation effects, convergence, and
capability boundaries.  A threshold may be met or exceeded, but neither an
individual framework nor the framework as a whole is classified as failed.
The frozen methods and thresholds below must not be tuned after the complete
matrix is viewed.

## Frozen physical scope

Every case uses the authoritative serialized Version-2 common contract.  The
four frozen solar-derived fixed-abundance line absorbers are always enabled
together:

- H2O;
- CO;
- CO2; and
- CH4.

Their abundances are the exact common-contract values.  H2 and He retain the
exact serialized background fill, so the six-species VMRs sum to one.  The
declared mean molecular weight remains `2.321438174776293 u`.  Stage 3 does not
recompute equilibrium chemistry, substitute round-number abundances, alter the
H2/He ratio, or vary mean molecular weight as a free factor.  Mean-molecular-
weight effects are identified through the fixed shared composition and the
declared framework representations, not through a composition counterfactual.

The temperature cases are:

- the frozen `1755 K` isothermal analytic control; and
- the frozen PG14 non-inverted primary physical profile.

The evaluated common-contract temperature arrays are supplied identically to
every framework.  The pressure ladder is `40`, `80`, and `160` cells over the
frozen `1e-5--100 bar` domain, with `80` cells primary.  The common-contract
pressure-edge/centre mappings are unchanged.  All comparisons use the frozen
`0.8--12 micron` domain, exact `6550 K` blackbody stellar normalization, and
flux-conserving `R=100` definition.

## Frozen CIA factorial

The four line absorbers remain on in every case.  The CIA treatments form the
following `2 x 2` factorial:

| Case label | H2--H2 CIA | H2--He CIA |
| --- | --- | --- |
| `molecular_only` | off | off |
| `molecular_plus_h2_h2_cia` | on | off |
| `molecular_plus_h2_he_cia` | off | on |
| `molecular_plus_h2_h2_and_h2_he_cia` | on | on |

This factorial is crossed with both temperature profiles and all three
vertical grids.  It isolates each CIA pair and their combined treatment while
holding line abundances, H2/He abundances, mean molecular weight, pressure,
temperature, gravity, stellar normalization, and spectral products fixed.

## Track A: shared mean-optical-depth closure

Track A supplies an identical pressure-cell-by-wavelength mean optical-depth
tensor to the genuinely compatible ROBERT and stable-petitRADTRANS pure-
absorption paths.  Molecular-line and CIA components, their declared sum, and
the exact-zero scattering state are retained separately so that the factorial
can be audited.  Gates apply only to quantities with matched definitions.

PICASO 4.0 is not assigned a Track-A gate.  Its exact-`omega0=0` native thermal
interface does not provide the finite identical-tensor path required for this
closure.  The independently implemented absorbing-formal reference remains a
separately labelled array/unit-exchange diagnostic and is not promoted to a
native PICASO result.

The following numerical gates are frozen before the complete matrix is
inspected:

| Matched Track-A diagnostic | Frozen limit | Observed | Result |
| --- | ---: | ---: | --- |
| Maximum ROBERT/pRT symmetric relative difference | `5e-4` | `1.726821e-2` | exceeded |
| Maximum ROBERT/pRT eclipse-depth difference | `0.1 ppm` | `7.129640 ppm` | exceeded |
| Maximum Track-A 80-to-160 eclipse-depth change | `0.1 ppm` | `0.769169 ppm` | exceeded |
| Maximum isothermal analytic-control eclipse difference | `0.1 ppm` | `0.002509 ppm` | met |
| Maximum absolute single-scattering albedo | `0` | `0` | met |

The maxima cover the declared Stage-3 Track-A cases unless the report records
an unsupported quantity before evaluating that gate.  Unsupported interfaces
remain explicit and are not replaced with invented values or acceptance tests.

## Track B: native construction and attribution

Track B holds the high-level physical cases fixed while ROBERT, PICASO, and
stable petitRADTRANS construct their native line and CIA opacity
representations.  Database choice, CIA source data, pressure-temperature
interpolation, correlated-k construction, opacity sampling, unit conversion,
and native radiative-transfer representation are attribution effects.  Track B
has no cross-framework acceptance gates.

PICASO's mandatory primary molecular path is the official PICASO-4 resort-
rebin correlated-k representation frozen in the common contract.  PICASO
opacity sampling remains secondary, separately labelled, and unsmoothed.  Its
native samples, flux-conserving R=100 product, sample counts, within-bin
variance where available, and the declared sampling-density diagnostic are
retained rather than treated as interchangeable with correlated-k.

Native-resolution spectra, R=100 products, optical-depth components and totals,
and pressure-resolved diagnostics are retained where genuinely supported.
Stable pRT's supported high-level native-flux interface does not expose a
stable layer optical-depth tensor, so its spectrum and supported native
emission-contribution product are retained while the missing tensor remains a
documented capability boundary.  Similar-looking framework-native vertical
diagnostics are not assumed to have identical definitions.

## Preserved prior-stage framing

Stage 1's eight-angle representation differs from the continuous-angle
analytic solution by `0.196897 ppm`.  Its dedicated angular relative gate is
met, but the independently frozen `0.01 ppm` analytic eclipse gate is not.
Stage 3 does not change that result: claims requiring sub-`0.01 ppm`
continuous-angle closure must not use the eight-angle Stage-1 product as
validated.

Stage 2 identified an out-of-tolerance, vertically converging Track-A regime.
Its maximum ROBERT/pRT eclipse-depth difference decreased from `10.083885 ppm`
at 40 cells to `2.541600 ppm` at 80 and `0.636478 ppm` at 160, while the
isothermal controls remained below `0.002510 ppm`.  Stage 3 preserves that
measured regime and its interpretation; it does not reinterpret it as a
framework failure or tune the discretization, quadrature, opacity, or gates to
force agreement.

## Resource pilot and launch decision

Before the complete matrix, run one representative 80-cell three-framework
pilot using the primary PG14 non-inverted case and the complete line-plus-both-
CIA treatment.  Measure wall time, process-tree peak resident memory, and
available memory at the decision point.  Project the complete workload from
the measured pilot.

The frozen resource gates are:

| Resource diagnostic | Frozen limit |
| --- | ---: |
| Projected complete-matrix wall time | `7200 s` |
| Largest process-tree peak RSS / available memory | `60%` |

Stop before the complete matrix if either limit is exceeded.  Do not replace a
stopped pilot with an unmeasured runtime or memory estimate.

### Pilot measurements

The representative
`pg14_non_inverted_molecular_plus_h2_h2_and_h2_he_cia_80_cells` pilot took
`12.825173 s`.  Applying the frozen workload multiplier of 36 projected
`461.706212 s`, well below `7200 s`.  The largest process peak RSS was
`3,952,787,456 bytes` against `10,001,924,096 bytes` available, or `39.5203%`,
below the frozen `60%` limit.  Both resource gates therefore authorized the
complete matrix.

The report records `192.703337 s` for the post-pilot matrix and retains the raw
per-case timings.  Pilot plus post-pilot matrix wall time was `205.528509 s`.

## Track-A results

| Resolution | Maximum symmetric relative difference | Maximum eclipse difference |
| ---: | ---: | ---: |
| 40 cells | `1.726821e-2` | `7.129640 ppm` |
| 80 cells | `4.352092e-3` | `1.792972 ppm` |
| 160 cells | `1.089822e-3` | `0.448767 ppm` |

The roughly fourfold decrease at each grid doubling is the same convergence
signature localized in Stage 2.  The maximum 80-to-160 Track-A change is
`0.769169 ppm`, so the shared calculation has not entered the frozen
`0.1 ppm` vertical-convergence regime at 160 cells.  The isothermal controls
remain much tighter: ROBERT's maximum analytic eclipse difference is
`0.002509 ppm` and stable pRT's is `0.001975 ppm`.  All retained cloud,
Rayleigh, and single-scattering-albedo arrays are exactly zero.

## CIA factorial and native attribution

The isothermal spectra are insensitive to the CIA factorial at numerical
precision because an isothermal pure-absorption atmosphere with its matching
thermal lower boundary is an analytic no-signal control.  On the primary
80-cell PG14 non-inverted case, adding both CIA pairs relative to the molecular-
only case changes the maximum eclipse depth by:

| Representation | Both CIA pairs versus molecular only |
| --- | ---: |
| ROBERT shared mean-tau | `8.128949 ppm` |
| stable-pRT shared mean-tau | `8.105700 ppm` |
| ROBERT native random overlap | `43.398931 ppm` |
| stable-pRT native correlated-k | `43.674657 ppm` |
| PICASO correlated-k | `0.000327 ppm` |
| PICASO opacity sampling | `51.574306 ppm` |

H2--H2 is the larger individual CIA effect in each representation that shows a
resolved physical signal.  For example, the 80-cell native ROBERT effects are
`37.766841 ppm` for H2--H2 and `9.473100 ppm` for H2--He with the other factor
off; the corresponding native pRT values are `38.053224 ppm` and
`9.500461 ppm`.  The native ROBERT and pRT factorial interactions are
`3.841009 ppm` and `3.879028 ppm`, respectively.  The much smaller PICASO
correlated-k CIA response and the larger opacity-sampling response are retained
as database/interpolation/representation attribution, not forced into a
shared-input agreement claim.

Track B remains ungated.  At 80 cells its maximum native R=100 differences are
`6.470394 ppm` for ROBERT versus stable pRT, `658.119872 ppm` for PICASO
correlated-k versus stable pRT, and `658.872868 ppm` for ROBERT versus PICASO
correlated-k.  These values combine native database, interpolation,
correlated-k, and RT-representation effects and do not rank the frameworks.

Native vertical convergence also remains representation dependent.  The
80-to-160 maximum changes are `0.876343 ppm` for ROBERT, `0.519259 ppm` for
stable pRT, `0.016944 ppm` for PICASO correlated-k, and `8.252217 ppm` for the
secondary PICASO opacity-sampling product.  The opacity-sampling density check
increases the native grid from 819 to 1638 samples and changes the R=100 eclipse
depth by as much as `214.295213 ppm`.  No smoothing was applied; the result is
finite-sampling evidence, not a correlated-k or shared-RT gate.

The full-mixture mean molecular weight stays fixed at
`2.321438174776293 u` in every case, so it cannot alias the CIA factorial.  No
counterfactual spectrum was run.  The report retains the derived H2/He-only
value `2.3045507066 u` and full-mixture/H2-He-only number-column ratio
`0.9927254284` as attribution context only, without modifying the contract.

## Tensor and diagnostic capabilities

Track A retains the four-molecule random-overlap tensor, separate H2--H2 and
H2--He mean-CIA tensors, selected totals, and the corresponding ROBERT and
stable-pRT spectra and vertical diagnostics.  ROBERT's native products retain
the supported component and total tensors.

PICASO's supported high-level path exposes total native `taugas` combining the
four molecules and requested CIA, but not separate native molecular/CIA
component tensors.  Its pressure-resolved arrays are absorbing-formal
diagnostics applied to that total, not native SH contribution definitions.
Stable pRT's high-level native-flux interface retains spectra and native
emission contributions but does not expose a stable native layer optical-depth
tensor.  These missing or non-identical interfaces remain declared capability
boundaries and are not populated with invented products or gates.

## Artifacts, provenance, and warnings

`stage_3_report.json` is accompanied by 22 numerical NPZ shards and
`stage_3_integrity.json`.  The shards cover shared optical depths, ROBERT and
stable-pRT shared/native products, PICASO correlated-k, PICASO opacity sampling,
and the opacity-sampling density check at the applicable resolutions.  The
integrity manifest records keys, shapes, dtypes, units, axes, finite policies,
sizes, and SHA-256 values; the Version-2 checksum file currently has 24 Stage-3
entries including the integrity manifest.  Final verification of these current
artifacts is recorded below.

The report records ROBERT `0.3.0`, PICASO `4.0`, and stable petitRADTRANS
`3.3.3` under their exact isolated interpreters, plus molecular, CIA, PICASO
database, reference, source, and output checksums.  The pRT-HDF CIA sources are
the frozen H2--H2 BoRi R831 and H2--He BoRi delta-wavenumber-2 tables; PICASO's
native CIA source is its frozen 661-bin continuum database.

ROBERT, PICASO 4.0, and stable petitRADTRANS remain process-isolated in the
three exact Version-2 interpreters.  Every PICASO import is preceded by the
frozen `picaso_refdata` setting and writable task-local Numba and Matplotlib
cache directories.  The harmless optional-Vega and exact-zero cloud/Rayleigh
warnings are recorded; no missing stellar data were downloaded, the warnings
were not suppressed, and exact zero was not replaced with an epsilon.

Detailed raw worker products remain ignored beneath the Version-2 Stage-3
output namespace.  Only committed Stage-3 data products under
`docs/data/emission_intercomparison/version_2/` enter the integrity and
checksum records.

## Verification

Final verification used the exact isolated interpreters and frozen data paths:

- focused Stage-3 contract and artifact tests: `32 passed`;
- strict SHA-256 and integrity-manifest verification: passed as part of the
  focused artifact suite for all 24 Stage-3 entries;
- Ruff over the repository: passed;
- exact-environment smoke tests: ROBERT `0.3.0`, PICASO `4.0`, and stable
  petitRADTRANS `3.3.3` all passed; and
- complete ROBERT suite: `501 passed, 5 skipped`.

The PICASO smoke emitted the recorded harmless optional-Vega warning.  The
scientific gate exceedances remain preserved measurements and were not treated
as orchestration or test failures.
