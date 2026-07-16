# Emission intercomparison: Stage 4

## Outcome

Stage 4 now compares native-opacity thermal spectra and pressure-resolved
contribution functions for isothermal, monotonic, inverted, and retrieved-like
temperature profiles.  The full matrix ran in strict process isolation using
`robert-exoplanets` for ROBERT, `picaso` for PICASO 3.2.2, and
`petitradtrans-stable` for petitRADTRANS 3.3.3.

The comparison uses 40, 80, and 160 ROBERT pressure cells.  PICASO receives
the corresponding 41, 81, and 161 cell edges as atmospheric levels.  pRT
receives the 40, 80, and 160 geometric cell centres as pressure nodes.  All
reported contribution functions therefore have the same physical pressure
coordinate and row count at a given resolution.

Stage 4 remains a Track-B characterization without a cross-code acceptance
gate.  ROBERT and pRT share source HDF correlated-k tables through independent
loaders.  PICASO uses its independent official SQLite opacity database, so its
large spectral and contribution differences retain the opacity attribution
established in Stage 3.

## Thermal structures

The four profiles are deterministic functions of log pressure from `1e-5` to
`100 bar`:

- isothermal: 1250 K;
- monotonic: 800 K at the top to 1700 K at the bottom;
- inverted: 850 K at the top, warming to 1500 K, cooling to 1300 K at greater
  pressure, then warming to 1750 K at the bottom;
- retrieved-like: a shallow upper atmosphere followed by a steep photospheric
  gradient and a deep-temperature plateau from 760 to 1660 K.

The same continuous definitions are sampled at ROBERT/PICASO edges and at
ROBERT/pRT cell centres, avoiding profile interpolation as a hidden source of
difference.

## Primary 80-cell comparison

Across the four profiles, ROBERT versus pRT has a worst R=100 spectral p95
symmetric relative difference of `1.30%`, a maximum eclipse-depth difference
of `7.80 ppm`, a worst contribution-centroid RMS difference of `0.0462 dex`,
and a worst contribution-profile p95 total-variation distance of `0.0573`.
The isothermal spectra agree to p95 `1.19e-6` symmetric relative difference,
which isolates the remaining non-isothermal difference to vertical source and
opacity discretization rather than normalization.

PICASO differs from ROBERT/pRT at the primary grid by spectral p95 values of
roughly `71--106%` depending on profile and pair.  Contribution-centroid RMS
differences are about `0.38--0.47 dex`.  This is consistent with the native
opacity attribution in Stage 3, not evidence of a new radiative-transfer
failure.

## Contribution definitions

- ROBERT reports its native disk-integrated layer source decomposition.  The
  blackbody lower-boundary term is folded into the deepest pressure cell before
  normalization.
- pRT reports its native normalized `emission_contribution` diagnostic on the
  supplied pressure nodes.
- PICASO exposes native total optical depths but not source-decomposed SH4
  layer fluxes through its public spectrum output.  Its Stage-4 contribution is
  therefore an independently implemented pure-absorption formal solution
  applied to PICASO's native gas, continuum, cloud, and Rayleigh optical-depth
  arrays (the latter two are zero or negligible in this clear case).  Stage 1
  already validates this absorbing RT limit across all three codes.

All contribution arrays are conservatively binned to the common R=100
wavelength grid and normalized independently at every wavelength.  Pairwise
metrics include pressure-centroid and peak offsets plus total-variation
distance between complete vertical profiles.

## Vertical convergence

From 80 to 160 cells/nodes:

- ROBERT's worst spectral p95 difference is `0.102%`, its maximum
  eclipse-depth change is `0.605 ppm`, its worst contribution-centroid RMS
  change is `0.000457 dex`, and its worst contribution p95 total variation is
  `0.000973`.
- PICASO's corresponding values are `0.201%`, `1.86 ppm`, `0.00183 dex`, and
  `0.00711`.
- pRT's corresponding values are `0.654%`, `0.872 ppm`, `0.0211 dex`, and
  `0.0272`.

The ROBERT--pRT contribution-centroid RMS difference decreases from at most
`0.0884 dex` at 40 cells to `0.0462 dex` at 80 and `0.0254 dex` at 160.  This
near factor-of-two convergence is the expected signature of aligning ROBERT
cell contributions with pRT node contributions at progressively finer grids.

## Reproduction

Run from the ROBERT environment:

```bash
conda activate robert-exoplanets
python examples/benchmark_emission_intercomparison_stage_4.py
```

The versioned summary is
`docs/data/emission_intercomparison/stage_4_report.json`.  Detailed native and
common-grid NPZ artifacts are written beneath
`examples/outputs/emission_intercomparison/stage_4/` and remain ignored by
Git.
