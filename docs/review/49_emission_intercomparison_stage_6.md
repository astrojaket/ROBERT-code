# Emission intercomparison: Stage 6

## Predeclared methods and acceptance gates

This section was written before the full Stage-6 matrix was run.  It fixes the
method and Track-A acceptance criteria independently of the observed results.

Stage 6 uses symmetric localized perturbations in log10 volume mixing ratio,

`q_s+/- (P) = q_s(P) 10^[+/- Delta L(P;P0)]`,

with `Delta=0.10 dex` and the Stage-5 localization
`L=exp[-0.5 (log10(P/P0)/0.35 dex)^2]` at `1e-4`, `1e-3`, `1e-2`, `1e-1`,
`1`, and `10 bar`.  The non-target molecular VMRs remain fixed.  H2 and He
are recomputed from the remaining atmosphere in an 85:15 ratio at every
pressure, after which the complete composition is explicitly checked to sum
to unity.  Mean molecular weight and H2-H2/H2-He CIA are recomputed from each
case composition.

The full 40/80/160-cell matrix covers H2O, CO, CO2, and CH4 for all four
Stage-4 temperature profiles; 80 cells is primary.  The primary-resolution
linearity audit additionally fixes `Delta=0.05`, `0.10`, and `0.20 dex` for
the monotonic and retrieved-like profiles.  It reports differences between
the central-difference Jacobians rather than assuming convergence.

Track A accepts only if every predeclared limit below is met.  The primary
cross-code limits apply across all profiles, species, and code pairs; the
80-to-160 limits apply across all profiles, species, and frameworks.

| Diagnostic | Primary limit | 80-to-160 limit |
| --- | ---: | ---: |
| R=100 signed-Jacobian p95 absolute difference / pair peak | 0.05 | 0.05 |
| Eclipse-Jacobian RMS difference | 0.50 ppm/dex | 0.50 ppm/dex |
| Pressure-centroid RMS difference | 0.15 dex | 0.15 dex |
| Full vertical-response p95 total variation | 0.08 | 0.08 |
| Cross-species-fraction p95 total variation | 0.08 | 0.08 |

Track A obtains case-specific plus/minus total optical depths from ROBERT's
independent loader of the shared source HDF opacity tables, including the
renormalized composition and CIA, and supplies those identical optical depths
and temperatures to all three pure-absorption RT paths.  Track B instead has
each framework recompute its own molecular opacity, correlated-k mixing, mean
molecular weight, and CIA.  ROBERT and pRT independently load the same source
HDF tables.  PICASO uses its official independent SQLite opacity database and
therefore has no inappropriate native cross-code acceptance gate.

## Results

The complete main matrix and linearity audit ran in `1358.6 s` using the
absolute interpreters documented below.  All unchanged, predeclared Track-A
gates pass.

Post-run metric validation identified one numerical-analysis defect before
the results were finalized.  The isothermal atmosphere with a blackbody lower
boundary has an analytically zero composition derivative, but raw subtraction
left cancellation values of only `3.8e-5` to `9.5e-5` flux units.  Normalizing
that noise created arbitrary one-hot pressure responses.  The analysis now
sets central-difference values at or below
`32 epsilon max|F_case|/Delta` to exact zero before normalization.  This floor
is defined from machine precision, the measured case-flux scale, and the
finite-difference step; it does not use observed cross-code differences.  The
raw worker fluxes remain checksummed, and none of the predeclared acceptance
limits was changed.

## Track A: shared abundance-induced optical depth

At the 80-cell primary resolution, the worst result over all four profiles,
all four species, and all code pairs is:

- signed R=100 Jacobian p95 absolute difference divided by pair peak:
  `0.410%` (gate `5%`);
- eclipse-depth Jacobian RMS difference: `0.225 ppm/dex` (gate
  `0.50 ppm/dex`);
- pressure-centroid RMS difference: `0.00341 dex` (gate `0.15 dex`);
- full vertical-response p95 total variation: `0.00541` (gate `0.08`);
- cross-species-fraction p95 total variation: `0.000593` (gate `0.08`).

The corresponding worst 80-to-160 values are `0.218%`, `0.115 ppm/dex`,
`0.00494 dex`, `0.00273`, and `0.000365`.  Every Track-A gate therefore
passes with substantial margin.  The identical case-specific total optical
depths successfully isolate RT differentiation from opacity construction.

## Track B: native composition-dependent opacity

At 80 cells, ROBERT versus pRT has worst values of `1.24%` for the signed
Jacobian p95 metric, `0.363 ppm/dex` for the eclipse-Jacobian RMS difference,
`0.0605 dex` for pressure-centroid RMS, `0.0651` for p95 vertical-response
total variation, and `0.00941` for p95 cross-species-fraction total variation.
ROBERT and pRT independently load the same source HDF tables, so this is the
relevant native-opacity same-basis comparison.

PICASO remains an opacity-database attribution result.  Pairs containing
PICASO reach `72.1%` in the p95 signed-Jacobian metric, about
`11.3 ppm/dex` in eclipse-Jacobian RMS, `1.76 dex` in pressure-centroid RMS,
and `0.50` in p95 vertical-response total variation.  These numerical results
are retained in full; no inappropriate PICASO pass/fail gate is imposed.

## Vertical convergence

Native Track-B 80-to-160 convergence is strong for all frameworks:

| Framework | Jacobian p95 | Eclipse RMS (ppm/dex) | Centroid RMS (dex) | Response TV p95 | Species-fraction TV p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROBERT | 0.204% | 0.0818 | 0.00176 | 0.00456 | 0.000763 |
| PICASO | 0.282% | 0.113 | 0.00661 | 0.0109 | 0.00261 |
| pRT | 0.481% | 0.0757 | 0.00221 | 0.00459 | 0.000987 |

The corresponding worst 40-to-80 Jacobian p95 values are `0.819%` for
ROBERT, `1.12%` for PICASO, and `1.79%` for pRT.  The report retains the full
40-to-80 and 80-to-160 results for signed spectra, eclipse depth, pressure
centroids, peak pressures, complete vertical profiles, and cross-species
fractions.

## Finite-difference linearity

The 80-cell audit covers all four molecules for the monotonic and
retrieved-like profiles.  Relative to the `0.10 dex` main derivative, the
worst p95 truncation/nonlinearity difference at `0.05 dex` is below `0.096%`
for every framework and track.  At `0.20 dex`, the worst p95 value is
`0.379%` in Track A and `0.312%` in Track B.  Worst relative-RMS differences
are `0.275%` at `0.05 dex` and `1.09%` at `0.20 dex` for Track A; the native
Track-B maxima are `0.213%` and `0.766%`.  The observed step-size dependence
is small but explicitly reported rather than treating `0.10 dex` as exact.

## Relation to Stage 4 and Stage 5

The response artifact carries the Stage-4 contribution projections and the
Stage-5 temperature responses alongside every Stage-6 composition response,
and the JSON report compares their centroids, peaks, and complete normalized
profiles species by species.

These are three different diagnostics.  A contribution function decomposes
emergent intensity by pressure.  A temperature Jacobian differentiates the
source function and, in the native track, temperature-dependent opacity.  A
composition Jacobian differentiates abundance and may change sign; it also
contains opacity redistribution, gas-overlap, CIA, mean-molecular-weight, and
background H2/He effects.  Similar pressure support is physically useful, but
the functions are not expected to be identical.

## Reproduction and records

Run from the ROBERT environment:

```bash
/opt/miniconda3/envs/robert-exoplanets/bin/python \
  examples/benchmark_emission_intercomparison_stage_6.py
```

PICASO runs with `/opt/miniconda3/envs/picaso/bin/python`; stable pRT runs with
`/opt/miniconda3/envs/petitradtrans-stable/bin/python`.  The report records
those absolute paths, package versions, the known PICASO reference warning,
all method and grid contracts, source/data/output checksums, raw case timings,
and summarized timings.

The versioned outputs are
`docs/data/emission_intercomparison/stage_6_report.json` and
`docs/data/emission_intercomparison/stage_6_response_tensors.npz`.  Detailed
process-isolated contracts and worker outputs remain under the ignored
`examples/outputs/emission_intercomparison/stage_6/` directory.

The post-Stage-6 programme is fixed in
`docs/emission_intercomparison_roadmap.md`: Stage 7 covers absorbing clouds,
Stage 8 covers cloud scattering and solver order, and Stage 9 performs both
cloud-free and cloudy directed cross-retrievals.  Stage-9 files and smoke runs
are prepared locally before the full sampler matrix is pulled onto and run on
Glamdring.
