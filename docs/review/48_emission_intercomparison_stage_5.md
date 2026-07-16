# Emission intercomparison: Stage 5

## Outcome

Stage 5 now compares localized thermal response functions and numerical
temperature Jacobians for the Stage-4 isothermal, monotonic, inverted, and
retrieved-like profiles.  The complete 40/80/160-cell matrix ran in strict
framework isolation using `robert-exoplanets` for ROBERT, `picaso` for PICASO
3.2.2, and `petitradtrans-stable` for petitRADTRANS 3.3.3.  The finalized run
completed in `293.1 s` and passed every predeclared shared-optical-depth Track-A
gate.

The vertical-grid contract is unchanged: ROBERT uses 40, 80, and 160 pressure
cells; PICASO receives the matching 41, 81, and 161 cell edges as atmospheric
levels; and pRT receives the ROBERT geometric cell centres as 40, 80, and 160
pressure nodes.  The 80-cell result is primary.

## Perturbation and Jacobian definitions

Six perturbation centres are fixed at `1e-4`, `1e-3`, `1e-2`, `1e-1`, `1`,
and `10 bar`.  At centre pressure `P0`, the applied shape is

`L(P;P0) = exp[-0.5 (log10(P/P0) / 0.35 dex)^2]`.

Each framework is run at `T + 10 K L` and `T - 10 K L`.  The reported
temperature Jacobian is the symmetric difference

`J(P0, lambda) = [F(T + 10 L) - F(T - 10 L)] / 20 K`.

The continuous perturbation is sampled independently on PICASO/ROBERT cell
edges and ROBERT/pRT geometric cell centres, just as the unperturbed Stage-4
profiles were.  Flux Jacobians are conservatively binned to R=100 before the
comparison.  Eclipse-depth Jacobians use the same 5800 K stellar blackbody and
planet/star radius ratio as Stages 1--4 and are reported in ppm/K.

For each wavelength, the absolute Jacobian is normalized over all six centres.
These complete normalized response profiles, the signed flux Jacobians, and
the eclipse-depth Jacobians are versioned in
`docs/data/emission_intercomparison/stage_5_response_profiles.npz` rather than
being reduced to only centroids or peaks.  Pairwise metrics compare pressure
centroids, peak pressures, and total-variation distance between the complete
profiles.

## Track A: shared optical depth

Track A freezes the baseline ROBERT gas+CIA total optical depth.  Its
correlated-k optical depth is g-weighted independently in every layer and
wavelength bin, then the identical frozen field is supplied to the three
pure-absorption RT paths for every positive and negative temperature
perturbation.  This deliberately isolates source-function and RT response; it
is not a native-opacity calculation.

At the 80-cell primary resolution, the worst results over all four profiles
and all code pairs are:

- R=100 spectral-Jacobian p95 absolute difference divided by the pair peak:
  `0.226%` (gate `5%`);
- eclipse-Jacobian RMS difference: `0.00142 ppm/K` (gate `0.02 ppm/K`);
- response-centroid RMS difference: `0.00263 dex` (gate `0.15 dex`);
- response-profile p95 total variation: `0.00282` (gate `0.08`).

The corresponding worst 80-to-160 results are `0.156%`, `0.000749 ppm/K`,
`0.00120 dex`, and `0.00237`; all pass their respective Track-A gates.  Peak
pressures are quantized onto the six perturbation centres, so the report also
retains the full profiles and continuous log-pressure centroids.

## Track B: native temperature-dependent opacity

Track B recomputes each framework's molecular and CIA opacity for every
positive and negative perturbation.  ROBERT and pRT continue to use the same
source HDF correlated-k tables through independent loaders.  PICASO uses its
independent official SQLite opacity database.

At 80 cells, ROBERT versus pRT has worst values over the four profiles of
`0.636%` for the R=100 spectral-Jacobian p95 metric, `0.00366 ppm/K` for the
eclipse-Jacobian RMS difference, `0.00588 dex` for response-centroid RMS, and
`0.0113` for p95 response total variation.  Pairs containing PICASO reach
about `19.5%`, `0.0778 ppm/K`, `0.282 dex`, and `0.288`, respectively.  As in
Stages 3--4, the PICASO difference is an opacity-database attribution result,
not an RT acceptance failure, and no inappropriate native cross-code gate is
imposed.

Native 80-to-160 convergence remains strong.  The worst per-framework values
are:

| Framework | Jacobian p95 | Eclipse RMS (ppm/K) | Centroid RMS (dex) | Response TV p95 |
| --- | ---: | ---: | ---: | ---: |
| ROBERT | 0.229% | 0.000601 | 0.00169 | 0.00243 |
| PICASO | 0.332% | 0.00740 | 0.0105 | 0.0155 |
| pRT | 0.231% | 0.000812 | 0.00145 | 0.00243 |

The report also quantifies 40-to-80 convergence.  Its worst native-Jacobian
p95 values are `0.790%` for ROBERT, `0.911%` for PICASO, and `0.854%` for pRT,
showing the expected improvement at 80-to-160.

## Relation to Stage-4 contribution functions

The Stage-4 baseline contribution functions are projected through the same
six localization kernels and normalized across perturbation centres.  Stage 5
then compares those projected profiles with the numerical temperature
responses for every framework, profile, resolution, and track.

The comparison must not be interpreted as an identity.  A contribution
function decomposes emergent intensity, whereas a temperature Jacobian
contains the Planck derivative `dB/dT`; Track B additionally contains opacity
derivatives.  At 80 cells, the native Track-B ROBERT and pRT response profiles
remain fairly close to their projected contribution functions (worst p95
total variation `0.0598` and `0.0779`), while PICASO reaches `0.278`.  The
frozen-opacity Track-A comparison is intentionally more different because the
Stage-4 reference contribution functions retain each code's native opacity
and because source contribution is not Planck sensitivity; its worst p95 total
variation is about `0.80`.

## Reproduction and records

Run from the `robert-exoplanets` environment:

```bash
conda activate robert-exoplanets
python examples/benchmark_emission_intercomparison_stage_5.py
```

When opacity data are outside a fresh clone, pass `--picaso-reference`,
`--picaso-database`, and `--prt-input` explicitly.  The versioned summary is
`docs/data/emission_intercomparison/stage_5_report.json`.  It records the
interpreter and package version for each framework, method definitions,
contracts, source and output checksums, timings, pairwise results, and both
40-to-80 and 80-to-160 convergence.  Detailed process artifacts remain under
the ignored `examples/outputs/emission_intercomparison/stage_5/` directory.
