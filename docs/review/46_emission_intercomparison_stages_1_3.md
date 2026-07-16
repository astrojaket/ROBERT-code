# Emission intercomparison: Stages 1--3

## Outcome

The first three emission-intercomparison stages have been rebuilt and rerun
with strict process isolation.  ROBERT runs in `robert-exoplanets`, PICASO in
`picaso`, and the stable petitRADTRANS reference in
`petitradtrans-stable`.  See `docs/emission_intercomparison_environments.md`
for the reproducible environment and data contract.

Stage 1 passes its analytic grey/isothermal acceptance gates.  Stage 2 passes
at the 80-layer primary resolution after the expected adaptive convergence
from 20 and 40 layers.  Stage 3 is a native-opacity characterization and does
not impose a cross-code pass/fail gate.

## Stage 1: grey/isothermal Track A

- Temperatures: 500, 1000, 1500, and 2000 K.
- Column optical depths: 1e-4, 1e-2, 1, and 100.
- Layer grids: 16, 32, 64, and 128.
- Eight-angle Gauss--Legendre disk integration.
- Shared optical-depth arrays are supplied independently to all three RT
  solvers.

Across 64 cases, the maximum pairwise symmetric relative difference is
`1.91e-5`, the maximum analytic symmetric relative difference is `1.91e-5`,
and the maximum eclipse-depth difference is `0.00137 ppm`.  All gates pass.

## Stage 2: single-molecule/shared-tau Track A

- Molecules: H2O, CO, CO2, and CH4.
- Temperature anchors: 500, 1000, 1500, and 2000 K, each with a 300 K
  top-to-bottom gradient.
- Volume mixing ratios: 1e-6 through 1e-1 in six logarithmic steps.
- Layer grids: 20, 40, and 80.
- 288 total cases.

The worst per-case p95 symmetric relative difference converges from `6.15%`
at 20 layers to `1.58%` at 40 layers and `0.399%` at the 80-layer primary
grid.  The primary-grid threshold is `0.5%`.  The maximum eclipse-depth
difference over the full matrix is `2.02 ppm`, and the worst 40-to-80-layer
convergence difference is `0.38 ppm`; both are below the `3 ppm` thresholds.

## Stage 3: four-molecule+CIA Track B

- Fixed VMRs: H2O `1e-3`, CO `3e-4`, CO2 `1e-4`, and CH4 `1e-5`.
- The remaining gas is H2/He in an 85:15 ratio.
- H2--H2 and H2--He collision-induced absorption is included.
- Layer grids: 40, 80, and 160; the 80-layer result is primary.
- Spectra are conservatively compared on a common R=100 wavelength grid.

ROBERT and pRT use the same source HDF correlated-k tables through independent
loaders and agree at the primary resolution to p95 `0.145%`, with a maximum
eclipse-depth difference of `1.25 ppm`.  PICASO uses its independent official
SQLite opacity database; its p95 differences from ROBERT/pRT are about `85%`
and reach about `80 ppm`.  This is an expected Stage-3 attribution result, not
an RT failure: Stage 1--2 already isolate and validate the RT path.

All three native solvers are converged from 80 to 160 layers to p95 below
`0.083%`; the maximum 80-to-160 eclipse-depth changes are `0.096 ppm` for
ROBERT, `0.78 ppm` for PICASO, and `0.020 ppm` for pRT.

## Reproduction

Run all stages from the ROBERT environment:

```bash
conda activate robert-exoplanets
python examples/benchmark_emission_intercomparison_stages_1_3.py
```

The launcher records versioned reports under
`docs/data/emission_intercomparison/` and detailed ignored NPZ artifacts under
`examples/outputs/emission_intercomparison/`.  Stage 4 thermal-structure and
contribution-function results are documented in
`docs/review/47_emission_intercomparison_stage_4.md`.
