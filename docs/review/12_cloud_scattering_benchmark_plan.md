# Cloud and Scattering Benchmark Plan

Date: 2026-07-05

ROBERT now has the first cloud/aerosol RT inputs and a conservative two-stream
reference backend. The next priority is benchmarking, not making the closure
more complicated immediately.

## Current Implementation

- `CloudOpticalProperties` stores layer-by-wavelength extinction optical depth,
  single-scattering albedo, and asymmetry factor.
- Clouds can be passed directly to `solve_clear_sky_emission(...)` or split into
  absorption and scattering `LayerOpticalDepth` contributors.
- `grey_cloud_deck(...)` and `power_law_haze(...)` provide small retrieval-ready
  parameterizations for early tests.
- `multiple_scattering_backend="two_stream"` activates the first conservative
  two-stream effective-extinction closure.
- `ClearSkyEmissionResult.extinction_optical_depth` preserves physical
  extinction tau when `total_optical_depth` records the solver's effective tau.

This is a reference hook. It should not be treated as validated cloud
scattering until the benchmark ladder below passes.

## Benchmark Ladder

### 1. Analytic Limits

Use small one-layer and two-layer atmospheres where the expected answer can be
written directly:

- no cloud and zero scattering;
- pure absorbing grey cloud;
- pure scattering grey cloud;
- mixed absorption/scattering cloud;
- optically thin and optically thick limits;
- isotropic scattering (`g=0`) and increasingly forward-scattering cases.

These tests should verify both spectra and diagnostics: extinction tau,
effective tau, single-scattering albedo, layer contribution, and metadata.

### 2. PICASO/Virga Optical-Property Parity

Before comparing spectra, compare the cloud optical properties themselves:

- pressure grid orientation and interpolation;
- wavelength grid and units;
- total extinction optical depth;
- single-scattering albedo;
- asymmetry factor;
- condensate species labels and provenance;
- later, phase-function moments or tabulated phase functions.

The ROBERT cloud object was designed around the same minimal set that a
PICASO/Virga-style pipeline naturally produces: `tau_ext`, `omega0`, and `g`.

### 3. Code-to-Code RT Cases

Run identical one-dimensional atmospheres through ROBERT and PICASO:

- clear atmosphere baseline;
- Rayleigh-only atmosphere;
- grey absorbing deck;
- grey scattering deck;
- power-law haze;
- Virga-generated MgSiO3/Mg2SiO4-like cloud columns when the adapter exists.

Each case should produce plots of the spectrum, residuals, tau, omega0, `g`,
and pressure-resolved contribution/weighting functions.

### 4. Published Science Cases

Useful literature anchors:

- PICASO reflected-light scattering benchmarks:
  [Batalha et al. 2019](https://arxiv.org/abs/1904.09355).
- Cloudy thermal phase curves:
  [Robbins-Blanch et al. 2022](https://arxiv.org/abs/2204.03545).
- PICASO plus Virga reflected phase curves:
  [Hamill et al. 2024](https://arxiv.org/abs/2411.14225).
- Virga condensation-cloud model benchmarks:
  [Batalha et al. 2025](https://arxiv.org/abs/2508.15102).
- PICASO 4.0 cloud/climate benchmarks:
  [Mang et al. 2026](https://arxiv.org/abs/2602.22468).

## Acceptance Criteria

- Analytic tests pass at tight numerical tolerance.
- ROBERT ingests or reproduces PICASO/Virga cloud optical properties without
  unit, orientation, or interpolation ambiguity.
- Simple code-to-code grey/haze cases agree within documented tolerance after
  matching geometry, boundary conditions, opacity, and scattering convention.
- Disagreements in published-case benchmarks are decomposed into cloud optical
  properties, RT geometry, scattering closure, opacity, and chemistry terms.

Only after this should the two-stream backend be promoted from reference hook to
validated cloudy RT path, or replaced by a fuller multiple-scattering solver.
