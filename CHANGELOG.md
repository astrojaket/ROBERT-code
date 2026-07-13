# Changelog

## Unreleased

- Generalized the active emission API from the legacy `ClearSky*` and
  `solve_clear_sky_emission*` names to `Emission*` and `solve_emission*`.
  Backward-compatible aliases remain for historical scripts.
- Designated PICASO and petitRADTRANS as the maintained gold-standard
  forward-model benchmarks and standardized their plots on a shared purple
  palette with `mediumpurple` for ROBERT model spectra.
- Archived the pre-YAML HAT-P-32b checks under
  `examples/Depreciated_Benchmarks/`; they are no longer part of active CI or
  the maintained benchmark workflow.

## v0.3.0 - Minimal Forward Model Foundation

- Added an independent optional float64 JAX/XLA implementation of conservative
  random-overlap recompression with explicit CPU/GPU device selection and no
  silent precision or platform fallback.
- Added a diagnostics-free fused correlated-k assembly path that scales
  k-coefficients and performs random overlap in one Numba kernel without
  materializing the species optical-depth cube; the species-resolved path
  remains the scientific reference.
- Classified ExoMol opacity sampling as a functional beta backend while
  retaining correlated-k as the validated retrieval default.
- Added sampler-independent retrieval problems, typed priors, coordinate-aware
  Gaussian likelihood evaluation, optimal estimation, and an optional
  UltraNest adapter.
- Added accuracy-preserving correlated-k spectral binning through `exo_k` using
  explicit observation bin edges; individual k coefficients are not
  wavelength-interpolated.
- Delegated zero-coefficient replacement to `exo_k.Ktable.remove_zeros` before
  binning, with replacement provenance, and added path-independent discovery
  of arbitrary molecular ExoMol/exo_k KTA products.
- Added pre-inference run manifests and stable JSON/NPZ retrieval results with
  configuration hashes, opacity identifiers, runtime provenance, and seeds.
- Hardened immutable data containers, pressure/spectral grid invariants,
  opacity archive manifest integrity, and explicit CIA extrapolation policy.
- Added deterministic injection-recovery validation contracts and a multi-gas
  HAT-P-32b-like H2O/CO2/NH3 RT case with explicit synthetic bins, seeded
  noise, convergence gating, recovery tolerances, and versioned reports.
- Promoted the validated retrieval physics into a reusable typed
  `EmissionForwardModel`; HAT-P-32b scripts now only assemble target
  inputs, opacity, priors, and outputs around the public package model.
- Added a Python-first `EmissionFactoryConfig`, typed ExoMol/exo_k
  opacity-source and binning configuration, opacity-derived pressure grids,
  factory provenance, and a standalone HAT-P-32b target configuration.
- Added Python-first complete retrieval-run configuration for optimal
  estimation and UltraNest, runtime temperature/chemistry emission models, a
  retrieval-tested FastChem adapter, and an eight-parameter HAT-P-32b
  FastChem/Madhusudhan-Seager comparison configuration.
- Added invalid-physics backtracking to diagnostic optimal estimation so trial
  P–T states outside opacity coverage do not crash a configured retrieval.
- Added a thermal P3/SH4 multiple-scattering backend following Rooney, Batalha
  & Marley, including explicit HG phase moments, optional delta-M scaling,
  stable banded multilayer boundary solves, source-function angular
  reconstruction, PICASO parity benchmarks, and retrieval-scale speed plots.
- Added an RT backend selection guide covering transmission, emission,
  reflected light, scattering order, and simple versus microphysical clouds.
- Added a controlled six-molecule ExoMolOP/exo_k emission comparison that sends
  an identical 20-g optical-depth cube to ROBERT and PICASO, with sub-0.06%
  maximum disk-integrated RT differences and molecular contribution plots.
- Added exact linear-in-optical-depth clear thermal source integration to both
  NumPy and Numba backends, reducing the molecular comparison RMS residual.
- Added absorption-dominated spherical transmission with exact shell chords,
  correlated-k impact-area integration, analytic limits, and a realistic
  FastChem molecular+CIA HAT-P-32b convergence benchmark.
- Vendored the NemesisPy v1.0.1 CIA reference table with BSD-3-Clause license,
  commit and checksum provenance; integrated CIA into parameterized retrievals.
- Added endpoint-inclusive configurable pressure grids, explicit
  NemesisPy-compatible opacity-boundary clipping, MPI-safe result writing, a
  finite UltraNest invalid-model floor, and four comparison plot products.
- Bundled the complete HAT-P-32b reference case for clone-local execution:
  saved NemesisPy products, six exo_k-binned opacity archives, FastChem input
  data and license, checksums, and reproducible generation provenance.
- Added a pinned cross-platform Conda environment, complete package extra,
  fresh-environment validation, and a configurable Slurm/UltraNest batch
  script with MPI launch guidance.
- Hardened long UltraNest runs for production clusters with resumable defaults,
  stable Slurm run directories, concurrent-writer locks, preserved initial and
  per-attempt manifests, live status/throughput records, checkpoint smoke tests,
  and pre-emption-aware automatic requeueing.
- Added a minimal Glamdring `addqueue` launcher following the official cluster
  contract: one Python invocation per allocated Slurm rank, no nested MPI
  launcher, and explicit memory and checkpoint-restart guidance.
- Added cloud-type-agnostic refractive-index retrievals and homogeneous-sphere
  Mie physics: tabulated/CSV/Exo-Skryer optical-constant readers, lognormal
  particle-size averaging, mass extinction and scattering, condensate mass
  fraction profiles, direct nodal `n`/`log10(k)` emission parameters, and exact
  scalar Legendre phase moments through degree four for SH4/delta-M transport.
- Added a commit-pinned, AGPL-separated snapshot of Exo Skryer's optical-
  constant catalogue with 44 selectable physical/reference materials,
  preserved source headers, checksums, duplicate-splice handling, and licence
  provenance.
- Added analytic one- and two-layer thermal cloud-scattering validation cases,
  plus pressure-resolved cloud-extinction and contribution maps in the
  PICASO/Virga-style benchmark output.
- Added a subprocess-isolated PICASO Toon reference runner and controlled
  grey-cloud RT comparison, then replaced the effective-extinction shortcut
  with a coupled, stably scaled hemispheric-mean thermal source-function solve.
  Controlled disk-integrated scattering cases agree with PICASO within 0.72%.
- Added explicit pressure-edge temperatures for linear-in-optical-depth Planck
  sources, Rutten formal-solution and Taylor et al. (2021) analytic-limit tests,
  physical extinction diagnostics, and a batched block-tridiagonal solver that
  reduces the 900-wavelength/four-g benchmark from 4.4 s to about 0.24 s.
- Added reproducible stable petitRADTRANS 3.3.3 benchmarks over 0.3--12 micron,
  including exact pRT correlated-k quadrature, explicit mass-to-volume mixing
  ratio conversion, six molecular absorbers, H2-H2/H2-He CIA, and H2/He
  Rayleigh extinction. Corrected gas-mixture Rayleigh physics to add the
  number-weighted molecular cross-sections rather than mixed refractivities.
  The six-molecule thermal-emission RT isolation agrees with pRT3 to 0.140%
  RMS, with contribution functions reproducing the same photospheric structure.
- Added native petitRADTRANS HDF5 CIA loading for H2-H2 and H2-He with explicit
  provenance and selectable log-coefficient temperature interpolation. A fully
  independent six-line-species plus CIA HDF emission run agrees with stable
  pRT3 to 0.108% RMS on matched 80-cell/81-node grids and shows regular
  pressure-grid convergence through 160 cells.
- Added the published 280-point WASP-69b JWST NIRCam+MIRI eclipse spectrum with
  checksummed VizieR provenance; exact two-region areal emission mixing;
  single-region dayside dilution; flux-conserving top-hat observation-bin
  integration; and named multi-dataset forward, likelihood, nuisance-offset,
  and retrieval contracts for the Schlawin et al. (2024) benchmark.
- Added a planet-independent disk-emission selector that composes ROBERT's
  existing regional forward models as one-region, diluted one-region, or
  two-region configurations, plus a parameterized gray-scattering regional
  model whose layer optical depths follow the hydrostatic mass column and use
  the validated SH4 multiple-scattering backend by default.

- Added atmospheric state, isothermal temperature, constant chemistry, and
  atmosphere-builder components.
- Added a zero-opacity fixture provider for pipeline wiring only.
- Added linear observation-grid response and independent Gaussian likelihood.
- Added a placeholder forward-model pipeline that returns native and observed
  spectra without claiming physical radiative-transfer behavior.
- Added a runnable minimal forward-model example.
- Added blackbody reference diagnostics and a plotting example for visual
  emission-spectrum sanity checks.
- Added a GitHub Actions CI workflow.
- Added a local-only HAT-P-32b benchmark plotting example and reusable external
  emission benchmark CSV loader.
- Added a v0.3 audit and next-build plan grounded in the local HAT-P-32b
  benchmark workflow.
- Added tabulated temperature-profile support and a local HAT-P-32b P-T
  diagnostic plotting example.
- Added retrieval-facing spline, Madhusudhan-Seager 2009, and
  Parmentier-Guillot 2014 style temperature-profile parameterizations.
- Added retrieval-facing free chemistry with explicit H2/He background-fill
  support and a VMR plotting diagnostic.
- Added optional composition-derived mean-molecular-weight calculation for
  complete VMR chemistry outputs.
- Added lightweight timing diagnostics and an atmosphere-build benchmark
  example for performance smoke testing.
- Added a model-setup factory that maps HAT-P-32b-style atmosphere config
  blocks into ROBERT pressure, temperature, chemistry, and MMW components.
- Added opacity metadata, coverage checks, and lightweight inspectors for
  ExoMol/ExoMolOP, exo_k `.kta`, HITRAN `.par`, HITRAN CIA, and future
  ROBERT compressed archive manifests.
- Added opacity-array benchmark diagnostics for future absorption and
  k-coefficient validation across wavelength, pressure, temperature, and
  g-ordinate axes.
- Added ROBERT-native opacity archive helpers for readable-manifest `.npy`
  directories and `.npz` exchange archives, plus a synthetic opacity archive
  I/O benchmark example.
- Added a validated `.kta` reader for ExoMolOP/exo_k correlated-k
  products, including header coverage metadata, full k-coefficient loading, and
  conversion into ROBERT native archives. The reader can optionally replace
  non-finite k-coefficients with an in-memory runtime floor while preserving the
  source table unchanged and recording replacement metadata.
- Added a native-grid correlated-k opacity evaluator that produces
  species-by-layer k-coefficients for exact pressure, temperature, wavelength,
  and g-ordinate benchmark cases, plus optional pressure-temperature
  interpolation in log-k.
- Added a local HAT-P-32b opacity benchmark example that compares exact
  evaluator slices against native `.kta` table values and writes diagnostic
  opacity plots, including missing-table-region diagnostics.
- Added gas optical-depth assembly from evaluated correlated-k opacity, plus
  cumulative tau, transmission, and layer weighting diagnostics for future
  RT contribution plots.
- Added a synthetic tau and transmission-weighting plotting example.
- Added a NumPy cloud-free thermal-emission reference solver with Planck
  source-function integration, optional disk averaging, eclipse-depth output,
  and layer contribution diagnostics.
- Added a local HAT-P-32b cloud-free emission benchmark script that compares the
  current gas-only ROBERT spectrum against an external benchmark and records
  remaining benchmark physics gaps explicitly.
- Added CIA/Rayleigh layer optical-depth helpers, random-overlap gas mixing,
  phase-aware Lobatto disc geometry, and a first diagnostic direct-beam
  single-scattering source treatment.
- Added a v0.3 RT benchmark audit with default, phase-geometry, and
  single-scattering benchmark results.
- Added hydrostatic radius/path geometry anchored at a reference pressure,
  optional spherical shell path factors in the emission solver, and a benchmark
  note showing that the HAT-P-32b comparison still matches best with the
  plane-parallel path default.
- Accelerated repeated forward-model calls by caching immutable log-k tables
  and prepared spectral indices, using a streaming heap merge for sorted
  correlated-k random overlap, and simplifying fixed SH4 source contractions.
  Representative six-molecule WASP-69b calls are 3.1--5.3x faster without
  changing their likelihoods; a deterministic random-overlap scaling benchmark
  and warmed `cProfile` support were added.
- Added an exact shared-atmosphere multi-dataset emission path that evaluates
  temperature, chemistry, mean molecular weight, and atmospheric state once
  while retaining mode-specific correlated-k opacity and RT. WASP-69b spectra
  and likelihoods remain bit-for-bit identical, with warmed three-mode calls
  about 12% faster and four-mode calls about 21% faster.
- Made shared-atmosphere orchestration the default public multi-instrument
  factory flow. The native-spectrum-then-bin diagnostic now has an explicit
  `NativeSpectrumMultiDatasetForwardModel` name, while the deliberately
  repeated-atmosphere path exists only inside the performance benchmark.
- Removed the unused v0.1 `Observation.validate()` compatibility no-op;
  observations already validate all invariants during construction.
- Added explicit WASP-69b `--dlogz` and provenance-rich `--fast-stop`
  controls for exploratory checkpoint finalization without changing scientific
  defaults, plus a complete clear-model plotting and published-comparison
  workflow that reports posterior effective sample size.

## v0.2.0 - Core Domain Foundation

- Renamed the target import namespace to `robert_exoplanets`.
- Added core grid and spectrum containers.
- Added `Planet` and `Star` body models.
- Added an instrument-facing `Observation` container.
- Added repository hygiene files.
- Kept the stub retrieval workflow runnable while no real physics is implemented.

## v0.1.0 - Skeleton

- Created the initial ROBERT repository skeleton.
- Added a stub retrieval workflow, tests, and architecture documents.
