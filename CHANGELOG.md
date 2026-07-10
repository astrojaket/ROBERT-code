# Changelog

## v0.3.0 - Minimal Forward Model Foundation

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
  `ClearSkyEmissionForwardModel`; HAT-P-32b scripts now only assemble target
  inputs, opacity, priors, and outputs around the public package model.
- Added a Python-first `ClearSkyEmissionFactoryConfig`, typed ExoMol/exo_k
  opacity-source and binning configuration, opacity-derived pressure grids,
  factory provenance, and a standalone HAT-P-32b target configuration.
- Added Python-first complete retrieval-run configuration for optimal
  estimation and UltraNest, runtime temperature/chemistry emission models, a
  retrieval-tested FastChem adapter, and an eight-parameter HAT-P-32b
  FastChem/Madhusudhan-Seager comparison configuration.
- Added invalid-physics backtracking to diagnostic optimal estimation so trial
  P–T states outside opacity coverage do not crash a configured retrieval.
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
- Added a NumPy clear-sky thermal-emission reference solver with Planck
  source-function integration, optional disk averaging, eclipse-depth output,
  and layer contribution diagnostics.
- Added a local HAT-P-32b clear-sky emission benchmark script that compares the
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
