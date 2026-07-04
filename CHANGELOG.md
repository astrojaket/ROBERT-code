# Changelog

## v0.3.0 - Minimal Forward Model Foundation

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
- Added a v0.3 audit and next-build plan grounded in the NemesisPy HAT-P-32b
  workflow.
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
  ExoMol/ExoMolOP, exo_k/NEMESIS `.kta`, HITRAN `.par`, HITRAN CIA, and future
  ROBERT compressed archive manifests.
- Added opacity-array benchmark diagnostics for future absorption and
  k-coefficient validation across wavelength, pressure, temperature, and
  g-ordinate axes.
- Added ROBERT-native opacity archive helpers for readable-manifest `.npy`
  directories and `.npz` exchange archives, plus a synthetic opacity archive
  I/O benchmark example.

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
