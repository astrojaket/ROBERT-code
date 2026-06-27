# Lessons Learned from NEMESIS and NemesisPy

Sources inspected:

- NEMESIS / Radtran: https://github.com/nemesiscode/radtrancode at commit `52c273f`.
- NemesisPy: https://github.com/astrojaket/NemesisPy at commit `afe2bae`.

## Successful Architectural Ideas

### Separate the Forward Model from the Retrieval Objective

NEMESIS has a recognizable separation between retrieval iteration (`coreret`) and forward-model calls (`forward*`, `cirsrtf*`, `cirsrad*`), even though the implementation is coupled through files and common blocks. This separation is worth preserving conceptually.

ROBERT should make it explicit:

```text
physical parameters -> atmosphere -> forward model -> observable -> likelihood
```

### Use Compact Parameterizations for Retrieval

Both codebases show that retrieval should not directly sample every atmospheric layer by default. Temperature, abundance, clouds, and calibration parameters need compact, scientifically motivated parameterizations.

ROBERT should preserve this, but replace integer model IDs and raw dict conventions with named, typed transforms.

### Keep Complete Reference Examples

NEMESIS example directories are extremely valuable because they include run files and outputs. They encode behavior that prose documentation cannot fully capture.

ROBERT should maintain small, versioned, automated reference cases from the beginning.

### Treat Opacity Preparation as a First-Class Workflow

Both codebases show that opacity data are not incidental. K-table coverage, rebinning, CIA coverage, stellar spectra, and cross-section data shape the entire retrieval.

ROBERT should make opacity preparation explicit, cached, validated, and reproducible.

### Preserve Diagnostic Retrieval Products

NEMESIS optimal estimation outputs covariance, averaging kernels, contribution/weighting information, and retrieval diagnostics. These are scientifically important.

ROBERT should not reduce retrieval output to posterior samples only.

## Unsuccessful Architectural Ideas

### Files as Internal Function Arguments

NEMESIS often communicates between stages by writing and reading sidecar files. This made sense historically, but it hides dependencies and makes tests fragile.

ROBERT should use structured in-memory objects internally and reserve files for user I/O, reproducibility, and fixtures.

### Global State as Shared Context

Common blocks in Fortran and import-time globals in Python both make behavior harder to isolate.

ROBERT should pass explicit context and make valid computational objects complete at construction time.

### One Executable per Variant

The many NEMESIS variants reduced risk during incremental development, but they created duplicated logic.

ROBERT should prefer one set of core algorithms with pluggable geometry, observation, and RT modes.

### Raw Dicts as Long-Term Domain Model

NemesisPy's modular layer is flexible, but raw dict access spreads schema rules throughout the code.

ROBERT should parse user config into typed, validated objects once.

### Samplers Embedded in Scripts

Sampler examples are useful, but placing MultiNest logic inside data/demo scripts prevents reuse.

ROBERT should make samplers adapters around a stable likelihood interface.

## Scientifically Robust Features

- Correlated-k radiative transfer with random overlap.
- Pressure/layer/path geometry handling.
- CIA treatment for H2/He-rich atmospheres.
- Planck/source-function emission integration.
- Optimal-estimation covariance and averaging-kernel diagnostics.
- Surface/lower-boundary handling for relevant Solar System cases.
- Scattering and cloud/aerosol concepts, especially in NEMESIS.
- Exoplanet disc integration and phase-curve concepts in NemesisPy.

These should be preserved through reference tests, not copied blindly.

## Technical Debt That Accumulated

### In NEMESIS

- Compile-time array sizes.
- Platform-specific build flags and record-length behavior.
- Large common-block state.
- Long argument lists combined with hidden state.
- Goto-based retrieval loop control.
- Integer-coded model registries.
- Many near-duplicate geometry/executable variants.
- Deep reliance on current working directory and run-name file families.

### In NemesisPy

- Committed build/cache artifacts.
- Multiple old/backup/test variants in importable package directories.
- Mutable backend object requiring a particular call sequence.
- Optional dependencies not cleanly declared or isolated.
- MPI, plotting, and I/O side effects in modules that should be data preparation only.
- Empty retrieval package with sampler logic in examples.
- Prototype comments, debugger imports, and TODO sections in scientific modules.

## Features That Complicated Future Development

- Implicit unit systems.
- Hidden interpolation/extrapolation policies.
- State support constraints embedded in retrieval loops.
- Combining chemistry, cloud, stellar, observation, and RT logic in broad methods.
- File-format-driven intermediate state.
- Multiple concepts named by short flags (`iscat`, `ilbl`, `iform`, `lin`) instead of typed modes.
- Rebinning and plotting coupled to k-table fetching.

## Opportunities Enabled by Modern Python

### Typed Scientific Domain Objects

Use dataclasses or Pydantic-like validation at I/O boundaries:

- `Planet`
- `Star`
- `PressureGrid`
- `AtmosphereProfile`
- `LayerGrid`
- `OpacitySet`
- `Observation`
- `InstrumentResponse`
- `RetrievalProblem`

### Backend Interfaces

Define small protocols:

- `OpacityProvider`
- `RadiativeTransferBackend`
- `TemperatureParameterization`
- `CompositionParameterization`
- `CloudModel`
- `Likelihood`
- `SamplerAdapter`

### Better Testing

Modern Python makes it straightforward to combine:

- unit tests for pure transforms,
- golden-data tests against legacy examples,
- property tests for units/monotonicity,
- benchmark tests for performance regressions.

### Cleaner Performance Strategy

ROBERT can keep readable reference kernels and add Numba/JAX/GPU backends behind stable interfaces.

### Reproducibility by Construction

Run manifests can capture:

- config hash,
- opacity file checksums,
- code version,
- sampler settings,
- random seeds,
- environment metadata,
- output schema version.

## Core Lesson

NEMESIS teaches ROBERT what must be scientifically respected. NemesisPy teaches ROBERT which exoplanet workflows matter and which modern Python conveniences are useful. Neither should dictate ROBERT's architecture.

ROBERT should be a new framework with validated inheritance of scientific behavior, not a transliteration of historical code shape.

