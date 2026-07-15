# Software Architecture Document

This document defines ROBERT's software structure: package hierarchy, public API
ownership, implementation boundaries, allowed dependencies, and forbidden
dependencies.

## 1. Architecture Style

ROBERT uses a layered, protocol-oriented architecture.

The central design rule is:

```text
science objects and protocols are stable;
implementations and backends are replaceable.
```

The core retrieval workflow is assembled from independent components:

```text
Observation + InstrumentResponse
Planet + Star
AtmosphereBuilder
OpacityProvider
RadiativeTransferBackend
Likelihood
ParameterSpace
SamplerAdapter
```

## 2. Target Package Hierarchy

```text
robert_exoplanets/
  cli/
  core/
  bodies/
  stellar/
  atmosphere/
  parameterizations/
  chemistry/
  clouds/
  opacity/
  rt/
  instruments/
  forward/
  likelihoods/
  retrieval/
  samplers/
  io/
  plugins/
  validation/
  analysis/
  visualization/
```

## 3. Package Contracts

### `cli`

Purpose:

- Provide command-line entry points such as `robert run config.yaml`.

Public API:

- CLI commands, not imported scientific APIs.

Internal implementation:

- Argument parsing.
- Logging setup.
- Runtime environment setup.
- Calls into `io.config`, `retrieval`, and `samplers`.

Allowed dependencies:

- Any public ROBERT package.

Forbidden dependencies:

- Private internals of physics packages.
- Heavy optional packages unless behind command-specific imports.

### `core`

Purpose:

- Provide low-level objects and utilities shared across ROBERT.

Public API:

- `PressureGrid`, `SpectralGrid`, `Spectrum`, result status types,
  exceptions, logging helpers, citation helpers.

Internal implementation:

- Unit-aware array validation.
- Shape and monotonicity checks.
- Base exception classes.

Allowed dependencies:

- Standard library.
- NumPy.
- Optional lightweight typing utilities.

Forbidden dependencies:

- Any other ROBERT package.
- Samplers.
- Plotting.
- File readers beyond simple serialization helpers.

### `bodies`

Purpose:

- Represent astronomical bodies.

Public API:

- `Planet`.
- `Star`.
- Geometry metadata types.

Internal implementation:

- Validation of radius, mass, gravity, orbital metadata, and stellar spectra.

Allowed dependencies:

- `core`.

Forbidden dependencies:

- `rt`, `opacity`, `samplers`, `likelihoods`.

### `stellar`

Purpose:

- Prepare stellar photosphere spectra for emission normalization,
  irradiation, and future transit-light-source-effect components.

Public API:

- `StellarSpectrumModel`.
- `PhoenixStellarSpectrumModel`.
- `BlackbodyStellarSpectrumModel`.

Internal implementation:

- Stellar-atmosphere catalog interpolation and coverage validation.
- Flux-conserving preparation on immutable spectral grids.
- Explicit surface-flux and radiance conventions.

Allowed dependencies:

- `core`.
- `bodies`.
- Optional catalog adapters imported at the package edge.

Forbidden dependencies:

- `rt`, `opacity`, `likelihoods`, `samplers`.
- File access during a likelihood call.

### `atmosphere`

Purpose:

- Represent the evaluated atmospheric state.

Public API:

- `AtmosphereState`.
- `AtmosphereBuilder`.
- Column/patch structures when implemented.

Internal implementation:

- Consistency checks for pressure, temperature, composition, clouds, density,
  altitude, and mean molecular weight.

Allowed dependencies:

- `core`.
- `bodies` only for gravity or geometry metadata.

Forbidden dependencies:

- `opacity`, `rt`, `samplers`, `io.config`.

### `parameterizations`

Purpose:

- Convert retrieval parameters into physical profiles.

Public API:

- `TemperatureProfile`.
- `CompositionProfile`.
- `CloudProfile`.
- Registries and built-in parameterization factories.

Internal implementation:

- Pure transforms.
- Parameter requirement declarations.
- Citations.

Allowed dependencies:

- `core`.
- `atmosphere` protocols.

Forbidden dependencies:

- File I/O.
- Opacity or RT code.
- Likelihoods or samplers.

### `chemistry`

Purpose:

- Provide composition models and chemistry backend adapters.

Public API:

- `ChemistryModel`.
- `FreeChemistry`.
- `EquilibriumChemistry`.
- Backend protocols.

Internal implementation:

- Free abundance transforms.
- Equilibrium grid adapters.
- Quench approximations when validated.

Allowed dependencies:

- `core`.
- `atmosphere`.
- Optional chemistry packages inside adapters only.

Forbidden dependencies:

- Samplers.
- Instrument code.
- Mandatory heavy chemistry dependencies in core imports.

### `clouds`

Purpose:

- Define cloud vertical and optical models.

Public API:

- `CloudModel`.
- `CloudProfile`.
- Cloud optical-property protocols.

Internal implementation:

- Gray cloud decks.
- Slabs.
- Mie/n,k adapters when implemented.

Allowed dependencies:

- `core`.
- `atmosphere`.

Forbidden dependencies:

- General gas opacity databases unless through public opacity protocols.
- Samplers.

### `opacity`

Purpose:

- Own opacity data access, validation, preparation, and interpolation.

Public API:

- `OpacityDatabase`.
- `OpacityProvider`.
- `PreparedOpacity`.
- `CoverageReport`.
- CIA, Rayleigh, correlated-k provider interfaces.

Internal implementation:

- File format readers.
- Species mapping.
- Coverage checks.
- Cache-key generation.
- Interpolation index preparation.

Allowed dependencies:

- `core`.
- `atmosphere` public state objects.
- `chemistry` protocols only when required.

Forbidden dependencies:

- `rt`.
- `forward`.
- `likelihoods`.
- `retrieval`.
- `samplers`.
- User config parsing.

### `rt`

Purpose:

- Compute spectra from atmosphere and opacity state.

Public API:

- `RadiativeTransferBackend`.
- `EmissionSolver`.
- `TransmissionSolver` when implemented.
- RT diagnostics.

Internal implementation:

- NumPy reference solvers.
- Optional accelerated backends.
- Contribution functions.

Allowed dependencies:

- `core`.
- `atmosphere`.
- `opacity` public protocols and prepared state.

Forbidden dependencies:

- `instruments`.
- `likelihoods`.
- `retrieval`.
- `samplers`.
- `io.config`.
- Plotting.

### `instruments`

Purpose:

- Represent observations and instrument response.

Public API:

- `Observation`.
- `Instrument`.
- `InstrumentResponse`.
- Calibration/noise group metadata.

Internal implementation:

- Binning.
- Convolution.
- Response matrix preparation.
- Offset, scale, and jitter group handling.

Allowed dependencies:

- `core`.
- `bodies` if a response needs stellar/planet metadata.

Forbidden dependencies:

- RT implementations.
- Opacity implementations.
- Sampler adapters.

### `forward`

Purpose:

- Assemble a complete prediction model.

Public API:

- `ForwardModel`.
- `ModelPrediction`.

Internal implementation:

- Calls atmosphere builder, opacity provider, RT backend, and instrument
  response in sequence.

Allowed dependencies:

- `core`, `bodies`, `atmosphere`, `parameterizations`, `chemistry`, `clouds`,
  `opacity`, `rt`, `instruments`.

Forbidden dependencies:

- Sampler implementations.
- CLI.
- Plotting.

### `likelihoods`

Purpose:

- Compare model predictions with observations.

Public API:

- `Likelihood`.
- `GaussianLikelihood`.
- `CovarianceLikelihood`.
- `NoiseModel`.

Internal implementation:

- Residuals.
- Masks.
- Covariance handling.
- Calibration nuisance parameters.

Allowed dependencies:

- `core`.
- `instruments`.
- `forward` prediction protocols.

Forbidden dependencies:

- Opacity internals.
- RT internals.
- Sampler implementations.

### `retrieval`

Purpose:

- Define a complete retrieval problem independent of any sampler.

Public API:

- `Parameter`.
- `Prior`.
- `ParameterSpace`.
- `RetrievalProblem`.
- `RetrievalResult`.

Internal implementation:

- Prior transforms.
- Log prior.
- Log likelihood wrapper.
- Result schema.

Allowed dependencies:

- `core`.
- `forward`.
- `likelihoods`.

Forbidden dependencies:

- Specific sampler libraries.
- Opacity file readers.
- RT private modules.

### `samplers`

Purpose:

- Adapt external inference engines to ROBERT retrieval problems.

Public API:

- `SamplerAdapter`.
- Built-in adapter classes such as `DynestySampler`.

Internal implementation:

- Unit-cube transforms.
- Sampler-specific callbacks.
- Checkpoint/resume handling.

Allowed dependencies:

- `core`.
- `retrieval`.
- Optional sampler package inside adapter module only.

Forbidden dependencies:

- `opacity` internals.
- `rt` internals.
- `instruments` internals.

### `io`

Purpose:

- Convert external files and user config into typed ROBERT objects.

Public API:

- Config loaders.
- Manifest writers.
- Result serializers.
- Legacy import adapters.

Internal implementation:

- YAML parsing.
- Pydantic validation.
- Schema versioning.
- Output format writers.

Allowed dependencies:

- Public domain packages.

Forbidden dependencies:

- Private implementation modules.
- Sampler-specific raw output assumptions except through result adapters.

### `plugins`

Purpose:

- Discover and validate built-in and external plugins.

Public API:

- Registry construction.
- Plugin metadata types.

Internal implementation:

- Entry-point discovery.
- Version compatibility checks.
- Conflict resolution.

Allowed dependencies:

- Public protocols.
- `importlib.metadata`.

Forbidden dependencies:

- Heavy plugin imports at package import time.
- Side-effectful plugin loading.

### `validation`

Purpose:

- Provide scientific and numerical validation utilities.

Public API:

- Reference case definitions.
- Comparison reports.
- Tolerance helpers.

Internal implementation:

- Golden-output comparison.
- Framework cross-check summaries.

Allowed dependencies:

- Public ROBERT APIs.

Forbidden dependencies:

- Production run orchestration that users must call.

### `analysis` and `visualization`

Purpose:

- Analyze and present retrieval outputs.

Public API:

- Posterior summaries.
- Spectrum envelopes.
- Profile diagnostics.
- Plot helpers.

Internal implementation:

- Optional plotting and ArviZ-like adapters.

Allowed dependencies:

- `core`.
- `retrieval` result objects.
- Optional analysis/plot packages.

Forbidden dependencies:

- Core retrieval execution requirements.

## 4. Dependency Graph Rules

Forbidden imports:

- `core` imports no ROBERT package.
- `rt` imports no `instruments`, `likelihoods`, `retrieval`, `samplers`, or
  `io.config`.
- `opacity` imports no `rt`, `forward`, `likelihoods`, `retrieval`, or
  `samplers`.
- `parameterizations` imports no `opacity`, `rt`, `likelihoods`, or `samplers`.
- `samplers` imports no opacity or RT implementation modules.
- `examples` are never imported by package code.

Cycle policy:

- Top-level package cycles are forbidden.
- Internal cycles require an architecture note and must be removed before v1.0.

## 5. Import-Time Side Effects

No package import may:

- Start MPI.
- Set BLAS/OpenMP thread environment variables.
- Select CUDA/JAX platforms.
- Download data.
- Read large opacity files.
- Configure global logging.
- Create output directories.

Such behavior belongs in CLI runtime setup or explicit object construction.
