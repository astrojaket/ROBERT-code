# ROBERT Software Architecture Specification v1.0

Status: design reference.

Scope: this document defines the target software architecture for ROBERT. It is
not an implementation plan for full physics. Future implementation work must
follow the boundaries, dependencies, and public interfaces specified here unless
this document is deliberately revised.

## 1. Project Identity

ROBERT is a scientific platform for atmospheric retrieval of exoplanet emission
spectra, designed first for JWST data and later for other observatories,
spectral modes, atmospheric models, and computational backends.

The published Python distribution name MUST be:

```text
robert-exoplanets
```

Reason: `robert` already exists as a pip-installable package name. The
architecture target is therefore:

```text
distribution name: robert-exoplanets
import namespace:  robert_exoplanets
command name:      robert
```

The v0.2 foundation uses `src/robert_exoplanets` as the import namespace. Future
code MUST NOT reintroduce the shorter `robert` import namespace.

## 2. Design Principles

ROBERT prioritizes, in order:

1. Scientific correctness.
2. Clarity.
3. Modularity.
4. Reproducibility.
5. Maintainability.
6. Extensibility.
7. Performance.

Consequences:

- Correct but slow reference code is preferred over fast unclear code.
- Raw user configuration MUST NOT enter physics kernels.
- Files are I/O products, not internal control flow.
- Optional performance backends MUST match reference behavior.
- New features MUST include validation strategy and failure behavior.
- The core package MUST remain useful without heavyweight optional samplers,
  GPU runtimes, or external opacity downloads.

## 3. Repository Layout

Target mature layout:

```text
ROBERT-code/
  AGENTS.md
  README.md
  pyproject.toml
  CHANGELOG.md
  CITATION.cff
  LICENSE
  docs/
    architecture/
    api/
    tutorials/
    how_to/
    theory/
    review/
    developer/
  examples/
    configs/
    data/
    scripts/
    notebooks/
  schemas/
  src/
    robert_exoplanets/
  tests/
    unit/
    integration/
    regression/
    scientific/
    fixtures/
  benchmarks/
  scripts/
  .github/
    workflows/
```

Directory responsibilities:

| Path | Exists because | Must not contain |
| --- | --- | --- |
| `src/robert_exoplanets/` | Installable package source | Large opacity data, notebooks, generated outputs |
| `docs/architecture/` | Normative design decisions | Ad hoc tutorials or historical notes |
| `docs/review/` | Evidence from external framework reviews | Current API promises unless promoted into architecture docs |
| `docs/api/` | Generated public API documentation | Manual architecture prose |
| `docs/tutorials/` | End-to-end learning paths | Unreviewed research scratch work |
| `docs/how_to/` | Short task-oriented guides | Long theory derivations |
| `docs/theory/` | Equations, assumptions, citations | Implementation details not tied to equations |
| `docs/developer/` | Contribution, release, and testing guides | User-facing science tutorials |
| `examples/configs/` | Small runnable retrieval configs | Production science outputs |
| `examples/data/` | Tiny fixture data only | Real JWST products or large opacity tables |
| `examples/scripts/` | Python scripts using public APIs | Private-module imports |
| `examples/notebooks/` | Tutorial notebooks using public APIs | Required test-only logic |
| `schemas/` | Versioned config schemas | Runtime-generated schemas without review |
| `tests/unit/` | Fast isolated behavior tests | Network, large files, slow samplers |
| `tests/integration/` | Cross-module wiring tests | Full science validation |
| `tests/regression/` | Golden-output checks | Benchmarks that vary by machine |
| `tests/scientific/` | Validated science cases | Smoke tests with arbitrary expected values |
| `tests/fixtures/` | Tiny local fixture data | Downloaded databases |
| `benchmarks/` | Performance measurement cases | Required CI correctness tests |
| `scripts/` | Developer utilities | User-facing public APIs |
| `.github/workflows/` | CI, docs, release automation | Local machine assumptions |

## 4. Package Hierarchy

Target package tree:

```text
robert_exoplanets/
  __init__.py
  cli/
  core/
  bodies/
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

### 4.1 `core`

Belongs here:

- Array shape conventions.
- Unit conversion helpers.
- Grid classes.
- Spectrum containers.
- Result/status containers.
- Exceptions.
- Logging helpers.
- Citation metadata helpers.

Does not belong here:

- Opacity file readers.
- Sampler-specific code.
- JWST-specific code.
- Physical parameterizations that depend on retrieval choices.

### 4.2 `bodies`

Belongs here:

- `Planet`.
- `Star`.
- Orbital and geometric metadata needed by forward models.

Does not belong here:

- Stellar contamination likelihoods.
- Instrument response functions.
- Retrieval priors.

### 4.3 `atmosphere`

Belongs here:

- `AtmosphereState`.
- Layer-centered temperature, composition, cloud, pressure, altitude, density,
  and mean-molecular-weight arrays.
- Validation of atmospheric shape and units.

Does not belong here:

- Opacity database access.
- Sampler logic.
- User config parsing.

### 4.4 `parameterizations`

Belongs here:

- Pure transforms from retrieval parameters to atmospheric profiles.
- Temperature profile families.
- Composition profile families.
- Cloud vertical profile families.
- Parameter requirements and citations for each model.

Does not belong here:

- File I/O.
- Likelihood evaluation.
- Opacity interpolation.
- Instrument convolution.

### 4.5 `chemistry`

Belongs here:

- Chemistry backend protocols.
- Free chemistry helpers.
- Equilibrium chemistry adapters.
- Disequilibrium/quench adapters when implemented.

Does not belong here:

- Hard dependency on every chemistry package.
- Retrieval sampler code.
- Cloud optical properties.

### 4.6 `clouds`

Belongs here:

- Cloud model protocols.
- Cloud vertical distribution models.
- Cloud optical-property models.
- Cloud citations and parameter definitions.

Does not belong here:

- General gas opacity handling.
- Instrument response.
- Sampler-specific priors.

### 4.7 `opacity`

Belongs here:

- Opacity database metadata.
- Opacity providers.
- Correlated-k tables.
- Opacity sampling and line-by-line providers when implemented.
- CIA, Rayleigh, and special continuum providers.
- Coverage checks.
- Cache keys and prepared opacity state.

Does not belong here:

- Radiative-transfer integration.
- Retrieval likelihoods.
- Instrument binning.
- Sampler imports.

### 4.8 `rt`

Belongs here:

- Radiative-transfer backend protocols.
- Reference NumPy emission solver.
- Optional Numba/JAX/compiled backends.
- RT diagnostics such as contribution functions.

Does not belong here:

- Opacity file parsing.
- Observation file loading.
- Sampler logic.
- Plotting.

### 4.9 `instruments`

Belongs here:

- `Observation`.
- `Instrument`.
- `InstrumentResponse`.
- JWST response/convolution/binning models.
- Calibration nuisance models: offset, scale, jitter.

Does not belong here:

- Atmospheric physics.
- Opacity interpolation.
- Sampler execution.

### 4.10 `forward`

Belongs here:

- `ForwardModel`.
- Assembly of planet, star, atmosphere builder, opacity provider, RT backend,
  and instrument response into a prediction callable.

Does not belong here:

- Sampler implementation.
- Config file parsing.
- Plotting.

### 4.11 `likelihoods`

Belongs here:

- Gaussian independent-error likelihood.
- Covariance likelihood.
- Multi-dataset likelihood composition.
- Noise and calibration parameter evaluation.

Does not belong here:

- Priors.
- Sampler APIs.
- Opacity loading.

### 4.12 `retrieval`

Belongs here:

- `RetrievalProblem`.
- Parameter definitions.
- Prior transforms.
- Posterior/log-probability wrappers.
- Retrieval result containers.

Does not belong here:

- Specific sampler engines.
- RT kernels.
- File format readers.

### 4.13 `samplers`

Belongs here:

- Sampler adapter protocols.
- Built-in adapters such as `dynesty`.
- Optional adapters for UltraNest, PyMultiNest, Nautilus, BlackJAX, NumPyro, or
  JAXNS.

Does not belong here:

- Physics code.
- Opacity code.
- Instrument code except through `RetrievalProblem`.

### 4.14 `io`

Belongs here:

- YAML config loading.
- Pydantic config schemas.
- Serialization of manifests, spectra, posterior summaries, and results.
- Legacy import adapters.

Does not belong here:

- Physics kernels.
- Sampler internals.
- Automatic downloads without explicit user action.

### 4.15 `plugins`

Belongs here:

- Plugin discovery.
- Entry-point loading.
- Plugin metadata validation.
- Registry construction.

Does not belong here:

- Built-in physics implementations unless they are only registration shims.

### 4.16 `validation`

Belongs here:

- Reference cases.
- Golden-data comparison helpers.
- Scientific acceptance thresholds.

Does not belong here:

- Production retrieval orchestration.
- Large external data.

### 4.17 `analysis` and `visualization`

Belongs here:

- Posterior summaries.
- Corner-plot helper wrappers.
- Spectrum and profile envelope extraction.
- Diagnostic plotting.

Does not belong here:

- Required retrieval logic.
- Core physics dependencies.

## 5. Public Module Interfaces

All public modules MUST document inputs, outputs, invariants, and error modes.
The following module contracts define the target v1.0 surface.

### 5.1 Core Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `core.grids` | Array-like grid values, units, edge/center mode | `PressureGrid`, `SpectralGrid`, `LayerGrid` | Validate monotonicity, shape, units, layer boundaries | Grids are immutable and one-dimensional unless explicitly multi-axis |
| `core.spectrum` | Wavelength/frequency arrays, flux-like arrays, units | `Spectrum` | Represent model or observed spectra independent of instrument | Spectral coordinate and value arrays have matching shape |
| `core.exceptions` | Error context | Typed exceptions | Provide stable error classes | Public APIs raise ROBERT exceptions or standard Python errors with clear messages |
| `core.logging` | Logger name, verbosity | Configured logger | Central logging policy | Libraries never call `basicConfig` at import |
| `core.citations` | Component citation metadata | Citation records | Preserve scientific provenance | Every nontrivial model can expose citations |

Expected behavior:

- Grid constructors reject non-finite, duplicate, or invalidly ordered values.
- Spectrum containers do not know about retrievals, samplers, or opacities.

### 5.2 Body Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `bodies.planet` | Radius, mass/gravity, orbit, name | `Planet` | Store planetary metadata used by atmosphere and RT | Either gravity is provided or derivable from mass/radius |
| `bodies.star` | Radius, effective temperature, spectrum, contamination model metadata | `Star` | Store stellar metadata and optional stellar spectrum | Stellar spectra have explicit units and source metadata |

Expected behavior:

- Missing optional stellar spectra are allowed for brown-dwarf or planet-only
  emission modes.
- Star objects do not perform contamination modeling themselves; they only own
  stellar data.

### 5.3 Atmosphere and Parameterization Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `atmosphere.state` | Pressure grid, temperature, composition, altitude, cloud state | `AtmosphereState` | Validate full atmospheric state | Layer dimension is consistent across all layer arrays |
| `parameterizations.temperature` | Parameters and pressure grid | Temperature profile | Temperature transforms | Temperature values are finite and positive |
| `parameterizations.composition` | Parameters, pressure grid, species list | Composition profiles | Free and profile-based abundance models | VMR/mass-fraction conventions are explicit |
| `parameterizations.clouds` | Parameters, pressure grid, atmosphere state | Cloud vertical state | Cloud profile transforms | Cloud optical depths or masses are non-negative |
| `parameterizations.registry` | Component names | Component classes/functions | Register built-in parameterizations | Aliases are explicit and documented |

Expected behavior:

- Parameterizations are pure functions or immutable callable objects.
- Parameterizations declare required parameter names, bounds/support, units, and
  citations.
- Parameterizations do not read files.

### 5.4 Chemistry Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `chemistry.base` | Atmosphere state, parameters | Chemistry result | Define backend protocol | Backend output species names match opacity species mapping |
| `chemistry.free` | Abundance parameters | Composition profiles | Free chemistry models | Fill gases and trace gases are normalized under declared convention |
| `chemistry.equilibrium` | T/P, metallicity, C/O, backend data | Equilibrium composition | Adapter to validated chemistry grids or solvers | Backend metadata and grid coverage are recorded |
| `chemistry.quench` | T/P, Kzz-like parameters, backend composition | Quenched profiles | Optional disequilibrium approximation | Quench assumptions are explicit |

Expected behavior:

- Chemistry backends fail before retrieval if required data are missing.
- Chemistry data checksums are written to the run manifest.

### 5.5 Opacity Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `opacity.database` | File path or database URI | `OpacityDatabase` | Describe available species, grids, formats | Metadata can be inspected without running RT |
| `opacity.providers` | Database, species, pressure/spectral grids | `OpacityProvider`, `PreparedOpacity` | Prepare and evaluate opacity | Coverage is validated before likelihood evaluation |
| `opacity.correlated_k` | K-tables, g-ordinates, species state | CK opacity state | Correlated-k interpolation and mixing | CK mode records mixing rule and quadrature |
| `opacity.cia` | CIA tables and atmosphere state | CIA opacity | CIA continuum contribution | CIA pair availability is validated |
| `opacity.rayleigh` | Species and wavelength grid | Rayleigh cross sections | Rayleigh contribution | Species formula/source is recorded |
| `opacity.cache` | Cache key fields | Cache key, cache store | Deterministic caching | Cache key includes checksums and grids |
| `opacity.coverage` | Atmosphere state and opacity metadata | Coverage report | Detect missing/out-of-range data | No silent extrapolation by default |

Expected behavior:

- Opacity providers are prepared outside the sampler hot loop.
- Opacity state is immutable once prepared.
- Extrapolation requires an explicit policy object and manifest entry.

### 5.6 Radiative-Transfer Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `rt.base` | Atmosphere, opacity, geometry | Spectrum and diagnostics | Define backend protocol | Backends return identical semantic outputs |
| `rt.emission` | Layer state, opacity, boundary conditions | Emission spectrum | Reference 1D thermal emission | Units and angular integration convention are documented |
| `rt.diagnostics` | RT intermediate arrays | Contribution functions, photosphere estimates | Diagnostics | Diagnostics align with spectral grid |
| `rt.backends.numpy` | NumPy arrays | Spectrum/diagnostics | Readable reference backend | Must be the validation baseline |
| `rt.backends.numba` | NumPy-compatible arrays | Spectrum/diagnostics | Optional accelerated backend | Must pass reference tests |

Expected behavior:

- RT backends do not know about observations or samplers.
- RT backends accept prepared opacity state, not opacity file paths.

### 5.7 Instrument Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `instruments.observation` | Observed wavelengths, flux/depth, uncertainty, masks, metadata | `Observation` | Store observed data | Data arrays have consistent shape and finite uncertainties |
| `instruments.response` | Model spectrum, response/kernel/bin definitions | Binned prediction | Convolution, integration, sampling | Response weights are normalized or explicitly not normalized |
| `instruments.jwst` | JWST mode metadata | `Instrument` or response objects | JWST-specific response setup | Instrument mode and data product provenance are recorded |
| `instruments.calibration` | Calibration parameters and observation groups | Calibration-adjusted prediction/data | Offsets, scales, jitter | Calibration parameter units are explicit |

Expected behavior:

- Observations may contain multiple segments or instruments.
- Instrument response does not evaluate atmospheric physics.
- Calibration models are explicit components of the likelihood.

### 5.8 Forward, Likelihood, Retrieval, and Sampler Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `forward.model` | Bodies, atmosphere builder, opacity provider, RT backend, instrument response | `ForwardModel` | Convert parameter vector to model prediction | No sampler imports |
| `likelihoods.gaussian` | Prediction, observation, uncertainties, calibration | Log likelihood | Independent Gaussian likelihood | Handles masks and invalid predictions deterministically |
| `likelihoods.covariance` | Prediction, observation, covariance | Log likelihood | Correlated Gaussian likelihood | Covariance is positive definite or rejected |
| `retrieval.parameters` | Parameter specs | `Parameter`, `ParameterSpace` | Names, bounds, priors, transforms | Parameter names are unique |
| `retrieval.problem` | Forward model, likelihood, parameter space, datasets | `RetrievalProblem` | Single complete retrieval definition | Callable log posterior is deterministic for fixed seed/state |
| `retrieval.results` | Samples, evidence, best-fit, diagnostics, manifest | `RetrievalResult` | Sampler-independent result container | Result schema is versioned |
| `samplers.base` | Retrieval problem | Sampler protocol | Adapter interface | Sampler receives no raw physics internals |
| `samplers.dynesty` | Retrieval problem, sampler config | Retrieval result | Dynesty integration | Resume and random seed behavior are recorded |

Expected behavior:

- `RetrievalProblem` is the only object samplers need.
- Likelihoods return finite log values or a documented invalid-model floor.
- Samplers write no files except through `io.results` or result writers.

### 5.9 I/O, Plugins, Validation, and Analysis Modules

| Module | Inputs | Outputs | Responsibilities | Invariants |
| --- | --- | --- | --- | --- |
| `io.config` | YAML/Python config | Validated config model | Parse user config into typed objects | Raw dicts stop here |
| `io.manifest` | Problem, environment, files, settings | Run manifest | Reproducibility record | Manifest is written before sampling starts |
| `io.serialization` | Results and spectra | Files | Stable output formats | Output schema has version |
| `plugins.registry` | Entry points and built-ins | Component registry | Discovery and validation | Plugin loading has no side effects beyond registration |
| `validation.reference_cases` | Tiny fixtures and expected outputs | Validation reports | Scientific regression cases | Tolerances are explicit |
| `analysis.posterior` | Retrieval result | Summaries | Posterior statistics | Does not rerun retrieval |
| `visualization.spectra` | Result/spectrum objects | Figures | Plotting helpers | Plotting is optional dependency-friendly |

Expected behavior:

- I/O modules may depend on domain objects but domain objects do not depend on
  I/O modules.
- Plugin discovery is deterministic and can be disabled for tests.

## 6. Core Data Model

### 6.1 `PressureGrid`

Required attributes:

- `edges`: pressure at layer boundaries.
- `centers`: pressure at layer centers.
- `unit`: pressure unit.
- `orientation`: explicit direction, for example high-to-low or low-to-high.

Optional attributes:

- `name`.
- `metadata`.

Ownership and lifecycle:

- Created by config or Python API.
- Owned by atmosphere builders and reused by opacity preparation.
- Immutable after construction.

### 6.2 `SpectralGrid`

Required attributes:

- `wavelength` or `wavenumber`.
- `unit`.
- `mode`: native, observed, opacity, or internal.

Optional attributes:

- Bin edges.
- Resolving power.
- Source metadata.

Lifecycle:

- Observation and opacity providers may each define grids.
- Instrument response maps model grids to observation grids.

### 6.3 `Planet`

Required attributes:

- `name`.
- `radius` or radius parameter reference.
- `gravity` or mass/radius sufficient to derive gravity.

Optional attributes:

- `mass`.
- `orbital_period`.
- `semi_major_axis`.
- `equilibrium_temperature`.
- `system_distance`.
- Metadata and citations.

Lifecycle:

- Constructed before retrieval problem.
- May contain fixed values or parameter references, but not sampled values.

### 6.4 `Star`

Required attributes:

- `name` or source identifier.

Optional attributes:

- `radius`.
- `effective_temperature`.
- `log_g`.
- `metallicity`.
- `spectrum`.
- Contamination-region metadata.

Lifecycle:

- Optional for brown-dwarf mode.
- Required for planet/star emission contrast and stellar contamination models.

### 6.5 `Observation`

Required attributes:

- Spectral coordinates.
- Observed value.
- Uncertainty or covariance.
- Observable type: eclipse depth, flux ratio, flux density, transit depth, etc.

Optional attributes:

- Mask.
- Instrument name.
- Segment/group labels.
- Response mode per bin.
- Time/visit metadata.
- Provenance.

Ownership:

- Owned by `RetrievalProblem`.
- Never mutated by likelihoods.

Lifecycle:

- Loaded from config or Python API.
- Validated before retrieval starts.

### 6.6 `Instrument` and `InstrumentResponse`

Required attributes:

- Instrument name.
- Mode or response definition.
- Method for mapping a model `Spectrum` to observation bins.

Optional attributes:

- Line-spread function.
- Throughput curve.
- Resolving power.
- Calibration group mapping.

Lifecycle:

- Prepared before retrieval.
- May cache convolution matrices or response weights.

### 6.7 `AtmosphereState`

Required attributes:

- `pressure_grid`.
- Layer temperature.
- Composition by species.
- Mean molecular weight.

Optional attributes:

- Altitude.
- Gravity profile.
- Density.
- Cloud state.
- Column/patch axis.
- Derived chemistry diagnostics.

Ownership:

- Produced by an atmosphere builder for a given parameter vector.
- Consumed by opacity providers and RT backends.

Lifecycle:

- Usually transient inside forward-model calls.
- May be saved for posterior diagnostics.

### 6.8 `OpacityDatabase`

Required attributes:

- Location or URI.
- Format.
- Available species.
- Pressure/temperature/spectral coverage.
- Checksums or content identifiers.

Optional attributes:

- Citation metadata.
- Version.
- Source database name.

Ownership:

- Owned by opacity provider configuration.

Lifecycle:

- Inspected before retrieval.
- Not mutated during retrieval.

### 6.9 `PreparedOpacity`

Required attributes:

- Species list.
- Spectral grid.
- Pressure/temperature grid mappings.
- Opacity arrays or lazy handles.
- Cache key.

Optional attributes:

- CK g-ordinates and weights.
- Interpolation indices.
- Memory mapping handles.

Lifecycle:

- Created outside sampler hot loop.
- Immutable during likelihood evaluation.

### 6.10 `Spectrum`

Required attributes:

- Spectral coordinate.
- Spectral value.
- Observable type.
- Units.

Optional attributes:

- Uncertainty.
- Resolution.
- Diagnostics.
- Metadata.

Lifecycle:

- Produced by RT.
- Transformed by instrument response.
- Stored in results.

### 6.11 `ForwardModel`

Required attributes:

- Planet.
- Optional star.
- Atmosphere builder.
- Opacity provider.
- RT backend.
- Instrument response.

Required behavior:

- `predict(parameters) -> ModelPrediction`.

Lifecycle:

- Built once per retrieval problem.
- Owns prepared immutable dependencies.
- Does not own sampler state.

### 6.12 `Likelihood`

Required attributes:

- Likelihood type.
- Observation or dataset collection.
- Calibration model.

Required behavior:

- `loglike(prediction, parameters) -> float`.

Lifecycle:

- Owned by `RetrievalProblem`.
- Stateless except for prepared covariance or calibration metadata.

### 6.13 `Parameter`, `Prior`, and `ParameterSpace`

Required attributes:

- Unique parameter name.
- Physical meaning.
- Unit or dimensionless marker.
- Prior.
- Transform between unit cube/internal and physical values.

Optional attributes:

- Initial value.
- Bounds.
- Fixed value.
- Citation or model owner.

Lifecycle:

- Defined by config or Python API.
- Compiled by retrieval problem before sampling.

### 6.14 `RetrievalProblem`

Required attributes:

- Forward model.
- Likelihood.
- Parameter space.
- Run metadata.

Required behavior:

- `log_likelihood(theta)`.
- `log_prior(theta)`.
- `log_posterior(theta)`.
- `prior_transform(unit_cube)` for nested samplers.

Lifecycle:

- Complete and validated before any sampler runs.
- Immutable once sampling begins.

### 6.15 `Sampler`

Required behavior:

- `run(problem, output) -> RetrievalResult`.
- `resume(problem, output) -> RetrievalResult` if supported.

Optional attributes:

- Random seed.
- Number of live points or equivalent.
- Checkpoint path.

Lifecycle:

- Adapter object owns sampler-specific state.
- Does not mutate `RetrievalProblem`.

### 6.16 `RetrievalResult`

Required attributes:

- Samples or posterior representation.
- Best-fit or representative parameter set.
- Evidence/diagnostics if sampler provides them.
- Best-fit spectrum.
- Run manifest.
- Schema version.

Optional attributes:

- Profile envelopes.
- Contribution functions.
- Corner-plot-ready tables.
- Sampler raw output references.

Lifecycle:

- Created by sampler adapter.
- Serialized by `io.serialization`.

### 6.17 `RunManifest`

Required attributes:

- ROBERT version.
- Config schema version.
- Config hash.
- Git commit if available.
- Opacity checksums.
- Random seeds.
- Sampler settings.
- Environment summary.
- Plugin versions.

Lifecycle:

- Written before sampling starts.
- Updated with result metadata after sampling finishes.

## 7. Dependency Rules

Preferred dependency direction:

```text
core
  bodies
  atmosphere
    parameterizations
    chemistry
    clouds
  opacity
  rt
  instruments
  forward
  likelihoods
  retrieval
  samplers
  io
  analysis / visualization
```

More precise allowed dependencies:

| Package | May depend on |
| --- | --- |
| `core` | Standard library, NumPy, typing helpers |
| `bodies` | `core` |
| `atmosphere` | `core`, `bodies` |
| `parameterizations` | `core`, `atmosphere`, `retrieval.parameters` protocols only |
| `chemistry` | `core`, `atmosphere`, optional backend packages behind adapters |
| `clouds` | `core`, `atmosphere` |
| `opacity` | `core`, `atmosphere`, `chemistry` protocols where necessary |
| `rt` | `core`, `atmosphere`, `opacity` public protocols |
| `instruments` | `core`, `bodies` for star/planet metadata where needed |
| `forward` | `core`, `bodies`, `atmosphere`, `parameterizations`, `opacity`, `rt`, `instruments` |
| `likelihoods` | `core`, `instruments`, `forward` prediction protocols |
| `retrieval` | `core`, `forward`, `likelihoods` |
| `samplers` | `core`, `retrieval` |
| `io` | All public domain packages, never private internals |
| `plugins` | Public protocols only |
| `analysis` | `core`, `retrieval.results`, optional plotting/analysis packages |
| `visualization` | `core`, `analysis`, optional plotting packages |

Forbidden dependencies:

- `core` MUST NOT import any ROBERT subpackage.
- `rt` MUST NOT import `instruments`, `likelihoods`, `retrieval`, `samplers`, or
  `io.config`.
- `opacity` MUST NOT import `rt`, `forward`, `likelihoods`, `retrieval`, or
  `samplers`.
- `parameterizations` MUST NOT import `opacity`, `rt`, `likelihoods`, or
  `samplers`.
- `likelihoods` MUST NOT import sampler adapters.
- `samplers` MUST NOT import opacity, RT, or instrument implementation modules
  directly.
- Public modules MUST NOT depend on example scripts or notebooks.
- Import-time side effects such as MPI initialization, environment variable
  mutation, data downloads, or GPU platform selection are forbidden.

Dependency graph rule:

- Cycles are forbidden across top-level packages.
- Cycles inside a package require explicit justification in a developer note and
  SHOULD be removed before v1.0.

## 8. API Philosophy

### 8.1 Naming

- Class names use `PascalCase`.
- Functions, modules, and parameters use `snake_case`.
- Units appear in names only at I/O boundaries, for example `wavelength_um`.
- Internal physical objects store units explicitly rather than in variable names.
- Boolean names start with `use_`, `include_`, `allow_`, `is_`, or `has_`.
- Plugin names are lowercase, stable, and documented.

### 8.2 Object Ownership

- Configuration owns user intent.
- Domain objects own validated physical data.
- Prepared providers own cached computational state.
- `RetrievalProblem` owns the complete immutable retrieval definition.
- Samplers own sampler state only.
- Result writers own serialization.

### 8.3 Mutability

- Domain objects SHOULD be immutable.
- Arrays in domain objects MUST NOT be mutated in place by downstream code.
- Caches may be mutable internally but MUST expose deterministic keys and clear
  invalidation behavior.
- Sampler state may be mutable inside sampler adapters only.

### 8.4 Configuration

- User configuration is parsed once at the boundary.
- Raw dictionaries MUST NOT pass into physics, opacity, RT, likelihood, or
  sampler internals.
- Python API users may construct the same typed objects directly.

### 8.5 Error Handling

- User-correctable errors raise `RobertConfigError`, `RobertDataError`, or
  `RobertCoverageError`.
- Invalid model states inside a sampler return a documented log-probability floor
  only when that is scientifically intended.
- Programmer errors should raise standard Python exceptions.
- Error messages MUST include the failing component name and relevant parameter
  or file path.

### 8.6 Logging

- Library modules use `logging.getLogger(__name__)`.
- No module configures global logging at import.
- CLI configures logging.
- Long retrievals emit structured progress messages from sampler adapters.
- Warnings that affect science are also recorded in the run manifest.

### 8.7 Typing

- Public functions require type annotations.
- Core dataclasses and protocols are typed.
- Array shapes are documented in docstrings.
- Static typing is used to catch architecture violations, not to obscure
  scientific code.

### 8.8 Documentation Style

- Public APIs use NumPy-style docstrings.
- Scientific models include equations or references in `docs/theory/`.
- Every implemented parameterization lists assumptions, required parameters, and
  citations.
- Examples import public APIs only.

## 9. Plugin Architecture

ROBERT supports plugins for:

- Temperature parameterizations.
- Composition and chemistry models.
- Cloud vertical profiles.
- Cloud optical-property models.
- Opacity providers.
- Radiative-transfer backends.
- Instrument response models.
- Likelihoods.
- Priors and transforms.
- Sampler adapters.
- Result exporters.

Discovery mechanism:

- Built-ins are registered in package-local registries.
- External plugins use `importlib.metadata.entry_points`.
- Entry-point groups:

```text
robert_exoplanets.temperature
robert_exoplanets.composition
robert_exoplanets.chemistry
robert_exoplanets.clouds
robert_exoplanets.opacity
robert_exoplanets.rt
robert_exoplanets.instruments
robert_exoplanets.likelihoods
robert_exoplanets.priors
robert_exoplanets.samplers
robert_exoplanets.exporters
```

Plugin contract:

- A plugin exposes a factory or class implementing the relevant protocol.
- A plugin declares:
  - stable name,
  - version,
  - ROBERT compatibility range,
  - citations,
  - required optional dependencies,
  - configuration schema fragment.
- Loading a plugin MUST NOT run downloads, initialize MPI, set GPU devices, or
  perform expensive computation.
- Plugins are validated before being added to registries.

Conflict policy:

- Built-in names are reserved.
- External plugins cannot silently override existing names.
- Explicit aliases are allowed only if registered without conflict.

## 10. Configuration System

ROBERT uses:

- YAML for user-facing run configuration.
- Pydantic v2 models for configuration validation at the I/O boundary.
- Frozen dataclasses for core scientific domain objects.
- Python object construction for advanced users and tests.
- TOML only for project metadata (`pyproject.toml`), not retrieval runs.

Rationale:

- YAML is readable for scientists and supports nested retrieval definitions.
- Pydantic provides strong validation, helpful error messages, and JSON schema
  generation.
- Dataclasses keep physics objects lightweight and dependency-minimal.
- Python APIs remain essential for notebooks and advanced workflows.

Configuration rules:

- Every config file has `schema_version`.
- Every run records the exact config file and normalized validated config.
- Config sections:

```text
run
bodies
observations
instruments
atmosphere
chemistry
clouds
opacity
rt
likelihood
parameters
sampler
outputs
runtime
plugins
```

Minimal shape:

```yaml
schema_version: 1
run:
  name: wasp43b_miri_emission
  mode: emission
bodies:
  planet:
    name: WASP-43b
observations:
  - name: miri_lrs
    path: data/wasp43b_miri.txt
    observable: eclipse_depth
atmosphere:
  pressure_grid:
    min_bar: 1.0e-6
    max_bar: 1.0e2
    layers: 100
  temperature:
    model: isothermal
  composition:
    model: free_constant
opacity:
  mode: correlated_k
rt:
  backend: numpy
  mode: emission_1d
likelihood:
  model: gaussian
sampler:
  engine: dynesty
parameters: []
```

Config parser behavior:

- Validate paths but do not download data.
- Validate plugin names before building problem objects.
- Convert config into typed domain objects.
- Fail early on unknown fields unless an explicit extension namespace is used.

## 11. Testing Philosophy

Testing tiers:

| Tier | Purpose | Runtime target | CI requirement |
| --- | --- | --- | --- |
| Unit | Validate pure functions and objects | Seconds | Required every PR |
| Integration | Validate module wiring | Seconds to minutes | Required every PR |
| Regression | Compare against golden outputs | Minutes | Required when affected |
| Scientific validation | Compare against trusted cases/literature/frameworks | Minutes to hours | Scheduled or release-gated |
| Benchmark | Track speed and memory | Variable | Scheduled, not correctness-gating |

Unit tests:

- Cover grids, data objects, parameter transforms, priors, likelihood math, and
  config validation.
- Use tiny in-repo fixtures only.

Regression tests:

- Use fixed random seeds.
- Store expected outputs with schema versions.
- Include tolerance rationale.

Scientific validation tests:

- Clear 1D emission atmosphere.
- Cloudy 1D emission atmosphere.
- Multi-instrument JWST emission retrieval.
- Cross-check against a trusted framework or analytic limit where possible.

Benchmark tests:

- Measure opacity preparation.
- Measure forward model evaluation.
- Measure likelihood calls per second.
- Measure memory footprint for representative opacity grids.

CI:

- Run unit and integration tests on Linux for supported Python versions.
- Run at least one macOS job before release.
- Optional dependencies are tested in separate jobs.
- Slow tests use markers and are not required for every commit.
- Docs build is CI-gated before release.

## 12. Performance Philosophy

Performance belongs behind stable interfaces.

Rules:

- Implement a readable NumPy reference backend first.
- Add vectorization where it does not hide scientific logic.
- Add Numba only after reference tests exist.
- Add JAX/GPU only after CPU behavior and cache semantics are stable.
- Parallelism is configured at runtime, never at import.
- Prepared opacities and instrument response matrices are built outside the
  sampler hot loop.

Caching:

- Cache keys include:
  - ROBERT version,
  - backend,
  - species,
  - pressure grid,
  - spectral grid,
  - opacity checksums,
  - RT mode,
  - cloud/chemistry assumptions.
- Caches can be disabled for debugging.
- Cache hits and misses are logged at debug level and recorded in manifests when
  scientifically relevant.

GPU/JAX policy:

- GPU support is optional.
- JAX backends must match NumPy reference results within documented tolerances.
- JAX configuration occurs in CLI/runtime setup, not module import.
- GPU nondeterminism must be documented.

Memory policy:

- Avoid copying large opacity arrays in hot loops.
- Prefer prepared read-only arrays and memory mapping when appropriate.
- Document expected memory use for benchmark cases.

## 13. Documentation Philosophy

Documentation types:

| Type | Location | Audience | Required content |
| --- | --- | --- | --- |
| Architecture | `docs/architecture/` | Developers | Boundaries, dependency rules, design decisions |
| Theory | `docs/theory/` | Scientists/developers | Equations, assumptions, citations |
| API | `docs/api/` | Users/developers | Generated public API references |
| Tutorials | `docs/tutorials/` | New users | End-to-end examples |
| How-to guides | `docs/how_to/` | Users | Specific tasks |
| Developer docs | `docs/developer/` | Contributors | Testing, release, style, plugin authoring |
| Review notes | `docs/review/` | Maintainers | Evidence from external frameworks |

Rules:

- Tutorials use public APIs only.
- Every public component has an API docstring.
- Every scientific model has citations.
- Every example has a small data fixture or a clearly documented external data
  requirement.
- Architecture docs are updated in the same PR as architecture-changing code.

## 14. Versioning Roadmap

Semantic versioning applies after v1.0. Before v1.0, minor versions may break API
with migration notes.

### v0.1: Skeleton

Status: current repository stage.

Includes:

- Minimal package.
- Stub observation/config/model/result objects.
- Stub end-to-end example.
- Initial tests.

Exit criteria:

- Tests pass.
- Architecture specification exists.

### v0.2: Naming and Core Domain

Includes:

- Distribution name `robert-exoplanets`.
- Import namespace `robert_exoplanets` implemented.
- `core` grids, spectra, exceptions, logging.
- `Planet`, `Star`, `Observation`.
- Config schema draft.

Exit criteria:

- No public code uses temporary package names.
- Config objects validate basic run files.

### v0.3: Observations, Instruments, and Likelihoods

Includes:

- JWST-style observation containers.
- Instrument response interface.
- Independent Gaussian likelihood.
- Calibration parameters: offset and jitter.
- Run manifest v1 draft.

Exit criteria:

- Multi-segment synthetic observation smoke test.

### v0.4: Opacity Infrastructure

Includes:

- Opacity database metadata.
- Coverage validation.
- Tiny correlated-k fixture format.
- CIA and Rayleigh fixture providers.
- Deterministic opacity cache keys.

Exit criteria:

- Opacity preparation works without external downloads.

### v0.5: Reference Emission Forward Model

Includes:

- 1D atmosphere builder.
- Isothermal and one non-isothermal temperature model.
- Free constant chemistry.
- NumPy thermal emission backend.
- Contribution diagnostics.

Exit criteria:

- Clear-atmosphere golden spectrum test.

### v0.6: Retrieval Problem and First Sampler

Includes:

- `ParameterSpace`.
- Priors and transforms.
- `RetrievalProblem`.
- dynesty adapter.
- Sampler-independent result object.

Exit criteria:

- Tiny end-to-end retrieval smoke test.

### v0.7: Scientific Validation and Documentation

Includes:

- Clear and cloudy validation cases.
- User tutorial for JWST emission retrieval.
- Developer plugin guide.
- API documentation build.

Exit criteria:

- Release candidate docs are complete.

### v0.8: Clouds and Extended Likelihoods

Includes:

- Basic gray cloud/deck model.
- Covariance likelihood.
- Multi-instrument likelihood composition.
- Optional second sampler adapter.

Exit criteria:

- Cloudy emission retrieval reference case.

### v0.9: API Freeze Candidate

Includes:

- Stable plugin protocols.
- Stable config schema v1.
- Performance benchmarks.
- Migration guide from earlier v0.x APIs.

Exit criteria:

- No known architecture blockers for v1.0.

### v1.0: Stable JWST Emission Platform

Includes:

- Stable public API for JWST 1D emission retrievals.
- Validated opacity, atmosphere, RT, likelihood, and sampler pipeline.
- Reproducible run manifests.
- Plugin discovery.
- Complete documentation.
- CI and release process.

Non-goals for v1.0:

- Full phase-curve retrieval.
- Full 3D retrieval.
- Polarisation.
- Mandatory GPU support.
- Multiple production chemistry backends.

## 15. Architecture Change Process

Any change that affects module boundaries, public data models, dependency rules,
configuration schema, plugin protocols, or result schema MUST:

1. Update this specification or add an accepted architecture decision record.
2. Include migration notes if user-facing behavior changes.
3. Include tests that enforce the new contract.
4. Avoid compatibility breaks after v1.0 unless released as v2.0 or guarded by a
   deprecation cycle.

## 16. Final Implementation Gate

No significant ROBERT physics, sampler, opacity, or instrument implementation
should begin until this specification has been reviewed and accepted. The first
implementation work after acceptance should be the v0.2 naming and core-domain
cleanup, especially preserving the `robert-exoplanets` distribution name and
`robert_exoplanets` import namespace.
