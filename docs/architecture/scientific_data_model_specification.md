# Scientific Data Model Specification

This document defines the scientific entities ROBERT uses to represent
retrieval problems. Objects should be typed, validated, and immutable wherever
possible.

## 1. Data Model Principles

- Scientific objects own validated physical data, not raw user config.
- Units and conventions are explicit.
- Objects have clear lifecycles.
- Arrays use documented shapes and axes.
- Derived quantities record how they were derived.
- Large data handles are separated from small metadata objects.

## 2. Core Entities

### `Planet`

Purpose:

- Represent the planetary body whose atmosphere is retrieved.

Required attributes:

- `name`.
- `radius` or parameter reference for radius.
- `gravity`, or enough information to derive gravity from mass and radius.

Optional attributes:

- `mass`.
- `semi_major_axis`.
- `orbital_period`.
- `inclination`.
- `eccentricity`.
- `system_distance`.
- `equilibrium_temperature`.
- Metadata and citations.

Responsibilities:

- Validate body-level parameters.
- Provide geometry quantities to atmosphere and RT components.

Ownership:

- Owned by `RetrievalProblem`.

Lifecycle:

- Created during config parsing or Python API construction.
- Immutable once retrieval begins.

Relationships:

- Used by `AtmosphereBuilder`, `ForwardModel`, and sometimes
  `InstrumentResponse`.

### `Star`

Purpose:

- Represent stellar properties and optional stellar spectrum.

Required attributes:

- `name` or source identifier.

Optional attributes:

- `radius`.
- `effective_temperature`.
- `log_g`.
- `metallicity`.
- `spectrum`.
- Stellar heterogeneity metadata.

Responsibilities:

- Store stellar information used for emission contrast, irradiation, and future
  contamination models.
- Supply effective temperature, log surface gravity, and metallicity to a
  selected stellar-spectrum model, or carry an explicitly prepared spectrum.

Ownership:

- Owned by `RetrievalProblem`.

Lifecycle:

- Optional for brown-dwarf or star-free emission modes.
- Immutable once retrieval begins.

Relationships:

- Used by stellar-spectrum preparation, RT, instrument, and stellar
  contamination components through public fields.

### `PressureGrid`

Purpose:

- Define atmospheric vertical discretization.

Required attributes:

- `edges`.
- `centers`.
- `unit`.
- `orientation`.

Optional attributes:

- `name`.
- `metadata`.

Responsibilities:

- Validate pressure monotonicity.
- Define layer and level conventions.

Ownership:

- Owned by atmosphere configuration and shared by atmosphere/opacity/RT.

Lifecycle:

- Constructed before opacity preparation.
- Immutable.

Relationships:

- Used by `AtmosphereState`, opacity providers, and RT backends.

### `SpectralGrid`

Purpose:

- Represent spectral coordinates for opacity, model, or observed spectra.

Required attributes:

- Coordinate values.
- Coordinate unit.
- Grid role: opacity, native model, observed, or internal.

Optional attributes:

- Bin edges.
- Resolving power.
- Source metadata.

Responsibilities:

- Validate monotonicity and shape.
- Preserve spectral convention.

Ownership:

- Owned by opacity providers, spectra, or instrument response.

Lifecycle:

- Prepared before retrieval.
- Immutable.

### `TemperatureProfile`

Purpose:

- Convert retrieval parameters into atmospheric temperature values.

Required attributes/behavior:

- `evaluate(parameters, pressure_grid)`.
- Declared required parameters.
- Declared citations.

Optional attributes:

- Valid parameter domain.
- Smoothness or monotonicity constraints.

Responsibilities:

- Produce finite positive temperatures.

Ownership:

- Owned by `AtmosphereBuilder`.

Lifecycle:

- Constructed before retrieval.
- Called for each sampled parameter vector.

### `ChemistryModel`

Purpose:

- Convert retrieval parameters and atmospheric state into composition.

Required behavior:

- `evaluate(parameters, pressure_grid, temperature)`.
- Declare species produced.

Optional attributes:

- Backend data metadata.
- Elemental abundance basis.
- Equilibrium grid coverage.

Responsibilities:

- Produce composition arrays under an explicit convention, such as VMR or mass
  fraction.
- Validate normalization rules.

Ownership:

- Owned by `AtmosphereBuilder`.

Lifecycle:

- Prepared before retrieval if it needs external data.
- Called during forward-model evaluation.

### `CloudModel`

Purpose:

- Represent cloud vertical structure and optical properties.

Required behavior:

- Evaluate cloud state from parameters and atmosphere.

Optional attributes:

- Cloud composition.
- Particle size distribution.
- Refractive-index data.

Responsibilities:

- Produce non-negative cloud quantities.
- State assumptions and citations.

Ownership:

- Vertical cloud profiles are owned by atmosphere construction.
- Cloud optical properties may be owned by opacity providers.

### `AtmosphereState`

Purpose:

- Hold the evaluated atmospheric state for one parameter vector.

Required attributes:

- `pressure_grid`.
- `temperature`.
- `composition`.
- `mean_molecular_weight`.

Optional attributes:

- `altitude`.
- `gravity`.
- `density`.
- `cloud_state`.
- Column/patch axis.
- Derived diagnostics.

Responsibilities:

- Validate shared layer dimension.
- Provide a complete input to opacity and RT.

Ownership:

- Produced by `ForwardModel` or `AtmosphereBuilder`.

Lifecycle:

- Transient for likelihood calls.
- Saved only for diagnostics or posterior samples.

### `OpacityDatabase`

Purpose:

- Describe available opacity data.

Required attributes:

- Location.
- Format.
- Species.
- Spectral coverage.
- Temperature/pressure coverage.
- Checksums or content identifiers.

Optional attributes:

- Citation metadata.
- Version.
- Source database.

Responsibilities:

- Provide inspectable metadata before retrieval.

Ownership:

- Owned by opacity provider config.

Lifecycle:

- Inspected during setup.
- Not mutated during retrieval.

### `PreparedOpacity`

Purpose:

- Store run-specific prepared opacity state.

Required attributes:

- Species list.
- Spectral grid.
- Pressure/temperature mappings.
- Cache key.

Optional attributes:

- Interpolation indices.
- CK quadrature.
- Lazy array handles.

Responsibilities:

- Provide fast, validated opacity evaluation during retrieval.

Ownership:

- Owned by `OpacityProvider`.

Lifecycle:

- Created before sampling.
- Immutable during sampling.

### `Spectrum`

Purpose:

- Represent model or observed spectral values.

Required attributes:

- `spectral_grid`.
- `values`.
- `observable`.
- `unit`.

Optional attributes:

- Uncertainty.
- Resolution.
- Diagnostics.
- Metadata.

Responsibilities:

- Preserve coordinate/value alignment.
- Preserve observable type.

Ownership:

- Produced by RT and instrument response.
- Stored by results.

### `Instrument`

Purpose:

- Represent instrument and observing mode metadata.

Required attributes:

- Name.
- Mode or response identifier.

Optional attributes:

- Line-spread function.
- Throughput.
- Resolving power.
- Detector/order/visit metadata.

Responsibilities:

- Provide metadata for instrument response construction.

Ownership:

- Owned by observation or retrieval problem.

### `Observation`

Purpose:

- Represent measured data.

Required attributes:

- Spectral coordinate.
- Observed values.
- Uncertainty or covariance.
- Observable type.

Optional attributes:

- Masks.
- Segment labels.
- Instrument metadata.
- Response modes.
- Provenance.

Responsibilities:

- Validate shape, finite data, and uncertainty/covariance.

Ownership:

- Owned by `RetrievalProblem`.

Lifecycle:

- Loaded before retrieval.
- Immutable.

### `ForwardModel`

Purpose:

- Convert physical parameters to observable model predictions.

Required behavior:

- `predict(parameters) -> ModelPrediction`.

Responsibilities:

- Build atmosphere.
- Evaluate opacity.
- Run RT.
- Apply instrument response.
- Return diagnostics.

Ownership:

- Owned by `RetrievalProblem`.

Lifecycle:

- Constructed once.
- Called repeatedly by likelihood/sampler.

### `Prior`

Purpose:

- Represent a parameter probability distribution.

Required behavior:

- `log_prob(value)`.
- `transform(unit_value)`.

Responsibilities:

- Validate support.
- Provide sampler-compatible transforms.

Ownership:

- Owned by `Parameter`.

### `Likelihood`

Purpose:

- Compute the probability of observations given model predictions.

Required behavior:

- `loglike(prediction, parameters) -> float`.

Responsibilities:

- Apply masks.
- Evaluate noise model.
- Handle invalid models deterministically.

Ownership:

- Owned by `RetrievalProblem`.

### `Sampler`

Purpose:

- Explore posterior distribution.

Required behavior:

- `run(problem, output) -> RetrievalResult`.

Responsibilities:

- Translate ROBERT problem into sampler-specific callbacks.
- Preserve seeds and resume state.

Ownership:

- User or CLI owns sampler adapter.

### `Posterior`

Purpose:

- Store posterior samples, weights, and evidence-like diagnostics.

Required attributes:

- Sample values.
- Parameter names.
- Weights or equal-weight marker.

Optional attributes:

- Evidence.
- Effective sample size.
- Diagnostics.

Ownership:

- Owned by `RetrievalResult`.

### `RetrievalResult`

Purpose:

- Stable output object for retrievals.

Required attributes:

- Posterior.
- Best-fit or representative parameters.
- Best-fit spectrum.
- Manifest.
- Schema version.

Optional attributes:

- Profile summaries.
- Contribution functions.
- Sampler raw output references.

Responsibilities:

- Provide sampler-independent access to results.

## 3. Ownership Summary

| Object | Primary owner |
| --- | --- |
| `Planet`, `Star` | `RetrievalProblem` |
| `Observation`, `InstrumentResponse` | `RetrievalProblem` |
| `PressureGrid` | Atmosphere setup |
| `OpacityDatabase` | Opacity provider configuration |
| `PreparedOpacity` | `OpacityProvider` |
| `AtmosphereState` | `ForwardModel` per evaluation |
| `Spectrum` | RT/instrument/result depending on lifecycle |
| `Prior` | `Parameter` |
| `Likelihood` | `RetrievalProblem` |
| `Sampler` | Caller or CLI |
| `Posterior`, `RetrievalResult` | Sampler adapter/result I/O |

## 4. Mutability Rules

- Config models are immutable after validation.
- Domain objects are immutable after construction.
- Transient evaluation objects are not mutated outside their owner.
- Caches may mutate internally but expose deterministic keys.
- Sampler state is the only long-lived mutable runtime state.
