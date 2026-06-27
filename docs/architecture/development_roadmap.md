# Development Roadmap

This roadmap stages ROBERT from architecture skeleton to stable scientific
release.

## v0.1 - Architecture and Infrastructure

Objectives:

- Establish repository skeleton.
- Produce RFC-0001 and companion architecture documents.
- Keep current example and tests running.
- Set distribution name to `robert-exoplanets`.

Deliverables:

- RFC-0001 document suite.
- Minimal package skeleton.
- Stub end-to-end example.
- Initial pytest suite.

Success criteria:

- `pytest` passes.
- Architecture docs are complete enough for review.
- No major physics implementation has started.

## v0.2 - Core Data Model

Objectives:

- Implement core immutable domain objects.
- Resolve final import namespace.
- Introduce config schema foundation.

Deliverables:

- `PressureGrid`, `SpectralGrid`, `Spectrum`.
- `Planet`, `Star`, `Observation`.
- Core exceptions and logging helpers.
- Initial YAML/Pydantic config schema.

Success criteria:

- Domain objects validate units, shapes, and invariants.
- Config can instantiate basic objects.
- Existing skeleton tests migrate to the new namespace.

## v0.3 - Minimal Forward Model

Objectives:

- Build a non-retrieval forward-model pipeline.
- Add simple atmosphere construction.

Deliverables:

- `AtmosphereState`.
- Isothermal temperature profile.
- Free constant chemistry.
- Minimal opacity fixture interface.
- NumPy reference emission placeholder upgraded to a documented physical
  minimal solver only after validation data exist.

Success criteria:

- A tiny clear-atmosphere forward model runs from config.
- Output includes native and observed-grid spectra.
- Tests cover each pipeline step.

## v0.4 - Instrument Support

Objectives:

- Support JWST-style observation handling and model-to-data transforms.

Deliverables:

- `InstrumentResponse`.
- Binning/convolution support.
- Multi-segment observations.
- Offset and jitter group metadata.

Success criteria:

- Synthetic JWST-like data can be loaded, modeled, and compared.
- Instrument response tests pass for known binning cases.

## v0.5 - Retrieval Engine

Objectives:

- Implement sampler-independent retrieval problem.
- Add first nested sampler adapter.

Deliverables:

- `Parameter`, `Prior`, `ParameterSpace`.
- `RetrievalProblem`.
- Gaussian likelihood.
- dynesty adapter.
- `RetrievalResult`.
- Run manifest writer.

Success criteria:

- Tiny end-to-end retrieval smoke test passes.
- Manifest records config hash, seed, versions, and opacity identifiers.

## v0.6 - JWST Validation

Objectives:

- Establish initial scientific trust for JWST emission retrievals.

Deliverables:

- Clear 1D emission validation case.
- Cloud-free JWST binned example.
- NEMESIS or published benchmark comparison where practical.
- Documentation tutorial.

Success criteria:

- Validation report documents agreement/tolerances.
- Tutorial can be run by a new user with small fixtures.

## v0.7 - Plugin Ecosystem

Objectives:

- Stabilize extension points.

Deliverables:

- Plugin registry.
- Entry-point discovery.
- Plugin metadata validation.
- Developer plugin guide.
- Example toy plugin.

Success criteria:

- A third-party-style plugin can register without modifying core code.
- Plugin compatibility errors are clear.

## v0.8 - Performance Optimization

Objectives:

- Optimize measured bottlenecks without changing public APIs.

Deliverables:

- Benchmark suite.
- Numba backend for one validated hot kernel if justified.
- Opacity/instrument response caching.
- Memory-use documentation.

Success criteria:

- Benchmarks quantify performance.
- Accelerated backend matches NumPy reference tests.

## v0.9 - Scientific Validation

Objectives:

- Prepare API and science release candidate.

Deliverables:

- Expanded validation matrix.
- Multi-instrument JWST emission case.
- Cloudy atmosphere case.
- Optional covariance likelihood if ready.
- API freeze proposal.

Success criteria:

- No known architecture blockers for v1.0.
- Validation docs are release-quality.

## v1.0 - Stable Scientific Release

Objectives:

- Provide a stable, documented JWST 1D emission retrieval platform.

Deliverables:

- Stable public API.
- Stable config schema v1.
- Stable result schema v1.
- Validated 1D emission retrieval pipeline.
- Plugin discovery.
- Complete docs and tutorials.
- CI/release workflow.

Success criteria:

- Independent developers can contribute compatible components.
- Users can run documented JWST emission retrieval examples.
- Scientific validation is documented and reproducible.

## Post-v1.0 Candidates

Possible future features:

- Transmission retrievals.
- Patchy/column atmospheres.
- Phase curves.
- Reflection spectroscopy.
- Multiple chemistry backends.
- JAX/GPU backend.
- Additional sampler adapters.

These should not destabilize the v1.0 emission architecture.
