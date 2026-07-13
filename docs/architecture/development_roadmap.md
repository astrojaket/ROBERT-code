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

## v0.3 - Forward-Model Foundation

Objectives:

- Build a non-retrieval forward-model pipeline.
- Add modular atmosphere, chemistry, opacity, and RT-facing components.
- Establish local benchmark and plotting workflows.

Deliverables:

- `AtmosphereState`.
- Temperature profiles: isothermal, tabulated, spline, Madhusudhan & Seager
  2009, and Parmentier & Guillot 2014 style.
- Free/background chemistry, mean-molecular-weight models, and optional
  FastChem equilibrium adapter.
- `.kta` import, ROBERT archive helpers, opacity coverage metadata, and
  correlated-k evaluation.
- Gas optical-depth assembly, random-overlap mixing, CIA/Rayleigh optical-depth
  terms, and tau/contribution plotting diagnostics.
- NumPy cloud-free thermal-emission reference solver with disc geometries and a
  first-order single-scattering source diagnostic.

Success criteria:

- A tiny clear-atmosphere forward model runs from config.
- Output includes native and observed-grid spectra.
- Tests cover public pipeline pieces.
- Local HAT-P-32b benchmark scripts run and write plots/JSON summaries.

## v0.4 - Benchmark RT Parity

Objectives:

- Close the remaining known gap between ROBERT and the local HAT-P-32b
  emission benchmark before retrieval integration.

Deliverables:

- Benchmark matrix for geometry/scattering/CIA/Rayleigh choices.
- Hydrostatic height/path geometry and reference-radius-pressure handling.
- Clear convention for layer boundaries, pressure orientation, and radius
  reference pressure.
- Cloud/aerosol optical-property containers for absorption, scattering,
  single-scattering albedo, and phase/asymmetry inputs.
- Decision note for the default fast opacity archive format.

Success criteria:

- The validation report documents agreement/tolerances and residual causes.
- Benchmark scripts can be rerun by a new user with local external inputs.
- RT-facing public APIs do not need breaking changes before sampler work.

## v0.5 - Instrument Support

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

## v0.6 - Retrieval Engine

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

## v0.7 - JWST Validation

Objectives:

- Establish initial scientific trust for JWST emission retrievals.

Deliverables:

- Clear 1D emission validation case.
- Cloud-free JWST binned example.
- External or published benchmark comparison where practical.
- Documentation tutorial.

Success criteria:

- Validation report documents agreement/tolerances.
- Tutorial can be run by a new user with small fixtures.

## v0.8 - Plugin Ecosystem

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

## v0.9 - Performance Optimization

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

## v0.10 - Scientific Validation

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
