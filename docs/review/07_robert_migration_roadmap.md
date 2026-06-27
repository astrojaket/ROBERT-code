# ROBERT Migration Roadmap

This roadmap follows the review of:

- NEMESIS / Radtran: https://github.com/nemesiscode/radtrancode at commit `52c273f`.
- NemesisPy: https://github.com/astrojaket/NemesisPy at commit `afe2bae`.

The roadmap assumes ROBERT is a new framework informed by these projects, not a rewrite.

## Ground Rule

Do not implement major ROBERT physics until the review documents are accepted and the first validation fixtures are selected.

## Phase 0: Review Acceptance

Deliverables:

- Repository Inventory.
- Scientific Components Inventory.
- Architecture Review.
- Performance Review.
- Lessons Learned from NEMESIS and NemesisPy.
- ROBERT Design Specification.
- ROBERT Migration Roadmap.

Acceptance criteria:

- The scientific scope for the first real ROBERT milestone is agreed.
- The validation source cases are chosen.
- The team agrees which NEMESIS/NemesisPy behavior is reference behavior.

## Phase 1: Validation Fixture Selection

Goal:

- Establish reference cases before porting algorithms.

Candidate fixtures:

- NEMESIS Jupiter CIRS nadir thermal emission example.
- NEMESIS direct-imaged exoplanet example.
- A small NemesisPy emission spectrum with bundled k-tables/CIA.
- A minimal analytic/isothermal grey atmosphere toy case.

Outputs:

- `validation/fixtures/` data manifest.
- Expected spectra and tolerances.
- Notes on units and grid conventions.

Acceptance criteria:

- Each fixture can be run or its expected output can be loaded.
- ROBERT has a documented comparison metric for each fixture.

## Phase 2: Core Domain Model

Goal:

- Build typed objects before physics.

Implement:

- `Planet`
- `Star`
- `PressureGrid`
- `SpectralGrid`
- `AtmosphereProfile`
- `Observation`
- `Spectrum`
- unit/orientation validation

Acceptance criteria:

- No radiative-transfer physics yet.
- Objects reject invalid units/shapes/orientations.
- Tests cover construction, serialization, and equality/metadata behavior.

## Phase 3: Parameterization Layer

Goal:

- Convert named retrieval parameters into physical profiles.

Implement first:

- isothermal T(P),
- simple Guillot-style T(P) only if formula/source is agreed,
- constant-with-altitude VMR with H2/He background,
- explicit VMR normalization policy.

Acceptance criteria:

- Parameterizations are pure and tested.
- Required parameters and support bounds are discoverable.
- No opacity or sampler imports.

## Phase 4: Opacity Readers and Coverage

Goal:

- Read enough opacity data to run the first fixture.

Implement:

- NEMESIS `.kta` reader or adapter.
- CIA table reader for selected pairs.
- opacity coverage metadata.
- simple local cache.

Acceptance criteria:

- Reader tests validate shapes, units, pressure/T grids, g-ordinates.
- Coverage errors are explicit.
- No radiative-transfer kernel depends on file paths.

## Phase 5: Clear-Sky Emission Forward Model

Goal:

- First real scientific forward model.

Implement:

- layer construction,
- hydrostatic height,
- Planck function,
- k-table interpolation,
- random-overlap correlated-k,
- CIA contribution,
- clear-sky thermal emission,
- simple disc integration only if needed by fixture.

Acceptance criteria:

- Unit tests pass.
- Golden fixture comparison passes within documented tolerance.
- Backend is pure NumPy first; Numba optional after correctness.

## Phase 6: Instrument and Likelihood Layer

Goal:

- Compare model spectra with JWST-like observations.

Implement:

- observation binning/convolution,
- Gaussian likelihood,
- optional diagonal jitter/offset calibration parameters,
- multi-dataset likelihood composition.

Acceptance criteria:

- A fixed toy problem has deterministic log likelihood.
- Instrument response is tested independently of RT.
- No sampler-specific imports in likelihood code.

## Phase 7: Retrieval Engine

Goal:

- Run a small end-to-end retrieval.

Implement:

- `ParameterSpace`
- prior transforms,
- `RetrievalProblem`
- one sampler adapter, preferably a lightweight dependency first,
- result object and run manifest.

Acceptance criteria:

- Stub example becomes a real small retrieval example.
- Random seed and manifest make run reproducible.
- Forward model remains independently testable.

## Phase 8: Performance Backend

Goal:

- Optimize only after reference behavior is pinned.

Implement selectively:

- Numba backend for hot kernels.
- opacity/layer cache.
- benchmark suite.

Acceptance criteria:

- Same scientific tests pass against reference backend.
- Benchmarks report speed and memory.
- Backend selection is explicit.

## Phase 9: Expanded Physics

Add features one at a time:

- additional CIA pairs,
- Rayleigh scattering extinction,
- cloud opacity,
- scattering source functions,
- transmission spectra,
- phase curves,
- stellar contamination/TLSE,
- FastChem/equilibrium chemistry,
- optimal estimation with Jacobians,
- line-by-line backend.

Acceptance criteria for each:

- design note,
- reference fixture,
- unit tests,
- integration test,
- documentation of valid range and limitations.

## Phase 10: User-Facing Workflows

Goal:

- Make ROBERT usable without weakening the core architecture.

Implement:

- CLI or notebook workflows,
- config parser,
- JWST example templates,
- report generation,
- plotting outside computational core.

Acceptance criteria:

- CLI uses the public API.
- Plotting imports do not affect core imports.
- Example outputs include manifests.

## Migration Principles

- Prefer validated behavior over line-by-line translation.
- Port one scientific component at a time.
- Keep old and new outputs comparable.
- Do not add abstraction until a second use case justifies it.
- Keep I/O adapters at the edge.
- Keep physics independent from retrieval algorithms.
- Keep plotting independent from computational core.

## Initial Milestone Proposal

Milestone name:

- `M1-clear-sky-emission-reference`

Scope:

- One-dimensional clear-sky thermal emission.
- Correlated-k gas opacity.
- H2/He CIA.
- JWST-like binned emission likelihood.
- One reference NEMESIS or NemesisPy comparison.

Out of scope:

- Scattering source functions.
- Full clouds.
- Transmission retrieval.
- Phase curves.
- GPU/JAX.
- Full optimal estimation.

This is the smallest milestone that turns ROBERT from a skeleton into a scientifically anchored retrieval framework without importing historical architectural debt.

