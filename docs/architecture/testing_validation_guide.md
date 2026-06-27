# Testing & Validation Guide

ROBERT must be tested as both software and science.

## 1. Test Tiers

| Tier | Purpose | Runs in normal PR CI | Data allowed |
| --- | --- | --- | --- |
| Unit | Isolated function/object behavior | Yes | Tiny in-repo fixtures |
| Integration | Cross-package wiring | Yes | Tiny in-repo fixtures |
| Regression | Golden outputs | Yes when small, otherwise scheduled | Versioned fixtures |
| Scientific validation | Physical trust and benchmark agreement | Release-gated or scheduled | Curated validation data |
| Benchmark | Speed and memory trends | Scheduled | Representative data |

## 2. Unit Tests

Required for:

- Grid validation.
- Spectrum containers.
- Parameter transforms.
- Priors.
- Temperature/composition/cloud parameterizations.
- Likelihood equations.
- Config validation.
- Cache-key construction.

Rules:

- No network.
- No large data.
- No external opacity databases.
- No slow samplers.

## 3. Integration Tests

Required for:

- Config to object construction.
- Observation plus instrument response.
- Atmosphere plus opacity plus RT wiring.
- Retrieval problem log-likelihood call.
- Result serialization round trip.

Rules:

- Use small fixtures.
- Use deterministic seeds.
- Avoid external services.

## 4. Regression Tests

Purpose:

- Detect unintended numerical changes.

Required cases by v1.0:

- Clear 1D emission spectrum.
- Cloudy 1D emission spectrum.
- CIA/Rayleigh contribution case.
- Multi-instrument binned prediction.
- Tiny sampler smoke retrieval.

Rules:

- Store fixture provenance.
- Store expected outputs with schema version.
- Use explicit tolerances.
- Explain tolerance choices.

## 5. Scientific Validation Tests

Purpose:

- Establish scientific trust.

Validation sources:

- NEMESIS reference calculations.
- Published benchmark cases.
- Analytic limits such as isothermal or optically thin/thick behavior.
- Cross-framework comparisons where appropriate.

Recommended validation matrix:

| Case | Purpose |
| --- | --- |
| Isothermal clear emission | Basic RT sanity |
| Non-isothermal clear emission | T-P sensitivity |
| CIA-dominated atmosphere | Continuum validation |
| Rayleigh/haze case | Scattering/continuum sanity |
| Gray cloud deck | Cloud parameterization sanity |
| JWST binned emission | Instrument response validation |
| Multi-instrument offset case | Likelihood/nuisance validation |

## 6. NEMESIS Regression Strategy

NEMESIS should be used as a trusted reference where overlap exists.

Strategy:

1. Select small, reproducible NEMESIS cases.
2. Record source commit, input files, opacity metadata, and outputs.
3. Convert inputs into ROBERT typed objects.
4. Compare spectra and diagnostics under documented tolerances.
5. Store only small derived fixtures in the ROBERT repo.

Do not copy NEMESIS source code.

## 7. Cross-Framework Validation

Use other frameworks for conceptual and numerical checks:

- petitRADTRANS for emission/transmission and opacity mode sanity.
- TauREx for contribution and retrieval workflow comparisons.
- POSEIDON for instrument and stellar-contamination workflows.
- PICASO for reflected/scattering/climate ideas when those modes exist.
- CHIMERA and Brewster for brown-dwarf and cloud retrieval workflows.
- Exo_Skryer for JAX/backend and YAML/sampler design comparisons.

Cross-framework validation must record:

- Framework version/commit.
- Config assumptions.
- Opacity data differences.
- Known non-identical physics.

## 8. Benchmark Tests

Measure:

- Opacity preparation time.
- Forward-model evaluation time.
- Likelihood calls per second.
- Peak memory.
- Cache hit/miss behavior.

Rules:

- Benchmarks are not correctness gates.
- Benchmarks record hardware and backend.
- Benchmark regressions trigger review, not automatic rejection.

## 9. Continuous Integration

Minimum CI before v1.0:

- Python supported-version matrix on Linux.
- macOS smoke job.
- Unit and integration tests.
- Documentation build.
- Package build check.
- Optional dependency jobs for samplers/backends as they are added.

Markers:

- `slow`.
- `scientific`.
- `benchmark`.
- `optional_dependency`.

## 10. Minimum Contribution Expectations

| Contribution type | Required tests |
| --- | --- |
| Pure helper | Unit tests |
| Data model | Unit tests and validation failures |
| Config schema | Config success/failure tests |
| Parameterization | Unit tests and at least one scientific sanity check |
| Opacity provider | Coverage tests and interpolation regression |
| RT backend | Reference comparison tests |
| Instrument model | Binning/convolution tests |
| Likelihood | Known residual/covariance tests |
| Sampler adapter | Tiny smoke retrieval |
| Result writer | Round-trip serialization |

## 11. Reproducibility Tests

Every complete retrieval test should verify:

- Manifest exists.
- Config hash is recorded.
- Seed is recorded.
- Opacity identifiers are recorded.
- Re-running with same inputs reproduces expected deterministic parts.
