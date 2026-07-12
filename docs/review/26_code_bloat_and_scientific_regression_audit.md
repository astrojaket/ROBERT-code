# ROBERT Code-Bloat and Scientific-Regression Audit

Date: 2026-07-12

## Scope

This audit covers the complete working tree, including the uncommitted RT,
cloud, transmission, multi-dataset, WASP-69b, and cross-framework benchmark
work present on the audit date. Existing work was treated as the current
implementation rather than compared only with `main`.

The review checked:

- explicit deprecation, compatibility, placeholder, and legacy markers;
- deprecated NumPy, SciPy, Python, and packaging API patterns;
- unreferenced and duplicate public definitions;
- package exports, importability, lint, tests, coverage, and distribution builds;
- every `examples/benchmark_*.py` entry point;
- deterministic scientific reports before and after cleanup, excluding timings.

## Findings

ROBERT did not contain any function formally marked as deprecated. It also did
not use the searched deprecated NumPy or SciPy aliases. The actionable bloat
was instead an obsolete prototype layer that remained publicly exported after
physical opacity, RT, likelihood, and retrieval implementations had replaced
it.

The obsolete layer contained:

- a zero-opacity fixture provider and its fixture-only data containers;
- a linear placeholder emission backend and placeholder forward-model wrapper;
- a second linear stub emission model and deterministic mock retrieval runner;
- two early configuration skeletons superseded by `RetrievalRunConfig` and the
  typed model-setup path;
- two stub examples and nine tests that tested only the deleted placeholders.

Two smaller dead-code findings were also confirmed:

- `core.logging.get_logger` only wrapped `logging.getLogger` and was not used;
- `opacity.inspectors.inspect_robert_npz_archive` duplicated and delegated to
  the actual implementation in `opacity.archive` while the package exported
  only the latter.

One benchmark-only defect was found. The HAT-P-32b emission benchmark computed
its spectrum successfully but attempted to pass an immutable `mappingproxy`
directly to `json.dumps`. Converting that metadata view to a plain dictionary
repairs report serialization without changing the calculation.

One notebook had a late import detected by Ruff. The import was moved to the
top of its cell; notebook behavior is unchanged.

## Removed Public Prototype API

The following names were removed because they exposed nonphysical or
superseded behavior:

- `CoverageReport`
- `EmissionModel`
- `EvaluatedOpacity`
- `FixtureOpacityProvider`
- `ForwardModel` (the placeholder implementation, not the architectural
  forward-model concept)
- `ModelPrediction` (the placeholder-specific container)
- `PlaceholderEmissionBackend`
- `PreparedOpacity` (the fixture-specific container)
- `RobertConfig`
- `RetrievalConfig`
- `StubRetrievalResult`
- `run_stub_retrieval`

The generic architecture documents still use terms such as `ForwardModel`,
`ModelPrediction`, and `PreparedOpacity` as interface concepts. Those concepts
remain valid; only their obsolete placeholder implementations were removed.

## Deliberately Retained

The following superficially legacy or lightly referenced code was retained:

- `.kta` readers and early archive-manifest compatibility, because they are
  explicit import adapters for real scientific data and have regression tests;
- the NumPy RT backend, because it is the scientific reference for accelerated
  implementations;
- public scattering, two-stream, Mie, SH4, opacity-inspection, and conversion
  functions even where repository-local call counts are low, because they
  expose real typed scientific behavior and are part of the documented public
  API;
- architecture and historical review references to prior skeleton milestones,
  because they are historical records rather than executable bloat.

Removing public scientific functions solely because this repository does not
call them would risk breaking external users without improving scientific
correctness. Further public-namespace reduction should therefore be an
intentional API-design change, not dead-code deletion.

## Verification

### Static and packaging checks

- Ruff: all source, tests, examples, and notebooks pass.
- Bytecode compilation: source and examples pass.
- Top-level API: all 181 names in `robert_exoplanets.__all__` resolve.
- Distribution: sdist and wheel build successfully with the installed build
  dependencies.
- Diff hygiene: `git diff --check` passes.

### Tests and coverage

- Before cleanup: 285 tests passed.
- After cleanup: 276 tests passed.
- The nine removed tests covered only the deleted prototype API.
- Branch coverage after cleanup: 77%, above the configured 70% threshold.

The only pytest warnings are Matplotlib-triggered PyParsing deprecation
warnings in the installed dependency stack. They do not originate in ROBERT.
Several benchmark processes also print an Intel OpenMP notice about
`omp_set_nested`; this likewise originates in the runtime dependency stack.

### Benchmark execution

All 15 benchmark programs completed successfully after cleanup:

| Benchmark | Result | Scientific check |
| --- | --- | --- |
| Atmosphere build | Pass | 100-layer build path executes |
| Cloud scattering PICASO/Virga | Pass | Deterministic report unchanged |
| HAT-P-32b opacity | Pass | All evaluator comparisons pass; report unchanged |
| HAT-P-32b retrieval plumbing | Pass | Deterministic OE report unchanged |
| HAT-P-32b transmission | Pass | Deterministic report unchanged |
| HAT-P-32b emission RT | Pass after report fix | Repeat reports identical except runtime; RMSE 16.7024768695 ppm |
| Native HDF emission convergence | Pass | Deterministic report unchanged |
| Opacity archive I/O | Pass | All four archive modes execute |
| petitRADTRANS 3 stable | Pass | Deterministic report unchanged |
| petitRADTRANS 3 multispecies emission | Pass | Deterministic report unchanged |
| petitRADTRANS 3 multispecies transmission | Pass | Deterministic report unchanged |
| petitRADTRANS 4 | Pass | Deterministic report unchanged |
| SH4 versus Toon/PICASO | Pass | Accuracy report unchanged |
| Taylor et al. (2021) Figures 1/2 | Pass | Deterministic report unchanged |
| WASP-69b multi-instrument | Pass | Both log likelihoods unchanged exactly |

Eleven pre/post JSON reports were directly comparable and matched after
timing-only fields were removed. The HAT-P-32b emission pre-cleanup execution
could not write a new report because of the serialization defect above; its
post-fix report was therefore run twice and matched exactly after removing only
`runtime_s`. The cleanup did not modify any atmosphere, opacity, RT,
likelihood, or inference equation.

## Environment Note

The shell's bare `python` currently resolves to Python 3.9, which is outside
ROBERT's declared Python 3.10--3.14 support range. Validation used the
`robert-exoplanets` Python 3.12 environment. The README already instructs users
to activate that environment before running ROBERT.

## Conclusion

The cleanup removes the nonphysical prototype surface, duplicate wrappers, and
their maintenance burden while retaining the scientific implementation and
data compatibility adapters. The available regression evidence shows no
scientific change from the cleanup.
