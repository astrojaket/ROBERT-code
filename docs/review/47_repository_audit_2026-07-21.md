# Detailed repository audit

Date: 2026-07-21

## Executive summary

ROBERT is no longer a minimal forward-model skeleton. It is a coherent,
typed validation-retrieval package with 91 Python modules, 65 test modules,
87 Python examples, strict schema-v2 YAML workflows, and reproducible result
products. The strongest areas are immutable scientific containers,
correlated-k preparation, clear emission, absorption-dominated transmission,
shared-atmosphere multi-dataset orchestration, independent Gaussian
likelihoods, and the breadth of analytic and cross-framework RT tests.

The package can support controlled validation retrievals in its documented
regimes. It should not yet be presented as a general production science model.
The largest scientific gaps are independent science-opacity validation of
cloudy end-to-end calculations, transmission scattering-return physics,
calibrated pipeline ingestion and covariance, and long-run posterior validation.
The largest engineering gaps are the absent plugin system promised by the
architecture, low coverage in workflow orchestration, and no static
type-checking gate despite a typed public package. The 5.23 GiB Git-history
finding was resolved in the cleanup follow-up recorded below.

No critical defect was found. The audit fixed two high-impact correctness
issues: the published L 98-59 b loader could not verify its tracked data, and
PSIS-LOO allowed an unnormalized Gaussian likelihood even when retrieved
uncertainty parameters made the omitted normalization vary between posterior
draws. CI was also allowing the entire optional LOO test module to skip.

## Audited baseline

The working tree was clean before the audit. The required environment existed
at `/Users/jaketaylor/miniforge3/envs/robert-exoplanets`, although the active
Anaconda executable could not resolve it by name and its editable installation
pointed at a different Dropbox checkout. The environment was refreshed from
this workspace with the declared development, opacity, retrieval, and
diagnostics extras.

Before changes, the full suite collected 433 tests plus one module-level skip:

- 429 passed;
- two L 98-59 b loader tests failed on a checksum mismatch;
- one subprocess post-processing test failed because it imported the older
  Dropbox checkout;
- the complete LOO module skipped because ArviZ was absent;
- the native MultiNest smoke test skipped unless explicitly enabled.

The subprocess failure disappeared after correcting the editable installation;
it was environment contamination rather than a repository behavior failure.

## Capability matrix

| Subsystem | Implementation | Tests and validation | Documentation | Current limitation |
| --- | --- | --- | --- | --- |
| Core grids and spectra | Implemented, typed, immutable | Strong unit and invariant coverage | Good | Units are string conventions rather than a fully unit-aware internal algebra |
| Planet, star, observation | Implemented, validated containers | Strong unit coverage and real target loaders | Good | No calibrated JWST pipeline-product model |
| Atmosphere and temperature | Isothermal, tabulated, spline, Madhusudhan-Seager, and PG14-style profiles | Unit, integration, benchmark, and retrieval coverage | Good | Broader production parameterizations and domain validation remain limited |
| Chemistry and MMW | Constant/free, background closure, CLR/phantom composition, FastChem adapter | Strong profile tests; POSEIDON CLR parity; selected retrieval checks | Good | FastChem is optional at runtime; disequilibrium chemistry is absent |
| Stellar spectra and TSLE | Blackbody and STScI PHOENIX preparation; typed Rackham/POSEIDON disk-mixture transform for transmission | Unit/integration tests and exact POSEIDON v1.4 transform oracle | Good within tested transform regime | PHOENIX needs external `PYSYN_CDBS`; stellar grids, evolution, spot crossings, and surface maps are not validated |
| Correlated-k opacity | KTA, ROBERT archives, exo_k preparation, coverage checks, empirical target-bin preparation | Strong interpolation, archive, coverage, and cross-framework tests | Good | Production runs require external molecular data; extrapolation remains deliberately strict |
| Opacity sampling | ExoMol cross-section HDF path | Functional tests and benchmarks | Explicitly beta | Not the validated retrieval default |
| Extinction | Gas, CIA, Rayleigh, deck/haze, and Mie contributions | Analytic and regression coverage | Good | Aerosol database and continuum breadth remain limited |
| Clear thermal emission | Disk integration, hydrostatic options, contribution diagnostics | Analytic limits and pRT/PICASO comparisons | Strong | Validation is regime-specific, not a universal production guarantee |
| Cloudy thermal emission | Toon two-stream and P3/SH4 with Mie moments and delta-M | Grey, analytic, PICASO, and shared-cloud benchmarks | Strong and appropriately qualified | Independent end-to-end science molecular-opacity/cloud validation remains open |
| Transmission | Exact spherical chords, reference radius/pressure, correlated-k/CIA/Rayleigh, inverse-square gravity | Analytic, pRT, PICASO, convergence, and injection-recovery tests | Strong | Absorption dominated; scattering-return and refraction are not implemented |
| Instruments | Linear, top-hat, and stratified-sampling responses; multi-dataset grouping | Unit and multi-instrument tests | Good | No science-grade throughput/LSF convolution or calibrated pipeline interface |
| Likelihoods | Independent Gaussian terms with masks, offsets, jitter, and error scaling | Exact equation, masking, nuisance, and pointwise tests | Good | No covariance or correlated-noise likelihood |
| Retrieval | Sampler-independent problems, OE, UltraNest, MultiNest, and hybrid flows | Unit/integration tests, injection recovery, optional native smoke | Good | Native sampler and long-run posterior validation are not normal PR gates |
| Configuration and provenance | Strict schema-v2 YAML, manifests, status, portable JSON/NPZ results | Extensive schema and round-trip coverage | Very good | Version/release numbering has not kept pace with delivered capabilities |
| Post-processing | Fit statistics, residuals, posterior plots, sampler comparisons | Unit and subprocess workflow tests | Good | Publication-scale posterior diagnostics remain incomplete |
| PSIS-LOO | Pointwise Gaussian ELPD, Pareto-k, weighted resampling, plots and serialization | Focused tests through ArviZ after this audit | Good after correction | Independent likelihood units only; unreliable points still require exact refits |
| Plugins | Architectural protocols only | None | Target design exists | Registry, entry-point discovery, metadata validation, and example plugin are absent |

## Findings

### Critical

No critical finding was identified.

### High

#### H1. Bundled L 98-59 b data failed its own integrity check — fixed

Files: `src/robert_exoplanets/io/l9859b.py`,
`data/observations/l98_59b_bello_arufe2025/`.

The loader stored the SHA-256 of the upstream Zenodo byte stream, but the two
Git-tracked text files omit the final line-ending byte. Consequently, the
default `verify_checksum=True` path always raised before loading the data.
The tracked numeric and header contents were compared to Zenodo record
`10.5281/zenodo.14676143` and are otherwise byte-for-byte identical. The loader
now verifies the repository-normalized SHA-256, while the data README records
both upstream MD5 and repository SHA-256 provenance.

Impact: every default L 98-59 b configured workflow was broken despite the
published retrieval report claiming the loader was verified.

#### H2. Unnormalized PSIS-LOO could be invalid with retrieved uncertainty — fixed

Files: `src/robert_exoplanets/diagnostics/leave_one_out.py`,
`docs/leave_one_out.md`, `tests/test_leave_one_out.py`.

For fixed Gaussian uncertainties, omitting the normalization shifts each
pointwise term by a posterior-constant value. With retrieved jitter or
uncertainty scale, the normalization changes with every posterior draw and
therefore changes the PSIS importance ratios, ELPD, and Pareto-k diagnostics.
The previous documentation claimed these diagnostics were always unchanged.
ROBERT now rejects unnormalized LOO when a retrieved uncertainty nuisance
parameter is present and explains the narrower fixed-uncertainty condition.

Impact: direct Python-API users could otherwise obtain plausible but
mathematically inconsistent model-criticism results.

### Medium

#### M1. Optional LOO tests silently skipped in CI — fixed

The quality job installed `dev`, `perf`, and `opacity`, but not the declared
`diagnostics` extra. Because `tests/test_leave_one_out.py` used a module-level
optional import, CI could remain green without exercising the feature. The
quality job now installs `diagnostics`, and a static workflow test preserves
that contract.

#### M2. Capability and roadmap documentation had drifted — fixed in entry points

The README and package metadata described ROBERT as emission-only, while the
repository contains a tested transmission and retrieval stack. Conversely, the
development roadmap still listed transmission and JAX as post-v1 candidates
and mapped already-delivered work to v0.4-v0.10. The README, package metadata,
agent guide, and roadmap now state the present capability and retain clear
validation-versus-production boundaries.

Historical review documents remain intentionally dated snapshots and were not
rewritten.

#### M3. Git history was 5.23 GiB — resolved in cleanup follow-up

The checked-out content is modest, but `.git` contains a 5.23 GiB pack. The
largest objects are deleted 40-94 MB NPZ arrays formerly stored beneath
`docs/data/emission_intercomparison/`; many historical versions accumulate to
several gigabytes.

The follow-up inventory distinguished required lightweight inputs and compact
benchmark test oracles from generated arrays, plots, chains, deprecated
k-tables, and full benchmark products. The latter paths were removed from all
local branches and tags with `git filter-repo`. The packed Git object store is
now 3.27 MiB; the complete `.git` directory is 3.5 MiB. The largest retained
blob is the required 1.35 MB packaged CIA table.

A verified 5.2 GiB no-hardlink mirror of the pre-rewrite repository is retained
locally outside the checkout. Publishing the rewritten refs remains a separate,
explicitly coordinated force-push step.

#### M4. Static typing is declared but not enforced — unresolved

The distribution ships `py.typed` and exposes a large typed API, but no mypy or
Pyright configuration or CI job exists. Ruff does not validate type contracts.

Impact: protocol drift and incorrect annotations can reach users undetected.

Recommended action: introduce a scoped type checker over `core`, `bodies`,
`instruments`, `likelihoods`, and retrieval interfaces first; ratchet coverage
outward rather than requiring a repository-wide cleanup in one change.

#### M5. Architecture promises a plugin system that does not exist — unresolved

RFC-0001 and the interface documents describe plugin-first registries across
RT, chemistry, clouds, opacity, instruments, likelihoods, and samplers. No
plugin package, entry-point discovery, compatibility metadata, or third-party
example exists.

Impact: built-in factories and central configuration switches will become
harder to evolve as backends grow, and the documented extension contract is
not currently usable.

Recommended action: implement one narrow vertical slice—preferably a sampler
or chemistry plugin—before generalizing registries across every subsystem.

#### M6. Validation gates do not cover every supported deployment — unresolved

Normal CI runs on Linux and covers Python 3.10-3.14, lint, coverage, build, and
an exo_k job. It does not include the architecture guide's macOS smoke job,
documentation validation, native MultiNest smoke, external PHOENIX data, or
scheduled scientific/benchmark jobs.

Impact: platform, documentation, and optional-backend regressions can appear
outside the normal PR signal.

Recommended action: separate fast PR gates from scheduled/release gates, then
add macOS import/forward smoke, native sampler smoke, link/config checks, and a
small curated scientific matrix with archived provenance.

### Low

#### L1. Version and maturity metadata lag implementation — unresolved

The package remains `0.3.0` and `Development Status :: 2 - Pre-Alpha` while it
contains far more than the original v0.3 roadmap. Pre-alpha remains a prudent
scientific maturity label, but the numeric version no longer communicates the
amount of API change. Rebaseline the next release after deciding whether public
interfaces are ready for an explicit compatibility policy.

#### L2. Third-party plotting warnings are noisy — unresolved

The full test suite emits numerous Matplotlib/PyParsing deprecation warnings.
They originate in installed third-party versions rather than ROBERT code and do
not currently indicate failing behavior. Avoid hiding all warnings; constrain a
future filter to the known third-party messages or update the affected stack.

## Changes made

- `.github/workflows/ci.yml`: exercise the diagnostics extra in the coverage job.
- `AGENTS.md`: align project intent and repository layout with current code.
- `CHANGELOG.md`: record the correctness, CI, and audit documentation changes.
- `README.md`: correct maturity wording, include transmission and LOO, and link this audit.
- `pyproject.toml`: broaden the distribution summary from emission-only to atmospheric retrievals.
- `src/robert_exoplanets/__init__.py`: align the package docstring with that summary.
- `data/observations/l98_59b_bello_arufe2025/README.md`: distinguish upstream and normalized repository checksums.
- `src/robert_exoplanets/io/l9859b.py`: verify the bytes actually tracked by Git.
- `src/robert_exoplanets/diagnostics/leave_one_out.py`: guard parameter-dependent Gaussian normalization and correct result interpretation.
- `docs/leave_one_out.md`: document the mathematical limit accurately.
- `docs/architecture/development_roadmap.md`: add a current status rebaseline while preserving historical milestones.
- `tests/test_ci_workflow.py`: prevent optional diagnostics from disappearing from CI.
- `tests/test_leave_one_out.py`: add the variable-uncertainty normalization regression.

## Validation results

All commands used the Miniforge `robert-exoplanets` environment explicitly.
The environment was refreshed with:

```bash
/Users/jaketaylor/miniforge3/bin/conda run -n robert-exoplanets \
  python -m pip install -e '.[dev,opacity,retrieval,diagnostics]'
```

Completed checks:

- full suite with coverage: 439 passed, one intentionally gated native
  MultiNest smoke skipped, 75% branch coverage against the 70% floor;
- focused audit regression suite: 16 passed;
- lowest coverage: `io/configured_tasks.py` 29%, `rt/random_overlap.py` 32%,
  `io/model_setup.py` and `rt/thermal_integration.py` 63%, `rt/sh4.py` 66%;
- Ruff: passed;
- source and wheel distributions: built successfully in a temporary output
  directory;
- maintained CI example smoke set: all six commands completed, including YAML
  validation and the atmosphere/archive benchmarks;
- all 25 tracked YAML configurations parsed successfully;
- 76 Markdown files had zero missing local links;
- 62 documented Python command references had zero missing scripts;
- compile/import audit: 91 modules imported, 690 declared exports resolved,
  and 274 top-level public names were available;
- `pip check`: no broken requirements;
- import provenance: resolved to this workspace's `src/robert_exoplanets`.

The suite emitted 252 warnings, all from the installed Matplotlib/PyParsing
stack. No ROBERT warning or failure was hidden.

## Prioritized next-step plan

### Immediate

1. **Rebaseline the release and compatibility policy.**
   Outcome: version, changelog, roadmap, and stability promises describe the
   same software. Acceptance: a release RFC identifies stable/experimental
   APIs, validated regimes, deprecations, and the next version boundary.
2. **Make validation tiers executable.**
   Outcome: PR, scheduled, and release suites map directly to the testing guide.
   Acceptance: markers and CI jobs exist for unit/integration, optional
   backends, scientific fixtures, and benchmarks, with artifact provenance.
3. **Plan the Git-history cleanup.**
   Outcome: normal clones no longer transfer gigabytes of deleted NPZ arrays.
   Acceptance: required artifacts are archived externally with checksums and a
   coordinated migration plan has owner, date, rollback, and contributor steps.

### Near term

1. **Add covariance and scientifically valid likelihood grouping.**
   Outcome: multi-instrument correlated errors and grouped LOO units can be
   represented explicitly. Acceptance: positive-definiteness validation,
   analytic likelihood tests, serialization, config coverage, and one realistic
   fixture.
2. **Implement a narrow plugin vertical slice.**
   Outcome: the documented extension model is real rather than aspirational.
   Acceptance: entry-point discovery, metadata/version checks, no import-time
   side effects, and a toy third-party-style plugin tested in isolation.
3. **Introduce incremental static type checking.**
   Outcome: the `py.typed` contract is enforced. Acceptance: a CI gate covers
   stable domain and retrieval interfaces with a documented ratchet policy.
4. **Strengthen orchestration and optional-backend tests.**
   Outcome: configuration dispatch, result/status failure paths, native sampler
   adapters, and accelerated/reference parity receive targeted behavioral tests.
   Acceptance: module-specific coverage and mutation-sensitive assertions rise
   without weakening the 70% aggregate gate.
5. **Create a release-quality user path.**
   Outcome: a new user can install, validate config, run a small bundled
   forward/retrieval case, and interpret results without machine-specific data.
   Acceptance: one maintained tutorial is exercised in CI and all commands,
   outputs, limitations, and runtime expectations are verified.

### Later

1. Independent end-to-end cloudy validation using science molecular opacity,
   followed by high-stream and microphysical sensitivity studies.
2. Transmission scattering-return and, if justified by target regimes,
   refraction and stellar-contamination physics.
3. Calibrated JWST product ingestion, throughput/LSF models, and broader
   atmospheric/chemistry parameterizations.
4. Long-run posterior calibration, repeated-seed recovery, evidence stability,
   and production performance characterization.
5. Phase-curve/reflection modes only after the existing emission/transmission
   validation matrix and extension interfaces are stable.

## Trust boundary

ROBERT can currently be trusted as a transparent, testable framework for
controlled clear-emission, benchmarked cloudy-emission, and
absorption-dominated transmission validation retrievals using explicitly
prepared opacity and documented assumptions. Opacity-sampling, cloudy
end-to-end science interpretation, optional accelerated backends, and long-run
posterior conclusions remain experimental or regime-specific. ROBERT should
not yet be used as a general production pipeline for arbitrary JWST data,
correlated multi-instrument noise, or unconstrained cloud/scattering physics.
