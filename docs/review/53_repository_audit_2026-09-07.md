# ROBERT repository audit

Date: 2026-09-07.

## Decision

Keep the scientific pipeline and typed domain objects. Do not rebuild ROBERT
from scratch. The present module sizes, broad public API, and split workflow
paths are not the end state to aim for. Further feature work should reduce
those costs, with tests at the shared interfaces.

This audit covers the working tree, including substantial changes and new
files present before the audit. It is not an audit of a clean release commit.
Existing user work and scientific result records were retained.

## Scope and evidence

The initial inventory found 117 source Python files (50,285 lines), 112 test
Python files (22,440 lines), and 413 top-level exports. All declared root
exports resolved. A static scan found no forbidden absolute imports in the
checked `core`, `atmosphere`, `stellar`, `opacity`, and `rt` boundaries. This
scan does not establish the absence of every runtime dependency or cycle.

Code, tests, YAML definitions, user guides, architecture documents, CI, and
local validation records were inspected. External scientific benchmarks and
cluster retrievals were not rerun. Result claims below refer to the stored
records, not new measurements from this audit.

## Capability contract

| Expected function | Current route and evidence | Limit |
| --- | --- | --- |
| Typed grids, spectra, bodies, atmospheric state | `core`, `bodies`, `atmosphere`; validation and immutability tests | Preserve these boundaries |
| Clear and cloudy emission; transmission | Python models and schema-v2 YAML; analytic and numerical RT tests | Validation is regime-specific; transmission omits scattering return and refraction |
| Free/equilibrium chemistry and clouds | Atmosphere builders, optional FastChem, deck/haze and Mie paths | This is not a complete chemistry or aerosol database |
| Correlated-k and opacity sampling | Python and YAML; coverage/interpolation tests | External production tables need explicit preparation and identity |
| Line-by-line opacity and H-minus | Python providers, explicit VMR state, dedicated benchmarks | LBL is not an `opacity.format` option in standard YAML |
| Binned and high-resolution responses | Python binning, convolution, velocity, filtering, order/frame objects | Standard YAML does not assemble the complete HRS path; calibrated pipeline ingestion remains absent |
| Independent and correlated likelihoods | Gaussian YAML; Python dense/semidefinite covariance and mixed likelihoods | Independent named blocks do not represent arbitrary cross-instrument covariance |
| OE, UltraNest, MultiNest | Configured runners and Python adapters | Not every problem supports OE or pointwise LOO |
| Run manifests, results, post-processing | Standard runners and dedicated target scripts | The device-compiled path needs explicit compatibility tests |
| JAX and accelerator calculations | Optional reference-parity kernels and dedicated device graphs | CPU tests do not validate Metal/CUDA operation or float32 science accuracy |
| Third-party plugins | Protocols and target architecture documents | Registry, entry-point discovery, and compatibility checks are absent |
| Shared-VMR WASP-77Ab real-data retrieval | Dedicated runner and operator checks | Full real-data HRS/LRS/joint sampler measurements remain deferred |

The source of truth for YAML options is `io/task_config.py`. In particular,
`OpacityConfig.format` allows `exomol_kta` and `exomol_cross_section_hdf`, and
`LikelihoodConfig.model` allows `gaussian`. API availability must not be
presented as availability through every runner.

## Findings and actions

### High: accelerator adapter drift breaks test collection

The initial full suite stopped with eight collection errors. The Metal problem
imported `retrieval.accounting.LikelihoodAccounting`, which does not exist in
the current tree. This prevented import of the Metal package even before a
hardware check. The adapter must use the current sampler contract; restoring
an otherwise unused accounting abstraction would add debt.

### High: optimal estimation can discard likelihood structure

`gaussian_inputs_from_vector` passed diagnostic arrays into optimal estimation,
which builds `diag(1 / uncertainty**2)`. A correlated likelihood returns the
covariance diagonal in those diagnostic arrays. Its off-diagonal terms were
therefore lost. The same interface cannot establish that a profiled or
cross-correlation objective is an independent Gaussian objective.

Unsupported likelihoods must fail before optimization. Nested sampling still
uses their full scalar likelihood. Supporting covariance in OE later requires
an explicit residual/whitening or covariance contract, with objective and
Jacobian tests; returning diagonal uncertainties is insufficient.

### High: direct Python APIs bypassed the CLR prior boundary

CLR priors have a joint unit-cube transform. Their scalar `log_probability`
returned zero inside bounds, and the summed `log_prior_from_vector` exposed
that support check as a density. Direct Python OE also accepted CLR priors
through scalar Gaussian approximations, although YAML already rejected them.

Scalar CLR density and valid CLR posterior-density calls now raise a clear
configuration error. Direct OE also rejects CLR priors before model evaluation.
The existing joint transform remains available to nested samplers, which do
not call this density API. No new prior-density formula was introduced.

### Medium: mutable state and incorrect diagnostic labels

`GasOpticalDepth` was frozen but stored a mutable metadata dictionary. It now
uses the existing immutable-mapping helper. Full emission diagnostics also
labelled every gas source as correlated-k. They now use the stored opacity
mode, with `gas_correlated_k` retained as the default.

### Medium: optional tests depended on process state

The isolated JAX CPU suite exposed SH4 tests that expected float64 values and
tolerances without selecting float64. They passed with `JAX_ENABLE_X64=1`.
Precision must be scoped by the tests rather than inherited from another test.
The numerical tolerances remain unchanged.

### Medium: large modules combine separate responsibilities

At the start of the audit, `opacity/line_by_line.py` had 2,297 lines,
`instruments/high_resolution.py` 1,736, `rt/emission.py` 1,590,
`forward/emission.py` 1,490, `likelihoods/high_resolution.py` 1,471,
`io/configured_tasks.py` 1,277, and `postprocessing.py` 1,166.
Size alone is not a defect. The concern is that data preparation, validation,
model construction, and execution can change for different reasons in the
same module. Split these responsibilities as they change. Preserve public
imports during the split and compare outputs against current regression
fixtures. A repository-wide file move would add risk without proving benefit.

### Medium: API and workflow contracts need one owner

The root package exports 413 names, while the strict YAML path exposes only a
subset. The new high-resolution and accelerator code adds separate assembly
paths. Define a small sampler-facing protocol shared by all retrieval problems,
and test it with the supported adapters. Keep response and likelihood
construction reusable across standard and target-specific runners. Do not
expand root exports by default; use subpackage APIs for specialized work.

### Medium: architecture and quality gates differ from implementation

The architecture describes a plugin ecosystem, a static typing contract, and
separate validation tiers. General plugin discovery and a mypy/Pyright gate do
not exist. CI has Linux tests, lint, coverage, builds, opacity integration,
and HRS helpers, but no macOS hardware job or executable release-wide science
matrix. The testing guide's proposed markers are not configured in
`pyproject.toml`.

Add a scoped type gate for stable domain/problem interfaces. Make optional
backend checks explicit. Implement a narrow plugin interface when there is a
real external consumer; do not build ten registries to satisfy a diagram.

### Medium: documentation mixes policy, design, and measured status

The old roadmap described completed capabilities as future work. The data
policy repeated target methods and result tables already held in review notes
and JSON records. These copies can disagree when results change.

The roadmap now uses outcome gates. The data policy links to existing target
records and retains storage, provenance, VMR, and release rules. README and the
configuration guide state the Python/YAML boundary. Architecture interface
examples are marked as target designs rather than stable implemented APIs.
Historical reviews remain dated records.

The post-processing guide also claimed that plots needed no new RT evaluation.
`RetrievalResult` stores parameters and inference arrays, not a complete set of
best-fit spectra. Spectral and atmospheric post-processing calls the forward
model again. The guide now states the external-input requirement. A portable
prediction artifact would improve long-term reuse without changing inference.

## Design direction

Resume policy also needs a sampler-specific contract. The compatibility check
compares scientific fields and `invalid_loglike_floor`, while other sampler
settings can change between attempts. Increasing a call budget is intentional
and tested. Changing live-point or stopping settings needs explicit adapter
rules. Attempt manifests retain the new settings; do not treat the original
manifest alone as the full history. This audit leaves that policy unchanged.

If starting again, use the same scientific sequence:

```text
parameters -> atmosphere -> opacity -> RT -> response -> likelihood -> sampler
```

Keep immutable validated arrays, explicit units and geometry, prepared opacity
and response operators, optional numerical adapters, and portable provenance.
Use fewer assembly paths and fewer public convenience aliases. Give general
workflow code ownership of run setup and results; give target adapters ownership
of acquisition and observing conventions. Keep numerical kernels separate from
I/O and runtime selection.

The next acceptance gates are in the
[development roadmap](../architecture/development_roadmap.md). The highest
priority is reliable integration of existing features, followed by a
release-quality validation matrix. New physics should have a stated observing
need and a reproducible validation case.

## Verification

All Python checks used the `robert-exoplanets` Conda environment and this
workspace's editable package.

- Full suite with `JAX_PLATFORMS=cpu`: **805 passed, 28 skipped**.
- Coverage with branch measurement enabled: **72%**, above the 70% gate.
- Separate opt-in accelerator CPU suite: **37 passed**. This includes isolated
  SH4 tests with scoped float64 selection; no global x64 environment setting
  was needed.
- CLR and related config/inference checks: **72 passed**.
- Ruff and `git diff --check`: passed.
- All 40 maintained YAML configurations parsed; the documented R100 forward
  validation command completed.
- All 120 source modules imported. All 413 declared root exports resolved.
- Source and wheel builds passed. `pip check` found no broken requirements.
- Eighteen current/architecture documents had no missing local link targets.

The full-suite skips cover optional JAX CPU execution (checked separately),
real Metal hardware, and native MultiNest. No Metal/CUDA hardware validation or
long-run science retrieval was performed. The default local JAX backend
initialization aborted in the native plug-in; CPU selection avoided that
environment issue without changing ROBERT's backend policy. The suite emitted
369 warnings from the installed stack, including NumPy/exo-k binary-compatibility
and plotting deprecations. These warnings were not suppressed.

Cleanup retained numerical kernels and public imports. Shared private helpers
now own repeated Planck evaluation, optical-depth grid checks, gravity
validation, and opacity interpolation brackets. A duplicate opacity coefficient
conversion was removed. The obsolete Metal accounting hook was removed.
Six source-inspection or environment-presence tests and two source-string
assertions were removed (88 lines); runtime numerical, parser, integrity,
resource, and provenance checks remain. A release test now checks sampler
records rather than searching unrelated dirty-tree paths for sampler names.
New regressions cover the correctness fixes.
