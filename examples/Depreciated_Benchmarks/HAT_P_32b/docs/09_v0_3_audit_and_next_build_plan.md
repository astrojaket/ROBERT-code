# v0.3 Audit and Next Build Plan

Date: 2026-07-02

Status: historical note. The current v0.3 RT/benchmark audit is
`docs/review/10_v0_3_rt_benchmark_audit.md`.

This note audits the current ROBERT state after the v0.3 foundation work and
maps the next steps against the NEMESIS/NemesisPy lessons and the local
HAT-P-32b emission model.

## Executive Assessment

ROBERT is on the right track.

The current implementation has deliberately built the same high-level workflow
that made the HAT-P-32b NemesisPy example useful:

```text
system/config -> pressure grid -> atmosphere state -> opacity boundary
              -> forward prediction -> observation grid -> likelihood/plots
```

but ROBERT has done this with typed domain objects, explicit validation, and
testable seams rather than raw dictionaries, hidden sidecar files, and broad
script-driven state. That is exactly the architectural correction identified in
the NEMESIS/NemesisPy review.

The main caution is that ROBERT should not jump directly from the current
placeholder forward model to a full radiative-transfer solver. The next work
should first pin the data conventions that the HAT-P-32b model depends on:

- pressure grid orientation and units,
- tabulated P-T interpolation,
- VMR shape and background H2/He policy,
- k-table metadata and coverage,
- benchmark comparison metrics.

## What Is Done

### Repository and Workflow

- GitHub repository connected: `astrojaket/ROBERT-code`.
- Draft PR open for v0.3: `#1`.
- Project-local conda environment supported via `environment.yml`.
- GitHub Actions CI added for tests and examples.
- Generated local outputs are ignored by git.

### Core Public Objects

- `PressureGrid`
- `SpectralGrid`
- `Spectrum`
- `Planet`
- `Star`
- `Observation`
- `RobertConfig`
- `RetrievalConfig`

These cover the minimum typed data model needed before adding real physics.

### v0.3 Forward-Model Foundation

- `AtmosphereState`
- `AtmosphereBuilder`
- `IsothermalTemperatureProfile`
- `ConstantChemistry`
- zero-opacity fixture provider
- prepared/evaluated opacity placeholder containers
- linear observation-grid response
- placeholder forward model returning native and observed spectra
- independent Gaussian likelihood with mask, offset, and jitter handling

The placeholder components are correctly labeled as placeholders and are tested
as pipeline wiring, not as physics.

### Diagnostics and Validation Anchors

- Planck blackbody helper.
- Blackbody eclipse-depth reference curve.
- Blackbody plotting example.
- External emission benchmark CSV loader.
- HAT-P-32b benchmark plotting example.
- Local HAT-P-32b benchmark path documented without committing large data.

This creates an early visual validation loop, which is important for emission
retrieval development.

### Tests

Current local suite covers:

- core grids and spectra,
- bodies and observations,
- atmosphere state construction,
- forward-model wiring,
- instrument response interpolation,
- Gaussian likelihood math,
- blackbody diagnostics,
- external benchmark CSV loading,
- legacy stub retrieval behavior.

## What Is Not Done Yet

These are known gaps, not failures:

- no real opacity reader,
- no `.kta` parser,
- no CIA table reader,
- no real radiative-transfer backend,
- no layer optical-depth calculation,
- no H2/He background-fill policy in ROBERT chemistry,
- no tabulated P-T profile component,
- no FastChem adapter,
- no run manifest,
- no sampler or retrieval problem,
- no config-file parser for the full HAT-P-32b workflow,
- no golden numerical comparison against the HAT-P-32b model output beyond
  loading and plotting the benchmark CSV.

## Alignment With NEMESIS/NemesisPy Lessons

### Good Alignment

ROBERT currently follows the strongest lessons from NEMESIS/NemesisPy:

- The forward model is separate from likelihood/retrieval.
- Atmosphere, opacity, instrument response, and likelihood are separate modules.
- Public behavior uses typed objects rather than raw dictionaries.
- Opacity preparation is already represented as a first-class boundary.
- Plots and benchmark loading are diagnostics, not computational core state.
- External data are referenced, not committed.
- Tests lock down the public skeleton behavior.

### Risks to Watch

The next work can drift off course if we:

- pass raw config dictionaries into opacity or RT kernels,
- let plotting scripts become the only way to run science workflows,
- silently extrapolate k-table or P-T coverage,
- hide units in filenames or assumptions,
- implement a clever fast RT kernel before a simple reference kernel is tested,
- treat the HAT-P-32b CSV as "truth" without also recording the exact input
  assumptions that produced it.

## HAT-P-32b NemesisPy Workflow Mapping

The local HAT-P-32b emission example does roughly this:

| NemesisPy example step | ROBERT target component |
| --- | --- |
| Python `CONFIG` dictionary | Typed config parser and `RobertConfig` extension |
| System block with `R_star`, `R_plt`, `T_star`, `M_plt`, `SMA` | `Planet`, `Star`, geometry/boundary metadata |
| Pressure grid block | `PressureGrid` |
| P-T CSV override | `TabulatedTemperatureProfile` |
| Temperature engine | temperature profile protocol |
| FastChem/free chemistry | chemistry profile protocol |
| H2/He background fill | explicit VMR background policy |
| k-table fetch for target wavelength grid | `.kta` database metadata and preparation |
| CIA file | CIA provider metadata and coverage |
| `EmissionForward.spectrum(...)` | cloud-free emission RT backend |
| `wavelength_um,value,bb_*` CSV | `Spectrum`, `EmissionBenchmark`, validation fixture |
| PT/VMR and emission plots | diagnostics examples |

This mapping should guide the next implementation order.

## Concrete Next Build Plan

### Step 1: Merge and Stabilize v0.3

Goal:

- Make the current branch the baseline.

Tasks:

- Review draft PR `#1`.
- Confirm GitHub Actions runs successfully.
- Merge once satisfied.
- Tag or record `v0.3.0` baseline if desired.

Acceptance criteria:

- `main` contains the v0.3 public skeleton.
- CI passes on GitHub.
- Local `.conda/bin/python -m pytest` passes.

### Step 2: Add HAT-P-32b Fixture Manifest

Goal:

- Record benchmark provenance before adding physical algorithms.

Tasks:

- Add a small validation manifest that points to the external HAT-P-32b CSV.
- Record expected columns, units, wavelength range, and model assumptions.
- Record the source script/config path used to generate the benchmark.
- Add a comparison helper that reports max/median fractional differences between
  two `Spectrum` objects on the same grid.

Acceptance criteria:

- ROBERT can load the benchmark and summarize its range and metadata.
- No large external files are committed.
- Tests use a tiny synthetic CSV fixture.

### Step 3: Add Tabulated P-T Profile Support

Goal:

- Reproduce the first atmosphere input used by the HAT-P-32b example.

Tasks:

- Add `TabulatedTemperatureProfile`.
- Load columns such as `pressure_bar` and `temperature_K`.
- Interpolate temperature in `log10(pressure)`.
- Make clipping/extrapolation policy explicit; default should reject uncovered
  pressure ranges unless configured otherwise.
- Add PT diagnostic plotting support using public objects.

Acceptance criteria:

- Tests cover interpolation, unit conversion, monotonic pressure handling, and
  out-of-coverage failures.
- The local HAT-P-32b PT CSV can be interpolated onto a ROBERT `PressureGrid`.

### Step 4: Strengthen Chemistry State

Goal:

- Match the VMR shape and background policy needed for H2/He atmospheres.

Tasks:

- Extend constant chemistry to support inactive/background gases.
- Add an explicit "fill remainder" policy for H2 or H2/He split.
- Validate VMR normalization by convention.
- Preserve species ordering for opacity lookup.

Acceptance criteria:

- Atmosphere state can represent H2O + H2 + He for the HAT-P-32b case.
- Background fill is visible in metadata and tests.
- No implicit chemistry defaults enter the RT path.

### Step 5: Add `.kta` Metadata Reader

Goal:

- Inspect opacity files safely before evaluating opacity.

Tasks:

- Define `OpacityDatabase` metadata object.
- Implement a read-only `.kta` header/metadata inspector.
- Extract species, wavelength coverage, pressure/temperature grids, g-ordinates,
  and checksum.
- Add coverage checks against `SpectralGrid`, `PressureGrid`, temperature, and
  species.

Acceptance criteria:

- The local HAT-P-32b `.kta` files can be inspected.
- Tests use a tiny synthetic or documented miniature fixture.
- Coverage failures are explicit and do not silently extrapolate.

### Step 6: Add CIA Metadata Reader

Goal:

- Represent the H2/He continuum input needed by the emission model.

Tasks:

- Add CIA database metadata object.
- Inspect supported pairs and temperature coverage.
- Add coverage validation against atmosphere state.

Acceptance criteria:

- CIA file availability and coverage are checked before RT.
- No RT code depends directly on file paths.

### Step 7: Build a NumPy Cloud-Free Emission Reference Backend

Goal:

- Produce the first real ROBERT emission spectrum.

Tasks:

- Add `RadiativeTransferBackend` protocol or base class.
- Implement a simple NumPy cloud-free thermal emission backend.
- Include Planck source function and layer optical-depth integration.
- Start with the simplest validated opacity path available.
- Return diagnostics such as total optical depth and contribution-like arrays.

Acceptance criteria:

- A toy analytic/isothermal case passes.
- A small HAT-P-32b comparison reports documented differences.
- The backend is slow but readable and independently testable.

### Step 8: Add Instrument/Validation Comparison Layer

Goal:

- Compare ROBERT predictions against NemesisPy-style output products.

Tasks:

- Add comparison functions for same-grid spectra.
- Add interpolation/rebin comparison for mismatched grids.
- Add a validation report writer for HAT-P-32b.

Acceptance criteria:

- Running one command/script produces:
  - ROBERT prediction,
  - HAT-P-32b benchmark overplot,
  - numerical residual summary,
  - manifest of inputs.

### Step 9: Retrieval Problem and Sampler Later

Goal:

- Avoid sampler work until forward-model correctness is anchored.

Tasks deferred:

- `ParameterSpace`
- priors,
- `RetrievalProblem`,
- sampler adapter,
- posterior/result schema.

Acceptance criteria before starting:

- Cloud-free emission backend has at least one benchmark comparison.
- Opacity/CIA coverage behavior is tested.
- Run manifest exists.

## Immediate Recommendation

The next code PR after v0.3 should be:

```text
Add validation manifest and tabulated P-T profile support for HAT-P-32b
```

That is the best next step because it moves directly toward reproducing the
NemesisPy HAT-P-32b setup while still avoiding premature radiative-transfer
implementation.
