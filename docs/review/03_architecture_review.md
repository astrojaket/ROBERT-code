# Architecture Review

Sources inspected:

- NEMESIS / Radtran: https://github.com/nemesiscode/radtrancode at commit `52c273f`.
- NemesisPy: https://github.com/astrojaket/NemesisPy at commit `afe2bae`.

## Executive Assessment

NEMESIS is a scientifically mature Fortran application suite whose architecture reflects decades of extension under fixed-memory Fortran, file-based workflows, and instrument-specific needs. Its science should be respected, but its architecture should not be copied.

NemesisPy is a useful Python bridge from NEMESIS into exoplanet workflows. Its Numba kernels and modular engine direction are valuable, but the inspected repository still mixes source, build artifacts, examples, data processing, plotting, MPI, I/O, and prototype code.

ROBERT should preserve scientific algorithms and validation cases, not historical coupling.

## NEMESIS Detailed Review

### Strengths

- Clear long-term scientific lineage.
- Many complete worked examples with real run files.
- Mature Radtran subsystem with opacity, path, scattering, and RT capabilities.
- Optimal-estimation machinery includes covariance, averaging kernels, gain matrix, and forward-model error propagation.
- Supports multiple observing geometries, limb paths, disc-averaged spectra, scattering modes, secondary transit, and primary transit variants.

### Tight Coupling

- Run-name sidecar files are both API and storage layer.
- `gsetrad`/`subprofretg` modify files that downstream RT routines read.
- Common blocks (`/diagnostic/`, `/solardat/`, path/layer include state, etc.) create implicit global inputs.
- Retrieval loop, physical support checks, profile generation, and radiative-transfer dispatch are interwoven.
- Instrument/geometric variants are split across separate executables and near-duplicate routines.

ROBERT implication:

- Treat sidecar file formats as legacy import/export fixtures only.
- Internal APIs should pass structured state, not write temporary files as function arguments.

### Duplicated Code

Examples:

- `nemesis`, `nemesisL`, `nemesisMCS`, `nemesisPT`, `nemesisdisc`.
- `coreret`, `coreretL`, `coreretMCS`, `coreretPT`, `coreretdisc`.
- `forwardnogX`, `forwardnogMCS`, `forwardnogPT`, `forwardnoglbl`, `forwardavfovX`, Venus/disc variants.
- Many model-specific extraction and modification routines.

Historical reason:

- Separate executables reduced risk when adding specific observing modes and avoided invasive rewrites.

ROBERT implication:

- Use one retrieval engine with pluggable geometry/instrument/RT backends.
- Avoid variant-specific top-level applications unless the variant is truly a separate product.

### Global State and Hidden Assumptions

- Compile-time arrays in `arrdef.f` and `arraylen.f`.
- Common blocks for solar data, scattering phase data, diagnostics, Mie flags, path/layer state.
- Assumed current working directory and fixed file names such as `aerosol.ref`.
- Unit conventions embedded in comments and file formats.
- Integer model IDs encode physical meaning.

ROBERT implication:

- Use explicit dimensions, units, and schemas.
- State transforms should be inspectable and serializable.
- No hidden current-working-directory dependency in computational core.

### Oversized Modules

- `subprofretg.f` is a large concentration of profile parameterization logic.
- `coreret.f` contains retrieval setup, constraint enforcement, forward calls, acceptance logic, reduced-wavelength behavior, file diagnostics, and final covariance.
- `cirsradg_wave.f` combines opacity interpolation, continuum, CIA, scattering flags, RT accumulation, and gradients.

ROBERT implication:

- Split responsibilities:
  - parameter transform,
  - physical validation,
  - opacity interpolation,
  - RT integration,
  - derivative mapping,
  - retrieval algorithm.

### Unclear Interfaces

- Many routines accept long argument lists plus hidden common-block state.
- Output may be returned through arrays, common blocks, or generated files.
- Several flags (`iscat`, `ilbl`, `iform`, `lin`) alter major control flow but are low-level integers.

ROBERT implication:

- Use typed enums for RT mode, opacity mode, geometry, and output units.
- Use named result objects.
- Make unsupported combinations fail early.

### Difficult-to-Test Code

- Many routines depend on file system state.
- Global common blocks make independent tests hard.
- Compile-time limits and platform-specific record lengths affect behavior.
- Reference examples exist, but automated unit/regression test boundaries are not apparent.

ROBERT implication:

- Create test fixtures from NEMESIS examples.
- Add small unit tests around isolated pure functions.
- Keep integration tests separate from scientific golden-data tests.

### Legacy Design Patterns

- Fixed-size arrays.
- Common blocks.
- Side-effect files as intermediate data.
- Goto-based control flow in retrieval convergence.
- Manual matrix routines and custom inversion checks.

ROBERT implication:

- Use modern arrays and typed interfaces.
- Keep numerical behavior, but modernize control structure.
- Use tested linear algebra libraries while preserving reproducibility checks.

## NemesisPy Detailed Review

### Strengths

- Brings core exoplanet RT workflows into Python.
- Numba kernels make performance-oriented code readable and inspectable.
- Clear scientific sequence in `calc_radiance.py` and `calc_tau_gas.py`.
- Modular engines are moving toward separation of temperature, chemistry, clouds, stellar effects, and k-table I/O.
- Supports practical exoplanet needs: phase curves, disc integration, transmission, offsets, bin widths, FastChem, stellar heterogeneity.

### Tight Coupling

- `modular.rt.forward_model.ForwardModel` wraps a mutable `radtran.forward_model.ForwardModel` backend.
- Backend holds planet and opacity data in object attributes and methods read from those attributes.
- Config dictionaries are passed broadly and interpreted at many levels.
- Cloud engine calls backend RT methods directly and applies observation offsets.
- K-table I/O uses plotting, MPI, and external `exo_k` in the same module.

ROBERT implication:

- Use explicit state/result objects and dependency injection.
- Keep plotting and MPI out of core I/O.
- Validate config into typed domain objects once, then avoid raw dict access in physics.

### Duplicated and Stale Code

Observed examples:

- `calc_radiance.py`, `calc_radiance_old.py`, `calc_radiance_jo.py`, `calc_radiance_backup_sept24.py`, `calc_radiance_pref_test.py`.
- `forward_model.py`, `forward_model_old.py`, `forward_model_pref_test.py`.
- `calc_layer.py`, `calc_layer_new.py`, `calc_layer_test.py`.
- Multiple cloud variants with `_old`, `_jo`, `_hazemult_test`.
- Committed `build/lib`, `.pyc`, `dist`, `.egg-info`, and `__MACOSX` artifacts.

ROBERT implication:

- Keep experimental variants in branches, notebooks, or examples, not importable package paths.
- Enforce package hygiene in CI.

### Circular and Unhelpful Dependency Direction

The package-level direction is mostly:

```text
modular -> radtran -> common
```

That is reasonable. Problems arise because:

- `data/` imports `radtran` and models for plotting/fitting examples.
- `modular` imports backend internals directly.
- setup, prior extraction, chemistry, and forward evaluation all share raw config dictionaries.

ROBERT implication:

- Keep `examples/` and `datasets/` outside core imports.
- Allow high-level application code to import physics, never the reverse.
- Define stable backend interfaces so wrappers do not depend on concrete internals.

### Global State and Hidden Assumptions

- Module import attempts for optional dependencies may change behavior.
- MPI rank globals are created at import time in `modular/io/ktables.py`.
- `matplotlib.use("Agg")` is set in an I/O module.
- Environment variables such as `PYSYN_CDBS` are changed inside stellar helper code.
- Mutable backend `ForwardModel` needs `set_planet_model` and `set_opacity_data` before evaluation.

ROBERT implication:

- Optional dependencies should be plugin/adapters with explicit availability checks.
- No MPI or plotting side effects at import time.
- Use constructors that create valid objects or raise.

### Unclear Interfaces

- Raw dict configs and params make valid keys hard to discover.
- Parameter names are sometimes aliases (`Tirr`, `T_irr`, `Kappa_IR`, etc.).
- Units are present in comments but not type-enforced.
- Observation dicts carry behavior flags (`fit_offset`, `fit_error_inflation`) rather than typed calibration models.

ROBERT implication:

- Use typed schemas and canonical parameter names.
- Support aliases only in config migration layers.
- Units should be explicit at I/O boundaries and standardized internally.

### Difficult-to-Test Code

- Numba JIT can hide Python exceptions and complicate small tests.
- External dependencies (`exo_k`, `h5py`, `mpi4py`, FastChem, pysynphot) are not fully isolated.
- The retrieval package is empty, while examples call samplers directly.
- Some functions do file loading, rebinning, plotting, and MPI coordination together.

ROBERT implication:

- Separate pure numerical kernels from file/network/HPC orchestration.
- Provide small deterministic fixtures for each numerical kernel.
- Provide slow reference tests and fast smoke tests.

### Performance-Oriented Design Risks

- JIT kernels are optimized around explicit loops but allocate large arrays repeatedly.
- Mutable backend attributes make caching possible but opaque.
- Runtime code may reconfigure planet model or opacity state repeatedly.

ROBERT implication:

- Design explicit caches for opacity interpolation and layer geometry.
- Make cache keys visible and test invalidation.
- Optimize after correctness benchmarks.

## Cross-Cutting Architecture Findings

### What Should Be Preserved

- NEMESIS scientific validation cases and file formats as reference fixtures.
- Correlated-k random-overlap algorithm behavior.
- Layer/path geometry semantics.
- CIA and continuum treatment, with coverage metadata.
- Optimal-estimation outputs: averaging kernels, covariance decomposition, gain matrix.
- Exoplanet-specific modular ideas from NemesisPy: chemistry engines, stellar contamination, multi-observation handling, k-table rebinning.

### What Should Be Modernized

- State-vector/profile mapping.
- Opacity readers and cache management.
- K-table interpolation API.
- Instrument/binning/convolution handling.
- Sampler integration.
- Configuration and validation.
- Packaging and optional dependencies.

### What Should Be Redesigned Completely

- Sidecar files as internal API.
- Common-block/global-state control flow.
- Integer-coded physical model registry.
- Retrieval loop embedding physical support constraints.
- Script-level sampler usage.
- Import-time MPI/plotting/environment mutation.

## Review by Future ROBERT Package Area

| Future ROBERT area | Lesson from NEMESIS/NemesisPy |
| --- | --- |
| `robert.core` | Needs typed state, units, and grids; no file side effects. |
| `robert.opacity` | Must isolate file formats, interpolation, coverage, and caching. |
| `robert.rt` | Should expose small backend-neutral RT interfaces and reference-tested kernels. |
| `robert.parameterizations` | Must replace integer model IDs with named transforms and support metadata. |
| `robert.instruments` | Must own binning, convolution, offsets, and multi-instrument calibration. |
| `robert.retrieval` | Must own likelihoods, prior transforms, and sampler adapters. |
| `robert.io` | Should translate legacy files and user configs into typed objects. |
| `robert.validation` | Should store NEMESIS/NemesisPy comparison fixtures and tolerances. |

