# Performance Strategy

ROBERT optimizes only after correctness, tests, and interfaces are stable.

## 1. Performance Principles

- Reference implementations come first.
- Faster backends must preserve public interfaces.
- Optimization choices are measured, not guessed.
- Runtime configuration is explicit.
- Import-time performance side effects are forbidden.

## 2. Backend Strategy

### NumPy

Role:

- Reference backend.
- First implementation for all core algorithms.

Requirements:

- Clear equations.
- Thorough tests.
- Used as the comparison target for accelerated backends.

### Numba

Role:

- Optional CPU acceleration for stable kernels.

Allowed when:

- NumPy reference exists.
- Tests cover numerical behavior.
- Speed benefit is measured.

Avoid:

- Letting Numba constraints dictate public APIs.
- Duplicating large amounts of logic without tests.

### JAX

Role:

- Future optional differentiable/GPU backend.

Allowed when:

- CPU reference is stable.
- Cache/device semantics are understood.
- Tests compare against NumPy.

Rules:

- JAX platform selection happens at runtime, not import.
- GPU nondeterminism is documented.
- JAX is not a required dependency for core installation.

### Compiled Extensions

Role:

- Future use for stable hot kernels if needed.

Requirements:

- Clear build isolation.
- Reference tests.
- Optional unless essential for v1.0 performance.

## 3. Vectorization

Use vectorization when it:

- Improves clarity.
- Removes Python loops from hot paths.
- Preserves shape readability.

Avoid vectorization when it:

- Obscures scientific equations.
- Creates large temporary arrays that dominate memory.
- Makes validation harder.

## 4. Caching

Cache candidates:

- Opacity interpolation indices.
- Prepared opacity arrays.
- Instrument response matrices.
- Chemistry interpolation grids.
- Repeated spectral grid transforms.

Cache keys must include:

- ROBERT version.
- Backend.
- Species list.
- Pressure grid.
- Spectral grid.
- Opacity checksums.
- Chemistry/cloud assumptions.
- RT mode.

Rules:

- Caches are disable-able.
- Cache hits/misses are debug-logged.
- Science-relevant cache metadata is recorded in manifests.

## 5. Parallelism

Supported strategies:

- Multiprocessing for independent forward-model evaluations.
- Sampler-native parallelism.
- MPI through optional sampler/runtime adapters.

Rules:

- No MPI initialization at import.
- No forced BLAS/OpenMP thread changes at import.
- Parallel settings live in runtime config and manifest.
- Serial execution remains supported.

## 6. GPU Support

GPU support is future optional functionality.

Requirements:

- CPU reference parity.
- Explicit runtime selection.
- Device and precision recorded in manifest.
- Documented fallback behavior.

GPU support should not:

- Be required for installation.
- Change user-facing data models.
- Make simple CPU tests fail.

## 7. Memory Management

Rules:

- Avoid copying large opacity arrays in hot loops.
- Prefer immutable prepared arrays.
- Use lazy loading or memory mapping where appropriate.
- Document expected memory for benchmark cases.

Memory-sensitive components:

- Correlated-k opacity cubes.
- Line-by-line/opacity-sampling data.
- Multi-instrument response matrices.
- Posterior predictive spectra.

## 8. Performance Testing

Benchmarks should measure:

- Setup time.
- Opacity preparation time.
- Single forward-model call.
- Likelihood calls per second.
- End-to-end tiny retrieval time.
- Peak memory.

Benchmarks should report:

- Hardware.
- Python version.
- Backend.
- Optional dependencies.
- Data fixture.
- Cache state.

## 9. Performance Anti-Patterns

Avoid:

- Premature GPU rewrites.
- Hidden global caches.
- Import-time backend setup.
- Optimizing unvalidated physics.
- Making performance-specific array layouts leak into public APIs.
- Broad dependency additions for small speedups.
