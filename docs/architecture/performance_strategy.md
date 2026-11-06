# Performance Strategy

ROBERT optimizes only after correctness, tests, and interfaces are stable.
Status reviewed: 2026-11-06. The
[BeAR performance audit](../review/58_bear_2026_performance_audit.md) records
source evidence, candidate improvements, and measurable completion gates.

## 1. Performance Principles

- Reference implementations come first.
- Faster backends must preserve public interfaces.
- Optimization choices are measured, not guessed.
- Runtime configuration is explicit.
- Import-time performance side effects are forbidden.

For Python specifically, ROBERT should be fast by keeping the hot path small:
use NumPy arrays for layer/spectral calculations, precompute reusable indices
and matrices before repeated model calls, avoid Python loops inside opacity and
radiative-transfer kernels, and only add compiled acceleration after profiling
identifies a stable bottleneck.

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

- Implemented optional experimental CPU/GPU paths, including JAX Metal and
  direct opacity sampling. These are not general production replacements.

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

GPU paths are implemented as optional experimental functionality. The current
Numba/NumPy route remains the production reference. Existing component parity
and single-proposal timings do not establish posterior equivalence, native
hardware accuracy, or long-run memory safety. See the
[supported-methods boundaries](../supported_methods.md#accelerator-boundaries).

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
- Host/device transfer volume and synchronization cost for accelerator paths.
- Full sampler wall time and number of likelihood calls, separately from
  fixed-proposal throughput.

Compare matched physics, opacity, grids, data, and proposal vectors. Record
compilation and setup separately from warm execution. Synchronize asynchronous
device work before timing it. Keep numerical error, posterior calibration, and
evidence checks beside each speed result. A direct-cross-section GPU timing is
not a speed measurement for the correlated-k/Virga/SH4 workflow.

Benchmarks should report:

- Hardware.
- Python version.
- Backend.
- Optional dependencies.
- Data fixture.
- Cache state.

Current lightweight benchmark helpers live in `robert_exoplanets.diagnostics`.
Use `time_callable` for smoke benchmarks that should stay dependency-free, and
use dedicated profilers for deeper kernel work. The atmosphere-build smoke
benchmark can be run with:

```bash
python examples/benchmark_atmosphere_build.py
```

The correlated-k random-overlap scaling benchmark exercises the principal
multi-molecule opacity-mixing kernel independently of data loading and RT:

```bash
python examples/benchmark_random_overlap.py
```

It reports molecule/grid scaling, input memory, throughput, and agreement with
the NumPy reference backend. The WASP-69b benchmark measures warmed full-model
calls and can also write a `cProfile` capture without including setup, warmup,
or Numba compilation:

```bash
python examples/benchmark_wasp69b_multi_instrument.py \
  --profile wasp69b-forward-model.prof
```

The matched six-gas clear versus MgSiO3/Mie/SH4 benchmark measures the cloudy
multiple-scattering overhead and can profile both warmed paths:

```bash
python examples/benchmark_wasp69b_mgsio3_cloud.py \
  --profile-clear clear.prof \
  --profile-cloudy cloudy.prof
```

The baseline, implemented Numba spectrum-only results, and remaining
physics-preserving SH4 acceleration plan are recorded in
`docs/review/33_mgsio3_cloud_forward_performance_plan.md`.

Environment variables `ROBERT_BENCH_REPEAT` and `ROBERT_BENCH_WARMUP` control
the number of measured and warmup calls.

## 9. Performance Anti-Patterns

Avoid:

- Premature GPU rewrites.
- Hidden global caches.
- Import-time backend setup.
- Optimizing unvalidated physics.
- Making performance-specific array layouts leak into public APIs.
- Broad dependency additions for small speedups.
