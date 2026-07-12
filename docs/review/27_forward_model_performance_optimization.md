# Forward-Model Performance Optimization

Date: 2026-07-12

## Scope and method

This pass profiled warmed forward-model calls before changing numerical code.
The representative workload was the 80-layer, six-molecule WASP-69b
multi-instrument model with 16 correlated-k ordinates on both its 1,591-point
native opacity grid and its 280-point observation grids. Separate profiles
covered SH4 emission, spherical transmission, opacity interpolation, and
random-overlap scaling.

The optimization gate was scientific equivalence with the existing NumPy
reference paths and benchmark fixtures. No layer, wavelength, angular, or
correlated-k resolution was reduced. No equation, convergence threshold,
precision, or physical approximation was changed.

Measurements below were made on macOS arm64 with Python 3.12.13, NumPy 2.2.6,
and Numba 0.61.2. Absolute timings vary with machine load; ratios and profile
hotspots are the useful quantities.

## Profile findings

The initial warmed retrieval path was dominated by correlated-k random
overlap. Its implementation sorted all `n_g**2` pairwise optical depths and
rescanned them for every target g interval at every layer and wavelength.
Repeated opacity evaluation also recalculated logarithms of immutable opacity
tables and repeatedly searched an unchanged spectral grid. SH4 source
reconstruction made thousands of small `einsum` calls. By contrast, the
80-layer spherical-transmission solve took about 30 ms and was not worth a
more invasive rewrite.

## Implemented changes

### Random overlap

The accelerated backend now rebins each sorted pair distribution in a single
streaming pass and reuses work buffers. Correlated-k inputs are nondecreasing
along the g axis, so the common path merges sorted outer sums with an `n_g`
element heap instead of sorting an `n_g**2` temporary. Unsorted caller input
retains the general argsort fallback, preserving public behavior.

The NumPy implementation remains the scientific reference. The new scaling
benchmark tests 1, 3, and 6 species at 280 and 1,591 wavelengths, records
throughput and memory, and checks a deterministic subset against NumPy on each
run. The largest observed absolute difference was `3.13e-13` in optical depth.

### Correlated-k interpolation

Prepared opacity objects now retain their validated spectral lookup indices,
removing repeated exact-grid searches from every model evaluation. Providers
using log-pressure/temperature/log-k interpolation can cache the logarithm of
their immutable coefficient tables. The cache is enabled by default, recorded
in prepared metadata, and can be disabled with `cache_log_kcoeff=False` when
memory is more important than evaluation speed.

This is an explicit speed/memory tradeoff. In the WASP-69b workload the four
observation-grid providers use about 106 MiB for cached log coefficients; the
full native-grid provider uses about 599 MiB. The coefficient values and
interpolation arithmetic remain float64. Cached and uncached paths are tested
for exact array equality.

### SH4 source reconstruction

Four fixed, four-term P3 source contractions now use their explicit sums
instead of dispatching many tiny general-purpose `einsum` operations. The
underlying SH4 equations and solve are unchanged.

## Performance results

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| WASP-69b, one native-grid model then instrument binning | 6.35 s | 1.21 s | 5.3x |
| WASP-69b, per-mode opacity binning and four model calls | 1.34 s | 0.43 s | 3.1x |
| pRT3 multispecies emission, ROBERT steady call | 3.12 s | 0.83 s | 3.8x |
| pRT3 multispecies transmission, ROBERT native case | 4.70 s | 1.12 s | 4.2x |
| Native-HDF emission convergence, load/evaluate all resolutions | 14.64 s | 4.56 s | 3.2x |
| SH4 benchmark, 64 x 900 x 4 grid with six angles | 0.87 s | 0.61 s | 1.4x |

The WASP-69b likelihoods were unchanged exactly at
`-61446.289006481966` for native-then-bin and `-61339.66439309031` for
per-mode opacity preparation. The two strategies intentionally need not agree
with each other because correlated-k recompression before RT is not equivalent
to integrating a native-grid spectrum after RT.

## Scientific and software verification

- All 16 `examples/benchmark_*.py` programs completed successfully.
- All 278 tests passed.
- Branch coverage was 76%, above the configured 70% requirement.
- Random-overlap NumPy/Numba differences stayed below `3.13e-13`.
- Cached and uncached log-k interpolation were exactly equal.
- HAT-P-32b transmission metrics were exactly unchanged.
- WASP-69b likelihoods were exactly unchanged.
- SH4/PICASO accuracy changed only at floating-point summation level: the
  largest report-field change was `2.18e-16` absolute.

The only warnings were Matplotlib/PyParsing deprecations and an Intel OpenMP
`omp_set_nested` notice from installed dependencies; ROBERT does not call
those deprecated APIs.

## Remaining measured opportunities

Random overlap remains the largest native-grid cost even after the speedup.
The next useful investigation is coarse-grained parallelism over independent
spectral blocks or independent likelihood calls, with thread oversubscription
measured explicitly. SH4 is now dominated by many small banded linear solves;
batching or a lower-level LAPACK path may help, but needs its own numerical
parity study. Peak-memory instrumentation should also be added before enabling
larger molecule sets, because full-table log-k caching deliberately exchanges
memory for speed.

Transmission integration and general gas optical-depth assembly are not
current priorities: their measured contribution is small enough that more
complex code would not materially shorten retrievals.
