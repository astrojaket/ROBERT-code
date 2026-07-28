# MgSiO3 Cloud Forward-Model Performance Review

Date: 2026-07-28

## Outcome

The first physics-preserving cloudy-retrieval optimization sequence is
complete on `codex/cloudy-performance`. On the matched four-dataset WASP-69b
benchmark, the warmed MgSiO3/Mie/SH4 forward call fell from 0.779 s to
0.497 s: a 36.1% time reduction, or 1.57x higher throughput. The matched clear
call fell from 0.259 s to 0.193 s, and the cloudy/clear ratio fell from 3.01x
to 2.58x.

No pressure layers, wavelength points, correlated-k ordinates, angular
quadrature, Mie moments, delta-M order, or scattering physics were removed.
The accelerated spectrum agrees with the full SciPy/NumPy diagnostic reference
to `3.47e-18` maximum absolute eclipse depth and `1.53e-15` maximum relative
difference.

The remaining cloudy cost is no longer dominated by Mie optics. A current
profile places the time in the compiled SH4 boundary solver, RORR gas-opacity
combination, and final spectrum reconstruction. Further work should be chosen
from a new profile rather than by changing the scientific resolution.

## Matched benchmark

The benchmark uses the catalog MgSiO3 refractive index and the full-band
WASP-69b retrieval configuration:

- H2O, CO2, CO, CH4, NH3, and HCN gas opacity;
- 80 pressure layers and 16 correlated-k ordinates;
- 280 points across F322W2, the overlap average, F444W, and MIRI/LRS;
- 0.1 micron monodisperse particles with density 3200 kg m-3;
- condensate mass fraction `1e-7` from `1e-4` to `3.162` bar;
- exact scalar Mie phase moments through degree four;
- P3/SH4 thermal multiple scattering with degree-four delta-M; and
- one OpenMP thread and one Numba thread.

The clear counterpart reuses the exact planet, star, atmosphere builder,
prepared correlated-k provider, CIA tables, pressure grid, spectral grid, and
disc geometry from the cloudy problem. Each spectral mode retains its own
prepared opacity and radiative-transfer solve. The shared object is the
evaluated atmospheric temperature, chemistry, and mean-molecular-weight state;
instrument convolution is still applied after the appropriate per-mode
radiative-transfer calculation.

The measurements used an Apple M1 Pro macOS arm64 host with Python 3.12.13,
NumPy 2.2.6, SciPy 1.18.0, and Numba 0.61.2. Two warmups preceded seven
alternating measurements.

| Model | Before median (s) | After median (s) | After throughput |
| --- | ---: | ---: | ---: |
| Clear emission | 0.259 | 0.193 | 5.181 calls/s |
| MgSiO3/Mie/SH4 | 0.779 | 0.497 | 2.010 calls/s |
| Cloudy/clear ratio | 3.01x | 2.58x | — |

The cloud changes the spectrum by up to `5.86e-7` in eclipse depth, so the
benchmark cannot pass by silently taking the clear path. Problem setup takes
about 3.01 s and is excluded from timed calls. The reported peak resident
memory was 1559.6 MiB; this is a process-wide high-water mark that includes
loaded opacity tables and compiled-kernel state, not an incremental
per-likelihood allocation.

Reproduce the benchmark and profiles with:

```bash
env -u NUMBA_CACHE_DIR conda run -n robert-exoplanets \
  python examples/benchmark_wasp69b_mgsio3_cloud.py \
  --repeats 7 \
  --warmups 2 \
  --output /tmp/robert-cloud-speed.json \
  --profile-clear /tmp/robert-clear.prof \
  --profile-cloudy /tmp/robert-cloudy.prof
```

## Implemented sequence

### 1. Reference and accelerated SH4 backends

The SciPy/NumPy implementation remains the diagnostic scientific reference.
The boundary solver now accepts `auto`, `scipy`, or `numba`; `auto` selects
Numba when it is installed. The selected backend is recorded in the solver
result and forward-model manifest.

Parity coverage includes one, two, and 80 layers; absorbing, mixed, and exactly
conservative scattering; diagnostic arrays; supplied phase moments; and a
forced-pivot system. The compiled solver uses pivoted banded LU, validates
normalized backward error against `1e-11`, and reports the failing column for
singular systems.

### 2. Spectrum-only and compiled boundary solution

Retrieval calls avoid materializing diagnostic layer, angle, and correlated-k
products. Fixed half-range contractions and the independent banded systems are
compiled, while diagnostic requests continue to use the full reference-capable
path.

A warmed post-change profile of one cloudy call reported approximately:

| Operation | Cumulative time (s) |
| --- | ---: |
| Complete call | 0.504 |
| SH4 spectrum | 0.276 |
| SH4 layer/boundary solution | 0.180 |
| Compiled banded solver | 0.114 |
| Gas optical depth and RORR | 0.105 |
| Spectrum reconstruction | 0.081 |
| MgSiO3 Mie optics | 0.029 |
| Compiled half-range eigen step | 0.008 |

These cumulative entries overlap where a parent contains a child. They show
that compiling Mie first would not have materially improved total throughput.

### 3. Shared atmospheric state

The specialized refractive-index cloud model now has an
`evaluate_atmosphere` path. The four-dataset wrapper evaluates the common
temperature, chemistry, and mean-molecular-weight state once per parameter
vector, then supplies that state to each independently prepared spectral mode.

A benchmark guard checks both object sharing and a non-zero cloudy/clear
spectral difference. This guard caught an early generic-wrapper mismatch that
would otherwise have made a cloudy timing appear artificially fast.

### 4. Smooth cloud boundaries

Cloud slabs now use the fractional pressure overlap of every atmospheric layer
rather than a hard layer-centre mask. The pressure-column integral is
conserved, and moving a boundary through a layer changes the optical depth
continuously.

At a deliberately chosen layer-boundary crossing, a `2e-8` dex perturbation
previously produced a 308.6 ppm MIRI/LRS jump. The fractional treatment reduces
the same jump to approximately 0.00027 ppm. This removes a discontinuity that
could waste nested-sampling likelihood calls without introducing a smoothing
approximation.

### 5. MultiNest accounting and MPI scaling

The cloudy example now uses MultiNest. Its status metadata records both the
native MultiNest likelihood-evaluation count and Python callback count, since
startup/finalization callbacks can differ slightly from the native count.

The MPI benchmark constructs and warms the complete likelihood independently
on every rank, uses one OpenMP and one Numba thread per rank, and times
rank-independent deterministic prior points. Aggregate strong scaling was:

| MPI ranks | Aggregate calls/s | Efficiency relative to one rank |
| ---: | ---: | ---: |
| 1 | 1.536 | 100% |
| 2 | 2.969 | 96.6% |
| 4 | 5.801 | 94.4% |

The one-region direct refractive-index parameterization took about 0.742 s per
likelihood on one rank, only about 14% longer per call than the catalog model,
but it has 24 parameters rather than 12.

Two bounded four-rank MultiNest pilots tested integration and call accounting;
they were intentionally stopped after 100 iterations and are not posterior or
evidence convergence claims:

| Cloud model | Dimensions | Live points | Native evaluations | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Catalog MgSiO3 | 12 | 400 | 514 | 104.3 s |
| Direct n,k | 24 | 1000 | 1101 | 234.0 s |

The pilot shows why retrieval wall time cannot be inferred from forward-call
speed alone: dimensionality and live-point count also control the number of
calls MultiNest requests.

## Trust and regression gates

The performance result is accepted only when all of the following remain true:

- the full test suite passes in the `robert-exoplanets` Conda environment
  (498 passed and 5 optional tests skipped on 2026-07-28);
- the opt-in native MultiNest smoke passes (6 tests, including a completed
  156-evaluation native run);
- Ruff and whitespace checks pass;
- accelerated SH4 output stays within the reference parity gate;
- clear and cloudy grids remain exactly equal and all spectra remain finite;
- the cloudy spectrum remains measurably different from the clear spectrum;
- atmosphere sharing is asserted without sharing prepared per-mode opacity;
- fractional cloud-boundary conservation and continuity tests pass; and
- MPI throughput is reported from the slowest rank after setup and JIT warmup.

## Next measured targets

The 2x cloudy/clear target has not yet been reached. The next investigation
should separately measure allocations and time in the remaining compiled
boundary solve, RORR combination, and reconstruction. Any next optimization
must retain the SciPy/NumPy reference, the current numerical parity limits, and
the complete pressure, spectral, correlated-k, angular, Mie, and SH4 physics.
