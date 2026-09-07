# Isolated JAX Metal backend: implementation and scientific audit

Date: 2026-08-03

## Outcome

ROBERT now has an isolated experimental JAX Metal backend under
`src/robert_exoplanets/metal/`.  It does not branch inside, replace, or alter
the NumPy/Numba CPU numerical implementation.  A strict runtime refuses CPU
fallback, and the sampler-facing `MetalRetrievalProblem` synchronizes only the
final likelihood scalar.

The implementation is scientifically credible as an experimental reference,
but it is **not recommended as ROBERT's default performance backend**.  On the
tested Apple M4 Pro, exact conservative RORR and pivoted SH4 are much slower
than ROBERT's optimized Numba CPU kernels.  Exact Mie is faster on Metal for a
128-particle batch, but Mie is not the dominant complete-retrieval cost.  A
complete clear likelihood is about 165 times slower than the production CPU
path at the tested shape.

This negative performance result is important: changing Python numerical code
to JAX does not reproduce BeAR's mature CUDA kernels, and Apple Metal's sort
and control-flow performance is presently a poor match for ROBERT's exact
RORR and pivoted band solve.

## Verified runtime

- Apple M4 Pro, 20 GPU cores, 24 GB unified memory;
- `jax==0.4.38`;
- `jaxlib==0.4.38`;
- `jax-metal==0.1.1`;
- Metal float32 only; and
- `ENABLE_PJRT_COMPATIBILITY=1` at process launch.

The versions are pinned in the `metal` optional dependency because newer JAX
and jaxlib packages were not compatible with the verified plug-in.  The
backend records package versions, device kind, platform, and dtype in retrieval
metadata.

## Implemented graph

The device-native components are:

1. Parmentier--Guillot 2014 temperature, including a float32-converged
   exponential-integral calculation;
2. common-grid pressure/temperature interpolation in log(k), with static
   spectral indices and no interpolation across correlated-k wavelength bins;
3. hydrostatic molecular columns and gas optical depth;
4. ROBERT's conservative target-bin RORR definition;
5. exact real-pair Lorenz--Mie recurrence, scalar phase moments through degree
   four, lognormal integration, and hydrostatic cloud optical depth;
6. exact linear-source clear thermal integration;
7. P3/SH4 with degree-four delta-M, analytic layer modes, a compact
   bandwidth-five partial-pivot LU, and normalized backward-error rejection;
8. fixed linear instrument projections;
9. normalized or unnormalized Gaussian likelihoods with masks, offset, jitter,
   uncertainty scaling, and ordered multi-dataset sums; and
10. one outer JIT from parameter vector through the final scalar likelihood.

The Mie cloud hardware test compiles a retrieved particle radius and cloud mass
fraction through Mie optics, cloud extinction/scattering, phase moments,
delta-M, SH4, projection, and likelihood.  The clear hardware test compiles
correlated-k interpolation, hydrostatic optical depth, RORR, thermal RT,
projection, and likelihood.  All outputs are asserted to reside on the Metal
device.

## Scientific parity

### Component tests

| Component | Comparison | Result |
| --- | --- | ---: |
| Conservative RORR, 3 x 40 x 64 x 8 | Metal float32 vs NumPy target-bin reference | RMS relative `2.52e-7`; max relative `2.06e-6`; max absolute `2.47e-6` |
| PG14 temperature, 80 layers | JAX float32 vs SciPy/NumPy float64 | `rtol=3e-5`, `atol=0.03 K` test passes |
| log(k) interpolation + hydrostatic tau | JAX float32 vs float64 equation | `rtol<=1.5e-5` tests pass |
| clear linear-source RT | JAX float32 vs NumPy reference | `rtol=2e-6`, `atol=2e-6` test passes |
| Mie x={1e-4, 1, 5, 20} | Metal algorithm on JAX CPU vs CPU Mie | Q/g `rtol<=3e-5`; moments `rtol<=5e-5` |
| Mie x=50, m=4+3i | bounded downward recurrence vs CPU Mie | Q/g `rtol<=8e-5`; moments `rtol<=1e-4` |
| Mie resonance sweep | x=0.1..30, nonabsorbing | Q `rtol<=6e-5`; g `atol<=8e-5`; cancellation-limited moments `atol<=4e-4` |
| lognormal Mie mass optics | JAX float32 vs CPU Mie | mass coefficients `rtol<=8e-5`; moments `rtol<=1.5e-4` |
| SH4 mixed/absorption/conservative/delta-M | JAX CPU x64 vs CPU reference | mixed max absolute `4.44e-16`; all parity tests pass |
| SH4, 80 x 4 x 4 | Metal float32 vs Numba CPU | RMS relative `2.17e-7`; max relative `2.73e-7` |

The broad 128-particle Mie timing sweep showed maximum absolute efficiency
error `2.05e-4` and maximum absolute phase-moment error `1.30e-3`.  Relative
errors become uninformative for symmetry-forced moments near zero.  These
broader errors are recorded rather than hidden; posterior acceptance requires
the actual configured cloud parameter bounds and observational spectrum, not
only the smaller unit fixtures.

### Complete-likelihood comparison

The self-contained complete clear benchmark used three species, 80 layers,
128 wavelengths, and eight g ordinates.  At a temperature perturbation of 10 K:

- CPU log likelihood: `-186.7363303`;
- Metal log likelihood: `-187.0821838`; and
- absolute difference: `0.3458535`.

This is inside the provisional single-point maximum of 0.5 proposed during
review. A resolved, sequential 21-point temperature-shift fixture over ±2 K
(no batching) gave:

- median `|delta logL| = 0.03649`;
- p99 `|delta logL| = 0.06961`;
- maximum `|delta logL| = 0.06963`;
- posterior mean difference `0.00937 K`, or `0.018` CPU posterior standard
  deviations;
- posterior standard deviations `0.519397 K` (CPU) and `0.519409 K` (Metal),
  a relative width difference of about `0.0024%`; and
- log-evidence difference `0.000223` on the identical discrete grid.

The posterior mean, width, p99/max likelihood, and evidence gates proposed in
review pass on this self-contained fixture. The stricter provisional median
`|delta logL| <= 0.01` gate does not pass. This is not a WASP-69 posterior or
sampler-convergence claim.

## Synchronized performance

Compilation and execution are separated; every Metal timing calls
`block_until_ready()`.

| Workload | Compile | Warm Metal | CPU reference | CPU/Metal speedup |
| --- | ---: | ---: | ---: | ---: |
| RORR 3 x 40 x 64 x 8 | 0.272 s | 6.63 ms | 1.31 ms Numba | 0.198x |
| exact Mie, 128 particles | 0.470 s | 16.06 ms | 29.16 ms CPU | 1.82x |
| SH4 80 x 4 x 4 | 9.12 s | 1.384 s | 2.07 ms Numba | 0.00150x |
| complete clear likelihood 3 x 80 x 128 x 8 | 1.33 s | 1.100 s | 6.69 ms production CPU | 0.00608x |

An 80 x 32 x 8 SH4 benchmark was stopped after exceeding three minutes.  It is
not presented as a timing result.  The earlier Python-unrolled graph was then
replaced by fixed-size `lax` loops, but the bounded 80 x 4 x 4 result still
shows that Metal control-flow performance is not competitive with Numba.

## Unified-memory incident and policy

A production-shape batch-32 clear graph caused a Metal bus error and the user
reported a laptop crash.  A batch-8 follow-up was stopped and is not reported.
The failure was a single oversized unified-memory allocation, not disk growth:
after restart macOS reported 85% available memory pressure, zero swap I/O, and
331 GiB free disk.

The backend now includes `MetalResourcePolicy`:

- conservative estimates include pair-distribution sort scratch and compiler
  intermediates, not just visible arrays;
- the default limit is the smaller of 1 GiB or 8% of physical RAM;
- unsafe graph shapes raise before compilation;
- laptop-safe public batching is hard-limited to one proposal, independent of
  a lower byte estimate, because outer `vmap` compiler scratch is not visible
  to primitive shape guards;
- the benchmark records batch 8 and 32 as refused before `jax.jit` rather than
  launching them;
- ROBERT's SH4 kernel cache is limited to four quadrature variants;
- its Mie kernel cache is limited to eight static-capacity variants;
- other compiled caches use bounded LRU storage; and
- `clear_metal_caches()` releases ROBERT-held compiled kernels between
  unrelated retrievals in a long-lived notebook.

`XLA_PYTHON_CLIENT_MEM_FRACTION` did not lower the Metal plug-in's logged
17.18 GB allocator ceiling and is therefore not presented as protection.  The
preflight policy and isolated benchmark processes are the enforced safeguards.
No benchmark in this report should be rerun with a higher limit on a 24 GB
laptop while interactive applications are open.

## BeAR and Exo Skryer context

The source review in
`docs/review/31_bear_exoskryer_correlated_k_review.md` explains the performance
gap.  BeAR uses persistent single-precision CUDA kernels over opacity sampling,
with one thread per spectral point and GPU RT/instrument/likelihood kernels.  It
does not solve ROBERT's conservative correlated-k random-overlap problem.

Exo Skryer uses a whole JAX graph, but its RORR sorts and interpolates a
different distribution and also offers approximate PRAS and transmission-only
factorization.  Those algorithms are not drop-in replacements for ROBERT's
conservative target-bin calculation.  ROBERT retained its scientific
definition, and the M4 timings show the cost of that decision on JAX Metal.

## Supported and unsupported use

The Metal backend accepts prepared arrays and supports fixed/free composition
inside the compiled graph.  It does not call CPU code from a JIT.

Correlated-k coverage is explicit: strict prepared tables return an invalid
device state outside pressure/temperature coverage, while clip-mode tables
clip both coordinates to the boundary before calculating interpolation
weights. Dynamic Mie capacity, particle radius, distribution width, density,
refractive index, gravity, pressure thickness, condensate mass fraction, and
opacity validity are checked on device; invalid values propagate to the
configured invalid likelihood rather than returning finite truncated physics.

FastChem remains an external CPU library and cannot participate in a JAX Metal
graph.  The configured WASP-69b equilibrium-chemistry retrieval is therefore
explicitly unsupported rather than using `pure_callback` or silently moving
chemistry to the CPU.  A genuine GPU equilibrium-chemistry implementation is a
separate scientific project and is not approximated here.

The local full WASP-69 direct-n/k validation is also blocked by the existing
empty/corrupt local STScI PHOENIX catalogue.  Self-contained scientific
fixtures cover all implemented kernels, but this report makes no WASP-69
posterior-reproducibility claim.

## Installation and execution

```bash
conda run -n robert-exoplanets python -m pip install -e ".[dev,opacity,retrieval,metal]"

ENABLE_PJRT_COMPATIBILITY=1 ROBERT_RUN_METAL_TESTS=1 \
  conda run -n robert-exoplanets python -m pytest tests/test_metal_hardware.py -q

ENABLE_PJRT_COMPATIBILITY=1 \
  conda run -n robert-exoplanets python scripts/benchmark_metal_backend.py
```

Algorithmic float64 SH4 parity must run in a separate process from float32
JAX tests:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 ROBERT_RUN_JAX_CPU_TESTS=1 \
  conda run -n robert-exoplanets python -m pytest tests/test_metal_sh4.py -q
```

## Verification summary

- complete CPU suite: `527 passed, 20 skipped`;
- optional JAX CPU float32/component suite: `25 passed`;
- separate JAX CPU float64 SH4 suite: `5 passed`;
- real Apple Metal suite: `3 passed`; and
- Ruff checks: pass for all new backend, test, and benchmark files.

## Decision

Keep this backend isolated and experimental.  It is a real GPU implementation,
not a CPU callback, but the measured M4 Pro performance does not justify a
production user selector yet.  Near-term performance work should continue on
the exact Numba CPU path or investigate a native Metal/C++ kernel specifically
for the measured RORR/SH4 bottlenecks.  JAX Metal should be reconsidered when
its sort and control-flow compiler/runtime improve, using these tests and
benchmarks as the acceptance contract.
