# MgSiO3 Cloud Forward-Model Performance Plan

Date: 2026-07-13

## Outcome

The warmed six-gas WASP-69b MgSiO3/Mie/SH4 forward model takes **1.483 s**
per four-dataset call, compared with **0.471 s** for the matched clear-emission
model. The cloudy model is **3.15x slower** and adds **1.013 s** per call.

The calculation is not Mie-limited. A warmed profile attributes 0.989 s of a
1.460 s cloudy call to SH4, including 0.608 s in the multilayer boundary solve
and 0.347 s in source reconstruction. Mie particle optics takes only 0.041 s.
The next optimization should therefore be an optional Numba SH4 execution
backend, following the reference-plus-accelerated pattern already used for
random-overlap resort/rebin (RORR).

No angular, vertical, spectral, correlated-k, Mie, phase-function, or delta-M
resolution should be reduced to obtain the speedup.

## Matched benchmark

The benchmark uses the catalog `MgSiO3` refractive index and the existing
WASP-69b full-band retrieval configuration:

- gases: H2O, CO2, CO, CH4, NH3, and HCN;
- 80 pressure layers;
- 16 correlated-k ordinates;
- 280 points across F322W2, the published overlap average, F444W, and MIRI/LRS;
- 0.1 micron monodisperse particles with density 3200 kg m-3;
- condensate mass fraction `1e-7` between `1e-4` and `3.162` bar;
- exact scalar Mie phase moments through degree four;
- P3/SH4 thermal multiple scattering with degree-four delta-M;
- one OpenMP thread and one Numba thread.

The clear counterpart is constructed from each cloudy model's exact planet,
star, atmosphere builder, prepared correlated-k provider, CIA tables,
Rayleigh setting, pressure grid, spectral grid, and disk geometry. The timed
difference is therefore cloud optics plus the multiple-scattering path, not a
different atmospheric or opacity setup.

The run used the `robert-exoplanets` conda environment on an Apple M1 Pro
MacBook Pro with macOS arm64, Python 3.12.13, NumPy 2.2.6, SciPy 1.18.0, and
Numba 0.61.2. Two warmups preceded five alternating measurements.

| Model | Timed calls (s) | Median (s) | Throughput (calls/s) |
| --- | --- | ---: | ---: |
| Clear emission | 0.462, 0.468, 0.521, 0.471, 0.474 | 0.471 | 2.123 |
| MgSiO3/Mie/SH4 | 1.501, 1.451, 1.498, 1.466, 1.483 | 1.483 | 0.674 |

Cloudy problem construction took 0.828 s. All outputs were finite, clear and
cloudy spectral grids were exactly equal, and the maximum absolute change in
eclipse depth was `5.231e-7`, confirming that the cloud case did not collapse
onto the clear solution.

Reproduce the measurements and warmed profiles with:

```bash
conda run -p /Users/jaketaylor/miniforge3/envs/robert-exoplanets \
  python examples/benchmark_wasp69b_mgsio3_cloud.py \
  --repeats 5 \
  --warmups 2 \
  --output /tmp/robert-cloud-speed.json \
  --profile-clear /tmp/robert-clear.prof \
  --profile-cloudy /tmp/robert-cloudy.prof
```

## Profile decomposition

The profile covers one warmed full cloudy call. Percentages use the 1.460 s
profiled wall time; cumulative entries overlap where a parent contains a
child.

| Operation | Cumulative time (s) | Profiled call |
| --- | ---: | ---: |
| SH4 total | 0.989 | 67.7% |
| SH4 multilayer boundary solution | 0.608 | 41.6% |
| 4,480 SciPy `solve_banded` calls | 0.305 | 20.9% |
| SH4 source reconstruction | 0.347 | 23.8% |
| Small-array `einsum` calls | 0.256 | 17.5% |
| Gas optical depth, including RORR | 0.207 | 14.2% |
| CIA optical depth | 0.153 | 10.5% |
| MgSiO3 Mie optics | 0.041 | 2.8% |

There is one 320-unknown boundary system (`4 * n_layers`) for each of the
`280 * 16 = 4,480` wavelength/g columns. Dispatching thousands of small SciPy
solves is expensive, while the columns are independent and naturally suited
to a compiled parallel loop. The diagnostic SH4 implementation also
materializes level moments and angle/layer/g contribution arrays even though
a retrieval call ultimately needs only the disk-integrated spectrum.

## Implemented spectrum-only result

The first acceleration stage now keeps the SciPy boundary solve unchanged and
uses a Numba kernel to reconstruct only the disk- and g-integrated spectrum.
The full NumPy/SciPy result remains available whenever diagnostics are enabled.
The retrieval path also uses the existing fused diagnostics-free RORR kernel.

On the same single-threaded laptop benchmark, two warmups and five alternating
measurements gave:

| Model | Before (s) | Spectrum-only (s) | Change |
| --- | ---: | ---: | ---: |
| Clear emission | 0.471 | 0.445 | normal run-to-run variation plus fused path |
| MgSiO3/Mie/SH4 | 1.483 | 1.191 | 19.7% less time, 1.25x faster |
| Cloudy/clear ratio | 3.15x | 2.68x | 14.9% smaller relative penalty |

The Numba reconstruction itself takes 0.120 s instead of 0.347 s, a 2.89x
kernel speedup. Total SH4 time falls from 0.989 s to 0.749 s. The unchanged
multilayer solution now dominates at 0.609 s, including 0.305 s in 4,480
SciPy `solve_banded` calls and 0.217 s in the remaining large `einsum`
contractions.

With eight Numba threads on the same M1 Pro, the clear and cloudy medians were
0.323 s and 0.972 s respectively. The cloud calculation therefore reaches
about one call per second on this laptop. The smaller gain over the new
single-thread result is expected because the SciPy boundary solves remain
serial.

The real four-dataset WASP-69b spectrum-only output agrees with the full
diagnostic cloud solver to `2.17e-18` maximum absolute eclipse depth and
`1.11e-15` maximum relative difference. Unit coverage also compares NumPy and
Numba for supplied phase moments, delta-M, mixed scattering, and the exactly
conservative limit.

The WASP-69b and WASP-80b Mie retrievals both use SH4,
`thermal_integration_backend="auto"`, and diagnostics disabled. Their Slurm
scripts request four one-CPU MPI tasks with `NUMBA_NUM_THREADS=1`, so every
likelihood worker selects this Numba spectrum-only path without nested thread
oversubscription. On larger CPU nodes, prefer more independent sampler/MPI
workers before increasing Numba threads inside each likelihood call, and
benchmark the product of both settings explicitly.

GPU work is not the next priority: the remaining dominant operation is the
SciPy banded boundary solve, while the accelerated reconstruction is only
about 10% of the cloudy call. A physics-equivalent batched CPU solve should be
implemented and profiled first. GPU/JAX work becomes worthwhile if large
native spectral grids or higher-order solvers leave a sufficiently large
batched kernel after that change.

Verification completed in the `robert-exoplanets` conda environment:

- all 303 tests pass;
- Ruff and formatting checks pass;
- WASP-69b catalog and direct-n,k smoke likelihoods are finite;
- WASP-80b catalog and direct-n,k smoke likelihoods are finite;
- every prepared WASP-69b and WASP-80b mode records
  `cloud_spectrum_only=true` and `cloud_sh4_spectrum_backend=numba`.

## Remaining physics-preserving implementation plan

### 1. Establish the backend and parity contract

Keep the current SciPy/NumPy implementation as the named scientific reference.
Add `backend="auto" | "scipy" | "numba"` to `solve_thermal_sh4`, with `auto`
selecting Numba when installed and falling back to SciPy otherwise. Thread an
explicit SH4 execution-backend option through the cloud configurations and
record the selected backend in result and manifest metadata.

Before changing the solver, extend `tests/test_rt_sh4.py` with reference arrays
covering:

- one, two, and 80 layers;
- optical depths from the optically thin limit through very thick layers;
- absorbing, mixed, and exactly conservative scattering;
- isotropic, backward, and strongly forward scattering;
- Henyey-Greenstein and supplied exact Mie moments;
- delta-M enabled and disabled;
- multiple emission angles and correlated-k ordinates;
- layer/angle contributions, bottom contribution, level moments, and final
  radiance.

Use the analytic tests at their existing tight tolerances. Gate accelerated
versus reference arrays at `rtol=2e-11, atol=2e-13` initially, require the
normalized boundary-system residual to remain below `1e-11`, and tighten the
array tolerance if the conditioning study supports it. Any failures should
fall back by explicit backend selection, not by silently changing the physics.

### 2. Add a spectrum-only SH4 retrieval path (completed)

Mirror the existing clear-emission optimization: retain the full diagnostic
solver, but add a path that returns only g-integrated, disk-integrated radiance.
It should not allocate `moment_levels`, per-layer contribution arrays, or
per-g diagnostic products when `compute_diagnostics=False`.

Compile the reconstruction loop with Numba and accumulate directly over
quadrature node, emission angle, layer, and g ordinate. Compute the fixed
Gauss-Legendre nodes and disk geometry outside the kernel. This changes only
which intermediate arrays are materialized; it must use the same source
function, quadrature, attenuation, boundary condition, and summation weights.

Acceptance criteria:

- spectrum agrees with the summed diagnostic reference within the parity gate;
- no diagnostic array is allocated in the retrieval path;
- at least 2x speedup for SH4 source reconstruction on the benchmark fixture;
- contribution-function requests continue to use the reference-capable full
  result path.

### 3. Compile the batched banded boundary solve (next)

Implement a Numba kernel over independent wavelength/g columns. Preserve the
same 11-row banded matrix and right-hand side constructed by the reference
equations. Use a bandwidth-aware Gaussian elimination with pivoting and an
explicit singular/ill-conditioned status per column; do not replace the solve
with an approximate iteration or lower-order scattering method.

Develop this in two separable steps:

1. compile only the existing banded matrix solve and compare its coefficients
   and residuals against SciPy `solve_banded`;
2. after parity is established, fuse coefficient/eigenbasis construction and
   band assembly into the column kernel to remove large temporaries and small
   `einsum` dispatches.

Acceptance criteria:

- coefficient and emergent-radiance parity across the validation matrix;
- singular cases raise the same typed ROBERT validation error with the failing
  column identified;
- at least 3x speedup for `_layer_solution` on the 80-layer fixture;
- deterministic serial results with `NUMBA_NUM_THREADS=1`.

### 4. Share atmospheric state across spectral modes

The current cloudy wrapper rebuilds an identical temperature/chemistry state
for each of four datasets. Add the same `evaluate_atmosphere`/typed
multi-dataset sharing pattern already validated for clear emission. Keep each
mode's opacity recompression, Mie evaluation, cloud optical depth, and SH4 solve
independent. Verify bit-for-bit identity against the current independent-mode
wrapper before adopting it in the retrieval workflow.

This is a smaller, orthogonal saving; it should follow the kernel work or be a
separate reviewable change rather than being mixed into the SH4 numerical
rewrite.

### 5. Re-profile before touching Mie or common opacity work

Mie is only 2.8% of the current cloudy call, so compiling it first cannot
materially improve retrieval throughput. After SH4 acceleration, re-run this
benchmark with one and all available Numba threads and measure peak resident
memory. Only then consider:

- preparing immutable optical-constant interpolation outside repeated Mie
  calls;
- compiling Mie coefficient/moment loops if they become a measured hotspot;
- preparing CIA interpolation indices, which affects clear and cloudy models;
- disabling species-resolved gas optical-depth storage in the cloudy
  diagnostics-free path so it uses the already validated fused RORR kernel.

Do not cache Mie outputs across changing radius, width, refractive index, or
cloud-composition parameters unless every physics-bearing input is part of an
explicit cache key.

## Delivery sequence and performance gate

Use small, independently reviewable changes:

1. parity fixtures, backend selection, metadata, and benchmark extensions;
2. Numba spectrum-only reconstruction;
3. Numba batched banded solve;
4. fused SH4 column kernel after both isolated kernels pass;
5. shared cloudy atmospheric state;
6. final full-suite, benchmark, and peak-memory report.

The initial end-to-end target is a warmed cloudy call below **0.94 s** (no more
than 2x the matched clear call) with no resolution or physics changes. A 3x
speedup of the measured SH4 section implies about 0.80 s end to end by Amdahl's
law, or roughly 1.7x the clear baseline. Treat that as the realistic first
target; further work must be justified by a new profile rather than an assumed
benefit.

## Non-goals

- No reduction in pressure layers, correlated-k ordinates, source quadrature,
  disk quadrature, or wavelength coverage.
- No replacement of exact Mie moments by Henyey-Greenstein moments.
- No change from SH4 to two-stream or pure absorption.
- No change to delta-M scaling or cloud vertical structure.
- No float32 path.
- No removal of the SciPy/NumPy scientific reference implementation.
