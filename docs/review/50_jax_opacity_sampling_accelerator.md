# JAX opacity-sampling accelerator benchmark

Date: 2026-08-03

## Outcome

Opacity sampling is a substantially better accelerator workload than
correlated-k random overlap.  ROBERT now has a separate JAX path that retains
sampled ExoMol cross sections at their shared physical wavelengths,
interpolates them in log cross section, directly sums species optical depths,
and evaluates PG14 clear emission and the likelihood without a CPU numerical
callback.

On the Apple M4 Pro, Metal reached a small real speed advantage only at the
larger tested spectral width.  At 12,791 wavelengths it was 1.138x faster than
the existing CPU/Numba path.  At 5,117 wavelengths it was 10.0% slower.  The
same float32 JAX graph on CPU was 2.80x faster than CPU/Numba at 5,117
wavelengths, showing that the remaining limitation is the experimental Metal
backend rather than the JAX graph.

| Backend | Native wavelengths | Cold compile + first call | Warm median | CPU/Numba warm median | CPU/JAX speedup |
|---|---:|---:|---:|---:|---:|
| JAX Metal | 5,117 | 1.676 s | 20.857 ms | 18.960 ms | 0.909x |
| JAX CPU float32 | 5,117 | 717.1 ms | 6.935 ms | 19.402 ms | 2.798x |
| JAX Metal | 12,791 | 3.254 s | 42.073 ms | 47.894 ms | 1.138x |

Setup, HDF5 loading, checksum calculation, and host-to-device table transfer
are excluded from both warm timings.  Every JAX timing synchronizes the device.
Only one proposal and device zero are used.

## Scientific comparison

The benchmark uses the six published ExoMolOP R=15,000 TauREx cross-section
files, 80 atmospheric layers, a non-isothermal Parmentier-Guillot 2014 profile
evaluated independently at centers and edges, four-point Gauss-Legendre disk
integration, free trace-gas chemistry with an H2/He background, and a
128-bin deterministic observation projection.

| Backend | Native wavelengths | Spectrum RMS relative difference | Maximum absolute eclipse-depth difference | Absolute log-likelihood difference |
|---|---:|---:|---:|---:|
| JAX Metal | 5,117 | 1.262e-4 | 9.169e-9 | 5.438e-3 |
| JAX CPU float32 | 5,117 | 4.658e-7 | 6.439e-9 | 1.392e-4 |
| JAX Metal | 12,791 | 1.267e-4 | 7.069e-9 | 6.011e-3 |

Increasing each of H2O, CO, CO2, CH4, NH3, and HCN by 0.5 dex produces a
nonzero spectral change in both the float64 CPU reference and JAX result.  The
worst response-vector RMS error, normalized by the peak CPU response, is
`4.76e-6` at 5,117 wavelengths and `3.00e-6` at 12,791 wavelengths.  Repeated
JAX likelihoods are bitwise stable within each measured process.
The greater Metal discrepancy is therefore attributable to the Metal
execution of the float32 graph, not an opacity-insensitive atmosphere.  It is
small for this likelihood, but must be tested against retrieval posterior
stability before Metal can be called scientifically interchangeable with the
CPU reference.

The compact measured record, including every warm timing sample, is stored in
`docs/data/jax_opacity_sampling_m4pro_20260803.json`.

## Memory safety

The public preparation path estimates resident tables, interpolation
temporaries, disk RT scratch, compiler scratch, and the likelihood projection
before transferring the table.  Laptop runs default to a one GiB ceiling and
batch size one.  The 12,791-wavelength case retained 182,403,060 bytes of
visible device arrays and passed an 874,499,696-byte conservative accelerator
guard below the one-GiB limit.  Measured peak host RSS was 4,699,865,088 bytes;
the smaller Metal case peaked at 2,922,954,752 bytes.  Checks after each
isolated process recorded zero swap input/output and memory recovery after
exit.  This is not proof that a long-lived retrieval is leak-free, so such
retrievals must continue to monitor resident memory and clear caches between
unrelated static graph shapes.

The reviewed path uses a compact prefix-sum bin projection rather than a dense
mostly-zero matrix, guards raw table transfers, and assembles optical depth in
log space before exponentiation so weak cross sections are not prematurely
erased by float32 underflow.

## Reproduction

Apple Metal, conservative laptop run:

```bash
ENABLE_PJRT_COMPATIBILITY=1 \
conda run -n robert-exoplanets python \
  scripts/benchmark_opacity_sampling_jax.py \
  --data-dir opacity_data/exomol_xsec \
  --platform metal --sampling-stride 6 --layers 80 \
  --likelihood-bins 128 --warm-repeats 5 \
  --maximum-working-set-gib 1 \
  --output opacity_sampling_metal.json
```

Glamdring, one CUDA GPU and the full native opacity grid:

```bash
conda run -n robert-exoplanets python \
  scripts/benchmark_opacity_sampling_jax.py \
  --data-dir /path/to/exomol_xsec \
  --platform cuda --sampling-stride 1 --layers 80 \
  --likelihood-bins 128 --warm-repeats 5 \
  --maximum-working-set-gib 8 \
  --output opacity_sampling_cuda_gpu0.json
```

The CUDA command reports all visible GPUs but deliberately selects device
zero.  Multi-GPU proposal distribution is a later benchmark and is not
silently enabled by this script.

## Scope boundary

These are single-proposal forward-model and likelihood measurements, not a
posterior-equivalence result.  Cloud scattering remains outside this phase.
The sampler-facing problem supports the nested-sampler scalar likelihood
contract; optimal estimation requires additional differentiable-problem
interfaces and is not yet claimed for this backend.
