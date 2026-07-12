# WASP-69b Fast RT and Opacity-Sampling Validation

Date: 2026-07-12

## Outcome

ROBERT now preserves its full diagnostic radiative-transfer implementation as
the scientific reference while exposing a retrieval-oriented spectrum-only
path. The fast path omits layer and quadrature-point contribution arrays and
avoids species-resolved optical-depth arrays for fused opacity sampling.
Contribution functions remain available through the diagnostic solver for
plotting and interpretation.

The opacity-sampling path now uses a compiled fused kernel for pressure and
temperature interpolation, exponentiation, abundance weighting, and species
summation. A deterministic stratified response selects native ExoMol samples
inside each observed spectral bin. Both features are opt-in; correlated-k
remains the retrieval default.

## Numerical checks

- The spectrum-only thermal solver agrees with the summed diagnostic solver to
  a maximum relative difference of `7.77e-16` in the performance fixture.
- Fused opacity-sampling optical depths agree with the species-resolved
  reference calculation to within `2e-13` relative tolerance.
- The spectrum-only RT integration reduced its benchmark time from 59.8 ms to
  17.4 ms, a 3.42x speedup, without changing the spectrum.

## WASP-69b native-bin benchmark

The benchmark uses the 274 Schlawin et al. (2024) native bins from the F322W2,
F444W, and LRS modes, 80 atmospheric layers, and six gases. Diagnostics are
disabled for every timed retrieval call. The accuracy reference evaluates all
23,907 ExoMol R=15,000 samples within the observed wavelength range before
integrating into the exact observation bins.

| Method | Opacity samples | Median complete call | Speed vs correlated-k | RMS error vs full ExoMol | RMS difference |
|---|---:|---:|---:|---:|---:|
| Correlated-k, 16 g ordinates | — | 168 ms | 1.00x | 2.92% | 0.80 sigma |
| Sampling, 2 per bin | 548 | 103 ms | 1.62x | 11.73% | 2.90 sigma |
| Sampling, 4 per bin | 1,096 | 113 ms | 1.49x | 7.83% | 1.93 sigma |
| Sampling, 8 per bin | 2,192 | 136 ms | 1.23x | 4.88% | 1.33 sigma |
| Sampling, 16 per bin | 4,384 | 171 ms | 0.98x | 3.48% | 0.83 sigma |
| Sampling, 24 per bin | 6,576 | 202 ms | 0.83x | 2.37% | 0.60 sigma |
| Full ExoMol sampling | 23,907 | 404 ms | 0.42x | reference | reference |

The correlated-k comparison includes a line-list difference: its CO data use
HITEMP while the ExoMol sampling data use Li2015. It therefore measures the
combined representation and opacity-source difference rather than only the
correlated-k approximation.

## Decision

Sparse midpoint sampling is not scientifically adequate for the WASP-69b bins.
It produces a modest speedup only while its error exceeds the observational
uncertainty. At 16–24 samples per bin it approaches the full-grid result, but
the speed advantage has disappeared. This is an algorithmic sampling limit;
moving the same calculation to Rust or C++ would not improve its convergence.

The production strategy is therefore:

1. Retain the typed NumPy diagnostic path as the scientific reference.
2. Use the compiled spectrum-only path during retrievals.
3. Keep correlated-k as the validated WASP-69b opacity method.
4. Compute contribution functions only for requested diagnostic products.
5. Profile chemistry and retrieval-level parallelism before introducing a
   native-language extension. A native extension is justified only for a
   remaining measured kernel, with parity tests against the reference path.

## Reproduction

```bash
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1 \
NUMBA_CACHE_DIR=/tmp/robert-numba-cache PYTHONPATH=src \
python examples/benchmark_wasp69b_opacity_sampling.py
```

The JSON result is written to
`examples/outputs/opacity_sampling/wasp69b_stratified_sampling_benchmark.json`.
