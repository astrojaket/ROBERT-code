# Optional JAX Conservative-RORR Backend

Date: 2026-07-12

## Decision

ROBERT now contains an independent, opt-in JAX/XLA implementation of the same
conservative random-overlap with resorting and rebinning calculation used by
the NumPy scientific reference. It is not selected by any `auto` backend and
does not modify the existing NumPy or Numba kernels.

The earlier transmission-product emission idea was not adopted. Although
species transmissions factor exactly for a constant source inside each layer,
ROBERT's production solver integrates a source that varies linearly with
optical depth. Replacing it would therefore introduce a new approximation
rather than a pure acceleration backend.

## Numerical method

For every layer and wavelength, the JAX backend sequentially combines active
species by:

1. constructing every pairwise sum of the current and next k distributions;
2. attaching the outer-product quadrature weights;
3. sorting the pair distribution by optical depth;
4. integrating `k * dg` cumulatively; and
5. differencing that integral at the original target-g-bin boundaries.

This is conservative target-bin averaging, not interpolation of log-k at
g-points. It retains float64 throughout and preserves the existing transparent-
species cutoff behavior.

## Platform policy

Backend selection is explicit:

```python
from robert_exoplanets import jax_random_overlap_species_tau

mixed_cpu = jax_random_overlap_species_tau(tau, weights, platform="cpu")
mixed_gpu = jax_random_overlap_species_tau(tau, weights, platform="gpu")
```

ROBERT raises when the requested platform is unavailable. It never silently
falls back from GPU to CPU or from float64 to float32.

- Apple Silicon M1-M4: supported through the official JAX CPU backend.
- macOS Apple GPU: not supported for this backend because the experimental
  Metal plug-in does not support float64; upstream JAX currently directs macOS
  users to its CPU installation.
- Linux x86_64/aarch64 CPU: supported through standard JAX wheels.
- Linux NVIDIA GPU: supported through the optional CUDA 12 or CUDA 13 JAX
  extras when the cluster driver and GPU meet upstream requirements.

Install CPU JAX with:

```bash
python -m pip install -e ".[jax]"
```

For Linux CUDA, select the version compatible with the cluster:

```bash
python -m pip install -e ".[jax-cuda12]"
# or
python -m pip install -e ".[jax-cuda13]"
```

Every science-grade JAX run must set `JAX_ENABLE_X64=1` before Python starts.

## Apple-Silicon CPU benchmark

The controlled fixture uses the six WASP-69b F322W2 correlated-k tables,
80 layers, 144 wavelength bins, and 16 g ordinates. Times are warmed medians of
five calls; JAX compilation is separate.

| Backend | Median time |
|---|---:|
| NumPy reference | 904 ms |
| Numba, 8 threads | 15.9 ms |
| JAX/XLA CPU including host roundtrip | 481 ms |

JAX compilation plus the first call took 2.22 s. JAX agreed with NumPy to
`5.67e-13` RMS relative and `5.05e-12` maximum relative. Numba remained about
30 times faster than warmed JAX on this CPU fixture.

The conclusion is unambiguous for Apple laptops: use Numba for production CPU
retrievals. The JAX implementation exists for portability and controlled
Linux/CUDA evaluation, where device parallelism and a future device-resident
forward model may change the tradeoff.

## Reproduction

```bash
JAX_ENABLE_X64=1 NUMBA_NUM_THREADS=8 \
PYTHONPATH=src:examples \
python examples/benchmark_jax_random_overlap.py --repeats 5
```

The benchmark reports visible JAX devices, compilation time, warmed timings,
and numerical differences against both NumPy and Numba.
