# Performance Review

Sources inspected:

- NEMESIS / Radtran: https://github.com/nemesiscode/radtrancode at commit `52c273f`.
- NemesisPy: https://github.com/astrojaket/NemesisPy at commit `afe2bae`.

This review identifies likely runtime hot spots and design opportunities. It does not include measured profiling runs, because the task was source review only.

## Where Runtime Is Spent

### Radiative Transfer Kernel

The dominant cost is expected in:

- gas k-table interpolation across wavelength, g-ordinate, gas, pressure, temperature, and layer,
- random-overlap k-distribution combination,
- CIA and continuum optical-depth evaluation,
- layer-by-layer transmission/source-function accumulation,
- disc/phase integration over angles or surface tiles,
- repeated forward-model calls inside retrieval loops.

NEMESIS concentrates this in `cirsradg_wave.f`, `get_kg.f`, `noverlapg.f`, `ngascon.f`, `nciacon.f`, and scattering routines.

NemesisPy concentrates this in `calc_tau_gas.py`, `calc_tau_cia.py`, `calc_radiance.py`, `calc_layer.py`, and backend `ForwardModel` methods.

### Retrieval Loop

NEMESIS optimal estimation repeatedly computes:

- forward spectrum,
- K matrix / Jacobians,
- gain matrix,
- cost function,
- covariance products,
- trial spectra for Marquardt brake acceptance.

Finite-difference Jacobians are especially expensive because they require repeated forward models.

Bayesian/nested sampling workflows will spend almost all runtime in forward-model evaluations, with likelihood overhead secondary unless I/O or Python object churn is excessive.

### I/O and Opacity Preparation

NEMESIS reads many sidecar files and opacity tables. NemesisPy modular k-table preparation can use `exo_k`, MPI, plotting, and disk cache writes.

For retrieval, opacity loading/rebinning must be outside the inner loop. Per-sample file I/O is unacceptable.

## Vectorization Opportunities

### Safe Near-Term Targets

- Precompute layer-independent wavelength/g-ordinate constants.
- Vectorize Planck function calls over wavelength and layer.
- Vectorize CIA pair interpolation where it does not change numerical policy.
- Vectorize observation binning/convolution outside the RT kernel.
- Batch multi-observation likelihood calculations when they share the same high-resolution spectrum.

### Higher-Risk Targets

- Vectorizing random-overlap ranking can change ordering/tie behavior and floating-point accumulation.
- Reordering loops in correlated-k kernels can alter precision and cache behavior.
- Broadcasting large `(wave, g, layer, gas)` arrays can improve clarity but explode memory.

Recommendation:

- Preserve a scalar/reference implementation for tests.
- Optimize only behind the same public interface.

## Caching Opportunities

### High-Value Caches

- Opacity tables loaded from disk.
- Regridded/rebinned k-tables per instrument wavelength grid.
- K-table interpolation indices and weights for a fixed pressure/temperature grid when appropriate.
- Layer geometry for fixed pressure grid, radius, gravity, and viewing geometry.
- Stellar spectra interpolated to model/observation grids.
- Instrument response matrices.

### Cache Invalidation Risks

- Temperature-dependent k interpolation invalidates with T(P).
- Hydrostatic height invalidates with temperature, mean molecular weight, gravity, radius, and reference pressure.
- VMR-dependent mean molecular weight invalidates with chemistry.
- Clouds may invalidate wavelength-dependent opacity and geometry-dependent transmission.

ROBERT recommendation:

- Use explicit cache keys.
- Keep caches outside pure scientific functions.
- Provide a debug mode that disables caches for reproducibility testing.

## Parallelism and MPI

### Existing Patterns

NEMESIS historically uses separate executables and external workflows. NemesisPy uses MPI in some k-table/TLSE/data-fitting scripts and has import-time MPI fallback in `modular/io/ktables.py`.

### Opportunities

- Parallelize independent likelihood evaluations across sampler workers.
- Parallelize phase-curve/disc tiles.
- Parallelize wavelength chunks for high-resolution forward models.
- Parallelize k-table rebinning/preparation across gases/instruments.

### Design Recommendation

ROBERT should not make MPI a hidden import-time behavior. Instead:

```text
serial core kernel
  -> multiprocessing/threading adapter
  -> MPI adapter
  -> cluster/scheduler adapter
```

Sampler adapters should own distributed execution where possible. The computational core should be deterministic in serial.

## Asynchronous I/O

Useful for:

- prefetching opacity assets,
- writing sampler checkpoints,
- streaming diagnostics,
- preparing multiple instrument grids before retrieval.

Not useful inside:

- numerical radiative-transfer kernels,
- per-sample likelihoods unless file I/O remains in the hot path, which should be avoided.

ROBERT recommendation:

- Use asynchronous I/O for workflow orchestration and data preparation only.
- Keep scientific kernels synchronous and testable.

## Numba

NemesisPy demonstrates that Numba can make explicit Python loops competitive while preserving a readable port of Fortran logic.

Strengths:

- Good for scalar loop kernels.
- Low barrier for porting Fortran-like algorithms.
- Avoids immediate C/Fortran extension complexity.

Risks:

- Harder debugging in `nopython` mode.
- Function signatures and array dtypes become implicit API.
- Some Python/NumPy behavior differs under JIT.
- Compilation overhead matters for short runs.

ROBERT recommendation:

- Keep Numba as a backend option, not the only implementation.
- Maintain pure NumPy/reference functions for small tests where feasible.
- Pin numerical tolerances and compile-time behavior in benchmarks.

## JAX

Potential advantages:

- Automatic differentiation for Jacobians.
- XLA compilation and GPU/TPU paths.
- Batch evaluation and vectorized sampler proposals.

Risks:

- Correlated-k ranking/random overlap may be awkward or nondifferentiable.
- Dynamic file/data structures must be pushed outside JAX functions.
- Scientific reproducibility can be harder across devices/dtypes.
- Rewriting kernels for JAX before validation could delay correctness.

ROBERT recommendation:

- Do not start with JAX as the primary backend.
- Design array APIs that could support a JAX backend later.
- Consider JAX first for differentiable parameterizations and simple RT subsets, not for the full legacy physics stack.

## GPU Acceleration

Potential targets:

- batched spectra over many parameter samples,
- large wavelength grids,
- phase-curve tile batches,
- interpolation-heavy opacity operations.

Risks:

- Data transfer overhead can dominate small spectra.
- Memory footprint of full opacity arrays can exceed GPU memory.
- Random-overlap sorting/ranking may not map cleanly.
- GPU floating-point behavior may complicate reference comparisons.

ROBERT recommendation:

- Defer GPU work until CPU reference kernels and benchmark suite are stable.
- Preserve backend boundaries so GPU kernels can be added later.

## Memory Layout

NEMESIS and NemesisPy inherit different array-order assumptions:

- Fortran arrays are column-major.
- NumPy arrays are row-major by default.
- NemesisPy comments explicitly note Fortran/Python multi-dimensional cycling differences.

High-risk arrays:

- `k_gas_w_g_p_t` shaped roughly `(gas, wave, g, pressure, temperature)`.
- `tau_total_w_g_l` shaped `(wave, g, layer)`.
- VMR shaped `(layer, gas)`.

ROBERT recommendation:

- Standardize dimension order in the data model.
- Document dimension names with every array.
- Consider `xarray`-like metadata for diagnostics, but keep kernels using plain arrays.
- Benchmark C-order vs Fortran-order arrays for each backend.

## Benchmarking Philosophy

ROBERT should maintain three benchmark tiers:

1. Microbenchmarks: k interpolation, random overlap, Planck, CIA, layer averaging.
2. Kernel benchmarks: one point spectrum, one transmission spectrum, one disc spectrum.
3. Retrieval benchmarks: fixed small dataset and fixed sampler/OE settings.

Each benchmark should record:

- wall time,
- memory allocation,
- numerical output checksum,
- hardware/backend,
- dependency versions.

Performance changes should not be accepted unless scientific reference tests still pass.

## Prioritized Recommendations

1. Build a reference test suite before optimization.
2. Move all I/O, plotting, and MPI outside the hot computational path.
3. Implement explicit opacity/layer/stellar caches with visible keys.
4. Keep a clear CPU reference backend.
5. Use Numba for early performance, but keep backend boundaries open.
6. Delay JAX/GPU until array contracts and reference outputs are stable.
7. Benchmark memory layout before large rewrites.

