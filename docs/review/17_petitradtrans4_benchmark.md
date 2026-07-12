# petitRADTRANS 4 JAX Benchmark

## Scope

This benchmark compares ROBERT with the JAX rewrite of petitRADTRANS 4, not
the stable petitRADTRANS 3 release. The installed reference is
petitRADTRANS `4.0.0b0` from the official `jax_dev` branch at commit
`8f612343f9cdef0b232c4ec469f0a96b9224ccd8`, using JAX `0.10.2` on the
Apple-silicon CPU backend.

Both calculations use 80 pressure points from `1e-5` to 100 bar, a temperature
profile from 900 to 1800 K, constant gravity of 15 m/s2, and 619 wavelength
bins from 2.8 to 5.2 micron. The molecular opacity is the official pRT-format
ExoMol POKAZATEL H2O R=1000 correlated-k table with 16 g ordinates. CIA,
clouds, and scattering are excluded from this first controlled comparison.

ROBERT now has a unit-explicit reader for pRT correlated-k HDF5 files. For the
strict emission comparison, pRT4's evaluated cumulative optical depths are
differenced and supplied directly to ROBERT. The residual therefore cannot be
attributed to a different opacity database or pressure-temperature opacity
interpolator.

## Results

### Transmission

The absorption-only transmission spectra agree well:

| Metric | ROBERT - pRT4 |
|---|---:|
| Median transit-depth difference | 1.72 ppm |
| RMS transit-depth difference | 1.73 ppm |
| Maximum absolute difference | 2.14 ppm |

The remaining difference includes spherical-annulus discretization and the
codes' pressure-boundary conventions. This result supports the current ROBERT
absorption transmission geometry at the precision of this benchmark. A later
test should add the same H2-H2 and H2-He CIA to both paths.

### Emission

At 80 pressure points the median emission residual is small (`-0.031%`), but
the RMS is `8.76%` and the maximum is `34.8%`. This initially looks like a
ROBERT discrepancy, but a pRT4-only pressure-resolution test reveals that the
beta reference is not self-converged for this molecular case:

| pRT4 pressure grids | RMS change | Maximum change |
|---|---:|---:|
| 40 vs 80 | 8.78% | 34.9% |
| 80 vs 160 | 16.4% | 52.2% |
| 160 vs 320 | 11.5% | 38.8% |
| 320 vs 640 | 5.46% | 22.0% |

The changes occur in wavelength blocks whose boundaries move with the number
of pressure points. That is not a physically plausible convergence sequence.
It is evidence of a pRT4 beta implementation defect on this commit/backend,
most likely in array indexing or batching, rather than evidence against
ROBERT's formal solution. ROBERT's same-opacity thermal solver has already
agreed with PICASO at `0.00133%` RMS for the HAT-P-32b molecular+CIA case.

The pRT4 emission result must therefore not be used as a science validation
reference until the behavior is reproduced against a newer pRT4 commit and,
ideally, reported upstream with this minimal case.

## Timings

Synchronized wall timings on this Apple-silicon CPU were:

| Calculation | First call | Steady median |
|---|---:|---:|
| pRT4 emission | 0.420 s | 0.0266 s |
| ROBERT emission | 1.30 s | 0.0598 s |
| pRT4 transmission | 0.300 s | 0.0206 s |
| ROBERT transmission | 0.142 s | 0.149 s |

The pRT4 first call includes JAX compilation. The ROBERT emission first call
includes Numba compilation. The scopes are not perfectly symmetric: pRT4 is a
high-level opacity-plus-RT call, while the strict ROBERT emission timing starts
from pRT4's pre-evaluated cumulative optical depth. These figures are useful
engineering measurements, not a definitive code-speed ranking. On this CPU,
pRT4 is faster after compilation; a GPU benchmark remains to be performed on
suitable hardware.

## Reproduction

Download or verify the minimal pRT input data:

```bash
python examples/setup_petitradtrans4_data.py
```

Run the comparison and timing benchmark:

```bash
python examples/benchmark_petitradtrans4.py
```

The opacity files are stored below
`opacity_data/petitRADTRANS/input_data/`, which is ignored by git. Outputs are
written to `examples/outputs/petitradtrans4/`.

## Recommended next action

Preserve the pRT4 benchmark as an expected-failure external diagnostic and
report its pressure-grid-dependent emission blocks upstream with the exact
commit, platform, input table, and convergence plot. Stable pRT 3.3.3 has now
been run over 0.3--12 micron with H2O and H2 CIA; the converged results are
documented in `docs/review/18_petitradtrans3_stable_benchmark.md` and supersede
pRT4 as the current petitRADTRANS science-validation reference.
