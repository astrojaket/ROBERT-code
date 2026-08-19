# Stage 9 Numba and Stronger Grey-Cloud Rerun

Date: 2026-08-19

## Outcome

ROBERT now uses the Numba CPU kernels by default for thermal integration, SH4
spectrum reconstruction, and the SH4 boundary solve. The NumPy and SciPy paths
remain available as explicit scientific references.

The frozen Stage 9 ROBERT adapter forced `backend="numpy"`. On the exact
80-layer, 3696-wavelength, 16-g Stage 9 scattering input, selecting the existing
Numba reconstruction reduced the local warm call from the archived
600.86-second 12-rank cluster observation to 8.01 seconds on one M4 process.
The machines and concurrency differ, so this is not a direct hardware speedup
ratio. The important result is numerical: the maximum eclipse-depth difference
from the archived NumPy spectrum was `2.82e-18`.

Adding the compiled, pivoted SH4 boundary solver reduced the same local warm
call to 5.25 seconds. Disabling unused species-tau retention reduced it to
5.03 seconds. The compiled result differed from the archived Stage 9 spectrum
by at most `4.12e-18` in eclipse depth (`4.12e-12 ppm`). The one-process peak
RSS was 7.46 GiB. No Metal or large JAX allocation was used.

Eight consecutive exact stronger-scattering calls had identical spectral
checksums. After explicit collection, live RSS stayed between 0.60 and
1.00 GiB with no monotonic growth. The larger process high-water mark is a
temporary full-grid allocation, not a retained per-call leak.

## Exact warm profile after the SH4 change

| Component | Time (s) | Complete-call share |
| --- | ---: | ---: |
| Complete forward call | 5.246 | 100.0% |
| SH4 spectrum | 4.336 | 82.6% |
| SH4 layer/boundary work | 2.631 | 50.2% |
| Compiled pivoted band solve | 1.688 | 32.2% |
| Compiled source reconstruction | 1.570 | 29.9% |
| Gas optical depth and RORR | 0.610 | 11.6% |
| Opacity interpolation | 0.142 | 2.7% |
| CIA | 0.086 | 1.6% |
| Planck construction | 0.004 | 0.1% |
| Final R=100 binning | 0.006 | 0.1% |

Parent timings include child timings and must not be added together.

## Stronger cloud selection

The original truth used optical depth 1 at 5 microns and a 0.01 bar cloud top.
Its archived RMS spectral effects relative to the clear non-inverted case were
24.4 ppm for ROBERT absorbing cloud and 42.9 ppm for ROBERT scattering cloud.

A bounded grid test selected:

- optical depth at 5 microns: `3.0`;
- cloud-top pressure: `0.003 bar`;
- absorbing albedo: `0.0`;
- scattering albedo: `0.9`.

These truths remain inside the existing priors. They are not on a prior edge.
For ROBERT, the absorbing-cloud effect becomes 94.8 ppm RMS with a 364.8 ppm
maximum. The scattering-cloud effect becomes 183.9 ppm RMS with a 517.9 ppm
maximum. This is 3.88 and 4.29 times the original RMS effect.

Use new scenario names and a new output root for the rerun. Do not replace the
frozen Stage 9 products. The rerun should contain only the two grey-cloud
scenarios, which gives 36 directed cross-framework retrievals across the three
noise tiers.

## Cluster adapter requirements

The Stage 9 ROBERT adapter must:

1. stop forcing `backend="numpy"`;
2. select `backend="numba"` and `boundary_backend="numba"`;
3. pass `retain_species_tau=False` for the diagnostics-free gas calculation;
4. retain one Numba and one OpenMP thread per MPI rank;
5. warm each rank once before timed or sampled calls; and
6. keep the NumPy/SciPy result as the parity oracle.

The stronger-cloud truth must be supplied from the scenario definition rather
than hard-coded in `parameter_definitions()`. Generate a new contract and new
injections. Keep the original contract and results read-only.

Reproduce the one-process check with:

```bash
conda run -n robert-exoplanets env \
  PYTHONPATH=$PWD/src \
  STAGE9_PRT_INPUT_DATA=/path/to/ROBERT-stage9/reference/petitradtrans/input_data \
  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/benchmark_stage9_robert_grey_cloud.py \
  /path/to/stage9-source /path/to/ROBERT-stage9 \
  --cloud-tau-5um 3 --cloud-top-pressure-bar 0.003 \
  --output /tmp/stage9-robert-stronger-grey-cloud.json
```
