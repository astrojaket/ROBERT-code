# Stable petitRADTRANS 3 Benchmark, 0.3--12 Micron

## Purpose

The petitRADTRANS 4 beta benchmark exposed non-physical, pressure-grid-sized
wavelength blocks in its emission spectra. This comparison therefore uses the
current stable release, petitRADTRANS 3.3.3, as the reference and expands the
default validation domain to 0.3--12 micron.

The controlled atmosphere has 80 pressure points from `1e-5` to 100 bar, a
temperature profile from 900 to 1800 K, and constant gravity of 15 m/s2. The
opacity sources are the pRT-format ExoMol POKAZATEL H2O R=1000 correlated-k
table and the pRT H2-H2 and H2-He collision-induced absorption tables. The
calculation has 3689 wavelength bins and uses the table's exact 16-point
`petit_samples` g quadrature.

## Composition convention

petitRADTRANS consumes mass mixing ratios, while ROBERT consumes volume mixing
ratios. The pRT atmosphere uses

- H2 mass fraction: 0.740,
- He mass fraction: 0.259,
- H2O mass fraction: 0.001.

ROBERT converts these using

`x_i = (X_i / M_i) / sum_j(X_j / M_j)`.

This gives an H2O volume mixing ratio of `1.2853674809066052e-4` and a mean
molar mass of `2.3156255071` amu. Treating the pRT mass fraction as a ROBERT
volume mixing ratio would make the atmospheres physically different by almost
an order of magnitude.

For the strict RT comparison, the total H2O+CIA opacity evaluated by pRT is
supplied to ROBERT. For emission, pRT's returned cumulative optical depth is
also supplied directly. The comparison therefore tests RT and geometry rather
than opacity interpolation or database differences.

## Results

| Wavelength band | Emission RMS | Emission maximum | Transmission RMS | Transmission maximum |
|---|---:|---:|---:|---:|
| 0.3--12 micron | 0.0733% | 0.208% | 3.90 ppm | 10.46 ppm |
| 0.3--1 micron | 0.0940% | 0.208% | 6.32 ppm | 10.46 ppm |
| 1--5 micron | 0.0729% | 0.145% | 1.82 ppm | 2.48 ppm |
| 5--12 micron | 0.0263% | 0.0346% | 1.72 ppm | 3.00 ppm |

The emission residual is smooth and sub-percent everywhere. The remaining
shape is consistent with the analytic formal integral in ROBERT versus pRT3's
Feautrier discretization and lower-boundary treatment. The transmission
residual is smallest in the near- and mid-infrared. Its larger optical value
occurs where the atmosphere becomes transparent and reference-radius and
pressure-cell boundary conventions dominate the absolute transit radius.

These results validate ROBERT's current absorption-only emission and spherical
transmission paths for molecular+CIA calculations over the requested domain.

## Stable-code self-convergence

Unlike the JAX beta, stable pRT3 shows regular near-second-order convergence:

| pRT3 pressure grids | RMS change | Maximum change |
|---|---:|---:|
| 40 vs 80 | 0.0994% | 0.396% |
| 80 vs 160 | 0.0238% | 0.0888% |
| 160 vs 320 | 0.00591% | 0.0225% |
| 320 vs 640 | 0.00146% | 0.00556% |

This establishes stable pRT3, rather than the current pRT4 beta commit, as the
appropriate petitRADTRANS emission reference for the present validation work.

## Timings

On the Apple-silicon CPU:

| Calculation | First call | Steady median |
|---|---:|---:|
| pRT3 emission | 0.112 s | 0.109 s |
| ROBERT emission | 1.44 s | 0.232 s |
| pRT3 transmission | 0.114 s | 0.113 s |
| ROBERT transmission | 0.454 s | 0.444 s |

The pRT calls include opacity evaluation and RT. The strict ROBERT paths start
from pRT-evaluated opacity or optical depth, so this is not a perfectly
symmetric end-to-end speed ranking. ROBERT's first emission call includes
Numba compilation. The stable Fortran pRT reference is currently faster on
this grid; optimization should follow physics validation, not precede it.

## Reproduction

The stable pRT calculation is isolated in the `petitradtrans-stable` conda
environment:

```bash
/Users/jaketaylor/miniforge3/envs/petitradtrans-stable/bin/python \
  examples/run_petitradtrans3_stable_reference.py \
  opacity_data/petitRADTRANS/input_data \
  examples/outputs/petitradtrans3_stable/reference_80.npz
python examples/benchmark_petitradtrans3_stable.py
```

The plots and machine-readable reports are written below
`examples/outputs/petitradtrans3_stable/`.

## Follow-up

The six-molecule+CIA+Rayleigh transmission comparison is complete and recorded
in `19_petitradtrans3_multispecies_transmission.md`. Transmission development
is now intentionally on hold while the shared opacity, chemistry, and
hydrostatic infrastructure is exercised in the emission path.
