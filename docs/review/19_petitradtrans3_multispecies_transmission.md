# Stable petitRADTRANS 3 Multispecies Transmission Benchmark

## Scope

This is the maintained multi-species transmission validation gate. It compares
ROBERT with stable petitRADTRANS 3.3.3 from 0.3 to
12 micron using 80 pressure points, 3689 wavelength bins, and the exact
16-point `petit_samples` quadrature stored in the pRT HDF tables. The atmosphere
contains H2O, CO, CO2, CH4, NH3, and HCN line opacity; H2-H2 and H2-He CIA; and
H2 and He Rayleigh extinction.

The pressure grid spans `1e-5` to 100 bar, temperature increases from 900 to
1800 K, gravity is 15 m/s2, the radius is anchored at `1e8` m and 0.01 bar, and
the stellar radius is 1.2 solar radii. Stable pRT3 is the only petitRADTRANS
reference used here; the pRT4 beta is excluded.

## Composition convention

petitRADTRANS receives mass mixing ratios and ROBERT receives volume mixing
ratios. The conversion is exact:

`x_i = (X_i / M_i) / sum_j(X_j / M_j)`.

For the adopted atmosphere this gives a mean molar mass of
`2.3242008615381975` amu. The trace-gas pRT mass fractions and ROBERT VMRs are:

| Species | pRT mass fraction | ROBERT VMR |
|---|---:|---:|
| H2O | 1.0e-3 | 1.2901275e-4 |
| CO | 3.0e-3 | 2.4893173e-4 |
| CO2 | 3.0e-4 | 1.5843403e-5 |
| CH4 | 1.0e-4 | 1.4487808e-5 |
| NH3 | 3.0e-5 | 4.0941807e-6 |
| HCN | 3.0e-5 | 2.5800278e-6 |

The benchmark converts pRT's species-resolved, mass-weighted opacity back to
effective molecular cross-sections before ROBERT combines the six gas
distributions with random overlap. This isolates the RT, mixture, and geometry
implementation from database and interpolation differences.

## Rayleigh physics

For a gas mixture, extinction is the number-weighted sum of each species'
molecular Rayleigh cross-section. It is not the square of a composition-
averaged refractivity. ROBERT now evaluates the H2 and He refractivity laws
separately and adds `x_H2 sigma_H2 + x_He sigma_He`. This is also structurally
consistent with pRT3, which computes and adds each requested Rayleigh species.

Two Rayleigh comparisons are retained:

1. ROBERT uses the continuum scattering opacity returned by pRT3. This is the
   strict RT and geometry isolation test.
2. ROBERT independently computes H2/He Rayleigh extinction. This additionally
   tests the refractivity and cross-section implementation.

Scattering is treated as removal from the unresolved direct stellar beam. No
scattering source term is returned to the beam; multiple scattering,
refraction, finite stellar angular size, and limb darkening are outside this
bounded comparison.

## Results

| Band | No-Rayleigh RMS | Shared-Rayleigh RMS | Native-Rayleigh RMS | pRT Rayleigh maximum effect |
|---|---:|---:|---:|---:|
| 0.3--12 micron | 4.28 ppm | 2.57 ppm | 3.28 ppm | 987.22 ppm |
| 0.3--1 micron | 6.30 ppm | 1.94 ppm | 4.05 ppm | 987.22 ppm |
| 1--12 micron | 2.82 ppm | 2.82 ppm | 2.83 ppm | 11.74 ppm |

The strict shared-Rayleigh result is the principal RT result. Its 1.94 ppm
optical RMS is tiny compared with the nearly 1000 ppm Rayleigh signature. The
independent ROBERT vertical Rayleigh optical depth is 1.0288 times the pRT3
value in the median over 0.3--1 micron. That 2.9% difference is retained as a
real difference between the molecular cross-section prescriptions rather than
removed by an empirical scale factor.

The larger no-Rayleigh optical residual occurs where the atmosphere is most
transparent and pressure-cell and reference-radius conventions control the
absolute baseline. In the molecular infrared the remaining structured peaks
are below 10 ppm and are consistent with the different spherical annulus and
pressure discretizations.

### Radius convention diagnostic

The positive shared-Rayleigh residual contains a systematic radius component.
A least-squares constant shift of -8.27 km to ROBERT's effective radius reduces
the full-band RMS from 2.57 to 0.98 ppm. This does **not** mean that either code
used the wrong requested reference pressure: log-pressure interpolation places
pRT's radius at 0.01 bar only 23.8 m below the requested `1e8` m, while ROBERT
anchors it exactly. Near the anchor, their pressure-center radii agree at the
metre-to-hundred-metre level.

The effective shift mainly measures their different discrete conventions. pRT
treats radii at the supplied pressure centres as annulus boundaries and uses a
trapezoidal chord integration. ROBERT constructs explicit pressure-cell edges,
treats each layer as a constant-property shell, and performs Gauss--Legendre
impact integration. Raw metrics remain the primary cross-code result; the
radius-aligned metric separates the largely retrievable absolute-radius mode
from wavelength-dependent spectral-shape differences.

## Timing

On the Apple-silicon CPU, pRT3 loads and constructs the atmosphere in 2.96 s.
Its first transmission calculation takes 0.682 s and its steady median is
0.591 s. One ROBERT native-Rayleigh case takes 1.02 s on its first call and
0.994 s at steady state. The three-case clear/shared/native diagnostic bundle
takes 1.80 s at steady state. Thus the current ROBERT native case is 1.68 times
slower than pRT3, or 0.59 times its throughput. This is substantially better
than the prior 2.42 s ROBERT measurement.

The comparison is not a pure opacity-I/O race: both steady-state timers exclude
table loading, but pRT evaluates its prepared native tables while ROBERT starts
from the species-resolved evaluated arrays exported by the reference run and
performs random overlap, native Rayleigh, and spherical transmission. The
timing is therefore a useful multi-gas performance gate, not a claim that the
two implementations perform identical work.

## Reproduction

```bash
/Users/jaketaylor/miniforge3/envs/petitradtrans-stable/bin/python \
  examples/run_petitradtrans3_multispecies_transmission.py \
  opacity_data/petitRADTRANS/input_data \
  examples/outputs/petitradtrans3_stable/multispecies_transmission_reference.npz

/Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/python \
  examples/benchmark_petitradtrans3_multispecies_transmission.py \
  --output-dir examples/outputs/multispecies_transmission_petitradtrans3
```

The first script writes its large pRT reference below the ignored runtime output
directory. The second writes a compact JSON report, compressed spectra, and a
four-panel plot to the tracked validation snapshot.

## Decision

The absorption/extinction spherical transmission core is validated for this
clear H2/He, molecular+CIA+Rayleigh domain. This does not claim validation for
refraction, aerosols, multiple scattering, patchy limbs, or finite-star
effects. This benchmark now acts as a regression gate while parameterized
transmission development continues toward configured multi-species retrievals
and aerosol physics.
