# Stable petitRADTRANS 3 Multispecies Transmission Benchmark

## Scope

This is the final transmission validation before transmission development is
placed on hold. It compares ROBERT with stable petitRADTRANS 3.3.3 from 0.3 to
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

## Timing

On the Apple-silicon CPU, pRT3 loads and constructs the atmosphere in 2.56 s.
Its first transmission calculation takes 0.712 s and its steady median is
0.601 s. One ROBERT native-Rayleigh case takes 2.44 s on its first call and
2.42 s at steady state. The three-case clear/shared/native diagnostic bundle
takes 3.22 s at steady state because gas random overlap is shared between its
three RT solves. The comparison is not a pure opacity-I/O race: pRT evaluates
its native tables, while ROBERT starts from the species-resolved arrays
exported by that reference run and performs random overlap plus RT. The stable
pRT3 implementation is therefore about four times faster for this end-to-end
case; optimizing ROBERT's multi-gas random-overlap path is emission-relevant
work, but must preserve the validated physics.

## Reproduction

```bash
/Users/jaketaylor/miniforge3/envs/petitradtrans-stable/bin/python \
  examples/run_petitradtrans3_multispecies_transmission.py \
  opacity_data/petitRADTRANS/input_data \
  examples/outputs/petitradtrans3_stable/multispecies_transmission_reference.npz

/Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/python \
  examples/benchmark_petitradtrans3_multispecies_transmission.py
```

The scripts write a JSON report, compressed spectra, and a four-panel benchmark
plot below `examples/outputs/petitradtrans3_stable/`.

## Decision

The absorption/extinction spherical transmission core is validated for this
clear H2/He, molecular+CIA+Rayleigh domain. This does not claim validation for
refraction, aerosols, multiple scattering, patchy limbs, or finite-star
effects. Transmission development is now frozen by design. The downloaded
opacities, composition conversion, correlated-k quadrature, random-overlap
combination, hydrostatic columns, and continuum machinery will next be reused
to finish the chemically complete emission validation.
