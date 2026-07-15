# Six-Molecule Transmission Injection-Recovery Review

## Purpose

This is the first configured multi-gas closure test of the parameterized
transmission foundation. It tests whether the YAML workflow can prepare six
real molecular databases once, retrieve all six mixing ratios together with
the reference radius, run MultiNest under MPI, and produce auditable recovery
and posterior outputs.

It complements rather than replaces the stable-petitRADTRANS benchmark. The
pRT comparison is an independent forward-physics test. This injection recovery
uses ROBERT to generate and retrieve the same data, so it tests inference and
integration closure.

## Contract

- Molecules: H2O, CO, CO2, CH4, NH3, and HCN.
- Opacity: ExoMolOP R=15000 cross sections converted to 8-point empirical
  correlated-k distributions inside each observation bin.
- Spectrum: 96 logarithmic bins over 0.6--12 micron with 8 ppm Gaussian noise.
- Atmosphere: 48 layers from `1e-7` to 10 bar at 1100 K, with an H2/He
  background, CIA, and Rayleigh extinction.
- Geometry: 1-bar reference radius, inverse-square gravity, and sixth-order
  impact quadrature.
- Retrieval: seven parameters, MultiNest, 2 MPI processes, 40 live points,
  `dlogz=0.5`, seed 20260716.

Forty live points are intentionally a validation-scale setting. They are enough
to demonstrate deterministic closure here, but are not a recommended setting
for production posterior or evidence claims.

## Recovery

| Parameter | Truth | Best fit | Absolute error | Posterior sigma |
|---|---:|---:|---:|---:|
| log10 H2O | -3.200 | -3.239 | 0.039 | 0.022 |
| log10 CO | -3.000 | -3.033 | 0.033 | 0.102 |
| log10 CO2 | -3.500 | -3.533 | 0.033 | 0.024 |
| log10 CH4 | -3.700 | -3.721 | 0.021 | 0.019 |
| log10 NH3 | -3.600 | -3.615 | 0.015 | 0.017 |
| log10 HCN | -3.800 | -3.862 | 0.062 | 0.037 |
| Radius scale | 1.003000 | 1.003160 | 0.000160 | 0.000108 |

All preregistered parameter tolerances pass. The best-fit chi-square is 83.79,
giving reduced chi-square 0.941 for 89 degrees of freedom. The weighted
posterior contains 660 samples with effective sample size 237.

## Radius and abundance degeneracy

The retrieved reference radius is meaningfully coupled to the gas abundances,
as expected for transmission spectroscopy. Its posterior correlations are
-0.71 with H2O, -0.59 with CO2, -0.60 with CH4, and -0.69 with NH3. The largest
gas-gas correlation is +0.81 between CH4 and NH3. Broad 0.6--12 micron coverage
still separates all six species in this high-information synthetic case.

This directly answers the radius-offset concern in the independent pRT plot:
when radius is allowed to move, the retrieval absorbs the nearly constant
absolute-depth mode while molecular band shapes retain abundance information.

## Performance

After one-time Numba compilation, one seven-parameter forward evaluation takes
about 0.023 seconds on the Apple-silicon CPU. MultiNest performed 8705
likelihood evaluations and converged in 151 seconds of sampler time; configured
setup, inference, saving, and plotting took 164 seconds end to end.

## Decision

The clear six-molecule parameterized transmission path is closed at validation
scale. The next physics increment should introduce a minimal aerosol model and
repeat this test with deliberately weaker molecular identifiability, while
retaining the clear case as the regression baseline.
