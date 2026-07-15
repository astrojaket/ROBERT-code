# Shared Deck-and-Haze Transmission Recovery Review

## Architectural outcome

The deck-and-haze prescription is not owned by either emission or
transmission. `ParameterizedDeckHazeCloudModel` converts retrieval parameters
and a common gas optical-depth state into `CloudOpticalProperties`. Both
forward models now use the same atmosphere builder, chemistry, thermal
structure, gas-opacity evaluation, CIA, Rayleigh, and cloud-extinction
assembly. They diverge only when the assembled atmospheric extinction is
mapped through disc-emission or spherical-transit geometry.

The model contains two components:

- a finite grey deck below a retrieved cloud-top pressure, with retrieved
  integrated vertical extinction optical depth; and
- a vertically well-mixed haze whose reference mass extinction is in square
  centimetres per gram of bulk atmosphere and whose wavelength dependence is
  a retrieved power law.

The shared optical properties retain single-scattering albedo and asymmetry.
Transmission treats scattering as extinction out of the direct stellar beam.
Emission sends the same absorption/scattering split to its configured
multiple-scattering backend, SH4 by default.

## Validation contract

- Molecules: H2O, CO, CO2, CH4, NH3, and HCN.
- Opacity: real ExoMolOP R=15000 cross sections, converted into eight-point
  empirical correlated-k distributions inside 96 observation bins from
  0.6--12 micron.
- Atmosphere: 48 layers from `1e-7` to 10 bar, 1100 K, H2/He background, CIA,
  Rayleigh extinction, inverse-square gravity, and a 1-bar reference radius.
- Aerosols: cloud top at `1e-3` bar, integrated deck optical depth 0.1, haze
  reference mass extinction `10^-4.3 cm2/g`, and haze slope -4.
- Data: deterministic 8 ppm Gaussian noise, seed 20260717.
- Retrieval: 11 parameters, MultiNest with two MPI processes, 50 live points,
  `dlogz=0.5`, and no iteration cap.

Fifty live points remain a validation-scale configuration. The result tests
end-to-end closure and exposes degeneracies; it is not a production posterior
or evidence recommendation.

## Recovery

| Parameter | Truth | Best fit | Absolute error | Posterior sigma |
|---|---:|---:|---:|---:|
| log10 H2O | -3.200 | -3.324 | 0.124 | 0.103 |
| log10 CO | -3.000 | -3.188 | 0.188 | 0.132 |
| log10 CO2 | -3.500 | -3.653 | 0.153 | 0.110 |
| log10 CH4 | -3.700 | -3.835 | 0.135 | 0.100 |
| log10 NH3 | -3.600 | -3.703 | 0.103 | 0.099 |
| log10 HCN | -3.800 | -3.923 | 0.123 | 0.107 |
| Radius scale | 1.003000 | 1.003795 | 0.000795 | 0.000789 |
| log10 cloud-top pressure (bar) | -3.000 | -2.855 | 0.145 | 0.097 |
| log10 deck optical depth | -1.000 | -0.953 | 0.047 | 0.045 |
| log10 haze mass extinction | -4.300 | -4.842 | 0.542 | 0.306 |
| Haze slope | -4.000 | -5.232 | 1.232 | 1.091 |

All validation-scale recovery criteria pass. The best-fit chi-square is 81.30,
or reduced chi-square 0.956 for 85 degrees of freedom. The weighted posterior
contains 1050 samples with effective sample size 337.

## What the posterior says

The molecular abundances and reference radius remain highly correlated after
adding aerosols. Radius and H2O have correlation -0.98, while CH4 and NH3 have
the largest pairwise correlation at +0.99. Radius and cloud-top pressure have
correlation +0.69.

The haze parameters are deliberately less identifiable than the deck:
reference mass extinction and slope correlate at +0.83. Their injected values
lie within the posterior 16--84% intervals, even though the single best-fit
point is displaced. This is why the quick-test tolerances for haze are broader
than those for the compact deck posterior. The result should be read as a
detected aerosol continuum with a weakly determined haze law, not a precise
haze characterization.

## Performance and decision

MultiNest used 86,720 likelihood evaluations and 1633 accepted replacements.
The sampler phase took 1453 seconds and configured inference, persistence, and
plotting took 1466 seconds on two local MPI processes.

The shared deck-and-haze foundation passes its first cloudy transmission
closure test. The next cloud step should move the existing Mie particle model
behind the same geometry-independent cloud protocol, then validate it in both
emission and transmission. Chemistry and thermal structure already enter both
modes through the shared `AtmosphereBuilder`; future additions should continue
to live there rather than in geometry-specific forward models.
