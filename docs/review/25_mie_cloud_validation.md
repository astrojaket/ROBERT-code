# Mie-cloud validation for the WASP-69b retrieval

## Implemented chain

The retrieval-facing path is now:

1. select tabulated `n(lambda), k(lambda)` or retrieve nodal values;
2. calculate homogeneous-sphere Lorenz-Mie efficiencies and scalar phase
   moments;
3. convert cross sections to opacity per condensate mass;
4. convert condensate mass fraction to layer optical depth using
   `delta_p / g`;
5. mix cloud and Rayleigh phase moments by scattering optical depth;
6. solve thermal multiple scattering with P3/SH4 and degree-four delta-M
   scaling; and
7. integrate the spectrum onto the four observed NIRCam/MIRI datasets.

The phase convention is `chi_0 = 1`, `chi_1 = 3g`. The SH4 solver consumes
`chi_0` through `chi_3`; the unresolved-forward fraction is `chi_4 / 9`.

## Independent Mie checks

ROBERT was compared with `miepython 3.2.0`, installed only in a temporary
validation directory and not added as a runtime dependency. Checks covered
size parameters 0.1, 1, 5, and 20, including absorbing particles. Extinction
and scattering efficiencies agreed to relative errors of order `1e-12` or
smaller, asymmetry to order `1e-15`, and phase moments to order `1e-13`.

For `x = 5` and `m = 1.6 + 0.1i`, independent 256-point Gauss-Legendre
integration of the normalized `miepython` phase function gives

```text
[1.000000000000000,
 2.324914198318782,
 3.181268856917814,
 3.596168330101814,
 3.979001987734556]
```

This value is retained as a regression test. The analytic Rayleigh limit is
also tested as `[1, 0, 0.5, 0, 0]` in the same convention.

## Size-distribution convergence

The public lognormal model defines
`r_eff = <r^3>/<r^2>` and integrates a number distribution over six standard
deviations in log radius. A convergence audit at `sigma_g = 1.5` showed that
12--24 radius points are not uniformly adequate over a broad radius prior:
for `r_eff = 1 micron`, 12 versus 48 points changed mass opacity by as much as
about 4.7% and individual moments by about 0.20; at `10 micron`, the opacity
difference was larger. Consequently the first WASP-69b model is explicitly
monodisperse (`sigma_g = 1`, one radius point). A distributed-size retrieval
must select its quadrature by convergence testing over that run's prior.

## Radiative-transfer checks

Exact non-Henyey-Greenstein moments produce a measurably different spectrum
from an HG closure with the same asymmetry. Splitting a cloud into separate
absorption/scattering layer contributors preserves the SH4 result. Rayleigh
and cloud moments are tested individually and in mixtures. The underlying SH4
implementation has a controlled matched-coefficient PICASO comparison with a
maximum relative point-radiance difference of `3.23e-6` for the tested grey
cases.

## WASP-69b readiness and limits

`examples/retrieve_wasp69b_mie_cloud.py` builds the 280-point NIRCam+MIRI
problem and supports:

- `catalog`: fixed laboratory optical constants (default `MgSiO3`), 12 fitted
  parameters;
- `direct-nk`: six real-index and six log-imaginary-index nodes in addition to
  the shared atmospheric/cloud parameters, 24 fitted parameters.

Both modes return finite likelihoods in smoke evaluation. The catalog mode is
the recommended first evidence run; direct optical-constant retrieval should
follow after the cloud evidence and prior sensitivity are established.

This validation does not make homogeneous spheres universal. Non-spherical or
aggregate particles, porosity, effective-medium mixtures, polarization, and
Kramers-Kronig coupling are outside this model. SH4 also remains a truncated
angular representation. Posterior samples with extreme forward scattering
should be checked against a high-stream solver before a final physical claim.
