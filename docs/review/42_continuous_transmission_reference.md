# Continuous transmission reference benchmark

## Question

The matched-PICASO audit showed that ROBERT and PICASO use different
finite-layer extinction and annulus conventions. This benchmark asks which
scheme approaches a continuous atmosphere more accurately, without assuming
that either code is the reference.

## Independent continuous contract

The atmosphere is isothermal at 1200 K with mean molecular weight 2.3 amu,
10 bar radius 75,567,044 m, gravity 8.42 m/s2 at that radius, and a 1e-9 bar
top. Gravity follows the exact inverse-square relation. Pressure and radius
therefore obey the analytic hydrostatic mapping

`ln(P/P0) = A (1/r - 1/r0)`.

A smooth analytic cross-section from 1--12 microns contains six broad
absorption bands. It is deliberately database-independent and separable from
the density profile. This isolates spherical transport and vertical
discretization from chemistry, opacity interpolation, correlated-k mixing,
and cloud physics.

The reference directly evaluates

`tau(b, lambda) = 2 sigma(lambda) integral n(sqrt(b^2+x^2)) dx`

and then integrates blocked area over impact parameter. The production
reference uses 1024-point Gauss--Legendre quadrature for both integrations.
Repeating it at 512 points changes the spectrum by only 2.73e-9 ppm RMS and
6.46e-9 ppm maximum, far below either code's finite-layer error.

## Code paths compared

ROBERT receives exact layer-integrated vertical optical depth and exact
analytic pressure-edge radii. Its production spherical-shell solver treats
each layer as a uniform-extinction shell and uses order-8 Gauss--Legendre
annulus integration.

PICASO receives the same pressure levels, exact analytic radii, temperature,
mean molecular weight, and cross-section. The external PICASO 3.2.2
environment runs the actual `picaso.fluxes.get_transit_1d` kernel. Giving it
the exact analytic radii is deliberately favourable: this benchmark measures
the transmission discretization, not PICASO's separate altitude builder.

## Results

| Layers | ROBERT RMS | PICASO RMS | ROBERT median | PICASO median |
|---:|---:|---:|---:|---:|
| 16 | 33.84 ppm | 80.15 ppm | +33.90 ppm | -80.85 ppm |
| 32 | 8.95 ppm | 32.89 ppm | +9.03 ppm | -33.33 ppm |
| 64 | 2.29 ppm | 13.38 ppm | +2.31 ppm | -13.57 ppm |
| 80 | 1.48 ppm | 10.02 ppm | +1.49 ppm | -10.16 ppm |
| 128 | 0.58 ppm | 5.46 ppm | +0.59 ppm | -5.54 ppm |
| 256 | 0.15 ppm | 2.25 ppm | +0.15 ppm | -2.29 ppm |
| 512 | 0.037 ppm | 0.946 ppm | +0.037 ppm | -0.963 ppm |

A log-log fit from 32--512 layers gives convergence orders of 1.98 for ROBERT
and 1.28 for PICASO. ROBERT is effectively second order for this smooth
atmosphere, while PICASO's level-density/rectangular-annulus method converges
more slowly. The opposite residual signs explain the cancellation observed
when the two finite-layer choices were mixed in the preceding audit.

At the 80-layer resolution used in the cloud benchmarks, ROBERT's maximum
absolute error is 1.53 ppm. PICASO's is 10.67 ppm. The earlier 1.9--4.2 ppm
matched-code differences are therefore compatible with ROBERT being closer to
the continuous result; agreement with PICASO should not be used as the
accuracy target for this case.

## Scope and next extension

This is a controlled smooth, isothermal, pure-absorption test. It establishes
that ROBERT's uniform-shell Gaussian integral is the better finite-resolution
approximation for this contract. It does not yet establish the same ordering
for sharp temperature/composition gradients, pressure-dependent molecular
cross-sections, cloud boundaries, refraction, or forward scattering.

The next useful extension is a non-isothermal continuous atmosphere with
pressure- and temperature-dependent analytic opacity, followed by an
edge-aligned and non-edge-aligned cloud-top convergence test. Those cases will
test the layer interpolation choices that this separable-opacity benchmark
intentionally removes.

## Artifacts

- `examples/outputs/continuous_transmission_reference/continuous_transmission_reference.json`
- `examples/outputs/continuous_transmission_reference/continuous_transmission_reference.png`
- `examples/outputs/continuous_transmission_reference/continuous_transmission_reference_spectra.npz`
- `examples/outputs/continuous_transmission_reference/picaso_continuous_transmission.npz`
