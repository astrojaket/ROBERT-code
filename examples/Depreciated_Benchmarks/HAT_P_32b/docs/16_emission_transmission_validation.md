# Realistic Emission and First Transmission Validation

## HAT-P-32b FastChem molecular+CIA emission

The bundled HAT-P-32b MAP state was evaluated with:

- the Madhusudhan–Seager temperature profile;
- FastChem equilibrium abundances for H2O, CO2, CO, CH4, NH3, HCN, H2, and He;
- six ExoMolOP/exo_k correlated-k opacity archives with random overlap;
- the provenance-pinned NemesisPy H2-H2/H2-He CIA table;
- 80 pressure layers, 117 wavelength bins, and 20 g ordinates.

ROBERT and PICASO received identical molecular+CIA layer optical depths, g
weights, pressure edges, edge temperatures, angular quadrature, and boundary
conditions. The pressure-grid storage order was explicitly converted to
top-to-bottom for PICASO's low-level routine.

| Metric | Result |
|---|---:|
| Maximum disk relative RT difference | `3.607e-5` (0.00361%) |
| RMS disk relative RT difference | `1.325e-5` (0.00133%) |
| Maximum point-radiance difference | `1.157e-4` (0.0116%) |
| Maximum CIA effect on eclipse depth | 4.51 ppm |

These are RT-only differences; the test deliberately controls opacity rather
than comparing independent opacity databases.

Reproduce with:

```bash
python examples/compare_hat_p_32b_emission_picaso.py
```

## Absorption-dominated spherical transmission

ROBERT now includes a first transmission solver for absorption/extinction in
concentric spherical shells. It uses exact shell chord lengths and integrates
`2 b [1 - T(b)] db` over impact parameter with Gauss–Legendre quadrature.
Correlated-k transmission is integrated over g before atmospheric annulus area.
The opaque base radius is explicit and was anchored at 10 bar for this
benchmark.

The same HAT-P-32b FastChem atmosphere, molecular opacity, and CIA were reused.

| Metric | Result |
|---|---:|
| Transit-depth range | 28,895.7–30,738.4 ppm |
| Molecular spectral modulation | 1,842.7 ppm |
| Maximum CIA transit-depth effect | 1.84 ppm |
| Order 8 vs order 16 impact integration | 0.0133 ppm maximum |
| Order 4 vs order 16 impact integration | 0.0987 ppm maximum |

The transparent and opaque limits and correlated-k-before-area ordering have
analytic tests. Current scope is deliberately absorption dominated. Scattered
light returning to the stellar beam, refraction, finite stellar angular size,
and limb darkening are explicitly not included.

Reproduce with:

```bash
python examples/benchmark_hat_p_32b_transmission.py
```

The next transmission validation should compare this solver with an independent
code using identical extinction coefficients and radius levels, followed by
reference-pressure/radius sensitivity and refractive-boundary tests.
