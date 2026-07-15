# Transmission Native-Opacity Convergence

## Scope

ROBERT transmission was compared with the same atmosphere evaluated using all
27,469 native ExoMolOP H2O POKAZATEL R=15,000 samples between the synthetic
observation limits. The reference calculation uses 240 pressure layers and
impact-quadrature order 24. Its source checksum is
`ba9d455db93809661aee79e7d322fde19da0236af1c1a9bb68f5b24735efb7fb`.

This is not an independent radiative-transfer comparison: it isolates
spectral compression and ROBERT numerical convergence. The next validation
stage must compare against a separate framework such as petitRADTRANS or
PICASO.

## Correlated-k convergence

The molecular-only sweep removes CIA and Rayleigh so the reported difference
isolates target-bin H2O k-distribution compression.

| g points | RMS (ppm) | Maximum absolute (ppm) | Mean bias (ppm) |
|---:|---:|---:|---:|
| 2 | 14.521 | 25.990 | -13.096 |
| 4 | 4.993 | 12.413 | -3.770 |
| 8 | 2.253 | 5.371 | -1.547 |
| 16 | 1.146 | 2.669 | -0.727 |
| 32 | 1.150 | 2.971 | -0.820 |
| 64 | 1.166 | 3.483 | -0.888 |

Sixteen points are the useful knee for this case. Higher orders do not improve
the discontinuous empirical inverse-CDF quadrature monotonically and cost more.

## Vertical and impact convergence

| Pressure layers | RMS (ppm) | Maximum absolute (ppm) |
|---:|---:|---:|
| 24 | 7.945 | 8.378 |
| 48 | 1.955 | 2.147 |
| 80 | 0.669 | 0.750 |
| 120 | 0.249 | 0.296 |
| 160 | 0.108 | 0.139 |

| Impact order | RMS (ppm) | Maximum absolute (ppm) |
|---:|---:|---:|
| 2 | 0.03139 | 0.03496 |
| 4 | 0.00492 | 0.00546 |
| 6 | 0.00159 | 0.00176 |
| 8 | 0.00069 | 0.00076 |

Pressure discretization matters much more than within-annulus impact
quadrature for this atmosphere.

## Recommended settings

- Production default: 16 g-points, 80 pressure layers, impact order 4. The
  combined difference is 1.072 ppm RMS and 2.150 ppm maximum.
- Fast exploratory retrieval: 8 g-points, 48 pressure layers, impact order 4
  or 6. The tested order-6 configuration is 1.755 ppm RMS and 4.119 ppm
  maximum.
- High-accuracy final checks: 16 g-points, at least 120 pressure layers, and
  impact order 4.

These recommendations apply to this clear, isothermal hot-Jupiter H2O case.
They should be rechecked for high-mean-molecular-weight atmospheres, sharp
temperature gradients, aerosols, and multi-species random overlap.
