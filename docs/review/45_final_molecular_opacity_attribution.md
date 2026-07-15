# Final PICASO--ROBERT molecular-opacity attribution

## Decision

The transmission diagnostic campaign closes with this comparison. No
production change is required. On the exact end-to-end retrieval atmosphere,
the active molecular bands agree closely when both codes use their full native
opacity grids. The remaining spectrum-scale difference is comparable to the
inter-code spread reported by Barstow et al. (2020), not evidence of an
incorrect transmission solver.

## Why the retrieval result was revisited

The quick PICASO injection-recovery used every fifth opacity sample. That was
appropriate for a fast retrieval gate, but the earlier official-PICASO
convergence benchmark showed that striding an R=15,000 grid is not a
flux-conserving resampling method at ppm precision.

For the final diagnostic, PICASO evaluated all 40,620 native samples from
0.8--12 microns and exported its H2O, CO, CO2, and CH4 layer optical depths.
ROBERT independently evaluated all 37,275 ExoMolOP samples from 1--12 microns.
Both results were integrated into the same JWST-like bins on the same 48-layer,
1100 K, constant-composition atmosphere. CIA remains present in the full
spectra but is not re-benchmarked; its implementation and matched-optical-depth
transmission validation were already completed.

## Sampling result

PICASO stride 5 differs from its own full-native spectrum by 30.85 ppm RMS and
106.77 ppm maximum. Replacing the quick injection with full-native PICASO
reduces the ROBERT-minus-PICASO truth-spectrum discrepancy from 42.51 to
38.84 ppm RMS and from 111.29 to 88.20 ppm maximum. The uncertainty-weighted
chi-square falls from 242.52 to 171.94 over 72 bins.

This does not mean the remaining 38.84 ppm is a molecular-opacity amplitude
error. It is the total end-to-end difference, including independent opacity
tabulation, within-bin opacity distributions and correlated-k treatment,
continuum prescriptions, cloud interaction, and the already quantified small
geometry differences.

## Species-resolved molecular opacity

To avoid ratios dominated by transparent windows, the primary statistic uses
only bins where either code's species column optical depth is at least one per
cent of that species' maximum.

| Species | Active bins | RMS log10 tau difference | Median log10 ratio | Maximum |
|---|---:|---:|---:|---:|
| H2O | 46 | 0.0155 dex | +0.0064 dex | 0.0560 dex |
| CO | 7 | 0.0658 dex | +0.0699 dex | 0.0987 dex |
| CO2 | 5 | 0.0395 dex | -0.0156 dex | 0.0837 dex |
| CH4 | 28 | 0.0213 dex | +0.0154 dex | 0.0420 dex |

H2O and CH4 agree to a few per cent in their active bands. CO differs by about
16 per cent in the median active band, while CO2 differs by about 4 per cent in
the median and less than 21 per cent at its largest active bin. At 2.84 microns
ROBERT's H2O column optical depth is 0.056 dex lower. At 4.18 microns the CO2
difference is only -0.010 dex, so the large single-bin retrieval residual there
cannot be interpreted as a simple CO2 opacity-scale error. Within-bin line
distribution, random-overlap/correlated-k treatment, and PICASO sampling are
the more plausible attribution.

Large log ratios remain in CO and CO2 windows where both species are
effectively transparent. They are real database-floor differences but have no
meaningful leverage on the spectrum or retrieved abundance and are excluded
from the active-band summary.

## Comparison with Barstow et al. (2020)

Barstow et al., *A comparison of exoplanet spectroscopic retrieval tools*,
MNRAS 493, 4884 ([doi:10.1093/mnras/staa548](https://doi.org/10.1093/mnras/staa548)),
compared NEMESIS, TauREx, and CHIMERA transmission spectra. Their realistic
multi-gas, CIA, and cloudy forward models generally agreed at the level of a
few tens of ppm. They explicitly attributed remaining differences primarily
to opacity line data and tabulation, and ran cross-retrievals with 30, 60, and
100 ppm uncertainty envelopes.

The ROBERT--PICASO full-native 38.84 ppm RMS difference lies in that same
published regime. The earlier retrieval's cloud top was unbiased and its main
abundance shift was only a few posterior standard deviations under an Asimov
20--35 ppm error model—more demanding than most of the Barstow et al. tiers.
It would be disproportionate to keep modifying a validated solver to force
agreement between independent opacity ecosystems at this level.

## Closure

The relevant conclusion is not that PICASO or ROBERT is the truth. It is that:

1. matched-opacity transmission solver tests already agree at a few ppm;
2. independent active molecular bands agree at roughly 0.02--0.07 dex;
3. the total full-native cross-code spectrum differs by about 39 ppm, consistent
   with the scale of a published three-code comparison; and
4. the remaining difference is dominated by legitimate opacity/tabulation and
   sampling choices rather than a demonstrated physics defect.

No additional transmission diagnostic is scheduled. Future work should be
driven by real science analyses or a separately scoped inter-code emission
study, not by pursuing zero residual against another framework.

## Artifacts

- `examples/outputs/picaso_jwst_molecular_opacity_discrepancy/molecular_opacity_discrepancy.json`
- `examples/outputs/picaso_jwst_molecular_opacity_discrepancy/molecular_opacity_discrepancy.png`
- `examples/outputs/picaso_jwst_molecular_opacity_discrepancy/molecular_opacity_discrepancy_compact.npz`
