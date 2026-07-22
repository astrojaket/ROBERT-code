# L 98-59 b flat-spectrum CLR ensemble

## Decision

The requested 100+100, 50-live-point MultiNest science look does **not**
support the hypothesis that a flat 3--5 micron spectrum generically drives the
CLR posterior to the heaviest available absorber. SO2 is not constrained in
any of 100 five-gas runs, and CO2 is not constrained in any of the paired 100
four-gas runs after SO2 is removed. The result is unchanged under the
preregistered threshold sensitivity checks.

These are prior/likelihood behavior diagnostics, not molecular detections or
science-grade evidence calculations. Representative closure-swap and
higher-live-point cluster runs are still warranted before drawing a general
conclusion about CLR parameterizations.

## Target and audit corrections

The request called the target “L98-89b.” The repository module, published
spectrum, Bello-Arufe et al. (2025) paper, and Zenodo record all identify the
planet as **L 98-59 b**, which is the target used here.

The pre-run audit found two provenance/runtime issues and repaired them with
tests:

- the committed Eureka spectrum differed from the upstream byte stream only by
  a terminal blank line, so verification now uses a newline-canonicalized
  scientific-content checksum while retaining the exact upstream checksum;
- the loader's paper DOI was corrected to `10.3847/2041-8213/adaf22`;
- the native MultiNest Fortran RNG rejects large integer seeds, so ROBERT now
  validates its supported range before launching the sampler.

The optional phantom species from commit `9415c18` was not used. H2S is the
opacity-bearing physical category recovered from unit-sum closure in both
ensembles. It contributes line opacity and its 34.0809 amu molecular mass. The
internal `background_species` name means mathematical remainder here, not an
assumed physical background. The POSEIDON-parity CLR transform is symmetric,
so the omitted coordinate has no special prior status.

## Reproducible contract

- Source: 218-point Eureka! L 98-59 b spectrum from Zenodo
  `10.5281/zenodo.14676143`, checksum verified.
- Flat depth: median 619.782 ppm.
- Per-bin uncertainty: median quoted error 37.166 ppm.
- Grid: 51 equal-log-width bins with exact outer edges at 3 and 5 microns;
  geometric-mean centers; exact center/width resolving power 99.8379573.
- Noise: 100 independent Gaussian realizations from NumPy PCG64 streams
  spawned from root `SeedSequence(20260719)`.
- A gases: H2O, CO2, CO, H2S, SO2.
- B gases: H2O, CO2, CO, H2S, using the exact same 100 NPZ spectra.
- Atmosphere: cloud-free, isothermal 100--800 K, one-bar reference-radius scale
  0.85--1.15, no conventional background, no phantom molecule.
- Opacity: real ExoMol R1000 correlated-k tables for every configured gas.
- Sampler: native MultiNest 3.10 through PyMultiNest 2.12, 50 live points,
  three MPI ranks, `dlogz=0.5`; A seeds 19000--19099 and B seeds 20000--20099.

The primary constraint definition was fixed before inspecting results: a gas
is “constrained” when its one-sided 95% posterior VMR lower bound exceeds 1%.
Sensitivity thresholds were 0.1%, 10%, and 50%. A separate interval-width
check asks whether the central 95% log10 VMR interval is narrower than two or
four dex.

## Ensemble result

| Diagnostic | A: SO2 available | B: SO2 removed |
|---|---:|---:|
| Completed / converged | 100 / 100 | 100 / 100 |
| Target gas | SO2 | CO2 |
| Primary constrained frequency | 0/100 | 0/100 |
| One-sided 95% upper limit on frequency after 0/100 | 2.95% | 2.95% |
| Constrained at 0.1%, 10%, or 50% threshold | 0/100 at each | 0/100 at each |
| Central 95% width below 2 or 4 dex | 0/100 at each | 0/100 at each |
| Median central 95% width | 11.62 dex | 11.60 dex |
| Target posterior median above 1% | 14/100 | 10/100 |
| Target median VMR across runs (16/50/84%) | 1.36e-7 / 3.87e-5 / 4.20e-3 | 4.17e-8 / 6.96e-6 / 2.70e-3 |
| Median reduced chi-square | 1.057 | 1.029 |
| Median ln Z | 441.552 | 442.459 |
| Median stored ln Z error | 0.045 | 0.049 |

The targets are simultaneously near both composition boundaries in most runs:
60/100 target posteriors approach the `log10(VMR)=-12` floor at their lower
side, while 97/100 have a 97.5th percentile above 95% VMR. This boundary-
spanning behavior explains why a target can sometimes have the largest
posterior median while never supplying a defensible lower-bound constraint.

## Abundance ranks and prior volume

| Median-abundance leader | A count | B count |
|---|---:|---:|
| H2O | 26 | 32 |
| CO2 | 12 | 12 |
| CO | 13 | 13 |
| H2S closure category | 29 | 43 |
| SO2 | 20 | -- |

A 50,000-draw reference from the exact configured CLR prior gives SO2 a 19.9%
chance of being most abundant in A and CO2 a 25.1% chance in B. SO2's observed
20/100 rank frequency is indistinguishable from symmetric prior corner volume;
CO2's 12/100 is lower than its prior expectation. The flat-spectrum likelihood
therefore does not single out the heaviest absorber. Instead, it leaves broad,
multimodal corner solutions, with H2O and H2S more commonly carrying the
largest posterior median in these configurations.

## MMW, temperature, radius, and evidence

The median of the per-run posterior-median MMW is 34.10 amu in A and 33.03 amu
in B. The paired MMW difference A minus B has median 3.94 amu and a 16--84%
range of 0.11--9.57 amu. The corresponding CLR-prior median MMW values are
34.08 and 30.87 amu, showing that much of the ensemble shift is already
available from composition prior volume rather than a target-gas constraint.

Temperature and reference radius carry the clearest degeneracy. Across runs,
the median posterior temperatures are 273 K (A) and 255 K (B), while the median
95% intervals span approximately 108--678 K and 109--640 K. Median radius
scales are 0.9594 and 0.9601, with median 95% intervals of 0.9320--0.9707 and
0.9303--0.9710. The median weighted temperature-radius correlations are -0.803
and -0.836. MMW-radius correlations are weaker (+0.209 and +0.300), and
MMW-temperature correlations are close to zero (+0.061 and -0.066).

The paired ln Z difference A minus B has median -0.949, a 16--84% range of
-1.276 to -0.616, and favors A in only 2/100 pairs. At this exploratory
resolution, the extra SO2 dimension receives a modest Occam penalty without
producing a constrained abundance. The low live-point count and approximately
0.05 stored evidence errors prohibit a final model-selection claim.

## Outputs and next validation

Tracked reproduction material is under `studies/l98_59b_flat_clr/`. Ignored run
artifacts under its `outputs/` directory include the generated-data manifest,
all 100 spectra, 200 resolved configs, opacity cache, per-run logs/manifests/
chains/results, compact CSV and JSON summaries, selected-fit plots, and final
ensemble figures.

The next targeted cluster validation should rerun a small set of representative
and boundary-extreme realizations with at least several hundred live points.
It should also swap the omitted CLR coordinate between H2S and another
non-target gas. Symmetric-prior and exact-forward-model invariance are expected;
any material change beyond nested-sampling uncertainty would identify a
coordinate/sampler pathology rather than atmospheric evidence.
