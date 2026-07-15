# Independent PICASO-generated JWST transmission retrieval

## Decision

The first external end-to-end transmission retrieval does **not** justify
implementing cloud sub-layering next. Cloud sub-layering remains a documented,
gated future optimization. The dominant actionable discrepancy is the
molecular-opacity/spectral-treatment difference between the official PICASO
database and ROBERT's ExoMolOP correlated-k path.

## Benchmark contract

PICASO 3.2.2 independently generated a cloudy hot-Jupiter transmission
spectrum using its official R=15,000 opacity database. The physical atmosphere
has 48 layers from 10 to 1e-5 bar, an isothermal 1100 K profile, constant H2O,
CO, CO2, and CH4 abundances, and the shared finite deck plus power-law haze.
The planet radius is anchored at 10 bar in both codes.

The spectrum was averaged into 72 contiguous JWST-like bins covering
0.8--12 microns. Representative uncertainties are 25 ppm for NIRISS/SOSS,
20 ppm for NIRSpec/G395H, and 35 ppm for MIRI/LRS. The benchmark is an Asimov
data set: the PICASO model mean is used without a random noise realization.
Consequently, recovered shifts and residuals measure cross-code model bias
rather than noise luck.

ROBERT was not changed for this benchmark. It retrieved four molecular
abundances, the 10-bar radius scale, cloud-top pressure, integrated deck
optical depth, haze mass extinction, and haze slope using MultiNest with 50
live points on two MPI processes.

## Result

At the identical injected parameters, ROBERT differs from PICASO by 42.51 ppm
RMS and 111.29 ppm maximum, corresponding to chi-square 242.52 over 72 bins.
The converged retrieval improves the residual RMS to 33.45 ppm, but its best
fit remains statistically inadequate: chi-square 131.59 for 63 degrees of
freedom, reduced chi-square 2.089, with a survival probability of 9.4e-7.

The mismatch is concentrated in molecular regions. NIRSpec/G395H contributes
85.17 of the total chi-square, compared with 17.27 from NIRISS/SOSS and 29.16
from MIRI/LRS. The largest residuals occur at 2.84 microns (+5.20 sigma),
4.18 microns (+4.63 sigma), and 6.52 microns (-3.25 sigma).

The posterior-mean shifts relative to their posterior standard deviations are:

| Parameter | Posterior mean | Shift |
|---|---:|---:|
| log10 H2O | -3.030 | -0.37 sigma |
| log10 CO | -3.319 | +1.00 sigma |
| log10 CO2 | -4.474 | +2.55 sigma |
| log10 CH4 | -5.402 | -0.06 sigma |
| 10-bar radius scale | 0.99926 | -1.11 sigma |
| log10 cloud-top pressure | -3.008 | -0.06 sigma |
| log10 deck optical depth | -0.470 | +0.40 sigma |
| log10 haze mass extinction | -2.931 | +0.47 sigma |
| haze slope | -3.480 | +0.52 sigma |

CO2 is the clear outlier. In contrast, the recovered cloud top is effectively
unbiased and every aerosol posterior mean lies within 0.53 sigma of the shared
truth. The maximum-likelihood haze slope moves farther, but its posterior is
broad and non-Gaussian; it is not evidence for a localized cloud-boundary
error.

## Interpretation and closing diagnostic

This is a useful failure rather than a reason to add geometry complexity. The
reference-pressure convention is controlled, the cloud-top pressure recovers,
and the structured residuals follow molecular bands. The retrieval used
PICASO's stride-5 opacity option for speed; it was therefore an exploratory
attribution test rather than the final native-resolution comparison.

Review 45 completes the one requested closing diagnostic using every native
PICASO and ExoMolOP sample on this exact atmosphere. No further transmission
diagnostic is recommended. The existing continuous cloud-boundary benchmark
remains the acceptance test if future science retrievals independently isolate
a material partial-shell aerosol bias.

## Reproduction

Run from the `robert-exoplanets` conda environment:

```bash
python examples/picaso_jwst_transmission_retrieval.py \
  --config configurations/picaso_jwst_transmission_retrieval_multinest.yaml

mpiexec -n 2 python run_retrieval.py \
  --config configurations/picaso_jwst_transmission_retrieval_multinest.yaml

python examples/picaso_jwst_transmission_retrieval.py \
  --config configurations/picaso_jwst_transmission_retrieval_multinest.yaml \
  --evaluate-result
```

Generated artifacts are under
`examples/outputs/picaso_jwst_transmission_retrieval/`, including the external
PICASO contract and spectrum, the ROBERT result, the quantitative benchmark
report, and posterior/fit plots.
