# WASP-77Ab real HRS/LRS comparison and retrieval protocol

Date: 2026-08-31

Status: the source files, deterministic HRS/LRS checks, real-grid LBL sampling
check, fixed-template PyMultiNest reference measurements, and operator-level
ROBERT injection validation are complete for the documented scope. Full
real-data shared-atmosphere ROBERT PyMultiNest retrievals are deferred to a
cluster because the laptop scope is complete. This document does not claim a
literature reproduction or real-data posterior recovery.

This protocol uses Smith et al. (2024) as the real high-resolution (HRS) and
low-resolution (LRS) comparison case. It keeps the public August et al.
(2023) NIRSpec spectrum separate from the Smith-produced IGRINS products.
The implementation uses one shared ROBERT VMR atmosphere for future physical
fits, with a fixed-template path available for deterministic validation.

## 1. Verified sources and data roles

The primary paper is [Smith et al. (2024), arXiv:2312.13069](https://arxiv.org/abs/2312.13069),
with [DOI 10.3847/1538-3881/ad17bf](https://doi.org/10.3847/1538-3881/ad17bf).
The supplementary files are from [Zenodo record 10382053](https://zenodo.org/records/10382053),
which has [DOI 10.5281/zenodo.10382053](https://doi.org/10.5281/zenodo.10382053)
and a CC-BY-4.0 record licence. The checked-in source record is
[`docs/data/wasp77ab_smith2024_sources.json`](../data/wasp77ab_smith2024_sources.json).
The acquisition command is:

```text
conda run -n robert-exoplanets python examples/fetch_wasp77ab_smith2024_data.py \
  --output-dir external_data/wasp77ab_smith2024
```

The acquisition helper verifies official Zenodo file sizes and MD5 values,
computes SHA-256 values, uses a bounded stream, and never writes into Git.
The seven primary files are present and verified:

| File | Role | SHA-256 |
| --- | --- | --- |
| `cube14v3.pic` | 2020-12-14 pre-eclipse IGRINS reduced cube | `999623a7ba4460aa72ba9bbe284de92e7b5a811ebe09e51d70667084ae592b26` |
| `20201214_info.csv` | pre-eclipse frame and observing-condition metadata | `68f5a80f930e5f1f99cd1861678754f824bf665972d675f617a7d66c4fdad11e` |
| `w77_1DRC_FULL.txt` | full HRS atmospheric template | `8f3fcdf11942e1c1db5b5aba61c8266bad65cf3c279aa237e55346ed9be7a14` |
| `w77_1DRC_H2O_ONLY.txt` | H2O-only HRS template | `f68acda835e0af3e1747504d1e6243560610e5b98ad675917233859a17ace330` |
| `w77_1DRC_CO_ONLY.txt` | CO-only HRS template | `b8c9fc1496b25d9b4030ac3fafac3623ac1a5e945cf0b2f3161e96d3e8e3c016` |
| `w77_pre_nirspec_best_fit_R500K_scaled.txt` | R=500,000 pre-eclipse template for NIRSpec comparison | `fe1fb7cd37b6dec52173a19ee1fd7ea1529c421946c76d728d0bdcf4285ad34d` |
| `NIRSpec_posterior_draws_median_spectrum.txt` | Smith NIRSpec posterior-median spectrum | `45858658108e62333226e16e13809f304666672803ffcdf089dabe3514b01e5a` |

The optional 2020-12-06 and 2020-12-21 cubes and their metadata are listed in
the source record but are not required for the primary pre-eclipse check.
They must be acquired before a three-night result is claimed.

The LRS comparison input is the [August et al. (2023) JWST/NIRSpec
paper](https://arxiv.org/abs/2305.07753), [DOI
10.3847/2041-8213/ace828](https://doi.org/10.3847/2041-8213/ace828), from the
[MAST archive DOI 10.17909/3fmp-zj55](https://doi.org/10.17909/3fmp-zj55).
The local Firefly table has SHA-256
`96159824870eb7c8132fd9c8bee30caf9ec95f05e128dbe2fea8627909a40d3f` and 150
rows. It returns two named detector data sets:

| Data set | Rows | Published wavelength range | Role |
| --- | ---: | --- | --- |
| `nirspec_g395h_nrs1` | 70 | 2.811--3.709 micron | August NRS1 LRS comparison input |
| `nirspec_g395h_nrs2` | 80 | 3.835--5.164 micron | August NRS2 LRS comparison input |

The outer bin edges are 2.808--3.712 micron and 3.831--5.168 micron. The
0.119 micron detector gap is retained. Published asymmetric errors are kept
in metadata; the deterministic Gaussian check uses their mean absolute value.
Smith uses `N=160` in one predictive paragraph. The public 150-row table plus
the public Smith NIRSpec template gives the reported fit statistic, so this
document uses 150 as the measured data count and records 160 as a manuscript
count discrepancy.

The stellar PHOENIX provenance is separate from both target-data sources. The
official STScI base URL is
[`https://ssb.stsci.edu/cdbs/grid/phoenix/`](https://ssb.stsci.edu/cdbs/grid/phoenix/).
The maintained minimal subset manifest is
[`wasp77ab_phoenix_sources.json`](../data/wasp77ab_phoenix_sources.json). It
pins the following files under `PYSYN_CDBS/grid/phoenix`; the manifest is an
official identity record and does not claim that the files are downloaded on
every machine.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `catalog.fits` | 1,497,600 | `c5cb54e12ca20f8cba38624c460b743d0783ab92fcc0ad53b8288819d1d4d471` |
| `phoenixm00/phoenixm00_5600.fits` | 9,901,440 | `688a5acf69c9ec65a1b5398084d6c62e43ad63d2799588e5a005f715123686ca` |
| `phoenixm00/phoenixm00_5700.fits` | 9,901,440 | `373a945f3c16f36473fdca542d5a97c090a16871ad598f7b069fd15bb4bbe027` |

PHOENIX mode in the ROBERT runner accepts an explicit `PYSYN_CDBS` root and
fails closed on these three sizes and hashes before opacity preparation.
Explicit blackbody mode does not require this subset. The fixed-template
reference uses the supplied Smith template stellar flux; the PHOENIX subset is
provenance for the shared physical runner.

## 2. Target, orbit, and observing products

The cited target constants and the two ephemerides are in
[`examples/wasp77ab_target.py`](../../examples/wasp77ab_target.py). The
production circular orbit uses the newer Cortés-Zuleta values from Smith
Table 1: `T_C = 2457420.88439` BJD and `P = 1.36002854` day. The older Maxted
epoch and period remain metadata and must not be mixed into one velocity
calculation.

The HRS line-of-sight velocity is:

```text
V_LOS(t) = gamma + V_bary(t) + Kp sin(2 pi phase(t)) + V_sys
```

Positive velocity is redshift. A science run must calculate the barycentric
term from the exact BJD, target coordinates, observatory, ephemeris, and
software version. It must not use a rounded paper value.

Smith used Gemini-South/IGRINS at about R=45,000 over 1.45--2.6 micron. The
public cleaned products have shape `(order, frame, pixel) = (44, 79, 1848)`
for the primary 2020-12-14 sequence, with phases 0.3259--0.4707. The real
LBL sampling check selects 19 complete orders in 1.90--2.45 micron. This K
subset is a declared comparison product; it is not a claim that the original
Smith analysis used only those orders.

## 3. Smith HRS reduction and model method

The method is pinned to the paper and to the public products:

1. The IGRINS pipeline extracts raw one-dimensional spectra. Each frame is
   aligned to the last frame in its sequence to correct wavelength drift,
   counts are normalised, eight low-throughput or strongly contaminated orders
   are discarded, and 200 edge pixels are discarded from each retained order.
2. Smith applies `numpy.linalg.svd` separately to each order. The primary
   pre-eclipse analysis removes the first four principal components and saves
   both the residual cube and the four-component scaling matrix. The post-
   eclipse sensitivity analysis uses three components as its preferred case;
   PC counts 2, 3, 4, 6, and 8 are sensitivity cases.
3. The HRS thermal-emission template is generated at R=250,000, broadened by
   an equatorial Gray rotation kernel with `v sin(i) = 4.2 km/s`, and convolved
   with a Gaussian IGRINS line-spread function at R=45,000. The stellar
   spectrum is a smoothed PHOENIX spectrum in the Smith method. The Smith
   template opacity set includes H2O, CO isotopologues, H2--H2/H2--He CIA,
   and other gases; its exact line-list versions are part of the literature
   method and are not silently equated with the ROBERT pRT tables.
4. The total velocity contains the systemic, barycentric, orbital, and
   systematic terms above. Smith cross-correlates on a `Kp`--`Vsys` grid,
   median subtracts the map, estimates its scale from a 3-sigma-clipped map,
   and reports a CCF SNR.

The local ROBERT temporal adapter preserves the public cube, frame times,
order grids, masks, and fixed velocity terms. Its exact Smith-equivalent
likelihood does the following for every trial model:

- forms the model ratio and injects `(1 + Fp/Fs) * data_scale`;
- recomputes the same order-wise SVD and removes the configured components;
- subtracts the frame mean from the model residual, while the stored data
  residual is not re-centred;
- applies the whole-order 3-sigma clip, zero-filling clipped data values but
  retaining their model/norm accounting; and
- evaluates the scalar Brogi--Line-style quadratic stream, one order at a
  time in the bounded diagnostic.

The deterministic HRS validator uses the response order
`Gray rotation -> Gaussian R=45,000 LSF -> interpolation onto real cube
pixels`. It does not pixel-bin the model. The real-grid LBL benchmark also
tests the explicit pixel-bin response on the 19 real K-band orders.

## 4. ROBERT data model and covariance policy

`load_smith2024_wasp77ab_hrs` returns a
`TimeResolvedHighResolutionObservation` with wavelength shape `(order, pixel)`
and flux/mask shape `(order, frame, pixel)`, plus phase, BJD, fixed velocity,
airmass, humidity, and SNR metadata. The data model is independent of the
atmosphere model. It does not deserialize or invent an unpublished covariance
matrix.

The public reduced products do not provide a covariance matrix used by the
current adapter. The primary comparison therefore treats named HRS orders
and the two NIRSpec detector data sets as independent likelihood terms. This
is an explicit assumption. If cross-order or cross-detector covariance is
obtained, it must be represented by a block covariance/operator and tested
with the exact quadratic form; it must not be replaced silently by diagonal
errors. For any linear filter `F`, propagate `C_out = F C_in F.T` and carry
the retained residual rank into the likelihood degrees of freedom.

## 5. Deterministic validation status

The checks below use fixed public templates. They validate source loading,
response handling, binning, and likelihood plumbing. They are not atmosphere
retrievals and do not provide sampler evidence.

| Check | Result | Measured values |
| --- | --- | --- |
| HRS pre-eclipse validator | Pass | 44 orders x 79 frames x 1848 pixels; 4 PCs; 6,423,648 active samples; 40,527 clipped samples; peak RSS 630,226,944 bytes |
| HRS full/H2O/CO diagnostics | Pass | local raw summed CCF values 26.0578 / 22.2704 / 4.7053; paper's map-normalised anchors are 9.4 / 9.2 / 3.6 and are not claimed reproduced |
| NIRSpec LRS validator | Pass | total chi-square 108.7132; chi-square/N = 0.7247545; NRS1 = 0.7425956; NRS2 = 0.7091435; peak RSS 211,304,448 bytes |
| Real-grid LBL convergence | Pass | HRS selected stride 2; LRS selected stride 25; peak RSS 1,663,959,040 bytes; one CPU; opacity estimate below 1023 MiB |

The detailed records are [`wasp77ab_smith2024_hrs_validation_20260831.json`](../data/wasp77ab_smith2024_hrs_validation_20260831.json),
[`wasp77ab_smith2024_lrs_validation_20260831.json`](../data/wasp77ab_smith2024_lrs_validation_20260831.json),
and [`wasp77ab_lbl_sampling_20260831.json`](../data/wasp77ab_lbl_sampling_20260831.json).
Their current SHA-256 values are, respectively,
`3bfe5ccd29ac572652a619b15acbfac7a39c7087311e4290c004da06348cac92`,
`6d9cd93f3de1c753c80a3d9f251759a9302870e699d52039d08e0aa64efa9736`, and
`365b4cc05a9e84576f6d494d526bffc4856d0a314bc0ae902f0e8713b8864106`.

The HRS LBL metrics are RMS/max fractional errors of `4.0866e-5/3.1854e-4`
at stride 2, `1.46796e-4/8.8350e-4` at stride 4,
`3.14572e-4/1.83998e-3` at stride 6, and
`5.53084e-4/3.45619e-3` at stride 8. The LRS sigma RMS values for strides
25, 50, 100, and 250 are 0.0293389, 0.0473251, 0.0797095, and 0.160474.
The selected values are the finest passing values in the completed report.
The recorded run used `NUMEXPR_NUM_THREADS=3` and the other six numerical
thread values equal to 1; this legacy record used the maximum-three policy.

### Fixed-template PyMultiNest reference result

The fixed-template reference workflow is complete and passed its declared
two-seed evidence gate. The report is
[`wasp77ab_smith2024_reference_pymultinest.json`](../data/wasp77ab_smith2024_reference_pymultinest.json),
SHA-256
`7bdbf6df636ca1bb4cd9e01a87cc5898558e79142c0204305aea9368fd36f1d9`.
It used PyMultiNest with the MultiNest backend, seeds 24680 and 24681, 64
live points, `max_iter=0`, one MPI process, and the legacy maximum-three
thread cap. All observed numerical thread values were 1. Peak RSS was
1,261,125,632 bytes, below the 1.9-GiB configured limit.

| Mode | Status | Likelihood calls (seed 24680/24681) | ΔlnZ | Combined error |
| --- | --- | ---: | ---: | ---: |
| HRS-only | Pass | 1,207 / 1,230 | 0.36144 | 0.22754 |
| LRS-only | Pass | 534 / 547 | 0.06953 | 0.09903 |
| Joint | Pass | 1,835 / 2,126 | 0.28603 | 0.07406 |

The HRS-only posterior medians are `Kp = 192.286/192.276 km/s` and
`dVsys = -5.672/-5.703 km/s` for the two seeds. The joint medians are
`Kp = 192.260/192.270 km/s` and `dVsys = -5.676/-5.697 km/s`. The LRS-only
fit has NRS1/NRS2 chi-square values `53.5369/53.7935` and
`53.5364/53.7941` for the two seeds. The report records the full posterior
intervals and per-component statistics.

This result is a fixed-template comparison. It is not a VMR retrieval: it
does not infer H2O, CO, H-, or electron VMRs. It is also not an exact Smith
method reproduction because the ROBERT response, opacity, stellar, and
likelihood implementations are independent choices documented above. The
result is therefore a reproducible reference for the data and operators, not
a claim that ROBERT recovered the Smith atmospheric model.

### Operator-level ROBERT injection validation

The compact record
[`wasp77ab_robert_injection_20260831.json`](../data/wasp77ab_robert_injection_20260831.json)
has status `pass`. It is an operator-level injection validation on the real
HRS/LRS data structures. It is not a real-data posterior recovery and it does
not use a sampler. The report is 15,080 bytes and its SHA-256 is
`0e0efb30ba27b6a5fc0599ef6d5f8b839f13ae7693bd7a172a2a1436eed0afe9`.

The final record uses a structured 19-order HRS noise calibration and a fixed
truth mapping. HRS synthetic noise uses fixed Gaussian seed `20260832`, with
the scale set to each prepared order's residual RMS. LRS synthetic noise uses
fixed seed `20260831`. The joint case combines those same deterministic HRS
and LRS products. The truth-minus-null log-likelihood differences are:

| Injection mode | Truth minus null | Truth minus wrong-RV |
| --- | ---: | ---: |
| HRS-only | `+91.65259` | `+42.53990` |
| LRS-only | `+261859.82925` | not applicable |
| Joint | `+261951.48184` | `+42.53990` |

The H-minus on/off truth comparisons are finite and numerically zero or
negligible in these windows. This validates both finite execution paths only;
it is not an H-minus detection. The Gray/pRT bound-free (BF) opacity is zero
above the exact `1.6421 micron` cutoff in this comparison window. The
free-free (FF) path remains composition-controlled by H and e- VMR, so the
injection result does not identify H-minus or electron abundance.

The injection resources passed with six numerical threads, all seven thread
variables recorded, one MPI process, and peak RSS `1,754,120,192` bytes. This
is strictly below the configured `1.9 GiB` limit of `2,040,109,465` bytes and
below the hard `2 GiB` ceiling. The measured resource result does not approve
full real-data sampling on a laptop.

## 6. Shared ROBERT atmosphere and H-minus scope

The physical runner is [`run_wasp77ab_robert_joint_pymultinest.py`](../../examples/run_wasp77ab_robert_joint_pymultinest.py).
It loads the real HRS and LRS data, prepares H2O and CO tabulated line-by-line
cross sections from the external pRT R=10^6 files, and uses one shared
`AtmosphereBuilder` for both data sets. ROBERT composition is VMR-only:
`log10_H2O_VMR`, `log10_CO_VMR`, `log10_Hminus_VMR`, and
`log10_electron_VMR` are VMR parameters. No mass-fraction state or
independent continuum amplitude is allowed.

The runner includes Rayleigh scattering and the NemesisPy H2--H2/H2--He CIA
table. pRT mass fractions are allowed only in an external oracle boundary,
where they are deterministically derived from the ROBERT VMR vector and
molecular masses. They are not retrieval parameters.

H-minus uses the physical Gray/pRT bound-free and free-free convention with
explicit H-, H, and e- VMR profiles. The bound-free cutoff is exactly
1.6421 micron, so bound-free opacity is zero throughout the 1.90--5.17 micron
WASP-77Ab comparison window. The selected long-wave free-free branch is used
where valid. Free-free strength depends on H and e-; the pilot fixes neutral H
at VMR 0.1 to break that product degeneracy and retrieves electron VMR. The
H- VMR parameter is retained for the same shared state and for future shorter
wavelength LRS tests, but it is expected to be prior-dominated in this window.
No Saha closure is imposed.

The present ROBERT pilot is isothermal, with 32 pressure layers from 10^-5 to
100 bar and a bounded 2500--2900 K temperature prior. Smith's published
retrieval uses a six-parameter non-isothermal P--T profile (`T0`, `log P1`,
`log P2`, `log P3`, `alpha1`, `alpha2`). The isothermal pilot is therefore an
engineering and plumbing result. It must not be called a Smith atmospheric
reproduction until a bounded non-isothermal ROBERT model is implemented and
validated.

## 7. Sampler policy and cluster deferral

All sampler claims for this target require PyMultiNest with the MultiNest
backend. The common bounded laptop contract is:

- seeds 24680 and 24681;
- one MPI process;
- all seven numerical thread variables explicitly recorded;
- 64 live points, `max_iter=0`, evidence tolerance 0.5, and sampling
  efficiency 0.8 for the first proof run;
- process RSS strictly below 2 GiB and LBL opacity preparation strictly below
  1023 MiB; and
- one order-sized temporary at a time in the HRS likelihood evaluation.

The completed fixed-template reference record uses a maximum of three
numerical threads, with one observed thread in its measured run. New
shared-atmosphere ROBERT jobs are permitted a maximum of six numerical
threads, with MPI still fixed at one process. Six is a ceiling, not a
production choice: select the production thread count only after a
representative run at that setting measures peak RSS strictly below 1.9 GiB.
The laptop has no completed full real-data shared-atmosphere ROBERT sampler
record. The operator-level injection record above is the completed laptop
validation. Full HRS-only, LRS-only, and joint real-data retrievals are
deferred to a cluster. This is an explicit scope decision, not a sampler
pass:

```text
ROBERT_PYMULTINEST_MEASUREMENTS = DEFERRED_TO_CLUSTER
```


## 8. Current boundary

The deterministic source, HRS, LRS, real-grid, fixed-template, and
operator-level injection records are complete for their declared scope. The
fixed-template fits are comparison records, and the operator injection is a
plumbing check; neither is a real-data atmospheric posterior recovery.

The shared-VMR real-data sampler remains deferred to a cluster. A six-parameter
non-isothermal P--T model and any published covariance definition are required
before a literature reproduction can be claimed. The public reduced cube also
limits the HRS result to partial validation of the original observing analysis.

Smith's main joint anchors remain useful comparison values: log10(H2O VMR)
`-4.02 +0.07/-0.06`, log10(CO VMR) `-3.91 +/- 0.13`, `T0 = 1400 +30/-40 K`,
`C/O = 0.57 +/- 0.06`, `dKp = -1.29 +0.75/-0.78 km/s`, and
`dVsys = -5.27 +0.48/-0.47 km/s`. Use them as literature context, not as
truth values for the current isothermal pilot.

Until the cluster measurements and model upgrades are complete, the correct
claim is: “ROBERT was evaluated on the Smith et al. WASP-77Ab HRS/LRS
comparison data under the documented independent model and likelihood choices;
full real-data PyMultiNest retrievals were deferred to a cluster.”
