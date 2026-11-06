# Elemental abundance sensitivity — 6 November 2026

## Status and scope

This local study is complete for its stated model. It tests how O/H, C/H, and
Si/H affect one synthetic transmission spectrum and checks six changed-truth
retrievals. It does not validate abundance coverage, a full opacity model, or
constraints for a planet observed with real data. The [compact record](../data/elemental_abundance_sensitivity_20261106.json)
contains full weighted intervals, run statistics, and input and product hashes.

The study extends the [baseline elemental recovery](59_local_elemental_recovery_2026-11-06.md)
and the [local Virga recovery](60_local_virga_recovery_2026-11-06.md).
The isolated output directory is
`/Users/jaketaylor/.codex/visualizations/2026/11/06/01a1b0d4-f26e-7571-83c0-e1a6be3d8b9a/elemental-identifiability-20261106`.
Large arrays, chains, and figures remain there, outside Git.

## Fixed model and data

| Setting | Value |
| --- | --- |
| Observation | 48 transmission bins with centers from 2.8 to 5.15 microns; saved bin edges are retained |
| Noise | One saved, fixed noise vector; independent 20 ppm uncertainty per bin |
| Temperature and pressure | Isothermal 1800 K; 80 layers from 100 to 1e-5 bar |
| Chemistry | Full FastChem equilibrium composition |
| Cloud model | Cloud-free |
| Opacity | Native R=1000 correlated-k KTA tables for H2O, CO, and SiO; all 20 g ordinates |
| Reference truth | log10(O/H)=-3.18, log10(C/H)=-3.70, log10(Si/H)=-4.28 |
| Uniform priors | O/H [-3.8,-2.8]; C/H [-4.3,-3.3]; Si/H [-4.9,-3.9] dex |

The opacity inputs are the shared native tables. The study did not copy or
modify them. The fixed noise has SHA-256
`c22bf6d69bba31361251a242430a517662cba507effc5a7f76077156d0eea5d5`.
The altered study and six changed-truth retrievals used clean source commit
`f8318defe95491e9bc50ddbd538e363327147ca9`. The reference 100-live-point
retrieval used `c211dcab7e748f9581ce6ff8bdcb21506fe7d267`.

## Spectral sensitivity and finite-grid profiles

The one-at-a-time spectra change one elemental abundance while holding the
other two at the reference truth. O/H makes the largest spectral change over
the tested range. C/H also changes the spectrum. The Si/H curves remain near
the 20 ppm per-bin reference scale. These are fixed-nuisance spectral changes;
they are not marginalized constraints.

The joint diagnostic stores spectra for 9,261 states on a 21 by 21 by 21
grid. It profiles the Gaussian likelihood over the other parameters for the
noiseless reference spectrum, the saved noisy spectrum, and alternate
low/mid/high abundance injections. In this grid, low and high mean prior
endpoints. The grid uses nominal 0.05 dex spacing, with the reference truth
inserted, so some neighboring steps are not uniform. The baseline O/H and
C/H posterior correlation is 0.9446 in the saved 100-live-point result.

The one-parameter 1-sigma and 2-sigma profile thresholds are delta-chi-square
1 and 4. The two-parameter thresholds are 2.30 and 6.18. These are approximate
finite-grid contours, not Bayesian intervals or calibrated confidence
limits. The whitened finite-difference Jacobian has a largest singular value
of about 109 per dex at the reference state. A 0.05 dex step can cross the
narrow O/C ridge. Coarse nuisance sampling then creates spikes or disconnected
segments in some C/H and Si/H profiles. The figures keep these disconnected
regions visible. Do not report grid crossings as precise constraints.

## Native changed-truth retrievals

Six native retrievals changed one injected truth at a time. The changed value
is 0.10 dex inside its prior edge. The other two truths remain at the
reference values. These interior truths differ from the endpoint values in
the finite-grid profiles.

| Case | log10(O/H) truth | log10(C/H) truth | log10(Si/H) truth |
| --- | ---: | ---: | ---: |
| O-low | -3.70 | -3.70 | -4.28 |
| O-high | -2.90 | -3.70 | -4.28 |
| C-low | -3.18 | -4.20 | -4.28 |
| C-high | -3.18 | -3.40 | -4.28 |
| Si-low | -3.18 | -3.70 | -4.80 |
| Si-high | -3.18 | -3.70 | -4.00 |

Each run used 100 live points, four MPI ranks, one numerical thread per rank,
`dlogz=0.5`, and no iteration cap. All six fits converged. All runs reused the
same noise vector. Therefore the cases compare compositions under one noise
realization; they do not measure coverage over noise realizations.

Intervals below use all native posterior weights and the exact central
1-sigma and 2-sigma quantile probabilities. Values are dex and rounded to
four decimals. The JSON file retains the full precision.

| Case | Ratio | Truth | Median | 1-sigma interval | 2-sigma interval |
| --- | --- | ---: | ---: | --- | --- |
| O-low | O/H | -3.7000 | -3.6528 | [-3.6986, -3.6024] | [-3.7411, -3.5656] |
| O-low | C/H | -3.7000 | -3.4984 | [-3.6127, -3.3751] | [-3.7206, -3.3153] |
| O-low | Si/H | -4.2800 | -4.0618 | [-4.1994, -3.9556] | [-4.3422, -3.9136] |
| O-high | O/H | -2.9000 | -2.9011 | [-2.9627, -2.8443] | [-2.9904, -2.8093] |
| O-high | C/H | -3.7000 | -3.7098 | [-4.0971, -3.4397] | [-4.2612, -3.3461] |
| O-high | Si/H | -4.2800 | -4.2359 | [-4.5840, -4.0090] | [-4.8391, -3.9132] |
| C-low | O/H | -3.1800 | -3.1135 | [-3.1672, -3.0318] | [-3.2175, -2.9874] |
| C-low | C/H | -4.2000 | -3.7554 | [-4.0775, -3.5017] | [-4.2599, -3.3606] |
| C-low | Si/H | -4.2800 | -4.2563 | [-4.6169, -4.0136] | [-4.8529, -3.9150] |
| C-high | O/H | -3.1800 | -3.2502 | [-3.3303, -3.1827] | [-3.4372, -3.1252] |
| C-high | C/H | -3.4000 | -3.5248 | [-3.6908, -3.3968] | [-4.0156, -3.3185] |
| C-high | Si/H | -4.2800 | -4.2445 | [-4.5396, -4.0336] | [-4.7713, -3.9326] |
| Si-low | O/H | -3.1800 | -3.1728 | [-3.2517, -3.0768] | [-3.2861, -3.0190] |
| Si-low | C/H | -3.7000 | -3.6927 | [-4.0482, -3.4474] | [-4.2210, -3.3255] |
| Si-low | Si/H | -4.8000 | -4.5290 | [-4.7536, -4.2909] | [-4.8808, -4.0416] |
| Si-high | O/H | -3.1800 | -3.1783 | [-3.2698, -3.0924] | [-3.3187, -3.0478] |
| Si-high | C/H | -3.7000 | -3.6179 | [-3.9404, -3.4127] | [-4.1913, -3.3182] |
| Si-high | Si/H | -4.0000 | -4.2023 | [-4.4604, -4.0018] | [-4.7877, -3.9240] |

All 18 truths fall inside their 2-sigma marginal intervals. Eight fall outside
their 1-sigma intervals: all three parameters in O-low; O/H and C/H in C-low;
O/H in C-high; and Si/H in Si-low and Si-high. The O-low truths are close to
their 2-sigma interval edges.

Across these six runs, the O/H 2-sigma interval width is 17.6–31.2% of its
1-dex prior. C/H spans 69.7–91.5% in five cases and 40.5% in O-low. Si/H
spans 83.9–93.8% in five cases and 42.9% in O-low. The narrower C/H and Si/H
intervals coincide with O-low, where C/O is one by number. One composition
and one noise vector cannot establish that this regime gives precise or
calibrated constraints.

The saved spectra and VMR products use 100 weighted posterior draws. The
postprocessing check passed for all six runs. It verified the full weighted
parameter quantiles, draw vectors, saved spectra, temperature, VMR, and
mean-molecular-weight products. The exact core quantile update changed only
the six new derived quantile summaries and the new comparison. It used saved
native samples and draws without new forward evaluations. It did not modify
the baseline products.

## Limits and next tests

This study fixes the temperature profile at 1800 K and has no clouds. Its
opacity set has H2O, CO, and SiO only. H-minus is configured but inactive in
this temperature and wavelength regime. FastChem also reports other species,
including C1H4, Fe, Mg, N2, and sulfur species, in parts of the atmosphere.
A maximum VMR alone does not show that a species changes the spectrum or the
retrieved abundances.

Next local tests should vary the temperature profile and cloud properties,
add justified opacity sources, and check a regime with a measured nonzero
H-minus signal. Repeat the recovery with independent noise vectors. Refine the
O/C grid around its ridge and compare its profiles with converged native
retrievals. Until then, treat O/H as the strongest sensitivity in this setup;
C/H and Si/H are usually weakly constrained by these priors, with a clear
O-low regime dependence. Do not interpret these results as constraints beyond
this fixed-temperature, limited-opacity model.

## Evidence and verification

The compact [machine-readable record](../data/elemental_abundance_sensitivity_20261106.json)
contains source commits, fixed truths and priors, exact quantile probabilities,
all weighted parameter summaries, run counts and resources, and input/product
hashes. The external checks found all six native retrievals converged and all
product checks passed. The saved baseline products remain unchanged.
