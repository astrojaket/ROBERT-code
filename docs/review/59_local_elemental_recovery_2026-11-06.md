# Local fixes and elemental recovery — 6 November 2026

## Code changes

The Virga test helper now sets `gravity_model` only for transmission. Emission
fixtures no longer fail before their cloud assertions. The production schema
continues to reject an explicit transmission-only option in emission.

The experimental Python joint-geometry API now exposes the underlying
likelihood to the manifest through an explicit wrapper contract. Its manifest
records likelihood type, known scientific settings, nuisance roles, and
local/global parameter bindings. A changed likelihood must block checkpoint
resume. Built-in covariance matrices and PSD tolerances are fingerprinted,
HRS velocity-parameter roles are retained, and prepared observations keep
their data identity. Custom covariance operators fail safely rather than
using an incomplete identity. The standard YAML runner still selects one
geometry.

Run creation also preserves the explicit null legacy metallicity/C/O
parameter names used by independent FastChem elemental configurations. The
round-trip test covers both top-level and regional chemistry.

These fixes do not complete joint posterior/profile collection or a physical
joint emission/transmission retrieval. See the
[joint API tutorial](../tutorials/04_joint_emission_transmission.md).

## Recovery protocol

This is one cloud-free, fixed-temperature transmission injection. It tests the
configured independent FastChem elemental parameters with full equilibrium
composition. It is not a fit to WASP-178b observations.

| Setting | Value |
| --- | --- |
| Temperature | Fixed, isothermal, 1800 K |
| Pressure | 80 layers from 100 to 1e-5 bar |
| Data | 48 bins from 2.8 to 5.15 microns |
| Uncertainty | Independent Gaussian, 20 ppm per bin |
| Molecular opacity | Native R=1000 H2O, CO, SiO KTA inputs, all 20 g ordinates |
| Gas combination | Random overlap |
| Geometry | Spherical transmission, inverse-square gravity, reference pressure 1 bar |
| Sampler | Separate PyMultiNest runs with 50 and 100 live points, four MPI ranks each |
| Thread limit | One numerical thread per rank |
| Evidence tolerance | dlogz = 0.5 |
| Iteration cap | None (`multinest_max_iterations: 0`) |
| Seed | 20261104, retained for comparison with the earlier test |

| Parameter | Injected value | Uniform prior bounds |
| --- | --- | --- |
| log10(n_O/n_H) | -3.18 | [-3.8, -2.8] |
| log10(n_C/n_H) | -3.70 | [-4.3, -3.3] |
| log10(n_Si/n_H) | -4.28 | [-4.9, -3.9] |

Hydrogen, helium, and the unmapped elements retain the input solar abundance
convention. The temperature, radii, gravity inputs, and cloud-free model are
fixed. The test does not measure degeneracy with a fitted temperature profile.

The source tables are the existing shared native-wavelength subsets prepared
on 4 November. Each file is 30,368,240 bytes and has 639 wavelength points,
22 pressure nodes, 27 temperature nodes, and 20 g ordinates. They retain exact
native coefficients, pressure/temperature grids, and quadrature weights.
They are not cross-section HDF files. No opacity file is copied into this run
or committed to Git.

The new injection evaluates the current model at the stated truth and adds
the exact saved noise vector. The noise SHA-256 is
`c22bf6d69bba31361251a242430a517662cba507effc5a7f76077156d0eea5d5`.
The compact result record retains the source hashes and code revision.

The audit initially missed the completed 4 November fixed-temperature
recovery. This new run is a fresh code/regression check with the same noise;
it is not an independent noise realization or a posterior calibration
ensemble. Existing checkpoints and earlier results remain unchanged.

## Interpretation and products

H-minus is configured, but its bound-free contribution is outside this band
and the implemented free-free policy is zero below 2500 K. This test therefore
checks elemental retrieval and the configured full-composition path; it does
not validate a nonzero H-minus signal.

Use the full weighted posterior for elemental interval summaries. Use the
same 100 weighted draws for spectra and atmospheric profiles. Save the draw
indices, parameter vectors, numerical products, and quantile probabilities.
The representative spectrum is the pointwise posterior median, with central
68.2689% and 95.45% intervals. Figures use the WASP-15b style and external
captions.

Full arrays, chains, configurations, driver scripts, and PNG/PDF figures stay
in the isolated external run. Git retains only the compact result record and
this description, in accordance with the [data policy](../data_policy.md).

## Measured results

Both runs converged without an iteration cap. The 50-live-point run used
`cef1216c7b7beb0177671edb16acdfecea0ec76b`. The requested 100-live-point
repeat used `c211dcab7e748f9581ce6ff8bdcb21506fe7d267`, whose source change
only updates plot legend text. Each imported the recorded clean checkout.
Their injected arrays, truth, noise, priors, and physical settings are exactly
equal. Both forward truths also match the 4 November full-state result.

The [compact result record](../data/elemental_injection_recovery_20261106.json)
contains input and product hashes, code revisions, intervals, and checks.

| Diagnostic | 50 live points | 100 live points |
| --- | --- | --- |
| Converged | Yes | Yes |
| Native likelihood evaluations | 560 | 937 |
| Weighted posterior samples | 264 | 571 |
| Posterior effective sample size | 156.28 | 350.20 |
| Monitored MPI job time | 30.25 s | 35.30 s |
| Peak monitored combined RSS | 1218.23 MiB | 1487.97 MiB |
| Reported log evidence | 449.0631 ± 0.0635 | 449.1304 ± 0.0748 |
| Posterior-median prediction chi-squared / 45 | 0.9569 | 0.9627 |

The evidence values are PyMultiNest's reported global-evidence fields and
match the importance-nested-sampling estimates in the native logs. The timing
includes MPI startup and import costs. It excludes opacity preparation and
posterior plotting. These small-case timings are not a scaling benchmark.
The chi-squared diagnostic uses the pointwise posterior median of the saved
100 draws, with 48 independent bins and three fitted parameters.

The 100-live-point elemental results are:

| log10 number ratio | Truth | Posterior median | 1-sigma interval | 2-sigma interval |
| --- | --- | --- | --- | --- |
| O/H | -3.180 | -3.194 | [-3.274, -3.062] | [-3.310, -3.026] |
| C/H | -3.700 | -3.747 | [-4.098, -3.413] | [-4.271, -3.322] |
| Si/H | -4.280 | -4.275 | [-4.616, -4.010] | [-4.849, -3.916] |

All three truths fall inside both intervals in both runs. C/H and Si/H remain
broad: their 100-live-point 2-sigma widths span 94.9% and 93.3% of their full
one-dex priors. O/H spans 28.5%. The increased live-point count changes the
sampled C/H shape and median; this does not establish precise C/H or Si/H
identifiability. Neither run is an independent noise realization.

The saved vectors match their selected posterior rows. Recomputed quantiles
match the saved spectra, temperature, VMR, and mean-molecular-weight products
exactly. Full-composition VMR sums differ from one by less than 3.2e-12 over
the saved draws. The 50-live-point samples and weights also reproduce the
earlier fixed-seed result exactly in this environment.

Both PNG/PDF figure pairs passed visual checks. Legends use 1-sigma and
2-sigma; numerical probabilities remain saved. The reusable WASP-15b helper
and ROBERT's general posterior plot legends use the same notation.

## Software checks

- Final code suite: 1089 passed, 40 optional/external checks skipped.
- Coverage: 73%, above the 70% floor.
- Ruff, the two-file mypy gate, and sdist/wheel builds passed.
- After the legend-only update: all 13 post-processing tests and lint passed.

All Python checks and calculations used the `robert-exoplanets` environment.
The recovery used four Open MPI 5.0.10 ranks with one numerical thread each.
No Glamdring connection or job submission was used.
