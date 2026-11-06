# Local Virga cloud recovery: 6 November 2026

**Result:** The two local retrieval pipelines ran to native convergence. The saved posterior products passed numerical checks. Emission contains all four injected parameters inside both the 1-sigma and 2-sigma intervals. Transmission misses three injected parameters at the 2-sigma level. Parameter-coverage validation remains open. These two injections do not establish general recovery performance.

## Scope and result

These are two separate synthetic self-injections. One fits thermal emission. One fits transmission. Each run generated and fitted its own noisy data, with independent noise seeds. They are not one joint emission/transmission retrieval, and they are not retrievals of physical observations.

The native MultiNest results report convergence. Each used 100 live points, four MPI workers, one numerical thread per worker, `dlogz=0.5`, and `max_iter=0` (no iteration cap). The result table reports weighted quantiles from all native posterior rows. The intervals are central intervals at the exact 1-sigma and 2-sigma probabilities saved in the JSON record.

| Geometry | Weighted rows | ESS | Likelihood calls | Timed fit wall time | Common-draw analysis |
|---|---:|---:|---:|---:|---:|
| Emission | 831 | 432.313 | 1,874 | 659.90 s | 100 draws, 368.69 s |
| Transmission | 900 | 400.494 | 3,182 | 1,025.77 s | 100 draws, 361.31 s |

The emission truths fall inside both intervals for all four fitted parameters. The transmission truths fall outside the 2-sigma interval for `T_upper`, `T_deep`, and `cloud_fsed`. `cloud_log10_kzz` falls inside the 2-sigma interval, but outside the 1-sigma interval.

| Geometry | Parameter | Truth | Weighted median | 1-sigma interval | 2-sigma interval | Truth in 1-sigma / 2-sigma |
|---|---|---:|---:|---:|---:|---|
| Emission | `T_upper` (K) | 850 | 839.522 | 825.099–851.080 | 814.889–862.480 | yes / yes |
| Emission | `T_deep` (K) | 1800 | 1809.115 | 1792.996–1827.306 | 1779.866–1838.517 | yes / yes |
| Emission | `cloud_fsed` | 1.5 | 2.032 | 1.268–3.149 | 0.949–3.747 | yes / yes |
| Emission | `cloud_log10_kzz` | 9.0 | 8.917 | 8.354–9.621 | 8.088–9.923 | yes / yes |
| Transmission | `T_upper` (K) | 850 | 795.995 | 765.309–826.688 | 739.238–846.115 | no / no |
| Transmission | `T_deep` (K) | 1800 | 1888.018 | 1844.505–1932.720 | 1814.195–1965.686 | no / no |
| Transmission | `cloud_fsed` | 1.5 | 2.841 | 2.163–3.566 | 1.669–3.941 | no / no |
| Transmission | `cloud_log10_kzz` | 9.0 | 8.593 | 8.427–8.790 | 8.340–9.007 | no / yes |

The representative spectral prediction is the pointwise median of the same 100 weighted posterior draws used for the saved spectral and profile products. There are 208 degrees of freedom: 212 bins minus four fitted parameters. Emission has `chi-square=194.7633`, or `chi-square/208=0.936362`. Transmission has `chi-square=216.8710`, or `chi-square/208=1.042649`. These are posterior-prediction checks. They are not maximum-likelihood chi-squared values.

The transmission coverage miss is real for this run. A separate three-call check found that the truth spectrum matches the saved noiseless injection bitwise. The saved centers, edges, flux, and uncertainty also match the current observation arrays. The maximum difference between `data - truth` and the saved noise is `1.72e-18`, from floating-point subtraction. The problem likelihood matches a direct normalized Gaussian calculation at truth, at the native maximum-likelihood point, and at the marginal posterior medians. At truth, `chi-square=219.2191`. At the maximum-likelihood point, `chi-square=216.7382`. Their difference is `delta chi-square=2.4809` for four parameters. The marginal-median point has `chi-square=220.9136`.

These checks rule out a mismatch in the saved truth spectrum, observation coordinates, noise construction, or likelihood calculation. They do not explain the shifted transmission marginals. This one noise draw does not measure coverage frequency, and the evidence does not identify a cause. Do not tune the data or priors to force this run to cover the truth.

## Forward model and injection protocol

Both runs use the same synthetic planet and star. The planet has mass `1.1502935e27 kg` and radius `8.0e7 m`. The derived gravity is a constant scalar, `11.9959436048 m/s2`, for gas optical depth and cloud calculations. Emission passes `self.gravity_m_s2` to gas optical-depth evaluation; see [`emission.py`](../../src/robert_exoplanets/forward/emission.py#L817). The helper that derives gravity from mass and radius is at [`emission.py`](../../src/robert_exoplanets/forward/emission.py#L1459). Transmission selects constant gravity in its configuration. Both Virga calls use uniform mean molecular weight, `2.3465386118 amu`.

The star is a `600000000 m` blackbody at `5000 K`. The atmosphere has 48 pressure layers from `1e-5` to `100 bar`. A spline temperature profile has two fitted controls:

| Control | Knot pressure | Truth | Uniform prior |
|---|---:|---:|---:|
| `T_upper` | `1e-6 bar` | `850 K` | `700–1200 K` |
| `T_deep` | `1000 bar` | `1800 K` | `1500–2200 K` |

The knots bracket the pressure grid. The profile is non-isothermal. The fitted cloud controls are:

| Control | Truth | Uniform prior |
|---|---:|---:|
| `cloud_fsed` | `1.5` | `0.5–4.0` |
| `cloud_log10_kzz` | `9.0` | `8.0–10.0` |

`Kzz` is in `cm2/s`. The gas VMRs are fixed: H2O `1e-3`, CO `1e-3`, CO2 `1e-5`, CH4 `1e-5`, NH3 `1e-6`, and HCN `1e-7`. The H2 and He background fractions are `0.8547` and `0.1453`. The saved posterior products contain eight gas species, including H2 and He. An independent check confirmed that their full VMR profiles normalize.

The cloud condensable is MgSiO3. Its bottom gas mass mixing ratio is `0.00275 g/g`; the metallicity is solar, the geometric particle-size width is `2`, and supersaturation is `0`. The input pressure-edge and temperature-edge grids give one Virga layer per ROBERT layer.

Each synthetic spectrum has 212 bins, with 213 geometric bin edges uniformly spaced in log wavelength from `0.6` to `5.0 micron`. Bin centers are the geometric means of adjacent edges. The target center-over-bin-width resolving power is `99.9872`. Each bin has independent Gaussian noise with `20 ppm` uncertainty. The noise seeds are `20261106` for emission and `20261107` for transmission.

The molecular opacity inputs are ExoMol NEMESIS KTA tables. The upstream `R1000` sources cover `0.3–50 micron`. ROBERT's current bundled tables are reduced to nominal coverage `0.3–15 micron`; their KTA headers give 391 wavelength points from `0.3015008` to `14.924962 micron`. Their measured effective resolution is `99.9479`, and the calculation retains **8 g ordinates**. These opacity settings are separate from the synthetic observation grid.

## Virga and Mie input

ROBERT's Virga adapter is an experimental path. It has not replaced the existing Mie cloud model; see [the Virga cloud scope and limits](56_virga_cloud_experiment.md).

The refractive-index input is glassy enstatite MgSiO3 from Dorschner et al. (1995). Its original 109 rows cover `0.2–500 micron`. The `.refrind` table has 574 log-spaced wavelengths from `0.4` to `7.0 micron`, at effective resolving power about `200.2`. The 35 source n/k points in this range were linearly interpolated in wavelength, with no extrapolation. The generated Virga 2.0.2/MiePython 3.3.1 `.mieff` file uses 60 log-spaced radius centers from `0.0001` to `542.39131 micron` and six sub-bin averages per radius. The observation edges lie inside the Mie wavelength coverage.

The Mie reader check passed. It found 574 increasing wavelengths, 60 increasing radii, and 34,440 efficiency rows. Qext, Qscat, and cosQ are finite. Qext and Qscat are non-negative. Qabs is non-negative within a relative tolerance of `2e-12`. The derived asymmetry values are finite and range from `1.59e-9` to `0.969874`, within `[-1, 1]`. These checks cover the table structure and basic physical bounds. They do not validate cloud formation against an independent cloud model.

The truth produced finite, nonzero condensate and cloud optical depth in both geometries. Removing clouds changed the maximum absolute spectrum by `27.12 ppm` in emission and `581.56 ppm` in transmission. Changing `cloud_fsed` from `1.5` to `2.5` changed the maximum absolute spectrum by `18.95 ppm` and `221.86 ppm`, respectively. Increasing `cloud_log10_kzz` by `0.5` changed it by `15.96 ppm` and `184.09 ppm`. These one-at-a-time tests show a measurable effect at the injection truth. They do not measure how much prior mass produces negligible or absent clouds.

The adapter uses `do_virtual=false`. Virga warned when a cloud deck reached the bottom of the `100 bar` grid and did not add a sublayer. Treat this as the boundary assumption used in these runs. It is not an independent cloud-base validation.

Emission uses the SH4 solver with a Henyey–Greenstein phase-function closure from Virga's asymmetry values. Transmission uses cloud extinction. It does not use a multiple-scattering transit solver.

## Saved products and checks

Each posterior analysis selected 100 rows with replacement from the normalized native weights, using draw seed `20261108`. It used the same rows for the spectrum, temperature, VMR, condensate, particle-radius, and cloud optical-depth products. The exact central quantile probabilities are `0.02275013194817921`, `0.15865525393145707`, `0.5`, `0.8413447460685429`, and `0.9772498680518208`. The parameter table above instead uses all native weighted rows.

The independent `verify_products.py` check passed for both geometries. It confirmed source-row, vector, weight, and quantile mappings; the full VMR sum for all eight gases; and cloud support and conditional-radius consistency. All 100 draws contain clouds. Every species-layer radius summary has full support. The cloud optical-depth grid has 212 centers and 213 edges, matching the observation grid. The products keep cloud optical depth on that grid; they do not interpolate it to a different grid.

Both geometries have four PNG/PDF figure types, for eight PNGs total. Root visually checked the PNGs and both final parameter PDFs. The parameter figures were rebuilt from the saved native samples, weights, and full weighted quantiles. This plot-only change did not alter numerical products. The full run arrays, posterior chains, configurations, and PNG/PDF files remain outside Git at:

`/Users/jaketaylor/.codex/visualizations/2026/11/06/01a1b0d4-f26e-7571-83c0-e1a6be3d8b9a/virga-local-recovery-20261106`

The compact hash and result record is [`virga_injection_recovery_20261106.json`](../data/virga_injection_recovery_20261106.json). It contains no arrays, chains, or figures.

## Validation status and next gates

The local retrieval and posterior-product pipeline is functional for these two separate cases. The parameter-coverage gate remains open because transmission misses three truths at the 2-sigma level. The result does not support a general claim that ROBERT recovers Virga parameters.

Before making a parameter-coverage claim, run transmission self-injections with independent noise realizations and report coverage across the runs. Keep priors, model, and a native convergence rule fixed before comparing those runs. Separately measure what fraction of the cloud-control prior produces negligible or absent condensate. Do not infer that fraction from the posterior draws, since all posterior draws here contain clouds. An independent cloud-base or high-stream cloud-scattering comparison also remains outside this validation.

## Performance follow-ups

These are proposed upgrades. None has been implemented or benchmarked. No speedup is claimed.

1. **Reuse one Virga result for radiative transfer and cloud profiles.** [`VirgaEddySedCloudModel.evaluate`](../../src/robert_exoplanets/forward/virga_cloud.py#L374) runs Virga and maps optical properties. `compute_cloud_profiles` starts a separate Virga calculation at lines 395–420. Posterior analysis also calculates the spectrum before asking for cloud diagnostics. Add a typed result that retains the one raw Virga evaluation and its mapped optical and physical profiles. Check exact profile and metadata parity, confirm one `virga.compute` call per draw, and compare spectra and likelihoods against the current path. Do not cache complete outputs across different proposals without a complete state identity.
2. **Cache decoded Mie tables through a supported Virga API.** In `virga-exo 2.0.2`, `virga.justdoit.compute` calls `get_mie` for each evaluation (`justdoit.py:87–121`); `get_mie` reads and parses the `.mieff` file (`justdoit.py:1830–1890`). Add an immutable table input or a process-local cache keyed by file-content hash and table options. Check numerical parity, invalidation after file replacement, array immutability, and independent use by MPI worker processes. Keep proposal-dependent sedimentation and radius-distribution optics uncached.
3. **Cache fixed wavelength-mapping setup.** `_cloud_optics_from_virga` interpolates three arrays at [`virga_cloud.py`](../../src/robert_exoplanets/forward/virga_cloud.py#L850), and `_interpolate_spectrum` repeats source sorting, logarithms, and `numpy.interp` for each layer at lines 1070–1098. Cache only the mapping setup. Key it by exact source and target grids, units, and interpolation policy. Check parity for both wavelength orders and exact endpoints, then compare spectra and likelihoods.

The truth-preparation cloud-optics and cloud-profile calls took about `1.07 s` each. The posterior analysis reports their combined adapter time as `2.270 s` per emission draw and `2.306 s` per transmission draw. These times include setup, Virga, and result mapping. They do not isolate Mie-table parsing.

## Source and related records

The runs imported ROBERT from commit `f8318defe95491e9bc50ddbd538e363327147ca9` in `/private/tmp/robert-local-upgrades-20261106`. The worktree was clean at run time. The package version was `0.3.0`. The environment used Python `3.12`, `virga-exo 2.0.2`, and `miepython 3.3.1`. The standard test set passed 77 tests, and all four real-Virga integration checks passed. The independent saved-product check also passed.

See the [prior fixes and local elemental recovery](59_local_elemental_recovery_2026-11-06.md) and the [elemental abundance sensitivity analysis](61_elemental_abundance_sensitivity_2026-11-06.md). The compact record is [`docs/data/virga_injection_recovery_20261106.json`](../data/virga_injection_recovery_20261106.json).
