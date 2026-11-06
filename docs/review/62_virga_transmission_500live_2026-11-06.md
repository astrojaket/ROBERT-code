# Virga transmission 500-live-point follow-up — 6 November 2026

**Result:** Both new 500-live-point transmission fits reached native MultiNest convergence with four MPI ranks and no iteration cap. The noisy fit contains all four injected values inside the 2-sigma intervals and none inside the 1-sigma intervals. The noiseless fit contains all four inside both intervals. The earlier 100-live-point noisy fit contained only one of four values inside 2-sigma. This is a controlled comparison for one noise realization, not a coverage study or a general Virga validation.

## Experiment

The noisy 500-live-point case reuses the exact data, noise vector, uncertainty vector, truth spectrum, opacity inputs, priors, and model from the earlier 100-live-point transmission case. It changes the live-point count and uses a fresh sampler output directory. The noiseless case uses the same injected truth with an all-zero noise vector and the same 20 ppm uncertainty in all 212 bins. It is also a fresh run. A source-guarded preflight regenerated the truth with the current clean checkout and matched the archived truth array exactly; it made one forward call for each case. No forward calls were made for the saved-likelihood checks.

Both runs use fixed gas VMRs and retrieve four values: `T_upper`, `T_deep`, `cloud_fsed`, and `cloud_log10_kzz`. The temperature profile has two fitted spline controls on a 48-layer atmosphere. The domain is 0.6–5.0 micron with 212 log-spaced bins and 213 edges, at nominal R=100. The uncertainties are independent 20 ppm Gaussian errors. The original priors are `T_upper=700–1200 K`, `T_deep=1500–2200 K`, `cloud_fsed=0.5–4`, and `cloud_log10_kzz=8–10`. The injected values are 850 K, 1800 K, 1.5, and 9.0, respectively.

The fixed trace-gas VMRs are H2O `1e-3`, CO `1e-3`, CO2 `1e-5`, CH4 `1e-5`, NH3 `1e-6`, and HCN `1e-7`. H2 and He use background proportions 0.8547 and 0.1453 to fill the remaining abundance; these are not their final normalized VMRs. Clouds use MgSiO3. Transmission includes cloud extinction and does not include a multiple-scattering transit solver.

`cloud_log10_kzz` means `log10[Kzz/(cm^2 s^-1)]`; `cloud_fsed` is dimensionless.

## Native retrieval results

| Case | Live points | Weighted rows | ESS | Native likelihood evaluations | Callback evaluations | Inference time | Timed process wall | Truth in 1-sigma | Truth in 2-sigma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Noisy reference | 100 | 900 | 400.494 | 3,182 | 3,285 | 1,012.02 s | 1,025.77 s | 0/4 | 1/4 |
| Noisy follow-up | 500 | 4,393 | 2,046.000 | 13,741 | 14,030 | 4,271.28 s | 4,286.49 s | 0/4 | 4/4 |
| Noiseless follow-up | 500 | 4,680 | 2,270.696 | 14,841 | 15,098 | 4,598.42 s | 4,613.53 s | 4/4 | 4/4 |

All three results report convergence. The new runs used `dlogz=0.5`, 500 live points, four MPI ranks, one numerical thread per rank, `max_iter=0`, and `resume=false`. There was no iteration, wall-time, or RSS kill limit. The configured noise seed was `20261107`; the effective MultiNest seed in the result metadata is `21594`.

Parameter intervals use every native weighted row. The central quantile probabilities are `0.02275013194817921`, `0.15865525393145707`, `0.5`, `0.8413447460685429`, and `0.9772498680518208`. Thus the 1-sigma interval spans the middle 68.2689% and the 2-sigma interval spans the middle 95.45% of posterior weight.

| Run | Parameter | Truth | Median | 1-sigma interval | 2-sigma interval | Truth inside 1-sigma / 2-sigma |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Noisy 100 | `T_upper` (K) | 850 | 795.995 | 765.309–826.688 | 739.238–846.115 | no / no |
| Noisy 100 | `T_deep` (K) | 1800 | 1888.018 | 1844.505–1932.720 | 1814.195–1965.686 | no / no |
| Noisy 100 | `cloud_fsed` | 1.5 | 2.841 | 2.163–3.566 | 1.669–3.941 | no / no |
| Noisy 100 | `cloud_log10_kzz` | 9.0 | 8.593 | 8.427–8.790 | 8.340–9.007 | no / yes |
| Noisy 500 | `T_upper` (K) | 850 | 804.784 | 773.056–838.477 | 742.063–866.747 | no / yes |
| Noisy 500 | `T_deep` (K) | 1800 | 1874.995 | 1824.627–1921.466 | 1779.813–1963.344 | no / yes |
| Noisy 500 | `cloud_fsed` | 1.5 | 2.638 | 1.826–3.483 | 1.232–3.924 | no / yes |
| Noisy 500 | `cloud_log10_kzz` | 9.0 | 8.635 | 8.435–8.914 | 8.336–9.220 | no / yes |
| Noiseless 500 | `T_upper` (K) | 850 | 826.971 | 789.504–861.203 | 750.526–888.515 | yes / yes |
| Noiseless 500 | `T_deep` (K) | 1800 | 1834.902 | 1781.955–1892.487 | 1737.316–1947.926 | yes / yes |
| Noiseless 500 | `cloud_fsed` | 1.5 | 1.860 | 1.254–2.804 | 0.881–3.683 | yes / yes |
| Noiseless 500 | `cloud_log10_kzz` | 9.0 | 8.858 | 8.566–9.156 | 8.364–9.492 | yes / yes |

The noisy 500-live-point result has broader marginal tails and places all four truths inside 2-sigma for this realization. It contains no noisy truth inside 1-sigma. This comparison does not identify a unique reason for the earlier interval misses.

## Evidence and likelihood checks

The result records report Importance Nested Sampling (INS) evidence. The native retrieval log also records the ordinary NS estimates. These are separate estimates:

| Case | Result INS `ln(Z)` | INS error | Native log `ln(ev)` | Native `ln(ev)` error | Final NS `ln(Z)` trace |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy 500 | 1980.581176 | 0.840470 | 1979.277499 | 0.139350 | 1979.009247 |
| Noiseless 500 | 2086.563923 | 0.034264 | 2087.047174 | 0.140823 | 2086.789879 |

`dlogz=0.5` is the stopping threshold for estimated remaining evidence. It is not the reported evidence uncertainty. The INS error in `result.json` and the ordinary NS error in the terminal log use different estimates and must not be compared as if they were the same statistic.

The independent saved-array Gaussian checks used no new forward calls. For the noisy case, truth `chi-square=219.219145`, saved maximum-likelihood `chi-square=216.623362`, and their difference is `2.595783` for four fitted parameters. The pointwise median prediction has `chi-square=216.352045`. For the noiseless case, truth `chi-square=0`, saved maximum-likelihood `chi-square=0.095483`, and pointwise median prediction `chi-square=0.391743`. The checks confirm data/truth/noise identity and likelihood arithmetic. They do not explain marginal coverage.

## Product checks and provenance

The independent verifier reports 175 saved-product checks passed, with no failures. It checks native run metadata, clean-source identity, data and input hashes, native-weight parameter quantiles and ESS, the 100 saved weighted-row identities, common draws across predictive/profile products, finite values, VMR closure, and cloud-grid and conditional-radius support. The same 100 weighted rows supply spectral, temperature, VMR, condensate, particle-radius, and cloud optical-depth products. Parameter intervals use all native weighted samples, not this 100-draw subset.

The archived truth was generated at `f8318defe95491e9bc50ddbd538e363327147ca9`. The fresh preflight and both fits record clean commit `6f1cf1c17aab982b3e5c8c1b4942fed75d27e910`, imported from `/private/tmp/robert-local-upgrades-20261106/src/robert_exoplanets/__init__.py`. The latter commit is a documentation-only successor; the source, data, scripts, project configuration, and runner were checked unchanged from the truth-generation commit. The preflight arrays match the archived truth exactly (maximum difference zero).

The noisy observation hash is `3f6a5823b9dfae030cf52315614eb791b10854fedd0700fb1294b65d56d5d912`; the shared noisy vector hash is `4a06389fdadf195b0454a30ffdcaa9ded3e06502feca7ad8743d544b41d13b1b`. The noiseless observation hash is `4a0210084665ba576f1c681bb16566d2742d97ac787178415464d751f64bde15`; its zero-vector hash is `bf75520ae2a2df40c3d8b29b71564bac7a99659315d2e1c83b750c96807a078d`. The exact truth-spectrum hash is `7d40f32ae65fcb5ca9bcb21eb6247364d467a8c61a66b1bc7b9102c72872a861`. The compact JSON record also stores injection, manifest, result-array, opacity, refractive-index, Mie-table, and source optical-constant hashes.

The six R=100 KTA inputs are the committed H2O, CO, CO2, CH4, NH3, and HCN tables, with eight g ordinates retained. Their identities are recorded in the compact JSON. The shared MgSiO3 Mie table hash is `f33137b611a6453f4a547e40cde7babcda9a3b50d37339fb80cbb7eed11221d9`; it covers 0.4–7.0 micron and was read back with finite, bounded efficiencies and asymmetry values. The recorded environment uses Python 3.12, `virga-exo 2.0.2`, and `miepython 3.3.1`.

Final visual review passed for all ten PNGs and ten rendered PDF pages, including the interval comparison and combined spectrum/residual plot. The two parameter figures were restyled for shorter labels and y-axis headroom. Root re-hashed ten protected numerical files; their hashes did not change. The combined plot labels its shaded band as a **model interval**, separate from the 20 ppm observation errors. The saved-product verifier passed 175 checks with zero failures. Run arrays, chains, logs, configurations, and figures remain outside Git at:

`/Users/jaketaylor/.codex/visualizations/2026/11/06/01a1b0d4-f26e-7571-83c0-e1a6be3d8b9a/virga-transmission-500live-20261106`

The compact record is [`virga_transmission_500live_20261106.json`](../data/virga_transmission_500live_20261106.json). It contains summary values and hashes only.

## Interpretation and remaining checks

The noiseless result shows that, for this model and prior, zero-noise data can place all four truths within 1-sigma at 500 live points. The noisy 500-live-point result improves the observed 2-sigma inclusion from one of four parameters in the 100-live-point baseline to four of four. This one noise draw does not estimate coverage frequency, and the comparison does not prove that sampler live-point count alone caused the change.

Keep these checks open:

1. Repeat the fixed noisy and noiseless 500-live-point cases with independent sampler seeds. Compare native-weighted medians and tails, and both NS and INS evidence estimates. Increase live points if the estimates still shift. The current noisy INS uncertainty is larger than the 100-live-point value; this pair does not identify a unique cause.
2. Repeat the noisy recovery with independent noise seeds to measure marginal coverage. Keep the priors, model, and convergence rule fixed. This is separate from checking sampler-seed stability.
3. Measure how much of the cloud-control prior produces negligible or absent clouds. The current posterior draws cannot estimate that prior fraction.
4. Compare cloud-base behavior and cloud scattering against independent references, including a high-stream check where relevant.
5. Measure per-evaluation and full-sampler costs before claiming or tuning performance improvements.
6. Keep claims limited to this fixed-gas, four-parameter, local synthetic transmission test. It does not validate general Virga cloud physics or target-science retrievals.

See the original 100-live-point result and broader local Virga scope in the [6 November recovery record](60_local_virga_recovery_2026-11-06.md), and the updated [repository audit](57_repository_audit_2026-11-06.md).
