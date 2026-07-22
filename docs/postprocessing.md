# Post-processing and plotting

ROBERT separates inference products from visualization. Retrievals always
write portable `result.json` and `result_arrays.npz` files, and configured
forward runs write `forward_model.npz`. These numerical products are the source
of truth; plots can be regenerated with different styles without rerunning the
forward model or sampler.

## Automatic plotting from YAML

Plotting is disabled by default. Enable it for either retrievals, forward runs,
or both:

```yaml
plotting:
  enabled: true
  retrieval: true
  forward: true
  style: default
  image_format: png
  dpi: 180
  max_posterior_samples: 20000
  posterior_predictive_samples: 200
  posterior_predictive_seed: 0
  corner_max_parameters: 20
  dataset_colors:
    f322w2: "#20639b"
    f444w: "#ef5675"
    lrs: "#2ca25f"
  parameter_labels:
    metallicity: "[M/H]"
    CtoO: "C/O"
  leave_one_out:
    enabled: false
    max_posterior_draws: 2000
    seed: 0
    pareto_k_threshold: null
```

After successful inference, MPI rank 0 post-processes every completed phase.
A hybrid run therefore receives separate OE and nested-sampling plot folders.
The numerical posterior summary always uses every sample.
`max_posterior_samples` limits rendered marginal/corner samples, while
`posterior_predictive_samples` controls the forward evaluations used for the
68% spectral and temperature-profile credible envelopes.

The benchmark configurations
`wasp69b_cloud_free_native_pg14_R1000.yaml` and
`wasp69b_mie_catalog_pg14_R1000.yaml` enable automatic plotting, and their
derived MultiNest/OE/hybrid configurations inherit that setting.

## Retrieval post-processing

From an isolated run directory:

```bash
python postprocess_retrieval.py --config configuration.yaml
```

ROBERT discovers each completed phase and writes beneath `outputs/plots/`:

- `fit_statistics.json`;
- `posterior_summary.json`;
- `fit_spectrum_residuals.png`;
- `temperature_profiles.png` when the forward model exposes a T-P profile;
- `posterior_corner.png` for nested posteriors up to `corner_max_parameters`;
- `posterior_marginals.png` or `optimal_estimation_parameters.png`;
- `parameter_correlation.png`; and
- `plot_manifest.json`.

The spectrum plot shows the observation-grid posterior model as open squares
and a 68% posterior envelope. For ExoMol correlated-k configurations, ROBERT
also evaluates and plots the median and 68% envelope on the exact native
opacity grid. Formats for which a native-grid correlation is not defined are
reported honestly and retain the observation-grid model rather than drawing
an interpolated curve and calling it native.

When `plotting.leave_one_out.enabled` is true for a nested-sampling result,
ROBERT additionally writes `leave_one_out.json`,
`leave_one_out_arrays.npz`, and `leave_one_out.png`. This PSIS-LOO diagnostic
reports expected out-of-sample predictive accuracy and Pareto-k reliability at
individual wavelength resolution. See
[Bayesian leave-one-out cross-validation](leave_one_out.md) for method,
configuration, comparison, and interpretation details.

Fit diagnostics include total and per-dataset chi-squared, reduced
chi-squared, degrees of freedom, chi-squared survival probability, RMSE,
standardized-residual diagnostics, recomputed log likelihood, AIC, AICc, and
BIC. Nested results additionally retain evidence and timing information; the
posterior summary records weighted quantiles and effective sample size. OE
results use the state and covariance Gaussian approximation.

Process one phase or select a custom appearance without changing YAML:

```bash
python postprocess_retrieval.py \
  --config configuration.yaml \
  --result-dir outputs/multinest \
  --style collaborator.mplstyle \
  --format pdf \
  --color f322w2=mediumpurple \
  --color f444w=darkorange \
  --label metallicity='[M/H]'
```

## Forward-model post-processing

Run or replot a configured forward model with:

```bash
python run_forward.py --config configuration.yaml
python postprocess_forward.py --config configuration.yaml
```

The post-processor writes `outputs/plots/forward/fit_statistics.json`, the
configured parameter values, a spectrum/residual plot, and a plot manifest.
Information criteria are included for consistency but are descriptive when
the forward parameters were prescribed rather than fitted.

## WASP-69b sampler benchmark comparison

After the ten clear/Mie UltraNest, MultiNest, OE, and hybrid runs finish:

```bash
python postprocess_wasp69b_sampler_benchmark.py \
  --project-dir /scratch/dp448/dc-tayl1/my_project/wasp69b_sampler_benchmark_128core
```

The comparison directory contains JSON and CSV summaries plus runtime,
core-hour, fit-statistic, evidence, and shared-parameter comparison plots. If a
run lacks its individual post-processing products, the comparison script
generates them first. By default it requires the complete ten-run matrix; use
`--allow-incomplete` for an interim comparison.

Method colours can be overridden without editing results:

```bash
python postprocess_wasp69b_sampler_benchmark.py \
  --project-dir /path/to/project \
  --method-color ultranest=navy \
  --method-color multinest=crimson \
  --format svg
```
