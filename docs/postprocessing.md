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
  dataset_colors:
    f322w2: "#20639b"
    f444w: "#ef5675"
    lrs: "#2ca25f"
  parameter_labels:
    metallicity: "[M/H]"
    CtoO: "C/O"
```

After successful inference, MPI rank 0 post-processes every completed phase.
A hybrid run therefore receives separate OE and nested-sampling plot folders.
The numerical posterior summary always uses every sample;
`max_posterior_samples` limits only the samples rendered in histograms.

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
- `posterior_marginals.png` or `optimal_estimation_parameters.png`;
- `parameter_correlation.png`; and
- `plot_manifest.json`.

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
