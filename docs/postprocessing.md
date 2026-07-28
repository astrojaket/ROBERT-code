# Post-processing and plotting

ROBERT separates inference and forward-model products from visualization.
Configured forward runs write `forward_model.npz`. Retrieval phases write
`result.json` and `result_arrays.npz`. Plots can therefore be regenerated
without rerunning radiative transfer or inference.

## YAML settings

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
    dataset_a: "#20639b"
    dataset_b: "#ef5675"
  parameter_labels:
    metallicity: "[M/H]"
    CtoO: "C/O"
  leave_one_out:
    enabled: false
    max_posterior_draws: 2000
    seed: 0
    pareto_k_threshold: null
```

`max_posterior_samples` limits samples rendered in marginal and corner plots;
summary statistics still use the complete weighted posterior.
`posterior_predictive_samples` controls reproducibly resampled forward
evaluations for spectral and temperature intervals.

## Retrieval products

Run:

```bash
python postprocess_retrieval.py --config configuration.yaml
```

ROBERT discovers completed `ultranest`, `multinest`,
`optimal_estimation`, and hybrid phase directories. Each receives its own
folder beneath `outputs/plots/` containing:

- `fit_statistics.json`;
- `posterior_summary.json`;
- `fit_spectrum_residuals.png`;
- `posterior_marginals.png` or `optimal_estimation_parameters.png`;
- `parameter_correlation.png`;
- `posterior_corner.png` when the state dimension permits;
- `temperature_profiles.png` when the model exposes an atmosphere builder;
- `posterior_predictive_quantiles.npz`; and
- `plot_manifest.json`.

The predictive NPZ stores the numerical q16, q50, and q84 curves behind the
plots. Dataset products use keys such as:

```text
spectrum_DATASET_wavelength_micron
spectrum_DATASET_q16
spectrum_DATASET_q50
spectrum_DATASET_q84
```

When native-opacity-grid evaluation is supported, it also stores
`native_wavelength_micron` and `native_q16`, `native_q50`, `native_q84`.
Temperature products use `temperature_REGION_pressure_bar` and
`temperature_REGION_q16_K`, `q50_K`, and `q84_K`.

Spectrum panels distinguish observations, the best-fit residual calculation,
the posterior median, and the central 68% posterior predictive interval.
Parameter names and order come from the serialized result. ROBERT does not add
truth or reference markers unless they are explicitly available.

Fit diagnostics include total and per-dataset chi-squared, reduced chi-squared,
degrees of freedom, survival probability, RMSE, standardized residuals,
recomputed log likelihood, AIC, AICc, and BIC. Nested results retain evidence,
timing, weighted quantiles, and effective sample size. Optimal-estimation
results use their state and covariance Gaussian approximation.

Process a particular phase or override appearance:

```bash
python postprocess_retrieval.py \
  --config configuration.yaml \
  --result-dir outputs/multinest \
  --style paper.mplstyle \
  --format pdf \
  --color dataset_a=mediumpurple \
  --label metallicity='[M/H]'
```

For a headless machine:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/scratch/matplotlib"
mkdir -p "$MPLCONFIGDIR"
python postprocess_retrieval.py --config configuration.yaml
```

The generated Slurm and Glamdring launchers set these automatically.

## Forward-model products

Run or regenerate forward diagnostics:

```bash
python run_forward.py --config configuration.yaml
python postprocess_forward.py --config configuration.yaml
```

ROBERT writes beneath `outputs/plots/forward/`:

- `fit_statistics.json`;
- `forward_parameters.json`;
- `forward_spectrum_residuals.png`; and
- `plot_manifest.json`.

Information criteria are included for a consistent schema but are descriptive
when the parameters were prescribed rather than inferred.
