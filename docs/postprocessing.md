# Post-processing and plotting

ROBERT separates inference and forward-model products from visualization.
Run these commands from the simulation directory outside the ROBERT checkout.
All relative `outputs/` paths below refer to that simulation directory.
Configured forward runs write `forward_model.npz`. Retrieval phases write
`result.json` and `result_arrays.npz`. They also save the best-fit spectrum and
observation for each supported dataset in `best_fit_prediction.json` and
`best_fit_prediction.npz`. These retain masks, bin edges, units, parameter
values, and provenance. Prepared HRS cubes and sampler-only device problems
record why a portable spectrum is unavailable; they are not flattened.

Best-fit and parameter plots can use saved products without opacity files or a
forward model. Posterior spectral intervals and atmospheric plots still require
forward evaluations. To plot only the saved products:

```python
from robert_exoplanets.postprocessing import postprocess_saved_best_fit_output

postprocess_saved_best_fit_output(
    "outputs/multinest", plot_dir="outputs/plots/saved_best_fit"
)
```

## YAML settings

```yaml
plotting:
  enabled: true
  retrieval: true
  forward: false
  style: default
  image_format: png
  dpi: 180
  max_posterior_samples: 20000
  posterior_predictive_samples: 100
  posterior_predictive_seed: 0
  corner_max_parameters: 20
  dataset_colors:
    dataset_a: mediumpurple
    dataset_b: mediumpurple
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
evaluations for spectral, temperature, VMR, and cloud intervals.

`style: default` and `style: robert` select the ROBERT science style used for
the WASP-178 b and WASP-15 b figures. This style uses the purple ROBERT
palette, black observations, filled posterior contours, and clean axes. A
Matplotlib style name or `.mplstyle` path remains a supported override.

## Retrieval products

Run:

```bash
python postprocess_retrieval.py --config configuration.yaml
```

ROBERT discovers completed `multinest`,
`optimal_estimation`, and hybrid phase directories. Each receives its own
folder beneath `outputs/plots/` containing:

- `fit_statistics.json`;
- `posterior_summary.json`;
- `fit_spectrum_residuals.png`;
- `posterior_marginals.png` or `optimal_estimation_parameters.png`;
- `parameter_correlation.png`;
- `posterior_corner.png` when the state dimension permits;
- `temperature_profiles.png` when the model exposes an atmosphere builder;
- `vmr_profiles.png` for its evaluated composition profiles;
- `cloud_profiles.png` for supported cloud models;
- `posterior_predictive_quantiles.npz`; and
- `plot_manifest.json`.

After retrieval, the plotting step selects 100 posterior draws with replacement,
using the sampler weights. It evaluates the same draws for each instrument and
atmospheric region. It plots the median and central 68.27% and 95.45% intervals
for spectra, temperature, VMR, and cloud profiles. The median is `mediumpurple`;
the 1-sigma band is darker than the 2-sigma band. Dataset colour overrides apply
to forward and best-fit-only curves.

The predictive NPZ stores the selected parameter vectors, quantile probabilities,
all five curves, coordinates, and units. The shared keys are
`posterior_draw_vectors`, `parameter_names`, `quantile_probabilities`, and
`quantile_labels`. Each product has `lower_2sigma`, `lower_1sigma`, `median`,
`upper_1sigma`, and `upper_2sigma` fields. These are pointwise intervals;
the median curves need not describe one joint physical model.
Cloud profiles use the actual model's
condensate mass fraction, particle radius, or labelled reference-wavelength layer
optical depth. Unsupported cloud diagnostics have an explicit unavailable record.

Spectrum panels distinguish observations, best-fit residuals, and posterior
intervals. Each instrument keeps its own wavelength grid and bin edges.
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
