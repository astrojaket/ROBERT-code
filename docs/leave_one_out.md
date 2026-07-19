# Bayesian leave-one-out cross-validation

ROBERT implements Bayesian leave-one-out cross-validation (LOO-CV) as an
optional retrieval diagnostic. The implementation follows
[Welbanks et al. (2023)](https://arxiv.org/abs/2212.03872) and uses
Pareto-smoothed importance sampling (PSIS) through
[ArviZ](https://python.arviz.org/). It is applicable to transmission and
emission retrievals that use ROBERT's independent Gaussian likelihoods.

Install the optional dependency with:

```bash
conda run -n robert-exoplanets python -m pip install -e ".[diagnostics]"
```

## What ROBERT computes

For posterior draws \(\theta^s\), ROBERT evaluates every independent
pointwise likelihood term

\[
\log p(y_i \mid \theta^s)
\]

and passes the resulting `posterior_draw x observation` matrix to ArviZ.
PSIS approximates the posterior obtained with observation \(i\) omitted and
returns the pointwise expected log predictive density

\[
\mathrm{elpd}_{\mathrm{LOO},i}
= \log p(y_i \mid y_{-i}),
\]

the sum over all observations, its standard error, the effective number of
parameters, and one Pareto-\(k\) reliability diagnostic per observation.
This needs one completed retrieval plus forward evaluations at posterior
draws; it does not require one complete nested-sampling run per wavelength.

Weighted nested-sampling output is converted to reproducible equal-weight
draws using systematic resampling. The configured seed and selected source
indices are stored with the diagnostic. Limiting the number of posterior
draws controls forward-model cost but should be tested for numerical
stability in publication analyses.

## Configuration

Enable automatic calculation after nested sampling under the existing
post-processing block:

```yaml
plotting:
  enabled: true
  retrieval: true
  leave_one_out:
    enabled: true
    max_posterior_draws: 2000
    seed: 0
    pareto_k_threshold: null
```

`pareto_k_threshold: null` uses ArviZ's current sample-size-dependent
threshold. Set it to `0.7` to reproduce the fixed threshold used by Welbanks
et al. Points above the resolved threshold are marked
`requires_exact_refit: true`. ROBERT deliberately does not replace those
values with an unreliable approximation: run an exact retrieval with that
observation excluded before interpreting the point.

The post-processor writes:

- `leave_one_out.json`, including the total and pointwise ELPD, Pareto-\(k\),
  dataset identity, wavelength, and reliability flags;
- `leave_one_out_arrays.npz`, containing the lossless pointwise likelihood
  matrix, ELPD, Pareto-\(k\), and selected posterior indices; and
- `leave_one_out.png` (or the configured image format), showing pointwise
  ELPD and Pareto-\(k\) against wavelength.

Optimal-estimation phases do not contain sampled posterior draws, so they are
reported as `not_available` rather than treated as PSIS input.

## Python API and model comparison

Use `run_psis_leave_one_out` with a retrieval problem and posterior samples,
or use `psis_leave_one_out` directly with a precomputed pointwise
log-likelihood matrix:

```python
from robert_exoplanets import (
    compare_psis_leave_one_out,
    run_psis_leave_one_out,
)

full = run_psis_leave_one_out(
    full_problem,
    full_result.samples,
    weights=full_result.weights,
    max_posterior_draws=2000,
    seed=17,
)
reduced = run_psis_leave_one_out(
    reduced_problem,
    reduced_result.samples,
    weights=reduced_result.weights,
    max_posterior_draws=2000,
    seed=17,
)
comparison = compare_psis_leave_one_out(full, reduced)
```

The comparison returns
\(\Delta\mathrm{elpd}=\mathrm{elpd}_{\rm full}-\mathrm{elpd}_{\rm reduced}\)
and the paired standard error from the pointwise differences. Positive values
favor the first model's expected out-of-sample prediction. Models must use the
same observations in the same order.

## Interpretation limits

LOO-CV complements Bayesian evidence; it does not replace it. In particular:

- LOO-CV does not apply an evidence-like Occam penalty for unused prior
  volume.
- A large Pareto-\(k\) says the PSIS approximation is unreliable for that
  point. It does not, by itself, say the observation or atmospheric model is
  wrong.
- The ELPD difference divided by its standard error is not a Gaussian
  detection significance.
- If `likelihood.include_normalization` is false, absolute ELPD is shifted by
  a data-dependent constant. Same-data model differences, point rankings, and
  Pareto-\(k\) diagnostics are unchanged. Enable normalization when absolute
  predictive densities will be reported.
- Correlated measurements cannot be treated as independently leaveable
  points without defining scientifically valid likelihood units. ROBERT's
  present function is therefore limited to independent Gaussian terms.

The central use is model criticism: identify which wavelength bins support or
undermine an atmospheric interpretation, and whether an apparent preference
depends on only one influential datum.
