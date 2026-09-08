# AGENTS.md

Guidance for coding agents working in the ROBERT repository.

## Project Intent

ROBERT is a production-ready radiative-transfer and atmospheric-retrieval
framework for its documented and benchmarked regimes. It combines typed planet, star, atmosphere, opacity, observation,
instrument, forward-model, likelihood, and inference components behind a
strict YAML workflow. It supports thermal-emission and transmission spectra,
correlated-k and opacity-sampling inputs, equilibrium and free chemistry,
cloud-free and cloudy atmospheres, one- and two-region emission, instrument
binning, Gaussian likelihoods, optimal estimation and PyMultiNest.
These workflows produce portable configuration snapshots, manifests, numerical
results, diagnostics, and plots; use the documented validation boundaries when
assessing science results. Implemented features do not by themselves establish
scientific validation.

The Python distribution name is `robert-exoplanets`; avoid introducing packaging or documentation that assumes the distribution is named `robert`.

## Working Guidelines

- When running Python, tests, package installation, retrievals, examples, or
  scientific validation, use the `robert-exoplanets` Conda environment. Prefer
  explicit commands such as `conda run -n robert-exoplanets python ...` and
  `conda run -n robert-exoplanets python -m pytest ...`; do not rely on the
  currently activated shell environment.
- Always use Git over SSH for GitHub operations, with remotes such as
  `git@github.com:owner/repository.git`. Do not require or use the GitHub CLI
  (`gh`); create commits locally and fetch or push them through the SSH remote.
- Keep physics-facing APIs explicit and typed so later scientific implementations can replace stubs without changing user-facing examples.
- Prefer small, well-tested modules over broad framework code.
- Do not add numerical approximations that look like real retrieval physics unless they are clearly labeled as placeholders.
- Keep examples runnable with the default development dependencies.
- Add tests for public behavior whenever you add or change an API.
- Follow `docs/data_policy.md`: retain required lightweight inputs and compact
  benchmark test oracles, but do not commit heavy opacity data or generated
  scientific products.
- Keep user simulations outside the ROBERT checkout. Use
  `scripts/create_run_directory.py` for run configurations, runner wrappers,
  outputs, prepared caches, scratch files, and cluster logs. Reuse shared
  opacity inputs across runs; do not download or copy them for each simulation.
- Use `opacity_data/ktables_exomol/` as the default R=1000 source for JWST
  work. The local tables were copied from Dropbox and verified by checksum.
  Keep each instrument's wavelength coverage and bin edges separate. Use
  bundled R=100 only when explicitly selected for quick validation.
- After retrieval, use 100 weighted posterior draws for median and central
  1-sigma/2-sigma spectra, T-P, VMR, and cloud profiles. Use the same draws
  across products and save their numerical values. Use `mediumpurple` for
  posterior curves and bands, unless the user selects another colour.
- When changing package boundaries, data models, interfaces, plugin design,
  configuration, or testing design, read the relevant guidance in
  `docs/rfcs/0001-robert-architectural-specification.md` and
  `docs/architecture/`.

## Repository Layout

- `src/robert_exoplanets/`: Python package source.
- `src/robert_exoplanets/core/`: Core grids, spectra, exceptions, and logging helpers.
- `src/robert_exoplanets/bodies/`: Planet and star domain objects.
- `src/robert_exoplanets/instruments/`: Observation and future instrument objects.
- `src/robert_exoplanets/atmosphere/`: Temperature, chemistry, and evaluated atmosphere state.
- `src/robert_exoplanets/opacity/`: Opacity metadata, readers, preparation, and interpolation.
- `src/robert_exoplanets/rt/`: Emission, transmission, extinction, geometry, and scattering solvers.
- `src/robert_exoplanets/forward/`: Reusable forward-model orchestration over atmosphere, opacity, RT, and instruments.
- `src/robert_exoplanets/retrieval/`: Priors, retrieval problems, manifests, inference, and sampler adapters.
- `src/robert_exoplanets/validation/`: Deterministic injection-recovery contracts.
- `examples/`: Runnable example scripts.
- `tests/`: Pytest suite.

## Useful Commands

```bash
conda run -n robert-exoplanets python -m pip install -e ".[dev,opacity,retrieval]"
conda run -n robert-exoplanets python -m pytest
conda run -n robert-exoplanets python examples/stub_emission_retrieval.py
```

## Current Scope Boundaries

The following remain outside the supported scope of current workflows:

- Calibrated JWST pipeline products and general cross-instrument covariance
  in configured workflows. Python APIs support covariance within declared
  observation blocks.
- Production-complete chemistry, clouds, and opacity databases.
- Science-grade multiple-scattering and instrument throughput/line-spread models.
- Broad sampler support and fully validated long-run posterior workflows.
