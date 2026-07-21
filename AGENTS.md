# AGENTS.md

Guidance for coding agents working in the ROBERT repository.

## Project Intent

ROBERT is an early-stage Python package for JWST exoplanet atmospheric
retrievals, initially focused on emission. It includes typed domain containers,
correlated-k and opacity-sampling preparation, cloud-free and cloudy emission,
absorption-dominated transmission, likelihood and optimal-estimation
infrastructure, and optional UltraNest and MultiNest adapters. These components
support validation retrievals but are not yet a production science model.

The Python distribution name is `robert-exoplanets`; avoid introducing packaging or documentation that assumes the distribution is named `robert`.

## Working Guidelines

- Always run Python, tests, package installation, retrievals, examples, and
  scientific validation in the `robert-exoplanets` Conda environment. Prefer
  explicit commands such as `conda run -n robert-exoplanets python ...` and
  `conda run -n robert-exoplanets python -m pytest ...`; do not rely on the
  currently activated shell environment.
- Keep physics-facing APIs explicit and typed so later scientific implementations can replace stubs without changing user-facing examples.
- Prefer small, well-tested modules over broad framework code.
- Do not add numerical approximations that look like real retrieval physics unless they are clearly labeled as placeholders.
- Keep examples runnable with the default development dependencies.
- Add tests for public behavior whenever you add or change an API.
- Follow `docs/data_policy.md`: retain required lightweight inputs and compact
  benchmark test oracles, but do not commit heavy opacity data or generated
  scientific products.
- Follow `docs/rfcs/0001-robert-architectural-specification.md` and the companion docs in `docs/architecture/` for package boundaries, dependency rules, data models, plugin design, configuration, and testing philosophy.

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

The following are deliberately out of scope for the initial skeleton:

- JWST pipeline calibration products.
- Calibrated JWST pipeline products and multi-instrument covariance.
- Production-complete chemistry, clouds, and opacity databases.
- Science-grade multiple-scattering and instrument throughput/line-spread models.
- Broad sampler support and fully validated long-run posterior workflows.
