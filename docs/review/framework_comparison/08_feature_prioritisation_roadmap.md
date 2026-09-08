# Feature Prioritisation Roadmap

The current retrieval baseline uses PyMultiNest for nested sampling and keeps
Optimal Estimation as a separate route for supported independent Gaussian
problems. The rows below describe remaining extensions beyond that baseline.

## Current baseline

These capabilities form the current supported release boundary. Scientific
evidence remains case-specific and is recorded in the validation summary.

| Feature | Why it is required | Evidence from ecosystem |
| --- | --- | --- |
| Typed config parsing | Prevents raw dict drift and silent invalid modes | TauREx, Exo_Skryer, pRT show config matters. |
| Pressure/layer grid objects | Every physics module depends on consistent grid conventions | Universal pattern. |
| 1D emission RT | Core science target | All reviewed exoplanet frameworks support emission. |
| Correlated-k opacity | Practical retrieval default | NEMESIS, TauREx, POSEIDON, pRT, PICASO, CHIMERA, Exo_Skryer. |
| CIA | Essential for H2/He atmospheres | Universal or near-universal support. |
| Rayleigh | Basic continuum/scattering contribution | Common support. |
| Free chemistry | Simplest robust retrieval mode | Universal retrieval pattern. |
| Basic temperature profiles | Enables compact retrieval | Universal retrieval pattern. |
| Simple cloud model | JWST spectra often require cloud/haze flexibility | All modern frameworks include clouds. |
| Instrument response/binning | Retrieval must compare model to real JWST data | pRT, POSEIDON, PICASO, Exo_Skryer. |
| Gaussian likelihood with jitter/offsets | Multi-instrument JWST needs nuisance parameters | POSEIDON, pRT, Brewster, Exo_Skryer. |
| PyMultiNest nested-sampling adapter | Tested nested-sampling baseline | MPI and MultiNest dependency isolation. |
| Optimal estimation route | Independent Gaussian retrieval method | Explicit observation and Jacobian contract. |
| Reproducible output manifest | Scientific traceability | Lessons from all mature workflows. |
| End-to-end smoke example | Prevents architecture from becoming theoretical | NEMESIS, CHIMERA, Brewster examples are valuable. |

## Remaining extensions

These require a stated observing need and a separate validation case.

| Feature | Benefit | Dependency |
| --- | --- | --- |
| Additional nested-sampler adapters | Independent comparisons after the PyMultiNest baseline is stable | A separate scientific need and validation case. |
| Covariance likelihood | Handles correlated data products | Observation/covariance schema. |
| Chemistry extensions | Broader disequilibrium or photochemistry beyond the current equilibrium and analytic pressure-quench paths | Chemistry interface and independent validation data. |
| Opacity-sampling / line-by-line mode | High-resolution and validation use cases | Opacity provider abstraction. |
| Stellar-contamination extensions | Surface maps, limb darkening, and active-region evolution beyond the current TSLE route | Stellar model, response contract, and independent validation data. |
| Patchy/cloud-column mixing | Cloud inhomogeneity without full 3D | Column abstraction. |
| Cloudy transmission extensions | Scattering return and refraction beyond the current absorption-dominated route | Atmosphere, opacity, and geometry interfaces stable. |
| ArviZ output | Standard posterior analysis | Sampler-independent result schema. |
| Documentation notebooks | User onboarding | Stable API and small fixtures. |

## Future

These are valuable but likely to introduce significant complexity.

| Feature | Why future | Trigger to prioritize |
| --- | --- | --- |
| Phase curves | Requires multidimensional geometry and multiple observations | A funded/active phase-curve science case. |
| Full 2D/3D retrieval | Large parameter and validation burden | Mature 1D retrievals plus column framework. |
| Reflection spectroscopy | Scattering and stellar/phase geometry complexity | Need for reflected-light JWST/future mission studies. |
| Polarisation | Specialized RT and data products | Specific collaborator/science requirement. |
| Photochemistry coupling | Expensive, stiff, backend-specific | Validated external backend and clear use case. |
| JAX/GPU backend | Install, cache, and determinism complexity | NumPy/Numba backend becomes bottleneck. |
| Variational inference | Hard to validate for retrieval posteriors | Need for fast approximate inference after exact baselines. |
| Hamiltonian methods | Requires differentiable constrained model | JAX backend and validated gradients. |
| Web config UI | Maintenance surface | CLI/YAML schema is stable and users need GUI. |

## Development Phases

1. Foundation: grids, state objects, config, manifests, observations.
2. Forward model: opacity provider, CIA/Rayleigh, emission RT, diagnostics.
3. Retrieval: likelihood, priors, PyMultiNest adapter, optimal-estimation route, result schema.
4. Scientific examples: clear, cloudy, multi-instrument JWST emission.
5. Extension: covariance, chemistry extensions, line-by-line/OS; defer additional samplers.
6. New modes: transmission extensions, patchy columns, and TSLE extensions.

## Non-Goals for the First Real Release

- Full phase-curve retrieval.
- GPU support.
- Multiple chemistry backends.
- Full NEMESIS file compatibility.
- A large plugin marketplace.
- A web app.
