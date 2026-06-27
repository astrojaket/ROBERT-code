# Feature Prioritisation Roadmap

## Must-Have

These are required before ROBERT should be used for real JWST emission retrievals.

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
| dynesty sampler adapter | Accessible nested-sampling baseline | Common Python retrieval choice. |
| Reproducible output manifest | Scientific traceability | Lessons from all mature workflows. |
| End-to-end smoke example | Prevents architecture from becoming theoretical | NEMESIS, CHIMERA, Brewster examples are valuable. |

## Nice-to-Have

These should follow once the must-have path is tested.

| Feature | Benefit | Dependency |
| --- | --- | --- |
| UltraNest adapter | Independent nested-sampling backend | Stable likelihood API. |
| PyMultiNest adapter | Cluster and legacy comparison support | Optional dependency isolation. |
| Covariance likelihood | Handles correlated data products | Observation/covariance schema. |
| Equilibrium chemistry backend | Common comparative retrieval mode | Chemistry interface and validation data. |
| Opacity-sampling / line-by-line mode | High-resolution and validation use cases | Opacity provider abstraction. |
| Stellar contamination model | Important for transmission and some emission analyses | Star model and instrument response. |
| Patchy/cloud-column mixing | Cloud inhomogeneity without full 3D | Column abstraction. |
| Transmission RT | Broadens science scope | Atmosphere and opacity interfaces stable. |
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
3. Retrieval: likelihood, priors, dynesty adapter, result schema.
4. Scientific examples: clear, cloudy, multi-instrument JWST emission.
5. Extension: covariance, equilibrium chemistry, line-by-line/OS, second sampler.
6. New modes: transmission, patchy columns, stellar contamination.

## Non-Goals for the First Real Release

- Full phase-curve retrieval.
- GPU support.
- Multiple chemistry backends.
- Full NEMESIS file compatibility.
- A large plugin marketplace.
- A web app.
