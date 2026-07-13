# Recommended Architecture for ROBERT

ROBERT should be a focused JWST exoplanet emission retrieval framework with a
clean path to transmission and multidimensional extensions. It should optimize
for scientific correctness, simplicity, modularity, reproducibility,
maintainability, and then performance.

## Adopt Directly

| Idea | Source frameworks | Recommendation | Justification |
| --- | --- | --- | --- |
| Dataset/instrument object | petitRADTRANS, POSEIDON, Exo_Skryer | Create `Observation` and `InstrumentResponse` as first-class objects | JWST emission retrievals are often multi-instrument and need offsets, jitter, binning, and masks. |
| Contribution diagnostics | NEMESIS, TauREx, pRT, PICASO | Return gas, CIA, Rayleigh, cloud, and total contribution diagnostics | Diagnostics are essential for scientific review and regression tests. |
| Component registry | TauREx, Exo_Skryer | Use simple registries for temperature, chemistry, cloud, opacity, RT, likelihood, sampler | Enables extension without raw string switches scattered through physics code. |
| RT engine boundary | petitRADTRANS | Build `RadiativeTransferBackend` with stable inputs and outputs | Keeps physics kernels reusable outside retrieval. |
| Sampler adapters | TauREx, pRT, Exo_Skryer | Wrap dynesty first; add other samplers behind the same interface | Retrieval logic should not know sampler-specific APIs. |
| Run manifest | NEMESIS examples, pRT data handling, PICASO data versioning | Save config hash, code version, opacity checksums, random seed, sampler settings | Reproducibility must be built in from the first real retrieval. |

## Modify Before Adopting

| Idea | Use with modification | Reason |
| --- | --- | --- |
| Plugin architecture | Keep a simple registry, not a full mixin system initially | TauREx-style plugins are powerful, but ROBERT's early surface should stay small. |
| JAX backend | Treat as future backend | Exo_Skryer shows value, but JAX changes array semantics, caching, and install complexity. |
| Multiple chemistry backends | Start with free chemistry and one equilibrium adapter later | Multiple chemistry engines introduce validation and dependency burden. |
| Multidimensional atmospheres | Design data structures to allow columns, but implement 1D first | POSEIDON/PICASO show value; early implementation would distract from robust JWST emission. |
| Broad sampler support | Start with dynesty, then UltraNest or PyMultiNest if needed | Exo_Skryer breadth is attractive but can outpace tests. |

## Avoid Entirely

- Sidecar files as internal state.
- Global `settings.runargs` or mutable process-wide retrieval tuples.
- Import-time MPI, BLAS thread, CUDA, or reference-data configuration.
- User-edited template scripts as the primary run interface.
- Raw dictionaries passed into RT, opacity, or likelihood kernels.
- Monolithic `run_retrieval` functions with many positional arguments.
- Silent extrapolation outside opacity coverage.

## Proposed Mature Package Shape

```text
src/robert_exoplanets/
  core/
    grids.py
    state.py
    spectrum.py
    results.py
  parameterizations/
    temperature.py
    composition.py
    clouds.py
    registry.py
  opacity/
    providers.py
    correlated_k.py
    cia.py
    rayleigh.py
    coverage.py
    cache.py
  rt/
    base.py
    emission.py
    diagnostics.py
    backends/
      numpy.py
      numba.py
  instruments/
    observation.py
    response.py
    jwst.py
    calibration.py
  likelihoods/
    gaussian.py
    covariance.py
  retrieval/
    problem.py
    priors.py
    transforms.py
    samplers/
      base.py
      dynesty.py
  io/
    config.py
    manifests.py
    tables.py
  validation/
    reference_cases.py
    comparisons.py
```

The repository should not create all of these directories before they are real.
The skeleton can grow toward this layout as features are implemented.

## Core Interfaces

```python
class TemperatureProfile:
    def evaluate(self, parameters, pressure_grid): ...

class CompositionProfile:
    def evaluate(self, parameters, pressure_grid, temperature): ...

class OpacityProvider:
    def prepare(self, spectral_grid, pressure_grid, species): ...
    def evaluate(self, atmosphere_state): ...

class RadiativeTransferBackend:
    def emission(self, atmosphere_state, opacity_state, geometry): ...

class InstrumentResponse:
    def observe(self, model_spectrum): ...

class Likelihood:
    def loglike(self, prediction, observation, parameters): ...

class SamplerAdapter:
    def run(self, problem, output_dir): ...
```

## Recommended MVP

The first substantial ROBERT release should include:

- 1D thermal emission.
- Correlated-k opacity provider with strict coverage validation.
- CIA and Rayleigh.
- Isothermal and Guillot-like temperature profiles.
- Free constant-with-altitude chemistry.
- Gray cloud deck or opacity slab.
- JWST-style observation object with binning, mask, offset, and jitter.
- Gaussian independent-error likelihood.
- dynesty sampler adapter.
- Run manifest and posterior output.
- Reference tests for a cloud-free atmosphere and a cloudy atmosphere.

## Testing Strategy

- Unit tests for every pure parameterization.
- Shape/unit tests for grids and spectra.
- Golden tests for opacity interpolation and simple emission spectra.
- Likelihood tests with known residuals and covariance.
- End-to-end smoke test with tiny opacity fixtures.
- Benchmark tests marked separately from CI-required tests.

## Performance Strategy

Start with NumPy reference kernels. Add a Numba backend once the equations and
tests are stable. Keep JAX/GPU as a later backend after ROBERT's object model,
cache keys, and output schema are settled.

## User Experience

ROBERT should support both:

```bash
robert run config.yaml
```

and a Python API:

```python
problem = RetrievalProblem.from_config("config.yaml")
result = DynestySampler().run(problem)
```

The CLI should be a wrapper around the same typed objects used by the Python API.
