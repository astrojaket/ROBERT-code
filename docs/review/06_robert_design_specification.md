# ROBERT Design Specification

This proposal follows the repository and science reviews of:

- NEMESIS / Radtran: https://github.com/nemesiscode/radtrancode at commit `52c273f`.
- NemesisPy: https://github.com/astrojaket/NemesisPy at commit `afe2bae`.

## Design Priorities

In priority order:

1. Scientific correctness.
2. Modularity.
3. Maintainability.
4. Reproducibility.
5. Extensibility.
6. Performance.

Performance optimization must not compromise correctness or maintainability.

## Package Structure

Proposed mature package layout:

```text
src/robert_exoplanets/
  core/
    units.py
    arrays.py
    grids.py
    state.py
    results.py
  parameterizations/
    temperature.py
    composition.py
    clouds.py
    registry.py
  opacity/
    providers.py
    kta.py
    cia.py
    cache.py
    coverage.py
  rt/
    interfaces.py
    layers.py
    planck.py
    correlated_k.py
    emission.py
    transmission.py
    scattering.py
    backends/
      numpy.py
      numba.py
  instruments/
    observations.py
    response.py
    jwst.py
    calibration.py
  likelihoods/
    gaussian.py
    covariance.py
    multi_dataset.py
  retrieval/
    problem.py
    priors.py
    transforms.py
    optimal_estimation.py
    samplers/
      base.py
      dynesty.py
      pymultinest.py
  io/
    config.py
    legacy_nemesis.py
    tables.py
    manifests.py
  validation/
    fixtures.py
    comparisons.py
  benchmarks/
    cases.py
```

The current skeleton can evolve toward this structure gradually. Do not create empty packages until there is a real use.

## Module Boundaries

### `core`

Owns domain objects and array conventions.

Rules:

- No sampler imports.
- No plotting.
- No file system side effects.
- Units and dimension order documented here.

### `parameterizations`

Maps retrieval parameters to physical profiles.

Rules:

- Pure transforms where possible.
- No opacity reading.
- No likelihood evaluation.
- Every parameterization declares required parameters, bounds/support, and citations.

### `opacity`

Owns opacity assets and interpolation data.

Rules:

- File formats are isolated here.
- Opacity coverage is queryable.
- No retrieval-loop logic.
- Cache keys are explicit.

### `rt`

Owns radiative-transfer kernels.

Rules:

- Accept structured arrays/state.
- Return structured spectra/diagnostics.
- No user config dict access.
- No observation file loading.
- Backends share the same interface.

### `instruments`

Owns observations, binning, convolution, offsets, jitter, and JWST-specific response handling.

Rules:

- Convert model spectra to observed quantities.
- Keep instrument response separate from atmospheric physics.

### `likelihoods`

Owns statistical comparison between model and data.

Rules:

- No sampler-specific code.
- Supports independent, block-diagonal, and full covariance cases.
- Noise inflation and offsets are explicit calibration parameters.

### `retrieval`

Owns retrieval algorithms and sampler adapters.

Rules:

- Depends on forward model and likelihood interfaces.
- Does not know opacity file formats.
- Sampler adapters are thin.

### `io`

Owns user config, legacy imports, run manifests, and serialization.

Rules:

- Raw user dictionaries stop here.
- Convert to typed objects before physics.
- Legacy NEMESIS file readers are adapters, not internal APIs.

## API Philosophy

### Explicit Over Implicit

Prefer:

```python
problem = RetrievalProblem(
    atmosphere_model=atmosphere_model,
    forward_model=forward_model,
    datasets=[miri_dataset, nirspec_dataset],
    likelihood=GaussianLikelihood(),
    parameters=parameter_space,
)
```

over:

```python
run("wasp43b")
```

The second can exist as a CLI wrapper, but not as the core API.

### Stable Small Interfaces

Core protocols:

```python
class Parameterization:
    def evaluate(self, parameters, grid): ...

class OpacityProvider:
    def opacity(self, atmosphere, spectral_grid): ...

class RadiativeTransferBackend:
    def emission(self, atmosphere, opacity, geometry): ...

class InstrumentResponse:
    def observe(self, spectrum): ...

class Likelihood:
    def loglike(self, model, data, parameters): ...
```

### No Raw Dicts in Physics

Configs are parsed at the boundary:

```text
YAML/TOML/Python config -> schema validation -> typed ROBERT objects -> physics
```

### Named Dimensions

ROBERT should standardize dimensions:

- pressure/profile level: `level`
- radiative-transfer layer: `layer`
- wavelength/wavenumber: `spectral`
- k ordinate: `g`
- gas species: `species`
- observation bin: `bin`
- dataset/order: `dataset`

Plain NumPy arrays can be used in kernels, but public objects should expose dimension names.

## Object Model

Key objects:

- `Planet`: radius, mass, gravity, reference pressure/radius.
- `Star`: radius, spectrum provider, contamination model.
- `PressureGrid`: ordered pressure levels and unit.
- `AtmosphereProfile`: pressure, temperature, VMR, optional clouds, derived height.
- `LayerGrid`: layer pressure, temperature, absorber amount, path scale.
- `OpacitySet`: gas, CIA, Rayleigh, cloud opacities on a spectral grid.
- `Spectrum`: spectral grid, values, unit, metadata.
- `Observation`: wavelengths, flux, uncertainty/covariance, instrument response.
- `Dataset`: one or more observations sharing calibration assumptions.
- `ParameterSpace`: named parameters and transforms.
- `RetrievalProblem`: immutable bundle of model, data, priors, likelihood.
- `RetrievalResult`: samples/OE solution, diagnostics, manifest.

## Data Model

### Internal Units

Suggested internal units:

- pressure: Pa,
- temperature: K,
- length: m,
- wavelength: um at public boundary, converted internally as needed,
- wavenumber: cm^-1 only where opacity formats require it,
- radiance/flux: explicit unit attached to `Spectrum`,
- VMR: dimensionless.

### Orientation

Choose one canonical vertical orientation. Recommended:

- profile arrays ordered top-to-bottom in pressure increasing downward, or bottom-to-top, but never implicit.

Because many legacy routines reverse arrays, ROBERT must make orientation explicit in object constructors and tests.

### Run Manifest

Every real retrieval should write a manifest with:

- ROBERT version and git commit,
- config hash,
- opacity checksums,
- data checksums,
- sampler/OE settings,
- random seeds,
- backend name and versions,
- hardware summary if benchmarking.

## Dependency Direction

Allowed:

```text
io -> core
io -> opacity/instruments/retrieval construction
parameterizations -> core
opacity -> core
rt -> core + opacity
instruments -> core
likelihoods -> core + instruments
retrieval -> core + likelihoods + rt interfaces
examples/cli -> all public APIs
```

Forbidden:

- `core` importing `retrieval`, `io`, plotting, MPI, or sampler packages.
- `rt` importing user config loaders.
- `opacity` importing samplers.
- `likelihoods` importing concrete sampler libraries.
- import-time MPI initialization.

## Testing Philosophy

### Required Test Tiers

1. Unit tests for parameter transforms, grids, units, Planck, MMW, interpolation.
2. Golden tests against small NEMESIS/NemesisPy reference cases.
3. Property tests for monotonic grids, nonnegative opacities, VMR normalization, finite spectra.
4. Integration tests for one emission and one transmission toy retrieval.
5. Slow validation tests gated separately from the default suite.

### Scientific Test Policy

Every ported scientific component needs:

- a small analytic or hand-checkable test,
- at least one legacy comparison fixture,
- documented tolerance and reason for tolerance.

## Benchmarking Philosophy

Benchmark correctness first, speed second.

Benchmark groups:

- micro: Planck, k interpolation, random overlap, CIA, layer averaging;
- forward: point emission, disc emission, transmission;
- retrieval: fixed small likelihood with fixed sampler settings.

Benchmark metadata must include backend, array order, dtypes, CPU/GPU, and dependency versions.

## Documentation Philosophy

Documentation should include:

- scientific equations and references,
- API examples,
- unit conventions,
- validation status,
- performance notes,
- migration notes for NEMESIS/NemesisPy users.

Each parameterization and opacity provider should state:

- scientific source,
- valid range,
- tested range,
- known limitations,
- whether it is production-ready or experimental.

## Initial ROBERT Scope Recommendation

Do not attempt the full NEMESIS feature set first.

Recommended first scientifically meaningful scope:

- 1D plane-parallel thermal emission.
- Correlated-k gas opacity.
- CIA for H2-H2 and H2-He at minimum.
- Cloud-free atmosphere, no scattering source function.
- Simple clouds only after cloud-free validation.
- JWST-like binned emission observation.
- Gaussian likelihood.
- One sampler adapter plus deterministic forward tests.

Everything else should be staged behind the roadmap.
