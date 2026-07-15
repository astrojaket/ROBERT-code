# Interface Specification

This document specifies the stable interfaces ROBERT components must implement.
The examples are protocol-shaped pseudocode, not production code.

## 1. General Interface Rules

- Inputs and outputs are typed domain objects.
- Raw user config does not enter component interfaces.
- Components declare required parameters, units, citations, and validity range.
- Implementations may be classes or functions, but public behavior must match
  the protocol.
- Optional backends must pass the same reference tests as built-ins.

## 2. Radiative Transfer Engines

Purpose:

- Convert atmosphere and opacity state into a native model spectrum.

Interface:

```python
class RadiativeTransferBackend:
    name: str

    def prepare(self, grid, options):
        ...

    def emission(self, atmosphere, opacity, geometry):
        ...

    def transmission(self, atmosphere, opacity, geometry):
        ...

    def diagnostics(self):
        ...
```

Inputs:

- `AtmosphereState`.
- `PreparedOpacity` or evaluated opacity state.
- Geometry/boundary condition object.

Outputs:

- `Spectrum`.
- RT diagnostics such as contribution functions, effective radii, and
  atmospheric-annulus area contributions.

Invariants:

- No observation file access.
- No sampler imports.
- Output grid and units are explicit.
- Invalid physical states fail deterministically.

## 3. Chemistry Engines

Purpose:

- Produce atmospheric composition from parameters and T-P structure.

Interface:

```python
class ChemistryModel:
    name: str
    species: tuple[str, ...]

    def required_parameters(self):
        ...

    def prepare(self, context):
        ...

    def evaluate(self, parameters, pressure_grid, temperature):
        ...
```

Inputs:

- Parameter values.
- Pressure grid.
- Temperature profile.
- Optional prepared backend data.

Outputs:

- Composition profiles.
- Optional chemistry diagnostics.

Invariants:

- Composition convention is explicit.
- Species names map cleanly to opacity species.
- Backend grid coverage is validated before sampling.

## 4. Opacity Providers

Purpose:

- Prepare and evaluate opacity contributions.

Interface:

```python
class OpacityProvider:
    name: str

    def inspect(self):
        ...

    def prepare(self, spectral_grid, pressure_grid, species):
        ...

    def coverage(self, atmosphere):
        ...

    def evaluate(self, atmosphere):
        ...
```

Inputs:

- Opacity database metadata.
- Spectral and pressure grids.
- Species list.
- Atmosphere state.

Outputs:

- `PreparedOpacity`.
- Evaluated opacity arrays.
- `CoverageReport`.

Invariants:

- No silent extrapolation.
- Cache key includes file checksums, species, grids, backend, and mode.
- Prepared state is immutable during retrieval.

## 5. Cloud Models

Purpose:

- Represent cloud vertical structure and cloud optical properties.

Interface:

```python
class CloudModel:
    name: str

    def required_parameters(self):
        ...

    def vertical_state(self, parameters, atmosphere):
        ...

    def optical_properties(self, cloud_state, spectral_grid):
        ...
```

Inputs:

- Parameter values.
- Atmosphere state.
- Spectral grid for optical properties.

Outputs:

- Cloud vertical profile.
- Cloud extinction/scattering properties.

Invariants:

- Cloud quantities are non-negative where physically required.
- Patch/column mixing weights sum under declared convention.
- Citations and assumptions are exposed.

## 6. Instrument Models

Purpose:

- Map native model spectra to observations.

Interface:

```python
class InstrumentResponse:
    name: str

    def prepare(self, observation):
        ...

    def observe(self, spectrum):
        ...
```

Inputs:

- Native `Spectrum`.
- `Observation`.
- Response/binning metadata.

Outputs:

- Binned `Spectrum` or `ModelPrediction`.

Invariants:

- Does not evaluate atmospheric physics.
- Response weights and integration convention are documented.
- Supports masks and segment labels.

## 7. Noise and Calibration Models

Purpose:

- Model nuisance terms such as offsets, scales, jitter, and covariance.

Interface:

```python
class NoiseModel:
    name: str

    def required_parameters(self):
        ...

    def apply(self, prediction, observation, parameters):
        ...

    def covariance(self, observation, parameters):
        ...
```

Inputs:

- Prediction.
- Observation.
- Calibration/noise parameters.

Outputs:

- Adjusted prediction or data.
- Effective uncertainty/covariance.

Invariants:

- Units are explicit.
- Segment/group mapping is validated.
- Noise inflation cannot silently make invalid uncertainties.

## 8. Likelihoods

Purpose:

- Compute statistical fit quality.

Interface:

```python
class Likelihood:
    name: str

    def loglike(self, prediction, observation, parameters):
        ...
```

Inputs:

- Model prediction.
- Observation.
- Parameters, including nuisance parameters.

Outputs:

- Scalar log likelihood.

Invariants:

- Masks are honored.
- Non-finite predictions are handled consistently.
- Covariance likelihoods validate positive definiteness.

## 9. Prior Distributions

Purpose:

- Define parameter probability models and sampler transforms.

Interface:

```python
class Prior:
    name: str

    def log_prob(self, value):
        ...

    def transform(self, unit_value):
        ...
```

Inputs:

- Physical value.
- Unit-cube value for nested samplers.

Outputs:

- Log probability.
- Physical value from unit cube.

Invariants:

- Support is explicit.
- Transform and log-probability agree.
- Invalid values return `-inf` or raise at validation time depending on context.

## 10. Samplers

Purpose:

- Run an inference backend against `RetrievalProblem`.

Interface:

```python
class SamplerAdapter:
    name: str

    def run(self, problem, output):
        ...

    def resume(self, problem, output):
        ...
```

Inputs:

- `RetrievalProblem`.
- Output/checkpoint configuration.

Outputs:

- `RetrievalResult`.

Invariants:

- Sampler does not import opacity or RT internals.
- Sampler records random seed, settings, backend version, and resume state.
- Sampler raw outputs are wrapped in ROBERT result schema.

## 11. Forward Model

Purpose:

- Compose atmosphere, opacity, RT, and instrument response.

Interface:

```python
class ForwardModel:
    def predict(self, parameters):
        ...
```

Inputs:

- Physical parameter mapping.

Outputs:

- `ModelPrediction` containing observed-grid prediction, optional native
  spectrum, and diagnostics.

Invariants:

- Deterministic for fixed parameters and prepared state.
- Does not mutate observations or prepared opacity.

## 12. Plugin Metadata

Every plugin must expose:

```python
class PluginMetadata:
    name: str
    version: str
    component_type: str
    robert_requires: str
    citations: tuple[str, ...]
    optional_dependencies: tuple[str, ...]
```

Rules:

- Plugin names are stable and lowercase.
- Built-in names are reserved.
- Plugins provide config schema fragments when configurable.
