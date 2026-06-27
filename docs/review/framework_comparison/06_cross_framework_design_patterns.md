# Cross-Framework Design Patterns

## Recurring Scientific Abstractions

| Pattern | Seen in | Maturity | ROBERT implication |
| --- | --- | --- | --- |
| Pressure grid plus layer grid | All frameworks | Very mature | Define `PressureGrid` and `LayerGrid` early, with explicit edge/center conventions. |
| Atmosphere state object | All, even when implicit | Very mature | Pass structured atmosphere state through physics; never raw config dicts. |
| Temperature parameterization | All retrieval frameworks | Very mature | Make parameterizations pure transforms from parameters to profiles. |
| Composition parameterization | All retrieval frameworks | Very mature | Support free constant/log profiles first; add equilibrium later as a backend. |
| Opacity provider / registry | TauREx, petitRADTRANS, PICASO, Exo_Skryer, POSEIDON | Mature | Make opacity coverage, species names, grids, and checksums queryable. |
| Contribution function / optical-depth terms | NEMESIS, TauREx, POSEIDON, pRT, PICASO, CHIMERA | Mature | Store per-term diagnostics for interpretability and validation. |
| Instrument response object | pRT, POSEIDON, TauREx, Exo_Skryer, PICASO | Mature | Separate model spectrum from observed/binned prediction. |
| Nuisance calibration parameters | POSEIDON, pRT, PICASO, Brewster, Exo_Skryer | Mature | Treat offsets, scale factors, and jitter as explicit parameter blocks. |
| Sampler adapter | TauREx, pRT, PICASO, CHIMERA, Exo_Skryer, Brewster | Mature | Keep samplers outside physics and likelihood internals. |

## Common Workflow Shape

Most frameworks converge on:

```text
config -> data loading -> opacity loading -> atmosphere parameterization
       -> radiative transfer -> instrument response -> likelihood -> sampler
       -> posterior products + spectra + diagnostics
```

ROBERT should make this workflow explicit in object names. The core likelihood
should be callable without a CLI, notebook, or sampler.

## Common API Designs

| API idea | Benefits | Failure mode | ROBERT decision |
| --- | --- | --- | --- |
| Config file drives run | Reproducible, scriptable | Raw config leaks into physics | Parse once into typed objects. |
| Python helper builds model | Interactive and notebook-friendly | Hidden defaults and mutable dicts | Keep helpers thin and validated. |
| Component registry | Extensible and discoverable | Opaque magic or naming collisions | Use simple registries with explicit aliases and citations. |
| Forward-model callable | Easy sampler integration | Can hide expensive mutable state | Make dependencies explicit and cache keys visible. |
| Dataset object | Multi-instrument retrievals become natural | Can absorb too many model concerns | Keep observed data and instrument response together, but not physics. |

## Mature Solutions Across Independent Projects

The ideas below recur often enough to treat as mature:

- Layer-centered pressure/temperature/composition arrays.
- Correlated-k as the default retrieval opacity mode.
- Optional high-resolution or opacity-sampling mode for special cases.
- CIA and Rayleigh as core continuum contributions.
- Separate nuisance calibration parameters per instrument or segment.
- Nested sampling as the default Bayesian retrieval workflow.
- Saving posterior samples plus best-fit spectra and profile envelopes.
- Small reference examples as a scientific regression suite.

## Repeated Sources of Technical Debt

- Historical file formats becoming internal APIs.
- Current-working-directory assumptions.
- Import-time environment mutation.
- Integer-coded physics modes without schema validation.
- Global caches with no run manifest.
- Sampler-specific code paths embedded in model construction.
- Notebooks or templates that users must edit for production runs.
- Optional data dependencies that fail before the relevant feature is used.

## Design Pattern to Standardize in ROBERT

Use this as the preferred dependency direction:

```text
io/config
  -> domain objects
    -> parameterizations
      -> opacity providers
        -> RT backend
          -> instrument response
            -> likelihood
              -> sampler adapter
```

Reverse imports should be prohibited. For example, `rt` should not import
`retrieval`, and `opacity` should not import a sampler.
