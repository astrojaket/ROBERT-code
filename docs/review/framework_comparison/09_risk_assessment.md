# Risk Assessment

| Risk | Likely source of complexity | Consequence | Mitigation | Priority |
| --- | --- | --- | --- | --- |
| Raw config leaks into physics | Convenience during early development | Hidden schemas, hard-to-test kernels | Parse config once into typed objects | High |
| Opacity coverage errors | Missing species/T/P/wavelength validation | Silent biased retrievals | Strict coverage checks and explicit extrapolation policy | High |
| Global cache state | Performance shortcuts | Cross-run contamination and irreproducibility | Cache keys with grid/species/checksum/backend metadata | High |
| Sampler coupling | Writing retrieval around first sampler | Hard to add or compare samplers | Stable `LogPosterior`/`RetrievalProblem` interface | High |
| Multi-instrument nuisance parameters | Offsets, jitter, scale factors | Parameter naming confusion and bad likelihoods | Dedicated calibration parameter block | High |
| Clouds as feature zoo | Many parameterizations and opacities | Unvalidated degeneracies and code sprawl | Start with one gray/cloud-deck model and diagnostics | High |
| Equilibrium chemistry backend | External grids and solvers | Dependency burden and hidden assumptions | Add after free chemistry is stable; save chemistry metadata | Medium |
| Line-by-line / opacity sampling | Memory and runtime | Slow retrievals and complicated opac APIs | Keep correlated-k first; add OS/LBL as separate provider | Medium |
| JAX/GPU backend | Device state, compilation, dtype, determinism | Hard installs and non-reproducible behavior | Future backend only; same tests as NumPy backend | Medium |
| MPI/thread environment | Import-time settings | Surprising process behavior | Configure only inside CLI runtime and record in manifest | Medium |
| Multidimensional atmospheres | Geometry, parameter explosion | Huge validation burden | Implement column abstraction before 2D/3D retrieval | Medium |
| Stellar contamination | Coupled stellar/planet/instrument model | Degeneracies and extra data dependencies | Keep as optional model with explicit priors | Medium |
| Photochemistry | Stiff solvers and many external assumptions | Technical debt and validation gaps | External interface only after equilibrium chemistry is mature | Low/Medium |
| Notebook-first workflows | Fast examples become production API | Non-reproducible edits | Notebooks call package APIs and checked YAML configs | Medium |
| Historical file compatibility | Temptation to support every old format | Parser sprawl | Support import adapters, not internal legacy formats | Medium |
| Overbroad plugin system | Premature extensibility | Opaque control flow | Start with simple registries and explicit names | Medium |
| Documentation drift | Rapid science feature changes | Users rely on wrong assumptions | Generate config reference from schemas where possible | Medium |
| Insufficient reference tests | Physics changes go unnoticed | Scientific regressions | Maintain small golden cases from the beginning | High |

## Highest-Risk Features to Defer

1. Full 3D retrievals.
2. GPU/JAX backend.
3. Photochemistry coupling.
4. Polarisation.
5. Variational/Hamiltonian inference.
6. Broad plugin marketplace.
7. Web UI.

## Red Flags During Implementation

- A function accepts both raw config and physical arrays.
- A module changes environment variables at import.
- A sampler imports an opacity module directly.
- A test requires external opacity downloads to pass.
- A retrieval result cannot report the exact opacity files and code version used.
- A physics mode is selected by an undocumented integer.
- A cache cannot be cleared or keyed deterministically.

## Guardrails

- Every public run writes a manifest.
- Every parameterization declares required parameters and citations.
- Every opacity provider exposes coverage before evaluation.
- Every backend passes the same reference tests.
- Every optional dependency is isolated behind an adapter.
- Every new feature includes at least one tiny example and one unit test.
