# Software Engineering Matrix

Legend for qualitative fields: `Strong`, `Medium`, `Weak`, or `Mixed`.

| Framework | Language / runtime | Layout | API cleanliness | Modularity | Physics/retrieval separation | Configuration philosophy | Plugin / registry pattern | Testing | Docs | CI / packaging | Type hints | Logging / errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEMESIS | Fortran | Large source plus examples | Weak by modern library standards | Medium scientific decomposition, weak state isolation | Medium conceptual separation, weak implementation separation | Run-name files and flags | No modern plugin layer | Example/regression style | Long-lived user knowledge plus examples | Make/build scripts | None | Fortran-era errors and file diagnostics |
| NemesisPy | Python, NumPy, Numba | Python package plus examples | Mixed | Medium | Medium in forward-model code, weak retrieval package | Dicts and scripts | Limited | Limited | Limited | Basic Python packaging | Sparse | Mixed |
| TauREx | Python | `src` package with domain subpackages | Strong | Strong | Strong | Config files mapped through class factories | Strong class/plugin discovery | Strong pytest suite | Strong ReadTheDocs | Strong CI, Poetry, nox, lint | Emerging/partial | Strong logging and validation |
| POSEIDON | Python, NumPy, Numba, MPI | Flat domain modules | Medium | Medium | Mixed; broad retrieval orchestration | Python dict builders | Limited | Medium pytest, data-dependent | Strong docs and notebooks | CI with conda, MPI, PyMultiNest | Sparse | Mixed; many exceptions/prints |
| petitRADTRANS | Python, Fortran, Cython/Meson | Mature package with retrieval subpackage | Strong for RT, medium for retrieval | Strong | Stronger than most, but spectral model can be broad | Object-based config | Limited formal plugins, strong opac/data objects | Strong reference tests | Strong ReadTheDocs | Strong packaging with Meson | Partial | Good validation, warnings |
| PICASO | Python, Numba | Package plus large reference data and notebooks | Medium | Medium | Mixed | Dict-like bundles and notebooks | Limited | Light pytest/notebook testing | Strong tutorials | Limited visible CI; modern pyproject | Sparse | Mixed; import-time env failures |
| CHIMERA | Python, Numba | Small package plus templates/notebooks | Weak/Medium | Medium kernels, weak workflow API | Weak; scripts own retrieval flow | Template scripts | No | Limited | Medium notebooks, sparse API docs | Setup exists, minimal CI evidence | None | Mostly prints/script failures |
| Brewster | Python, Fortran, f2py | Flat repository with templates, data, notebooks | Weak | Medium scientific functions, weak library boundaries | Weak; globals/templates | Python templates and globals | No | Manual check script and old pytest dependency | Basic README/templates | Makefile and pinned requirements | None | Mostly prints and hard-coded failures |
| Exo_Skryer | Python, JAX | Package with kernels, registries, samplers, experiments | Medium/Strong | Strong kernel decomposition | Stronger than many, CLI still orchestrates many steps | YAML schema plus CLI/web generator | Strong kernel registries | Light source tests, doctest config | Mixed; YAML docs strong, many stubs | Modern pyproject, ReadTheDocs config | Partial | Clear validation in many modules, many prints |

## Engineering Practices Worth Adopting

- TauREx: discoverable components with explicit input keywords, fit-parameter
  introspection, output/citation hooks.
- petitRADTRANS: reusable RT object boundary, dataset object with covariance,
  photometry, offset, and model-resolution concerns.
- Exo_Skryer: explicit YAML schema, kernel registries, sampler adapters with a
  common forward-model callable.
- POSEIDON: precompute expensive stellar/instrument/chemistry structures before
  the retrieval loop.
- PICASO: reference-data/version awareness and notebook-driven scientific
  tutorials.

## Engineering Practices to Avoid

- Import-time mutation of thread/MPI/GPU environment.
- Global runtime tuples such as `settings.runargs`.
- Hand-edited templates as the only production configuration interface.
- Monolithic `run_retrieval(...)` signatures with dozens of parameters.
- Raw dictionaries inside physics kernels.
- Failure modes that depend on current working directory or unvalidated external
  data paths.
