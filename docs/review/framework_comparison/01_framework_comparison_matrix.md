# Framework Comparison Matrix

This comparison reviews architectural ideas from established atmospheric
retrieval and radiative-transfer projects. It is a design reference for ROBERT,
not a code migration plan. No source code was copied.

## Evidence Base

| Framework | Source inspected | Evidence level | Notes |
| --- | --- | --- | --- |
| NEMESIS | https://github.com/nemesiscode/radtrancode at `52c273f` | Direct source | Mature Fortran retrieval and radiative-transfer system. |
| NemesisPy | https://github.com/astrojaket/NemesisPy at `afe2bae` | Direct source | Python/Numba exoplanet adaptation of NEMESIS concepts. |
| TauREx | https://github.com/ucl-exoplanets/taurex3 at `3ad6518` | Direct source | Modern Python package with plugins, typed-ish factories, tests, docs, CI. |
| POSEIDON | https://github.com/MartianColonist/POSEIDON at `d163221` | Direct source | Broad exoplanet retrieval feature set, including multidimensional atmospheres. |
| petitRADTRANS | https://gitlab.com/mauricemolli/petitRADTRANS at `0d9303c` | Direct source | Production RT package with Fortran kernels and retrieval tooling. |
| PICASO | https://github.com/natashabatalha/picaso at `0369089` | Direct source | Reflected/emitted/transit spectra plus climate and phase-curve workflows. |
| CHIMERA | https://github.com/mrline/CHIMERA at `0081994` | Direct source | Research-oriented retrieval scripts with Numba kernels and strong examples. |
| Brewster | https://github.com/substellar/brewster at `c12d1fb` | Direct source | Brown-dwarf and giant-planet emission retrieval with Python plus Fortran. |
| Exo_Skryer | https://github.com/ELeeAstro/Exo_Skryer at `abde13e` | Direct source | JAX-accelerated retrieval code with YAML configs and many sampler adapters. |

## High-Level Matrix

| Framework | Scientific center | Package organisation | RT implementation | Retrieval implementation | Configuration | Strongest idea for ROBERT | Main caution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NEMESIS | Solar System and exoplanet spectra; optimal estimation heritage | Large Fortran source tree, examples, build/run-name file families | Mature line-by-line and correlated-k kernels, scattering/cloud support, geometry variants | Optimal estimation plus retrieval executables | Run-name sidecar files and integer flags | Keep diagnostic depth: averaging kernels, covariance, contribution functions, validated examples | Do not inherit file-driven internals, global common blocks, compile-time shape limits, or variant executables |
| NemesisPy | Exoplanet emission, transmission, phase curves | Python package with calculation, data, radiative-transfer, examples | Numba-oriented correlated-k/opacity workflows under active evolution | Retrieval package is skeletal; samplers live mostly in scripts | Dicts and scripts | Preserve compact exoplanet workflows and Numba-friendly kernels | Avoid raw dicts as domain model and mutable backend call-order coupling |
| TauREx | General exoplanet forward models and retrievals | `src/taurex` split into model, chemistry, opacity, optimizer, plugins, instruments, output | Contribution pattern builds optical-depth terms and model outputs | Optimizer abstraction; Nestle, MultiNest, PolyChord variants | Config factory discovers classes by keywords | Adopt contribution/plugin ideas and fit-parameter introspection | Factory/mixin machinery can become too magical for a focused ROBERT MVP |
| POSEIDON | Transmission, emission, reflection, high-res, multidimensional atmospheres | Flat `POSEIDON` module set with domain files | Dedicated transmission/emission/reflection/high-res functions, Numba and optional GPU paths | PyMultiNest-centered `run_retrieval` with MPI, stellar contamination, high-res likelihoods | Python dict objects assembled by helper functions | Adopt serious instrument, stellar-contamination, multidimensional, and precomputation ideas | Avoid broad orchestration functions, import-time MPI/env mutation, and hidden global state |
| petitRADTRANS | Exoplanet spectral synthesis and retrievals | Mature package with `radtrans`, `spectral_model`, `retrieval`, `opacities`, Fortran source | `Radtrans` object wraps c-k/lbl opacity loading, clouds, CIA, Rayleigh, scattering | `RetrievalConfig`, `Data`, and sampler runner with MultiNest/UltraNest | Python object configuration | Adopt reusable RT engine boundary and dataset abstraction | Do not let one spectral-model class accumulate all workflow concerns |
| PICASO | Reflected light, thermal emission, transmission, 1D/3D climate | Package plus reference data, notebooks, docs | Toon/spherical-harmonics scattering, climate, phase curves, Virga clouds, Numba kernels | Dynesty/UltraNest/grid-analysis workflows, not as cleanly separated as RT | Dict-like input bundles, notebooks, reference data env vars | Adopt climate/opacity data curation lessons and 3D/phase geometry concepts | Avoid import-time hard reference-data failures and monolithic top-level functions |
| CHIMERA | HST/JWST emission/transmission retrieval prototypes | Small package plus `MASTER_ROUTINES` templates/notebooks and outputs | Correlated-k, Numba, two-stream scattering/reflection, Ackerman-Marley clouds | PyMultiNest and dynesty scripts, user-modifiable templates | Notebook/script templates | Adopt reference notebooks and explicit free vs equilibrium retrieval examples | Avoid notebook-as-primary API, committed generated outputs, and hand-edited scripts as production interface |
| Brewster | Brown-dwarf and giant-planet emission spectra | Flat repository with Python templates, Fortran kernels, data files, notebooks | Fortran forward model, DISORT-related files, CIA readers, cloud postprocessing | emcee and PyMultiNest templates, MPI via schwimmbad/mpi4py | Python template scripts | Adopt cloud/patch model expressiveness and brown-dwarf calibration workflow ideas | Avoid hard-coded file paths, pinned old dependencies, template mutation, and global `settings.runargs` |
| Exo_Skryer | JAX-accelerated substellar retrievals | Package with kernels, registries, samplers, experiments, docs, web config app | JAX kernels for emission/transit, OS/CK opacity, CIA, Rayleigh, clouds, 1.5D transit | Many sampler adapters: dynesty, UltraNest, Nautilus, PyMultiNest, JAXNS, BlackJAX, NumPyro | YAML reference plus CLI and Streamlit generator | Adopt kernel registry, explicit YAML schema, sampler adapter breadth, JAX performance boundary | Avoid premature sampler/backend sprawl and global opacity registries before core API stabilizes |

## Immediate Lessons

- The cleanest systems expose a small forward-model boundary, then adapt samplers
  around it.
- The most scientifically capable systems tend to become difficult when physics,
  data loading, likelihoods, and output writing are fused into one run function.
- Opacity data handling is not peripheral. Every successful code treats opacity
  loading, validation, grid coverage, and caching as core infrastructure.
- ROBERT should begin with JWST emission retrievals, but its abstractions should
  not bake in emission-only assumptions.
