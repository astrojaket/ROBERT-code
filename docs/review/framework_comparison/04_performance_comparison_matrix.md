# Performance Comparison Matrix

| Framework | MPI / distributed | Threads / multiprocessing | GPU | JAX | Numba | Compiled kernels | Vectorisation | Caching / lazy loading | Memory strategy | Scalability lesson for ROBERT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEMESIS | HPC-style runs possible, not modern Python MPI | Compiler/runtime dependent | No | No | No | Fortran | Strong array loops | File/data reuse patterns | Static arrays and compiled dimensions | Scientific kernels can be fast, but compile-time limits and file-state coupling age poorly |
| NemesisPy | Some MPI-oriented scripts | Python multiprocessing possible | No clear stable support | No | Yes | Numba | Good in kernels | Partial | Mutable backend/cache patterns | Keep JIT kernels behind stable APIs, not as user-facing state machines |
| TauREx | Optional sampler-level parallelism | NumPy/Numba optional | No core GPU | No | Optional | Mostly Python/NumPy | Good | Opacity caches/class factories | Reasonable | Optimize by modular contribution assembly and optional JIT, not by making all users install HPC stacks |
| POSEIDON | MPI via `mpi4py`; PyMultiNest-centered | Forces BLAS/OpenMP thread limits at import | Experimental CuPy paths | No | Yes | Numba | Strong in kernels | Chemistry, stellar spectra, instrument precomputation | MPI shared-memory chemistry loading | Precompute aggressively, but environment and MPI choices must be explicit runtime options |
| petitRADTRANS | MPI in retrieval workflows | NumPy/Fortran | No general GPU | No | No | Fortran, Cython/Meson/f2py-style modules | Strong | Opacity loading and HDF5 tables | Warns about inefficient opacity modes; supports sampling/rebinning | Compiled RT kernels plus immutable-ish RT objects are a good production pattern |
| PICASO | Joblib for phase curves; some MPI retrieval helpers | Joblib, Numba | No | No | Yes | Numba | Strong | Reference data and opacity factories | xarray/reference data workflows | Numba is valuable for scattering/climate, but import-time data checks hurt usability |
| CHIMERA | Cluster-oriented PyMultiNest/dynesty templates | Multi-core notebooks/scripts | No | No | Yes | Numba | Medium/Strong | Precomputed CK/chemistry data | User-managed large files | Research templates scale scientifically but are hard to maintain as a library |
| Brewster | MPI through schwimmbad/mpi4py; cluster scripts | emcee walkers and MPI pool | No | No | No | Fortran via f2py | Fortran kernels | User-managed line lists/CIA/cloud files | Large external opacity/cloud data | Fortran kernels can remain valuable, but old dependency pins and f2py build complexity are a maintenance risk |
| Exo_Skryer | Sampler-dependent; not MPI-centric | JAX CPU thread configuration | CUDA and Metal paths configured | Yes | No | XLA/JAX kernels | Strong | Global opacity registries and master wavelength grids | JAX device arrays and registry caches | JAX is attractive, but backend selection and global registries need careful reproducibility controls |

## Performance Recommendations for ROBERT

1. Keep a readable NumPy reference implementation for every core equation.
2. Add Numba or compiled backends only behind a stable protocol.
3. Treat JAX/GPU as a future backend, not a core design assumption.
4. Precompute instrument response, opacity interpolation indices, and chemistry
   grids outside the likelihood call.
5. Make cache keys explicit: opacity checksum, wavelength grid, pressure grid,
   species list, RT mode, and backend version.
6. Avoid import-time changes to `OMP_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`,
   MPI communicators, or JAX platform settings.
7. Define benchmark cases early: clear emission, cloudy emission, multi-instrument
   JWST emission, and one larger opacity case.
