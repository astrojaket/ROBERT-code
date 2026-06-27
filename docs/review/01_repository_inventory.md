# Repository Inventory

Sources inspected on 2026-06-26:

- NEMESIS / Radtran: https://github.com/nemesiscode/radtrancode at commit `52c273f`.
- NemesisPy: https://github.com/astrojaket/NemesisPy at commit `afe2bae`.

This inventory is intended to guide ROBERT design. It is not a claim that ROBERT should reproduce either repository's architecture.

## NEMESIS / Radtran

### Repository Character

NEMESIS is a Fortran application suite plus data archive, examples, manuals, and build scripts. The root repository combines:

- numerical recipes and matrix utilities,
- the Radtran radiative-transfer libraries,
- NEMESIS retrieval drivers,
- reference gas, CIA, solar, stellar, continuum, and planetary data,
- example calculation directories with complete input/output file sets,
- manuals and workshop material.

The code is file-format and executable driven. A retrieval is primarily a run-name plus a family of sidecar files (`.inp`, `.ref`, `.apr`, `.spx`, `.kls`, `.cia`, `.set`, `.fla`, `.xsc`, `.sca`, `.pat`, `.drv`, `.raw`, `.mre`, etc.).

### Top-Level Directory Structure

```text
radtrancode/
  Example_calculations/
    Direct_image_exoplanet/
    Jupiter_CIRS_nadir_thermal_emission/
    Scattering_giant_planet_visible_nearIR/
  FOVgreg/
  frecipes/
  idl/
  manuals/
    NEMCEE/
    NEMESIS4Newbies/
  nemesis/
    dev/
  nemesisPY/
  raddata/
  radtran/
    ciatable/
    cirsrad/
    cirsradg/
    includes/
    makefiles/
    matrices/
    monteck/
    multinest_mc/
    path/
    radtran/
    rtm_util/
    scatter/
    spec_data/
```

Observed file mix:

- about 843 `.f` Fortran files,
- many reference `.dat`, `.tab`, `.ref`, `.kta`, `.spx`, and run-control files,
- manuals in PDF/DOCX/TeX,
- legacy IDL helpers,
- Docker support and platform-specific build instructions.

### Major Subsystems

| Area | Main location | Responsibility |
| --- | --- | --- |
| Retrieval drivers | `nemesis/` | Executables such as `nemesis`, `nemesisL`, `nemesisdisc`, `nemesisPT`; setup, apriori reading, optimal-estimation loops, output writing. |
| State/profile mapping | `nemesis/subprofretg.f`, `readapriori.f`, `npvar.f` | Convert compact retrieval vector and parameterization IDs into full atmospheric profiles. |
| Forward-model orchestration | `nemesis/forward*.f`, `gsetrad*.f` | Build temporary profile/path/scattering files, call Radtran kernels, assemble spectra and Jacobians. |
| Gradient radiative transfer | `radtran/cirsradg/` | Correlated-k or LBL-table radiative transfer with gradients. |
| Non-gradient radiative transfer | `radtran/cirsrad/` | Thermal/scattering radiative transfer variants. |
| Path/layer geometry | `radtran/path/` | Atmospheric path construction, hydrostatic adjustment, layer integration, ray tracing. |
| Opacity processing | `radtran/radtran/`, `radtran/spec_data/`, `radtran/ciatable/` | K-table, LBL table, CIA, continuum, line database, and cross-section processing. |
| Scattering | `radtran/scatter/`, `radtran/monteck/` | Rayleigh, Mie/Henyey-Greenstein phase functions, Raman, single/multiple scattering, Monte Carlo utilities. |
| Matrix math | `radtran/matrices/`, `frecipes/` | Linear algebra, matrix inversion, numerical recipes. |
| Reference data | `raddata/` | Gas metadata, CIA tables, stellar/solar spectra, continuum and cross-section references. |

### Dependency Graph

High-level dependency direction:

```text
nemesis executables
  -> nemesis retrieval/setup/profile routines
    -> radtran path/layer routines
    -> radtran cirsrad/cirsradg kernels
      -> radtran opacity, CIA, scattering, Planck, interpolation
    -> radtran matrix utilities
  -> raddata and run-directory files
```

This direction is only conceptual. In practice, common blocks, include files, generated sidecar files, and shared unit numbers create hidden bidirectional coupling.

### Main Retrieval Data Flow

1. `nemesis.f` prompts for a run name.
2. `checkfiles` validates the run-name sidecar file set.
3. Header files are read: reference atmosphere, input settings, k-table/LBL mode, solar/cell files, spectra, previous retrieval files.
4. Wavelength grids are assembled by `wavesetb` or `wavesetc`.
5. `readapriori` constructs the state vector `xa`, covariance `sa`, variable descriptors `varident`, `varparam`, and log flags.
6. `coreret` runs the optimal-estimation retrieval.
7. The forward model writes/updates `.pat`, `.prf`, `.xsc`, `.sca`, and related files via `gsetrad`/`subprofretg`.
8. `cirsrtf_wave` or `cirsrtfg_wave` reads paths/opacities, calls radiative-transfer kernels, convolves spectra, and maps gradients back to state-vector elements.
9. `coreret` updates the state vector, covariance, averaging kernels, gain matrix, and convergence state.
10. `writeout`, `writeraw`, and `write_covariance` emit `.mre`, `.raw`, covariance, and diagnostic products.

### Main Forward-Model Call Graph

Representative thermal/correlated-k gradient path:

```text
nemesis.f
  -> coreret.f
    -> forwardavfovX.f or forwardnogX.f
      -> setup.f
      -> gsetrad.f
        -> subprofretg.f
        -> gwritepat.f / subpath.f / scattering file updates
      -> cirsrtfg_wave.f or cirsrtf_wave.f
        -> subpathg.f
        -> read_klist.f / read_klbllist.f
        -> get_scatter.f / get_xsec.f
        -> cirsradg_wave.f
          -> get_kg.f / get_klblg.f
          -> ngascon.f / nciacon.f / nparacon_all.f
          -> noverlapg.f / noverlapg1.f
          -> get_hg.f / Rayleigh/scattering helpers
          -> planck_wave.f / planckg_wave.f
        -> map2pro.f
        -> map2xvec.f
        -> cirsconv.f or lblconv1.f
```

Finite-difference/scattering path:

```text
coreret.f
  -> forwardnogX.f
    -> gsetrad.f
    -> cirsrtf_wave.f
    -> perturb state vector elements
    -> repeat forward calculations for K matrix
```

### Main Retrieval Loop Call Graph

```text
nemesis.f
  -> readapriori.f
  -> coreret.f
    -> dinvertm.f on Sa and Se
    -> setifix.f
    -> initial forward model and K matrix
    -> calc_gain_matrix.f
    -> calc_phiret.f
    -> assess.f
    -> for iter in 1..kiter
      -> calcnextxn.f
      -> embedded support checks on state vector/profile validity
      -> trial forward model and K matrix
      -> calc_phiret.f
      -> Marquardt brake update via `alambda`
      -> accept/reject iteration
      -> calc_gain_matrix.f
      -> optional reduced-wavelength refinement
    -> calc_serr.f
  -> writeout.f / writeraw.f / write_covariance.f
```

### Major NEMESIS Strengths

- Decades of validated scientific behavior.
- Explicit support for many Solar System geometries and instrument modes.
- Mature correlated-k/LBL table handling.
- Mature optimal-estimation machinery with Jacobian/covariance products.
- Broad CIA, continuum, scattering, and path-geometry coverage.
- Complete run examples with reference outputs.

### Major NEMESIS Risks for ROBERT

- Global state through common blocks and include files.
- Compile-time array limits.
- Heavy coupling to sidecar files and current working directory.
- Scientific constraints embedded inside retrieval control flow.
- Multiple near-duplicate executables for geometry variants.
- Hard-to-test routines with implicit I/O and unit-number side effects.
- Historical build complexity and platform-specific behavior.

## NemesisPy

### Repository Character

NemesisPy is a Python package focused on exoplanet spectra, especially emission and phase curves. It contains two layers:

- an older `nemesispy.radtran` package that ports key radiative-transfer routines and uses Numba for speed,
- a newer `nemesispy.modular` layer that wraps the backend with config-driven temperature, chemistry, cloud, observation, stellar, and k-table handling.

The repository also includes package build artifacts (`build/`, `dist/`, `.egg-info`), `.pyc` caches, and `__MACOSX` resource-fork files. Those should not be copied into ROBERT.

### Directory Structure

```text
NemesisPy/
  nemesispy/
    common/
    data/
      cia/
      gcm/
      stellar/
      xsecs/
    models/
    modular/
      engines/
      io/
      rt/
    radtran/
    retrieval/
  build/
  dist/
  nemesispy.egg-info/
```

### Major Python Modules and Classes

| Area | Main files | Responsibility |
| --- | --- | --- |
| Backend forward model | `radtran/forward_model.py` | Mutable `ForwardModel` class storing planet/opacity data and exposing point, disc, transmission, weighting, and cloud spectrum methods. |
| Radiance kernels | `radtran/calc_radiance.py` | Numba JIT emission/transmission/weighting functions. |
| Gas optical depth | `radtran/calc_tau_gas.py` | K-table interpolation and random-overlap correlated-k combination. |
| CIA | `radtran/calc_tau_cia.py` | CIA optical-depth calculation for H2-H2, H2-He, N2, CH4 pairs, plus partial H-minus support. |
| Rayleigh/cloud optical depth | `radtran/calc_tau_rayleigh.py`, `calc_tau_cloud.py` | Continuum/scattering optical-depth approximations. |
| Layering/hydrostatics | `radtran/calc_layer.py`, `common/calc_hydrostat.py` | Pressure/height layers, absorber-amount weighted layer averaging, hydrostatic altitude. |
| Model profiles | `models/TP_profiles.py`, `models/gas_profiles.py`, `models/tmap_*` | Temperature, gas, and longitudinal/latitudinal temperature map helpers. |
| Modular setup | `modular/rt/setup.py` | Dict-config pressure grid, planet structure, prior extraction. |
| Modular forward wrappers | `modular/rt/forward_model.py`, `emission_forward.py` | Transmission/emission wrappers around backend `ForwardModel`. |
| Modular engines | `modular/engines/*.py` | Temperature, chemistry, cloud, FastChem, stellar contamination/TLSE logic. |
| Modular I/O | `modular/io/*.py` | Config loading, observation loading, stellar spectra, k-table fetching/rebinning. |
| Retrieval package | `retrieval/transm_retr.py` | Empty in inspected commit. Retrieval examples live elsewhere. |

### Dependency Graph

Observed internal package-level edges:

```text
nemesispy.radtran -> nemesispy.common
nemesispy.radtran -> nemesispy.radtran
nemesispy.modular -> nemesispy.common
nemesispy.modular -> nemesispy.models
nemesispy.modular -> nemesispy.radtran
nemesispy.modular -> nemesispy.modular
nemesispy.data -> nemesispy.common/models/radtran/data
nemesispy.models -> nemesispy.common/models
```

Core direction:

```text
modular.io/config/setup
  -> modular.engines
    -> radtran.forward_model
      -> radtran.calc_layer
      -> radtran.calc_radiance
        -> calc_tau_gas / calc_tau_cia / calc_tau_rayleigh / calc_tau_cloud
```

Concern: `data/` imports executable modeling code for plotting/fitting examples. ROBERT should keep examples and datasets outside the importable scientific core.

### NemesisPy Forward-Model Data Flow

Older backend flow:

1. Instantiate `nemesispy.radtran.forward_model.ForwardModel`.
2. Call `set_planet_model` with mass, radius, gas IDs, isotope IDs, and layer count.
3. Call `set_opacity_data` to read k-tables and CIA tables.
4. Provide pressure, temperature, VMR, optional height model, path angle, stellar spectrum.
5. Compute mean molecular weight and hydrostatic height.
6. Split model into layers with `calc_layer` or `calc_layer_transm`.
7. Compute radiance/transmission through Numba kernels.

Modular transmission flow:

1. `build_model_structure(config)` builds pressure grid, planet values, and gas list.
2. `ForwardModel(model_structure, config, kta_paths, cia_path)` creates temperature, chemistry, and cloud engines.
3. `evaluate(params, observations)` builds T(P), VMR(P), backend planet model, and delegates to `CloudEngine`.
4. `CloudEngine` calls backend clear/cloudy transmission routines and applies offsets/TLSE.

Modular emission flow:

1. Build wavelength grid and prepare k-tables.
2. `EmissionForward` builds chemistry/temperature engines and backend `ForwardModel`.
3. `spectrum(params, nmu, units)` evaluates T(P), VMR(P), calls `calc_disc_spectrum_uniform`, and normalizes to flux ratio or ppm.

### NemesisPy Forward-Model Call Graph

```text
modular.rt.emission_forward.EmissionForward.spectrum
  -> TemperatureEngine.evaluate
  -> ChemistryEngine.vmr_matrix
  -> radtran.forward_model.ForwardModel.calc_disc_spectrum_uniform
    -> calc_mmw
    -> calc_hydrostat
    -> calc_layer
    -> calc_radiance
      -> calc_tau_cia
      -> calc_tau_rayleigh
      -> calc_tau_gas
        -> interp_k
        -> noverlapg
          -> rank
      -> calc_planck
```

```text
modular.rt.forward_model.ForwardModel.evaluate
  -> TemperatureEngine.evaluate
  -> ChemistryEngine.vmr_matrix
  -> CloudEngine.compute_fluxes
    -> backend ForwardModel.calc_transm_spectrum
      -> calc_hydrostat_pref_test
      -> calc_layer_transm
      -> calc_transm
        -> calc_tau_cia
        -> calc_tau_rayleigh
        -> calc_tau_gas
        -> calc_tau_cloud
```

### NemesisPy Retrieval Loop Inventory

The inspected `nemesispy/retrieval/transm_retr.py` is empty. Retrieval behavior is demonstrated in scripts such as `nemesispy/data/gcm/fit_gcm/mpi_trial.py`, where `pymultinest.run` is called directly from a data/demo script with inline `Prior` and `LogLikelihood` functions.

This is useful historical evidence but not a reusable architecture. ROBERT should define retrieval as:

```text
ParameterSpace -> PriorTransform -> ForwardModel -> Likelihood -> SamplerAdapter -> Result
```

rather than as scripts beside specific datasets.

### Major NemesisPy Strengths

- Numba-accelerated correlated-k kernels with clear numerical loops.
- Exoplanet-specific emission, transmission, disc integration, and phase-curve direction.
- Modular engines for temperature, chemistry, clouds, stellar corrections, and k-table preparation.
- Practical JWST/multi-observation concerns: offsets, bin widths, stellar contamination, k-table rebinning.
- More inspectable than Fortran for modern Python development.

### Major NemesisPy Risks for ROBERT

- Mutable backend object with many optional attributes.
- Dict-driven configs without typed schema validation.
- Multiple old/backup/test variants in production package.
- Build artifacts and caches committed to source tree.
- I/O, plotting, MPI, and scientific calculations interleaved in several places.
- Empty retrieval package despite retrieval-related examples.
- Optional dependencies not represented cleanly in package metadata (`exo_k`, `h5py`, `mpi4py`, FastChem/pysynphot paths).
- Some scientific TODOs and comments indicate incomplete components.

