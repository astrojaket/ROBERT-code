# Full Code and Cross-Framework Audit

Date: 2026-07-13

## Executive verdict

ROBERT now has a coherent, tested validation-retrieval stack rather than an
architecture skeleton. Clear thermal emission and absorption-dominated
transmission reproduce the stable petitRADTRANS 3.3.3 reference closely when
the opacity or optical-depth boundary is controlled. The P3/SH4 cloudy-emission
solver reproduces PICASO to numerical precision in matched grey cases and to
better than 0.004% in a realistic matched molecular case. The production
MgSiO3/Mie/SH4 retrieval path is internally consistent and runs in about one
second per four-dataset call on the audit laptop.

The package is ready for validation retrievals. It is not yet a production
science model for arbitrary clouds. A subsequent benchmark now independently
reproduces an end-to-end cloudy spectrum from one physical MgSiO3 contract in
ROBERT and PICASO/Virga; see review 35. Its gas opacity is an analytic
validation fixture, so an independent science molecular-opacity comparison and
a cloudy-transmission scattering reference remain open. These boundaries must
remain explicit in release claims.

## Audited baseline and environment

The outstanding configuration, retrieval, catalogue, documentation, and test
work was committed directly to `main` as `fb9879a` because the repository was
already on `main`; no separate branch existed to merge. This audit then found
and repaired one example-level regression in the WASP-69b multi-instrument
benchmark and added a regression test for it.

All Python tests, linters, imports, and benchmarks in this audit ran through:

```text
/Users/jaketaylor/miniforge3/envs/robert-exoplanets
```

The measured runtime was macOS arm64 on an Apple M1 Pro with Python 3.12.13,
NumPy 2.2.6, SciPy 1.18.0, Numba 0.61.2, and Pydantic 2.13.4. External
comparisons ran in their isolated pRT and PICASO environments, but the ROBERT
side and every repository test or benchmark ran from the requested environment.

## Verification summary

| Check | Result |
| --- | --- |
| Pytest | 335 passed; 14 third-party Matplotlib/PyParsing warnings |
| Branch coverage | 74%, above the configured 70% gate |
| Ruff lint | Passed |
| Ruff formatting of changed Python files | Passed |
| Package compile/import | Passed |
| Public API exports | 216 unique exports; all resolve |
| Dependency consistency | `pip check` passed |
| Benchmark programs | 21 of 21 completed |
| Source/test inventory | 81 package modules and 54 test modules |

Repository-wide formatting was inspected but not applied: 134 older files use
the repository's previous formatting and would be mechanically rewritten.
Mixing that unrelated churn into this scientific audit would obscure review.

The lowest measured branch coverage remains concentrated in orchestration and
specialized numerical paths: `io/configured_tasks.py` at 22%,
`rt/random_overlap.py` at 32%, and `rt/sh4.py` at 66%. The aggregate gate is
healthy, but these modules deserve targeted behavioral and failure-path tests.

## Completed capability audit

The following capabilities are implemented, connected, and exercised:

- typed pressure/spectral grids, bodies, observations, spectra, and atmosphere
  state;
- isothermal, spline, tabulated, Madhusudhan-Seager, and
  Parmentier-Guillot-style temperature profiles;
- free and FastChem chemistry, mean molecular weight, and retrieval-time
  temperature/chemistry evaluation;
- `.kta`, ROBERT archive, and native pRT HDF correlated-k opacity preparation;
- multi-gas random-overlap resort/rebin with NumPy, Numba, and optional JAX
  implementations;
- CIA and Rayleigh optical depths plus diagnostic decomposition;
- clear thermal emission and spherical-shell transmission solvers;
- Toon two-stream and P3/SH4 thermal multiple scattering, exact Mie moments,
  delta-M scaling, and spectrum-only accelerated execution;
- shared-atmosphere multi-dataset forward models, instrument response,
  offsets, jitter, masks, Gaussian likelihoods, optimal estimation, and an
  optional UltraNest adapter;
- strict versioned YAML retrieval configuration, manifests, deterministic
  injection/recovery checks, and versioned JWST emission catalogues;
- runnable HAT-P-32b, WASP-69b, and WASP-80b validation/retrieval examples.

No obvious forbidden reverse imports or import cycles were found. The public
API surface resolves cleanly. The architecture remains modular despite the
substantial physics growth, although the version and roadmap have not caught
up with the implementation: package version 0.3.0 now contains much of the
roadmap's v0.4-v0.10 scope. Release gates and versioning should be rebaselined
before making broader stability claims.

## petitRADTRANS 3.3.3 validation

Fresh pRT 3.3.3 references were generated from the local pRT opacity database.
Stable pRT 3 is the release reference because the tested pRT 4.0.0 beta showed
large, layer-count-dependent emission changes in its own output. Its
transmission remained consistent, but its beta emission result is not suitable
as a scientific acceptance target.

### Final emission and transmission spectra

| Matched case | ROBERT versus pRT result | Interpretation |
| --- | --- | --- |
| H2O+CIA emission, identical evaluated optical depth | median -0.0324%; RMS 0.0733%; max 0.208% | thermal integration agrees; pRT 80-to-160-layer self-change is up to 0.0888% |
| H2O+CIA transmission, identical evaluated optical depth | median +1.736 ppm; RMS 3.898 ppm; max 10.463 ppm | final spherical transmission is consistent |
| Six-gas+CIA+Rayleigh transmission, shared Rayleigh | RMS 2.569 ppm; max 9.805 ppm | geometry and absorption boundary are consistent |
| Six-gas+CIA emission, shared species optical depth | median -0.0869%; RMS 0.140%; max 0.735% | final multi-gas thermal RT is consistent |
| Six-gas emission, independently recombined opacity | median -1.421%; RMS 1.980%; max 4.313% | remaining difference is primarily correlated-k combination/convention, not the thermal solver |

For the six-gas transmission case, ROBERT's native Rayleigh optical depth was
2.88% higher in the median than pRT's vertical optical depth. The maximum
native optical residual was 5.234 ppm, while including Rayleigh changed the
pRT spectrum by as much as 987.2 ppm. This is small but measurable and should
be closed at the cross-section/convention boundary.

The six-gas emission contribution-function centroid differs by 0.0856 dex RMS
in pressure. That is consistent with the small spectral difference at the
shared-optical-depth boundary and should be retained as a regression metric,
not only the final spectrum.

### Performance

Single-thread warmed timings are intentionally reported without claiming a
universal speed ratio:

| Case | pRT 3.3.3 | ROBERT |
| --- | ---: | ---: |
| H2O+CIA emission | 0.107 s | 0.632 s |
| H2O+CIA transmission | 0.111 s | 0.610 s |
| Six-gas emission | 0.473 s | 2.902 s |
| Six-gas transmission | 0.588 s | 2.963 s for the native case |

ROBERT currently prioritizes transparent, replaceable reference physics and
retrieval orchestration over matching pRT's mature optimized kernels. The
timings identify RORR and SH4 batching as optimization targets; they do not
justify lowering spectral, angular, vertical, or correlated-k resolution.

## PICASO and cloud-scattering validation

PICASO comparisons used an external PICASO subprocess with the ROBERT side
executed in `robert-exoplanets`. Supplying identical layer optical depth,
single-scattering albedo, phase moments, source, and boundary conditions
isolates the multiple-scattering solver from cloud microphysics and opacity
database differences.

### Controlled grey thermal cases

| Case | ROBERT backend | Maximum disk-relative difference | Maximum point-angle difference |
| --- | --- | ---: | ---: |
| Isothermal absorption | clear reference | 4.78e-6 | negligible |
| Temperature-gradient absorption | Toon | 0.838% | 9.54% |
| `w=0.5`, `g=0` | Toon | 0.716% | 1.49% |
| `w=0.9`, `g=0` | Toon | 0.577% | 1.20% |
| `w=0.9`, `g=0.6` | Toon | 0.665% | 35.33% |
| Isotropic and forward scattering | matched SH4 | about 3.22e-6 | about 3.22e-6 |

The SH4 result demonstrates that the controlled cloudy thermal-RT boundary is
correct to numerical precision. Toon remains useful as a fast
disk-integrated approximation, but the 35% forward-scattering point-angle
error confirms that it must not be described as angle-resolved high-fidelity
cloud RT.

SH4 costs 0.671 s versus 0.273 s for Toon on the matched
`64 x 900 x 4 x 6` one-thread shape, or 2.46 times as much. The fidelity cost is
well bounded and appropriate for retrieval-scale validation.

### Molecular and realistic atmosphere cases

With identical six-gas molecular optical depth over 80 layers, 117
wavelengths, and 20 correlated-k ordinates, the PICASO disk comparison has
0.0152% RMS and 0.0507% maximum relative difference; the maximum point-angle
difference is 0.280%. For a realistic HAT-P-32b FastChem molecular+CIA case,
the disk RMS is `1.325e-5`, the maximum is `3.607e-5` (0.00361%), and the
maximum point-angle difference is 0.0116%. CIA changes the eclipse spectrum by
up to 4.512 ppm in that setup.

### Production MgSiO3/Mie/SH4 path

The full four-dataset WASP-69b case uses six gases, 80 layers, 16 g ordinates,
280 observed points, a 0.1 micron MgSiO3 particle, exact Mie moments through
degree four, and delta-M SH4. Its one-thread median clear and cloudy times were
0.386 s and 1.146 s, respectively: a 2.97 times cloudy/clear ratio and a
0.760 s cloud cost. The cloudy spectrum differs from the clear spectrum by up
to `5.231e-7` in eclipse depth.

Most importantly, the optimized spectrum-only path matches the full diagnostic
solver to `3.47e-18` absolute and `1.27e-15` relative eclipse depth. This
closes the risk that retrieval optimization changed the final emission
spectrum.

The run exposed a reproducibility issue rather than a physics failure: only an
older flat prepared-opacity cache existed, while the benchmark now requires a
resolution-scoped `R1000` cache. The local cache was migrated with explicit
resolution metadata. Generated opacity caches remain untracked and need a
documented preparation/validation command or manifest-driven automatic
migration for other machines.

The Taylor et al. 2021/NEMESIS benchmark remains qualitative: its Figure 1
median differences range from roughly -0.8% to -9.3%, but it uses a fitted
-4 dex mapping between NEMESIS cloud opacity and ROBERT mass extinction. It is
useful for shape and solver checks, not strict cloud-microphysics parity.
The older PICASO/Virga interchange benchmark defaults to synthetic cloud
optical properties and an internal solver comparison. Review 35 supersedes
that limitation with separately evaluated ROBERT and Virga Mie optics, vertical
cloud optical depth, gas validation optical depth, and final SH4 spectra.

No PICASO cloudy-transmission comparison currently exists. The independent
transmission reference is pRT, and the implemented solver is explicitly
absorption-dominated.

## Other benchmark findings

- HAT-P-32b clear emission has a 16.70 ppm RMSE against the existing external
  reference, with median depths of 982.53 ppm and 968.29 ppm.
- HAT-P-32b transmission spans 1842.75 ppm; CIA changes it by at most
  1.838 ppm. Impact quadrature converges from 0.615 ppm maximum error at order
  2 to 0.0132 ppm at order 8 relative to order 16.
- Native pRT HDF opacity gives 0.108% RMS and 0.647% maximum 80-layer emission
  difference; ROBERT's 80-to-160-layer self-change is 0.0408% RMS.
- The conservative JAX RORR output agrees with NumPy to `5.05e-12` maximum
  relative difference. On the tested CPU shape, Numba is much faster than JAX
  after compilation (0.064 s versus 0.486 s) and remains the appropriate
  default.
- Shared-atmosphere multi-instrument evaluation is exactly equivalent to the
  independent-mode path and is 1.28 times faster in the measured case.
- ROBERT sampling opacity at the same pRT R1000 source resolution gives 0.372%
  RMS and 1.474% maximum spectral difference. Lower-resolution or sparsely
  sampled source opacities can instead produce 5-22% maximum differences.
- In the full WASP-69b sampling comparison, correlated-k versus full ExoMol
  sampling differs by 50.0 ppm RMS and 164.6 ppm maximum, or 0.796 observational
  sigma RMS. Opacity resolution remains scientifically consequential.

## Audit repair

`examples/benchmark_wasp69b_multi_instrument.py` imported `PATTERNS` and
`PRT_DATA` from a retrieval example that no longer defines them, so even its
help command failed. The benchmark now owns those stable data-discovery
constants and uses package-aware imports for the live shared helpers. A test
imports the benchmark and verifies that all configured species and the pRT
data root are present. The repaired benchmark completes with exactly matching
spectra and likelihoods between the independent and shared-atmosphere paths.

## Prioritized remaining upgrades

### Priority 0: scientific release blockers

1. Add versioned, checksum-pinned final-spectrum regression fixtures for the
   stable pRT 3 emission and transmission cases and the matched PICASO SH4
   cases. Run small fixtures in normal CI and the complete external matrix in
   scheduled or release CI.
2. Extend the completed physical-contract PICASO/Virga cloud benchmark to the
   official PICASO science molecular-opacity database, comparing molecular and
   cloud optical depths separately before the final spectrum.
3. Implement and validate cloudy transmission, including slant scattering or
   an explicitly justified approximation, against an independent reference.
4. Close configured-task integration and error-path coverage. A 22%-covered
   production orchestrator is too weak for unattended science runs.
5. Rebaseline the roadmap and package version, define validation tolerances and
   release gates, and distinguish `experimental`, `validation`, and `stable`
   public APIs.

### Priority 1: accuracy and performance

1. Isolate and close the 1-4% independent multi-gas emission difference at the
   correlated-k/RORR convention boundary. Add abundance, layer, g-grid, and
   species-order invariance tests around `rt/random_overlap.py`.
2. Make opacity resolution and convergence a first-class run requirement. A
   science manifest should record the source line list, source resolution,
   sampling/binning method, cache checksum, and convergence evidence.
3. Implement a parity-gated batched SH4 boundary solve, retaining SciPy/NumPy
   as the reference. The current thousands of small serial banded solves are
   the dominant cloudy-emission cost.
4. Share the atmospheric state across cloudy spectral modes just as the clear
   path already does, while retaining mode-specific opacity, Mie, and SH4
   evaluation.
5. Expand Mie validation over particle-size distributions, refractive-index
   interpolation, conservative limits, vertical cloud boundaries, and
   high-stream strongly anisotropic phase functions.

### Priority 2: production completeness

1. Add calibrated JWST product ingestion and multi-instrument covariance.
2. Stabilize plugin discovery and metadata validation.
3. Add broader sampler support and long-run posterior diagnostics only after
   the forward-model release gates are automated.
4. Lock and continuously test the complete development environment. This audit
   found that the named environment lacked Pydantic even though the environment
   specification declares it.
5. Document generated opacity-cache creation, resolution migration, storage,
   and checksum validation so benchmarks are reproducible on a clean machine.

## Release acceptance recommendation

ROBERT can be described as suitable for clear-atmosphere emission and
absorption-dominated transmission validation retrievals, and for controlled
cloudy-emission experiments using SH4. The independently evaluated MgSiO3
validation-opacity spectrum supports end-to-end cloud-optics parity, but claims
of production-complete molecular-plus-cloud physics or scattering transmission
should wait for the remaining Priority 0 benchmarks. Preserve both shared-input
solver tests and independent end-to-end tests: the former localize numerical
errors, while the latter expose opacity and microphysics convention errors
that a matched-optical-depth benchmark intentionally removes.
