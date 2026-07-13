# ROBERT

ROBERT is an early-stage JWST exoplanet emission retrieval code. It now
contains typed core domain objects, modular atmosphere and chemistry
components, opacity import/archive helpers, RT-facing optical-depth objects, a
NumPy emission reference solver, first cloud/aerosol scattering hooks, optimal
estimation, and an optional UltraNest adapter. The retrieval layer is suitable
for validation runs, but the physical model is not yet a production science
retrieval.

The Python distribution name is `robert-exoplanets` to avoid colliding with the existing `robert` package on PyPI.

## Python Support

ROBERT supports Python 3.10 through 3.14. The reproducible full environment
uses Python 3.12, while CI tests every supported version.

## Architecture

The project architecture is governed by [RFC-0001: ROBERT Architectural Specification](docs/rfcs/0001-robert-architectural-specification.md), with companion documents listed in [docs/architecture](docs/architecture/README.md). Future substantial contributions should follow that document suite.

## Quick Start

```bash
conda env create --file environment.yml
conda activate robert-exoplanets
pytest
python examples/plot_blackbody_reference.py
python examples/plot_synthetic_tau_weighting.py
python examples/plot_cloud_scattering_reference.py
python examples/benchmark_cloud_scattering_picaso_virga.py
```

For an already-created environment, refresh the editable install with:

```bash
python -m pip install -e ".[dev]"
```

Install all optional runtime integrations without Conda with:

```bash
python -m pip install -e ".[complete]"
```

For the runnable Jupyter examples, include the notebook environment:

```bash
python -m pip install -e ".[dev,notebooks,opacity,retrieval]"
```

Science runs use a strict, versioned YAML file rather than an edited Python
workflow. Validate the supplied WASP-69b NIRCam+MIRI example without loading
its external data or opacity:

```bash
python run_retrieval.py \
  --config configurations/wasp69b_clear_R1000.yaml \
  --validate-only
```

See [Configuring and running ROBERT](docs/configuration.md) for portable paths,
directory initialization, opacity preparation, forward modelling, retrievals,
and 64-rank Slurm submission.

The plotting example writes local figures under `examples/outputs/`, which is
ignored by git.

The validated HAT-P-32b FastChem benchmark is bundled with the repository. A
fresh clone can immediately check its reference forward model:

```bash
python examples/compare_hat_p_32b_fastchem_retrieval.py
```

Run UltraNest locally with MPI:

```bash
mpiexec -n 4 python examples/compare_hat_p_32b_fastchem_retrieval.py \
  --method ultranest --run-retrieval \
  --live-points 400 --max-ncalls 500000 --dlogz 0.5 \
  --mpi-nprocs 4 \
  --output-dir retrieval_runs/hat_p_32b
```

See [Running the HAT-P-32b Retrieval on Slurm](docs/running_hat_p_32b_on_slurm.md)
for the supplied batch script, cluster setup, parameter overrides, and result
checks. Bundle provenance and third-party licensing are documented under
[`examples/data/hat_p_32b`](examples/data/hat_p_32b/README.md).

The multi-gas RT injection-recovery validation can be run with:

```bash
python examples/validate_hat_p_32b_injection_recovery.py --method optimal_estimation
```

It uses H2O, CO2, and NH3 on explicitly defined synthetic wavelength bins and
writes the injected truth, retrieval products, a pass/fail validation report,
and a residual plot. UltraNest is also available with `--method ultranest`;
non-converged sampler runs cannot pass validation.

## What Exists Today

- A minimal `robert_exoplanets` Python package.
- Core grid, spectrum, planet, star, and observation containers.
- Strict YAML task configuration plus typed internal configuration for
  forward modelling, optimal estimation, and UltraNest workflows.
- Blackbody reference diagnostics for visual sanity checks.
- Opacity metadata, coverage checks, and lightweight inspectors for ExoMol,
  ExoMolOP/exo_k `.kta`, HITRAN `.par`, HITRAN CIA, and future ROBERT archives.
- A validated ExoMolOP/exo_k `.kta` reader and converter into ROBERT
  native archives, with an optional in-memory floor for missing non-finite
  k-coefficients while leaving source tables unchanged.
- ROBERT-native opacity archive helpers for readable-manifest `.npy`
  directories and compact `.npz` exchange files, with an I/O benchmark example.
- A native-grid correlated-k opacity evaluator for exact benchmark cases and
  optional log-pressure, temperature, log-k interpolation.
- A functional beta ExoMol cross-section opacity-sampling backend for
  convergence experiments; correlated-k remains the validated retrieval
  default.
- Accuracy-preserving spectral-bin preparation through `exo_k`, which
  re-sorts and recompresses each correlated-k distribution onto observation
  bin edges instead of interpolating individual g ordinates.
- Species-generic loading of precomputed ExoMol/exo_k KTA and HDF5
  correlated-k products, including exo_k-native replacement of missing zero
  coefficients with fully recorded provenance.
- A local HAT-P-32b opacity benchmark example that reports exact evaluator
  agreement, records missing opacity-table regions, and plots k-coefficient
  slices.
- Gas optical-depth assembly from evaluated correlated-k opacity, including
  random-overlap multi-gas mixing, CIA/Rayleigh optical-depth contributors, and
  plot-ready cumulative tau and transmission-weighting diagnostics.
- An independent optional float64 JAX/XLA conservative-RORR backend for
  explicit Apple-Silicon/Linux CPU experiments and Linux CUDA benchmarks;
  NumPy and Numba remain the reference and default CPU paths.
- Hydrostatic radius/path geometry anchored at a reference pressure, available
  as an optional spherical-shell path model for RT experiments.
- A NumPy clear-sky thermal-emission reference solver with Planck source
  integration, disk quadrature, eclipse-depth normalization, and layer
  contribution diagnostics.
- A reusable, typed `ClearSkyEmissionForwardModel` that maps retrieval
  parameters into constant trace-gas abundances, an optional temperature
  offset and radius scale, prepared correlated-k opacity, and eclipse-depth RT;
  target examples contain only assembly data and priors.
- A Python-first `ClearSkyEmissionFactoryConfig` and public factory that turn
  typed planet, star, temperature-profile, opacity-source, and `exo_k` binning
  choices into a prepared model; `examples/hat_p_32b_config.py` is the validated
  target configuration to copy for new planets.
- A complete Python `RetrievalRunConfig` for OE or UltraNest and a
  parameterized clear-sky model that evaluates Madhusudhan-Seager P–T profiles
  and FastChem `[M/H]`/C/O chemistry inside every likelihood call.
- The provenance-pinned NemesisPy v1.0.1 CIA reference table, configurable
  endpoint-inclusive logarithmic pressure grids, explicit NemesisPy-compatible
  opacity-boundary clipping, and reusable ROBERT/NemesisPy comparison plots.
- Cloud/aerosol optical-property containers with extinction optical depth,
  single-scattering albedo, asymmetry factor, phase moments, and
  absorption/scattering splits.
- PICASO/Virga-style cloud optical-property interchange readers for dense
  `.npz` arrays and long-table `.csv` files.
- Cloud-type-agnostic homogeneous-sphere Mie optics from measured or retrieved
  complex refractive indices, including exact scalar phase moments through
  degree four, lognormal particle-size averaging, condensate-mass hydrostatic
  optical depths, and a retrieval-facing nodal `n(lambda)`/`log10(k(lambda))`
  emission model.
- A pinned, separately licensed Exo Skryer optical-constant catalogue with 44
  selectable physical/reference materials and source-checksum provenance.
- First-order direct-beam single-scattering source diagnostics for phase-aware
  geometries.
- A coupled hemispheric-mean Toon thermal two-stream source-function backend,
  validated in controlled disk-integrated grey cases against PICASO while
  retaining explicit limits for strongly anisotropic angle-resolved use.
- A four-term spherical-harmonics (P3/SH4) thermal multiple-scattering backend
  for higher-fidelity cloudy emission, with physical or HG phase moments,
  delta-M scaling, matched PICASO tests, and retrieval-scale timing benchmarks.
- An independently evaluated end-to-end MgSiO3 cloud benchmark: ROBERT and
  PICASO/Virga separately assemble validation gas opacity, Mie efficiencies,
  particle-averaged cloud optics, vertical optical depth, and SH4 emission from
  one checksum-pinned physical contract.
- Exact linear-in-optical-depth clear thermal source integration and a first
  absorption-dominated spherical-shell transmission solver with correlated-k
  impact-parameter integration.
- A Numba-backed thermal source integration path for thermal-only RT, with a
  NumPy reference backend retained for tests and debugging.
- A coordinate-checked Gaussian likelihood with masks, offsets, and jitter.
- Shared-atmosphere multi-dataset emission as the default multi-instrument
  retrieval flow: one temperature/chemistry state per physical region fans out
  to arbitrary mode-specific correlated-k opacity and RT calculations.
- Typed retrieval parameters and priors, diagnostic optimal estimation, and an
  optional UltraNest adapter behind one `RetrievalProblem` interface.
- Versioned run manifests written before inference, including configuration
  hashes, opacity identifiers, code/runtime provenance, settings, and seeds,
  plus method-independent JSON/NPZ results.
- Deterministic injection-recovery validation helpers with explicit parameter
  tolerances, fit-quality bounds, convergence gating, and versioned reports.
- Unit, integration, regression, and scientific benchmark tests.

## What Comes Later

- Calibrated JWST pipeline-product ingestion and multi-instrument covariance.
- Production atmospheric parameterizations and broader sampler support.
- End-to-end PICASO/Virga parity using independent science molecular-opacity
  databases, plus high-stream validation beyond the matched SH4 closure.
- Long-run posterior diagnostics and science-grade validation suites.
