# ROBERT

ROBERT is an early-stage JWST exoplanet emission retrieval code. It now
contains typed core domain objects, modular atmosphere and chemistry
components, opacity import/archive helpers, RT-facing optical-depth objects, a
NumPy emission reference solver, first cloud/aerosol scattering hooks, and
benchmark examples. A full retrieval engine is not implemented yet.

The Python distribution name is `robert-exoplanets` to avoid colliding with the existing `robert` package on PyPI.

## Architecture

The project architecture is governed by [RFC-0001: ROBERT Architectural Specification](docs/rfcs/0001-robert-architectural-specification.md), with companion documents listed in [docs/architecture](docs/architecture/README.md). Future substantial contributions should follow that document suite.

## Quick Start

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
pytest
python examples/stub_emission_retrieval.py
python examples/minimal_forward_model.py
python examples/plot_blackbody_reference.py
python examples/plot_synthetic_tau_weighting.py
python examples/plot_cloud_scattering_reference.py
```

For an already-created environment, refresh the editable install with:

```bash
python -m pip install -e ".[dev]"
```

The plotting example writes local figures under `examples/outputs/`, which is
ignored by git.

If the external HAT-P-32b benchmark is available locally, you can also run:

```bash
python examples/plot_hat_p_32b_benchmark.py
python examples/plot_hat_p_32b_pt_profile.py
python examples/inspect_opacity_metadata.py
python examples/benchmark_opacity_archive_io.py
python examples/benchmark_hat_p_32b_opacity.py
python examples/benchmark_hat_p_32b_emission_rt.py
```

Set `HAT_P_32B_EMISSION_CSV` to override the default Dropbox benchmark path.
Set `HAT_P_32B_PT_CSV` to override the default Dropbox P-T profile path.
Set `HAT_P_32B_KTA_DIR` to override the default Dropbox k-table directory.

## What Exists Today

- A minimal `robert_exoplanets` Python package.
- Core grid, spectrum, planet, star, and observation containers.
- Retrieval configuration containers.
- A placeholder emission model.
- A stub retrieval runner that returns deterministic mock results.
- A minimal non-retrieval forward-model pipeline with explicit placeholder
  atmosphere, opacity, instrument-response, and likelihood components.
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
- A local HAT-P-32b opacity benchmark example that reports exact evaluator
  agreement, records missing opacity-table regions, and plots k-coefficient
  slices.
- Gas optical-depth assembly from evaluated correlated-k opacity, including
  random-overlap multi-gas mixing, CIA/Rayleigh optical-depth contributors, and
  plot-ready cumulative tau and transmission-weighting diagnostics.
- Hydrostatic radius/path geometry anchored at a reference pressure, available
  as an optional spherical-shell path model for RT experiments.
- A NumPy clear-sky thermal-emission reference solver with Planck source
  integration, disk quadrature, eclipse-depth normalization, and layer
  contribution diagnostics.
- Cloud/aerosol optical-property containers with extinction optical depth,
  single-scattering albedo, asymmetry factor, and absorption/scattering splits.
- First-order direct-beam single-scattering source diagnostics for phase-aware
  geometries.
- A first conservative two-stream multiple-scattering reference backend behind
  the RT interface, intended for benchmarking and replacement by fuller
  scattering solvers.
- Tests that lock in the intended skeleton behavior.

## What Comes Later

- Real JWST data ingestion.
- Retrieval parameterization and sampler integration.
- PICASO/Virga cloud-scattering benchmark parity and fuller scattering solvers.
- Likelihood evaluation.
- Sampler integration.
