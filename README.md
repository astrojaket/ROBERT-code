# ROBERT

ROBERT is a starter repository for a JWST exoplanet emission retrieval code. It currently provides the package layout, typed core domain objects, tests, and a stubbed end-to-end example. Full retrieval physics is not implemented yet.

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
- A validated ExoMolOP/exo_k/NEMESIS `.kta` reader and converter into ROBERT
  native archives.
- ROBERT-native opacity archive helpers for readable-manifest `.npy`
  directories and compact `.npz` exchange files, with an I/O benchmark example.
- Tests that lock in the intended skeleton behavior.

## What Comes Later

- Real JWST data ingestion.
- Atmospheric parameterization.
- Radiative transfer and opacity handling.
- Likelihood evaluation.
- Sampler integration.
